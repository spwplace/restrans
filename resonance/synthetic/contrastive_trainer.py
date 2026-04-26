"""
Contrastive Training Harness
============================

Group-based contrastive training for the Resonance Transformer.

The Resonance Transformer is a dual-representation model:
  - Semantic embeddings encode program meaning
  - Phase embeddings encode proof/procedural structure

This harness trains the model to:
  1. Predict the next token (standard language modeling)
  2. Map all programs proving the same statement to nearby points
     in embedding space (contrastive discrimination of proof groups)

The combination forces the model to learn the topology of judgemental
 deduction: syntactically different proofs of the same proposition are
 semantically equivalent and must cluster together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Local imports
from .lambda_generator import Term
from .proof_walk_generator import ProofWalkDataset
from .mutation_engine import BisimilarMutation, MutatedProgram, training_schedule


# =============================================================================
# 1. Contrastive Losses
# =============================================================================

class ContrastiveGroupLoss(nn.Module):
    """
    Fast vectorized contrastive loss for proof groups.

    Uses a fully-vectorized InfoNCE implementation that replaces the
    O(N*k^2) Python loops with matrix operations on the GPU.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 1.0,
        loss_type: str = "infonce",
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.loss_type = loss_type

    def forward(
        self,
        embeddings: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Parameters
        ----------
        embeddings : torch.Tensor [N, D]
            Embeddings for N programs (concatenated across batch groups).
        positive_mask : torch.Tensor [N, N] bool
            True where (i,j) is a positive pair (same group).
        negative_mask : torch.Tensor [N, N] bool, optional
            True where (i,j) is a negative pair (different groups).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if negative_mask is None:
            negative_mask = ~positive_mask
            diag = torch.arange(embeddings.size(0), device=embeddings.device)
            negative_mask[diag, diag] = False

        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        if self.loss_type == "infonce":
            return self._infonce_loss(similarity, positive_mask, negative_mask)
        elif self.loss_type == "triplet":
            return self._triplet_loss(similarity, positive_mask, negative_mask)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _infonce_loss(
        self,
        similarity: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized InfoNCE: log(exp(pos) / sum(exp(all_valid)))."""
        N = similarity.size(0)
        diag_mask = torch.eye(N, device=similarity.device, dtype=torch.bool)

        # Exclude self-similarity
        sim = similarity.masked_fill(diag_mask, float("-inf"))
        sim_exp = torch.exp(sim)

        # Sum over positives and all valid entries
        pos_sum = (sim_exp * positive_mask.float()).sum(dim=1)  # [N]
        valid_sum = (sim_exp * (~diag_mask).float()).sum(dim=1)  # [N]

        has_pos = positive_mask.sum(dim=1) > 0
        has_neg = negative_mask.sum(dim=1) > 0
        valid = has_pos & has_neg

        if valid.sum() == 0:
            return torch.tensor(0.0, device=sim.device, requires_grad=True)

        loss = -torch.log((pos_sum[valid] + 1e-8) / (valid_sum[valid] + 1e-8))
        return loss.mean()

    def _triplet_loss(
        self,
        similarity: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized triplet margin loss."""
        # Compute mean positive and mean negative similarity per anchor
        pos_counts = positive_mask.sum(dim=1, keepdim=True).clamp(min=1)
        neg_counts = negative_mask.sum(dim=1, keepdim=True).clamp(min=1)

        pos_mean = (similarity * positive_mask.float()).sum(dim=1) / pos_counts.squeeze(1)
        neg_mean = (similarity * negative_mask.float()).sum(dim=1) / neg_counts.squeeze(1)

        has_pos = positive_mask.sum(dim=1) > 0
        has_neg = negative_mask.sum(dim=1) > 0
        valid = has_pos & has_neg

        if valid.sum() == 0:
            return torch.tensor(0.0, device=similarity.device, requires_grad=True)

        loss = F.relu(neg_mean[valid] - pos_mean[valid] + self.margin)
        return loss.mean()


class DualContrastiveLoss(nn.Module):
    """
    Dual-representation contrastive loss for the Resonance Transformer.

    The model produces two embeddings per program:
      - semantic_embed : encodes denotational meaning
      - phase_embed    : encodes proof structure / computational path

    The semantic embeddings should cluster by normal form (bisimilarity).
    The phase embeddings should cluster by proof strategy (syntactic family).

    Parameters
    ----------
    semantic_weight : float
        Weight for semantic contrastive loss.
    phase_weight : float
        Weight for phase contrastive loss.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        semantic_weight: float = 1.0,
        phase_weight: float = 0.5,
        loss_type: str = "infonce",
    ) -> None:
        super().__init__()
        self.semantic_loss = ContrastiveGroupLoss(temperature, loss_type=loss_type)
        self.phase_loss = ContrastiveGroupLoss(temperature, loss_type=loss_type)
        self.semantic_weight = semantic_weight
        self.phase_weight = phase_weight

    def forward(
        self,
        semantic_embeds: torch.Tensor,
        phase_embeds: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: Optional[torch.Tensor] = None,
        phase_positive_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute dual contrastive loss.

        Returns
        -------
        total_loss : torch.Tensor
        components : dict with "semantic", "phase", "total"
        """
        sem_loss = self.semantic_loss(semantic_embeds, positive_mask, negative_mask)

        if phase_positive_mask is not None:
            ph_loss = self.phase_loss(phase_embeds, phase_positive_mask, negative_mask)
        else:
            ph_loss = self.phase_loss(phase_embeds, positive_mask, negative_mask)

        total = self.semantic_weight * sem_loss + self.phase_weight * ph_loss
        return total, {
            "semantic": sem_loss.detach(),
            "phase": ph_loss.detach(),
            "total": total.detach(),
        }


# =============================================================================
# 2. Simple Tokenizer for Programs
# =============================================================================

class ProgramTokenizer:
    """
    Character-level tokenizer for lambda calculus programs.

    Maps each character to an integer ID. Handles padding/truncation.
    This is intentionally simple -- a production system would use BPE
    or a learned tokenizer.
    """

    def __init__(self, max_length: int = 128) -> None:
        # Vocabulary: all ASCII chars that appear in our program syntax
        special = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        chars = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            "()λ. :->_ "
        )
        self.vocab = special + list(chars)
        self.char2id = {c: i for i, c in enumerate(self.vocab)}
        self.id2char = {i: c for i, c in enumerate(self.vocab)}
        self.max_length = max_length
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> List[int]:
        ids = [self.sos_id]
        for c in text[: self.max_length - 2]:
            ids.append(self.char2id.get(c, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        chars = []
        for i in ids:
            if i == self.eos_id:
                break
            if i not in (self.pad_id, self.sos_id):
                chars.append(self.id2char.get(i, "?"))
        return "".join(chars)

    def batch_encode(
        self, texts: List[str], device: str = "cpu"
    ) -> torch.Tensor:
        """Encode a batch of texts, padding to max_length."""
        batch_ids = []
        for text in texts:
            ids = self.encode(text)
            # Pad
            while len(ids) < self.max_length:
                ids.append(self.pad_id)
            batch_ids.append(ids[: self.max_length])
        return torch.tensor(batch_ids, dtype=torch.long, device=device)


# =============================================================================
# 3. Minimal Resonance Transformer Stub
# =============================================================================

class MinimalResonanceTransformer(nn.Module):
    """
    Minimal dual-representation transformer for testing the training harness.

    Not the full Resonance Transformer -- just enough to produce two
    embedding vectors per input program for the contrastive objective.

    A real Resonance Transformer would have:
      - Separate semantic and phase encoders
      - Phase-coupled attention (resonance mechanism)
      - Complex phase-space dynamics

    Parameters
    ----------
    vocab_size : int
    d_model : int
    n_layers : int
    n_heads : int
    d_semantic : int
        Dimension of semantic embeddings.
    d_phase : int
        Dimension of phase embeddings.
    max_length : int
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_semantic: int = 64,
        d_phase: int = 64,
        max_length: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Semantic head: mean-pooled representation
        self.semantic_proj = nn.Linear(d_model, d_semantic)

        # Phase head: max-pooled representation (captures salient features)
        self.phase_proj = nn.Linear(d_model, d_phase)

        # LM head for next-token prediction
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.max_length = max_length

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor [B, L]
            Token IDs.

        Returns
        -------
        logits : torch.Tensor [B, L, V]
            Language modeling logits.
        semantic_embeds : torch.Tensor [B, d_semantic]
        phase_embeds : torch.Tensor [B, d_phase]
        """
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        tok_emb = self.token_embed(x)
        pos_emb = self.pos_embed(positions)
        h = tok_emb + pos_emb

        # Transformer
        mask = (x == 0).bool()  # PAD masking
        h = self.encoder(h, src_key_padding_mask=mask)

        # Semantic: mean pool over non-pad positions
        mask_exp = (~mask).float().unsqueeze(-1)
        semantic = (h * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        semantic = self.semantic_proj(semantic)

        # Phase: max pool
        phase = h.masked_fill(mask.unsqueeze(-1), -1e9).max(dim=1)[0]
        phase = self.phase_proj(phase)

        # LM logits
        logits = self.lm_head(h)

        return logits, semantic, phase


# =============================================================================
# 4. Training Loop
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for contrastive training."""

    n_epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 3e-4
    lm_weight: float = 1.0
    contrastive_weight: float = 1.0
    max_length: int = 128
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    log_interval: int = 10
    mutation_schedule: str = "linear"
    max_mutations: int = 1000
    temperature: float = 0.07
    gradient_clip: float = 1.0


def train_contrastive(
    model: nn.Module,
    dataset: ProofWalkDataset,
    config: TrainingConfig,
    mutation_engine: Optional[BisimilarMutation] = None,
    tokenizer: Optional[ProgramTokenizer] = None,
) -> Dict[str, List[float]]:
    """
    Train a model with combined LM + contrastive objectives.

    Parameters
    ----------
    model : nn.Module
        Model with signature: forward(x) -> (logits, semantic_embed, phase_embed)
    dataset : ProofWalkDataset
    config : TrainingConfig
    mutation_engine : BisimilarMutation, optional
        If provided, apply scheduled mutations each epoch.
    tokenizer : ProgramTokenizer, optional
        If None, a default tokenizer is created.

    Returns
    -------
    history : dict
        Training curves for lm_loss, contrastive_loss, total_loss.
    """
    tokenizer = tokenizer or ProgramTokenizer(max_length=config.max_length)
    device = torch.device(config.device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    contrastive_loss_fn = DualContrastiveLoss(
        temperature=config.temperature,
        loss_type="infonce",
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    history: Dict[str, List[float]] = {
        "lm_loss": [],
        "contrastive_loss": [],
        "total_loss": [],
        "epoch": [],
    }

    for epoch in range(config.n_epochs):
        model.train()
        epoch_lm_loss = 0.0
        epoch_ctr_loss = 0.0
        epoch_total = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            # batch["programs"] is List[List[str]]: B groups, each with k programs
            programs_batch = batch["programs"]  # List of k_i programs per group
            statements = batch["statements"]

            # Flatten all programs and compute embeddings
            all_programs: List[str] = []
            group_sizes: List[int] = []
            for group_progs in programs_batch:
                all_programs.extend(group_progs)
                group_sizes.append(len(group_progs))

            if not all_programs:
                continue

            # Tokenize
            tokens = tokenizer.batch_encode(all_programs, device=device)

            # Forward pass
            logits, sem_embeds, phase_embeds = model(tokens)

            # 1. Language modeling loss
            # Predict next token: shift targets
            targets = tokens[:, 1:].contiguous()
            lm_logits = logits[:, :-1, :].contiguous()
            lm_loss = F.cross_entropy(
                lm_logits.reshape(-1, lm_logits.size(-1)),
                targets.reshape(-1),
                ignore_index=tokenizer.pad_id,
            )

            # 2. Contrastive loss
            # Build positive mask across the flattened batch
            total = sum(group_sizes)
            positive_mask = torch.zeros(total, total, dtype=torch.bool, device=device)
            offset = 0
            for size in group_sizes:
                positive_mask[offset : offset + size, offset : offset + size] = True
                offset += size

            # Build negative mask: different groups
            negative_mask = ~positive_mask
            diag = torch.arange(total, device=device)
            negative_mask[diag, diag] = False

            ctr_loss, _ = contrastive_loss_fn(
                sem_embeds, phase_embeds, positive_mask, negative_mask
            )

            # Combined loss
            total_loss = config.lm_weight * lm_loss + config.contrastive_weight * ctr_loss

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

            epoch_lm_loss += lm_loss.item()
            epoch_ctr_loss += ctr_loss.item()
            epoch_total += total_loss.item()
            n_batches += 1

            if (batch_idx + 1) % config.log_interval == 0:
                print(
                    f"Epoch {epoch+1}/{config.n_epochs} "
                    f"Batch {batch_idx+1} | "
                    f"LM: {lm_loss.item():.4f} | "
                    f"CTR: {ctr_loss.item():.4f} | "
                    f"Total: {total_loss.item():.4f}"
                )

        avg_lm = epoch_lm_loss / max(n_batches, 1)
        avg_ctr = epoch_ctr_loss / max(n_batches, 1)
        avg_total = epoch_total / max(n_batches, 1)

        history["lm_loss"].append(avg_lm)
        history["contrastive_loss"].append(avg_ctr)
        history["total_loss"].append(avg_total)
        history["epoch"].append(epoch + 1)

        print(
            f"\n=== Epoch {epoch+1} Summary ==="
            f"\n  LM Loss:        {avg_lm:.4f}"
            f"\n  Contrastive:    {avg_ctr:.4f}"
            f"\n  Total:          {avg_total:.4f}"
        )

        # Mutation scheduling
        if mutation_engine is not None:
            n_mutations = training_schedule(
                epoch,
                config.n_epochs,
                schedule=config.mutation_schedule,
                max_mutations=config.max_mutations,
            )
            print(f"  Mutation count: {n_mutations}")
            # Note: applying mutations to the dataset would require regenerating
            # terms. In a real loop, you'd mutate the cached programs here.

    return history


# =============================================================================
# 5. Self-test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Contrastive Trainer - Self Test")
    print("=" * 60)

    from .lambda_generator import LambdaGenerator, T_INT

    # Build a tiny dataset
    dataset = ProofWalkDataset(
        n_groups=4,
        programs_per_group=4,
        max_term_depth=3,
        max_term_size=10,
        generator_seed=42,
    )

    # Build tokenizer
    tokenizer = ProgramTokenizer(max_length=64)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Encode/decode test
    sample_prog = dataset[0]["programs"][0]
    ids = tokenizer.encode(sample_prog)
    decoded = tokenizer.decode(ids)
    print(f"\nEncode/decode test:")
    print(f"  Original: {sample_prog[:60]}...")
    print(f"  Decoded:  {decoded[:60]}...")

    # Build minimal model
    model = MinimalResonanceTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_semantic=32,
        d_phase=32,
        max_length=64,
    )

    # Quick forward test
    tokens = tokenizer.batch_encode([sample_prog], device="cpu")
    logits, sem, phase = model(tokens)
    print(f"\nModel output shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  semantic: {sem.shape}")
    print(f"  phase: {phase.shape}")

    # Test contrastive loss
    loss_fn = DualContrastiveLoss(temperature=0.1)
    embeds = torch.randn(8, 32)
    pos_mask = torch.zeros(8, 8, dtype=torch.bool)
    pos_mask[:4, :4] = True
    pos_mask[4:, 4:] = True
    neg_mask = ~pos_mask
    diag = torch.arange(8)
    neg_mask[diag, diag] = False
    total_loss, comps = loss_fn(embeds, embeds, pos_mask, neg_mask)
    print(f"\nContrastive loss test: {total_loss.item():.4f}")
    print(f"  Components: {comps}")

    # Test training (1 epoch, tiny)
    print("\n--- Mini training run (1 epoch) ---")
    config = TrainingConfig(
        n_epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        max_length=64,
        d_model=64,
        n_layers=2,
        n_heads=2,
        log_interval=1,
    )
    history = train_contrastive(model, dataset, config, tokenizer=tokenizer)
    print(f"\nHistory: {history}")

    print("\nAll tests passed.")
