#!/usr/bin/env python3
"""
Meaningful Experiment: Proof-Walk Prior Impact on Language Learning
=====================================================================

Tests whether pre-training on synthetic proof-walks (lambda calculus)
helps a small transformer learn natural language (TinyStories) faster
or better.

Four conditions:
  1. baseline     -- Standard transformer, train from scratch on text
  2. resonance    -- Resonance transformer, train from scratch on text
  3. proof_prior  -- Pre-train on proof-walks, then fine-tune on text
  4. proof_only   -- Train only on proof-walks (control)

Hardware: Optimized for AMD GPU (ROCm) with HSA_OVERRIDE_GFX_VERSION=11.0.0
          Falls back to CPU with all threads.

Usage:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python experiment_text.py --condition baseline --epochs 20
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python experiment_text.py --condition proof_prior --pretrain_epochs 10 --finetune_epochs 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from resonance.device import get_device, set_seed, enable_deterministic
from resonance.config import StandardConfig, ResonanceConfig
from resonance.models import StandardTransformer, ResonanceTransformer
from resonance.training import train_model

# Synthetic pipeline (may fail if not available)
try:
    from synthetic.proof_walk_generator import ProofWalkDataset, CrawlMode
    from synthetic.mutation_engine import BisimilarMutation, training_schedule
    from synthetic.contrastive_trainer import (
        MinimalResonanceTransformer,
        ProgramTokenizer,
        DualContrastiveLoss,
    )
    SYNTHETIC_AVAILABLE = True
except Exception as e:
    SYNTHETIC_AVAILABLE = False
    print(f"Warning: synthetic pipeline not available: {e}")


# =============================================================================
# 1. TinyStories Dataset
# =============================================================================

class TinyStoriesDataset(Dataset):
    """Word-level TinyStories dataset."""

    def __init__(
        self,
        texts: List[str],
        tokenizer: WordTokenizer,
        seq_len: int = 128,
    ) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self._tokens: List[List[int]] = []
        self._build()

    def _build(self) -> None:
        for text in self.texts:
            tokens = self.tokenizer.encode(text)
            # Slide a window over the tokenized text
            for i in range(0, max(1, len(tokens) - self.seq_len), self.seq_len // 2):
                chunk = tokens[i : i + self.seq_len + 1]
                if len(chunk) >= 2:
                    self._tokens.append(chunk[: self.seq_len + 1])

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self._tokens[idx]
        # Pad if necessary
        if len(tokens) < self.seq_len + 1:
            tokens = tokens + [self.tokenizer.pad_id] * (self.seq_len + 1 - len(tokens))
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return {"input_ids": x, "labels": y}


class WordTokenizer:
    """Simple word-level tokenizer with limited vocabulary."""

    def __init__(self, vocab_size: int = 5000) -> None:
        self.vocab_size = vocab_size
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.pad_id = 0
        self.unk_id = 1
        self.sos_id = 2
        self.eos_id = 3
        self._specials = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]

    def train(self, texts: List[str]) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text.lower().split())
        most_common = counter.most_common(self.vocab_size - len(self._specials))
        words = self._specials + [w for w, _ in most_common]
        self.word2id = {w: i for i, w in enumerate(words)}
        self.id2word = {i: w for w, i in self.word2id.items()}
        print(f"Tokenizer vocab: {len(self.word2id)} / {self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        tokens = [self.word2id.get(w, self.unk_id) for w in text.lower().split()]
        return tokens

    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            if i == self.eos_id:
                break
            if i not in (self.pad_id, self.sos_id):
                words.append(self.id2word.get(i, "<UNK>"))
        return " ".join(words)


class GPT2TokenizerWrapper:
    """Wrapper around HuggingFace GPT-2 tokenizer for standardized benchmarking."""

    def __init__(self, max_length: int = 256) -> None:
        from transformers import GPT2Tokenizer

        self._tok = GPT2Tokenizer.from_pretrained("gpt2")
        self._tok.pad_token = self._tok.eos_token
        self.vocab_size = self._tok.vocab_size
        self.pad_id = self._tok.pad_token_id
        self.max_length = max_length
        print(f"GPT-2 tokenizer loaded: vocab={self.vocab_size}")

    def train(self, texts: List[str]) -> None:
        """No-op: GPT-2 tokenizer is pre-trained."""
        pass

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


# =============================================================================
# 2. Proof-Walk Pre-Training
# =============================================================================

def pretrain_on_proofs(
    model: nn.Module,
    n_groups: int = 2000,
    programs_per_group: int = 8,
    epochs: int = 10,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Pre-train a model on synthetic proof-walks with contrastive + LM loss."""
    if not SYNTHETIC_AVAILABLE:
        raise RuntimeError("Synthetic pipeline not available")

    tokenizer = ProgramTokenizer(max_length=128)
    dataset = ProofWalkDataset(
        n_groups=n_groups,
        programs_per_group=programs_per_group,
        max_term_depth=5,
        max_term_size=16,
        generator_seed=seed,
        crawl_mode=CrawlMode.HYBRID,
        structured_ratio=0.5,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    contrastive_loss_fn = DualContrastiveLoss(temperature=0.07, loss_type="infonce")

    history: Dict[str, List[float]] = {"train_lm": [], "train_ctr": [], "train_total": []}

    for epoch in range(epochs):
        model.train()
        epoch_lm = 0.0
        epoch_ctr = 0.0
        epoch_total = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            programs_batch = batch["programs"]
            all_programs: List[str] = []
            group_sizes: List[int] = []
            for group_progs in programs_batch:
                all_programs.extend(group_progs)
                group_sizes.append(len(group_progs))

            if not all_programs:
                continue

            tokens = tokenizer.batch_encode(all_programs, device=str(device))
            logits, sem_embeds, phase_embeds = model(tokens)

            targets = tokens[:, 1:].contiguous()
            lm_logits = logits[:, :-1, :].contiguous()
            lm_loss = F.cross_entropy(
                lm_logits.reshape(-1, lm_logits.size(-1)),
                targets.reshape(-1),
                ignore_index=tokenizer.pad_id,
            )

            total = sum(group_sizes)
            positive_mask = torch.zeros(total, total, dtype=torch.bool, device=device)
            offset = 0
            for size in group_sizes:
                positive_mask[offset : offset + size, offset : offset + size] = True
                offset += size
            negative_mask = ~positive_mask
            diag = torch.arange(total, device=device)
            negative_mask[diag, diag] = False

            ctr_loss, _ = contrastive_loss_fn(
                sem_embeds, phase_embeds, positive_mask, negative_mask
            )

            total_loss = lm_loss + ctr_loss
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_lm += lm_loss.item()
            epoch_ctr += ctr_loss.item()
            epoch_total += total_loss.item()
            n_batches += 1

        history["train_lm"].append(epoch_lm / max(n_batches, 1))
        history["train_ctr"].append(epoch_ctr / max(n_batches, 1))
        history["train_total"].append(epoch_total / max(n_batches, 1))
        print(
            f"  [Pretrain Epoch {epoch+1}/{epochs}] "
            f"LM={history['train_lm'][-1]:.4f} CTR={history['train_ctr'][-1]:.4f} "
            f"Total={history['train_total'][-1]:.4f}"
        )

    return history


# =============================================================================
# 3. Main Experiment Harness
# =============================================================================

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proof-Walk Prior Experiment")
    parser.add_argument(
        "--condition",
        choices=["baseline", "resonance", "proof_prior", "proof_only"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=20, help="Text training epochs")
    parser.add_argument("--pretrain_epochs", type=int, default=10, help="Proof pre-training epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--train_samples", type=int, default=100000, help="TinyStories train samples")
    parser.add_argument("--val_samples", type=int, default=10000)
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--tokenizer", choices=["word", "gpt2"], default="word", help="Tokenizer type")
    parser.add_argument("--use_phase_stream", type=lambda x: x.lower() == "true", default=True, help="Enable phase embedding stream (resonance only)")
    parser.add_argument("--use_resonance_bias", type=lambda x: x.lower() == "true", default=True, help="Enable resonance attention bias (resonance only)")
    parser.add_argument(
        "--architecture",
        choices=[
            "standard", "resonance",
            "llama", "llama_moe", "resonant_llama", "resonant_llama_moe",
            "qwen3_5", "qwen3_5_moe", "resonant_qwen3_5", "resonant_qwen3_5_moe",
            "gemma4", "gemma4_moe", "resonant_gemma4", "resonant_gemma4_moe",
        ],
        default=None,
        help="Model architecture (overrides --condition default)",
    )
    # Resonance variant flags
    parser.add_argument("--resonance_kernel", default="cosine", help="Resonance kernel name")
    parser.add_argument("--phase_embedding", default="real", help="Phase embedding variant")
    parser.add_argument("--bias_mode", default="additive", help="Resonance bias application mode")
    parser.add_argument("--init_preset", default="default", choices=["default", "wide", "strong", "very_strong", "normalized"], help="Initialization preset")
    parser.add_argument("--n_frequencies", type=int, default=32, help="Number of phase frequencies")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Override max sequence length")
    return parser.parse_args()


def load_tinystories(n_train: int, n_val: int, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Load TinyStories texts."""
    from datasets import load_dataset

    print(f"Loading TinyStories (train={n_train}, val={n_val})...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    texts_train = []
    texts_val = []
    rng = random.Random(seed)

    for i, example in enumerate(ds):
        if i >= n_train + n_val:
            break
        text = example["text"].strip()
        if not text:
            continue
        if i < n_train:
            texts_train.append(text)
        else:
            texts_val.append(text)

    print(f"  Train: {len(texts_train)} stories")
    print(f"  Val:   {len(texts_val)} stories")
    return texts_train, texts_val


def main() -> None:
    args = get_args()
    set_seed(args.seed)
    enable_deterministic(True)

    device = get_device(args.device)
    print(f"Device: {device}")

    out_dir = Path(args.output_dir or f"resonance/outputs/experiment_{args.condition}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    # Save config
    config_dict = vars(args)
    config_dict["device"] = str(device)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print("=" * 60)
    print(f"Condition: {args.condition}")
    print(f"Model: {args.embed_dim}d, {args.n_layers}L, {args.n_heads}H")
    print("=" * 60)

    # ======================================================================
    # Load data and build tokenizer
    # ======================================================================
    texts_train, texts_val = load_tinystories(args.train_samples, args.val_samples, args.seed)
    if args.tokenizer == "gpt2":
        tokenizer: WordTokenizer | GPT2TokenizerWrapper = GPT2TokenizerWrapper(max_length=args.seq_len)
    else:
        tokenizer = WordTokenizer(vocab_size=args.vocab_size)
    tokenizer.train(texts_train)

    train_ds = TinyStoriesDataset(texts_train, tokenizer, seq_len=args.seq_len)
    val_ds = TinyStoriesDataset(texts_val, tokenizer, seq_len=args.seq_len)
    print(f"Train tokens: {len(train_ds)} sequences")
    print(f"Val tokens:   {len(val_ds)} sequences")

    # ======================================================================
    # Build model
    # ======================================================================
    actual_vocab = getattr(tokenizer, "vocab_size", len(getattr(tokenizer, "word2id", {})))
    max_seq_len = args.max_seq_len or args.seq_len

    # Determine architecture
    arch = args.architecture
    if arch is None:
        if args.condition in ("baseline", "proof_prior"):
            arch = "standard"
        else:
            arch = "resonance"

    # Legacy architectures
    if arch == "standard":
        from resonance.config import StandardConfig
        from resonance.models import StandardTransformer
        config = StandardConfig(
            vocab_size=actual_vocab, max_seq_len=max_seq_len, embed_dim=args.embed_dim,
            n_layers=args.n_layers, n_heads=args.n_heads, ff_dim=args.ff_dim,
            batch_size=args.batch_size, learning_rate=3e-4,
        )
        model = StandardTransformer(config)
    elif arch == "resonance":
        from resonance.config import ResonanceConfig
        from resonance.models import ResonanceTransformer
        config = ResonanceConfig(
            vocab_size=actual_vocab, max_seq_len=max_seq_len, embed_dim=args.embed_dim,
            n_layers=args.n_layers, n_heads=args.n_heads, ff_dim=args.ff_dim,
            n_frequencies=args.n_frequencies, batch_size=args.batch_size, learning_rate=3e-4,
            phonetic_init=False, use_phase_stream=args.use_phase_stream,
            use_resonance_bias=args.use_resonance_bias,
            resonance_kernel=args.resonance_kernel,
            phase_embedding=args.phase_embedding,
            bias_mode=args.bias_mode,
            init_preset=args.init_preset,
        )
        model = ResonanceTransformer(config)
    # Modern architectures
    elif arch in ("llama", "llama_moe", "resonant_llama", "resonant_llama_moe"):
        from resonance.modern.llama import LlamaConfig, LlamaTransformer, LlamaMoE, ResonantLlama, ResonantLlamaMoE
        cfg = LlamaConfig(
            vocab_size=actual_vocab, max_seq_len=max_seq_len, embed_dim=args.embed_dim,
            n_layers=args.n_layers, n_heads=args.n_heads, n_kv_heads=args.n_heads // 2,
            ff_dim=args.ff_dim, use_moe="moe" in arch, use_resonance="resonant" in arch,
            n_frequencies=args.n_frequencies, resonance_kernel=args.resonance_kernel,
            phase_embedding=args.phase_embedding, bias_mode=args.bias_mode,
            init_preset=args.init_preset,
        )
        model = {
            "llama": LlamaTransformer, "llama_moe": LlamaMoE,
            "resonant_llama": ResonantLlama, "resonant_llama_moe": ResonantLlamaMoE,
        }[arch](cfg)
    elif arch in ("qwen3_5", "qwen3_5_moe", "resonant_qwen3_5", "resonant_qwen3_5_moe"):
        from resonance.modern.qwen3_5 import Qwen3_5Config, Qwen3_5Transformer, Qwen3_5MoE, ResonantQwen3_5, ResonantQwen3_5MoE
        cfg = Qwen3_5Config(
            vocab_size=actual_vocab, max_seq_len=max_seq_len, embed_dim=args.embed_dim,
            n_layers=args.n_layers, n_heads=args.n_heads, n_kv_heads=args.n_heads // 2,
            ff_dim=args.ff_dim, use_moe="moe" in arch, use_resonance="resonant" in arch,
            n_frequencies=args.n_frequencies, resonance_kernel=args.resonance_kernel,
            phase_embedding=args.phase_embedding, bias_mode=args.bias_mode,
            init_preset=args.init_preset,
        )
        model = {
            "qwen3_5": Qwen3_5Transformer, "qwen3_5_moe": Qwen3_5MoE,
            "resonant_qwen3_5": ResonantQwen3_5, "resonant_qwen3_5_moe": ResonantQwen3_5MoE,
        }[arch](cfg)
    elif arch in ("gemma4", "gemma4_moe", "resonant_gemma4", "resonant_gemma4_moe"):
        from resonance.modern.gemma4 import Gemma4Config, Gemma4Transformer, Gemma4MoE, ResonantGemma4, ResonantGemma4MoE
        cfg = Gemma4Config(
            vocab_size=actual_vocab, max_seq_len=max_seq_len, embed_dim=args.embed_dim,
            n_layers=args.n_layers, n_heads=args.n_heads, n_kv_heads=args.n_heads // 2,
            ff_dim=args.ff_dim, use_moe="moe" in arch, use_resonance="resonant" in arch,
            use_ple=False,  # disable PLE for small-scale experiments
            n_frequencies=args.n_frequencies, resonance_kernel=args.resonance_kernel,
            phase_embedding=args.phase_embedding, bias_mode=args.bias_mode,
            init_preset=args.init_preset,
        )
        model = {
            "gemma4": Gemma4Transformer, "gemma4_moe": Gemma4MoE,
            "resonant_gemma4": ResonantGemma4, "resonant_gemma4_moe": ResonantGemma4MoE,
        }[arch](cfg)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # ======================================================================
    # Optional: Resume from checkpoint
    # ======================================================================
    start_epoch = 0
    pretrain_history = None
    text_history = {"train_loss": [], "val_loss": [], "val_ppl": []}

    resume_ckpt = out_dir / "checkpoints" / "latest.pt"
    if args.resume and resume_ckpt.exists():
        print(f"\n[Resume] Loading checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        text_history = ckpt.get("text_history", text_history)
        pretrain_history = ckpt.get("pretrain_history", None)
        print(f"  Resuming from epoch {start_epoch}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ======================================================================
    # Optional: Pre-train on proof-walks (skip if resuming after pretrain)
    # ======================================================================
    if start_epoch == 0 and args.condition in ("proof_prior", "proof_only"):
        if not SYNTHETIC_AVAILABLE:
            raise RuntimeError("Synthetic pipeline required for proof_prior/proof_only")
        print("\n[Phase 1] Pre-training on synthetic proof-walks...")
        pretrain_history = pretrain_on_proofs(
            model=model,
            n_groups=2000,
            programs_per_group=8,
            epochs=args.pretrain_epochs,
            batch_size=16,
            device=device,
            seed=args.seed,
        )
        ckpt = out_dir / "checkpoints" / "pretrain.pt"
        torch.save({"model_state": model.state_dict(), "history": pretrain_history}, ckpt)
        print(f"Pre-training checkpoint saved: {ckpt}")

    # ======================================================================
    # Train on text
    # ======================================================================
    if args.condition != "proof_only":
        remaining_epochs = args.epochs - start_epoch
        if remaining_epochs <= 0:
            print("\n[Resume] All epochs already completed.")
            history = text_history
        else:
            print(f"\n[Phase 2] Training on TinyStories (epochs {start_epoch + 1}–{args.epochs})...")

            def save_checkpoint(epoch: int, model: nn.Module, results: dict) -> None:
                ckpt_path = out_dir / "checkpoints" / "latest.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "config": config,
                        "text_history": results,
                        "pretrain_history": pretrain_history,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"  -> Checkpoint saved: {ckpt_path}")

            history = train_model(
                model=model,
                config=config,
                train_dataset=train_ds,
                val_dataset=val_ds,
                n_epochs=remaining_epochs,
                device=device,
                log_interval=max(1, len(train_ds) // args.batch_size // 10),
                epoch_end_callback=save_checkpoint,
            )
            # Merge with any previously accumulated history
            for key in ("train_loss", "val_loss", "val_ppl"):
                text_history[key] = text_history.get(key, []) + history.get(key, [])
            history = text_history

        # Save final
        final_path = out_dir / "checkpoints" / "final.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config,
                "history": history,
                "pretrain_history": pretrain_history,
                "args": vars(args),
            },
            final_path,
        )
        with open(out_dir / "history.json", "w") as f:
            json.dump(
                {
                    "text_history": history,
                    "pretrain_history": pretrain_history,
                },
                f,
                indent=2,
            )
        print(f"\nSaved to {out_dir}")
        print(f"Final val PPL: {history['val_ppl'][-1]:.2f}")
    else:
        print("\n[proof_only] Skipping text training.")
        with open(out_dir / "history.json", "w") as f:
            json.dump({"pretrain_history": pretrain_history}, f, indent=2)


if __name__ == "__main__":
    main()
