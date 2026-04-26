"""Transformer model classes: Standard (vanilla) and Resonance (dual embedding)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ResonanceConfig, StandardConfig


# ---------------------------------------------------------------------------
# Standard (vanilla) transformer components
# ---------------------------------------------------------------------------

class StandardAttention(nn.Module):
    """Vanilla multi-head causal self-attention.

    Uses a single fused QKV projection followed by split into heads,
    scaled dot-product attention, and an output projection.
    """

    def __init__(self, config: StandardConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply standard multi-head self-attention.

        Args:
            x: Input tensor of shape ``(batch, seq_len, embed_dim)``.
            mask: Optional causal mask of shape ``(1, seq_len, seq_len)``
                or broadcastable variant.  ``0`` indicates masked positions.

        Returns:
            Attention output of shape ``(batch, seq_len, embed_dim)``.
        """
        batch_size, seq_len, _ = x.shape

        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(out)


class StandardBlock(nn.Module):
    """A single transformer block for the standard model.

    Pre-norm residual: ``x = x + attn(ln1(x))`` then ``x = x + ff(ln2(x))``.
    """

    def __init__(self, config: StandardConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = StandardAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through one standard transformer block.

        Args:
            x: Input tensor ``(batch, seq_len, embed_dim)``.
            mask: Optional causal attention mask.

        Returns:
            Updated tensor of the same shape.
        """
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class StandardTransformer(nn.Module):
    """Vanilla causal transformer language model (GPT-like).

    Uses token + positional embeddings, a stack of ``StandardBlock`` layers,
    and a final linear language-modelling head tied to the token embedding.
    """

    def __init__(self, config: StandardConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [StandardBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_final = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        # Tie weights between input embedding and output projection
        self.lm_head.weight = self.token_embed.weight

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)),
        )

        self.apply(self._init_weights)
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module) -> None:
        """Standard GPT-style weight initialisation."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass and optional next-token loss computation.

        The loss is computed as ``cross_entropy(logits[:, :-1], labels[:, 1:])``
        so that every position predicts the *next* token.

        Args:
            input_ids: Token indices ``(batch, seq_len)``.
            labels: Optional target token indices of the same shape.  If
                provided, the ``"loss"`` key is included in the returned dict.

        Returns:
            Dictionary with at least ``"logits"`` (shape
            ``(batch, seq_len, vocab_size)``).  Optionally ``"loss"``.
        """
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)

        mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        logits = self.lm_head(self.ln_final(x))

        result: dict[str, Any] = {"logits": logits}
        if labels is not None:
            # Shift so that each position predicts the next token
            result["loss"] = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return result


# ---------------------------------------------------------------------------
# Resonance transformer components
# ---------------------------------------------------------------------------

class ResonanceEmbedding(nn.Module):
    """Dual embedding combining semantic and phase representations.

    Each token receives:
    1. A *semantic* embedding (standard, learned).
    2. A *phase* embedding (structural, learned, lower-dimensional).
    3. A *learnable blend* parameter (sigmoid-gated) controlling the
       interpolation between semantic and projected phase embeddings.

    Phase embeddings may be initialised randomly or from phonetic (rhyme)
    structure via :func:`_init_phonetic_phases`.
    """

    def __init__(
        self,
        config: ResonanceConfig,
        rhyme_index: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        # Semantic embedding (standard)
        self.semantic = nn.Embedding(config.vocab_size, config.embed_dim)

        # Phase embedding (structural / phonetic)
        self.phase = nn.Embedding(config.vocab_size, config.n_frequencies)
        self.phase_proj = nn.Linear(config.n_frequencies, config.embed_dim, bias=False)

        # Learnable blend parameter (per-dimension interpolation)
        self.blend = nn.Parameter(
            torch.full((config.embed_dim,), config.resonance_blend)
        )

        # Positional encoding
        self.position = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Initialise
        self._init_weights()
        if config.phonetic_init and rhyme_index is not None:
            self._init_phonetic_phases(rhyme_index)
        else:
            self._init_random_phases()

    def _init_weights(self) -> None:
        """Initialise semantic, projection, and positional embeddings."""
        nn.init.normal_(self.semantic.weight, std=0.02)
        nn.init.normal_(self.phase_proj.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)

    def _init_random_phases(self) -> None:
        """Initialise phase embeddings from a Gaussian distribution."""
        nn.init.normal_(self.phase.weight, std=0.3)

    def _init_phonetic_phases(self, rhyme_index: dict[str, Any]) -> None:
        """Initialise phase embeddings so rhyming words are close in phase space.

        Each rhyme group receives a shared base phase vector; tokens in the
        group are perturbed by a small amount of Gaussian noise so they are
        similar but not identical.
        """
        rhyme_to_tokens: dict[str, list[int]] = rhyme_index.get("rhyme_to_tokens", {})

        # Assign a base phase vector to each rhyme group
        rhyme_to_phase = {
            rhyme: torch.randn(self.config.n_frequencies) * 0.3
            for rhyme in rhyme_to_tokens.keys()
        }

        with torch.no_grad():
            self.phase.weight.data = torch.randn_like(self.phase.weight) * 0.3

            phonetic_count = 0
            for rhyme, tokens in rhyme_to_tokens.items():
                base_phase = rhyme_to_phase[rhyme]
                for token_id in tokens:
                    self.phase.weight.data[token_id] = (
                        base_phase + torch.randn(self.config.n_frequencies) * 0.03
                    )
                    phonetic_count += 1

        self.phonetic_count = phonetic_count

    def get_resonance_matrix(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute pairwise resonance scores from phase similarity.

        The resonance between two tokens is defined as the mean cosine of
        their phase differences across all frequency dimensions.

        Args:
            token_ids: Integer token ids ``(batch, seq_len)``.

        Returns:
            Resonance matrix ``(batch, seq_len, seq_len)`` with values in
            ``[-1, 1]``.
        """
        if not self.config.use_phase_stream:
            batch_size, seq_len = token_ids.shape
            return torch.zeros(batch_size, seq_len, seq_len, device=token_ids.device)
        phases = self.phase(token_ids)
        # Broadcast: (B, S, 1, F) - (B, 1, S, F) -> (B, S, S, F)
        phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)
        resonance = torch.cos(phase_diff).mean(dim=-1)
        return resonance

    def forward(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce blended embeddings and the resonance matrix.

        Args:
            token_ids: Token indices ``(batch, seq_len)``.

        Returns:
            A tuple of ``(embeddings, resonance)`` where *embeddings* has
            shape ``(batch, seq_len, embed_dim)`` and *resonance* has shape
            ``(batch, seq_len, seq_len)``.
        """
        batch_size, seq_len = token_ids.shape

        sem = self.semantic(token_ids)

        if self.config.use_phase_stream:
            ph = self.phase(token_ids)
            ph_proj = self.phase_proj(ph)
            blend = torch.sigmoid(self.blend)
            embeddings = (1 - blend) * sem + blend * ph_proj
        else:
            embeddings = sem

        positions = torch.arange(seq_len, device=token_ids.device)
        embeddings = embeddings + self.position(positions)

        resonance = self.get_resonance_matrix(token_ids)
        embeddings = self.dropout(embeddings)

        return embeddings, resonance


class ResonanceAttention(nn.Module):
    """Multi-head causal self-attention with a resonance bias.

    In addition to the standard scaled dot-product term, the attention
    logits receive an additive bias proportional to the pairwise resonance
    scores.  The per-head weighting of this bias is learnable.
    """

    def __init__(self, config: ResonanceConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.resonance_weight = nn.Parameter(
            torch.full((config.n_heads,), config.resonance_attn_weight)
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        resonance: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply resonance-biased multi-head attention.

        Args:
            x: Input tensor ``(batch, seq_len, embed_dim)``.
            resonance: Pairwise resonance matrix ``(batch, seq_len, seq_len)``.
            mask: Optional causal mask broadcastable to
                ``(batch, n_heads, seq_len, seq_len)``.

        Returns:
            Attention output ``(batch, seq_len, embed_dim)``.
        """
        batch_size, seq_len, _ = x.shape

        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Add resonance bias: broadcast across heads with per-head weights
        if self.config.use_resonance_bias:
            attn = attn + resonance.unsqueeze(1) * self.resonance_weight.view(
                1, self.n_heads, 1, 1
            )

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(out)


class ResonanceBlock(nn.Module):
    """A single transformer block for the Resonance model.

    Identical structure to ``StandardBlock`` except that the attention
    sub-layer uses ``ResonanceAttention`` and therefore requires the
    *resonance* matrix as an additional input.
    """

    def __init__(self, config: ResonanceConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = ResonanceAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        resonance: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through one resonance transformer block.

        Args:
            x: Input tensor ``(batch, seq_len, embed_dim)``.
            resonance: Pairwise resonance matrix ``(batch, seq_len, seq_len)``.
            mask: Optional causal attention mask.

        Returns:
            Updated tensor of the same shape.
        """
        x = x + self.attn(self.ln1(x), resonance, mask)
        x = x + self.ff(self.ln2(x))
        return x


class ResonanceTransformer(nn.Module):
    """Full Resonance transformer language model.

    Uses :class:`ResonanceEmbedding` to produce dual semantic/phase
    representations, applies a stack of :class:`ResonanceBlock` layers,
    and projects back to vocabulary logits.  The output projection is
    tied to the *semantic* embedding weights.
    """

    def __init__(
        self,
        config: ResonanceConfig,
        rhyme_index: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.embedding = ResonanceEmbedding(config, rhyme_index)
        self.blocks = nn.ModuleList(
            [ResonanceBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_final = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.semantic.weight

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)),
        )

        self.apply(self._init_weights)
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module) -> None:
        """Standard GPT-style weight initialisation."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass and optional next-token loss computation.

        Args:
            input_ids: Token indices ``(batch, seq_len)``.
            labels: Optional target token indices of the same shape.

        Returns:
            Dictionary with at least ``"logits"``.  Optionally ``"loss"``.
        """
        batch_size, seq_len = input_ids.shape

        x, resonance = self.embedding(input_ids)
        mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0)

        for block in self.blocks:
            x = block(x, resonance, mask)

        logits = self.lm_head(self.ln_final(x))

        result: dict[str, Any] = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return result
