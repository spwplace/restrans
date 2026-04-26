"""
LLaMA architecture for training — dense and MoE, standard and resonant variants.

Extracted from vLLM inference codebase and reimplemented in plain PyTorch
for small-scale training (5M–50M params on TinyStories).

Architecture variants
---------------------
* ``LlamaTransformer``    – dense standard (pre-norm, GQA, RoPE, SwiGLU)
* ``LlamaMoE``            – sparse standard (MoE every ``moe_layer_freq`` layers)
* ``ResonantLlama``       – dense with resonance bias attention
* ``ResonantLlamaMoE``    – sparse with resonance bias + MoE

All models return a dictionary from ``forward(input_ids, labels=None)``:
  * ``logits``              – [B, S, vocab_size]
  * ``loss`` (optional)     – cross-entropy when ``labels`` provided
  * ``aux_loss`` (MoE only) – load-balancing loss accumulated over MoE layers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    RMSNorm,
    RoPE,
    SwiGLU,
    MoELayer,
    ModernResonanceEmbedding,
    ResonanceBiasAttention,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LlamaConfig:
    """Configuration for LLaMA-family models."""

    vocab_size: int = 32000
    max_seq_len: int = 2048
    embed_dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8           # GQA
    ff_dim: int = 11008
    dropout: float = 0.0
    rmsnorm_eps: float = 1e-6
    rope_theta: float = 10000.0
    # Sliding window per layer (None = full causal attention)
    sliding_window: list[int | None] | None = None
    # MoE options
    use_moe: bool = False
    moe_layer_freq: int = 2       # every Nth layer is MoE
    moe_num_experts: int = 4
    moe_top_k: int = 2
    # Resonance options
    use_resonance: bool = False
    n_frequencies: int = 32
    resonance_kernel: str = "cosine"
    phase_embedding: str = "real"
    bias_mode: str = "additive"
    init_preset: str = "default"


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _make_sliding_window_mask(
    seq_len: int, window_size: int, device: torch.device
) -> torch.Tensor:
    """Create a [1, 1, seq_len, seq_len] causal + sliding-window mask.

    Each position can attend to itself and the previous ``window_size - 1``
    tokens.
    """
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    window = (
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        .tril()
        .triu(-(window_size - 1))
    )
    return (causal & window).view(1, 1, seq_len, seq_len)


# ---------------------------------------------------------------------------
# Standard Attention (RoPE + GQA + SDPA)
# ---------------------------------------------------------------------------

class _LlamaAttention(nn.Module):
    """Standard LLaMA attention with RoPE, GQA, and optional sliding window.

    Uses :func:`torch.nn.functional.scaled_dot_product_attention` for
    efficiency.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.dropout = nn.Dropout(config.dropout)

        self.q_proj = nn.Linear(
            config.embed_dim, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.embed_dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.embed_dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.embed_dim, bias=False
        )

        self.rope = RoPE(self.head_dim, config.max_seq_len, config.rope_theta)

        sw = config.sliding_window[layer_idx] if config.sliding_window else None
        self.sliding_window = sw

    def _get_mask(self, seq_len: int, device: torch.device) -> torch.Tensor | None:
        if self.sliding_window is not None and self.sliding_window > 0:
            return _make_sliding_window_mask(seq_len, self.sliding_window, device)
        return None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, S)
        k = self.rope(k, S)

        # Repeat K/V heads for GQA
        if self.n_kv_heads < self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        if mask is not None:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask.bool(),
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )

        out = out.transpose(1, 2).reshape(B, S, -1)
        out = self.dropout(out)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------

class _LlamaDenseLayer(nn.Module):
    """Pre-norm LLaMA decoder layer with dense SwiGLU FFN."""

    def __init__(self, config: LlamaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.attn = _LlamaAttention(config, layer_idx)
        self.ffn_norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.ffn = SwiGLU(config.embed_dim, config.ff_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.attn._get_mask(x.size(1), x.device)
        x = x + self.attn(self.attn_norm(x), mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x, x.new_tensor(0.0)


class _LlamaMoEDecoderLayer(nn.Module):
    """Pre-norm LLaMA decoder layer with MoE FFN."""

    def __init__(self, config: LlamaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.attn = _LlamaAttention(config, layer_idx)
        self.ffn_norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.ffn = MoELayer(
            dim=config.embed_dim,
            hidden_dim=config.ff_dim,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.attn._get_mask(x.size(1), x.device)
        x = x + self.attn(self.attn_norm(x), mask=mask)
        ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        return x, aux_loss


class _ResonantLlamaDenseLayer(nn.Module):
    """Pre-norm LLaMA decoder layer with resonance bias attention + dense FFN."""

    def __init__(self, config: LlamaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.attn = ResonanceBiasAttention(
            embed_dim=config.embed_dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            dropout=config.dropout,
            use_resonance=True,
            bias_mode=config.bias_mode,
            init_preset=config.init_preset,
            rope=RoPE(
                config.embed_dim // config.n_heads,
                config.max_seq_len,
                config.rope_theta,
            ),
        )
        self.ffn_norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.ffn = SwiGLU(config.embed_dim, config.ff_dim)

        sw = config.sliding_window[layer_idx] if config.sliding_window else None
        self.sliding_window = sw

    def _get_mask(self, seq_len: int, device: torch.device) -> torch.Tensor | None:
        if self.sliding_window is not None and self.sliding_window > 0:
            return _make_sliding_window_mask(seq_len, self.sliding_window, device)
        return None

    def forward(
        self, x: torch.Tensor, resonance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self._get_mask(x.size(1), x.device)
        x = x + self.attn(self.attn_norm(x), resonance=resonance, mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x, x.new_tensor(0.0)


class _ResonantLlamaMoEDecoderLayer(nn.Module):
    """Pre-norm LLaMA decoder layer with resonance bias attention + MoE FFN."""

    def __init__(self, config: LlamaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.attn = ResonanceBiasAttention(
            embed_dim=config.embed_dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            dropout=config.dropout,
            use_resonance=True,
            bias_mode=config.bias_mode,
            init_preset=config.init_preset,
            rope=RoPE(
                config.embed_dim // config.n_heads,
                config.max_seq_len,
                config.rope_theta,
            ),
        )
        self.ffn_norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.ffn = MoELayer(
            dim=config.embed_dim,
            hidden_dim=config.ff_dim,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
        )

        sw = config.sliding_window[layer_idx] if config.sliding_window else None
        self.sliding_window = sw

    def _get_mask(self, seq_len: int, device: torch.device) -> torch.Tensor | None:
        if self.sliding_window is not None and self.sliding_window > 0:
            return _make_sliding_window_mask(seq_len, self.sliding_window, device)
        return None

    def forward(
        self, x: torch.Tensor, resonance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self._get_mask(x.size(1), x.device)
        x = x + self.attn(self.attn_norm(x), resonance=resonance, mask=mask)
        ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        return x, aux_loss


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LlamaTransformer(nn.Module):
    """Dense standard LLaMA transformer.

    Pre-norm RMSNorm, GQA attention with RoPE, SwiGLU FFN, causal masking,
    and weight tying between token embedding and LM head.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            self.layers.append(_LlamaDenseLayer(config, i))
        self.norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.embed.weight

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, Any]:
        x = self.embed(input_ids)
        for layer in self.layers:
            x, _ = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        result: dict[str, Any] = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return result

    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


class LlamaMoE(nn.Module):
    """Sparse standard LLaMA transformer with MoE layers.

    Every ``moe_layer_freq``-th layer replaces the dense SwiGLU FFN with a
    :class:`~.common.MoELayer`. The model accumulates an auxiliary
    load-balancing loss across all MoE layers.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            is_moe = (i + 1) % config.moe_layer_freq == 0
            if is_moe:
                self.layers.append(_LlamaMoEDecoderLayer(config, i))
            else:
                self.layers.append(_LlamaDenseLayer(config, i))
        self.norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, Any]:
        x = self.embed(input_ids)
        aux_loss = x.new_tensor(0.0)
        for layer in self.layers:
            x, layer_aux = layer(x)
            aux_loss = aux_loss + layer_aux

        x = self.norm(x)
        logits = self.lm_head(x)

        result: dict[str, Any] = {"logits": logits, "aux_loss": aux_loss}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return result

    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


class ResonantLlama(nn.Module):
    """Dense LLaMA transformer with resonance bias attention.

    Replaces the standard token embedding with
    :class:`~.common.ModernResonanceEmbedding` and the standard attention
    with :class:`~.common.ResonanceBiasAttention`. The pairwise resonance
    matrix is computed once at the embedding layer and passed to every
    attention layer.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = ModernResonanceEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            n_frequencies=config.n_frequencies,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            phase_embedding_type=config.phase_embedding,
            resonance_kernel=config.resonance_kernel,
            init_preset=config.init_preset,
        )
        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            self.layers.append(_ResonantLlamaDenseLayer(config, i))
        self.norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        # Tie with semantic embedding
        self.lm_head.weight = self.embed.semantic.weight

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, Any]:
        x, resonance = self.embed(input_ids)
        for layer in self.layers:
            x, _ = layer(x, resonance)
        x = self.norm(x)
        logits = self.lm_head(x)

        result: dict[str, Any] = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return result

    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


class ResonantLlamaMoE(nn.Module):
    """Sparse LLaMA transformer with resonance bias attention and MoE layers.

    Combines :class:`ResonantLlama` attention with MoE FFN layers at every
    ``moe_layer_freq``-th position.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = ModernResonanceEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            n_frequencies=config.n_frequencies,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            phase_embedding_type=config.phase_embedding,
            resonance_kernel=config.resonance_kernel,
            init_preset=config.init_preset,
        )
        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            is_moe = (i + 1) % config.moe_layer_freq == 0
            if is_moe:
                self.layers.append(_ResonantLlamaMoEDecoderLayer(config, i))
            else:
                self.layers.append(_ResonantLlamaDenseLayer(config, i))
        self.norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.semantic.weight

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, Any]:
        x, resonance = self.embed(input_ids)
        aux_loss = x.new_tensor(0.0)
        for layer in self.layers:
            x, layer_aux = layer(x, resonance)
            aux_loss = aux_loss + layer_aux

        x = self.norm(x)
        logits = self.lm_head(x)

        result: dict[str, Any] = {"logits": logits, "aux_loss": aux_loss}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return result

    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
