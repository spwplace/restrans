"""
Common components for modern transformer architectures.
=======================================================

Shared building blocks extracted from vLLM reference implementations:
  - RMSNorm
  - RoPE (rotary position embeddings)
  - SwiGLU / GELUAndMul activations
  - MoE Router + MoE Layer
  - Resonance utilities (embedding, matrix computation)

All implemented in plain PyTorch — no vLLM/TP/quantization dependencies.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels import build_kernel
from ..phase_embeddings import build_phase_embedding
from ..bias_modes import build_bias_mode


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in LLaMA, Qwen, Gemma)."""

    def __init__(self, dim: int, eps: float = 1e-6, has_weight: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.has_weight = has_weight
        if has_weight:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.has_weight and self.weight is not None:
            norm = norm * self.weight
        return norm


# ---------------------------------------------------------------------------
# Position Embeddings
# ---------------------------------------------------------------------------

class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE) — neox style.

    Supports per-layer theta values (e.g. Gemma4 uses different theta for
    sliding vs full attention layers).
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin tables
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, head_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, head_dim]
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, num_heads, seq_len, head_dim].
        Returns:
            Rotated tensor of same shape.
        """
        cos = self.cos[:, :, :seq_len, :]
        sin = self.sin[:, :, :seq_len, :]
        return x * cos + self.rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (used in LLaMA, Qwen)."""

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)      # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)      # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)      # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GELUAndMul(nn.Module):
    """GELU-gated multiplication (used in Gemma4).

    Input is split in half along last dim; second half is gated by GELU
    and multiplied into the first half.
    """

    def __init__(self, approximate: str = "tanh") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate, approximate=self.approximate)


class Gemma4MLP(nn.Module):
    """Gemma4-style MLP: up-proj → GELUAndMul → down-proj.

    The up-projection doubles hidden size because GELUAndMul splits it.
    """

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(dim, hidden_dim * 2, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.act = GELUAndMul(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_up_proj(x)))


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

class MoERouter(nn.Module):
    """Top-k router for Mixture of Experts."""

    def __init__(self, dim: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            topk_probs: [batch, seq_len, top_k]
            topk_indices: [batch, seq_len, top_k]
            all_probs: [batch, seq_len, num_experts]
        """
        logits = self.gate(x)
        all_probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(all_probs, self.top_k, dim=-1)
        # Renormalize top-k probabilities
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return topk_probs, topk_indices, all_probs


class MoELayer(nn.Module):
    """Sparse Mixture of Experts layer.

    Replaces a dense MLP with a router + set of expert MLPs.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        expert_type: str = "swiglu",
        expert_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = MoERouter(dim, num_experts, top_k)

        if expert_type == "swiglu":
            self.experts = nn.ModuleList([
                SwiGLU(dim, hidden_dim, bias=expert_bias) for _ in range(num_experts)
            ])
        elif expert_type == "gemma4":
            self.experts = nn.ModuleList([
                Gemma4MLP(dim, hidden_dim, bias=expert_bias) for _ in range(num_experts)
            ])
        else:
            raise ValueError(f"Unknown expert_type: {expert_type}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            output: [batch, seq_len, dim]
            aux_loss: scalar load-balancing loss
        """
        B, S, D = x.shape
        topk_probs, topk_indices, all_probs = self.router(x)

        x_flat = x.view(-1, D)
        output = torch.zeros_like(x_flat)

        # Count tokens per expert for load-balancing
        expert_counts = torch.zeros(self.num_experts, device=x.device)

        for expert_idx, expert in enumerate(self.experts):
            # Mask of tokens that route to this expert
            mask = (topk_indices == expert_idx).any(dim=-1)  # [B, S]
            flat_mask = mask.view(-1)
            count = flat_mask.sum().item()
            expert_counts[expert_idx] = count

            if count == 0:
                continue

            expert_input = x_flat[flat_mask]
            expert_out = expert(expert_input)

            # Weight by the routing probability for this expert
            pos_mask = (topk_indices == expert_idx)  # [B, S, top_k]
            weights = (pos_mask.float() * topk_probs).sum(dim=-1)  # [B, S]
            flat_weights = weights.view(-1)[flat_mask].unsqueeze(-1)
            output[flat_mask] = output[flat_mask] + expert_out * flat_weights

        # Load balancing aux loss (Switch Transformer style)
        # f_i = fraction of tokens routed to expert i
        # P_i = mean routing probability to expert i
        f = expert_counts / max(expert_counts.sum(), 1)
        P = all_probs.mean(dim=[0, 1])  # [num_experts]
        aux_loss = self.num_experts * (f * P).sum()

        return output.view(B, S, D), aux_loss


# ---------------------------------------------------------------------------
# Initialization presets
# ---------------------------------------------------------------------------

_PRESET_VALUES: dict[str, tuple[float, float]] = {
    "default":     (0.3, 0.1),
    "wide":        (1.0, 0.3),
    "strong":      (1.2, 1.0),
    "very_strong": (2.0, 2.0),
    "normalized":  (0.3, 0.1),
}


def get_preset_values(preset: str) -> tuple[float, float]:
    """Return (phase_init_std, resonance_weight) for a given preset."""
    return _PRESET_VALUES.get(preset, (0.3, 0.1))


# ---------------------------------------------------------------------------
# Resonance utilities for modern architectures
# ---------------------------------------------------------------------------

class ModernResonanceEmbedding(nn.Module):
    """Dual semantic + phase embedding (standalone, config-free).

    Can be dropped into any architecture that expects a standard
    nn.Embedding-like interface plus a resonance matrix.

    Supports swappable phase embeddings and resonance kernels via
    the modular registries.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_frequencies: int = 32,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        phase_embedding_type: str = "real",
        resonance_kernel: str = "cosine",
        kernel_gamma: float = 1.0,
        kernel_rank: int | None = None,
        init_preset: str = "default",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_frequencies = n_frequencies
        self.resonance_kernel = resonance_kernel

        phase_init_std, _ = get_preset_values(init_preset)

        self.semantic = nn.Embedding(vocab_size, embed_dim)
        self.phase = build_phase_embedding(
            phase_embedding_type, vocab_size, n_frequencies, init_std=phase_init_std
        )
        self.phase_proj = nn.Linear(n_frequencies, embed_dim, bias=False)
        self.blend = nn.Parameter(torch.full((embed_dim,), 0.3))
        self.position = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.kernel = build_kernel(
            resonance_kernel, n_frequencies, gamma=kernel_gamma, rank=kernel_rank
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.semantic.weight, std=0.02)
        nn.init.normal_(self.phase_proj.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)

    def get_resonance_matrix(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute pairwise resonance via the configured kernel."""
        from ..phase_embeddings import ComplexAnglePhaseEmbedding
        if (
            self.resonance_kernel in ("complex_magnitude", "complex_real")
            and isinstance(self.phase, ComplexAnglePhaseEmbedding)
        ):
            phases = self.phase.to_complex(token_ids)  # [B, S, F] complex
        else:
            phases = self.phase(token_ids)  # [B, S, F]
        return self.kernel(phases)  # [B, S, S]

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (embeddings, resonance_matrix)."""
        B, S = token_ids.shape
        sem = self.semantic(token_ids)
        ph = self.phase(token_ids)
        ph_proj = self.phase_proj(ph)
        blend = torch.sigmoid(self.blend)
        embeddings = (1 - blend) * sem + blend * ph_proj
        pos = torch.arange(S, device=token_ids.device)
        embeddings = embeddings + self.position(pos)
        resonance = self.get_resonance_matrix(token_ids)
        embeddings = self.dropout(embeddings)
        return embeddings, resonance


class ResonanceBiasAttention(nn.Module):
    """Standard GQA attention with optional resonance bias and optional RoPE.

    This is a drop-in replacement for standard multi-head attention that
    accepts an optional resonance matrix and adds it to attention logits.
    When ``rope`` is provided, rotary position embeddings are applied to
    Q and K before the attention matmul (LLaMA-style).

    Supports swappable bias application modes via the bias mode registry.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        use_resonance: bool = True,
        rope: RoPE | None = None,
        bias_mode: str = "additive",
        init_preset: str = "default",
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = embed_dim // n_heads
        self.use_resonance = use_resonance
        self.scale = self.head_dim ** -0.5
        self.rope = rope

        # GQA: Q has n_heads, K/V have n_kv_heads
        self.q_proj = nn.Linear(embed_dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        if use_resonance:
            _, attn_weight = get_preset_values(init_preset)
            self.resonance_weight = nn.Parameter(torch.full((n_heads,), attn_weight))
            self.bias_mode = build_bias_mode(bias_mode, n_heads=n_heads)
        else:
            self.bias_mode = None

    def forward(
        self,
        x: torch.Tensor,
        resonance: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q = self.rope(q, S)
            k = self.rope(k, S)

        # Repeat K/V heads for GQA
        if self.n_kv_heads < self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.use_resonance and resonance is not None and self.bias_mode is not None:
            # resonance: [B, S, S] → broadcast to [B, n_heads, S, S]
            resonance_bias = resonance.unsqueeze(1) * self.resonance_weight.view(1, self.n_heads, 1, 1)
            attn = self.bias_mode(attn, resonance_bias, mask)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Causal mask utility
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Returns a [1, 1, seq_len, seq_len] causal mask (1 = attend, 0 = mask)."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)
