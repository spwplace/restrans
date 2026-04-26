"""Gemma4 architecture — dense, MoE, and resonant variants.

Clean PyTorch training implementation extracted from the vLLM inference codebase.
Supports:
  - Full / sliding-window dual attention
  - Heavy RMSNorm (4× per layer)
  - QK norm with per-head learnable scales
  - Per-Layer Embeddings (PLE)
  - YOCO KV sharing
  - Attention logit soft-capping
  - GELUAndMul FFN (Gemma-style)
  - Gemma4-specific MoE router (RMSNorm → per-dim scale → top-k)
  - Resonance bias injected before soft-capping
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    RMSNorm,
    RoPE,
    Gemma4MLP,
    ModernResonanceEmbedding,
    make_causal_mask,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Gemma4Config:
    """Training-friendly configuration for Gemma4 variants."""

    vocab_size: int = 256000
    max_seq_len: int = 128000
    embed_dim: int = 2560
    n_layers: int = 40
    n_heads: int = 20
    n_kv_heads: int = 10           # GQA
    ff_dim: int = 10240
    dropout: float = 0.0
    rmsnorm_eps: float = 1e-6
    # Dual attention
    use_sliding_window: bool = True
    sliding_window_size: int = 4096
    sliding_window_every: int = 2  # every Nth layer is sliding
    # QK norm
    use_qk_norm: bool = True
    # PLE (Per-Layer Embeddings)
    use_ple: bool = True
    # YOCO KV sharing
    use_yoco_kv_share: bool = True
    num_kv_shared_layers: int = 2
    # Attention soft-capping
    attn_logits_soft_cap: float | None = 100.0
    # MoE options
    use_moe: bool = False
    moe_num_experts: int = 4
    moe_top_k: int = 2
    # Resonance options
    use_resonance: bool = False
    n_frequencies: int = 32


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def make_sliding_window_mask(
    seq_len: int, window_size: int, device: torch.device
) -> torch.Tensor:
    """Binary causal mask with sliding window: [1, 1, seq_len, seq_len]."""
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device))
    window = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=-window_size + 1)
    return (causal * window).view(1, 1, seq_len, seq_len)


def _get_layer_types(config: Gemma4Config) -> list[str]:
    """Return attention type for each layer."""
    if not config.use_sliding_window:
        return ["full_attention"] * config.n_layers
    return [
        "sliding_attention" if (i + 1) % config.sliding_window_every == 0 else "full_attention"
        for i in range(config.n_layers)
    ]


def _get_kv_donor_indices(
    config: Gemma4Config, layer_types: list[str]
) -> dict[int, int | None]:
    """Map layer index → donor layer index for YOCO KV sharing."""
    donors: dict[int, int | None] = {}
    if not config.use_yoco_kv_share or config.num_kv_shared_layers <= 0:
        for i in range(config.n_layers):
            donors[i] = None
        return donors

    first_shared = max(0, config.n_layers - config.num_kv_shared_layers)
    for i in range(config.n_layers):
        if i < first_shared:
            donors[i] = None
            continue
        lt = layer_types[i]
        donor: int | None = None
        for j in range(first_shared - 1, -1, -1):
            if layer_types[j] == lt:
                donor = j
                break
        donors[i] = donor
    return donors


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Gemma4Attention(nn.Module):
    """Gemma4 GQA attention with QK norm, RoPE, soft-capping, and optional resonance.

    Supports YOCO KV sharing: when ``kv_donor_idx`` is set, K/V are retrieved
    from a donor layer's cache rather than projected locally.
    """

    def __init__(
        self,
        config: Gemma4Config,
        layer_idx: int,
        is_sliding: bool,
        kv_donor_idx: int | None = None,
        use_resonance: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = is_sliding
        self.kv_donor_idx = kv_donor_idx
        self.is_kv_shared = kv_donor_idx is not None
        self.use_resonance = use_resonance

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.embed_dim // config.n_heads
        # Gemma4 uses scaling=1.0; Q/K norms handle scale implicitly.
        self.scaling = 1.0

        self.q_proj = nn.Linear(config.embed_dim, self.n_heads * self.head_dim, bias=False)
        if not self.is_kv_shared:
            self.k_proj = nn.Linear(config.embed_dim, self.n_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.embed_dim, self.n_kv_heads * self.head_dim, bias=False)
        else:
            self.k_proj = None
            self.v_proj = None
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.embed_dim, bias=False)

        # Q/K/V norms — applied per-head.
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rmsnorm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rmsnorm_eps)
            self.v_norm = RMSNorm(self.head_dim, eps=config.rmsnorm_eps, has_weight=False)
        else:
            self.q_norm = self.k_norm = self.v_norm = None

        # RoPE — different base for sliding vs full (default 10k for both).
        rope_base = 10000.0 if not is_sliding else getattr(config, "sliding_rope_theta", 10000.0)
        self.rope = RoPE(self.head_dim, max_seq_len=config.max_seq_len, base=rope_base)

        self.soft_cap = config.attn_logits_soft_cap
        self.dropout_p = config.dropout

        if self.use_resonance:
            self.resonance_weight = nn.Parameter(torch.full((self.n_heads,), 0.1))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        mask: torch.Tensor | None = None,
        resonance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            kv_cache: mutable dict mapping layer_idx → (k, v)
            mask: binary causal mask [1, 1, seq_len, seq_len]
            resonance: optional resonance matrix [batch, seq_len, seq_len]
        """
        B, S, _ = x.shape

        q = self.q_proj(x)

        # -----------------------------------------------------------------
        # KV path: shared layers pull from donor cache.
        # -----------------------------------------------------------------
        if self.is_kv_shared and kv_cache is not None and self.kv_donor_idx in kv_cache:
            k, v = kv_cache[self.kv_donor_idx]
        else:
            assert self.k_proj is not None and self.v_proj is not None
            k = self.k_proj(x)
            v = self.v_proj(x)

            if self.k_norm is not None:
                k = self.k_norm(k.view(B, S, self.n_kv_heads, self.head_dim)).view(B, S, -1)
                v = self.v_norm(v.view(B, S, self.n_kv_heads, self.head_dim)).view(B, S, -1)

            if kv_cache is not None:
                kv_cache[self.layer_idx] = (k, v)

        if self.q_norm is not None:
            q = self.q_norm(q.view(B, S, self.n_heads, self.head_dim)).view(B, S, -1)

        # Reshape to [B, H, S, D] for multi-head attention.
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        q = self.rope(q, S)
        k = self.rope(k, S)

        # Repeat K/V heads for GQA.
        if self.n_kv_heads < self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # Attention scores.
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Resonance bias (added BEFORE soft-capping).
        if self.use_resonance and resonance is not None:
            attn = attn + resonance.unsqueeze(1) * self.resonance_weight.view(1, self.n_heads, 1, 1)

        # Soft-capping.
        if self.soft_cap is not None:
            attn = torch.tanh(attn / self.soft_cap) * self.soft_cap

        # Causal / sliding-window mask.
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout_p, training=self.training)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

class Gemma4MoERouter(nn.Module):
    """Gemma4 MoE router.

    Preprocessing: RMSNorm (no weight) → scale by ``hidden_size**-0.5``
    → learned per-dim scale → linear projection to num_experts.
    """

    def __init__(self, dim: int, num_experts: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = RMSNorm(dim, eps=eps, has_weight=False)
        self.scale = nn.Parameter(torch.ones(dim))
        self.register_buffer("root_size", torch.tensor(dim ** -0.5), persistent=False)
        self.proj = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x * self.root_size.to(x.dtype)
        x = x * self.scale.to(x.dtype)
        return self.proj(x)


class Gemma4MoELayer(nn.Module):
    """Gemma4 sparse MoE layer that replaces the dense FFN.

    Routing: softmax over ALL experts → top-k → renormalize.
    Per-expert output scales are folded into routing weights.
    """

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.router = Gemma4MoERouter(config.embed_dim, self.num_experts, eps=config.rmsnorm_eps)
        self.per_expert_scale = nn.Parameter(torch.ones(self.num_experts))
        self.experts = nn.ModuleList([
            Gemma4MLP(config.embed_dim, config.ff_dim, bias=False)
            for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        router_logits = self.router(x)  # [B, S, E]

        # Softmax over ALL experts, then top-k.
        all_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(all_probs, self.top_k, dim=-1)

        # Renormalize top-k.
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Fold per-expert scales into top-k weights.
        expert_scales = self.per_expert_scale[topk_indices]  # [B, S, K]
        topk_probs = topk_probs * expert_scales.to(topk_probs.dtype)

        x_flat = x.view(-1, D)
        output = torch.zeros_like(x_flat)
        expert_counts = torch.zeros(self.num_experts, device=x.device)

        for expert_idx, expert in enumerate(self.experts):
            mask = (topk_indices == expert_idx).any(dim=-1)  # [B, S]
            flat_mask = mask.view(-1)
            count = flat_mask.sum().item()
            expert_counts[expert_idx] = count
            if count == 0:
                continue

            expert_input = x_flat[flat_mask]
            expert_out = expert(expert_input)

            # Weighted contribution of this expert.
            pos_mask = (topk_indices == expert_idx)  # [B, S, K]
            weights = (pos_mask.float() * topk_probs).sum(dim=-1)  # [B, S]
            flat_weights = weights.view(-1)[flat_mask].unsqueeze(-1)
            output[flat_mask] = output[flat_mask] + expert_out * flat_weights

        # Switch-style load-balancing aux loss.
        f = expert_counts / max(expert_counts.sum(), 1)
        P = all_probs.mean(dim=[0, 1])  # [E]
        aux_loss = self.num_experts * (f * P).sum()

        return output.view(B, S, D), aux_loss


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class Gemma4DecoderLayer(nn.Module):
    """Single Gemma4 decoder layer with heavy normalization."""

    def __init__(
        self,
        config: Gemma4Config,
        layer_idx: int,
        is_sliding: bool,
        kv_donor_idx: int | None,
        use_moe: bool = False,
        use_resonance: bool = False,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = is_sliding

        # Attention.
        self.self_attn = Gemma4Attention(
            config,
            layer_idx=layer_idx,
            is_sliding=is_sliding,
            kv_donor_idx=kv_donor_idx,
            use_resonance=use_resonance,
        )

        # FFN / MoE.
        self.use_moe = use_moe
        if use_moe:
            self.mlp = Gemma4MoELayer(config)
        else:
            self.mlp = Gemma4MLP(config.embed_dim, config.ff_dim, bias=False)

        # Heavy normalization: 4 RMSNorms per layer.
        self.input_layernorm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.post_attention_layernorm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)

        # PLE gating & projection inside the layer.
        if config.use_ple:
            ple_dim = max(64, config.embed_dim // 8)
            self.ple_dim = ple_dim
            self.ple_gate = nn.Linear(config.embed_dim, ple_dim, bias=False)
            self.ple_proj = nn.Linear(ple_dim, config.embed_dim, bias=False)
            self.ple_norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        else:
            self.ple_gate = None
            self.ple_proj = None
            self.ple_norm = None

        # Layer scalar (loaded from checkpoint; init to 1.0).
        self.register_buffer("layer_scalar", torch.ones(1), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor | None = None,
        kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        mask: torch.Tensor | None = None,
        resonance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (hidden_states, aux_loss) where aux_loss is 0 for dense layers.
        """
        # -----------------------------------------------------------------
        # Attention sub-layer.
        # -----------------------------------------------------------------
        residual = hidden_states
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(
            hidden_states,
            kv_cache=kv_cache,
            mask=mask,
            resonance=resonance,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        # -----------------------------------------------------------------
        # FFN / MoE sub-layer.
        # -----------------------------------------------------------------
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(residual)
        if self.use_moe:
            hidden_states, aux_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            aux_loss = hidden_states.new_tensor(0.0)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        # -----------------------------------------------------------------
        # Per-Layer Embedding (PLE).
        # -----------------------------------------------------------------
        if per_layer_input is not None and self.ple_gate is not None:
            gate = self.ple_gate(hidden_states)
            gate = F.gelu(gate, approximate="tanh")
            gated = gate * per_layer_input
            ple_out = self.ple_proj(gated)
            ple_out = self.ple_norm(ple_out)
            hidden_states = hidden_states + ple_out

        # Layer scalar.
        hidden_states = hidden_states * self.layer_scalar

        return hidden_states, aux_loss


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class _Gemma4Base(nn.Module):
    """Shared base for all Gemma4 model variants."""

    def __init__(
        self,
        config: Gemma4Config,
        use_moe: bool = False,
        use_resonance: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_moe = use_moe
        self.use_resonance = use_resonance

        # Embeddings.
        if use_resonance:
            self.embed = ModernResonanceEmbedding(
                vocab_size=config.vocab_size,
                embed_dim=config.embed_dim,
                n_frequencies=config.n_frequencies,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )
        else:
            self.embed = nn.Embedding(config.vocab_size, config.embed_dim)

        # Per-Layer Embeddings (PLE) — model-level table.
        if config.use_ple:
            ple_dim = max(64, config.embed_dim // 8)
            self.ple_dim = ple_dim
            self.ple_embed = nn.Embedding(config.vocab_size, config.n_layers * ple_dim)
            self.register_buffer("ple_scale", torch.tensor(ple_dim ** 0.5), persistent=False)
        else:
            self.ple_embed = None

        # Decoder layers.
        layer_types = _get_layer_types(config)
        kv_donors = _get_kv_donor_indices(config, layer_types)
        self.layers = nn.ModuleList([
            Gemma4DecoderLayer(
                config,
                layer_idx=i,
                is_sliding=(layer_types[i] == "sliding_attention"),
                kv_donor_idx=kv_donors[i],
                use_moe=use_moe,
                use_resonance=use_resonance,
            )
            for i in range(config.n_layers)
        ])

        # Final norm.
        self.norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)

        # LM head (tied with semantic embedding).
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        if use_resonance:
            self.lm_head.weight = self.embed.semantic.weight
        else:
            self.lm_head.weight = self.embed.weight

        self.embed_scale = math.sqrt(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "norm" in name or "Norm" in name:
                continue
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        # Re-init embeddings to small std.
        if self.use_resonance:
            nn.init.normal_(self.embed.semantic.weight, std=0.02)
        else:
            nn.init.normal_(self.embed.weight, std=0.02)

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """
        Args:
            input_ids: [batch, seq_len]
            labels: optional [batch, seq_len] token IDs for loss computation.
        Returns:
            Dict with ``logits``, ``loss`` (optional), and ``aux_loss`` (MoE only).
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Token embedding + optional resonance matrix.
        if self.use_resonance:
            h, resonance = self.embed(input_ids)
        else:
            h = self.embed(input_ids)
            resonance = None

        h = h * self.embed_scale
        h = self.dropout(h)

        # PLE inputs.
        if self.ple_embed is not None:
            ple = self.ple_embed(input_ids) * self.ple_scale  # [B, S, n_layers * ple_dim]
            ple = ple.view(B, S, self.config.n_layers, self.ple_dim)
        else:
            ple = None

        # Pre-compute masks.
        full_mask = make_causal_mask(S, device)
        sw_mask = None
        if self.config.use_sliding_window:
            sw_mask = make_sliding_window_mask(S, self.config.sliding_window_size, device)

        # Run decoder layers.
        kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = (
            {} if self.config.use_yoco_kv_share else None
        )
        total_aux_loss = h.new_tensor(0.0)

        for i, layer in enumerate(self.layers):
            mask = sw_mask if layer.is_sliding else full_mask

            # Mask resonance to sliding window when needed.
            layer_resonance = resonance
            if resonance is not None and layer.is_sliding and sw_mask is not None:
                layer_resonance = resonance * sw_mask.view(S, S)

            layer_ple = ple[:, :, i, :] if ple is not None else None

            h, aux = layer(
                h,
                per_layer_input=layer_ple,
                kv_cache=kv_cache,
                mask=mask,
                resonance=layer_resonance,
            )
            total_aux_loss = total_aux_loss + aux

        h = self.norm(h)
        logits = self.lm_head(h)

        loss: torch.Tensor | None = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        result: dict[str, torch.Tensor | None] = {"logits": logits, "loss": loss}
        if self.use_moe:
            result["aux_loss"] = total_aux_loss
        return result


# ---------------------------------------------------------------------------
# Public model classes
# ---------------------------------------------------------------------------

class Gemma4Transformer(_Gemma4Base):
    """Dense Gemma4 transformer (standard, non-MoE, non-resonant)."""

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__(config, use_moe=False, use_resonance=False)


class Gemma4MoE(_Gemma4Base):
    """Sparse Gemma4 transformer with MoE layers replacing dense FFN."""

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__(config, use_moe=True, use_resonance=False)


class ResonantGemma4(_Gemma4Base):
    """Dense Gemma4 with resonance embeddings and resonance-biased attention."""

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__(config, use_moe=False, use_resonance=True)


class ResonantGemma4MoE(_Gemma4Base):
    """Sparse Gemma4 with both resonance attention and MoE FFN."""

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__(config, use_moe=True, use_resonance=True)
