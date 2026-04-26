"""
Gemma4 Transformer — Dense + MoE + Resonance Variants
======================================================

Clean PyTorch training implementation extracted from vLLM's inference codebase.

Key Gemma4 features preserved:
  - Dual attention: full + sliding window (different head_dim per type)
  - Heavy RMSNorm: input, post-attn, pre-ffn, post-ffn
  - QK norm (learnable weight), V norm (no weight)
  - GELUAndMul activation
  - PLE (Per-Layer Embeddings)
  - Attention logit soft-capping
  - Layer scalar buffers
  - Custom MoE router with per-dim scaling + per-expert output scales
  - YOCO-style KV sharing (simplified for training)

Resonance integration:
  - Resonance bias added to attention logits BEFORE soft-capping
  - Sliding window layers mask resonance to local window
  - PLE is left untouched (separate path)
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
    GELUAndMul,
    Gemma4MLP,
    ModernResonanceEmbedding,
    get_preset_values,
    make_causal_mask,
    MoELayer,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Gemma4Config:
    vocab_size: int = 256000
    max_seq_len: int = 128000
    embed_dim: int = 2560
    n_layers: int = 40
    n_heads: int = 20
    n_kv_heads: int = 10
    head_dim: int = 256          # for sliding attention
    global_head_dim: int = 256   # for full attention (can differ)
    ff_dim: int = 10240
    dropout: float = 0.0
    rmsnorm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_local_theta: float = 10000.0
    # Layer types: list of "full_attention" or "sliding_attention"
    layer_types: list[str] | None = None
    sliding_window: int = 4096
    # QK norm
    use_qk_norm: bool = True
    # Attention soft-capping
    attn_logits_soft_cap: float | None = 100.0
    # PLE
    use_ple: bool = True
    hidden_size_per_layer_input: int = 256
    vocab_size_per_layer_input: int | None = None
    # YOCO KV sharing
    use_yoco_kv_share: bool = True
    num_kv_shared_layers: int = 2
    use_double_wide_mlp: bool = True
    # MoE
    use_moe: bool = False
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_ff_dim: int = 10240
    # Resonance
    use_resonance: bool = False
    n_frequencies: int = 32
    resonance_kernel: str = "cosine"
    phase_embedding: str = "real"
    bias_mode: str = "additive"
    init_preset: str = "default"


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Gemma4Attention(nn.Module):
    """GQA attention with QK norm, V norm, RoPE, sliding window, and soft-capping."""

    def __init__(self, config: Gemma4Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx] if config.layer_types else "full_attention"
        self.is_sliding = self.layer_type == "sliding_attention"
        self.is_full_attention = self.layer_type == "full_attention"

        # Head dim can differ per layer type
        self.head_dim = config.global_head_dim if self.is_full_attention else config.head_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.scaling = 1.0  # Gemma4 uses 1.0, norms handle scaling

        # Projections
        self.q_proj = nn.Linear(config.embed_dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.embed_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.embed_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.embed_dim, bias=False)

        # QK norms
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rmsnorm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rmsnorm_eps)
            self.v_norm = RMSNorm(self.head_dim, eps=config.rmsnorm_eps, has_weight=False)
        else:
            self.q_norm = self.k_norm = self.v_norm = None

        # RoPE
        rope_theta = config.rope_local_theta if self.is_sliding else config.rope_theta
        self.rope = RoPE(self.head_dim, max_seq_len=config.max_seq_len, base=rope_theta)

        # KV sharing config
        self.is_kv_shared = False
        self.kv_share_source_idx = -1
        if config.use_yoco_kv_share and config.num_kv_shared_layers > 0:
            first_shared = config.n_layers - config.num_kv_shared_layers
            if layer_idx >= first_shared > 0:
                self.is_kv_shared = True
                # Find last non-shared layer of same type
                for i in range(first_shared - 1, -1, -1):
                    if config.layer_types[i] == self.layer_type:
                        self.kv_share_source_idx = i
                        break

        # Resonance weight
        if config.use_resonance:
            _, attn_weight = get_preset_values(config.init_preset)
            self.resonance_weight = nn.Parameter(torch.full((config.n_heads,), attn_weight))
        else:
            self.resonance_weight = None

        # Soft-capping
        self.attn_logits_soft_cap = config.attn_logits_soft_cap

    def forward(
        self,
        x: torch.Tensor,
        resonance: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if not self.is_kv_shared:
            k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        else:
            # YOCO: reuse K/V from source layer
            if kv_cache is not None and self.kv_share_source_idx in kv_cache:
                k, v = kv_cache[self.kv_share_source_idx]
            else:
                # Fallback: compute own K/V if source not cached
                k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply QK norms (per-head)
        if self.q_norm is not None:
            q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
            if not self.is_kv_shared:
                k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)
                v = self.v_norm(v.transpose(1, 2)).transpose(1, 2)

        # Apply RoPE
        q = self.rope(q, S)
        if not self.is_kv_shared:
            k = self.rope(k, S)

        # Cache K/V for YOCO sharing
        if kv_cache is not None and not self.is_kv_shared:
            kv_cache[self.layer_idx] = (k, v)

        # GQA: repeat K/V heads
        if self.n_kv_heads < self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Resonance bias
        if resonance is not None:
            if self.resonance_weight is not None:
                attn = attn + resonance.unsqueeze(1) * self.resonance_weight.view(1, self.n_heads, 1, 1)
            else:
                attn = attn + resonance.unsqueeze(1)

        # Soft-capping
        if self.attn_logits_soft_cap is not None:
            attn = torch.tanh(attn / self.attn_logits_soft_cap) * self.attn_logits_soft_cap

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# MoE Router & Layer
# ---------------------------------------------------------------------------

class Gemma4Router(nn.Module):
    """Custom router: RMSNorm(no weight) → root_size scale → per-dim scale → projection."""

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps, has_weight=False)
        self.register_buffer("root_size", torch.tensor(config.embed_dim ** -0.5), persistent=False)
        self.scale = nn.Parameter(torch.ones(config.embed_dim))
        self.proj = nn.Linear(config.embed_dim, config.moe_num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x * self.root_size.to(x.dtype)
        x = x * self.scale.to(x.dtype)
        return self.proj(x)


class Gemma4MoELayer(nn.Module):
    """Gemma4 MoE with custom routing and per-expert output scales."""

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.router = Gemma4Router(config)
        self.per_expert_scale = nn.Parameter(torch.ones(config.moe_num_experts))

        # Experts: each is a Gemma4MLP
        self.experts = nn.ModuleList([
            Gemma4MLP(config.embed_dim, config.moe_ff_dim, bias=False)
            for _ in range(config.moe_num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        router_logits = self.router(x)  # [B, S, num_experts]
        all_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(all_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Apply per-expert scales to routing weights
        scales = self.per_expert_scale.to(topk_probs.dtype)
        topk_probs = topk_probs * scales[topk_indices]

        x_flat = x.view(-1, D)
        output = torch.zeros_like(x_flat)
        expert_counts = torch.zeros(self.num_experts, device=x.device)

        for expert_idx, expert in enumerate(self.experts):
            mask = (topk_indices == expert_idx).any(dim=-1)
            flat_mask = mask.view(-1)
            count = flat_mask.sum().item()
            expert_counts[expert_idx] = count
            if count == 0:
                continue
            expert_input = x_flat[flat_mask]
            expert_out = expert(expert_input)
            pos_mask = (topk_indices == expert_idx)
            weights = (pos_mask.float() * topk_probs).sum(dim=-1)
            flat_weights = weights.view(-1)[flat_mask].unsqueeze(-1)
            output[flat_mask] = output[flat_mask] + expert_out * flat_weights

        # Load balancing loss
        f = expert_counts / max(expert_counts.sum(), 1)
        P = all_probs.mean(dim=[0, 1])
        aux_loss = self.num_experts * (f * P).sum()

        return output.view(B, S, D), aux_loss


# ---------------------------------------------------------------------------
# Decoder Block
# ---------------------------------------------------------------------------

class Gemma4Block(nn.Module):
    """Gemma4 decoder layer with attention, MLP/MoE, PLE, and 4 RMSNorms."""

    def __init__(self, config: Gemma4Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.use_moe = config.use_moe
        self.use_ple = config.use_ple and config.hidden_size_per_layer_input > 0

        # Attention
        self.self_attn = Gemma4Attention(config, layer_idx)

        # MLP (always present; MoE is parallel/additional)
        first_kv_shared = config.n_layers - config.num_kv_shared_layers if config.use_yoco_kv_share else config.n_layers
        is_kv_shared = layer_idx >= first_kv_shared > 0
        layer_ff_dim = config.ff_dim * (2 if (config.use_double_wide_mlp and is_kv_shared) else 1)
        self.mlp = Gemma4MLP(config.embed_dim, layer_ff_dim, bias=False)

        # MoE (parallel to MLP)
        if self.use_moe:
            self.moe = Gemma4MoELayer(config)
            self.post_ff_norm_1 = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
            self.post_ff_norm_2 = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
            self.pre_ff_norm_2 = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        else:
            self.moe = None

        # Norms
        self.input_layernorm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.post_attention_layernorm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)

        # PLE
        if self.use_ple:
            self.ple_gate = nn.Linear(config.embed_dim, config.hidden_size_per_layer_input, bias=False)
            self.ple_proj = nn.Linear(config.hidden_size_per_layer_input, config.embed_dim, bias=False)
            self.post_ple_norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        else:
            self.ple_gate = None

        # Layer scalar
        self.register_buffer("layer_scalar", torch.ones(1), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        per_layer_input: torch.Tensor | None = None,
        resonance: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Attention path
        residual = x
        x = self.input_layernorm(residual)
        x = self.self_attn(x, resonance=resonance, mask=mask, kv_cache=kv_cache)
        x = self.post_attention_layernorm(x)
        x = x + residual

        # FFN path
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)

        if self.use_moe and self.moe is not None:
            x1 = self.post_ff_norm_1(x)
            router_logits = self.moe.router(residual)
            x2 = self.pre_ff_norm_2(residual)
            x2, aux_loss = self.moe(x2)  # Use the MoE layer directly
            x2 = self.post_ff_norm_2(x2)
            x = x1 + x2
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        x = self.post_feedforward_layernorm(x)
        x = x + residual

        # PLE
        if self.use_ple and per_layer_input is not None and self.ple_gate is not None:
            gate = self.ple_gate(x)
            gate = F.gelu(gate, approximate="tanh")
            gated = gate * per_layer_input
            ple_out = self.ple_proj(gated)
            ple_out = self.post_ple_norm(ple_out)
            x = x + ple_out

        # Layer scalar
        x = x * self.layer_scalar

        return x, aux_loss


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class Gemma4Transformer(nn.Module):
    """Gemma4 causal language model (dense or MoE)."""

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.config = config

        # Default layer types if not specified
        if config.layer_types is None:
            config.layer_types = [
                "sliding_attention" if i % config.sliding_window == 0 else "full_attention"
                for i in range(config.n_layers)
            ]

        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_dim)
        self.register_buffer("normalizer", torch.tensor(config.embed_dim ** 0.5), persistent=False)

        # PLE embeddings
        self.use_ple = config.use_ple and config.hidden_size_per_layer_input > 0
        if self.use_ple:
            ple_vocab = config.vocab_size_per_layer_input or config.vocab_size
            total_ple_dim = config.hidden_size_per_layer_input * config.n_layers
            self.embed_tokens_per_layer = nn.Embedding(ple_vocab, total_ple_dim)
            self.ple_model_proj = nn.Linear(config.embed_dim, total_ple_dim, bias=False)
            self.ple_proj_norm = RMSNorm(config.hidden_size_per_layer_input, eps=config.rmsnorm_eps)
            self.register_buffer("ple_scale", torch.tensor(config.hidden_size_per_layer_input ** 0.5), persistent=False)
            self.register_buffer("ple_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False)
            self.register_buffer("ple_proj_scale", torch.tensor(config.embed_dim ** -0.5), persistent=False)
        else:
            self.embed_tokens_per_layer = None

        # Blocks
        self.blocks = nn.ModuleList([
            Gemma4Block(config, i) for i in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _compute_ple_inputs(self, input_ids: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Compute per-layer embeddings [B, S, n_layers, ple_dim]."""
        if not self.use_ple:
            return None
        B, S = input_ids.shape
        # Token embeddings per layer
        ple_tokens = self.embed_tokens_per_layer(input_ids) * self.ple_scale  # [B, S, total_dim]
        # Project from hidden states
        proj = self.ple_model_proj(hidden_states) * self.ple_proj_scale  # [B, S, total_dim]
        # Combine
        combined = (ple_tokens + proj) * self.ple_input_scale
        # Reshape to per-layer
        ple = combined.view(B, S, self.config.n_layers, self.config.hidden_size_per_layer_input)
        # Normalize per-layer dim
        ple = self.ple_proj_norm(ple)
        return ple

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        h = self.embed_tokens(input_ids) * self.normalizer.to(self.embed_tokens.weight.dtype)

        # PLE inputs
        ple_inputs = self._compute_ple_inputs(input_ids, h)

        # Causal mask
        mask = make_causal_mask(S, device)
        if self.config.sliding_window > 0:
            # For sliding window, we still use full causal mask but the attention
            # module will apply additional sliding window masking
            pass

        # KV cache for YOCO sharing
        kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        total_aux_loss = torch.tensor(0.0, device=device)

        for i, block in enumerate(self.blocks):
            ple_input = ple_inputs[:, :, i, :] if ple_inputs is not None else None
            h, aux_loss = block(
                h,
                per_layer_input=ple_input,
                mask=mask,
                kv_cache=kv_cache,
            )
            total_aux_loss = total_aux_loss + aux_loss

        h = self.norm(h)
        logits = self.lm_head(h)

        result: dict[str, Any] = {"logits": logits, "aux_loss": total_aux_loss}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return result


# ---------------------------------------------------------------------------
# Resonant Variants
# ---------------------------------------------------------------------------

class ResonantGemma4(nn.Module):
    """Gemma4 with resonance-biased attention."""

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.config = config

        if config.layer_types is None:
            config.layer_types = [
                "sliding_attention" if i % max(config.sliding_window, 1) == 0 else "full_attention"
                for i in range(config.n_layers)
            ]

        self.embedding = ModernResonanceEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            n_frequencies=config.n_frequencies,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            phase_embedding_type=config.phase_embedding,
            resonance_kernel=config.resonance_kernel,
            init_preset=config.init_preset,
        )
        self.register_buffer("normalizer", torch.tensor(config.embed_dim ** 0.5), persistent=False)

        # PLE
        self.use_ple = config.use_ple and config.hidden_size_per_layer_input > 0
        if self.use_ple:
            ple_vocab = config.vocab_size_per_layer_input or config.vocab_size
            total_ple_dim = config.hidden_size_per_layer_input * config.n_layers
            self.embed_tokens_per_layer = nn.Embedding(ple_vocab, total_ple_dim)
            self.ple_model_proj = nn.Linear(config.embed_dim, total_ple_dim, bias=False)
            self.ple_proj_norm = RMSNorm(config.hidden_size_per_layer_input, eps=config.rmsnorm_eps)
            self.register_buffer("ple_scale", torch.tensor(config.hidden_size_per_layer_input ** 0.5), persistent=False)
            self.register_buffer("ple_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False)
            self.register_buffer("ple_proj_scale", torch.tensor(config.embed_dim ** -0.5), persistent=False)
        else:
            self.embed_tokens_per_layer = None

        # Blocks (reuse Gemma4Block but with resonance passed)
        self.blocks = nn.ModuleList([
            Gemma4Block(config, i) for i in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.semantic.weight

        self.apply(self._init_weights)
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _compute_ple_inputs(self, input_ids: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if not self.use_ple:
            return None
        B, S = input_ids.shape
        ple_tokens = self.embed_tokens_per_layer(input_ids) * self.ple_scale
        proj = self.ple_model_proj(hidden_states) * self.ple_proj_scale
        combined = (ple_tokens + proj) * self.ple_input_scale
        ple = combined.view(B, S, self.config.n_layers, self.config.hidden_size_per_layer_input)
        ple = self.ple_proj_norm(ple)
        return ple

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        B, S = input_ids.shape
        device = input_ids.device

        h, resonance = self.embedding(input_ids)
        h = h * self.normalizer.to(h.dtype)

        ple_inputs = self._compute_ple_inputs(input_ids, h)
        mask = make_causal_mask(S, device)
        kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        total_aux_loss = torch.tensor(0.0, device=device)

        for i, block in enumerate(self.blocks):
            ple_input = ple_inputs[:, :, i, :] if ple_inputs is not None else None
            h, aux_loss = block(
                h,
                per_layer_input=ple_input,
                resonance=resonance,
                mask=mask,
                kv_cache=kv_cache,
            )
            total_aux_loss = total_aux_loss + aux_loss

        h = self.norm(h)
        logits = self.lm_head(h)

        result: dict[str, Any] = {"logits": logits, "aux_loss": total_aux_loss}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return result


class Gemma4MoE(Gemma4Transformer):
    """Convenience wrapper: Gemma4Transformer with use_moe=True."""

    def __init__(self, config: Gemma4Config) -> None:
        config.use_moe = True
        super().__init__(config)


class ResonantGemma4MoE(ResonantGemma4):
    """Convenience wrapper: ResonantGemma4 with use_moe=True."""

    def __init__(self, config: Gemma4Config) -> None:
        config.use_moe = True
        super().__init__(config)
