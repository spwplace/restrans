"""Qwen3.5 architecture in pure PyTorch.

Supports dense / MoE and standard / resonant variants with hybrid
(full + linear) attention, QK norm, attention output gating, and RoPE.

Architecture reference:
- Qwen3.5 dense:  pre-norm RMSNorm, hybrid attention (full GQA + linear),
  QK norm, attention output gating, SwiGLU FFN.
- Qwen3.5 MoE:   same attention but with shared-expert + sparse-expert FFN.
- Resonant variants add a ``ModernResonanceEmbedding`` and inject the
  resonance matrix into full-attention logits (same pattern as
  ``ResonanceBiasAttention``) or into linear-attention gating/state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    make_causal_mask,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Qwen3_5Config:
    """Hyper-parameters for the Qwen3.5 family."""

    vocab_size: int = 151936
    max_seq_len: int = 32768
    embed_dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    ff_dim: int = 11008
    dropout: float = 0.0
    rmsnorm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    # Hybrid attention
    use_linear_attention: bool = True
    linear_attention_layers: list[int] | None = None

    # QK norm
    use_qk_norm: bool = True

    # Attention output gating
    use_attn_gate: bool = True

    # MoE options
    use_moe: bool = False
    moe_layer_freq: int = 2
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_shared_expert: bool = True

    # Resonance options
    use_resonance: bool = False
    n_frequencies: int = 32
    linear_resonance_mode: str = "gate"  # "gate", "skip", or "state"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_linear_layer(config: Qwen3_5Config, layer_idx: int) -> bool:
    """Return ``True`` if *layer_idx* should use linear attention."""
    if not config.use_linear_attention:
        return False
    if config.linear_attention_layers is not None:
        return layer_idx in config.linear_attention_layers
    # Default: alternate, even layers = full, odd layers = linear
    return layer_idx % 2 == 1


def _is_moe_layer(config: Qwen3_5Config, layer_idx: int) -> bool:
    """Return ``True`` if *layer_idx* should use an MoE FFN."""
    if not config.use_moe:
        return False
    return (layer_idx + 1) % config.moe_layer_freq == 0


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class Qwen3_5FullAttention(nn.Module):
    """Full GQA self-attention with QK norm, RoPE, and optional output gate.

    Uses :func:`torch.nn.functional.scaled_dot_product_attention` for the
    core attention computation.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.embed_dim, config.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.embed_dim, config.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.embed_dim, config.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.n_heads * self.head_dim, config.embed_dim, bias=False
        )

        self.use_attn_gate = config.use_attn_gate
        if config.use_attn_gate:
            self.gate_proj = nn.Linear(
                config.embed_dim, config.n_heads * self.head_dim, bias=False
            )

        self.use_qk_norm = config.use_qk_norm
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rmsnorm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rmsnorm_eps)

        self.rope = RoPE(
            self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_theta,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        resonance: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``[batch, seq_len, embed_dim]``
            resonance: Unused (present for API compatibility).
            mask: Optional custom mask. If ``None``, causal masking is applied
                via ``scaled_dot_product_attention``.

        Returns:
            ``[batch, seq_len, embed_dim]``
        """
        B, S, _ = x.shape

        q = (
            self.q_proj(x)
            .view(B, S, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(B, S, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(B, S, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # QK norm (per-head, over head_dim)
        if self.use_qk_norm:
            q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)

        # RoPE
        q = self.rope(q, S)
        k = self.rope(k, S)

        # GQA: repeat K/V heads
        if self.n_kv_heads < self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # Scaled dot-product attention
        dropout_p = self.dropout.p if self.training else 0.0
        if mask is not None:
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p
            )
        else:
            attn = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=dropout_p
            )

        attn = attn.transpose(1, 2).reshape(B, S, -1)

        # Attention output gating
        if self.use_attn_gate:
            gate = torch.sigmoid(self.gate_proj(x))
            attn = attn * gate

        return self.o_proj(attn)


class Qwen3_5LinearAttention(nn.Module):
    """Simplified linear (recurrent-style) self-attention.

    Computes causal attention *without* softmax:

    .. math::
        o_t = \\sum_{j \\le t} (q_t \\cdot k_j) v_j
              / \\sum_{j \\le t} (q_t \\cdot k_j)

    For training this is implemented as a causal matrix multiplication,
    which is acceptable for small-scale experiments.

    Supports three resonance adaptations (controlled by
    ``config.linear_resonance_mode``):

    * ``"skip"``  – standard learned gate (or no gate).
    * ``"gate"``  – resonance-derived attention output gate.
    * ``"state"`` – boost the recurrent state update for tokens with strong
      self-resonance.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.embed_dim, config.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.embed_dim, config.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.embed_dim, config.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.n_heads * self.head_dim, config.embed_dim, bias=False
        )

        self.use_attn_gate = config.use_attn_gate
        if config.use_attn_gate:
            self.gate_proj = nn.Linear(
                config.embed_dim, config.n_heads * self.head_dim, bias=False
            )

        self.use_qk_norm = config.use_qk_norm
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rmsnorm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rmsnorm_eps)

        self.use_resonance = config.use_resonance
        if config.use_resonance:
            if config.linear_resonance_mode == "gate":
                self.resonance_gate_weight = nn.Parameter(torch.tensor(0.1))
            elif config.linear_resonance_mode == "state":
                self.resonance_lambda = nn.Parameter(torch.tensor(0.1))

        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("eps", torch.tensor(1e-6))

    def forward(
        self,
        x: torch.Tensor,
        resonance: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``[batch, seq_len, embed_dim]``
            resonance: Pairwise resonance matrix ``[batch, seq_len, seq_len]``.
            mask: Ignored (causal masking is always applied internally).

        Returns:
            ``[batch, seq_len, embed_dim]``
        """
        B, S, _ = x.shape

        q = (
            self.q_proj(x)
            .view(B, S, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(B, S, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(B, S, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.use_qk_norm:
            q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)

        # GQA: repeat K/V heads
        if self.n_kv_heads < self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # Causal linear attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]
        causal = torch.tril(torch.ones(S, S, device=x.device)).view(1, 1, S, S)
        scores = scores * causal

        # Resonance "state" mode: boost value contributions by self-resonance
        if (
            self.use_resonance
            and self.config.linear_resonance_mode == "state"
            and resonance is not None
        ):
            diag = resonance.diagonal(dim1=1, dim2=2)  # [B, S]
            boost = 1.0 + self.resonance_lambda * diag  # [B, S]
            v_boosted = v * boost.unsqueeze(1).unsqueeze(-1)  # [B, H, S, d]
            out = torch.matmul(scores, v_boosted)  # [B, H, S, d]
        else:
            out = torch.matmul(scores, v)  # [B, H, S, d]

        denom = scores.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        out = out / denom

        out = out.transpose(1, 2).reshape(B, S, -1)

        # Attention output gate
        if self.use_attn_gate:
            if (
                self.use_resonance
                and self.config.linear_resonance_mode == "gate"
                and resonance is not None
            ):
                # gate_i = sigmoid(mean_j R[i,j] * w_g)
                gate_val = (
                    resonance.mean(dim=-1, keepdim=True) * self.resonance_gate_weight
                )  # [B, S, 1]
                gate = torch.sigmoid(gate_val)
                gate = gate.expand(-1, -1, out.size(-1))
            else:
                gate = torch.sigmoid(self.gate_proj(x))
            out = out * gate

        return self.o_proj(out)


class ResonantQwen3_5FullAttention(nn.Module):
    """Full GQA attention with QK norm, RoPE, gate, and resonance bias.

    The resonance bias is added to attention logits **before** softmax,
    following the same pattern as ``ResonanceBiasAttention`` in
    ``common.py``.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.embed_dim, config.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.embed_dim, config.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.embed_dim, config.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.n_heads * self.head_dim, config.embed_dim, bias=False
        )

        self.use_attn_gate = config.use_attn_gate
        if config.use_attn_gate:
            self.gate_proj = nn.Linear(
                config.embed_dim, config.n_heads * self.head_dim, bias=False
            )

        self.use_qk_norm = config.use_qk_norm
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rmsnorm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rmsnorm_eps)

        self.rope = RoPE(
            self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_theta,
        )
        self.dropout = nn.Dropout(config.dropout)

        # Resonance bias weight (one scalar per head)
        self.resonance_weight = nn.Parameter(torch.full((config.n_heads,), 0.1))

    def forward(
        self,
        x: torch.Tensor,
        resonance: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``[batch, seq_len, embed_dim]``
            resonance: ``[batch, seq_len, seq_len]`` pairwise resonance.
            mask: Optional custom mask broadcastable to
                ``[batch, n_heads, seq_len, seq_len]``.

        Returns:
            ``[batch, seq_len, embed_dim]``
        """
        B, S, _ = x.shape

        q = (
            self.q_proj(x)
            .view(B, S, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(B, S, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(B, S, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.use_qk_norm:
            q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)

        q = self.rope(q, S)
        k = self.rope(k, S)

        if self.n_kv_heads < self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # Manual attention so resonance bias can be injected pre-softmax
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]

        if resonance is not None:
            attn = attn + resonance.unsqueeze(1) * self.resonance_weight.view(
                1, self.n_heads, 1, 1
            )

        causal = torch.tril(torch.ones(S, S, device=x.device)).view(1, 1, S, S)
        attn = attn.masked_fill(causal == 0, float("-inf"))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, S, -1)

        if self.use_attn_gate:
            gate = torch.sigmoid(self.gate_proj(x))
            out = out * gate

        return self.o_proj(out)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

class Qwen3_5MoELayer(nn.Module):
    """Qwen3.5-style MoE FFN.

    Combines a *shared* expert (always active) with a set of *sparse*
    experts (top-k routed).  A learned scalar gate blends the two
    branches per token:

    .. math::
        \\text{out} = g \\cdot \\text{shared}(x)
                   + (1 - g) \\cdot \\text{sparse}(x)
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.shared_expert = SwiGLU(config.embed_dim, config.ff_dim)
        self.sparse_experts = MoELayer(
            dim=config.embed_dim,
            hidden_dim=config.ff_dim,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            expert_type="swiglu",
        )
        self.shared_expert_gate = nn.Linear(config.embed_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(output, aux_loss)``."""
        shared_out = self.shared_expert(x)
        sparse_out, aux_loss = self.sparse_experts(x)
        gate = torch.sigmoid(self.shared_expert_gate(x))  # [B, S, 1]
        out = gate * shared_out + (1.0 - gate) * sparse_out
        return out, aux_loss


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class Qwen3_5Block(nn.Module):
    """Single Qwen3.5 decoder layer (pre-norm)."""

    def __init__(self, config: Qwen3_5Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        is_linear = _is_linear_layer(config, layer_idx)
        if is_linear:
            self.attn = Qwen3_5LinearAttention(config)
        else:
            self.attn = Qwen3_5FullAttention(config)

        self.input_layernorm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.embed_dim, eps=config.rmsnorm_eps
        )

        if _is_moe_layer(config, layer_idx):
            self.mlp = Qwen3_5MoELayer(config)
        else:
            self.mlp = SwiGLU(config.embed_dim, config.ff_dim)

    def forward(
        self,
        x: torch.Tensor,
        resonance: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            ``(hidden_states, aux_loss)`` where ``aux_loss`` is zero for
            dense FFN layers.
        """
        # Pre-norm attention with residual
        h = x + self.attn(self.input_layernorm(x), resonance, mask)

        # Pre-norm FFN with residual
        if isinstance(self.mlp, Qwen3_5MoELayer):
            mlp_out, aux_loss = self.mlp(self.post_attention_layernorm(h))
            out = h + mlp_out
        else:
            out = h + self.mlp(self.post_attention_layernorm(h))
            aux_loss = torch.tensor(0.0, device=x.device)

        return out, aux_loss


class ResonantQwen3_5Block(nn.Module):
    """Single Qwen3.5 decoder layer with resonance support (pre-norm)."""

    def __init__(self, config: Qwen3_5Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        is_linear = _is_linear_layer(config, layer_idx)
        if is_linear:
            # Linear attention handles its own resonance modes internally
            self.attn = Qwen3_5LinearAttention(config)
        else:
            self.attn = ResonantQwen3_5FullAttention(config)

        self.input_layernorm = RMSNorm(config.embed_dim, eps=config.rmsnorm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.embed_dim, eps=config.rmsnorm_eps
        )

        if _is_moe_layer(config, layer_idx):
            self.mlp = Qwen3_5MoELayer(config)
        else:
            self.mlp = SwiGLU(config.embed_dim, config.ff_dim)

    def forward(
        self,
        x: torch.Tensor,
        resonance: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            ``(hidden_states, aux_loss)`` where ``aux_loss`` is zero for
            dense FFN layers.
        """
        h = x + self.attn(self.input_layernorm(x), resonance, mask)

        if isinstance(self.mlp, Qwen3_5MoELayer):
            mlp_out, aux_loss = self.mlp(self.post_attention_layernorm(h))
            out = h + mlp_out
        else:
            out = h + self.mlp(self.post_attention_layernorm(h))
            aux_loss = torch.tensor(0.0, device=x.device)

        return out, aux_loss


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class Qwen3_5Transformer(nn.Module):
    """Dense standard Qwen3.5 transformer.

    Pre-norm RMSNorm, hybrid attention, QK norm, attention output gating,
    SwiGLU FFN, causal masking, and tied input/output embeddings.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = replace(config, use_moe=False, use_resonance=False)

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.embed_dim)
        self.dropout = nn.Dropout(self.config.dropout)

        self.blocks = nn.ModuleList(
            [Qwen3_5Block(self.config, i) for i in range(self.config.n_layers)]
        )

        self.norm = RMSNorm(self.config.embed_dim, eps=self.config.rmsnorm_eps)
        self.lm_head = nn.Linear(
            self.config.embed_dim, self.config.vocab_size, bias=False
        )
        self.lm_head.weight = self.embed_tokens.weight

        self.register_buffer(
            "causal_mask",
            make_causal_mask(self.config.max_seq_len, torch.device("cpu")),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass and optional next-token loss.

        Args:
            input_ids: Token indices ``[batch, seq_len]``.
            labels: Optional target indices of the same shape.

        Returns:
            Dict with ``"logits"``, ``"aux_loss"`` (scalar, always ``0.0``
            for the dense model), and optionally ``"loss"``.
        """
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids)
        x = self.dropout(x)

        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        mask = self.causal_mask[:, :, :S, :S]

        for block in self.blocks:
            x, aux_loss = block(x, resonance=None, mask=mask)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.norm(x)
        logits = self.lm_head(x)

        result: dict[str, Any] = {"logits": logits, "aux_loss": total_aux_loss}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return result


class Qwen3_5MoE(nn.Module):
    """Sparse standard Qwen3.5 transformer with MoE layers.

    Same architecture as :class:`Qwen3_5Transformer` but replaces select
    FFN layers with a shared-expert + sparse-expert MoE block.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = replace(config, use_moe=True, use_resonance=False)

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.embed_dim)
        self.dropout = nn.Dropout(self.config.dropout)

        self.blocks = nn.ModuleList(
            [Qwen3_5Block(self.config, i) for i in range(self.config.n_layers)]
        )

        self.norm = RMSNorm(self.config.embed_dim, eps=self.config.rmsnorm_eps)
        self.lm_head = nn.Linear(
            self.config.embed_dim, self.config.vocab_size, bias=False
        )
        self.lm_head.weight = self.embed_tokens.weight

        self.register_buffer(
            "causal_mask",
            make_causal_mask(self.config.max_seq_len, torch.device("cpu")),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids)
        x = self.dropout(x)

        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        mask = self.causal_mask[:, :, :S, :S]

        for block in self.blocks:
            x, aux_loss = block(x, resonance=None, mask=mask)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.norm(x)
        logits = self.lm_head(x)

        result: dict[str, Any] = {"logits": logits, "aux_loss": total_aux_loss}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return result


class ResonantQwen3_5(nn.Module):
    """Dense Qwen3.5 transformer with resonance embeddings.

    Uses :class:`ModernResonanceEmbedding` for dual semantic/phase
    representations.  Full GQA layers add the resonance matrix to
    attention logits (:class:`ResonantQwen3_5FullAttention`).  Linear
    attention layers apply one of the three resonance adaptations
    (skip / gate / state) controlled by ``config.linear_resonance_mode``.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = replace(config, use_moe=False, use_resonance=True)

        self.embedding = ModernResonanceEmbedding(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            n_frequencies=self.config.n_frequencies,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout,
        )

        self.blocks = nn.ModuleList(
            [
                ResonantQwen3_5Block(self.config, i)
                for i in range(self.config.n_layers)
            ]
        )

        self.norm = RMSNorm(self.config.embed_dim, eps=self.config.rmsnorm_eps)
        self.lm_head = nn.Linear(
            self.config.embed_dim, self.config.vocab_size, bias=False
        )
        self.lm_head.weight = self.embedding.semantic.weight

        self.register_buffer(
            "causal_mask",
            make_causal_mask(self.config.max_seq_len, torch.device("cpu")),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        B, S = input_ids.shape
        x, resonance = self.embedding(input_ids)

        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        mask = self.causal_mask[:, :, :S, :S]

        for block in self.blocks:
            x, aux_loss = block(x, resonance=resonance, mask=mask)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.norm(x)
        logits = self.lm_head(x)

        result: dict[str, Any] = {"logits": logits, "aux_loss": total_aux_loss}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return result


class ResonantQwen3_5MoE(nn.Module):
    """Sparse Qwen3.5 transformer with resonance embeddings and MoE FFN.

    Combines :class:`ResonantQwen3_5` attention with MoE layers.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = replace(config, use_moe=True, use_resonance=True)

        self.embedding = ModernResonanceEmbedding(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            n_frequencies=self.config.n_frequencies,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout,
        )

        self.blocks = nn.ModuleList(
            [
                ResonantQwen3_5Block(self.config, i)
                for i in range(self.config.n_layers)
            ]
        )

        self.norm = RMSNorm(self.config.embed_dim, eps=self.config.rmsnorm_eps)
        self.lm_head = nn.Linear(
            self.config.embed_dim, self.config.vocab_size, bias=False
        )
        self.lm_head.weight = self.embedding.semantic.weight

        self.register_buffer(
            "causal_mask",
            make_causal_mask(self.config.max_seq_len, torch.device("cpu")),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        B, S = input_ids.shape
        x, resonance = self.embedding(input_ids)

        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        mask = self.causal_mask[:, :, :S, :S]

        for block in self.blocks:
            x, aux_loss = block(x, resonance=resonance, mask=mask)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.norm(x)
        logits = self.lm_head(x)

        result: dict[str, Any] = {"logits": logits, "aux_loss": total_aux_loss}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return result
