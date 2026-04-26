"""Resonance bias application modes for the Resonance Transformer.

The default is additive: attn = QK + R·w
This module provides alternatives that may better integrate the
resonance signal into the attention computation.

All modes implement:
    forward(qk_logits, resonance_bias, mask) -> modified_logits
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiasMode(nn.Module):
    """Base class for resonance bias application modes."""

    def forward(
        self,
        qk_logits: torch.Tensor,
        resonance_bias: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class AdditiveBias(BiasMode):
    """Default additive bias: attn = QK + R·w

    The resonance bias is added directly to the QK logits before
    masking and softmax.
    """

    def forward(
        self,
        qk_logits: torch.Tensor,
        resonance_bias: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return qk_logits + resonance_bias


class MultiplicativeGateBias(BiasMode):
    """Multiplicative gating: attn = QK · (1 + σ(R·w))

    The resonance bias passes through a sigmoid and multiplicatively
    gates the QK logits.  This preserves the sign structure of QK
    while modulating magnitudes.
    """

    def __init__(self, gate_init: float = 0.0) -> None:
        super().__init__()
        # Learnable gate bias: initialized near zero so 1+sigmoid(0) ≈ 1.5
        self.gate_bias = nn.Parameter(torch.tensor(gate_init))

    def forward(
        self,
        qk_logits: torch.Tensor,
        resonance_bias: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate = 1.0 + torch.sigmoid(resonance_bias + self.gate_bias)
        return qk_logits * gate


class ResidualGateBias(BiasMode):
    """Residual gating with learned scale: attn = QK + gate·R·w

    A learnable per-head (or global) gate controls how much of the
    resonance bias is added.  When gate → 0, the model reverts to
    standard attention.
    """

    def __init__(self, n_heads: int = 1) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.full((n_heads,), 0.1))

    def forward(
        self,
        qk_logits: torch.Tensor,
        resonance_bias: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # resonance_bias: [B, H, S, S]
        # gate: [H] → broadcast to [1, H, 1, 1]
        gate = torch.sigmoid(self.gate).view(1, -1, 1, 1)
        return qk_logits + gate * resonance_bias


class TemperatureScaledBias(BiasMode):
    """Temperature-scaled combined logits: attn = (QK + R·w) / τ

    A learned per-head temperature rescales the combined logits.
    This is useful when resonance and QK contributions have different
    natural scales.
    """

    def __init__(self, n_heads: int = 1, init_temp: float = 1.0) -> None:
        super().__init__()
        self.log_temp = nn.Parameter(
            torch.full((n_heads,), math.log(init_temp))
        )

    def forward(
        self,
        qk_logits: torch.Tensor,
        resonance_bias: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        temp = torch.exp(self.log_temp).view(1, -1, 1, 1)
        return (qk_logits + resonance_bias) / temp


class SoftmaxReWeightedBias(BiasMode):
    """Softmax-reweighted attention: attn = softmax(QK) · R·w + QK

    The resonance bias is used to reweight the QK softmax distribution.
    This makes the bias act like a learned attention prior that
    multiplicatively interacts with the query-key dynamics.
    """

    def forward(
        self,
        qk_logits: torch.Tensor,
        resonance_bias: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Compute standard attention probs
        qk = qk_logits.clone()
        if mask is not None:
            qk = qk.masked_fill(mask == 0, float("-inf"))
        attn_probs = F.softmax(qk, dim=-1)
        # Reweight by resonance and add back
        return qk_logits + attn_probs * resonance_bias


class OnlyResonanceBias(BiasMode):
    """Ablation: attention uses ONLY resonance bias, no QK term.

    attn = R·w

    This is a diagnostic mode to measure how much predictive signal
    the resonance matrix alone carries.
    """

    def forward(
        self,
        qk_logits: torch.Tensor,
        resonance_bias: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return resonance_bias


# =============================================================================
# Registry
# =============================================================================

_BIAS_REGISTRY: dict[str, Callable[..., BiasMode]] = {
    "additive": AdditiveBias,
    "multiplicative_gate": MultiplicativeGateBias,
    "residual_gate": ResidualGateBias,
    "temperature_scaled": TemperatureScaledBias,
    "softmax_reweighted": SoftmaxReWeightedBias,
    "only_resonance": OnlyResonanceBias,
}


def build_bias_mode(name: str, n_heads: int = 1, **kwargs) -> BiasMode:
    """Build a bias application mode by name.

    Args:
        name: One of the keys in ``_BIAS_REGISTRY``.
        n_heads: Number of attention heads (for modes that need it).
        **kwargs: Extra constructor args.

    Returns:
        Instantiated ``BiasMode`` subclass.
    """
    if name not in _BIAS_REGISTRY:
        raise ValueError(
            f"Unknown bias mode: {name!r}. "
            f"Available: {sorted(_BIAS_REGISTRY.keys())}"
        )
    cls = _BIAS_REGISTRY[name]
    sig = cls.__init__.__code__.co_varnames
    filtered = {k: v for k, v in kwargs.items() if k in sig}
    if "n_heads" in sig:
        filtered["n_heads"] = n_heads
    return cls(**filtered)


def list_bias_modes() -> list[str]:
    """Return all registered bias mode names."""
    return sorted(_BIAS_REGISTRY.keys())
