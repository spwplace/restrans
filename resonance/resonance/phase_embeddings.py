"""Phase embedding variants for the Resonance Transformer.

The default is a learned real-valued embedding table.
This module provides alternatives that may capture different
inductive biases about how tokens relate in phase space.

All variants implement:
    forward(token_ids: [B, S]) -> phases: [B, S, n_frequencies]
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseEmbedding(nn.Module):
    """Base class for phase embedding variants."""

    def __init__(self, vocab_size: int, n_frequencies: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_frequencies = n_frequencies

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RealPhaseEmbedding(PhaseEmbedding):
    """Standard learned real-valued phase embedding.

    This is the original implementation: each token learns an
    n_frequencies-dimensional real vector, and kernels compute
    relationships from these values.
    """

    def __init__(self, vocab_size: int, n_frequencies: int, init_std: float = 0.3) -> None:
        super().__init__(vocab_size, n_frequencies)
        self.weight = nn.Parameter(torch.randn(vocab_size, n_frequencies) * init_std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids, self.weight)


class FourierFixedPhaseEmbedding(PhaseEmbedding):
    """Fixed sinusoidal phase features (not learned).

    Inspired by positional encodings but applied to token identities.
    Each token gets a deterministic phase signature based on its ID:

        φ_t^f = sin(ω_f · t)   or   cos(ω_f · t)

    where ω_f = 1 / 10000^(2f/n_frequencies).

    This embeds tokens into a fixed Fourier basis.  The resonance kernel
    then measures how similar two tokens' spectral signatures are.
    """

    def __init__(self, vocab_size: int, n_frequencies: int) -> None:
        super().__init__(vocab_size, n_frequencies)
        # Precompute sinusoidal basis: [vocab_size, n_frequencies]
        positions = torch.arange(vocab_size).float().unsqueeze(1)  # [V, 1]
        div_term = torch.exp(
            torch.arange(0, n_frequencies, 2).float()
            * (-math.log(10000.0) / n_frequencies)
        )  # [F/2]
        pe = torch.zeros(vocab_size, n_frequencies)
        pe[:, 0::2] = torch.sin(positions * div_term)
        if n_frequencies % 2 == 1:
            pe[:, 1::2] = torch.cos(positions * div_term[: n_frequencies // 2])
        else:
            pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids, self.pe)


class ComplexAnglePhaseEmbedding(PhaseEmbedding):
    """Complex-valued phase embedding: learned angle, fixed amplitude.

    Each token learns an angle θ_t^f.  The phase vector is:
        z_t^f = exp(i · θ_t^f)

    For kernels that expect real inputs, we output the angle θ directly.
    For complex-aware kernels, the angle is the natural input.

    This is more principled than RealPhaseEmbedding because phases are
    inherently circular: θ and θ+2π should be identical.  The cosine
    kernel naturally respects this, but initialization and optimization
    on a real line can drift.  ComplexAngle keeps values in [-π, π].
    """

    def __init__(self, vocab_size: int, n_frequencies: int, init_std: float = 0.3) -> None:
        super().__init__(vocab_size, n_frequencies)
        # Initialize angles from a normal distribution
        self.angle = nn.Parameter(torch.randn(vocab_size, n_frequencies) * init_std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Return the angle (modulo 2π is implicit in how kernels use it)
        return self.angle[token_ids]

    def to_complex(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return complex unit vectors exp(i·θ)."""
        return torch.exp(1j * self.angle[token_ids])


class HierarchicalPhaseEmbedding(PhaseEmbedding):
    """Multi-scale phase embedding.

    Uses multiple smaller embedding tables at different scales,
    concatenated to form the full phase vector.  This lets the model
    learn coarse token relationships from low-frequency scales and
    fine-grained relationships from high-frequency scales.

    Example: n_frequencies=32, n_scales=4 → 4 tables of 8 dims each.
    """

    def __init__(
        self,
        vocab_size: int,
        n_frequencies: int,
        n_scales: int = 4,
        init_std: float = 0.3,
    ) -> None:
        super().__init__(vocab_size, n_frequencies)
        assert n_frequencies % n_scales == 0, "n_frequencies must divide n_scales"
        self.n_scales = n_scales
        self.dim_per_scale = n_frequencies // n_scales
        self.scales = nn.ModuleList([
            nn.Embedding(vocab_size, self.dim_per_scale)
            for _ in range(n_scales)
        ])
        self._init_weights(init_std)

    def _init_weights(self, init_std: float) -> None:
        # Initialize each scale with different standard deviations
        # so coarse scales span more of the manifold
        for i, emb in enumerate(self.scales):
            std = init_std * (1.0 + i * 0.5)  # larger std for coarser scales
            nn.init.normal_(emb.weight, std=std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        parts = [emb(token_ids) for emb in self.scales]
        return torch.cat(parts, dim=-1)


class FactorizedPhaseEmbedding(PhaseEmbedding):
    """Factorized phase embedding: amplitude × frequency basis.

    Instead of learning V×F parameters, learn:
        - amplitudes: [vocab_size, rank]
        - basis:      [rank, n_frequencies]

    Then φ_t = amplitudes[t] @ basis.

    This is lower-parameter and forces the model to represent phases
    in a shared frequency basis.
    """

    def __init__(
        self,
        vocab_size: int,
        n_frequencies: int,
        rank: int = 8,
        init_std: float = 0.3,
    ) -> None:
        super().__init__(vocab_size, n_frequencies)
        self.rank = rank
        self.amplitudes = nn.Parameter(torch.randn(vocab_size, rank) * init_std)
        self.basis = nn.Parameter(torch.randn(rank, n_frequencies) * init_std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        amp = self.amplitudes[token_ids]  # [B, S, rank]
        return torch.matmul(amp, self.basis)  # [B, S, F]


# =============================================================================
# Registry
# =============================================================================

_PHASE_REGISTRY: dict[str, Callable[..., PhaseEmbedding]] = {
    "real": RealPhaseEmbedding,
    "fourier_fixed": FourierFixedPhaseEmbedding,
    "complex_angle": ComplexAnglePhaseEmbedding,
    "hierarchical": HierarchicalPhaseEmbedding,
    "factorized": FactorizedPhaseEmbedding,
}


def build_phase_embedding(
    name: str,
    vocab_size: int,
    n_frequencies: int,
    **kwargs,
) -> PhaseEmbedding:
    """Build a phase embedding variant by name.

    Args:
        name: One of ``real``, ``fourier_fixed``, ``complex_angle``,
            ``hierarchical``, ``factorized``.
        vocab_size: Token vocabulary size.
        n_frequencies: Phase dimension.
        **kwargs: Extra constructor args (``init_std``, ``rank``, etc.).

    Returns:
        Instantiated ``PhaseEmbedding`` subclass.
    """
    if name not in _PHASE_REGISTRY:
        raise ValueError(
            f"Unknown phase embedding: {name!r}. "
            f"Available: {sorted(_PHASE_REGISTRY.keys())}"
        )
    cls = _PHASE_REGISTRY[name]
    sig = cls.__init__.__code__.co_varnames
    filtered = {k: v for k, v in kwargs.items() if k in sig}
    return cls(vocab_size, n_frequencies, **filtered)


def list_phase_embeddings() -> list[str]:
    """Return all registered phase embedding names."""
    return sorted(_PHASE_REGISTRY.keys())
