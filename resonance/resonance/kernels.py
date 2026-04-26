"""Modular resonance kernels for the Resonance Transformer.

The default kernel is a cosine-difference kernel (Fourier-like):
    R[i,j] = mean_f cos(φ_i^f - φ_j^f)

This module provides alternative kernels that can be swapped via config:
    cosine, cosine_weighted, dot, rbf, laplace, bilinear,
    complex_magnitude, complex_real.

All kernels are nn.Module subclasses so learned parameters (weights,
bilinear forms, etc.) participate in gradient descent automatically.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResonanceKernel(nn.Module):
    """Base class for resonance kernels.

    Each kernel takes phase embeddings ``[batch, seq_len, n_frequencies]``
    and returns a pairwise resonance matrix ``[batch, seq_len, seq_len]``.
    """

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CosineKernel(ResonanceKernel):
    """Default kernel: mean cosine of phase differences.

    R[i,j] = (1/F) Σ_f cos(φ_i^f - φ_j^f)

    This is bounded in [-1, 1] and invariant to global phase shifts.
    """

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)
        return torch.cos(phase_diff).mean(dim=-1)


class WeightedCosineKernel(ResonanceKernel):
    """Learned per-frequency weighted cosine kernel.

    R[i,j] = Σ_f w_f · cos(φ_i^f - φ_j^f)  where Σ_f w_f = 1

    Allows the model to learn which frequency bands matter.
    """

    def __init__(self, n_frequencies: int) -> None:
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(n_frequencies))

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.raw_weights, dim=0)
        phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)
        cos_terms = torch.cos(phase_diff)  # [B, S, S, F]
        return (cos_terms * weights.view(1, 1, 1, -1)).sum(dim=-1)


class DotKernel(ResonanceKernel):
    """Simple dot-product kernel over phase frequencies.

    R[i,j] = (1/F) Σ_f φ_i^f · φ_j^f

    Unbounded; behaves like a linear similarity.
    """

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        return torch.matmul(phases, phases.transpose(-1, -2)) / phases.size(-1)


class RBFKernel(ResonanceKernel):
    """Radial basis function (Gaussian) kernel over phase differences.

    R[i,j] = exp(-γ · (1/F) Σ_f (φ_i^f - φ_j^f)²)

    Produces a soft, localized similarity.
    """

    def __init__(self, gamma: float = 1.0, learnable_gamma: bool = False) -> None:
        super().__init__()
        if learnable_gamma:
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)
        sq_dist = (phase_diff ** 2).mean(dim=-1)
        return torch.exp(-self.gamma * sq_dist)


class LaplaceKernel(ResonanceKernel):
    """Laplacian (exponential) kernel over phase differences.

    R[i,j] = exp(-γ · (1/F) Σ_f |φ_i^f - φ_j^f|)
    """

    def __init__(self, gamma: float = 1.0, learnable_gamma: bool = False) -> None:
        super().__init__()
        if learnable_gamma:
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)
        abs_dist = phase_diff.abs().mean(dim=-1)
        return torch.exp(-self.gamma * abs_dist)


class BilinearKernel(ResonanceKernel):
    """Learned low-rank bilinear kernel.

    R[i,j] = (1/F) · φ_i^T W φ_j

    W is factorized as W = U^T V for efficiency, where U and V are
    [rank, n_frequencies] matrices.  Rank defaults to n_frequencies//2.
    """

    def __init__(self, n_frequencies: int, rank: int | None = None) -> None:
        super().__init__()
        rank = rank or max(1, n_frequencies // 2)
        self.u = nn.Parameter(torch.randn(rank, n_frequencies) * 0.02)
        self.v = nn.Parameter(torch.randn(rank, n_frequencies) * 0.02)
        self.scale = 1.0 / n_frequencies

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        # phases: [B, S, F]
        # Project: [B, S, rank]
        proj_u = torch.matmul(phases, self.u.T)
        proj_v = torch.matmul(phases, self.v.T)
        return torch.matmul(proj_u, proj_v.transpose(-1, -2)) * self.scale


class ComplexMagnitudeKernel(ResonanceKernel):
    """Complex exponential kernel (magnitude of coherence).

    Treats each phase frequency as a complex unit vector:
        z_i^f = exp(i · φ_i^f)

    Then computes the magnitude of mean complex coherence:
        R[i,j] = | (1/F) Σ_f exp(i(φ_i^f - φ_j^f)) |
               = | (1/F) Σ_f z_i^f · conj(z_j^f) |

    This is the *coherence* between two phase spectra — the FFT-like
    quantity the user asked about.  Bounded in [0, 1].

    Accepts either real angles or pre-computed complex unit vectors.
    """

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        if phases.is_complex():
            z = phases  # already unit vectors
        else:
            z = torch.exp(1j * phases)
        coherence = torch.matmul(z, z.conj().transpose(-1, -2)) / z.size(-1)
        return coherence.abs().real


class ComplexRealKernel(ResonanceKernel):
    """Complex exponential kernel (real part only).

    R[i,j] = Re( (1/F) Σ_f exp(i(φ_i^f - φ_j^f)) )
           = (1/F) Σ_f cos(φ_i^f - φ_j^f)

    This is identical to CosineKernel!  Provided for explicit naming
    and as a bridge to complex-valued generalizations.

    Accepts either real angles or pre-computed complex unit vectors.
    """

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        if phases.is_complex():
            z = phases  # already unit vectors
        else:
            z = torch.exp(1j * phases)
        coherence = torch.matmul(z, z.conj().transpose(-1, -2)) / z.size(-1)
        return coherence.real


class AttentionKernel(ResonanceKernel):
    """Softmax-normalized dot-product kernel.

    R[i,j] = softmax_j( (1/√F) · φ_i · φ_j )

    Produces a proper attention distribution over the phase space.
    This makes the resonance matrix act more like a soft nearest-neighbor
    selector in phase space.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(phases.size(-1))
        logits = torch.matmul(phases, phases.transpose(-1, -2)) * scale
        return F.softmax(logits / self.temperature, dim=-1)


# =============================================================================
# Registry
# =============================================================================

_KERNEL_REGISTRY: dict[str, Callable[..., ResonanceKernel]] = {
    "cosine": CosineKernel,
    "cosine_weighted": WeightedCosineKernel,
    "dot": DotKernel,
    "rbf": RBFKernel,
    "laplace": LaplaceKernel,
    "bilinear": BilinearKernel,
    "complex_magnitude": ComplexMagnitudeKernel,
    "complex_real": ComplexRealKernel,
    "attention": AttentionKernel,
}


def build_kernel(name: str, n_frequencies: int, **kwargs) -> ResonanceKernel:
    """Build a resonance kernel by name.

    Args:
        name: Kernel name.  One of the keys in ``_KERNEL_REGISTRY``.
        n_frequencies: Number of phase frequency dimensions.
        **kwargs: Extra arguments passed to the kernel constructor
            (e.g. ``gamma`` for RBF, ``rank`` for bilinear).

    Returns:
        An instantiated ``ResonanceKernel`` subclass.
    """
    if name not in _KERNEL_REGISTRY:
        raise ValueError(
            f"Unknown resonance kernel: {name!r}. "
            f"Available: {sorted(_KERNEL_REGISTRY.keys())}"
        )
    cls = _KERNEL_REGISTRY[name]
    sig = cls.__init__.__code__.co_varnames
    # Only pass kwargs that the constructor accepts
    filtered = {k: v for k, v in kwargs.items() if k in sig}
    if "n_frequencies" in sig:
        filtered["n_frequencies"] = n_frequencies
    return cls(**filtered)


def list_kernels() -> list[str]:
    """Return all registered kernel names."""
    return sorted(_KERNEL_REGISTRY.keys())
