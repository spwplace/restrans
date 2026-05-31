"""Modular resonance kernels for the Resonance Transformer.

The default kernel is a cosine-difference kernel (Fourier-like):
    R[i,j] = mean_f cos(φ_i^f - φ_j^f)

This module provides alternative kernels that can be swapped via config:
    cosine, harmonic_cosine, cosine_weighted, dot, rbf, laplace,
    bilinear, complex_magnitude, complex_real.

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


class HarmonicCosineKernel(ResonanceKernel):
    """Multi-harmonic cosine-difference kernel.

    The default cosine kernel uses only the fundamental phase difference.
    This variant computes a compact Fourier-style relation basis:

        R[i,j] = Σ_h w_h · mean_f cos(h · (φ_i^f - φ_j^f))

    where ``h`` ranges from 1 to ``n_harmonics``.  It keeps the same scalar
    relation-matrix interface while testing whether higher harmonics carry
    useful structural aliasing / periodicity information.
    """

    def __init__(self, n_frequencies: int, rank: int | None = None) -> None:
        super().__init__()
        del n_frequencies
        n_harmonics = rank or 4
        self.register_buffer(
            "harmonics",
            torch.arange(1, n_harmonics + 1, dtype=torch.float32),
        )
        self.raw_weights = nn.Parameter(torch.zeros(n_harmonics))

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)
        harmonic_diff = phase_diff.unsqueeze(-1) * self.harmonics.view(1, 1, 1, 1, -1)
        harmonic_terms = torch.cos(harmonic_diff).mean(dim=-2)  # [B, S, S, H]
        weights = F.softmax(self.raw_weights, dim=0)
        return (harmonic_terms * weights.view(1, 1, 1, -1)).sum(dim=-1)


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


class PairMLPKernel(ResonanceKernel):
    """Learned pair-specific relation kernel.

    This is intentionally small but more expressive than fixed distance or
    cosine kernels. It constructs each relation from both endpoint phase
    vectors, their difference, and their elementwise interaction.
    """

    def __init__(
        self,
        n_frequencies: int,
        rank: int = 64,
        temperature: float = 1.0,
        **_: object,
    ) -> None:
        super().__init__()
        hidden = max(4, int(rank))
        self.temperature = max(float(temperature), 1e-6)
        self.net = nn.Sequential(
            nn.Linear(n_frequencies * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        batch, seq_len, freq = phases.shape
        left = phases.unsqueeze(2).expand(batch, seq_len, seq_len, freq)
        right = phases.unsqueeze(1).expand(batch, seq_len, seq_len, freq)
        features = torch.cat([left, right, left - right, left * right], dim=-1)
        return self.net(features).squeeze(-1) / self.temperature


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


class DirectionalComplexKernel(ResonanceKernel):
    """Learned directional phase-difference kernel.

    The default cosine-difference kernel is symmetric: ``R[i,j] == R[j,i]``.
    Many structural relations are not symmetric (binder -> use, cause -> effect,
    opener -> closer), so this kernel keeps both cosine and sine terms:

        R[i,j] = Σ_f wc_f cos(φ_i^f - φ_j^f) + ws_f sin(φ_i^f - φ_j^f)

    The sine branch is antisymmetric and can encode order/direction while still
    remaining a compact scalar relation matrix for the existing attention path.
    """

    def __init__(self, n_frequencies: int) -> None:
        super().__init__()
        self.cos_weights = nn.Parameter(torch.zeros(n_frequencies))
        self.sin_weights = nn.Parameter(torch.zeros(n_frequencies))
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)
        cos_weights = F.softmax(self.cos_weights, dim=0)
        # Sine weights are signed but normalized to keep the scale predictable.
        sin_weights = torch.tanh(self.sin_weights)
        sin_weights = sin_weights / sin_weights.abs().sum().clamp(min=1.0)
        cos_part = (torch.cos(phase_diff) * cos_weights.view(1, 1, 1, -1)).sum(dim=-1)
        sin_part = (torch.sin(phase_diff) * sin_weights.view(1, 1, 1, -1)).sum(dim=-1)
        return self.output_scale * (cos_part + sin_part)


class WalkKernel(ResonanceKernel):
    """Walkformer kernel: a learnable simplex mixture of analytic walk operators.

    The sequence positions ARE the vertices of a latent 1-D line/cycle of length
    ``S``.  Each *atom* is an analytic continuous-time-quantum-walk (CTQW) /
    classical-walk operator on that graph, expressed purely as a function of the
    relative offset ``d = i - j``.  Because every atom is Toeplitz (a function of
    ``i - j`` only), the resulting bias matrix is translation-equivariant and
    length-extrapolating: the same per-offset profile is reused at every length.

    The output bias is the simplex mixture::

        R[i, j] = Σ_a softmax(π)_a · W_a[i - j]

    Atoms (each maps directly onto an operator formalized in graphplay):

      1. ``path_heat``   — CTQW / diffusion heat kernel on the path graph,
         ``exp(-τ L_path)``.  v1 uses the analytic Gaussian decay
         ``exp(-τ d²)`` in the relative offset (the continuum heat kernel).
      2. ``shift_k``     — soft shift-by-k Toeplitz (causal previous-token
         addressing), a Gaussian bump centred at offset ``-k`` with learnable
         soft ``k``.
      3. ``circulant``   — learnable circular convolution over a small band of
         relative offsets (diagonalizable by FFT; here a learnable per-offset
         band is used directly).
      4. ``chiral``      — **the key novel atom**: a U(1)-phased quantum walk
         ``e^{i φ d}`` with learnable phase-rate ``φ``, optionally driven by the
         ``complex_angle`` phase stream.  We take ``Re(·)`` to get a real,
         direction-aware band.  This is what real SSMs / Hyena / FNet cannot
         express (their kernels are real / phase-locked).  Ablatable via
         ``use_chiral=False``.
      5. ``cheb``        — a learnable low-order Chebyshev polynomial in the
         (normalized) relative offset — a learnable band-pass positional filter
         (a polynomial in the path Laplacian).

    All atoms are O(S) banded / dense-Toeplitz constructions and share only a
    handful of scalar parameters, so the kernel is extremely param-light — the
    selling point versus a learned-QK relative-position table.

    The causal upper-triangle is *not* masked here: the attention module masks
    it (``masked_fill(mask == 0, -inf)``) downstream, exactly as for the other
    kernels.  We do gate the chiral/shift atoms so their natural support is the
    causal (``d >= 0``) half.
    """

    def __init__(
        self,
        n_frequencies: int,
        walk_atoms: int = 5,
        use_chiral: bool = True,
        walk_band: int = 8,
        walk_use_phase_drive: bool = True,
        walk_atom_set: str = "v1",
        **_: object,
    ) -> None:
        super().__init__()
        self.n_frequencies = n_frequencies
        self.use_chiral = bool(use_chiral)
        self.band = int(walk_band)
        self.use_phase_drive = bool(walk_use_phase_drive)

        # ``walk_atom_set`` selects which atoms carry mixture mass.  This is the
        # byte-stable v1/v2 switch.  ``"v1"`` forces the five v2 atoms (indices
        # 5..9) to zero contribution and renormalizes, so the kernel is exactly
        # the original 5-atom walkformer (the running v1 gauntlet stays
        # comparable).  ``"v2"`` enables all ten atoms.  A set of explicit atom
        # names (e.g. ``"bessel"``) enables the v1 five plus just those v2 atoms
        # — used by the per-atom ablation probes.
        self.walk_atom_set = str(walk_atom_set)
        # Per-v2-atom enable mask (indices 5..9), resolved from walk_atom_set.
        self._v2_names = ("bessel", "coined", "twohorn", "powerlaw", "learnedH")
        self.v2_enabled = self._resolve_v2_enabled(self.walk_atom_set)

        # Mixture logits over the (now up to 10) atoms.  Atoms are always built;
        # when ``use_chiral`` is False the chiral atom (index 3) is forced to
        # zero contribution so the ablation is exact (no leakage through the
        # simplex).  The five v2 atoms (indices 5..9) start at a small negative
        # logit so they begin with low mixture mass — the kernel's initial
        # behaviour stays close to v1 (graceful).
        self.n_atoms = 10
        init_logits = torch.zeros(self.n_atoms)
        init_logits[5:] = -2.0
        self.mix_logits = nn.Parameter(init_logits)

        # Atom 1: path-heat / CTQW diffusion.  log τ keeps τ > 0.
        self.log_tau_heat = nn.Parameter(torch.tensor(math.log(0.25)))

        # Atom 2: soft shift-k.  Learnable (soft) offset and width.
        self.shift_k = nn.Parameter(torch.tensor(1.0))
        self.log_shift_width = nn.Parameter(torch.tensor(math.log(0.5)))

        # Atom 3: circulant / learnable banded relative-offset conv.  One weight
        # per offset in [-band, band].
        self.circ_weights = nn.Parameter(torch.zeros(2 * self.band + 1))

        # Atom 4: chiral U(1)-phased walk.  Learnable phase-rate φ and an
        # exponential envelope so the band stays local.
        self.chiral_phase = nn.Parameter(torch.tensor(0.6))
        self.log_chiral_decay = nn.Parameter(torch.tensor(math.log(0.15)))
        # Optional scalar that lets the complex-angle phase stream modulate φ.
        self.chiral_phase_drive = nn.Parameter(torch.tensor(0.0))

        # Atom 5: Chebyshev positional band-pass.  Low-order coefficients.
        self.cheb_order = 4
        self.cheb_coeffs = nn.Parameter(torch.zeros(self.cheb_order))

        # === v2 atoms (indices 5..9) ============================================

        # Atom 5 (mix index 5): Bessel CTQW — genuine continuous-time quantum
        # walk on Z.  Amplitude j -> i at time τ is J_{i-j}(2τ).  Computed
        # differentiably from the Jacobi-Anger generating function via FFT.
        self.log_tau_bessel = nn.Parameter(torch.tensor(math.log(1.0)))
        self.bessel_fft_min = 512  # minimum FFT grid (rounded up to pow2 >= 4S)

        # Atom 6 (mix index 6): complex-coined directed walk — a phase-carrying
        # directed walk that does NOT square-collapse.  Learnable coin angle φ,
        # a local envelope, and a learnable forward/backward (sign-of-d)
        # asymmetry so look-back (d>0) and look-ahead (d<0) get different mass.
        self.coined_phase = nn.Parameter(torch.tensor(0.6))
        self.log_coined_decay = nn.Parameter(torch.tensor(math.log(0.2)))
        self.coined_asym = nn.Parameter(torch.tensor(0.5))

        # Atom 7 (mix index 7): ballistic two-horn — the DTQW signature
        # distribution, a symmetric pair of Gaussian bumps at ±speed.
        self.log_horn_speed = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_horn_width = nn.Parameter(torch.tensor(math.log(0.75)))

        # Atom 8 (mix index 8): power-law / Lévy heavy tail 1 / (1 + |d|)^α.
        self.log_alpha = nn.Parameter(torch.tensor(math.log(1.0)))

        # Atom 9 (mix index 9): learnable circulant Hermitian Hamiltonian.
        # Learn the low-frequency Fourier symbol of a circulant generator H and
        # propagate exp(-iτ_H Ĥ); the per-offset bias row is Re(ifft(...)).
        self.h_symbol_modes = 8
        self.h_symbol = nn.Parameter(torch.zeros(self.h_symbol_modes))
        self.log_tau_hamiltonian = nn.Parameter(torch.tensor(math.log(0.5)))

        # Global output scale.
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def _resolve_v2_enabled(self, atom_set: str) -> list[bool]:
        """Map the ``walk_atom_set`` switch to a per-v2-atom enable mask.

        ``"v1"`` -> all five disabled (kernel == v1).  ``"v2"``/``"all"`` ->
        all enabled.  A comma/space-separated list of v2 atom names enables
        just those (v1 five always stay on); unknown names raise.
        """
        s = atom_set.strip().lower()
        if s in ("v1", "", "none"):
            return [False] * len(self._v2_names)
        if s in ("v2", "all"):
            return [True] * len(self._v2_names)
        canon = {name.lower(): name for name in self._v2_names}
        wanted_raw = [tok.strip() for tok in s.replace(",", " ").split() if tok.strip()]
        unknown = [tok for tok in wanted_raw if tok not in canon]
        if unknown:
            raise ValueError(
                f"unknown walk v2 atom(s) {sorted(unknown)}; "
                f"available: {list(self._v2_names)} (or 'v1'/'v2')"
            )
        wanted = {canon[tok] for tok in wanted_raw}
        return [name in wanted for name in self._v2_names]

    def _offsets(self, seq_len: int, device, dtype) -> torch.Tensor:
        """Relative-offset matrix d[i, j] = i - j, shape [S, S]."""
        pos = torch.arange(seq_len, device=device, dtype=dtype)
        return pos.unsqueeze(1) - pos.unsqueeze(0)

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        # phases: [B, S, F] (real angles) or complex unit vectors.
        if phases.is_complex():
            phase_angle = phases.angle()
            batch, seq_len, _ = phase_angle.shape
            real_dtype = phase_angle.dtype
            device = phase_angle.device
        else:
            batch, seq_len, _ = phases.shape
            phase_angle = phases
            real_dtype = phases.dtype
            device = phases.device

        d = self._offsets(seq_len, device, real_dtype)  # [S, S], d = i - j
        d_abs = d.abs()

        atoms: list[torch.Tensor] = []

        # --- Atom 1: path-heat / CTQW diffusion: exp(-τ d²) ---
        tau = F.softplus(self.log_tau_heat) + 1e-4
        w_heat = torch.exp(-tau * d * d)
        atoms.append(w_heat)

        # --- Atom 2: soft shift-k (causal previous-token addressing) ---
        # Mass concentrated at offset d = +k (attend k tokens back), Gaussian bump.
        width = F.softplus(self.log_shift_width) + 1e-3
        w_shift = torch.exp(-((d - self.shift_k) ** 2) / (2.0 * width * width))
        atoms.append(w_shift)

        # --- Atom 3: circulant / learnable banded relative-offset conv ---
        # circ_weights indexes offsets [-band, band]; outside the band -> 0.
        idx = (d + self.band).round().long()
        in_band = (idx >= 0) & (idx <= 2 * self.band)
        idx_clamped = idx.clamp(0, 2 * self.band)
        w_circ = self.circ_weights[idx_clamped] * in_band.to(real_dtype)
        atoms.append(w_circ)

        # --- Atom 4: chiral U(1)-phased quantum walk: Re(e^{i φ d}) · envelope ---
        phi = self.chiral_phase
        if self.use_phase_drive and phase_angle.shape[-1] > 0:
            # Let the mean complex-angle phase modulate the local rotation rate.
            # phase_angle: [B, S, F] -> per-token scalar drive, then relative.
            drive = phase_angle.mean(dim=-1)  # [B, S]
            drive_rel = drive.unsqueeze(2) - drive.unsqueeze(1)  # [B, S, S]
            phi_eff = phi + self.chiral_phase_drive * drive_rel  # [B, S, S]
            ang = phi_eff * d.unsqueeze(0)  # broadcast d -> [B, S, S]
            decay = torch.exp(-(F.softplus(self.log_chiral_decay) + 1e-4) * d_abs)
            w_chiral = torch.cos(ang) * decay.unsqueeze(0)  # [B, S, S]
        else:
            decay = torch.exp(-(F.softplus(self.log_chiral_decay) + 1e-4) * d_abs)
            w_chiral = torch.cos(phi * d) * decay  # [S, S]
        if not self.use_chiral:
            w_chiral = torch.zeros_like(w_chiral)
        atoms.append(w_chiral)

        # --- Atom 5: Chebyshev positional band-pass ---
        # Normalize offset to [-1, 1] over the band, evaluate Chebyshev T_k.
        x = (d / max(self.band, 1)).clamp(-1.0, 1.0)
        t_prev = torch.ones_like(x)
        t_cur = x
        cheb = self.cheb_coeffs[0] * t_prev + (
            self.cheb_coeffs[1] * t_cur if self.cheb_order > 1 else 0.0
        )
        for k in range(2, self.cheb_order):
            t_next = 2.0 * x * t_cur - t_prev
            cheb = cheb + self.cheb_coeffs[k] * t_next
            t_prev, t_cur = t_cur, t_next
        # Localize the band-pass to the same band as the other local atoms.
        cheb_env = torch.exp(-(d_abs / max(self.band, 1)))
        w_cheb = cheb * cheb_env
        atoms.append(w_cheb)

        # === v2 atoms (indices 5..9) ============================================

        d_long = d.round().long()  # exact integer offsets for table indexing

        # --- Atom 5 (index 5): Bessel CTQW on Z — exact J_{i-j}(2τ) ---
        # The genuine continuous-time quantum walk on the integer line.  Via the
        # Jacobi-Anger generating function  e^{i x sinθ} = Σ_d J_d(x) e^{i d θ},
        # the Bessel coefficients J_d(x) are exactly the Fourier coefficients of
        # g(θ) = exp(i x sinθ).  Sampling θ on an N-point grid and taking the FFT
        # of g recovers J_d(x) for d = 0..N-1 (and J_{-d} = (-1)^d J_d).  Using
        # torch.fft makes the whole thing autograd-differentiable in τ.
        #   Oscillatory with a ballistic light-cone front at d ≈ 2τ — contrast
        #   the v1 heat atom (atom 1), which is classical Gaussian diffusion.
        tau_b = F.softplus(self.log_tau_bessel) + 1e-4
        n_fft = 1
        target = max(self.bessel_fft_min, 4 * seq_len)
        while n_fft < target:
            n_fft *= 2
        theta = torch.arange(n_fft, device=device, dtype=real_dtype) * (
            2.0 * math.pi / n_fft
        )
        x_arg = 2.0 * tau_b
        g = torch.exp(1j * (x_arg * torch.sin(theta)))  # [N], complex
        bessel = (torch.fft.fft(g) / n_fft).real  # J_d(2τ) for d=0..N-1
        # Index by |d| then apply the parity relation J_{-d} = (-1)^d J_d.  For
        # d >= 0, J_d = (-1)^d J_{-d} too, so using |d| with the (-1)^|d|... no:
        # J_d for d>=0 is bessel[d]; for d<0, J_d = (-1)^{|d|} J_{|d|}.
        d_idx = d_long.abs().clamp(max=n_fft - 1)
        parity = torch.where(
            d_long < 0,
            (-1.0) ** d_long.abs().to(real_dtype),
            torch.ones_like(d, dtype=real_dtype),
        )
        w_bessel = bessel[d_idx] * parity  # [S, S]
        atoms.append(w_bessel)

        # --- Atom 6 (index 6): complex-coined directed walk (phase + direction) ---
        # A phase-carrying directed walk that does NOT square-collapse to
        # |amplitude|²: prior QW-transformers (CTQWformer / GQWformer) keep only
        # the magnitude; here we carry phase AND direction into the bias.
        #   bias(d) = Re( e^{i φ d} ) · envelope(d) · dir_gain(d)
        # with a learnable coin angle φ, a local exponential envelope, and a
        # learnable forward/backward asymmetry (look-back d>0 vs look-ahead d<0).
        coined_decay = torch.exp(
            -(F.softplus(self.log_coined_decay) + 1e-4) * d_abs
        )
        # Directional gain: sigmoid(asym) weights d>0, (1-·) weights d<0.
        fwd = torch.sigmoid(self.coined_asym)
        dir_gain = torch.where(d > 0, fwd, torch.where(d < 0, 1.0 - fwd, 0.5 * torch.ones_like(d)))
        w_coined = torch.cos(self.coined_phase * d) * coined_decay * dir_gain
        atoms.append(w_coined)

        # --- Atom 7 (index 7): ballistic two-horn (DTQW signature) ---
        # Discrete-time quantum walk transport: probability mass concentrates at
        # d ≈ ±c (two outward "horns" racing apart at ±c·t), the opposite of the
        # heat atom's single central peak.  Implemented as a symmetric pair of
        # Gaussian bumps at ±speed with learnable speed and width.
        speed = F.softplus(self.log_horn_speed) + 1e-3
        horn_w = F.softplus(self.log_horn_width) + 1e-3
        w_horn = torch.exp(-((d_abs - speed) ** 2) / (2.0 * horn_w * horn_w))
        atoms.append(w_horn)

        # --- Atom 8 (index 8): power-law / Lévy heavy tail ---
        # Heavy-tailed (polynomial) long-range coupling 1 / (1 + |d|)^α with
        # learnable α — a fractional-Laplacian (-Δ)^s style walk.  Contrast the
        # heat / ALiBi exponential decay: this keeps non-negligible mass at large
        # |d| (long-range), governed by a single learnable exponent.
        alpha = F.softplus(self.log_alpha) + 1e-3
        w_power = torch.pow(1.0 + d_abs, -alpha)
        atoms.append(w_power)

        # --- Atom 9 (index 9): learnable circulant Hermitian Hamiltonian ---
        # Learn the walk's dispersion relation, constrained to the CIRCULANT
        # (hence EQUITABLE) family.  We learn the low-frequency Fourier symbol of
        # a circulant Hermitian generator H, build the full real symbol Ĥ of
        # length S, propagate exp(-iτ_H Ĥ) in the Fourier basis, and read the
        # per-offset bias as W[i,j] = Re( ifft(exp(-iτ_H Ĥ)) )[(i-j) mod S].
        #   Learnable yet certified: it automatically inherits graphplay's
        #   verified equitable-quotient / mixing semantics, unlike an
        #   unconstrained learnable Laplacian.  Subsumes heat / wave / Bessel /
        #   chiral as special spectral choices.
        tau_h = F.softplus(self.log_tau_hamiltonian) + 1e-4
        n_modes = min(self.h_symbol_modes, seq_len)
        # Build a real, conjugate-symmetric circulant symbol Ĥ of length S by
        # placing the learnable low-frequency values at k = 1..n_modes-1 and
        # mirroring them to k = S-1..S-n_modes+1 (Ĥ_k = Ĥ_{S-k}); k=0 (DC) is the
        # zeroth learnable value.  Symmetry in k makes the propagator kernel real.
        k_axis = torch.arange(seq_len, device=device)
        h_full = torch.zeros(seq_len, device=device, dtype=real_dtype)
        h_full[0] = self.h_symbol[0]
        for m in range(1, n_modes):
            h_full = h_full + self.h_symbol[m] * (
                (k_axis == m).to(real_dtype) + (k_axis == (seq_len - m)).to(real_dtype)
            )
        propagator = torch.exp(-1j * tau_h * h_full.to(torch.complex64))
        kernel_offsets = torch.fft.ifft(propagator).real  # [S], indexed by offset mod S
        off_mod = (d_long % seq_len).clamp(0, seq_len - 1)
        w_hamiltonian = kernel_offsets[off_mod]  # [S, S]
        atoms.append(w_hamiltonian)

        # --- Simplex mixture over atoms ---
        mix = F.softmax(self.mix_logits, dim=0)
        # v1/v2 lock: zero the contribution of any disabled v2 atom (indices
        # 5..9) and renormalize, exactly analogous to the chiral ablation.  In
        # "v1" mode all five are zeroed so the kernel reduces to the v1 mixture.
        v2_mask = torch.ones_like(mix)
        for k, enabled in enumerate(self.v2_enabled):
            if not enabled:
                v2_mask[5 + k] = 0.0
        if not v2_mask.all():
            mix = mix * v2_mask
            mix = mix / mix.sum().clamp(min=1e-8)
        if not self.use_chiral:
            # Force the chiral atom's mixture mass to zero and renormalize so the
            # ablation truly removes the U(1) atom (no leakage).
            mask = torch.ones_like(mix)
            mask[3] = 0.0
            mix = mix * mask
            mix = mix / mix.sum().clamp(min=1e-8)

        result = phases.new_zeros((batch, seq_len, seq_len), dtype=real_dtype)
        for a, w in enumerate(atoms):
            if w.dim() == 2:
                w = w.unsqueeze(0)  # [1, S, S] broadcast over batch
            result = result + mix[a] * w
        return self.output_scale * result


# =============================================================================
# Registry
# =============================================================================

_KERNEL_REGISTRY: dict[str, Callable[..., ResonanceKernel]] = {
    "cosine": CosineKernel,
    "harmonic_cosine": HarmonicCosineKernel,
    "cosine_weighted": WeightedCosineKernel,
    "pair_mlp": PairMLPKernel,
    "dot": DotKernel,
    "rbf": RBFKernel,
    "laplace": LaplaceKernel,
    "bilinear": BilinearKernel,
    "complex_magnitude": ComplexMagnitudeKernel,
    "complex_real": ComplexRealKernel,
    "attention": AttentionKernel,
    "directional_complex": DirectionalComplexKernel,
    "walk": WalkKernel,
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
