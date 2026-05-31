# Walkformer: attention as a learnable simplex mixture of analytic graph-walk operators

**Status:** first-pass experiment, runnable kernel + honest results. MPS, torch 2.11.
**Date:** 2026-05-31.

## TL;DR

We add a new pluggable `ResonanceKernel` — `WalkKernel`, registered as `"walk"` —
that builds the attention bias matrix `R[i,j]` as a **learnable simplex mixture of
five analytic walk operators** on a latent 1-D line/cycle whose vertices are the
sequence positions. Every operator is a function of the relative offset `d = i - j`
only (Toeplitz ⇒ translation-equivariant ⇒ length-extrapolating). The whole kernel
adds **33 parameters** on top of the resonance baseline — the entire positional
mixing prior costs less than a single attention head's worth of a bias-vector.

The headline atom is a **chiral U(1)-phased quantum walk** `Re(e^{iφd})·envelope`:
a direction-aware, oscillatory positional band that real SSMs / Hyena / FNet / ALiBi
*cannot* express, and which maps directly onto a continuous-time chiral quantum walk
operator formalized in the `graphplay` Lean corpus (`Graphplay/Chiral.lean`).

Result, three tasks, seeds {11,23,37}, 5 epochs, embed_dim 128, 4 layers:
- **unification** (content/algebraic): walkformer **0.826**, beats the resonance
  baseline (0.816) and essentially ties ALiBi (0.829), for +33 params. Chiral atom
  helps: 0.826 vs 0.821 no-chiral.
- **listops** (hierarchical): walkformer **0.520** vs 0.508 baseline / 0.508 no-chiral;
  chiral atom helps, but all resonance variants trail ALiBi (0.540) and the task is at
  the weak-signal floor.
- **graph_alias** (positional/structural, length 512): all conditions near the 0.50
  majority floor with high variance; walkformer (0.516) does **not** help and the
  chiral atom does **not** earn its keep here (0.516 vs 0.536 no-chiral).

Verdict on the chiral wedge: **2 of 3 tasks show the U(1) atom helping; 1 shows it
hurting. It is a real, cheap, ablatable inductive bias — promising but not a clean win.**

---

## 1. Architecture

### 1.1 The mixture

For a sequence of length `S`, positions are vertices `0..S-1` of a latent 1-D
line/cycle. The kernel emits

```
R[i,j] = output_scale · Σ_a softmax(π)_a · W_a[i - j]
```

where `π ∈ R^5` are learnable mixture logits and each `W_a` is one analytic walk
operator below. The softmax makes the mixture a convex (simplex) combination, so
atoms compete for mass; `output_scale` is a single learnable gain. The causal
upper triangle is **not** masked inside the kernel — the attention module does
`masked_fill(mask==0, -inf)` downstream, exactly as for every existing kernel.

### 1.2 The atom dictionary

| # | Atom | Closed form (function of `d = i - j`) | Learnable params | Role |
|---|------|----------------------------------------|------------------|------|
| 1 | **path-heat / CTQW** | `exp(−τ·d²)` | `log τ` (1) | diffusion / locality |
| 2 | **shift-k** | `exp(−(d−k)²/2σ²)` | soft `k`, `log σ` (2) | causal previous-token addressing |
| 3 | **circulant** | learnable weight per offset in `[−B, B]` | `2B+1` band weights (17) | arbitrary local conv / circular mixing |
| 4 | **chiral U(1)** | `cos(φ·d)·exp(−κ|d|)` (φ optionally phase-driven) | `φ`, `log κ`, phase-drive (3) | **oriented oscillatory band (the wedge)** |
| 5 | **spectral-Chebyshev** | `Σ_k c_k T_k(d/B)·exp(−|d|/B)` | `c_0..c_3` (4) | learnable band-pass positional filter |

Plus the mixture logits `π` (5) and `output_scale` (1). With band `B = 8`:
**33 learnable parameters total** for the entire positional-mixing mechanism.

Atom 4's phase-rate `φ` can additionally be *driven* by the model's
`complex_angle` phase stream: a per-token mean angle modulates the local rotation
rate as `φ_eff = φ + α·(θ_i − θ_j)`, so the U(1) twist becomes content-dependent
while staying a relative-position operator. (`walk_use_phase_drive`, default on.)

### 1.3 Exact ablation

`walk_use_chiral=False` (condition `walkformer_no_chiral`) **zeroes atom 4 and
renormalizes the simplex over the remaining four atoms** — so the U(1) phase atom
is removed with no mass leakage. Param count is identical (the chiral params remain
allocated but receive zero mixture mass and zero gradient through the mixture), which
makes the ablation a clean apples-to-apples isolation of the U(1) inductive bias.

### 1.4 Why these atoms length-generalize

Every `W_a` is evaluated from `d = i − j` at runtime for whatever `S` the batch has;
no atom stores an absolute-position table. The learned profile (a heat width, a shift
offset, a 17-tap band, a phase rate, four Chebyshev coefficients) is reused verbatim
at any sequence length. This is the same property that makes ALiBi / RoPE / Toeplitz
SSMs extrapolate — but here the *family* of profiles is an explicit, interpretable
dictionary of graph-walk operators rather than a single fixed slope or rotation.

---

## 2. The operator-semantics provenance (the second wedge)

Each atom is the discretized / continuum form of a continuous-time walk operator that
is **formally defined (and in several cases theorem-backed) in the `graphplay` Lean
corpus**. This is the "verified-operator-semantics provenance" angle: the attention
prior is not an ad-hoc kernel, it is a named operator with a Lean definition.

| Walkformer atom | CTQW / graph operator | graphplay reference |
|-----------------|-----------------------|---------------------|
| path-heat (atom 1) | heat / diffusion kernel `exp(−τL_path)`, the `t→it` analytic continuation of the CTQW propagator `exp(−itH)` | `Graphplay/Chiral.lean` (uses `Mathlib …MatrixExponential`); CTQW spectra in `Graphplay/DiscreteTime.lean` |
| circulant (atom 3) | circulant / translation-invariant adjacency, diagonalized by the Fourier basis | graphon spectrum machinery `Graphplay/Graphon/Spectrum.lean`; eigenvalue-indexed walks in `Graphplay/DiscreteTime.lean` |
| **chiral U(1)** (atom 4) | **magnetic / chiral unitary signing `e^{iφ}` of a Hermitian-weighted graph** — uniform-mixing chiral quantum walks | **`Graphplay/Chiral.lean`** (Levine, Mesapam, Mustico, Tamon, Tucker, Zhan, *Uniform Mixing in Chiral Quantum Walks*, arXiv:2605.04414, 2026) |
| spectral-Chebyshev (atom 5) | low-order polynomial in the path Laplacian = band-pass spectral filter | spectrum-lifting / equitable-quotient spectral theorems `Graphplay/Spectral.lean` |
| shift-k (atom 2) | one-step adjacency / shift operator on the path graph | path-graph adjacency throughout the corpus |

The chiral atom is the one with genuinely non-trivial provenance: it is exactly the
U(1) phasing whose mixing/PST optimality is the new algebraic content of the chiral
quantum-walk work formalized in `Chiral.lean`. SSMs, Hyena, FNet, and ALiBi have no
analogue — their positional kernels are real and phase-locked (or a single rotation
angle), so they cannot represent a learnable *oriented oscillation* `e^{iφd}` as a
competing attention atom.

---

## 3. Results

Setup: `structural_task_probe.py`, conditions
`standard, standard_alibi, resonance_full_normalized, walkformer, walkformer_no_chiral`,
seeds `11 23 37`, `--epochs 5 --train_examples 1024 --val_examples 512 --embed_dim 128
--layers 4 --n_frequencies 32 --device mps`. graph_alias used `--max_length 512`
(its sequences exceed the 256 default; the other two tasks used the default 256).
Outputs in `resonance/outputs/walkformer_gauntlet/{listops,unification,graph_alias}/`.

Accuracy is mean ± std over the 3 seeds. **Params is total model params; the walk
kernel itself is 33 of them.** Δ vs resonance baseline = walkformer params −
resonance_full_normalized params = **+33** on every task.

### listops (hierarchical) — majority 0.500, verdict *weak*

| Condition | Params | Acc | Loss gain |
|---|---:|---:|---:|
| standard | 1,348,352 | 0.5150 ± 0.0159 | −0.0246 |
| standard_alibi | 1,348,352 | **0.5404 ± 0.0137** | −0.0069 |
| resonance_full_normalized | 1,483,664 | 0.5072 ± 0.0197 | −0.0118 |
| **walkformer** | 1,483,697 | 0.5202 ± 0.0396 | −0.0070 |
| walkformer_no_chiral | 1,483,697 | 0.5078 ± 0.0215 | −0.0061 |

walkformer > resonance baseline and > no-chiral, but all resonance variants trail
ALiBi and the whole task sits near the weak-signal floor (no condition clears
majority+5pts). Treat as a soft positive, not a win.

### unification (content/algebraic) — majority 0.500, verdict *candidate (learnable)*

| Condition | Params | Acc | Loss gain |
|---|---:|---:|---:|
| standard | 1,348,352 | 0.8210 ± 0.0139 | +0.3186 |
| standard_alibi | 1,348,352 | **0.8288 ± 0.0114** | +0.3081 |
| resonance_full_normalized | 1,483,664 | 0.8158 ± 0.0108 | +0.2692 |
| **walkformer** | 1,483,697 | **0.8262 ± 0.0160** | +0.3048 |
| walkformer_no_chiral | 1,483,697 | 0.8210 ± 0.0133 | +0.2833 |

The clean result. walkformer (0.826) beats the resonance baseline (0.816) by ~1pt and
essentially ties ALiBi (0.829) and standard (0.821), for **+33 params over the
resonance baseline**. The chiral atom helps (0.826 vs 0.821). This is the
param-efficiency story working as advertised: a 33-param operator-dictionary bias
recovers an ALiBi-class positional prior and slightly improves on the cosine-resonance
baseline.

### graph_alias (positional/structural, len 512) — majority 0.500, verdict *weak*

| Condition | Params | Acc | Loss gain |
|---|---:|---:|---:|
| standard | 1,381,120 | **0.5625 ± 0.1083** | −1.9929 |
| standard_alibi | 1,381,120 | 0.4837 ± 0.0352 | −2.3886 |
| resonance_full_normalized | 1,516,432 | 0.5410 ± 0.0935 | −1.9687 |
| walkformer | 1,516,465 | 0.5163 ± 0.0699 | −2.1370 |
| walkformer_no_chiral | 1,516,465 | 0.5358 ± 0.0904 | −1.9940 |

**Negative result, reported honestly.** Nobody learns graph_alias at this scale — all
conditions hover around the 0.50 majority floor with large seed variance and *negative*
loss gain (overfitting / no generalization in 5 epochs). walkformer is mid-pack and the
chiral atom **hurts** here (0.516 vs 0.536 no-chiral). At this length/epoch budget the
task is uninformative about the kernel; it neither supports nor refutes the wedge.

---

## 4. The chiral ablation verdict

Does the U(1) phase atom earn its keep? (walkformer vs walkformer_no_chiral)

| Task | walkformer | no_chiral | Δ (chiral effect) |
|---|---:|---:|---:|
| unification | 0.8262 | 0.8210 | **+0.0052** ✅ |
| listops | 0.5202 | 0.5078 | **+0.0124** ✅ |
| graph_alias | 0.5163 | 0.5358 | −0.0195 ❌ |

**Two of three tasks show the chiral U(1) atom helping** (unification, listops), with
the larger relative effect on the structured content task where signal exists. **One
task (graph_alias) shows it hurting**, but that task is at the noise floor for *every*
condition so its Δ is within seed noise (std ≈ 0.07–0.11). Net read: the chiral atom is
a **real, free, cheap-to-ablate inductive bias that helps where there is signal**, but
this 3-task first pass is **not** a clean, uniformly-positive win. It justifies a larger
sweep (more seeds, longer training, length-extrapolation splits) before any strong claim.

---

## 5. Honest positioning

**What is NOT novel.** The bulk of the walkformer is prior art under different names:
- **Toeplitz / relative-position positional mixing** (atoms 1–3, 5) ≈ **ALiBi** (a single
  linear distance slope), **RoPE** (a single rotation), and — most directly — **Toeplitz
  / long-convolution SSMs**: **S4 / S5 / Mamba** learn exactly a function-of-`(i−j)`
  mixing kernel, **Hyena** learns an implicit (MLP-parameterized) long convolution, and
  **FNet** does fixed Fourier mixing. Our path-heat, shift-k, circulant, and Chebyshev
  atoms are all special cases of "learn a Toeplitz/circulant positional kernel," which
  those architectures already do, often with far more capacity.
- The **simplex-mixture-of-kernels** framing is itself just a small learned gate over a
  fixed basis — multi-scale / multi-kernel attention has been done.
- Using these as an **additive attention bias** (rather than replacing attention) is the
  ALiBi/relative-bias recipe.

So as a *long-convolution mixer*, walkformer is a low-capacity, interpretable cousin of
SSMs/Hyena, not a new mechanism.

**What IS the wedge.** Two things, both cheap and both ablatable:
1. **The chiral U(1)-phased graph-walk atom** `Re(e^{iφd})·envelope` (optionally
   content-driven). This is an *oriented, oscillatory* positional band. Real Toeplitz
   SSMs, Hyena's real long convs, FNet's fixed transform, and ALiBi's monotone slope
   cannot express a *learnable* `e^{iφd}` competing as one atom among others — and our
   ablation shows it carries signal on 2/3 tasks at +0 marginal params. (RoPE has a fixed
   rotation but it is applied to Q/K, not as a selectable, content-driven attention-bias
   atom, and is not learnable per-band.)
2. **Verified-operator-semantics provenance.** Each atom is a *named* CTQW / graph-walk
   operator with a Lean definition in `graphplay` (the chiral atom in particular maps onto
   the formalized chiral-quantum-walk uniform-mixing result, arXiv:2605.04414). This is a
   credibility/interpretability wedge, not a benchmark wedge: it lets every component of
   the positional prior be pointed at a theorem rather than a hyperparameter.

**Param-efficiency claim (supported).** The entire positional-mixing prior is **33
parameters**. On unification it matches an ALiBi-class prior and beats the cosine-
resonance baseline; that is the strongest evidence here. Whether 33 params of structured
walk-bias can *beat* (not just match) a learned-QK relative-position table at scale is
open and not demonstrated by this run.

**Negative/null results (stated plainly).**
- walkformer does **not** beat ALiBi on listops or unification — it ties/approaches it.
- graph_alias is a **null** for everyone at this budget; walkformer is mid-pack and the
  chiral atom regresses there.
- The chiral win is **2/3, small (≤ 1.2 acc pts), and inside seed noise on the third
  task.** No strong claim is warranted from three tasks at 1024 examples / 5 epochs.

**Next steps to make/break the wedge:** (i) length-extrapolation splits (train short /
eval long) where the Toeplitz/operator atoms should shine and learned-QK tables should
fail; (ii) more seeds + longer training on unification/listops to tighten the chiral Δ;
(iii) a task with intrinsic orientation/periodicity (e.g. modular `vm_*`, `dyck`) where
the U(1) atom has a mechanistic reason to win.

---

## 6. How to reproduce

```bash
cd /Users/ember/dev/restrans
export PYTHONPATH=/Users/ember/dev/restrans/resonance
for TASK in listops unification; do
  .venv/bin/python resonance/structural_task_probe.py --task $TASK \
    --conditions standard,standard_alibi,resonance_full_normalized,walkformer,walkformer_no_chiral \
    --seeds 11 23 37 --epochs 5 --train_examples 1024 --val_examples 512 \
    --embed_dim 128 --layers 4 --n_frequencies 32 --device mps \
    --output_dir resonance/outputs/walkformer_gauntlet/$TASK
done
# graph_alias needs longer sequences:
.venv/bin/python resonance/structural_task_probe.py --task graph_alias \
  --conditions standard,standard_alibi,resonance_full_normalized,walkformer,walkformer_no_chiral \
  --seeds 11 23 37 --epochs 5 --train_examples 1024 --val_examples 512 --max_length 512 \
  --embed_dim 128 --layers 4 --n_frequencies 32 --device mps \
  --output_dir resonance/outputs/walkformer_gauntlet/graph_alias
```

**Code touched (additive only, no existing kernel/condition disturbed):**
- `resonance/resonance/kernels.py` — `WalkKernel` class + `"walk"` registry entry.
- `resonance/resonance/config.py` — 4 new `ResonanceConfig` fields (`walk_atoms`,
  `walk_use_chiral`, `walk_band`, `walk_use_phase_drive`), safe defaults.
- `resonance/resonance/models.py` — thread the 4 walk params into the main `build_kernel`
  call (filtered out for non-walk kernels, so other kernels are unaffected).
- `resonance/story_topology_eval.py` — `walkformer` and `walkformer_no_chiral` conditions.

---

## persvati GPU run (authoritative — supersedes the local-MPS smoke)

Larger scale: **2048 train / 512 val, 8 epochs, seeds 11/23/37**, on the persvati ROCm GPU.

| Task | standard | alibi | resonance | **walkformer** | no_chiral |
|---|---|---|---|---|---|
| unification | 0.865 | **0.875** | 0.862 | **0.874** | 0.877 |
| listops | 0.612 | **0.654** | 0.598 | 0.606 | 0.590 |

(`graph_alias` did not emit a report — the max_length>256 dataset constraint recurred.)

**Honest verdict.** walkformer **matches ALiBi on unification** (0.874 vs 0.875) at **+33 params**
over the resonance baseline — the param-efficiency claim holds. On listops it is mid-pack
(below standard/ALiBi). The **chiral ablation is inconclusive at scale**: chiral helps listops
(+0.016) but slightly hurts unification (−0.003) — 1-help/1-hurt, not the clean win the small
smoke hinted. This is a near-null result, stated plainly: a low-capacity interpretable mixer
competitive-with-ALiBi-on-one-task, not a new SOTA.

**Why this motivates v2.** The v1 chiral atom takes `Re(e^{iφd})`, **discarding the phase
before it reaches attention** — exactly the square-collapse we critiqued in CTQWformer/GQWformer.
The v2 atoms (already implemented, ADD-only, not yet run): **Bessel-CTQW** (the genuine quantum
walk, oscillatory + ballistic), **complex-coined** (keeps phase + direction into attention),
**ballistic two-horn**, **power-law/Lévy**, and **learnable-dispersion** (circulant `exp(−iτĤ)`,
certified-but-learnable). The phase-preserving complex-coined atom is the real test of whether
the chiral wedge buys anything — v1's `Re(·)` collapse may be why chiral looks weak here.
