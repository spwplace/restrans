# Literature Integration Plan

Date: 2026-04-28

This document integrates the external literature report into the active
research program. It is project policy for the next execution phase.

## Main Reframe

The broad architectural category is not novel. Relation-aware attention,
content/position disentanglement, graph transformer structural biases,
Abstractor-style relational bottlenecks, and Dual Attention Transformer-style
sensory/relational splits already occupy much of the space.

Our strongest claim is therefore narrower:

> A compact, explicitly bottlenecked structural stream may improve models on
> structure-dependent tasks when the data and training objective actually
> reward structural invariants; and the stream should be more probeable,
> compressible, and causally implicated in structural decisions than the
> ordinary residual stream.

The current scalar additive relation bias is not the thesis. It is one
candidate readout. Current experiments already suggest that phase-bearing and
phase-only conditions can tie, so the main object of study is the structural
channel plus the pressures that make it meaningful.

## What Changes Now

### 1. Baselines become non-negotiable

Future architecture claims need these comparators:

- vanilla transformer with RoPE or equivalent standard position handling;
- vanilla transformer with ALiBi or learned relative-position bias;
- DeBERTa-style disentangled content/position split;
- one explicit relational-stream comparator inspired by Abstractor or Dual
  Attention Transformer;
- Graphormer/GRPE-style structural bias when the input has known graph
  structure;
- syntax-aware baseline only when we use parser/dependency supervision.

Without these, a positive result is only "a pairwise bias or extra stream helped
some task", which is too weak.

### 2. The current `R` attention bias is demoted

Keep `R_ij = mean_f cos(phi_i^f - phi_j^f)` as an ablation and diagnostic, but
do not treat it as the privileged mechanism.

Higher-priority mechanisms:

- phase/structural stream with auxiliary structural objectives;
- contrastive invariance losses over same-structure examples;
- supervised structural losses where labels exist;
- dynamic per-layer phase state;
- phase-conditioned Q/K or value mixing;
- residual/FFN gating from structural state;
- directional relation kernels for ordered dependencies.

### 3. Microvalidations are instrumentation, not gates for all scaling

Small probes should catch broken datasets, shortcut saturation, and useless
adapters. They should not delay a properly instrumented 10M-30M medium run.

The next medium run can be exploratory if it includes:

- matched baselines;
- task families where structure is actually required;
- phase/residual probes;
- ablations that zero, shuffle, freeze, or patch the structural stream;
- logs and checkpoints sufficient for post-hoc interpretability.

## Benchmark Priorities

### Tier A: cheap instrumentation

Use these to debug adapters and verify structural signal:

- Dyck / cross-Dyck;
- ListOps or equivalent parse-composition task;
- Linzen agreement and BLiMP subsets;
- MSGS and HANS as shortcut-preference diagnostics;
- cap matching, ordinary unification, and algebraic protocol closure;
- small ContraCode/EquiBench-style transformed program pairs when available.

Tier A success is not a paper. It only tells us which setups are not broken.

### Tier B: main evidence

Pick two families for the first serious run:

- natural-language structural generalization: SLOG plus CFQ or COGS;
- formal/code structure: EquiBench if available, otherwise our cap matching /
  unification / algebraic protocol suite plus a public code-equivalence proxy;
- proof structure: LeanProgress is the preferred first Lean-family target once
  staged, because it is cheaper than full proof search.

The most attractive first paper result is a 10M-30M model that beats matched
baselines on at least two controlled structural families and shows the stream
is causally used.

### Tier C: transfer to natural language

Use BabyLM or TinyStories as the natural corpus, but evaluate on structural
probes rather than only perplexity:

- BLiMP;
- MSGS;
- HANS;
- Linzen;
- SLOG/COGS/CFQ if using seq2seq or probe adapters.

The core transfer question is:

> Does structural pretraining or multitask pressure make the structural stream
> useful on natural language syntax and semantic generalization?

Recommended conditions:

- natural-only vanilla;
- natural-only structural stream;
- synthetic-structural warmup then natural;
- natural plus structural multitask mixture;
- natural plus auxiliary structural losses.

## Architecture Matrix For The Next Medium Run

Do not run every clever idea immediately. The first medium matrix should be:

1. `standard_rope`
2. `standard_relative_or_alibi`
3. `deberta_split_lite`
4. `structural_phase_aux_only`
5. `structural_phase_contrastive`
6. `structural_phase_supervised` where labels exist
7. `resonance_logit_bias` as the legacy mechanism
8. `phase_conditioned_qk` or `residual_gate`, but not both unless compute is
   abundant
9. `relational_stream_lite` inspired by Abstractor/DAT

For every condition, log:

- train and validation loss;
- task metric by structural split;
- phase-vs-residual probe accuracy by layer;
- effective rank/eigenspectrum of phase, residual, and relation matrices;
- phase ablation, phase shuffle, and phase freeze deltas;
- attention or relation heatmaps on fixed examples.

## Interpretability Standard

Probe-only results are not enough. A useful structural stream should satisfy at
least two of:

- structure labels are more linearly accessible from the structural stream than
  from matched residual states;
- the stream or relation matrix is more compressible on structure-rich data;
- ablating or corrupting the stream selectively hurts structural behavior;
- patching the stream rescues structural decisions;
- sparse or geometric decompositions are cleaner than for the residual stream.

If the stream is interpretable but not useful, say that. If it is useful but not
interpretable, say that. Either result is better than an overbroad claim.

## Concrete Next Steps

1. Update the paper/site language:
   - remove metaphorical or physics-flavored framing;
   - stop calling the cosine relation or phase stream novel by itself;
   - frame the contribution around bottlenecked structural representations,
     training pressure, controlled tasks, and causal interpretability.

2. Implement and strengthen required baselines:
   - relative/ALiBi baseline is implemented as `standard_alibi`;
   - DeBERTa-style split baseline is implemented as `standard_deberta_lite`;
   - first Abstractor/DAT-lite comparator is implemented as
     `relational_stream_lite`;
   - stronger graph-aware and relational-stream parity baselines remain open.

3. Stage missing Tier B datasets:
   - EquiBench or a nearest public equivalent;
   - LeanProgress;
   - ListOps if not already staged.

4. Build a medium-run driver:
   - first pass is `scripts/run_literature_medium_suite.sh`;
   - config summary is `configs/literature_medium_suite.json`;
   - remaining work: resumable checkpoints, fixed eval examples for plots,
     and cloud-launch wrappers.

5. Run the first medium suite:
   - 10M-ish models first;
   - SLOG plus CFQ/COGS;
   - cap matching/unification/algebraic protocol or EquiBench;
   - BabyLM/TinyStories transfer only after the structural suite is logging
     cleanly.

6. Only then scale to 30M-50M or cloud-wide sweeps.

## Stop / Go Criteria

Proceed to larger runs if all three are true:

- selective gains appear on controlled structural splits;
- the structural stream is more probeable or compressible for structural labels;
- stream ablations or patching show causal involvement.

Reframe or stop if:

- gains disappear against relative-bias, DeBERTa-style, or relational-stream
  baselines;
- phase-only, inert, and full resonance are indistinguishable across real tasks;
- improvements occur only on toy tasks or easy random splits;
- the stream is unused under ablation.

The strongest negative result is still valuable: it would show that this class
of low-dimensional structural streams is either captured by existing
relation-aware methods or requires explicit structural pressure to become
useful.
