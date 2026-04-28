# Next Directions: Resonance, Topology, and Natural Language

Date: 2026-04-26

## Thesis

The research question is not whether resonance helps arbitrary language
modeling. The question is whether an architecture with a compact phase/relation
channel can learn, expose, preserve, and steer **structural equivalence** more
effectively than an ordinary transformer.

The plan is therefore to build around topology-bearing data:

- formal proof/program equivalence,
- semantic graph equivalence,
- natural-language realizations of the same proposition/event graph,
- hard negatives that preserve words while breaking structure.

## Guiding Hypotheses

1. **Structural factors can be made inspectable.**
   If phase/resonance is useful, it should align with known structural labels:
   normal form, proof step, AST role, semantic graph role, discourse edge, or
   causal/temporal relation.

2. **Topology should be causal, not decorative.**
   Phase patching, phase ablation, or resonance-bias patching should change
   structural behavior while leaving lexical/semantic content comparatively
   intact.

3. **The bridge to natural language is a semantic graph.**
   We should not jump directly from lambda terms to free-form text. We should
   generate multiple surface stories from one verified proposition/event graph,
   then generate near-miss hard negatives by changing one graph edge.

4. **A useful result must survive controls.**
   Every claim needs a corrected standard baseline, an iso-parameter baseline,
   bias-only/stream-only/inert ablations, multiple seeds, and intervention
   evidence.

## Track 1: Interpretability Tools

### Build

- Trace every internal stream:
  - semantic embeddings,
  - raw phase embeddings,
  - projected phase stream,
  - blend gate,
  - resonance matrix,
  - per-layer `QK` logits,
  - per-layer resonance logits,
  - per-layer combined attention logits,
  - attention deltas caused by resonance,
  - hidden states.
- Add intervention helpers:
  - phase noise,
  - phase permutation,
  - phase zeroing,
  - resonance weight scaling,
  - phase patching between clean/corrupt examples,
  - semantic patching while phase is fixed.
- Add diagnostics:
  - resonance entropy,
  - effective rank,
  - phase PCA rank,
  - phase drift over training,
  - correlation with known structure labels,
  - attention delta attribution by head/layer.

### Experiments

- Same-normal-form proof pairs:
  - patch phase from equivalent proof into corrupted proof,
  - measure retrieval/logit recovery.
- Same-story-graph paraphrases:
  - swap phase between paraphrases,
  - swap semantic stream while holding phase fixed.
- Hard negatives:
  - same words/entities, relation flipped,
  - verify whether phase/resonance detects the graph break.

### Interpretability Success Criteria

- Phase/resonance features have higher label purity than residual-stream
  features on known structure labels.
- Phase interventions causally affect structural tasks.
- SAEs on phase/projection streams are simpler or more monosemantic than SAEs
  on the ordinary residual stream, measured by feature purity and causal
  faithfulness, not vibes.

## Track 1b: Intrinsic Interpretability and Geometry

Recent interpretability work points in two useful directions for this project:

- Do not rely only on post-hoc SAEs.  Add interpretability pressure during
  training and check whether the performance/interpretability tradeoff is real
  for phase/resonance models.
- Do not assume every useful representation is sparse and axis-aligned.  The
  phase stream may be cleanly sparse on formal/syntactic tasks, but natural
  language and multimodal structure may be better described as low-dimensional
  regions, convex mixtures, or archetype coordinates.

### Built-In Interpretability Experiments

- Add sparse penalties to the structural path:
  - activation L1 or Hoyer sparsity on raw phase,
  - activation L1 or Hoyer sparsity on projected phase,
  - top-k or sparse-gated resonance features,
  - sparse contrastive projection heads for same-structure groups.
- Compare four training regimes:
  - ordinary standard transformer,
  - ordinary resonance transformer,
  - standard transformer with matched sparse bottleneck,
  - resonance transformer with sparse structural bottleneck.
- Evaluate the Pareto frontier:
  - task accuracy/loss,
  - phase/resonance label purity,
  - causal patching faithfulness,
  - SAE feature purity,
  - probe sparsity and stability across seeds.

### Geometry Experiments

- Fit archetype/hull probes to phase vectors, resonance rows, and residual
  stream states.
- Measure whether examples occupy:
  - sparse feature axes,
  - low-rank subspaces,
  - convex mixtures of a small archetype set,
  - graph/spectral neighborhoods induced by known structure.
- Use this as a guardrail: if phase is not monosemantic, that is not
  automatically a failure.  It may still be interpretable as geometry.

### Success Criteria

- Intrinsic sparsity improves interpretability without destroying the structural
  task signal.
- Phase/resonance geometry is simpler than residual-stream geometry under at
  least one faithful probe family.
- The best probe family is allowed to vary by modality: formal programs,
  syntax, semantic graphs, stories, and code do not need the same geometry.

## Track 2: Synthetic Topological Priors to Natural Language

### Representation

Use a small typed semantic graph:

```text
entities: people / objects / places
events: predicate(subject, object, optional target/location)
edges: before, causes, enables, prevents, because, believes, wants
truth operators: negation, quantifier-like all/some, modal/belief
```

Each graph has:

- a canonical signature,
- multiple text renderings,
- hard negative graph mutations,
- generated QA checks whose answers are invariant across paraphrases.

### Positive Groups

Same graph, different surface:

- lexical substitutions,
- active/passive alternation,
- temporal reordering with discourse markers,
- pronoun/name alternation,
- short/simple vs story-like rendering,
- explicit vs implicit cause.

### Hard Negatives

Same lexical material, changed topology:

- reverse cause/effect,
- swap giver/receiver,
- flip before/after,
- remove negation,
- change belief holder,
- change object identity,
- change enabling/preventing relation.

### Objectives

- language modeling,
- same-graph contrastive loss,
- hard-negative discrimination,
- relation reconstruction,
- graph distance / edge prediction,
- phase consistency across paraphrases.

## Track 3: Actual Natural Language

### Curriculum

1. Verified formal equivalence:
   - STLC/proofs/program mutations.
2. Controlled semantic graphs:
   - generated propositions and small stories.
3. Controlled natural-language stories:
   - same event graph, many surfaces, hard negatives.
4. Existing semantic resources:
   - AMR-style graph data,
   - NLI datasets,
   - paraphrase corpora,
   - syntax benchmarks,
   - code clone/equivalence datasets.
5. Ordinary language mixture:
   - keep auxiliary topology objectives alive while adding broad LM data.

### Evaluations

- same-graph retrieval,
- graph reconstruction from hidden/phase states,
- relation-flip sensitivity,
- paraphrase invariance,
- NLI consistency,
- syntax/proof transfer,
- phase/semantic causal patching,
- interpretability feature purity.

## Track 4: Making It Matter

### Strong Claim We Want To Earn

> A compact phase/relation channel gives transformers a measurable advantage in
> learning and exposing structural equivalence classes, especially under
> semantics-preserving variation and topology-breaking hard negatives.

### Paperable Results

1. Iso-parameter resonance beats standard on verified formal equivalence.
2. Resonance improves same-semantic-graph retrieval and hard-negative
   discrimination in controlled natural language.
3. Phase/resonance interventions causally control structural interpretation.
4. Phase/resonance features are easier to interpret than residual features under
   SAE/probing metrics.
5. Formal/topological pretraining transfers to natural-language structural
   tasks without merely improving generic PPL.

### Falsifiers

- Explicit graph encodings beat phase kernels everywhere.
- Phase embeddings collapse to token identity or near-constant bias.
- Phase patching has no causal effect.
- Iso-parameter standard models match or beat resonance on all structural tasks.
- Benefits vanish with scale or real text.

## Immediate Work Queue

1. Land resonance-native trace and intervention utilities.
2. Land a verified semantic-story generator.
3. Train/evaluate same-graph natural-language retrieval.
4. Add iso-parameter mode to proof and story topology evaluations.
5. Add phase initialization from known topology:
   - proof normal-form group,
   - AST path,
   - graph role,
   - Laplacian/spectral coordinates.
6. Add SAE/probe experiments on:
   - phase stream,
   - projected phase stream,
   - residual stream,
   - resonance rows.
7. Write a clean “structural equivalence learning” paper outline after the first
   story-graph runs.

## Current Findings

### 2026-04-26: Story Retrieval V0

The first same-story-graph retrieval harness is useful plumbing but not yet a
strong benchmark. A modest CPU run saturates quickly:

```text
standard:        same-graph retrieval 1.0000, margin 0.4282
resonance_full:  same-graph retrieval 1.0000, margin 0.4220
bias_only:       same-graph retrieval 1.0000, margin 0.4556
inert:           same-graph retrieval 1.0000, margin 0.4552
```

Interpretation: the task can be solved by lexical/entity overlap and shallow
surface cues. It should stay in the suite as a regression test, but it is not
evidence for or against the resonance hypothesis.

The intervention metrics are still informative. In the full model, phase
permutation/zeroing damages retrieval and margin, while resonance-weight scaling
has essentially no effect. Bias-only and inert are indistinguishable under these
defaults. That means the current additive resonance-bias path is too weak/flat
to support strong claims; future tests need either stronger phase structure,
larger phase variance, larger learned weights, or a different kernel.

### 2026-04-26: Harder Natural-Language Bridge

Added a temporal story-query harness:

```text
event facts + before edges + question -> answer yes/no
```

The label is computed from transitive closure over a rendered event graph. This
is a better bridge from synthetic topology to natural language because lexical
overlap alone should not solve held-out graph queries.

Medium CPU run:

```text
standard:                   answer acc 0.5495, margin 0.0627
resonance_full:             answer acc 0.5729, margin 0.0674
resonance_full_story_prior: answer acc 0.5208, margin 0.0189
bias_only:                  answer acc 0.5365, margin 0.0628
```

Strong-bias run (`phase_init_std=1.2`, `resonance_attn_weight=2.0`):

```text
standard:       answer acc 0.5495, margin 0.0627
resonance_full: answer acc 0.5234, margin 0.0351
bias_only:      answer acc 0.5312, margin 0.0624
```

Row-normalized default-bias run:

```text
standard:                  answer acc 0.5495, margin 0.0627
resonance_full_normalized: answer acc 0.5755, margin 0.0669
bias_only_normalized:      answer acc 0.5365, margin 0.0629
```

Interpretation: the task is hard enough to avoid immediate saturation. There is
a small default full-resonance accuracy lead, but it is not iso-param and does
not survive stronger random bias settings as an attention-bias effect. The
story-role prior is naive and hurts. Row-normalized full resonance preserves the
small lead, but normalized bias-only remains weak, so the current evidence still
points more toward the projected phase stream than the additive bias path.

Reproduction command:

```bash
.venv/bin/python resonance/story_query_eval.py \
  --output_dir resonance/outputs/story_query_eval_medium \
  --device cpu \
  --epochs 5 \
  --train_examples 256 \
  --val_examples 128 \
  --seeds 11 23 37 \
  --conditions standard,resonance_full,resonance_full_story_prior,bias_only \
  --interventions
```

### 2026-04-26: Kernel Sweep

The default attention-bias kernel is nearly inert:

```text
F=32, phase_init_std=0.3, weight=0.1
R std:          0.0223
effective rank: 1.7539
attention delta: 0.000027
```

Increasing phase variance and bias weight helps mechanically:

```text
F=32, phase_init_std=1.2, weight=1.0 -> attention delta 0.002221
F=32, phase_init_std=2.0, weight=2.0 -> attention delta 0.006857
```

Row-normalizing the resonance bias helps more directly:

```text
F=32, phase_init_std=0.3, weight=0.1, normalized -> attention delta 0.001138
F=32, phase_init_std=0.3, weight=1.0, normalized -> attention delta 0.016252
```

But the strong-bias temporal-query run got worse, so simply turning up the
kernel is not enough. The normalized-bias run did not make bias-only useful
either. The next serious architecture work should compare:

- row-normalized resonance kernels with stable variance and clipping;
- learned temperature/scale per layer;
- relation-aware baselines with explicit edge labels;
- phase initialization from verified graph coordinates, not token role buckets.

### Near-Term Benchmark Upgrades

- Add a no-lexical-overlap variant where same graph nodes are rendered with
  different aliases and the alias map is explicitly introduced.
- Add edge-mask and connective-mask evaluations to separate discourse-token
  memorization from graph inference.
- Add phase-kernel sweeps over `phase_init_std`, `resonance_attn_weight`, and
  `n_frequencies`; current defaults make the attention-bias contribution tiny.
- Add explicit graph baselines:
  - graph features concatenated to token embeddings,
  - shortest-path / Laplacian positional encodings,
  - relation-aware attention bias.

### 2026-04-26: Regime Probe Before Ablation

The ICLR 2026 representation-dispersion paper suggests a useful gate for this
project: before running architecture ablations, measure whether the task regime
is learnable, non-saturated, and geometrically active.

Added:

```bash
.venv/bin/python resonance/regime_probe.py \
  --output_dir resonance/outputs/regime_probe_medium \
  --device cpu \
  --epochs 5 \
  --train_examples 256 \
  --val_examples 128 \
  --seeds 11 23 37 \
  --conditions standard,resonance_full,resonance_full_normalized,bias_only_normalized
```

Current result:

```text
Verdict: weak: behavior improves but hidden geometry is not label-separated

standard:                  acc +0.0521 over majority, loss gain +0.0010, label gap -0.1716
resonance_full:            acc +0.0677 over majority, loss gain +0.0213, label gap -0.0849
resonance_full_normalized: acc +0.0677 over majority, loss gain +0.0213, label gap -0.0849
bias_only_normalized:      acc +0.0781 over majority, loss gain +0.0156, label gap -0.0983
```

Interpretation: this temporal-query regime is not yet a trustworthy ablation
target. It is slightly learnable, but the hidden states are not separating the
answer structure cleanly. The next objective is therefore regime discovery:

- vary data generators before varying architecture;
- require accuracy above majority, meaningful loss reduction, and positive
  geometry separation before claiming a benchmark is useful;
- test push-away and squeeze dispersion losses as a sensitivity check;
- only then run iso-parameter resonance-vs-baseline comparisons.

### 2026-04-26: Aliased Graph Regime Search

Added an aliased graph-walk generator and probe:

```bash
.venv/bin/python resonance/graph_alias_probe.py \
  --output_dir resonance/outputs/graph_alias_probe_shared_vocab_1step_8n_e3 \
  --device cpu \
  --epochs 3 \
  --train_examples 512 \
  --val_examples 256 \
  --n_graphs 32 \
  --n_nodes 8 \
  --alias_pool_size 96 \
  --out_degree 2 \
  --walk_length 4 \
  --query_steps 1 \
  --max_length 256 \
  --conditions standard,resonance_full,bias_only_normalized \
  --val_graphs new
```

The generator supports three validation splits:

- `same`: held-out walks over the same latent graphs and aliases.
- `same_aliases_new_edges`: same alias vocabulary/grouping, rewired edges.
- `new`: new latent graphs. With `alias_pool_size > 0`, train/validation draw
  aliases from the same token pool but use new alias groupings; this avoids the
  untrained-token failure of fully fresh random aliases.

Boundary results:

```text
known graph / held-out walks:      saturated at 1.0000 acc for every condition
same aliases / rewired edges:      saturated at 1.0000 acc for every condition
new aliases with no shared pool:   near chance or worse; token generalization bottleneck
shared vocab / new 16-node graph:  near chance; overfits training graph patterns
shared vocab / new 8-node graph:   first nontrivial band
```

Best current nontrivial band, single seed:

```text
shared vocab, new 8-node graphs, direct edge query, 3 epochs

standard:             acc 0.4883, acc-maj -0.0117, label gap -0.7234
resonance_full:       acc 0.6445, acc-maj +0.1445, label gap +0.9742
bias_only_normalized: acc 0.6602, acc-maj +0.1602, label gap +1.0954
```

Caveat: validation cross-entropy still worsens despite accuracy and geometry
improving. That means this is **not yet a passed regime**. It is, however, the
first place where resonance/bias variants separate behavior and hidden geometry
from standard without saturating.

Three-seed check with dispersion push-away (`dispersion_lambda=0.01`) does **not**
yet hold:

```text
standard:             acc 0.4805 +/- 0.0440, acc-maj -0.0195, label gap -0.7673
resonance_full:       acc 0.5638 +/- 0.1152, acc-maj +0.0638, label gap -0.2492
bias_only_normalized: acc 0.5911 +/- 0.0867, acc-maj +0.0911, label gap -0.2015
```

Per-epoch logging did find one seed where `resonance_full` briefly aligns
behavior and loss (`epoch 2: acc 0.6523, val loss 0.6640`), and another point
where behavior and geometry align (`epoch 3: acc 0.6953, label gap +0.3034`).
Those signals do not survive seeds. Treat them as a debugging target, not a
result.

Immediate next steps:

- rerun this band with at least 3 seeds before treating it as signal;
- log validation metrics per epoch and select by validation loss or calibrated
  accuracy, not final training epoch;
- add calibration metrics because current improvements are overconfident;
- sweep `alias_pool_size`, `n_nodes`, `query_steps`, and training graph count;
- compare against explicit relation-aware attention and graph-feature baselines.
