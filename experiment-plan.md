# Experiment Plan

Date: 2026-04-27

## Purpose

We are not trying to prove that resonance improves arbitrary language modeling.
The current question is narrower:

> Does a compact phase/relation channel help a transformer learn, expose, and
> causally use structural equivalence or structural validity?

The first experiment wave should find a regime where training is meaningful.
Only after a task passes that regime gate should we spend GPU time on full
iso-parameter ablations and interpretability work.

## Global Evaluation Gate

A task is ablation-ready only if it satisfies all three:

- validation accuracy beats majority by at least 5 points;
- validation loss improves during training;
- hidden-state label geometry separates, with positive `label_gap` /
  `between_minus_within`.

Additional filters:

- If every condition reaches 95%+ immediately, the task is saturated.
- If all conditions stay at chance, the task/generator/adapter is not ready.
- If resonance helps behavior but not loss, treat it as calibration/debugging
  signal rather than a result.
- If only `bias_only_normalized` wins, the useful object may be the relation
  bias rather than the phase stream. That is still interesting, but it changes
  the claim.

## Current Status Notes

Remote execution on `persvati` is working through `uv` with ROCm PyTorch
2.5.1. Use `--device cuda` because ROCm exposes the HIP device through the
PyTorch CUDA API, and set:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

The broad primary small sweep is too slow on the integrated AMD GPU. Focused
small probes are practical.

Results pulled into `resonance/outputs/experiment_plan` currently support a
narrow, cautious read:

- Linzen simple/dependency and the first two BLiMP agreement probes do not pass
  the regime gate in smoke runs.
- BLiMP `wh_island` is the most promising natural-language probe so far.
- With 700 training examples, `wh_island` saturates across all conditions, so it
  is not useful for architecture claims.
- With 50 examples, training is too weak.
- With 100 examples, the phase-bearing conditions beat standard/bias/inert on
  accuracy, but the 7-seed confirmation is only marginal on loss and label
  geometry:

| Run | Condition | Acc | Acc-Maj | Loss Gain | Label Gap |
|---|---|---:|---:|---:|---:|
| `blimp_wh_island_lowdata_100_confirm` | `standard` | 0.5819 | +0.0819 | -0.1568 | -0.2135 |
| `blimp_wh_island_lowdata_100_confirm` | `resonance_full_normalized` | 0.6429 | +0.1429 | +0.0066 | -0.0056 |
| `blimp_wh_island_lowdata_100_confirm` | `phase_stream_only_normalized` | 0.6438 | +0.1438 | +0.0100 | +0.0009 |
| `blimp_wh_island_lowdata_100_confirm` | `bias_only_normalized` | 0.6062 | +0.1062 | -0.1221 | -0.0087 |
| `blimp_wh_island_lowdata_100_confirm` | `resonance_inert_normalized` | 0.6048 | +0.1048 | -0.1222 | -0.0077 |

Interpretation: there is a real low-data accuracy hint for the phase stream,
but not yet evidence that the learned resonance bias is doing the work. Full
resonance and phase-stream-only are effectively tied; bias-only and inert are
also effectively tied. The next ablation target should therefore be phrased as
"phase path / additional structural channel under low data" until a task shows a
separate advantage for relation bias.

## Conditions

First pass:

- `standard`
- `resonance_full_normalized`
- `bias_only_normalized`

Second pass, only on tasks that pass the gate:

- `standard`
- `resonance_full_normalized`
- `phase_stream_only_normalized`
- `bias_only_normalized`
- `resonance_inert_normalized`
- explicit relation-aware baseline, once implemented

## First Executable Wave

These are runnable now through `resonance/run_experiment_plan.py`.

### 1. Linzen Simple Agreement

Path:

- `data/processed/rnn_agreement_simple/rnn_agr_simple/numpred.train`
- `data/processed/rnn_agreement_simple/rnn_agr_simple/numpred.val`

Hypothesis:

Subject-verb number across natural prefixes should reward a structural channel
that tracks agreement target/attractor roles rather than raw lexical
continuation.

Expected signal:

- resonance or bias-only improves hard-prefix accuracy;
- label geometry separates singular/plural contexts;
- no immediate saturation.

Risk:

The adapted prompt predicts number agreement rather than the exact verb token.
This is good for a structural probe, but it is not the original Linzen language
modeling objective.

### 2. Linzen Dependency TSV

Path:

- `data/processed/rnn_agreement/agr_50_mostcommon_10K.tsv`

Hypothesis:

The larger dependency TSV should tell us whether any agreement signal survives
broader prefix diversity and dependency metadata.

Expected signal:

- similar or stronger separation than simple agreement;
- sensitivity to attractors/distance in follow-up slicing.

Risk:

The current adapter uses a yes/no prompt over singular/plural candidate number.
We should later add metrics binned by `n_diff_intervening`, `distance`, and
`max_depth`.

### 3. BLiMP Agreement With Relative-Clause Distractors

Path:

- `data/processed/blimp/distractor_agreement_relative_clause.jsonl`

Hypothesis:

This is a direct structural syntax stress test. Resonance should help only if it
preserves dependency geometry across distractor nouns.

Expected signal:

- non-saturated grammaticality accuracy;
- resonance/bias gap over standard;
- positive label geometry.

### 4. BLiMP Anaphor Number Agreement

Path:

- `data/processed/blimp/anaphor_number_agreement.jsonl`

Hypothesis:

Anaphor agreement stresses binding-like structure. If the phase stream is a
useful structural path, it should separate binding-consistent and
binding-inconsistent contexts more cleanly than the residual stream.

Expected signal:

- less seed variance than standard;
- phase/resonance geometry with higher label purity.

### 5. BLiMP Wh-Island

Path:

- `data/processed/blimp/wh_island.jsonl`

Hypothesis:

Island constraints are harder and more abstract than agreement. A positive
signal here would be high value, but a negative result is not fatal.

Expected signal:

- any above-majority learnability is useful;
- if saturated or chance, treat as a data/task difficulty finding.

### 6. Aliased Graph Boundary

Synthetic task:

- `graph_alias`
- shared alias vocabulary
- new validation graphs
- 8 nodes, 32 train graphs, direct one-step query

Hypothesis:

Aliased graph walks control lexical identity while requiring latent graph-state
tracking. This is the current best synthetic boundary for the architecture.

Expected signal:

- resonance/bias conditions beat standard;
- positive label geometry;
- validation loss must improve for this to become a real ablation target.

Risk:

Prior runs had behavior/geometry blips that did not survive seeds. This is a
debugging probe until it passes the full gate.

### 7. Template Equivalence With Zero Lexical Overlap

Synthetic task:

- `template_equivalence`
- 80 templates
- depth 5

Hypothesis:

If the model can recognize latent template identity under surface substitution,
phase/resonance should help ignore lexical accidents.

Expected signal:

- useful as an indicator of structural invariance;
- not a headline result because it is synthetic and generator-dependent.

### 8. First-Order Term Unification

Synthetic task:

- `unification`
- depth 4

Hypothesis:

Unification stresses variable binding and tree alignment. Resonance should help
only if the phase path learns structural equivalence under renaming.

Expected signal:

- positive geometry is more important than raw accuracy;
- failed learning means generator/curriculum needs simplification.

### 9. Cross-Serial Dyck

Synthetic task:

- `dyck`
- cross mode
- depth 8
- 3 bracket types

Hypothesis:

Cross-serial matching tests nonlocal structural state. This is a debugging probe
for whether the current model/harness can express matching structure.

Expected signal:

- moderate non-saturated accuracy;
- do not treat a win here as natural-language evidence.

### 10. Causal Intervention Stories

Synthetic task:

- `causal_intervention`
- 8 events
- edge probability 0.30

Hypothesis:

Causal intervention questions should punish temporal-order heuristics and reward
edge-sensitive story topology.

Expected signal:

- if temporal-order heuristics dominate, redesign the generator;
- if phase/resonance separates connected vs disconnected interventions, promote
  this to the story-graph curriculum.

### 11. Story Graph Topology With Interventions

Harness:

- `resonance/story_topology_eval.py --interventions`

Hypothesis:

If resonance learns graph-level structure rather than lexical overlap, same-graph
retrieval and hard-negative separation should improve, and phase/resonance
interventions should be causal.

Expected signal:

- same-graph retrieval is not enough; hard negatives must separate;
- phase zero/permutation/noise should damage structural metrics more than
  lexical fluency;
- `resonance_inert_normalized` should not match full resonance.

Risk:

Earlier story retrieval saturated from lexical overlap. Hard negatives and
intervention effects matter more than raw retrieval.

## Execution Commands

Preflight dry run:

```bash
cd /Users/ember/dev/restrans
export PYTHONPATH=resonance
.venv/bin/python resonance/run_experiment_plan.py \
  --tier all \
  --size smoke \
  --device cuda \
  --dry_run
```

Recommended first GPU pass:

```bash
cd /Users/ember/dev/restrans
export PYTHONPATH=resonance
.venv/bin/python resonance/run_experiment_plan.py \
  --tier primary \
  --size ten_m \
  --device cuda \
  --continue_on_error
```

Then run synthetic indicators:

```bash
.venv/bin/python resonance/run_experiment_plan.py \
  --tier synthetic \
  --size small \
  --device cuda \
  --continue_on_error
```

Then run story topology:

```bash
.venv/bin/python resonance/run_experiment_plan.py \
  --tier story \
  --size small \
  --device cuda \
  --continue_on_error
```

If the GPU has enough memory, rerun the story tier at `--size ten_m`.

The runner writes command manifests under:

- `resonance/outputs/experiment_plan/*_manifest.json`

Each experiment writes:

- `results.json`
- `report.md`

## What To Hand Back

For each run, collect:

- `resonance/outputs/experiment_plan/**/results.json`
- `resonance/outputs/experiment_plan/**/report.md`
- any terminal logs if a run crashed
- GPU model / VRAM / CUDA version
- exact git commit or `git status --short`

Most useful summary table:

```text
task | condition | seed mean acc | acc-majority | val loss gain | label gap | verdict
```

## Second-Wave Preparation

The second wave means: use the datasets already staged in the lake but not part
of the first hand-written probe set.  The core plumbing now exists:

- `resonance/prepare_external_probe_data.py`
  - converts HANS, MSGS, SLOG, COGS, CFQ, POJ-104, BigCloneBench, and
    CodeSearchNet into generic yes/no probe JSONL;
  - writes to `data/processed/probes`;
  - currently uses `--limit 10000` per split/paradigm as a probe-staging cap.
- `resonance/structural_task_probe.py --task generic_probe`
  - consumes these converted probe JSONLs.
- `resonance/run_experiment_plan.py --tier second_wave`
  - runs the selected second-wave probes.
- `resonance/summarize_experiment_plan.py`
  - walks output directories and emits `summary.csv` / `summary.md`.

Prepared probe outputs:

- HANS entailment probes:
  - `data/processed/probes/hans/train.jsonl`
  - `data/processed/probes/hans/evaluation.jsonl`
- MSGS structural-feature probes:
  - `data/processed/probes/msgs/...`
- SLOG semantic-form probes:
  - `data/processed/probes/slog/...`
- COGS semantic-form probes:
  - `data/processed/probes/cogs/...`
- CFQ query probes:
  - `data/processed/probes/cfq/mcd1`
  - `data/processed/probes/cfq/mcd2`
  - `data/processed/probes/cfq/mcd3`
- Program/code probes:
  - `data/processed/probes/poj104`
  - `data/processed/probes/bigclonebench`
  - `data/processed/probes/codesearchnet`

Second-wave command:

```bash
cd /Users/ember/dev/restrans
export PYTHONPATH=resonance
.venv/bin/python resonance/prepare_external_probe_data.py \
  --data_dir data \
  --output_dir data/processed/probes \
  --limit 10000

.venv/bin/python resonance/run_experiment_plan.py \
  --tier second_wave \
  --size small \
  --device cuda \
  --continue_on_error

.venv/bin/python resonance/summarize_experiment_plan.py \
  --root resonance/outputs/experiment_plan
```

Second-wave hypotheses:

- HANS: tests resistance to lexical-overlap, subsequence, and constituent NLI
  heuristics.
- MSGS: tests linguistic-feature generalization against surface shortcuts.
- SLOG/COGS/CFQ: tests semantic-form/query matching under compositional split
  pressure.
- POJ-104: tests program-to-problem structural classification.
- BigCloneBench: tests program semantic-clone recognition.

Still not implemented:

1. Intrinsic interpretability runner
   - Add sparse structural bottleneck variants.
   - Run matched standard sparse-bottleneck controls.
   - Fit SAE, low-rank, and archetype/hull probes.
   - Run phase/resonance patching on tasks that pass the gate.

2. Full ablation runner for passed tasks
   - Re-run only promoted tasks with full condition set.
   - Use 3+ seeds.
   - Save models for interpretability.

## Decision Rule After Kimi Runs This

- If no first-wave task passes the gate, do not run ablations. Improve adapters
  and generators.
- If Linzen/BLiMP passes, prioritize syntax and MSGS/HANS converters.
- If graph/story tasks pass, prioritize semantic graph curriculum and
  intervention tooling.
- If only code/program tasks later pass, pivot toward program equivalence and
  formal-methods data first, then natural language transfer.
- If resonance never beats `bias_only_normalized`, rewrite the claim around
  relation-aware structural bias rather than the full phase-stream story.
