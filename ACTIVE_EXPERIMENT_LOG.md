# Active Experiment Log

Updated: 2026-04-28T22:55:18Z

This file tracks the live execution state so the repo is handoff-readable even
while nextop and persvati are running.

## Current Research Read

- The original broad claim has narrowed into a testable one: an explicit
  compact structural stream may help when the task contains controlled
  relational invariants and the model is forced to use them.
- We should not treat pairwise attention bias, phase coordinates, or cosine
  differences as novel by themselves. The relevant comparisons are ALiBi,
  DeBERTa-style splits, relation-aware attention, and Abstractor/DAT-like
  relational streams.
- The most informative evidence now comes from three simultaneous checks:
  selective task gains, cleaner structural geometry, and causal/intervention
  sensitivity. Accuracy-only bumps are scouting signal.

## Completed Since Last Handoff

- Stopped the old nextop matrix during `listops_depth4`; ListOps was slow and
  not discriminating the structural claim.
- Preserved and summarized the completed nextop partial matrix:
  `resonance/outputs/architecture_matrix_nextop_2026_04_28_medium_mps`.
- Added a focused local launcher:
  `scripts/run_signal_matrix_nextop.sh`.
- Added literature-inspired variants:
  `harmonic_cosine_normalized`, `relation_value_mix_normalized`,
  `relation_value_qk_film_normalized`, and `harmonic_relation_value_normalized`.
- Added `resonance/plot_matrix_results.py` for per-run heatmaps, matrix CSVs,
  and compact reports.
- Extended smoke coverage for the harmonic kernel and relation-value routing.
- Added a pair-specific MLP relation kernel for future sweeps:
  `pair_mlp_kernel_normalized` and `pair_mlp_relation_value_normalized`.

## Persvati Second-Wave Results

Existing completed directory:
`resonance/outputs/hyper_second_wave_2026_04_28`.

Read:

- `cogs_gen_semantic_parse_gpu` is a candidate regime. It is learnable and
  non-saturated; structural variants were ahead of standard in this setup.
- `slog_gen_cogs_lf_gpu` is too weak as configured. It barely trains and should
  not be used as evidence until the task/data presentation is fixed.
- `poj104_problem_id_gpu` is not in a meaningful regime. Treat it as failed
  regime selection, not an architecture result.
- `bigclonebench_clone_probe_gpu` failed because `max_length=1024` was too
  short for examples. Rerun only with higher max length and small batch size.

## Active Runs

### nextop

Root:
`resonance/outputs/signal_matrix_nextop_2026_04_28_mps`

Command:

```bash
ROOT=resonance/outputs/signal_matrix_nextop_2026_04_28_mps \
DEVICE=mps SEEDS="941 943 947" EPOCHS=4 \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
bash scripts/run_signal_matrix_nextop.sh
```

Task set:

- `unification_depth5`
- `cap_matching_depth4`
- `algebraic_protocol_depth4`
- `graph_alias_new`
- `cogs_gen_semantic_parse`
- `equibench_oj_va`

Partial read after `unification_depth5` and `cap_matching_depth4`:

- Both tasks are learnable and non-saturated.
- `unification_depth5` favors `standard` and `standard_iso_deberta_lite`.
- `cap_matching_depth4` is closer: `phase_dynamic_qk_film_normalized`,
  `standard_deberta_lite`, and `standard` are effectively tied at the top.
- Full resonance, phase-only, harmonic cosine, and relation-value mix currently
  track each other closely. This is evidence against the original additive
  relation-bias mechanism as a standalone win in these regimes.

### persvati

Root:
`resonance/outputs/signal_matrix_persvati_2026_04_28_rocm_v2`

Command shape:

```bash
source .venv-exp/bin/activate
export UV_PROJECT_ENVIRONMENT=.venv-exp
export HSA_OVERRIDE_GFX_VERSION=11.0.0
ROOT=resonance/outputs/signal_matrix_persvati_2026_04_28_rocm_v2 \
DEVICE=cuda SEEDS="951 953" EPOCHS=5 \
TRAIN_EXAMPLES=2500 VAL_EXAMPLES=800 \
HARD_TRAIN_EXAMPLES=1800 HARD_VAL_EXAMPLES=600 \
COGS_TRAIN_EXAMPLES=3000 COGS_VAL_EXAMPLES=1000 \
CODE_TRAIN_EXAMPLES=500 CODE_VAL_EXAMPLES=160 \
EMBED_DIM=128 LAYERS=4 HEADS=4 FF_DIM=512 \
BATCH_SIZE=16 COGS_BATCH_SIZE=16 CODE_BATCH_SIZE=2 \
bash scripts/run_signal_matrix_nextop.sh
```

Note: persvati's ROCm torch is installed in `.venv-exp`. Plain `uv run` without
`UV_PROJECT_ENVIRONMENT=.venv-exp` does not see torch there.

## Post-Run Commands

For each completed root:

```bash
uv run python resonance/summarize_architecture_matrix.py \
  --root <root> \
  --output_json <root>/summary.json \
  --output_md <root>/summary.md

uv run python resonance/plot_matrix_results.py --root <root>
```

Then rebuild the global notebook:

```bash
uv run python resonance/build_lab_notebook.py \
  --root resonance/outputs \
  --output_dir resonance/outputs/lab_notebook_hyper_live
```

If persvati results are not local, sync them:

```bash
rsync -az -e "ssh -i ~/.ssh/id_aws" \
  persvati:restrans-exp/resonance/outputs/signal_matrix_persvati_2026_04_28_rocm_v2/ \
  resonance/outputs/signal_matrix_persvati_2026_04_28_rocm_v2/
```
