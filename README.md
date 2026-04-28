# Resonance Transformer Research Harness

This repo contains the current experimental harness for evaluating phase /
structural channels in small transformer models.

The current literature-grounded execution policy is in
[`research/literature_integration.md`](research/literature_integration.md).
The full report-to-workstream tracker is in
[`research/literature_backlog.md`](research/literature_backlog.md).
In short: do not claim novelty for pairwise relation bias or a phase stream by
itself; compare against existing relation-aware and dual-stream baselines, and
judge the project by controlled structural tasks plus causal interpretability.

## Fresh Setup

```bash
git clone <this-repo>
cd restrans
uv sync
bash scripts/setup_external_deps.sh
```

Large datasets are not committed. See `dataset-lake.md` for staged datasets and
download notes. The external cap-matching reference implementation is cloned to
`external/reu_unif` by `scripts/setup_external_deps.sh`; the training probes do
not depend on a machine-local `~/dev/reu_unif` path.

For cloud handoff, use [`AWS_RUNPOD_HANDOFF.md`](AWS_RUNPOD_HANDOFF.md).  It
spells out acquisition, staging, GPU job shape, artifact requirements, and the
criteria for treating a result as real rather than a toy-probe artifact.

## Public Asset Acquisition

```bash
bash scripts/acquire_research_assets.sh
```

This stages public data and converts it to generic probes.  It includes
EquiBench by default.  It does not fetch licensed AMR/LDC data.  LeanDojo-v2 is
handled separately because it has a large proof-assistant dependency stack:

```bash
export GITHUB_ACCESS_TOKEN=<token>
bash scripts/setup_leandojo_progress.sh
```

## Structural Probe Smoke Tests

```bash
export PYTHONPATH=resonance
uv run python resonance/structural_task_probe.py \
  --task cap_matching \
  --output_dir resonance/outputs/smoke_cap_matching \
  --conditions standard,phase_stream_only_normalized \
  --seeds 1 \
  --epochs 1 \
  --train_examples 96 \
  --val_examples 96 \
  --device cpu
```

Other self-contained tasks include `unification`, `algebraic_protocol`,
`dyck`, `graph_alias`, `template_equivalence`, `causal_intervention`,
`structural_paraphrase`, and `temporal_query`.

## Literature-Grounded Medium Suite

```bash
DEVICE=mps bash scripts/run_literature_medium_suite.sh
```

The suite includes standard, iso-param, ALiBi, DeBERTa-lite,
phase-only/contrastive, legacy resonance, dynamic phase, directional kernel,
and relational-stream-lite conditions. The config summary is
`configs/literature_medium_suite.json`.

Set `SAVE_MODELS=1` to preserve checkpoints for interpretability audits, and
`RUN_INTERP=1` to run the audit automatically at the end:

```bash
DEVICE=cuda SAVE_MODELS=1 RUN_INTERP=1 bash scripts/run_literature_medium_suite.sh
```

## Active Experiment Scripts

- `scripts/run_hyper_local_mps_algebraic.sh`
- `scripts/run_hyper_local_cpu_algebraic.sh`
- `scripts/run_hyper_persvati.sh`
- `scripts/run_hyper_persvati_cpu_remainder.sh`
- `scripts/run_hyper_persvati_second_wave.sh`
- `scripts/watch_hyper_results.sh`

After runs complete:

```bash
export PYTHONPATH=resonance
uv run python resonance/build_lab_notebook.py \
  --root resonance/outputs \
  --output_dir resonance/outputs/lab_notebook_hyper_live
```
