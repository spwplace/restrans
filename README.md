# Resonance Transformer Research Harness

This repo contains the current experimental harness for evaluating phase /
structural channels in small transformer models.

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
