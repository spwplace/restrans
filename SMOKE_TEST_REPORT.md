# Smoke Test Report

Date: 2026-04-28
Machine: nextop

Command:

```bash
ROOT=/private/tmp/restrans_entrypoint_smoke_20260428_212439_rerun \
  bash scripts/smoke_all_entrypoints.sh
```

Result:

- Pass: `59`
- Fail: `0`
- Report root: `/private/tmp/restrans_entrypoint_smoke_20260428_212439_rerun`

Coverage:

- `bash -n` syntax check for every `scripts/*.sh` entrypoint.
- `--help` or dry-run check for handoff shell scripts.
- Recursive `py_compile` over `resonance/**/*.py`.
- `--help` check for every top-level argparse Python entrypoint.
- Direct repository smoke test: `resonance/tests/test_smoke.py`.
- Tiny structural-task training with checkpoint saving.
- Interpretability audit over the saved tiny checkpoints.
- Tiny resonance-kernel sweep.
- Tiny EquiBench generic-probe training path.
- Lab notebook build over smoke outputs.
- Example scripts:
  - `example_layer_config.py`
  - `example_modern_preset.py`
  - `example_sae.py`

Notes:

- Persvati was intentionally left untouched because it may still be running
  prior experiments.
- Remote and overnight scripts were syntax-checked only; running them would SSH
  into persvati or launch long jobs.
- Handoff-critical local paths received real tiny execution checks.
