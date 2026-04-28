# AWS / RunPod Handoff

Date: 2026-04-28

This repo is ready to hand to a colleague with AWS or RunPod credentials.  The
goal is not to prove the architecture from toy probes.  The goal is to run a
baseline-rich medium matrix, keep all artifacts, and promote only the task
families that survive performance, geometry, and causal-intervention checks.

## Required Access

Public data is enough for the next run.

Needed:

- Hugging Face access for public datasets and higher rate limits.
- AWS credentials if using S3 / EC2 / Batch.
- RunPod account if using Pods.
- Optional GitHub token for LeanDojo-v2 / LeanProgress setup.

Not needed yet:

- AMR 3.0 / LDC2020T02.  This is licensed LDC data and is explicitly deferred.
- CETBench.  Its dataset/code location still needs confirmation, and EquiBench
  already covers the immediate program-equivalence role.

## Fresh Machine Bootstrap

```bash
git clone <repo-url> restrans
cd restrans
uv sync
bash scripts/setup_external_deps.sh
PYTHONPATH=resonance uv run python resonance/tests/test_smoke.py
DRY_RUN=1 bash scripts/run_literature_medium_suite.sh
```

If `pytest` is unavailable, use the direct smoke test above.  It is the
repository-native check.

## Stage 1: Acquire Public Assets

Run this on CPU or a cheap mixed instance, not an expensive GPU:

```bash
bash scripts/acquire_research_assets.sh
```

Fuller cloud acquisition:

```bash
BABYLM_FULL=1 \
INCLUDE_BIG_CODE=1 \
INCLUDE_CODESEARCHNET=1 \
INCLUDE_LEANDOJO=1 \
bash scripts/acquire_research_assets.sh
```

Outputs:

- `data/dataset_manifest.json`
- `data/processed/probes`
- `data/processed/contrastive/equibench`

LeanProgress is isolated because it has a large proof-assistant dependency
stack:

```bash
export GITHUB_ACCESS_TOKEN=<token>
bash scripts/setup_leandojo_progress.sh
```

Treat LeanProgress as a CPU/storage workstream until a sample JSONL export is
confirmed.

## Stage 2: Medium Matrix Smoke

Run this first on one CUDA machine.  RunPod RTX 4090/5090 is fine; A100/H100 is
better if available.

```bash
DEVICE=cuda \
EPOCHS=4 \
TRAIN_EXAMPLES=1500 \
VAL_EXAMPLES=700 \
EMBED_DIM=128 \
LAYERS=4 \
HEADS=4 \
FF_DIM=512 \
BATCH_SIZE=24 \
bash scripts/run_literature_medium_suite.sh
```

This checks for runtime failures and task saturation.  It is not the final
experiment.

## Stage 3: Full 10M-ish Matrix

Use S3 or equivalent object storage as the artifact store.  Shard by task
family first, then condition, then seed.

Recommended all-condition run:

```bash
DEVICE=cuda \
ROOT=resonance/outputs/stage3_litmatrix_$(date -u +%Y_%m_%d_%H%M%S) \
SAVE_MODELS=1 \
RUN_INTERP=1 \
EPOCHS=8 \
TRAIN_EXAMPLES=5000 \
VAL_EXAMPLES=1500 \
EMBED_DIM=256 \
LAYERS=6 \
HEADS=8 \
FF_DIM=1024 \
BATCH_SIZE=32 \
bash scripts/run_literature_medium_suite.sh
```

If memory is tight, reduce `BATCH_SIZE` first.  If time is tight, keep
`SAVE_MODELS=1` but set `RUN_INTERP=0`, then run:

```bash
DEVICE=cuda bash scripts/run_interpretability_on_suite.sh <ROOT>
```

## First Launch Batch

Use separate jobs if possible:

```text
cpu-1: acquire/convert EquiBench, update manifest
cpu-2: LeanDojo-v2 install + sample LeanProgress export
gpu-1: cap_matching + unification + algebraic_protocol, all conditions, 3 seeds
gpu-2: listops + dyck, all conditions, 3 seeds
gpu-3: SLOG + COGS, all conditions, 3 seeds
gpu-4: CFQ + EquiBench + HANS + MSGS, all conditions, 3 seeds
```

After this, stop sweeping every variant everywhere.  Choose the top two task
families and top four conditions.

## What The Task Families Can And Cannot Tell Us

The small formal tasks are not proof of a scalable architecture.  They are
debuggers and microscopes.  They tell us whether the implementation can learn a
known structure, whether baselines already erase the advantage, and whether the
structural stream is causal.

The medium tasks are the actual evidence:

- SLOG / COGS / CFQ: language-side structural generalization.
- EquiBench: semantics-preserving / semantics-breaking program pairs.
- Cap matching / unification / algebraic protocol: controlled formal structure
  where shortcuts can be inspected.
- HANS / MSGS: shortcut diagnostics, not headline wins.

The result we want is selective:

1. Structural variants win only where structure matters.
2. The win survives `standard_iso`, `standard_alibi`, `standard_deberta_lite`,
   and `relational_stream_lite`.
3. Phase/structural states show cleaner structural geometry than residual
   states.
4. Phase ablation/shuffle/noise or resonance scaling selectively hurts
   structure-sensitive decisions.

If bigger, plainer baselines erase all gains, that is a useful negative result.
It means the next paper should be about task design, training pressure, and
structural-stream falsification rather than architecture novelty.

## Artifact Checklist

Every completed run should preserve:

- `results.json`
- `report.md`
- `logs/*.log`
- `checkpoints/*.pt` when `SAVE_MODELS=1`
- `interpretability_audit/results.json`
- `interpretability_audit/report.md`
- `lab_notebook/lab-notebook.md`

Recommended sync:

```bash
aws s3 sync resonance/outputs s3://<bucket>/restrans/outputs/
aws s3 sync data/dataset_manifest.json s3://<bucket>/restrans/manifests/
```

RunPod users should sync out before terminating pods.
