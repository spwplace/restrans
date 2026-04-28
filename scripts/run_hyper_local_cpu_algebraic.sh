#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export UV_CACHE_DIR=/tmp/uv-cache
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"

ROOT="resonance/outputs/hyper_cpu_algebraic_2026_04_28"
mkdir -p "$ROOT/logs"

CONDITIONS="standard,standard_iso,resonance_full_normalized,phase_stream_only_normalized,phase_dynamic_qk_film"

run_probe() {
  local name="$1"
  shift
  echo "===== $name ====="
  uv run python resonance/structural_task_probe.py \
    --output_dir "$ROOT/$name" \
    --device cpu \
    --conditions "$CONDITIONS" \
    --embed_dim 64 \
    --layers 2 \
    --heads 4 \
    --ff_dim 256 \
    --batch_size 32 \
    --eval_each_epoch \
    "$@" 2>&1 | tee "$ROOT/logs/$name.log"
}

run_probe cap_matching_depth6_cpu \
  --task cap_matching \
  --seeds 601 603 607 \
  --epochs 4 \
  --train_examples 2200 \
  --val_examples 700 \
  --depth 6 \
  --max_length 448

run_probe algebraic_protocol_depth4_cpu \
  --task algebraic_protocol \
  --seeds 601 603 607 \
  --epochs 4 \
  --train_examples 2200 \
  --val_examples 700 \
  --depth 4 \
  --max_length 640

uv run python resonance/build_lab_notebook.py \
  --root resonance/outputs \
  --output_dir resonance/outputs/lab_notebook_hyper_live || true
