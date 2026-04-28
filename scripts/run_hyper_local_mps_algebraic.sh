#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export UV_CACHE_DIR=/tmp/uv-cache
export PYTHONUNBUFFERED=1

ROOT="resonance/outputs/hyper_mps_algebraic_2026_04_28"
mkdir -p "$ROOT/logs"

CONDITIONS="standard,standard_iso,resonance_full_normalized,phase_stream_only_normalized,phase_dynamic_attn,phase_qk_film,phase_dynamic_qk_film,complex_directional_normalized"

run_probe() {
  local name="$1"
  shift
  echo "===== $name ====="
  uv run python resonance/structural_task_probe.py \
    --output_dir "$ROOT/$name" \
    --device mps \
    --conditions "$CONDITIONS" \
    --embed_dim 96 \
    --layers 3 \
    --heads 4 \
    --ff_dim 384 \
    --batch_size 16 \
    --eval_each_epoch \
    "$@" 2>&1 | tee "$ROOT/logs/$name.log"
}

run_probe cap_matching_depth5_mps \
  --task cap_matching \
  --seeds 501 503 507 \
  --epochs 5 \
  --train_examples 1800 \
  --val_examples 600 \
  --depth 5 \
  --max_length 384

run_probe algebraic_protocol_depth3_mps \
  --task algebraic_protocol \
  --seeds 501 503 507 \
  --epochs 5 \
  --train_examples 1800 \
  --val_examples 600 \
  --depth 3 \
  --max_length 512

run_probe unification_depth5_mps \
  --task unification \
  --seeds 501 503 507 \
  --epochs 5 \
  --train_examples 1800 \
  --val_examples 600 \
  --depth 5 \
  --max_length 256

uv run python resonance/build_lab_notebook.py \
  --root resonance/outputs \
  --output_dir resonance/outputs/lab_notebook_hyper_live || true
