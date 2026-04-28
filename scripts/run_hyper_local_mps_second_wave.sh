#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export UV_CACHE_DIR=/tmp/uv-cache
export PYTHONUNBUFFERED=1

ROOT="resonance/outputs/hyper_mps_second_wave_2026_04_28"
mkdir -p "$ROOT/logs"

CONDITIONS="standard,standard_iso,resonance_full_normalized,phase_stream_only_normalized,phase_dynamic_attn,phase_qk_film,phase_dynamic_qk_film,complex_directional_normalized"

run_generic() {
  local name="$1"
  local train_path="$2"
  local val_path="$3"
  local max_len="$4"
  local train_n="$5"
  local val_n="$6"
  echo "===== $name ====="
  uv run python resonance/structural_task_probe.py \
    --task generic_probe \
    --data_path "$train_path" \
    --val_data_path "$val_path" \
    --output_dir "$ROOT/$name" \
    --device mps \
    --conditions "$CONDITIONS" \
    --seeds 401 403 407 \
    --epochs 5 \
    --train_examples "$train_n" \
    --val_examples "$val_n" \
    --embed_dim 96 \
    --layers 3 \
    --heads 4 \
    --ff_dim 384 \
    --batch_size 16 \
    --max_length "$max_len" \
    --eval_each_epoch 2>&1 | tee "$ROOT/logs/$name.log"
}

run_generic cogs_gen_probe_mps_repeat \
  data/processed/probes/cogs/train.jsonl \
  data/processed/probes/cogs/gen.jsonl \
  640 1200 500

run_generic slog_gen_cogs_lf_mps \
  data/processed/probes/slog/slog_cogs_lf_train.jsonl \
  data/processed/probes/slog/slog_gen_cogs_lf.jsonl \
  640 1200 500

run_generic hans_evaluation_mps \
  data/processed/probes/hans/train.jsonl \
  data/processed/probes/hans/evaluation.jsonl \
  384 1200 500

run_generic code_clone_poj104_mps \
  data/processed/probes/poj104/train.jsonl \
  data/processed/probes/poj104/validation.jsonl \
  768 1600 500

uv run python resonance/build_lab_notebook.py \
  --root resonance/outputs \
  --output_dir resonance/outputs/lab_notebook_hyper_live || true
