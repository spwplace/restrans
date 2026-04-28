#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export UV_CACHE_DIR=/tmp/uv-cache
export PYTHONUNBUFFERED=1

ROOT="resonance/outputs/hyper_mps_2026_04_28"
mkdir -p "$ROOT/logs"

CONDITIONS="standard,standard_iso,resonance_full_normalized,phase_stream_only_normalized,phase_dynamic_attn,phase_qk_film,phase_dynamic_qk_film"

run_blimp() {
  local name="$1"
  local path="$2"
  echo "===== $name ====="
  uv run python resonance/structural_task_probe.py \
    --task blimp \
    --data_path "$path" \
    --output_dir "$ROOT/$name" \
    --device mps \
    --conditions "$CONDITIONS" \
    --seeds 101 103 107 \
    --epochs 5 \
    --train_examples 150 \
    --val_examples 300 \
    --embed_dim 96 \
    --layers 3 \
    --heads 4 \
    --ff_dim 384 \
    --batch_size 16 \
    --max_length 224 \
    --eval_each_epoch 2>&1 | tee "$ROOT/logs/$name.log"
}

run_generic() {
  local name="$1"
  local train_path="$2"
  local val_path="$3"
  echo "===== $name ====="
  uv run python resonance/structural_task_probe.py \
    --task generic_probe \
    --data_path "$train_path" \
    --val_data_path "$val_path" \
    --output_dir "$ROOT/$name" \
    --device mps \
    --conditions "$CONDITIONS" \
    --seeds 101 103 107 \
    --epochs 5 \
    --train_examples 1200 \
    --val_examples 500 \
    --embed_dim 96 \
    --layers 3 \
    --heads 4 \
    --ff_dim 384 \
    --batch_size 16 \
    --max_length 640 \
    --eval_each_epoch 2>&1 | tee "$ROOT/logs/$name.log"
}

run_blimp blimp_complex_np_island_n150_mps data/processed/blimp/complex_NP_island.jsonl
run_blimp blimp_sentential_subject_island_n150_mps data/processed/blimp/sentential_subject_island.jsonl
run_blimp blimp_principle_a_c_command_n150_mps data/processed/blimp/principle_A_c_command.jsonl
run_blimp blimp_distractor_agreement_relative_clause_n150_mps data/processed/blimp/distractor_agreement_relative_clause.jsonl

run_generic cogs_gen_probe_mps \
  data/processed/probes/cogs/train.jsonl \
  data/processed/probes/cogs/gen.jsonl

uv run python resonance/build_lab_notebook.py \
  --root resonance/outputs \
  --output_dir resonance/outputs/lab_notebook || true
