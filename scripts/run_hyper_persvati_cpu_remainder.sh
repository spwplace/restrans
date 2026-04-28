#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/restrans-exp"
. .venv-exp/bin/activate
export PYTHONPATH=resonance
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

ROOT="resonance/outputs/hyper_cpu_remainder_2026_04_28"
mkdir -p "$ROOT/logs"

CONDITIONS="standard,standard_iso,resonance_full_normalized,phase_stream_only_normalized,phase_dynamic_qk_film"

run_probe() {
  local name="$1"
  shift
  echo "===== $name ====="
  python resonance/structural_task_probe.py \
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

run_probe unification_depth5_cpu_fixed \
  --task unification \
  --seeds 201 203 207 \
  --epochs 4 \
  --train_examples 1500 \
  --val_examples 500 \
  --depth 5 \
  --max_length 256

run_probe cap_matching_depth5_cpu \
  --task cap_matching \
  --seeds 201 203 207 \
  --epochs 4 \
  --train_examples 1500 \
  --val_examples 500 \
  --depth 5 \
  --max_length 384

run_probe algebraic_protocol_depth4_cpu \
  --task algebraic_protocol \
  --seeds 201 203 207 \
  --epochs 4 \
  --train_examples 1500 \
  --val_examples 500 \
  --depth 4 \
  --max_length 640

run_probe dyck_cross_depth8_cpu \
  --task dyck \
  --seeds 201 203 207 \
  --epochs 4 \
  --train_examples 1500 \
  --val_examples 500 \
  --depth 8 \
  --n_types 3 \
  --dyck_mode cross \
  --max_length 256

run_probe slog_gen_cogs_lf_cpu \
  --task generic_probe \
  --data_path data/processed/probes/slog/slog_cogs_lf_train.jsonl \
  --val_data_path data/processed/probes/slog/slog_gen_cogs_lf.jsonl \
  --seeds 201 203 207 \
  --epochs 4 \
  --train_examples 1500 \
  --val_examples 500 \
  --max_length 640

python resonance/build_lab_notebook.py \
  --root resonance/outputs \
  --output_dir resonance/outputs/lab_notebook_hyper_cpu_remainder || true
