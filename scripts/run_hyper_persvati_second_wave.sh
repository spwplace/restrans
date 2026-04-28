#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/restrans-exp"
. .venv-exp/bin/activate
export PYTHONPATH=resonance
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

ROOT="resonance/outputs/hyper_second_wave_2026_04_28"
mkdir -p "$ROOT/logs"

echo "Waiting for first-wave GPU structural sweep to finish..."
while pgrep -af 'structural_task_probe.py .*hyper_2026_04_28' >/dev/null; do
  date -u '+%Y-%m-%dT%H:%M:%SZ first-wave GPU still active'
  sleep 120
done

CONDITIONS="standard,standard_iso,resonance_full_normalized,phase_stream_only_normalized,phase_dynamic_qk_film,complex_directional_normalized"

run_generic() {
  local name="$1"
  local train_path="$2"
  local val_path="$3"
  local max_len="$4"
  local train_n="$5"
  local val_n="$6"
  echo "===== $name ====="
  python resonance/structural_task_probe.py \
    --task generic_probe \
    --data_path "$train_path" \
    --val_data_path "$val_path" \
    --output_dir "$ROOT/$name" \
    --device cuda \
    --conditions "$CONDITIONS" \
    --seeds 301 303 307 \
    --epochs 5 \
    --train_examples "$train_n" \
    --val_examples "$val_n" \
    --embed_dim 128 \
    --layers 4 \
    --heads 4 \
    --ff_dim 512 \
    --batch_size 16 \
    --max_length "$max_len" \
    --eval_each_epoch 2>&1 | tee "$ROOT/logs/$name.log"
}

run_generic cogs_gen_semantic_parse_gpu \
  data/processed/probes/cogs/train.jsonl \
  data/processed/probes/cogs/gen.jsonl \
  768 2500 1000

run_generic slog_gen_cogs_lf_gpu \
  data/processed/probes/slog/slog_cogs_lf_train.jsonl \
  data/processed/probes/slog/slog_gen_cogs_lf.jsonl \
  768 2500 1000

run_generic poj104_problem_id_gpu \
  data/processed/probes/poj104/train.jsonl \
  data/processed/probes/poj104/validation.jsonl \
  1024 3000 1000

run_generic bigclonebench_clone_probe_gpu \
  data/processed/probes/bigclonebench/train.jsonl \
  data/processed/probes/bigclonebench/validation.jsonl \
  1024 3000 1000

echo "===== babylm_variant_lm_gpu ====="
python resonance/train_corpus_lm.py \
  --train_path data/processed/babylm_strict_small/train.jsonl \
  --val_path data/processed/babylm_strict_small/validation.jsonl \
  --output_dir "$ROOT/babylm_variant_lm_gpu" \
  --device cuda \
  --conditions "standard,resonance_full_normalized,phase_stream_only_normalized,phase_dynamic_qk_film,complex_directional_normalized" \
  --train_examples 30000 \
  --val_examples 3000 \
  --train_docs 300000 \
  --val_docs 30000 \
  --vocab_size 16000 \
  --seq_len 128 \
  --epochs 4 \
  --embed_dim 256 \
  --layers 5 \
  --heads 4 \
  --ff_dim 1024 \
  --batch_size 24 \
  --gradient_accumulation 2 \
  --log_interval 150 \
  2>&1 | tee "$ROOT/logs/babylm_variant_lm_gpu.log"

python resonance/build_lab_notebook.py \
  --root resonance/outputs \
  --output_dir resonance/outputs/lab_notebook_second_wave || true
