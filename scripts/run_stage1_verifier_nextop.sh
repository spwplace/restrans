#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="${STAMP:-$(date +%Y_%m_%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-resonance/outputs/stage1_verifier_nextop_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/stage1_verifier_nextop_${STAMP}}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
PY="${PY:-uv run python}"
DEVICE="${DEVICE:-mps}"
CONDITIONS="${CONDITIONS:-standard_alibi,standard_deberta_lite,phase_dynamic_qk_film_alibi_normalized,complex_directional_normalized,harmonic_relation_value_normalized}"
SEEDS="${SEEDS:-861}"

run_sequence() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "=== sequence $name ===" | tee -a "$LOG_DIR/driver.log"
  $PY resonance/train_sequence_prediction.py \
    --output_dir "$OUT_ROOT/$name" \
    --device "$DEVICE" \
    --conditions "$CONDITIONS" \
    --seeds $SEEDS \
    --epochs 6 \
    --train_examples 2048 \
    --val_examples 768 \
    --max_length 768 \
    --vocab_size 8192 \
    --embed_dim 160 \
    --layers 4 \
    --heads 4 \
    --ff_dim 640 \
    --batch_size 32 \
    --dropout 0.05 \
    --eval_each_epoch \
    "$@" \
    2>&1 | tee "$log"
}

run_probe() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "=== probe $name ===" | tee -a "$LOG_DIR/driver.log"
  $PY resonance/structural_task_probe.py \
    --output_dir "$OUT_ROOT/$name" \
    --device "$DEVICE" \
    --conditions "$CONDITIONS" \
    --seeds $SEEDS \
    --epochs 5 \
    --train_examples 2048 \
    --val_examples 768 \
    --max_length 768 \
    --vocab_size 8192 \
    --embed_dim 160 \
    --layers 4 \
    --heads 4 \
    --ff_dim 640 \
    --batch_size 32 \
    --dropout 0.05 \
    --eval_each_epoch \
    "$@" \
    2>&1 | tee "$log"
}

run_sequence "vm_step_next_len8_to12" --task vm_step_next --depth 8 --val_depth 12 --modulus 17
run_probe "vm_trace_len8_to12" --task vm_trace --depth 8 --val_depth 12 --modulus 17
run_sequence "lambda_beta_next_depth4_to6" --task lambda_beta_next --depth 4 --val_depth 6 --max_size 14 --max_trace_steps 5

echo "done: $OUT_ROOT" | tee -a "$LOG_DIR/driver.log"
