#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="${STAMP:-$(date +%Y_%m_%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-resonance/outputs/sequence_prediction_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/sequence_prediction_${STAMP}}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

DEVICE="${DEVICE:-mps}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
PY="${PY:-uv run python}"
CONDITIONS="${CONDITIONS:-standard_alibi,standard_deberta_lite,phase_dynamic_qk_film_alibi_normalized,complex_directional_normalized,harmonic_relation_value_normalized}"
SEEDS="${SEEDS:-831 832}"
EPOCHS="${EPOCHS:-8}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-4096}"
VAL_EXAMPLES="${VAL_EXAMPLES:-1024}"
EMBED_DIM="${EMBED_DIM:-192}"
LAYERS="${LAYERS:-4}"
HEADS="${HEADS:-4}"
FF_DIM="${FF_DIM:-768}"
BATCH_SIZE="${BATCH_SIZE:-32}"

run_task() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "=== sequence prediction $name ===" | tee -a "$LOG_DIR/driver.log"
  $PY resonance/train_sequence_prediction.py \
    --output_dir "$OUT_ROOT/$name" \
    --device "$DEVICE" \
    --conditions "$CONDITIONS" \
    --seeds $SEEDS \
    --epochs "$EPOCHS" \
    --train_examples "$TRAIN_EXAMPLES" \
    --val_examples "$VAL_EXAMPLES" \
    --max_length 896 \
    --vocab_size 8192 \
    --embed_dim "$EMBED_DIM" \
    --layers "$LAYERS" \
    --heads "$HEADS" \
    --ff_dim "$FF_DIM" \
    --batch_size "$BATCH_SIZE" \
    --eval_each_epoch \
    "$@" \
    2>&1 | tee "$log"
}

run_task "vm_step_next_len8_to12" --task vm_step_next --depth 8 --val_depth 12 --modulus 17
run_task "lambda_beta_next_depth5_to7" --task lambda_beta_next --depth 5 --val_depth 7 --max_size 20 --max_trace_steps 8
run_task "lambda_normal_form_depth5" --task lambda_normal_form --depth 5 --val_depth 5 --max_size 18 --max_trace_steps 8

echo "done: $OUT_ROOT" | tee -a "$LOG_DIR/driver.log"
