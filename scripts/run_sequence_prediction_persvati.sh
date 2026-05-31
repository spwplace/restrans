#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

STAMP="${STAMP:-$(date +%Y_%m_%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-resonance/outputs/sequence_prediction_persvati_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/sequence_prediction_persvati_${STAMP}}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
PY="${PY:-.venv-exp/bin/python}"
CONDITIONS="${CONDITIONS:-standard_alibi,standard_deberta_lite,phase_dynamic_qk_film_alibi_normalized,complex_directional_normalized,harmonic_relation_value_normalized}"
SEEDS="${SEEDS:-841 842 843}"
EPOCHS="${EPOCHS:-10}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-12000}"
VAL_EXAMPLES="${VAL_EXAMPLES:-3000}"
EMBED_DIM="${EMBED_DIM:-256}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-8}"
FF_DIM="${FF_DIM:-1024}"
BATCH_SIZE="${BATCH_SIZE:-64}"

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
    --max_length 1024 \
    --vocab_size 12000 \
    --embed_dim "$EMBED_DIM" \
    --layers "$LAYERS" \
    --heads "$HEADS" \
    --ff_dim "$FF_DIM" \
    --batch_size "$BATCH_SIZE" \
    --eval_each_epoch \
    "$@" \
    2>&1 | tee "$log"
}

run_task "vm_step_next_len10_to14" --task vm_step_next --depth 10 --val_depth 14 --modulus 23
run_task "lambda_beta_next_depth6_to8" --task lambda_beta_next --depth 6 --val_depth 8 --max_size 24 --max_trace_steps 12
run_task "lambda_normal_form_depth6" --task lambda_normal_form --depth 6 --val_depth 6 --max_size 24 --max_trace_steps 12

echo "done: $OUT_ROOT" | tee -a "$LOG_DIR/driver.log"
