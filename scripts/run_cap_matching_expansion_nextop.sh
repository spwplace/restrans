#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="${STAMP:-$(date +%Y_%m_%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-resonance/outputs/cap_matching_expansion_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/cap_matching_expansion_${STAMP}}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

DEVICE="${DEVICE:-mps}"
PY="${PY:-uv run python}"

CONDITIONS="${CONDITIONS:-standard_alibi,standard_deberta_lite,phase_dynamic_qk_film_alibi_normalized,relation_value_qk_film_alibi_normalized,resonance_full_normalized,complex_directional_normalized,harmonic_relation_value_normalized}"
SEEDS="${SEEDS:-701 702 703}"
EPOCHS="${EPOCHS:-8}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-4096}"
VAL_EXAMPLES="${VAL_EXAMPLES:-2048}"
EMBED_DIM="${EMBED_DIM:-192}"
LAYERS="${LAYERS:-4}"
HEADS="${HEADS:-4}"
FF_DIM="${FF_DIM:-768}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"

echo "cap_matching expansion"
echo "out=$OUT_ROOT"
echo "conditions=$CONDITIONS"
echo "seeds=$SEEDS"

run_probe() {
  local name="$1"
  local depth="$2"
  local val_depth="$3"
  local log="$LOG_DIR/${name}.log"
  echo "=== $name depth=$depth val_depth=$val_depth ===" | tee -a "$LOG_DIR/driver.log"
  $PY resonance/structural_task_probe.py \
    --task cap_matching \
    --output_dir "$OUT_ROOT/$name" \
    --device "$DEVICE" \
    --conditions "$CONDITIONS" \
    --seeds $SEEDS \
    --epochs "$EPOCHS" \
    --train_examples "$TRAIN_EXAMPLES" \
    --val_examples "$VAL_EXAMPLES" \
    --depth "$depth" \
    --val_depth "$val_depth" \
    --max_length 512 \
    --vocab_size 8192 \
    --embed_dim "$EMBED_DIM" \
    --layers "$LAYERS" \
    --heads "$HEADS" \
    --ff_dim "$FF_DIM" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --dropout 0.05 \
    --eval_each_epoch \
    2>&1 | tee "$log"
}

# Same-depth learnability curve.
run_probe "depth4_same" 4 4
run_probe "depth5_same" 5 5
run_probe "depth6_same" 6 6
run_probe "depth7_same" 7 7

# Out-of-depth generalization: train shallow, validate deeper.
run_probe "depth4_to6" 4 6
run_probe "depth5_to7" 5 7

echo "done: $OUT_ROOT" | tee -a "$LOG_DIR/driver.log"
