#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export PYTHONUNBUFFERED=1

DEVICE="${DEVICE:-cpu}"
ROOT="${ROOT:-resonance/outputs/unified_$(date -u +%Y_%m_%d_%H%M%S)}"
mkdir -p "$ROOT"

SEEDS="${SEEDS:-123 456}"
EPOCHS="${EPOCHS:-3}"
CONDITIONS="${CONDITIONS:-standard_alibi,phase_dynamic_qk_film_normalized}"

for seed in $SEEDS; do
  for condition in $(echo "$CONDITIONS" | tr ',' ' '); do
    outdir="$ROOT/${condition}_seed${seed}"
    echo "===== $condition seed=$seed ====="
    uv run python resonance/train_unified.py \
      --output_dir "$outdir" \
      --device "$DEVICE" \
      --condition "$condition" \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --embed_dim "${EMBED_DIM:-192}" \
      --layers "${LAYERS:-4}" \
      --heads "${HEADS:-4}" \
      --ff_dim "${FF_DIM:-768}" \
      --seq_len "${SEQ_LEN:-128}" \
      --batch_size "${BATCH_SIZE:-32}" \
      --vocab_size "${VOCAB_SIZE:-12000}" \
      --lm_max_examples "${LM_MAX_EXAMPLES:-15000}" \
      --lm_val_max_examples "${LM_VAL_MAX_EXAMPLES:-2000}" \
      --task_train_examples "${TASK_TRAIN_EXAMPLES:-2000}" \
      --task_val_examples "${TASK_VAL_EXAMPLES:-500}" \
      --lm_weight "${LM_WEIGHT:-1.0}" \
      --task_weight "${TASK_WEIGHT:-1.0}" \
      2>&1 | tee "$outdir/run.log"
  done
done

echo "Wrote results to $ROOT"
