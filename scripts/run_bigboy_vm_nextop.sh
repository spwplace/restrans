#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

STAMP="${STAMP:-$(date +%Y_%m_%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-resonance/outputs/bigboy_vm_nextop_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/bigboy_vm_nextop_${STAMP}}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

PY="${PY:-uv run python}"
DEVICE="${DEVICE:-mps}"
CONDITIONS="${CONDITIONS:-standard_alibi,phase_dynamic_qk_film_alibi_normalized,complex_directional_normalized}"
SEEDS="${SEEDS:-902}"

# Smaller companion run for nextop. Persvati is the main big model.
VOCAB_SIZE="${VOCAB_SIZE:-12000}"
EMBED_DIM="${EMBED_DIM:-320}"
LAYERS="${LAYERS:-8}"
HEADS="${HEADS:-8}"
FF_DIM="${FF_DIM:-1280}"
BATCH_SIZE="${BATCH_SIZE:-48}"
LR="${LR:-2e-4}"
DROPOUT="${DROPOUT:-0.05}"

echo "nextop VM architecture companion"
echo "out=$OUT_ROOT"

$PY resonance/train_sequence_prediction.py \
  --output_dir "$OUT_ROOT/vm_step_next_len12_to24" \
  --device "$DEVICE" \
  --conditions "$CONDITIONS" \
  --seeds $SEEDS \
  --epochs 6 \
  --train_examples 32768 \
  --val_examples 4096 \
  --max_length 256 \
  --vocab_size "$VOCAB_SIZE" \
  --embed_dim "$EMBED_DIM" \
  --layers "$LAYERS" \
  --heads "$HEADS" \
  --ff_dim "$FF_DIM" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --dropout "$DROPOUT" \
  --skip_before_eval \
  --eval_each_epoch \
  --task vm_step_next \
  --depth 12 \
  --val_depth 24 \
  --modulus 29 \
  2>&1 | tee "$LOG_DIR/vm_step_next_len12_to24.log"

echo "done: $OUT_ROOT" | tee -a "$LOG_DIR/driver.log"
