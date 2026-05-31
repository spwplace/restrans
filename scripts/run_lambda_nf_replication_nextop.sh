#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

STAMP="${STAMP:-$(date +%Y_%m_%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-resonance/outputs/lambda_nf_replication_nextop_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/lambda_nf_replication_nextop_${STAMP}}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

PY="${PY:-uv run python}"
DEVICE="${DEVICE:-mps}"
CONDITIONS="${CONDITIONS:-standard_alibi,phase_dynamic_qk_film_alibi_normalized,complex_directional_normalized,harmonic_relation_value_normalized}"
SEEDS="${SEEDS:-911 912 913}"

echo "lambda normal-form replication"
echo "out=$OUT_ROOT"
echo "conditions=$CONDITIONS"
echo "seeds=$SEEDS"

$PY resonance/train_sequence_prediction.py \
  --output_dir "$OUT_ROOT/lambda_normal_form_depth6_to9_multiseed" \
  --device "$DEVICE" \
  --conditions "$CONDITIONS" \
  --seeds $SEEDS \
  --epochs 5 \
  --train_examples 32768 \
  --val_examples 4096 \
  --max_length 384 \
  --vocab_size 12000 \
  --embed_dim 320 \
  --layers 8 \
  --heads 8 \
  --ff_dim 1280 \
  --batch_size 24 \
  --lr 2e-4 \
  --dropout 0.05 \
  --n_frequencies 32 \
  --skip_before_eval \
  --eval_each_epoch \
  --task lambda_normal_form \
  --depth 6 \
  --val_depth 9 \
  --max_size 28 \
  --max_trace_steps 8 \
  2>&1 | tee "$LOG_DIR/lambda_normal_form_depth6_to9_multiseed.log"

echo "done: $OUT_ROOT" | tee -a "$LOG_DIR/driver.log"
