#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

STAMP="${STAMP:-$(date +%Y_%m_%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-resonance/outputs/bigboy_vm_trace_nextop_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/bigboy_vm_trace_nextop_${STAMP}}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

PY="${PY:-uv run python}"
DEVICE="${DEVICE:-mps}"
CONDITIONS="${CONDITIONS:-standard_alibi,phase_dynamic_qk_film_alibi_normalized,complex_directional_normalized}"
SEEDS="${SEEDS:-903}"

$PY resonance/train_structural_posttrain.py \
  --output_dir "$OUT_ROOT/vm_trace_exact_rl_len12_to24" \
  --device "$DEVICE" \
  --conditions "$CONDITIONS" \
  --seeds $SEEDS \
  --sft_epochs 2 \
  --rl_epochs 3 \
  --rl_algorithm exact_rl \
  --reference after_sft \
  --curriculum_tasks vm_step \
  --curriculum_epochs 1 \
  --curriculum_train_examples 16384 \
  --curriculum_val_examples 1024 \
  --train_examples 16384 \
  --val_examples 4096 \
  --max_length 1024 \
  --vocab_size 12000 \
  --embed_dim 320 \
  --layers 8 \
  --heads 8 \
  --ff_dim 1280 \
  --batch_size 16 \
  --lr 2e-4 \
  --dropout 0.05 \
  --phase_contrastive_weight 0.05 \
  --skip_before_eval \
  --phase_ablation_eval \
  --cascade_eval \
  --eval_each_epoch \
  --task vm_trace \
  --depth 12 \
  --val_depth 24 \
  --modulus 29 \
  2>&1 | tee "$LOG_DIR/vm_trace_exact_rl_len12_to24.log"

echo "done: $OUT_ROOT" | tee -a "$LOG_DIR/driver.log"
