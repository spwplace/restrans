#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

STAMP="${STAMP:-$(date +%Y_%m_%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-resonance/outputs/formal_followup_nextop_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/formal_followup_nextop_${STAMP}}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

PY="${PY:-uv run python}"
DEVICE="${DEVICE:-mps}"
CONDITIONS="${CONDITIONS:-standard_alibi,phase_dynamic_qk_film_alibi_normalized,complex_directional_normalized}"
SEEDS="${SEEDS:-907}"

echo "formal follow-up nextop"
echo "out=$OUT_ROOT"
echo "conditions=$CONDITIONS"
echo "device=$DEVICE"

run_lambda_normal_form() {
  local log="$LOG_DIR/lambda_normal_form_depth6_to9.log"
  echo "=== lambda_normal_form depth6->9 sequence prediction ===" | tee -a "$LOG_DIR/driver.log"
  $PY resonance/train_sequence_prediction.py \
    --output_dir "$OUT_ROOT/lambda_normal_form_depth6_to9" \
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
    2>&1 | tee "$log"
}

run_lambda_trace_rl() {
  local log="$LOG_DIR/lambda_trace_exact_rl_depth6_to9.log"
  echo "=== lambda_trace depth6->9 exact-RL with lambda_beta_step curriculum ===" | tee -a "$LOG_DIR/driver.log"
  $PY resonance/train_structural_posttrain.py \
    --output_dir "$OUT_ROOT/lambda_trace_exact_rl_depth6_to9" \
    --device "$DEVICE" \
    --conditions "$CONDITIONS" \
    --seeds $SEEDS \
    --sft_epochs 2 \
    --rl_epochs 3 \
    --rl_algorithm exact_rl \
    --reference after_sft \
    --curriculum_tasks lambda_beta_step \
    --curriculum_epochs 1 \
    --curriculum_train_examples 8192 \
    --curriculum_val_examples 512 \
    --train_examples 8192 \
    --val_examples 2048 \
    --max_length 768 \
    --vocab_size 12000 \
    --embed_dim 320 \
    --layers 8 \
    --heads 8 \
    --ff_dim 1280 \
    --batch_size 16 \
    --lr 2e-4 \
    --dropout 0.05 \
    --n_frequencies 32 \
    --phase_contrastive_weight 0.05 \
    --skip_before_eval \
    --phase_ablation_eval \
    --cascade_eval \
    --eval_each_epoch \
    --task lambda_trace \
    --depth 6 \
    --val_depth 9 \
    --max_size 28 \
    --max_trace_steps 8 \
    2>&1 | tee "$log"
}

run_harder_vm_step() {
  local log="$LOG_DIR/vm_step_next_len24_to48.log"
  echo "=== vm_step_next len24->48 harder extrapolation ===" | tee -a "$LOG_DIR/driver.log"
  $PY resonance/train_sequence_prediction.py \
    --output_dir "$OUT_ROOT/vm_step_next_len24_to48" \
    --device "$DEVICE" \
    --conditions "$CONDITIONS" \
    --seeds $SEEDS \
    --epochs 5 \
    --train_examples 32768 \
    --val_examples 4096 \
    --max_length 256 \
    --vocab_size 12000 \
    --embed_dim 320 \
    --layers 8 \
    --heads 8 \
    --ff_dim 1280 \
    --batch_size 48 \
    --lr 2e-4 \
    --dropout 0.05 \
    --n_frequencies 32 \
    --skip_before_eval \
    --eval_each_epoch \
    --task vm_step_next \
    --depth 24 \
    --val_depth 48 \
    --modulus 29 \
    2>&1 | tee "$log"
}

run_lambda_normal_form
run_lambda_trace_rl
run_harder_vm_step

echo "done: $OUT_ROOT" | tee -a "$LOG_DIR/driver.log"
