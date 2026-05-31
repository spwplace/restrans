#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

STAMP="${STAMP:-$(date +%Y_%m_%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-resonance/outputs/bigboy_vm_persvati_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/bigboy_vm_persvati_${STAMP}}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

PY="${PY:-.venv-exp/bin/python}"
DEVICE="${DEVICE:-cuda}"

# Keep the comparator set small enough that each condition gets real training.
CONDITIONS="${CONDITIONS:-standard_alibi,phase_dynamic_qk_film_alibi_normalized,complex_directional_normalized}"
SEEDS="${SEEDS:-901}"

# About 40-55M parameters depending on condition with vocab=16000.
VOCAB_SIZE="${VOCAB_SIZE:-16000}"
EMBED_DIM="${EMBED_DIM:-512}"
LAYERS="${LAYERS:-12}"
HEADS="${HEADS:-8}"
FF_DIM="${FF_DIM:-2048}"
N_FREQ="${N_FREQ:-32}"

# Persvati's ROCm stack is usable but memory-fragile; keep batch modest.
BATCH_SIZE="${BATCH_SIZE:-48}"
LR="${LR:-2e-4}"
DROPOUT="${DROPOUT:-0.05}"

echo "bigboy VM architecture test"
echo "out=$OUT_ROOT"
echo "conditions=$CONDITIONS"
echo "model: vocab=$VOCAB_SIZE d=$EMBED_DIM layers=$LAYERS heads=$HEADS ff=$FF_DIM"

run_sequence() {
  local log="$LOG_DIR/vm_step_next_len12_to24.log"
  echo "=== sequence vm_step_next len12->24 ===" | tee -a "$LOG_DIR/driver.log"
  $PY resonance/train_sequence_prediction.py \
    --output_dir "$OUT_ROOT/vm_step_next_len12_to24" \
    --device "$DEVICE" \
    --conditions "$CONDITIONS" \
    --seeds $SEEDS \
    --epochs 8 \
    --train_examples 65536 \
    --val_examples 8192 \
    --max_length 256 \
    --vocab_size "$VOCAB_SIZE" \
    --embed_dim "$EMBED_DIM" \
    --layers "$LAYERS" \
    --heads "$HEADS" \
    --ff_dim "$FF_DIM" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --dropout "$DROPOUT" \
    --n_frequencies "$N_FREQ" \
    --skip_before_eval \
    --eval_each_epoch \
    --task vm_step_next \
    --depth 12 \
    --val_depth 24 \
    --modulus 29 \
    2>&1 | tee "$log"
}

run_trace_rl() {
  local log="$LOG_DIR/vm_trace_exact_rl_len12_to24.log"
  echo "=== exact-RL vm_trace len12->24 with vm_step curriculum ===" | tee -a "$LOG_DIR/driver.log"
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
    --curriculum_train_examples 32768 \
    --curriculum_val_examples 2048 \
    --train_examples 32768 \
    --val_examples 8192 \
    --max_length 1024 \
    --vocab_size "$VOCAB_SIZE" \
    --embed_dim "$EMBED_DIM" \
    --layers "$LAYERS" \
    --heads "$HEADS" \
    --ff_dim "$FF_DIM" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --dropout "$DROPOUT" \
    --n_frequencies "$N_FREQ" \
    --phase_contrastive_weight 0.05 \
    --skip_before_eval \
    --phase_ablation_eval \
    --cascade_eval \
    --eval_each_epoch \
    --task vm_trace \
    --depth 12 \
    --val_depth 24 \
    --modulus 29 \
    2>&1 | tee "$log"
}

run_sequence
run_trace_rl

echo "done: $OUT_ROOT" | tee -a "$LOG_DIR/driver.log"
