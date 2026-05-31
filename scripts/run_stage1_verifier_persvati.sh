#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

STAMP="${STAMP:-$(date +%Y_%m_%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-resonance/outputs/stage1_verifier_persvati_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/stage1_verifier_persvati_${STAMP}}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

PY="${PY:-.venv-exp/bin/python}"
DEVICE="${DEVICE:-cuda}"
CONDITIONS_FAST="${CONDITIONS_FAST:-standard_alibi,standard_deberta_lite,phase_dynamic_qk_film_alibi_normalized,complex_directional_normalized,harmonic_relation_value_normalized}"
CONDITIONS_RL="${CONDITIONS_RL:-standard_alibi,phase_dynamic_qk_film_alibi_normalized,complex_directional_normalized}"
SEEDS="${SEEDS:-851 852}"
PROBE_EPOCHS="${PROBE_EPOCHS:-6}"
SEQ_EPOCHS="${SEQ_EPOCHS:-8}"
RL_EPOCHS="${RL_EPOCHS:-3}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-4096}"
VAL_EXAMPLES="${VAL_EXAMPLES:-1536}"
EMBED_DIM="${EMBED_DIM:-224}"
LAYERS="${LAYERS:-5}"
HEADS="${HEADS:-8}"
FF_DIM="${FF_DIM:-896}"
BATCH_SIZE="${BATCH_SIZE:-48}"

run_probe() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "=== probe $name ===" | tee -a "$LOG_DIR/driver.log"
  $PY resonance/structural_task_probe.py \
    --output_dir "$OUT_ROOT/$name" \
    --device "$DEVICE" \
    --conditions "$CONDITIONS_FAST" \
    --seeds $SEEDS \
    --epochs "$PROBE_EPOCHS" \
    --train_examples "$TRAIN_EXAMPLES" \
    --val_examples "$VAL_EXAMPLES" \
    --max_length 768 \
    --vocab_size 10000 \
    --embed_dim "$EMBED_DIM" \
    --layers "$LAYERS" \
    --heads "$HEADS" \
    --ff_dim "$FF_DIM" \
    --batch_size "$BATCH_SIZE" \
    --dropout 0.05 \
    --eval_each_epoch \
    "$@" \
    2>&1 | tee "$log"
}

run_posttrain() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "=== posttrain $name ===" | tee -a "$LOG_DIR/driver.log"
  $PY resonance/train_structural_posttrain.py \
    --output_dir "$OUT_ROOT/$name" \
    --device "$DEVICE" \
    --conditions "$CONDITIONS_RL" \
    --seeds $SEEDS \
    --sft_epochs 2 \
    --rl_epochs "$RL_EPOCHS" \
    --rl_algorithm exact_rl \
    --reference after_sft \
    --train_examples "$TRAIN_EXAMPLES" \
    --val_examples "$VAL_EXAMPLES" \
    --max_length 768 \
    --vocab_size 10000 \
    --embed_dim "$EMBED_DIM" \
    --layers "$LAYERS" \
    --heads "$HEADS" \
    --ff_dim "$FF_DIM" \
    --batch_size "$BATCH_SIZE" \
    --dropout 0.05 \
    --phase_contrastive_weight 0.05 \
    --phase_ablation_eval \
    --cascade_eval \
    --eval_each_epoch \
    "$@" \
    2>&1 | tee "$log"
}

run_sequence() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "=== sequence $name ===" | tee -a "$LOG_DIR/driver.log"
  $PY resonance/train_sequence_prediction.py \
    --output_dir "$OUT_ROOT/$name" \
    --device "$DEVICE" \
    --conditions "$CONDITIONS_FAST" \
    --seeds $SEEDS \
    --epochs "$SEQ_EPOCHS" \
    --train_examples "$TRAIN_EXAMPLES" \
    --val_examples "$VAL_EXAMPLES" \
    --max_length 768 \
    --vocab_size 10000 \
    --embed_dim "$EMBED_DIM" \
    --layers "$LAYERS" \
    --heads "$HEADS" \
    --ff_dim "$FF_DIM" \
    --batch_size "$BATCH_SIZE" \
    --dropout 0.05 \
    --eval_each_epoch \
    "$@" \
    2>&1 | tee "$log"
}

# Fast exact formal sequence tasks first.
run_sequence "vm_step_next_len8_to12" --task vm_step_next --depth 8 --val_depth 12 --modulus 17
run_probe "vm_trace_len8_to12" --task vm_trace --depth 8 --val_depth 12 --modulus 17
run_posttrain "vm_trace_exact_rl_curriculum" \
  --task vm_trace \
  --depth 8 \
  --val_depth 12 \
  --modulus 17 \
  --curriculum_tasks vm_step \
  --curriculum_epochs 2 \
  --curriculum_train_examples "$TRAIN_EXAMPLES" \
  --curriculum_val_examples 512

# Smaller lambda tasks after the VM path proves the run is healthy.
run_sequence "lambda_beta_next_depth4_to6" --task lambda_beta_next --depth 4 --val_depth 6 --max_size 14 --max_trace_steps 5
run_probe "lambda_trace_depth4_to6" --task lambda_trace --depth 4 --val_depth 6 --max_size 14 --max_trace_steps 6

echo "done: $OUT_ROOT" | tee -a "$LOG_DIR/driver.log"
