#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="${STAMP:-$(date +%Y_%m_%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-resonance/outputs/sequence_verifier_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/sequence_verifier_${STAMP}}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

DEVICE="${DEVICE:-mps}"
PY="${PY:-uv run python}"

CONDITIONS="${CONDITIONS:-standard_alibi,standard_deberta_lite,phase_dynamic_qk_film_alibi_normalized,complex_directional_normalized,harmonic_relation_value_normalized}"
SEEDS="${SEEDS:-811 812}"
EPOCHS="${EPOCHS:-6}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-3072}"
VAL_EXAMPLES="${VAL_EXAMPLES:-1536}"
EMBED_DIM="${EMBED_DIM:-192}"
LAYERS="${LAYERS:-4}"
HEADS="${HEADS:-4}"
FF_DIM="${FF_DIM:-768}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"

echo "sequence verifier sweep"
echo "out=$OUT_ROOT"
echo "conditions=$CONDITIONS"
echo "seeds=$SEEDS"

run_probe() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "=== probe $name ===" | tee -a "$LOG_DIR/driver.log"
  $PY resonance/structural_task_probe.py \
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
    --lr "$LR" \
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
    --conditions "$CONDITIONS" \
    --seeds $SEEDS \
    --sft_epochs 2 \
    --rl_epochs 3 \
    --rl_algorithm exact_rl \
    --reference after_sft \
    --train_examples "$TRAIN_EXAMPLES" \
    --val_examples "$VAL_EXAMPLES" \
    --max_length 768 \
    --vocab_size 8192 \
    --embed_dim "$EMBED_DIM" \
    --layers "$LAYERS" \
    --heads "$HEADS" \
    --ff_dim "$FF_DIM" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --dropout 0.05 \
    --phase_contrastive_weight 0.05 \
    --phase_ablation_eval \
    --cascade_eval \
    --eval_each_epoch \
    "$@" \
    2>&1 | tee "$log"
}

# Local-rule probes: exact, verifiable sequence transitions.
run_probe "lambda_beta_step_depth5" --task lambda_beta_step --depth 5 --val_depth 5 --max_size 18
run_probe "vm_step_len8" --task vm_step --depth 8 --val_depth 8 --modulus 17

# Whole-cascade probes: do local rules compose into multi-step validity?
run_probe "lambda_trace_depth5" --task lambda_trace --depth 5 --val_depth 5 --max_size 18 --max_trace_steps 8
run_probe "lambda_trace_depth5_to7" --task lambda_trace --depth 5 --val_depth 7 --max_size 20 --max_trace_steps 10
run_probe "vm_trace_len8" --task vm_trace --depth 8 --val_depth 8 --modulus 17
run_probe "vm_trace_len8_to12" --task vm_trace --depth 8 --val_depth 12 --modulus 17

# Equivalence probes: behavioral equality, closer to the original thesis.
run_probe "lambda_equiv_depth5" --task lambda_equivalence --depth 5 --val_depth 5 --max_size 18 --max_trace_steps 12
run_probe "vm_equiv_len10" --task vm_equivalence --depth 10 --val_depth 10 --modulus 17
run_probe "unification_depth5" --task unification --depth 5 --val_depth 5

# RLVR-style posttraining: one-step curriculum, then whole-trace target.
run_posttrain "lambda_trace_exact_rl_curriculum" \
  --task lambda_trace \
  --depth 5 \
  --val_depth 7 \
  --max_size 20 \
  --max_trace_steps 10 \
  --curriculum_tasks lambda_beta_step \
  --curriculum_epochs 2 \
  --curriculum_train_examples "$TRAIN_EXAMPLES" \
  --curriculum_val_examples 512

run_posttrain "vm_trace_exact_rl_curriculum" \
  --task vm_trace \
  --depth 8 \
  --val_depth 12 \
  --modulus 17 \
  --curriculum_tasks vm_step \
  --curriculum_epochs 2 \
  --curriculum_train_examples "$TRAIN_EXAMPLES" \
  --curriculum_val_examples 512

echo "done: $OUT_ROOT" | tee -a "$LOG_DIR/driver.log"
