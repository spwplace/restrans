#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HOME="${HF_HOME:-$ROOT/data/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/data/hf_datasets}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"

STAMP="${STAMP:-$(date +%Y_%m_%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-resonance/outputs/sequence_verifier_persvati_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/sequence_verifier_persvati_${STAMP}}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
PY="${PY:-.venv-exp/bin/python}"

CONDITIONS="${CONDITIONS:-standard_alibi,standard_deberta_lite,phase_dynamic_qk_film_alibi_normalized,complex_directional_normalized,harmonic_relation_value_normalized}"
SEEDS="${SEEDS:-821 822 823}"
EPOCHS="${EPOCHS:-8}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-8192}"
VAL_EXAMPLES="${VAL_EXAMPLES:-3072}"
EMBED_DIM="${EMBED_DIM:-256}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-8}"
FF_DIM="${FF_DIM:-1024}"
BATCH_SIZE="${BATCH_SIZE:-48}"
LR="${LR:-3e-4}"

echo "persvati sequence verifier sweep"
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
    --max_length 1024 \
    --vocab_size 12000 \
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
    --rl_epochs 4 \
    --rl_algorithm exact_rl \
    --reference after_sft \
    --train_examples "$TRAIN_EXAMPLES" \
    --val_examples "$VAL_EXAMPLES" \
    --max_length 896 \
    --vocab_size 12000 \
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

run_probe "lambda_beta_step_depth6" --task lambda_beta_step --depth 6 --val_depth 6 --max_size 22
run_probe "lambda_trace_depth6_to8" --task lambda_trace --depth 6 --val_depth 8 --max_size 24 --max_trace_steps 12
run_probe "vm_step_len10" --task vm_step --depth 10 --val_depth 10 --modulus 23
run_probe "vm_trace_len10_to14" --task vm_trace --depth 10 --val_depth 14 --modulus 23
run_probe "lambda_equiv_depth6" --task lambda_equivalence --depth 6 --val_depth 6 --max_size 22 --max_trace_steps 16
run_probe "vm_equiv_len12" --task vm_equivalence --depth 12 --val_depth 12 --modulus 23
run_probe "unification_depth6" --task unification --depth 6 --val_depth 6

run_posttrain "lambda_trace_exact_rl_curriculum" \
  --task lambda_trace \
  --depth 6 \
  --val_depth 8 \
  --max_size 24 \
  --max_trace_steps 12 \
  --curriculum_tasks lambda_beta_step \
  --curriculum_epochs 2 \
  --curriculum_train_examples "$TRAIN_EXAMPLES" \
  --curriculum_val_examples 1024

run_posttrain "vm_trace_exact_rl_curriculum" \
  --task vm_trace \
  --depth 10 \
  --val_depth 14 \
  --modulus 23 \
  --curriculum_tasks vm_step \
  --curriculum_epochs 2 \
  --curriculum_train_examples "$TRAIN_EXAMPLES" \
  --curriculum_val_examples 1024

echo "done: $OUT_ROOT" | tee -a "$LOG_DIR/driver.log"
