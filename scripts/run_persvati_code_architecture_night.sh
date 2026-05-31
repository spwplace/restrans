#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ember/restrans-exp}"
cd "$ROOT"

STAMP="${STAMP:-$(date +%Y_%m_%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-resonance/outputs/code_architecture_night_${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/code_architecture_night_${STAMP}}"
PY="${PY:-.venv-exp/bin/python}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

export HF_HOME="${HF_HOME:-$ROOT/data/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/data/hf_datasets}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"

COMMON_ARGS=(
  --device cuda
  --amp bf16
  --tokenizer gpt2
  --hf_cache "$HF_HOME"
  --seq_len 512
  --lr 2e-4
  --weight_decay 0.1
  --dropout 0.1
  --num_workers 2
  --torch_threads 12
  --lm_fraction 0.35
  --limit_per_answer_file 8000
  --val_limit_per_file 1200
  --limit_per_pair_file 2000
  --val_per_source 800
  --synthetic_train_examples 3000
  --synthetic_val_examples 600
  --synthetic_depth 5
  --save_final
)

run_one() {
  local scale="$1"
  local condition="$2"
  local seed="$3"
  local steps="$4"
  local batch="$5"
  local accum="$6"
  local eval_every="$7"
  local out="$OUT_ROOT/${scale}_${condition}_s${seed}"
  local log="$LOG_DIR/${scale}_${condition}_s${seed}.log"
  echo "=== $(date) scale=$scale condition=$condition seed=$seed steps=$steps ===" | tee -a "$LOG_DIR/driver.log"
  "$PY" resonance/train_code_structural.py \
    --output_dir "$out" \
    --scale "$scale" \
    --condition "$condition" \
    --seed "$seed" \
    --max_steps "$steps" \
    --batch_size "$batch" \
    --grad_accum "$accum" \
    --eval_every "$eval_every" \
    --log_every 25 \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$log"
}

# Main matched full-model architecture run.  The 125M jobs establish signal.
run_one 125M standard_alibi 801 1500 4 8 300
run_one 125M phase_dynamic_qk_film_alibi_normalized 801 1500 4 8 300
run_one 125M relation_value_qk_film_alibi_normalized 801 1500 4 8 300

# Serious midpoint.  These run after the 125M cells; if time runs out, partial
# histories are still useful.
run_one 350M standard_alibi 901 700 1 16 175
run_one 350M phase_dynamic_qk_film_alibi_normalized 901 700 1 16 175
run_one 350M relation_value_qk_film_alibi_normalized 901 700 1 16 175

echo "done: $OUT_ROOT" | tee -a "$LOG_DIR/driver.log"
