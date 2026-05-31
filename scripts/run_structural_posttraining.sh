#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

DEVICE="${DEVICE:-mps}"
ROOT="${ROOT:-resonance/outputs/posttraining_structural_$(date -u +%Y_%m_%d_%H%M%S)}"
mkdir -p "$ROOT/logs"

# Deliberately small condition set. This is not another architecture sweep; it
# asks whether explicit verifiable posttraining can make the structural stream
# useful relative to a strong ALiBi baseline.
CONDITIONS="${CONDITIONS:-standard_alibi,phase_dynamic_qk_film_alibi_normalized,relation_value_qk_film_alibi_normalized}"
ALGORITHMS="${ALGORITHMS:-dpo,exact_rl,grpo_enum}"
SEEDS="${SEEDS:-1301}"

COMMON_ARGS=(
  --device "$DEVICE"
  --conditions "$CONDITIONS"
  --seeds $SEEDS
  --sft_epochs "${SFT_EPOCHS:-2}"
  --rl_epochs "${RL_EPOCHS:-4}"
  --train_examples "${TRAIN_EXAMPLES:-1200}"
  --val_examples "${VAL_EXAMPLES:-512}"
  --embed_dim "${EMBED_DIM:-128}"
  --layers "${LAYERS:-4}"
  --heads "${HEADS:-4}"
  --ff_dim "${FF_DIM:-512}"
  --batch_size "${BATCH_SIZE:-8}"
  --lr "${LR:-3e-4}"
  --phase_contrastive_weight "${PHASE_CONTRASTIVE_WEIGHT:-0.05}"
  --eval_each_epoch
  --phase_ablation_eval
  --cascade_eval
)

run_task() {
  local algorithm="$1"
  local name="$2"
  shift 2
  echo "===== $algorithm / $name ====="
  uv run python resonance/train_structural_posttrain.py \
    --output_dir "$ROOT/$algorithm/$name" \
    --rl_algorithm "$algorithm" \
    "${COMMON_ARGS[@]}" \
    "$@" 2>&1 | tee "$ROOT/logs/${algorithm}_${name}.log"
}

IFS=',' read -ra ALG_ARRAY <<< "$ALGORITHMS"
for algorithm in "${ALG_ARRAY[@]}"; do
  run_task "$algorithm" lambda_trace_depth4 \
    --task lambda_trace \
    --depth 4 \
    --max_size 14 \
    --max_trace_steps 5 \
    --curriculum_tasks lambda_beta_step \
    --curriculum_epochs "${CURRICULUM_EPOCHS:-2}" \
    --curriculum_train_examples "${CURRICULUM_TRAIN_EXAMPLES:-1200}" \
    --curriculum_val_examples "${CURRICULUM_VAL_EXAMPLES:-256}" \
    --max_length 640

  run_task "$algorithm" vm_trace_len9 \
    --task vm_trace \
    --depth 9 \
    --modulus 17 \
    --curriculum_tasks vm_step \
    --curriculum_epochs "${CURRICULUM_EPOCHS:-2}" \
    --curriculum_train_examples "${CURRICULUM_TRAIN_EXAMPLES:-1200}" \
    --curriculum_val_examples "${CURRICULUM_VAL_EXAMPLES:-256}" \
    --max_length 520

  run_task "$algorithm" dfa_equivalence_states7 \
    --task dfa_equivalence \
    --depth 7 \
    --max_length 520

  run_task "$algorithm" cap_matching_depth6 \
    --task cap_matching \
    --depth 6 \
    --max_length 448

  run_task "$algorithm" algebraic_protocol_depth4 \
    --task algebraic_protocol \
    --depth 4 \
    --max_length 640
done

uv run python resonance/summarize_posttraining.py --root "$ROOT" || true

echo "Wrote $ROOT"
