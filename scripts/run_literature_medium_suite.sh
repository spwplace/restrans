#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the literature-grounded medium structural suite.

Environment knobs:
  DEVICE=cpu|mps|cuda
  ROOT=resonance/outputs/custom_dir
  CONDITIONS=comma,separated,conditions
  SEEDS="701 703 707"
  EPOCHS=6 TRAIN_EXAMPLES=2500 VAL_EXAMPLES=1000
  EMBED_DIM=160 LAYERS=4 HEADS=4 FF_DIM=640 BATCH_SIZE=24
  SAVE_MODELS=1
  DRY_RUN=1

Example:
  DEVICE=mps bash scripts/run_literature_medium_suite.sh
EOF
  exit 0
fi

DEVICE="${DEVICE:-cpu}"
ROOT="${ROOT:-resonance/outputs/literature_medium_$(date -u +%Y_%m_%d_%H%M%S)}"
mkdir -p "$ROOT/logs"

CONDITIONS="${CONDITIONS:-standard,standard_iso,standard_alibi,standard_deberta_lite,phase_stream_only_normalized,phase_stream_only_normalized_phase_contrastive,resonance_full_normalized,phase_dynamic_qk_film,complex_directional_normalized,relational_stream_lite}"
SEEDS="${SEEDS:-701 703 707}"

run_probe() {
  local name="$1"
  shift
  echo "===== $name ====="
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN structural_task_probe $*"
    return 0
  fi
  local save_args=()
  if [[ "${SAVE_MODELS:-0}" == "1" ]]; then
    save_args+=(--save_models)
  fi
  uv run python resonance/structural_task_probe.py \
    --output_dir "$ROOT/$name" \
    --device "$DEVICE" \
    --conditions "$CONDITIONS" \
    --seeds $SEEDS \
    --epochs "${EPOCHS:-6}" \
    --train_examples "${TRAIN_EXAMPLES:-2500}" \
    --val_examples "${VAL_EXAMPLES:-1000}" \
    --embed_dim "${EMBED_DIM:-160}" \
    --layers "${LAYERS:-4}" \
    --heads "${HEADS:-4}" \
    --ff_dim "${FF_DIM:-640}" \
    --batch_size "${BATCH_SIZE:-24}" \
    --phase_contrastive_weight "${PHASE_CONTRASTIVE_WEIGHT:-0.0}" \
    --eval_each_epoch \
    "${save_args[@]}" \
    "$@" 2>&1 | tee "$ROOT/logs/$name.log"
}

run_generic() {
  local name="$1"
  local train_path="$2"
  local val_path="$3"
  local max_length="$4"
  local train_examples="$5"
  local val_examples="$6"
  run_probe "$name" \
    --task generic_probe \
    --data_path "$train_path" \
    --val_data_path "$val_path" \
    --max_length "$max_length" \
    --train_examples "$train_examples" \
    --val_examples "$val_examples"
}

run_generic_samefile() {
  local name="$1"
  local path="$2"
  local max_length="$3"
  local train_examples="$4"
  local val_examples="$5"
  run_probe "$name" \
    --task generic_probe \
    --data_path "$path" \
    --max_length "$max_length" \
    --train_examples "$train_examples" \
    --val_examples "$val_examples" \
    --batch_size "${CODE_BATCH_SIZE:-8}"
}

# Tier A/B bridge: exact formal labels, self-contained.
run_probe cap_matching_depth6 \
  --task cap_matching \
  --depth 6 \
  --max_length 448

run_probe algebraic_protocol_depth4 \
  --task algebraic_protocol \
  --depth 4 \
  --max_length 640

run_probe unification_depth5 \
  --task unification \
  --depth 5 \
  --max_length 256

run_probe listops_depth5 \
  --task listops \
  --depth 5 \
  --n_types 4 \
  --max_length 384

# Main natural-language structural-generalization branch.
run_generic cogs_gen \
  data/processed/probes/cogs/train.jsonl \
  data/processed/probes/cogs/gen.jsonl \
  768 2500 1000

run_generic slog_gen_cogs_lf \
  data/processed/probes/slog/slog_cogs_lf_train.jsonl \
  data/processed/probes/slog/slog_gen_cogs_lf.jsonl \
  768 2500 1000

run_generic cfq_mcd1 \
  data/processed/probes/cfq/mcd1/train.jsonl \
  data/processed/probes/cfq/mcd1/test.jsonl \
  768 3000 1000

if [[ "${RUN_EQUIBENCH:-1}" == "1" && -f data/processed/probes/equibench/OJ_VA/train.jsonl ]]; then
  run_generic_samefile equibench_oj_va \
    data/processed/probes/equibench/OJ_VA/train.jsonl \
    1536 300 100

  run_generic_samefile equibench_stoke \
    data/processed/probes/equibench/STOKE/train.jsonl \
    1536 300 100
fi

# Shortcut diagnostics: not headline evidence, but required guardrails.
run_generic hans_evaluation \
  data/processed/probes/hans/train.jsonl \
  data/processed/probes/hans/evaluation.jsonl \
  384 2500 1000

run_generic msgs_syntactic_category_relative_position \
  data/processed/probes/msgs/syntactic_category_control/train.jsonl \
  data/processed/probes/msgs/syntactic_category_relative_position/test.jsonl \
  384 2500 1000

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  if [[ "${RUN_INTERP:-0}" == "1" ]]; then
    bash scripts/run_interpretability_on_suite.sh "$ROOT" || true
  fi
  uv run python resonance/build_lab_notebook.py \
    --root resonance/outputs \
    --output_dir "$ROOT/lab_notebook" || true
fi

echo "Wrote $ROOT"
