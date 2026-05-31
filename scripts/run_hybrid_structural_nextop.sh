#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

DEVICE="${DEVICE:-mps}"
ROOT="${ROOT:-resonance/outputs/hybrid_structural_$(date -u +%Y_%m_%d_%H%M%S)}"
mkdir -p "$ROOT/logs"

# This matrix intentionally drops the old static additive resonance-bias path.
# Current evidence says the promising question is whether a compact structural
# stream can condition Q/K/V computation when paired with a strong positional
# prior and/or explicit phase training pressure.
CONDITIONS="${CONDITIONS:-standard,standard_alibi,standard_deberta_lite,standard_iso,standard_iso_alibi,phase_stream_only_normalized,phase_dynamic_qk_film_normalized,phase_dynamic_qk_film_alibi_normalized,phase_dynamic_qk_film_alibi_normalized_phase_contrastive,relation_value_qk_film_normalized,relation_value_qk_film_alibi_normalized}"
SEEDS="${SEEDS:-809}"
EPOCHS="${EPOCHS:-6}"

run_probe() {
  local name="$1"
  shift
  echo "===== $name ====="
  uv run python resonance/structural_task_probe.py \
    --output_dir "$ROOT/$name" \
    --device "$DEVICE" \
    --conditions "$CONDITIONS" \
    --seeds $SEEDS \
    --epochs "$EPOCHS" \
    --train_examples "${TRAIN_EXAMPLES:-2500}" \
    --val_examples "${VAL_EXAMPLES:-1000}" \
    --embed_dim "${EMBED_DIM:-160}" \
    --layers "${LAYERS:-4}" \
    --heads "${HEADS:-4}" \
    --ff_dim "${FF_DIM:-640}" \
    --batch_size "${BATCH_SIZE:-24}" \
    --phase_contrastive_weight "${PHASE_CONTRASTIVE_WEIGHT:-0.0}" \
    --eval_each_epoch \
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

run_generic cogs_gen \
  data/processed/probes/cogs/train.jsonl \
  data/processed/probes/cogs/gen.jsonl \
  768 2500 1000

run_generic hans_evaluation \
  data/processed/probes/hans/train.jsonl \
  data/processed/probes/hans/evaluation.jsonl \
  384 2500 1000

uv run python resonance/summarize_architecture_matrix.py \
  --root "$ROOT" \
  --output_json "$ROOT/summary.json" \
  --output_md "$ROOT/summary.md"

uv run python resonance/plot_matrix_results.py --root "$ROOT"

echo "Wrote $ROOT"
