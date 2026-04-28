#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run a tiny local architecture matrix over every currently registered condition.

This is an implementation health and directional-signal test, not a paper
result.  It exercises all architectural variants on small formal/compositional
and code-equivalence probes, then writes a summary.

Environment knobs:
  DEVICE=cpu|mps|cuda
  ROOT=resonance/outputs/architecture_matrix_smoke_<timestamp>
  CONDITIONS=comma,separated,conditions
  SEEDS="901"
  EPOCHS=1
  EMBED_DIM=32 LAYERS=1 HEADS=4 FF_DIM=64
  BATCH_SIZE=8 CODE_BATCH_SIZE=4
EOF
  exit 0
fi

DEVICE="${DEVICE:-cpu}"
ROOT="${ROOT:-resonance/outputs/architecture_matrix_smoke_$(date -u +%Y_%m_%d_%H%M%S)}"
mkdir -p "$ROOT/logs"

CONDITIONS="${CONDITIONS:-standard,standard_iso,standard_alibi,standard_deberta_lite,standard_iso_alibi,standard_iso_deberta_lite,resonance_full_normalized,phase_stream_only_normalized,bias_only_normalized,resonance_inert_normalized,phase_stream_only_normalized_phase_contrastive,resonance_full_normalized_phase_contrastive,phase_dynamic_mlp_normalized,phase_dynamic_attn_normalized,phase_qk_film_normalized,phase_dynamic_qk_film_normalized,resonance_dynamic_mlp_normalized,complex_directional_normalized,structural_heads_1_normalized,relational_stream_lite_normalized}"
SEEDS="${SEEDS:-901}"

run_probe() {
  local name="$1"
  shift
  echo "===== $name ====="
  uv run python resonance/structural_task_probe.py \
    --output_dir "$ROOT/$name" \
    --device "$DEVICE" \
    --conditions "$CONDITIONS" \
    --seeds $SEEDS \
    --epochs "${EPOCHS:-1}" \
    --embed_dim "${EMBED_DIM:-32}" \
    --layers "${LAYERS:-1}" \
    --heads "${HEADS:-4}" \
    --ff_dim "${FF_DIM:-64}" \
    --batch_size "${BATCH_SIZE:-8}" \
    --eval_each_epoch \
    "$@" 2>&1 | tee "$ROOT/logs/$name.log"
}

run_probe unification_depth4 \
  --task unification \
  --train_examples "${TRAIN_EXAMPLES:-96}" \
  --val_examples "${VAL_EXAMPLES:-48}" \
  --depth 4 \
  --max_length 192

run_probe listops_depth4 \
  --task listops \
  --train_examples "${TRAIN_EXAMPLES:-96}" \
  --val_examples "${VAL_EXAMPLES:-48}" \
  --depth 4 \
  --n_types 4 \
  --max_length 256

if [[ -f data/processed/probes/equibench/OJ_VA/train.jsonl ]]; then
  run_probe equibench_oj_va \
    --task generic_probe \
    --data_path data/processed/probes/equibench/OJ_VA/train.jsonl \
    --train_examples "${CODE_TRAIN_EXAMPLES:-40}" \
    --val_examples "${CODE_VAL_EXAMPLES:-24}" \
    --batch_size "${CODE_BATCH_SIZE:-4}" \
    --max_length 1536
else
  echo "Skipping equibench_oj_va: data/processed/probes/equibench/OJ_VA/train.jsonl missing"
fi

uv run python resonance/summarize_architecture_matrix.py \
  --root "$ROOT" \
  --output_json "$ROOT/summary.json" \
  --output_md "$ROOT/summary.md"

echo "Wrote $ROOT"
