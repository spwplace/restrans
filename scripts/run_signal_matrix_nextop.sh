#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the focused signal matrix on nextop.

This is the post-smoke local matrix: it deliberately skips weak/slow canaries
like ListOps and spends Apple MPS time on tasks that can plausibly distinguish
structural interfaces:

  - algebraic unification and cap matching
  - algebraic protocol synthesis
  - aliased graph-walk generalization
  - COGS semantic-parse generalization
  - a small EquiBench OJ_VA slice when available

Environment knobs:
  DEVICE=mps|cpu
  ROOT=resonance/outputs/signal_matrix_nextop_<timestamp>
  CONDITIONS=comma,separated,conditions
  SEEDS="941 943 947"
  EPOCHS=4
  TRAIN_EXAMPLES=1536 VAL_EXAMPLES=384
  COGS_TRAIN_EXAMPLES=2000 COGS_VAL_EXAMPLES=800
  CODE_TRAIN_EXAMPLES=320 CODE_VAL_EXAMPLES=80
  EMBED_DIM=64 LAYERS=2 HEADS=4 FF_DIM=256
  BATCH_SIZE=24 COGS_BATCH_SIZE=12 CODE_BATCH_SIZE=2
EOF
  exit 0
fi

DEVICE="${DEVICE:-mps}"
ROOT="${ROOT:-resonance/outputs/signal_matrix_nextop_$(date -u +%Y_%m_%d_%H%M%S)}"
mkdir -p "$ROOT/logs"

CONDITIONS="${CONDITIONS:-standard,standard_alibi,standard_deberta_lite,standard_iso_deberta_lite,resonance_full_normalized,phase_stream_only_normalized,phase_dynamic_qk_film_normalized,relation_value_mix_normalized,relation_value_qk_film_normalized,harmonic_cosine_normalized,harmonic_relation_value_normalized,relational_stream_lite_normalized}"
SEEDS="${SEEDS:-941 943 947}"

run_probe() {
  local name="$1"
  shift
  echo "===== $name ====="
  uv run python resonance/structural_task_probe.py \
    --output_dir "$ROOT/$name" \
    --device "$DEVICE" \
    --conditions "$CONDITIONS" \
    --seeds $SEEDS \
    --epochs "${EPOCHS:-4}" \
    --embed_dim "${EMBED_DIM:-64}" \
    --layers "${LAYERS:-2}" \
    --heads "${HEADS:-4}" \
    --ff_dim "${FF_DIM:-256}" \
    --batch_size "${BATCH_SIZE:-24}" \
    --eval_each_epoch \
    "$@" 2>&1 | tee "$ROOT/logs/$name.log"
}

run_probe unification_depth5 \
  --task unification \
  --train_examples "${HARD_TRAIN_EXAMPLES:-1024}" \
  --val_examples "${HARD_VAL_EXAMPLES:-256}" \
  --depth 5 \
  --max_length 256

run_probe cap_matching_depth4 \
  --task cap_matching \
  --train_examples "${TRAIN_EXAMPLES:-1536}" \
  --val_examples "${VAL_EXAMPLES:-384}" \
  --depth 4 \
  --max_length 256

run_probe algebraic_protocol_depth4 \
  --task algebraic_protocol \
  --train_examples "${TRAIN_EXAMPLES:-1536}" \
  --val_examples "${VAL_EXAMPLES:-384}" \
  --depth 4 \
  --max_length 256

run_probe graph_alias_new \
  --task graph_alias \
  --train_examples "${TRAIN_EXAMPLES:-1536}" \
  --val_examples "${VAL_EXAMPLES:-384}" \
  --n_graphs 48 \
  --n_nodes 12 \
  --aliases_per_node 3 \
  --alias_pool_size 144 \
  --out_degree 2 \
  --walk_length 4 \
  --query_steps 1 \
  --val_graphs new \
  --max_length 512

run_probe cogs_gen_semantic_parse \
  --task generic_probe \
  --data_path data/processed/probes/cogs/train.jsonl \
  --val_data_path data/processed/probes/cogs/gen.jsonl \
  --train_examples "${COGS_TRAIN_EXAMPLES:-2000}" \
  --val_examples "${COGS_VAL_EXAMPLES:-800}" \
  --batch_size "${COGS_BATCH_SIZE:-12}" \
  --max_length 768

if [[ -f data/processed/probes/equibench/OJ_VA/train.jsonl ]]; then
  run_probe equibench_oj_va \
    --task generic_probe \
    --data_path data/processed/probes/equibench/OJ_VA/train.jsonl \
    --train_examples "${CODE_TRAIN_EXAMPLES:-320}" \
    --val_examples "${CODE_VAL_EXAMPLES:-80}" \
    --batch_size "${CODE_BATCH_SIZE:-2}" \
    --max_length 1536
else
  echo "Skipping equibench_oj_va: data/processed/probes/equibench/OJ_VA/train.jsonl missing"
fi

uv run python resonance/summarize_architecture_matrix.py \
  --root "$ROOT" \
  --output_json "$ROOT/summary.json" \
  --output_md "$ROOT/summary.md"

cat > "$ROOT/README.md" <<EOF
# Signal Matrix Nextop

This run intentionally skips ListOps and other weak canaries. It focuses on
tasks where we currently expect structural interfaces to matter:

- unification depth 5 and cap matching
- algebraic protocol generalization
- aliased graph walk generalization
- COGS semantic parse generalization
- a small EquiBench OJ_VA slice when available

Condition set:

\`$CONDITIONS\`
EOF

echo "Wrote $ROOT"
