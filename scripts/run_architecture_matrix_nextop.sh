#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run a substantive local architecture matrix on nextop.

This is larger than the smoke test but still intended as a local regime-finding
matrix, not a final paper benchmark. Defaults are chosen to keep the Apple MPS
device busy for hours while covering all currently registered architecture
conditions across formal, compositional, graph, story, and code-equivalence
probes.

Environment knobs:
  DEVICE=mps|cpu
  ROOT=resonance/outputs/architecture_matrix_nextop_<timestamp>
  CONDITIONS=comma,separated,conditions
  SEEDS="921 922 923"
  EPOCHS=4
  TRAIN_EXAMPLES=1536 VAL_EXAMPLES=384
  CODE_TRAIN_EXAMPLES=320 CODE_VAL_EXAMPLES=80
  EMBED_DIM=64 LAYERS=2 HEADS=4 FF_DIM=256
  BATCH_SIZE=24 CODE_BATCH_SIZE=2
EOF
  exit 0
fi

DEVICE="${DEVICE:-mps}"
ROOT="${ROOT:-resonance/outputs/architecture_matrix_nextop_$(date -u +%Y_%m_%d_%H%M%S)}"
mkdir -p "$ROOT/logs"

CONDITIONS="${CONDITIONS:-standard,standard_iso,standard_alibi,standard_deberta_lite,standard_iso_alibi,standard_iso_deberta_lite,resonance_full_normalized,phase_stream_only_normalized,bias_only_normalized,resonance_inert_normalized,phase_stream_only_normalized_phase_contrastive,resonance_full_normalized_phase_contrastive,phase_dynamic_mlp_normalized,phase_dynamic_attn_normalized,phase_qk_film_normalized,phase_dynamic_qk_film_normalized,resonance_dynamic_mlp_normalized,complex_directional_normalized,structural_heads_1_normalized,relational_stream_lite_normalized}"
SEEDS="${SEEDS:-921 922 923}"

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

run_probe unification_depth4 \
  --task unification \
  --train_examples "${TRAIN_EXAMPLES:-1536}" \
  --val_examples "${VAL_EXAMPLES:-384}" \
  --depth 4 \
  --max_length 224

run_probe unification_depth5 \
  --task unification \
  --train_examples "${HARD_TRAIN_EXAMPLES:-1024}" \
  --val_examples "${HARD_VAL_EXAMPLES:-256}" \
  --depth 5 \
  --max_length 256

run_probe listops_depth4 \
  --task listops \
  --train_examples "${TRAIN_EXAMPLES:-1536}" \
  --val_examples "${VAL_EXAMPLES:-384}" \
  --depth 4 \
  --n_types 4 \
  --max_length 512

run_probe dyck_cross_depth8 \
  --task dyck \
  --train_examples "${TRAIN_EXAMPLES:-1536}" \
  --val_examples "${VAL_EXAMPLES:-384}" \
  --depth 8 \
  --n_types 3 \
  --dyck_mode cross \
  --max_length 192

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

run_probe template_equivalence_depth5 \
  --task template_equivalence \
  --train_examples "${TRAIN_EXAMPLES:-1536}" \
  --val_examples "${VAL_EXAMPLES:-384}" \
  --n_templates 80 \
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

run_probe causal_intervention \
  --task causal_intervention \
  --train_examples "${TRAIN_EXAMPLES:-1536}" \
  --val_examples "${VAL_EXAMPLES:-384}" \
  --n_events 8 \
  --edge_prob 0.30 \
  --max_length 256

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

echo "Wrote $ROOT"
