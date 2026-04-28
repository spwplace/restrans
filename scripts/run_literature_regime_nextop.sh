#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run a follow-up literature-regime matrix on nextop.

The architecture matrix asks "which interface is promising?".  This script asks
"which training pressure / bottleneck / geometry regime makes a structural
stream meaningful?"  It covers the subset currently implemented in the repo:
phase dimension, supervised contrastive pressure on phase states, dispersion
regularization, and attention-interface choices inspired by ALiBi, DeBERTa,
Abstractor/DAT-style relational streams, and Q/K conditioning.

Environment knobs:
  DEVICE=mps|cpu
  ROOT=resonance/outputs/literature_regime_nextop_<timestamp>
  SEEDS="931 932 933"
  EPOCHS=4
  TRAIN_EXAMPLES=1536 VAL_EXAMPLES=384
  EMBED_DIM=64 LAYERS=2 HEADS=4 FF_DIM=256 BATCH_SIZE=24
EOF
  exit 0
fi

DEVICE="${DEVICE:-mps}"
ROOT="${ROOT:-resonance/outputs/literature_regime_nextop_$(date -u +%Y_%m_%d_%H%M%S)}"
mkdir -p "$ROOT/logs"

SEEDS="${SEEDS:-931 932 933}"

COMMON_ARGS=(
  --device "$DEVICE"
  --seeds $SEEDS
  --epochs "${EPOCHS:-4}"
  --embed_dim "${EMBED_DIM:-64}"
  --layers "${LAYERS:-2}"
  --heads "${HEADS:-4}"
  --ff_dim "${FF_DIM:-256}"
  --batch_size "${BATCH_SIZE:-24}"
  --train_examples "${TRAIN_EXAMPLES:-1536}"
  --val_examples "${VAL_EXAMPLES:-384}"
  --eval_each_epoch
)

run_regime() {
  local name="$1"
  shift
  echo "===== $name ====="
  uv run python resonance/structural_task_probe.py \
    --output_dir "$ROOT/$name" \
    "${COMMON_ARGS[@]}" \
    "$@" 2>&1 | tee "$ROOT/logs/$name.log"
}

# Literature comparator surface: ALiBi, DeBERTa-lite, Q/K conditioning, dynamic
# phase updates, structural heads, and relational-stream-lite in one task.
run_regime comparator_unification_depth4 \
  --task unification \
  --conditions standard,standard_alibi,standard_deberta_lite,standard_iso_deberta_lite,phase_stream_only_normalized,phase_dynamic_attn_normalized,phase_qk_film_normalized,phase_dynamic_qk_film_normalized,relation_value_mix_normalized,relation_value_qk_film_normalized,structural_heads_1_normalized,relational_stream_lite_normalized \
  --depth 4 \
  --max_length 224

# Phase bottleneck width: tests whether the stream needs enough latent capacity,
# or whether small circular/low-dimensional geometry is actually enough.
for freq in 8 16 32 64; do
  run_regime "phase_dim_${freq}_unification" \
    --task unification \
    --conditions phase_stream_only_normalized,resonance_full_normalized,phase_qk_film_normalized,phase_dynamic_qk_film_normalized,relation_value_mix_normalized \
    --n_frequencies "$freq" \
    --depth 4 \
    --max_length 224
done

# Kernel family: explicitly tests whether the useful object is a cosine phase
# relation, a learned per-band weighting, a multi-harmonic Fourier-style basis,
# a localized RBF relation, or attention-normalized phase nearest neighbors.
run_regime kernel_family_unification \
  --task unification \
  --conditions resonance_full_normalized,weighted_cosine_normalized,harmonic_cosine_normalized,rbf_kernel_normalized,attention_kernel_normalized,pair_mlp_kernel_normalized,harmonic_relation_value_normalized,pair_mlp_relation_value_normalized \
  --depth 4 \
  --max_length 224

# Objective pressure: tests Sparse-CLIP / contrastive-training lesson in the
# limited form currently implemented: supervised contrastive loss on answer
# phase states.
for weight in 0.03 0.10 0.30; do
  run_regime "phase_contrastive_${weight}_unification" \
    --task unification \
    --conditions phase_stream_only_normalized,resonance_full_normalized,phase_dynamic_qk_film_normalized \
    --phase_contrastive_weight "$weight" \
    --depth 4 \
    --max_length 224
done

# Geometry regularization: tests whether mild dispersion pressure improves the
# representation geometry without becoming the real explanation for accuracy.
for lambda in 0.001 0.003 0.010; do
  run_regime "dispersion_${lambda}_unification" \
    --task unification \
    --conditions standard,phase_stream_only_normalized,phase_qk_film_normalized,relational_stream_lite_normalized \
    --dispersion_lambda "$lambda" \
    --depth 4 \
    --max_length 224
done

# Shortcut-controlled structural tasks with the most promising interface set.
run_regime graph_alias_literature_interfaces \
  --task graph_alias \
  --conditions standard,standard_deberta_lite,phase_stream_only_normalized,phase_qk_film_normalized,phase_dynamic_qk_film_normalized,relation_value_mix_normalized,relational_stream_lite_normalized,harmonic_cosine_normalized \
  --n_graphs 48 \
  --n_nodes 12 \
  --aliases_per_node 3 \
  --alias_pool_size 144 \
  --out_degree 2 \
  --walk_length 4 \
  --query_steps 1 \
  --val_graphs new \
  --max_length 512

run_regime template_equivalence_literature_interfaces \
  --task template_equivalence \
  --conditions standard,standard_deberta_lite,phase_stream_only_normalized,phase_qk_film_normalized,phase_dynamic_qk_film_normalized,relation_value_mix_normalized,relational_stream_lite_normalized,harmonic_cosine_normalized \
  --n_templates 80 \
  --depth 5 \
  --max_length 256

uv run python resonance/summarize_architecture_matrix.py \
  --root "$ROOT" \
  --output_json "$ROOT/summary.json" \
  --output_md "$ROOT/summary.md"

cat > "$ROOT/README.md" <<EOF
# Literature Regime Nextop Matrix

This follow-up complements the architecture matrix. It tests currently
implemented lessons from the literature:

- ALiBi and DeBERTa-lite comparator baselines.
- Abstractor/DAT-style relational-stream-lite comparator.
- Q/K phase conditioning and dynamic Q/K phase conditioning.
- Phase bottleneck dimension sweep.
- Supervised phase contrastive objective sweep.
- Dispersion geometry regularization sweep.
- Shortcut-controlled graph alias and template equivalence tasks.

Still not covered here: full Graphormer/SAN graph encodings, exact DAT or
Abstractor parity, dependency/proof supervision, sparse structural activations,
SAE/transcoder training objectives, and LeanProgress/COGS/SLOG/CFQ medium runs.
EOF

echo "Wrote $ROOT"
