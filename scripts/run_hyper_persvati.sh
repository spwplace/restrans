#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/restrans-exp"
. .venv-exp/bin/activate
export PYTHONPATH=resonance
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export OMP_NUM_THREADS=8

ROOT="resonance/outputs/hyper_2026_04_28"
mkdir -p "$ROOT/logs"

CONDITIONS="standard,standard_iso,resonance_full_normalized,phase_stream_only_normalized,resonance_inert_normalized,phase_dynamic_mlp,phase_dynamic_attn,phase_qk_film,phase_dynamic_qk_film,resonance_dynamic_mlp_normalized,complex_directional_normalized,structural_heads_1_normalized"

run_probe() {
  local name="$1"
  shift
  echo "===== $name ====="
  python resonance/structural_task_probe.py \
    --output_dir "$ROOT/$name" \
    --device cuda \
    --conditions "$CONDITIONS" \
    --embed_dim 96 \
    --layers 3 \
    --heads 4 \
    --ff_dim 384 \
    --batch_size 16 \
    --eval_each_epoch \
    "$@" 2>&1 | tee "$ROOT/logs/$name.log"
}

run_probe wh_island_n150_variants \
  --task blimp \
  --data_path data/processed/blimp/wh_island.jsonl \
  --seeds 11 23 37 41 53 \
  --epochs 5 \
  --train_examples 150 \
  --val_examples 300 \
  --max_length 224

run_probe wh_island_n100_variants \
  --task blimp \
  --data_path data/processed/blimp/wh_island.jsonl \
  --seeds 11 23 37 41 53 67 79 \
  --epochs 5 \
  --train_examples 100 \
  --val_examples 300 \
  --max_length 224

run_probe graph_alias_variants \
  --task graph_alias \
  --seeds 11 23 37 41 53 \
  --epochs 5 \
  --train_examples 1000 \
  --val_examples 500 \
  --n_graphs 32 \
  --n_nodes 8 \
  --alias_pool_size 96 \
  --out_degree 2 \
  --walk_length 4 \
  --query_steps 1 \
  --val_graphs new \
  --max_length 256

run_probe template_equivalence_variants \
  --task template_equivalence \
  --seeds 11 23 37 41 53 \
  --epochs 5 \
  --train_examples 1000 \
  --val_examples 500 \
  --n_templates 80 \
  --depth 5 \
  --max_length 256

python resonance/build_lab_notebook.py \
  --root resonance/outputs \
  --output_dir resonance/outputs/lab_notebook_hyper || true
