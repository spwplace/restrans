#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export OMP_NUM_THREADS=4

ROOT="resonance/outputs/local_overnight_2026_04_27"
mkdir -p "$ROOT/logs"

run_cpu_probe() {
  local name="$1"
  shift
  echo "===== $name ====="
  uv run python resonance/structural_task_probe.py \
    --output_dir "$ROOT/$name" \
    --device cpu \
    --conditions standard,resonance_full_normalized,phase_stream_only_normalized,bias_only_normalized,resonance_inert_normalized \
    --seeds 101 103 107 \
    --epochs 4 \
    --train_examples 600 \
    --val_examples 300 \
    --embed_dim 80 \
    --layers 3 \
    --heads 4 \
    --ff_dim 320 \
    --batch_size 16 \
    --eval_each_epoch \
    "$@" >"$ROOT/logs/$name.log" 2>&1
}

run_cpu_probe blimp_wh_island_n75 --task blimp --data_path data/processed/blimp/wh_island.jsonl --train_examples 75 --val_examples 300 --max_length 224 &
run_cpu_probe blimp_wh_island_n125 --task blimp --data_path data/processed/blimp/wh_island.jsonl --train_examples 125 --val_examples 300 --max_length 224 &
run_cpu_probe blimp_adjunct_island_n75 --task blimp --data_path data/processed/blimp/adjunct_island.jsonl --train_examples 75 --val_examples 300 --max_length 224 &
run_cpu_probe graph_alias_new_graphs_small --task graph_alias --n_graphs 32 --n_nodes 8 --alias_pool_size 96 --out_degree 2 --walk_length 4 --query_steps 1 --val_graphs new --max_length 256 &
wait

run_cpu_probe template_equivalence_depth5 --task template_equivalence --n_templates 80 --depth 5 --max_length 256 &
run_cpu_probe unification_depth4 --task unification --depth 4 --max_length 256 &
run_cpu_probe causal_intervention_8ev --task causal_intervention --n_events 8 --edge_prob 0.30 --max_length 256 &
run_cpu_probe structural_paraphrase --task structural_paraphrase --max_length 256 &
wait

uv run python resonance/build_lab_notebook.py --root resonance/outputs --output_dir resonance/outputs/lab_notebook_local
