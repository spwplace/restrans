#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/restrans-exp"
. .venv-exp/bin/activate
export PYTHONPATH=resonance
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export OMP_NUM_THREADS=8

ROOT="resonance/outputs/overnight_2026_04_27"
mkdir -p "$ROOT/logs"

run_probe() {
  local name="$1"
  shift
  echo "===== $name ====="
  python resonance/structural_task_probe.py \
    --output_dir "$ROOT/$name" \
    --device cuda \
    --conditions standard,resonance_full_normalized,phase_stream_only_normalized,bias_only_normalized,resonance_inert_normalized \
    --seeds 11 23 37 41 53 \
    --epochs 5 \
    --train_examples 300 \
    --val_examples 500 \
    --embed_dim 96 \
    --layers 3 \
    --heads 4 \
    --ff_dim 384 \
    --batch_size 16 \
    --eval_each_epoch \
    "$@" 2>&1 | tee "$ROOT/logs/$name.log"
}

run_lowdata_blimp() {
  local paradigm="$1"
  local train_n="$2"
  run_probe "blimp_${paradigm}_n${train_n}" \
    --task blimp \
    --data_path "data/processed/blimp/${paradigm}.jsonl" \
    --train_examples "$train_n" \
    --val_examples 300 \
    --max_length 224
}

run_generic() {
  local name="$1"
  local train_path="$2"
  local val_path="$3"
  local max_length="$4"
  local train_n="$5"
  local val_n="$6"
  run_probe "$name" \
    --task generic_probe \
    --data_path "$train_path" \
    --val_data_path "$val_path" \
    --max_length "$max_length" \
    --train_examples "$train_n" \
    --val_examples "$val_n"
}

run_synth() {
  local name="$1"
  shift
  run_probe "$name" \
    --train_examples 1000 \
    --val_examples 500 \
    "$@"
}

run_lowdata_blimp wh_island 100
run_lowdata_blimp wh_island 150
run_lowdata_blimp adjunct_island 100
run_lowdata_blimp complex_NP_island 100
run_lowdata_blimp left_branch_island_simple_question 100
run_lowdata_blimp coordinate_structure_constraint_object_extraction 100
run_lowdata_blimp principle_A_c_command 100
run_lowdata_blimp distractor_agreement_relative_clause 100

run_generic hans_small data/processed/probes/hans/train.jsonl data/processed/probes/hans/evaluation.jsonl 256 1200 1200
run_generic msgs_main_verb_length data/processed/probes/msgs/main_verb_control/train.jsonl data/processed/probes/msgs/main_verb_length/test.jsonl 256 1000 500
run_generic msgs_syntactic_category_relative_position data/processed/probes/msgs/syntactic_category_control/train.jsonl data/processed/probes/msgs/syntactic_category_relative_position/test.jsonl 256 1000 500
run_generic cogs_gen_probe data/processed/probes/cogs/train.jsonl data/processed/probes/cogs/gen.jsonl 384 1200 600
run_generic slog_gen_probe data/processed/probes/slog/slog_cogs_lf_train.jsonl data/processed/probes/slog/slog_gen_cogs_lf.jsonl 384 1200 600
run_generic cfq_mcd1_probe data/processed/probes/cfq/mcd1/train.jsonl data/processed/probes/cfq/mcd1/test.jsonl 512 1200 600
run_generic poj104_probe data/processed/probes/poj104/train.jsonl data/processed/probes/poj104/validation.jsonl 768 1000 500
run_generic bigclonebench_probe data/processed/probes/bigclonebench/train.jsonl data/processed/probes/bigclonebench/validation.jsonl 1024 800 400

run_synth graph_alias_new_graphs --task graph_alias --n_graphs 32 --n_nodes 8 --alias_pool_size 96 --out_degree 2 --walk_length 4 --query_steps 1 --val_graphs new --max_length 256
run_synth template_equivalence_depth5 --task template_equivalence --n_templates 80 --depth 5 --max_length 256
run_synth unification_depth4 --task unification --depth 4 --max_length 256
run_synth dyck_cross_depth8 --task dyck --dyck_mode cross --depth 8 --n_types 3 --max_length 192
run_synth causal_intervention_8ev --task causal_intervention --n_events 8 --edge_prob 0.30 --max_length 256

python resonance/summarize_experiment_plan.py --root "$ROOT" || true
python resonance/build_lab_notebook.py --root resonance/outputs --output_dir resonance/outputs/lab_notebook_remote || true
