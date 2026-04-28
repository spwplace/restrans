#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-resonance}"
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

ROOT="${ROOT:-/private/tmp/restrans_entrypoint_smoke}"
LOG_DIR="$ROOT/logs"
REPORT="$ROOT/report.md"
mkdir -p "$LOG_DIR"

PASS=0
FAIL=0
FAILURES=()

run_check() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name//[^A-Za-z0-9_.-]/_}.log"
  echo "===== $name ====="
  echo "\$ $*" >"$log"
  if "$@" >>"$log" 2>&1; then
    echo "PASS $name"
    PASS=$((PASS + 1))
  else
    local code=$?
    echo "FAIL $name (exit $code)"
    FAIL=$((FAIL + 1))
    FAILURES+=("$name")
  fi
}

write_report() {
  {
    echo "# Entry Point Smoke Test"
    echo
    echo "- Pass: \`$PASS\`"
    echo "- Fail: \`$FAIL\`"
    echo "- Root: \`$ROOT\`"
    echo
    if [[ "$FAIL" -gt 0 ]]; then
      echo "## Failures"
      echo
      for item in "${FAILURES[@]}"; do
        echo "- $item"
      done
      echo
    fi
    echo "## Notes"
    echo
    echo "- Remote and overnight scripts are syntax-checked only; running them would SSH into persvati or launch long jobs."
    echo "- The handoff-critical local paths get real tiny execution checks."
  } >"$REPORT"
}

echo "Smoke root: $ROOT"

# 1. Shell script syntax for every shell entrypoint.
for script in scripts/*.sh; do
  run_check "bash-n:${script}" bash -n "$script"
done

# 2. Help/dry-run checks for handoff shell entrypoints.
run_check "help:acquire_research_assets" bash scripts/acquire_research_assets.sh --help
run_check "help:setup_leandojo_progress" bash scripts/setup_leandojo_progress.sh --help
run_check "help:run_interpretability_on_suite" bash scripts/run_interpretability_on_suite.sh --help
run_check "help:run_literature_medium_suite" bash scripts/run_literature_medium_suite.sh --help
run_check "dry-run:run_literature_medium_suite" env DRY_RUN=1 RUN_EQUIBENCH=1 SAVE_MODELS=1 RUN_INTERP=1 ROOT="$ROOT/literature_dryrun" bash scripts/run_literature_medium_suite.sh

# 3. Python syntax for all package and top-level scripts.
mapfile -t python_files < <(find resonance -type f -name '*.py' -not -path '*/__pycache__/*' | sort)
run_check "py_compile:resonance" uv run python -m py_compile "${python_files[@]}"

# 4. --help for argparse top-level scripts. These should not train.
help_scripts=(
  resonance/analysis.py
  resonance/build_lab_notebook.py
  resonance/download_datasets.py
  resonance/experiment_text.py
  resonance/found_suite.py
  resonance/graph_alias_probe.py
  resonance/kernel_quick_compare.py
  resonance/prepare_external_probe_data.py
  resonance/regime_probe.py
  resonance/research_eval.py
  resonance/resonance_kernel_sweep.py
  resonance/run_ablations.py
  resonance/run_experiment_plan.py
  resonance/run_experiments.py
  resonance/run_interpretability_audit.py
  resonance/run_standardized.py
  resonance/story_query_eval.py
  resonance/story_topology_eval.py
  resonance/structural_task_probe.py
  resonance/summarize_experiment_plan.py
  resonance/sweep_variants.py
  resonance/topology_eval.py
  resonance/train.py
  resonance/train_corpus_lm.py
)
for script in "${help_scripts[@]}"; do
  run_check "help:${script}" uv run python "$script" --help
done

# 5. Tiny real execution checks for handoff-critical paths.
run_check "direct-smoke:test_smoke" env PYTHONPATH=resonance uv run python resonance/tests/test_smoke.py

run_check "tiny:structural_task_checkpoint" uv run python resonance/structural_task_probe.py \
  --task unification \
  --output_dir "$ROOT/structural_checkpoint" \
  --conditions standard,phase_stream_only_normalized,resonance_full_normalized \
  --seeds 3 \
  --epochs 1 \
  --train_examples 64 \
  --val_examples 32 \
  --embed_dim 32 \
  --layers 1 \
  --heads 4 \
  --ff_dim 64 \
  --batch_size 16 \
  --max_length 128 \
  --device cpu \
  --save_models

run_check "tiny:interpretability_on_suite" env MAX_CHECKPOINTS=3 MAX_EVAL_EXAMPLES=32 BATCH_SIZE=16 DEVICE=cpu bash scripts/run_interpretability_on_suite.sh "$ROOT/structural_checkpoint"

run_check "tiny:kernel_sweep" uv run python resonance/resonance_kernel_sweep.py \
  --output_dir "$ROOT/kernel_sweep" \
  --seeds 11 \
  --phase_stds 0.3 \
  --weights 0.1 \
  --frequencies 8 \
  --vocab_size 64 \
  --seq_len 16 \
  --batch_size 2 \
  --embed_dim 32 \
  --n_heads 4

run_check "tiny:generic_probe_equibench" uv run python resonance/structural_task_probe.py \
  --task generic_probe \
  --data_path data/processed/probes/equibench/OJ_VA/train.jsonl \
  --output_dir "$ROOT/equibench_probe" \
  --conditions standard,standard_alibi,standard_deberta_lite,phase_stream_only_normalized \
  --seeds 5 \
  --epochs 1 \
  --train_examples 24 \
  --val_examples 16 \
  --embed_dim 32 \
  --layers 1 \
  --heads 4 \
  --ff_dim 64 \
  --batch_size 4 \
  --max_length 1536 \
  --device cpu

run_check "tiny:build_lab_notebook" uv run python resonance/build_lab_notebook.py \
  --root "$ROOT" \
  --output_dir "$ROOT/lab_notebook"

# 6. Example scripts run from tmp so any demo files stay outside the repo.
EXAMPLE_CWD="$ROOT/example_cwd"
mkdir -p "$EXAMPLE_CWD"
run_check "example:layer_config" bash -lc "cd '$EXAMPLE_CWD' && PYTHONPATH='/Users/ember/dev/restrans/resonance' uv --directory /Users/ember/dev/restrans run python /Users/ember/dev/restrans/resonance/example_layer_config.py"
run_check "example:modern_preset" bash -lc "cd '$EXAMPLE_CWD' && PYTHONPATH='/Users/ember/dev/restrans/resonance' uv --directory /Users/ember/dev/restrans run python /Users/ember/dev/restrans/resonance/example_modern_preset.py"
run_check "example:sae" bash -lc "cd '$EXAMPLE_CWD' && PYTHONPATH='/Users/ember/dev/restrans/resonance' uv --directory /Users/ember/dev/restrans run python /Users/ember/dev/restrans/resonance/example_sae.py"

write_report
echo "Report: $REPORT"

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
