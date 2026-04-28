#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export UV_CACHE_DIR=/tmp/uv-cache
export PYTHONUNBUFFERED=1

INTERVAL_SECONDS="${INTERVAL_SECONDS:-900}"
MAX_ROUNDS="${MAX_ROUNDS:-48}"
LOG_DIR="resonance/outputs"
mkdir -p "$LOG_DIR"

for round in $(seq 1 "$MAX_ROUNDS"); do
  echo "===== watch round $round $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="

  rsync -az --partial -e 'ssh -i ~/.ssh/id_aws' \
    persvati:restrans-exp/resonance/outputs/ \
    resonance/outputs/ || true

  uv run python resonance/build_lab_notebook.py \
    --root resonance/outputs \
    --output_dir resonance/outputs/lab_notebook_hyper_live || true

  remote_active="$(ssh -i ~/.ssh/id_aws persvati 'pgrep -af "run_hyper_persvati|run_hyper_persvati_cpu|structural_task_probe|train_corpus_lm" || true' || true)"
  local_active="$(pgrep -af 'run_hyper_local_mps|structural_task_probe|train_corpus_lm' || true)"
  echo "remote_active:"
  echo "$remote_active"
  echo "local_active:"
  echo "$local_active"

  if [[ -z "$remote_active" && -z "$local_active" ]]; then
    echo "No active experiment processes found; final sync complete."
    break
  fi

  sleep "$INTERVAL_SECONDS"
done
