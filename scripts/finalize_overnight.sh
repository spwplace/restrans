#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance

LOG="resonance/outputs/finalize_overnight_2026_04_27.log"
mkdir -p resonance/outputs

{
  echo "finalizer started $(date)"
  while pgrep -f "resonance/structural_task_probe.py|resonance/train_corpus_lm.py" >/dev/null; do
    echo "local probes still running $(date)"
    sleep 300
  done

  while ssh -i ~/.ssh/id_aws persvati 'pgrep -af "resonance/structural_task_probe.py|run_overnight_persvati|run_post_persvati|resonance/train_corpus_lm.py" | grep -v pgrep >/dev/null'; do
    echo "remote probes still running $(date)"
    sleep 300
  done

  echo "pulling remote outputs $(date)"
  rsync -avz --partial -e 'ssh -i ~/.ssh/id_aws' \
    persvati:restrans-exp/resonance/outputs/overnight_2026_04_27/ \
    resonance/outputs/overnight_2026_04_27/
  rsync -avz --partial -e 'ssh -i ~/.ssh/id_aws' \
    persvati:restrans-exp/resonance/outputs/lab_notebook_remote/ \
    resonance/outputs/lab_notebook_remote/ || true

  echo "building local lab notebook $(date)"
  uv run python resonance/summarize_experiment_plan.py --root resonance/outputs/experiment_plan || true
  uv run python resonance/build_lab_notebook.py --root resonance/outputs --output_dir resonance/outputs/lab_notebook
  echo "finalizer done $(date)"
} >>"$LOG" 2>&1
