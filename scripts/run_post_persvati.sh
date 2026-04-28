#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/restrans-exp"
. .venv-exp/bin/activate
export PYTHONPATH=resonance
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export OMP_NUM_THREADS=8

LOG="resonance/outputs/post_persvati_2026_04_27.log"
mkdir -p resonance/outputs

{
  echo "post-poll started $(date)"
  while pgrep -af "run_overnight_persvati|resonance/structural_task_probe.py" | grep -v "run_post_persvati" | grep -v pgrep >/dev/null; do
    echo "waiting for overnight probes $(date)"
    sleep 300
  done

  echo "starting corpus LM $(date)"
  python resonance/train_corpus_lm.py \
    --output_dir resonance/outputs/corpus_lm_babylm_rocm_2026_04_27 \
    --device cuda \
    --conditions standard,resonance_full_normalized,phase_stream_only_normalized \
    --train_docs 200000 \
    --val_docs 20000 \
    --train_examples 12000 \
    --val_examples 1500 \
    --vocab_size 12000 \
    --seq_len 96 \
    --epochs 3 \
    --embed_dim 192 \
    --layers 4 \
    --heads 4 \
    --ff_dim 768 \
    --batch_size 24 \
    --gradient_accumulation 2 \
    --log_interval 100

  python resonance/build_lab_notebook.py --root resonance/outputs --output_dir resonance/outputs/lab_notebook_remote || true
  echo "post-poll done $(date)"
} >>"$LOG" 2>&1
