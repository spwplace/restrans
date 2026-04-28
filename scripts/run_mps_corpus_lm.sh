#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export PYTHONPATH=resonance

uv run python resonance/train_corpus_lm.py \
  --output_dir resonance/outputs/corpus_lm_babylm_mps_2026_04_27 \
  --device mps \
  --conditions standard,resonance_full_normalized,phase_stream_only_normalized \
  --train_docs 50000 \
  --val_docs 10000 \
  --train_examples 8000 \
  --val_examples 1000 \
  --vocab_size 10000 \
  --seq_len 96 \
  --epochs 2 \
  --embed_dim 192 \
  --layers 4 \
  --heads 4 \
  --ff_dim 768 \
  --batch_size 32 \
  --gradient_accumulation 2 \
  --log_interval 50
