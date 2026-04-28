#!/usr/bin/env bash
set -euo pipefail

# Focused CPU LM runner for persvati.
# Trains phase_dynamic_qk_film against strong baselines on BabyLM and TinyStories.
# Intended to run while the GPU signal matrix occupies ROCm.
#
# Environment:
#   ROOT=output/root/path
#   DEVICE=cpu
#   SEEDS="123 456"
#   EPOCHS=3

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

DEVICE="${DEVICE:-cpu}"
ROOT="${ROOT:-resonance/outputs/cpu_lm_focused_$(date -u +%Y_%m_%d_%H%M%S)}"
mkdir -p "$ROOT"

SEEDS="${SEEDS:-123 456}"
EPOCHS="${EPOCHS:-3}"

# Shared modest-lm hparams for CPU tractability
SHARED_ARGS=(
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --embed_dim 128
  --layers 3
  --heads 4
  --ff_dim 512
  --seq_len 96
  --batch_size 32
  --gradient_accumulation 1
  --lr 3e-4
  --weight_decay 0.01
  --dropout 0.1
  --vocab_size 10000
  --log_interval 100
)

CONDITIONS="${CONDITIONS:-standard,standard_alibi,standard_iso_deberta_lite,phase_dynamic_qk_film_normalized}"

# Ensure TinyStories JSONL exists
TINY_PROCESSED="data/processed/tinystories"
if [[ ! -f "$TINY_PROCESSED/train.jsonl" ]]; then
  echo "Converting TinyStories HF dataset to JSONL..."
  mkdir -p "$TINY_PROCESSED"
  source .venv-exp/bin/activate 2>/dev/null || true
  python3 - <<PY
import json
from pathlib import Path
from datasets import load_from_disk
root = Path("$TINY_PROCESSED")
ds = load_from_disk("data/hf_datasets/tinystories")
for split in ["train", "validation"]:
    out = root / f"{split}.jsonl"
    with out.open("w") as f:
        for row in ds[split]:
            f.write(json.dumps({"text": row["text"]}) + "\n")
    print(f"Wrote {out}")
PY
fi

run_lm() {
  local name="$1"
  local train_path="$2"
  local val_path="$3"
  local train_ex="$4"
  local val_ex="$5"
  local seed="$6"

  local outdir="$ROOT/${name}_seed${seed}"
  mkdir -p "$outdir"

  echo "===== $name seed=$seed ====="
  uv run python resonance/train_corpus_lm.py \
    --output_dir "$outdir" \
    --train_path "$train_path" \
    --val_path "$val_path" \
    --conditions "$CONDITIONS" \
    --seed "$seed" \
    --train_examples "$train_ex" \
    --val_examples "$val_ex" \
    "${SHARED_ARGS[@]}" 2>&1 | tee "$outdir/run.log"
}

for seed in $SEEDS; do
  # BabyLM strict small
  run_lm babylm \
    data/processed/babylm_strict_small/train.jsonl \
    data/processed/babylm_strict_small/validation.jsonl \
    6000 800 "$seed"

  # TinyStories subset
  run_lm tinystories \
    data/processed/tinystories/train.jsonl \
    data/processed/tinystories/validation.jsonl \
    8000 1000 "$seed"
done

echo "Wrote results to $ROOT"
