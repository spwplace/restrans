#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  cat <<'EOF'
Run lightweight interpretability audits over checkpoints in a completed suite.

Usage:
  bash scripts/run_interpretability_on_suite.sh resonance/outputs/literature_medium_...

Environment knobs:
  DEVICE=cpu|mps|cuda
  MAX_CHECKPOINTS=12
  MAX_EVAL_EXAMPLES=256
  BATCH_SIZE=32
EOF
  exit 0
fi

ROOT="$1"
OUT="${ROOT}/interpretability_audit"
mkdir -p "$OUT"

mapfile -t checkpoints < <(find "$ROOT" -path "*/checkpoints/*.pt" -type f | sort)
if [[ "${#checkpoints[@]}" -eq 0 ]]; then
  echo "No checkpoints found under $ROOT. Re-run training with SAVE_MODELS=1." >&2
  exit 0
fi

max="${MAX_CHECKPOINTS:-12}"
selected=("${checkpoints[@]:0:$max}")

uv run python resonance/run_interpretability_audit.py \
  "${selected[@]}" \
  --output_dir "$OUT" \
  --device "${DEVICE:-cpu}" \
  --max_eval_examples "${MAX_EVAL_EXAMPLES:-256}" \
  --batch_size "${BATCH_SIZE:-32}"

echo "Wrote $OUT"
