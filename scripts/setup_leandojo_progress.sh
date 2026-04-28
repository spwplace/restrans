#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Set up LeanDojo-v2 in an isolated uv environment and attempt a LeanProgress
sample export.

This is intentionally separate from the main project environment because
LeanDojo-v2 has a large theorem-proving / HF / Lightning dependency stack.

Required for real tracing:
  export GITHUB_ACCESS_TOKEN=...

Optional:
  export HF_TOKEN=...

Environment knobs:
  LEANDOJO_DIR=data/source_repos/LeanDojo-v2
  LEANDOJO_VENV=.venv-leandojo
  OUTPUT=data/processed/leanprogress/sample_leanprogress_dataset.jsonl
  INSTALL=1
  EXPORT_SAMPLE=1
EOF
  exit 0
fi

LEANDOJO_DIR="${LEANDOJO_DIR:-data/source_repos/LeanDojo-v2}"
LEANDOJO_VENV="${LEANDOJO_VENV:-.venv-leandojo}"
OUTPUT="${OUTPUT:-data/processed/leanprogress/sample_leanprogress_dataset.jsonl}"

if [[ ! -d "$LEANDOJO_DIR/.git" ]]; then
  mkdir -p "$(dirname "$LEANDOJO_DIR")"
  git clone --depth 1 https://github.com/lean-dojo/LeanDojo-v2.git "$LEANDOJO_DIR"
fi

if [[ "${INSTALL:-1}" == "1" ]]; then
  uv venv "$LEANDOJO_VENV" --python 3.11
  uv pip install --python "$LEANDOJO_VENV/bin/python" -e "$LEANDOJO_DIR"
  uv pip install --python "$LEANDOJO_VENV/bin/python" git+https://github.com/stanford-centaur/PyPantograph
fi

if [[ "${EXPORT_SAMPLE:-1}" == "1" ]]; then
  mkdir -p "$(dirname "$OUTPUT")"
  "$LEANDOJO_VENV/bin/python" -m lean_dojo_v2.lean_progress.create_sample_dataset \
    --output "$OUTPUT"
  echo "Wrote $OUTPUT"
fi
