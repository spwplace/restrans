#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=resonance
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Acquire and convert the public research assets for the structural-stream suite.

This intentionally does not fetch licensed AMR/LDC data.  It does stage the
public assets we can use now, including EquiBench.

Environment knobs:
  DATA_DIR=data
  PROFILE=full|core
  BABYLM_FULL=0|1
  INCLUDE_BIG_CODE=0|1
  INCLUDE_CODESEARCHNET=0|1
  INCLUDE_LEANDOJO=0|1
  OPTIONAL_EXPORT_LIMIT=10000
  PROBE_LIMIT=10000

Examples:
  bash scripts/acquire_research_assets.sh
  BABYLM_FULL=1 INCLUDE_BIG_CODE=1 INCLUDE_LEANDOJO=1 bash scripts/acquire_research_assets.sh
EOF
  exit 0
fi

DATA_DIR="${DATA_DIR:-data}"
PROFILE="${PROFILE:-full}"

download_args=(
  --data_dir "$DATA_DIR"
  --profile "$PROFILE"
  --optional_export_limit "${OPTIONAL_EXPORT_LIMIT:-10000}"
)

if [[ "${BABYLM_FULL:-0}" == "1" ]]; then
  download_args+=(--babylm_full)
fi
if [[ "${INCLUDE_CODESEARCHNET:-0}" == "1" ]]; then
  download_args+=(--include_codesearchnet)
fi
if [[ "${INCLUDE_BIG_CODE:-0}" == "1" ]]; then
  download_args+=(--include_big_code)
fi
if [[ "${INCLUDE_LEANDOJO:-0}" == "1" ]]; then
  download_args+=(--include_leandojo)
fi

uv sync
bash scripts/setup_external_deps.sh
uv run python resonance/download_datasets.py "${download_args[@]}"

uv run python resonance/prepare_external_probe_data.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$DATA_DIR/processed/probes" \
  --datasets all \
  --limit "${PROBE_LIMIT:-10000}"

echo "Acquisition complete."
echo "Manifest: $DATA_DIR/dataset_manifest.json"
echo "Generic probes: $DATA_DIR/processed/probes"
echo "Contrastive pairs: $DATA_DIR/processed/contrastive"
