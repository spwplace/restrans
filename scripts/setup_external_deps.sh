#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

REU_UNIF_URL="${REU_UNIF_URL:-https://github.com/emberian/reu_unif.git}"
REU_UNIF_REV="${REU_UNIF_REV:-c0da84c2d8d392031ab252e3617d43a9c9de96d0}"
REU_UNIF_DIR="${REU_UNIF_DIR:-external/reu_unif}"

mkdir -p "$(dirname "$REU_UNIF_DIR")"

if [[ ! -d "$REU_UNIF_DIR/.git" ]]; then
  git clone "$REU_UNIF_URL" "$REU_UNIF_DIR"
fi

git -C "$REU_UNIF_DIR" fetch --all --tags
git -C "$REU_UNIF_DIR" checkout "$REU_UNIF_REV"

echo "reu_unif checked out at $(git -C "$REU_UNIF_DIR" rev-parse HEAD)"
echo "path: $REU_UNIF_DIR"
