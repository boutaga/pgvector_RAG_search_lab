#!/usr/bin/env bash
# Tear the whole stack down, including volumes.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo "[down] removing containers + volumes ..."
dc down -v
echo "[down] done."
