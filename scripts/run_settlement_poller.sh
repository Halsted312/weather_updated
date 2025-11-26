#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

source "$REPO/.env"

export PYTHONPATH="$REPO"

python scripts/poll_settlements.py \
  --cities all \
  --days-back 3 \
  --refresh-cf6 \
  --loop \
  --interval-seconds 1800
