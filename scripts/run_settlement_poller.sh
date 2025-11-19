#!/usr/bin/env bash
set -euo pipefail

REPO="/home/halsted/Documents/python/kalshi_weather"
cd "$REPO"

source "$REPO/.env"

export PYTHONPATH="$REPO"

python scripts/poll_settlements.py \
  --cities all \
  --days-back 3 \
  --refresh-cf6 \
  --loop \
  --interval-seconds 1800
