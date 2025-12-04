#!/bin/bash
# Full backfill for all 6 cities, both station and city location types
# Estimated time: 4-6 hours
# Run with: nohup ./scripts/run_full_backfill.sh > logs/backfill_$(date +%Y%m%d).log 2>&1 &

set -e

echo "Starting full backfill at $(date)"
echo "=============================================="

# Activate virtual environment
source .venv/bin/activate

# Run backfill for all cities with both location types
python scripts/backfill_vc_historical_forecasts.py \
    --start-date 2022-12-23 \
    --lead-days 0,1,2,3 \
    --initial-delay 0.005 \
    --target-delay 0.005 \
    --max-delay 0.02

echo "=============================================="
echo "Backfill completed at $(date)"
