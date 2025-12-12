#!/usr/bin/env python3
"""
TOD v1 Per-City Health Check

Validates training data quality for all 6 cities to identify degenerate models.
Outputs a markdown table showing:
- Row counts per city
- Unique delta classes
- Value distributions
- Degenerate flags (< 1000 rows or < 3 unique classes)

Usage:
    .venv/bin/python scripts/health_check_tod_v1.py
"""

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.db.connection import get_db_session
from models.data.tod_dataset_builder import build_tod_snapshot_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# All 6 Kalshi weather cities
CITIES = ['chicago', 'austin', 'denver', 'los_angeles', 'miami', 'philadelphia']

# Thresholds for flagging degenerate datasets
MIN_ROWS = 1000
MIN_UNIQUE_CLASSES = 3


def run_health_check():
    """Run health check on TOD v1 training data for all cities."""

    logger.info("=" * 80)
    logger.info("TOD v1 HEALTH CHECK - Per-City Training Data Validation")
    logger.info("=" * 80)

    # Use same date range as training
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=700)

    logger.info(f"Date range: {start_date} to {end_date} ({(end_date - start_date).days} days)")
    logger.info(f"Snapshot interval: 15 minutes")
    logger.info("")

    results = []

    with get_db_session() as session:
        for city in CITIES:
            logger.info(f"Building dataset for {city}...")

            try:
                df = build_tod_snapshot_dataset(
                    cities=[city],
                    start_date=start_date,
                    end_date=end_date,
                    session=session,
                    snapshot_interval_min=15,
                    include_forecast_features=True,
                )

                if df.empty:
                    results.append({
                        'city': city,
                        'rows': 0,
                        'n_days': 0,
                        'n_delta_classes': 0,
                        'degenerate': True,
                        'notes': 'No data returned',
                        'value_counts': {},
                    })
                    continue

                # Compute statistics
                n_rows = len(df)
                n_days = df['day'].nunique() if 'day' in df.columns else 0
                n_delta_classes = df['delta'].nunique() if 'delta' in df.columns else 0

                # Value counts for delta
                if 'delta' in df.columns:
                    value_counts = df['delta'].value_counts().head(15).to_dict()
                else:
                    value_counts = {}

                # Determine if degenerate
                is_degenerate = n_rows < MIN_ROWS or n_delta_classes < MIN_UNIQUE_CLASSES

                # Notes
                notes = []
                if n_rows < MIN_ROWS:
                    notes.append(f"Low rows ({n_rows} < {MIN_ROWS})")
                if n_delta_classes < MIN_UNIQUE_CLASSES:
                    notes.append(f"Low classes ({n_delta_classes} < {MIN_UNIQUE_CLASSES})")
                if not notes:
                    notes.append("OK")

                results.append({
                    'city': city,
                    'rows': n_rows,
                    'n_days': n_days,
                    'n_delta_classes': n_delta_classes,
                    'degenerate': is_degenerate,
                    'notes': '; '.join(notes),
                    'value_counts': value_counts,
                })

                logger.info(f"  {city}: {n_rows} rows, {n_days} days, {n_delta_classes} delta classes")

            except Exception as e:
                logger.error(f"  Failed to build dataset for {city}: {e}")
                results.append({
                    'city': city,
                    'rows': 0,
                    'n_days': 0,
                    'n_delta_classes': 0,
                    'degenerate': True,
                    'notes': f'ERROR: {str(e)[:50]}',
                    'value_counts': {},
                })

    # Print markdown table
    print("\n")
    print("=" * 80)
    print("TOD v1 HEALTH CHECK RESULTS")
    print("=" * 80)
    print("")
    print("## Summary Table")
    print("")
    print("| City | Rows | Days | Delta Classes | Degenerate? | Notes |")
    print("|------|------|------|---------------|-------------|-------|")

    for r in results:
        degenerate_str = "YES" if r['degenerate'] else "no"
        print(f"| {r['city']:<14} | {r['rows']:>6,} | {r['n_days']:>4} | {r['n_delta_classes']:>13} | {degenerate_str:^11} | {r['notes']} |")

    print("")

    # Print detailed value counts for each city
    print("## Delta Value Counts (Top 15)")
    print("")

    for r in results:
        if r['value_counts']:
            print(f"### {r['city'].title()}")
            print("```")
            for delta, count in sorted(r['value_counts'].items()):
                pct = count / r['rows'] * 100 if r['rows'] > 0 else 0
                print(f"  delta={delta:>3}: {count:>6,} ({pct:>5.1f}%)")
            print("```")
            print("")

    # Summary
    n_healthy = sum(1 for r in results if not r['degenerate'])
    n_degenerate = sum(1 for r in results if r['degenerate'])

    print("## Summary")
    print("")
    print(f"- **Healthy cities**: {n_healthy}/6")
    print(f"- **Degenerate cities**: {n_degenerate}/6")
    print("")

    if n_degenerate > 0:
        degenerate_cities = [r['city'] for r in results if r['degenerate']]
        print(f"**Degenerate cities needing attention**: {', '.join(degenerate_cities)}")
        print("")
        print("Degenerate cities should NOT be used for live trading until fixed.")
    else:
        print("All cities have healthy training data.")

    print("")
    print("=" * 80)

    return results


if __name__ == '__main__':
    results = run_health_check()

    # Exit with error code if any city is degenerate
    n_degenerate = sum(1 for r in results if r['degenerate'])
    sys.exit(1 if n_degenerate > 0 else 0)
