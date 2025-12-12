#!/usr/bin/env python3
"""Train models on RECENT data only (rolling window approach).

This addresses non-stationarity by only using last N months of data.

Usage:
    python scripts/train_recent_data_only.py --city miami --months 6 --threshold 1.5
"""

import argparse
import sys
from pathlib import Path

# Your imports here - reuse existing EdgeClassifier logic but filter to recent data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', required=True)
    parser.add_argument('--months', type=int, default=6, help="Use last N months")
    parser.add_argument('--threshold', type=float, default=1.5)
    args = parser.parse_args()

    print(f"Training on last {args.months} months only...")
    print(f"This tests if recent data is more predictive than full historical data.")

    # Filter edge data to recent months
    import pandas as pd
    cache_path = Path(f"models/saved/{args.city}/edge_training_data_realistic.parquet")
    df = pd.read_parquet(cache_path)

    df['date'] = pd.to_datetime(df['day'])
    cutoff = df['date'].max() - pd.DateOffset(months=args.months)
    df_recent = df[df['date'] >= cutoff].copy()

    # Drop the temporary 'date' column (not a feature)
    df_recent = df_recent.drop(columns=['date'])

    print(f"Full data: {len(df):,} rows")
    print(f"Recent {args.months} months: {len(df_recent):,} rows")

    # Temporarily backup original cache and replace with recent data
    cache_path = Path(f"models/saved/{args.city}/edge_training_data_realistic.parquet")
    backup_path = Path(f"models/saved/{args.city}/edge_training_data_realistic.backup.parquet")

    # Backup original
    if cache_path.exists():
        import shutil
        shutil.copy(cache_path, backup_path)
        print(f"Backed up original to: {backup_path}")

    # Write recent data as cache
    df_recent.to_parquet(cache_path, index=False)
    print(f"Wrote recent data to cache: {cache_path}")

    # Train EdgeClassifier (will use the recent-only cache)
    print(f"\nTraining EdgeClassifier on recent {args.months} months...")
    import subprocess
    result = subprocess.run([
        'python', 'scripts/train_edge_classifier.py',
        '--city', args.city,
        '--threshold', str(args.threshold),
        '--sample-rate', '4',
        '--trials', '80',
        '--optuna-metric', 'auc',
        '--workers', '12',
    ])

    # Restore original cache
    if backup_path.exists():
        shutil.copy(backup_path, cache_path)
        print(f"\nRestored original cache from backup")
        backup_path.unlink()  # Remove backup

    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
