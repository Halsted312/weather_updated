#!/usr/bin/env python3
"""
Rebuild feature_completeness.csv artifacts for already-trained models.

This script generates the feature completeness audit CSVs without re-training,
using the date ranges encoded in the walk-forward window directory names.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import re
from pathlib import Path
from datetime import datetime, timedelta, date
from ml.dataset import build_training_dataset

WIN_RE = re.compile(r"win_(\d{8})_(\d{8})")


def parse_dates(win_dir: Path):
    """
    Parse train/test dates from window directory name.

    Format: win_YYYYMMDD_YYYYMMDD where first is train_start, second is test_end.
    With 90→7 windows: train period is 90 days, test period is 7 days.
    """
    m = WIN_RE.search(win_dir.name)
    if not m:
        return None

    train_start = datetime.strptime(m.group(1), "%Y%m%d").date()
    test_end = datetime.strptime(m.group(2), "%Y%m%d").date()

    # With 90→7 windows: train period is 90 days ending 1 day before test
    # test period is last 7 days
    test_start = test_end - timedelta(days=6)
    train_end = test_start - timedelta(days=1)

    return train_start, train_end, test_start, test_end


def rebuild(city: str, bracket: str, model_root: Path):
    """
    Rebuild feature completeness CSVs for all walk-forward windows.
    """
    print(f"\nRebuilding feature completeness for {city}/{bracket}")
    print(f"Model directory: {model_root}")

    win_dirs = sorted(model_root.glob("win_*"))
    if not win_dirs:
        print(f"No win_* directories found in {model_root}")
        return

    for win_dir in win_dirs:
        parsed = parse_dates(win_dir)
        if not parsed:
            print(f"⚠️  Could not parse dates from {win_dir.name}")
            continue

        train_start, train_end, test_start, test_end = parsed
        csv_path = win_dir / "feature_completeness.csv"

        print(f"\nProcessing {win_dir.name}:")
        print(f"  Train: {train_start} to {train_end} (90 days)")
        print(f"  Test:  {test_start} to {test_end} (7 days)")

        # Remove existing CSV to start fresh
        if csv_path.exists():
            csv_path.unlink()
            print(f"  Removed existing {csv_path.name}")

        try:
            # Rebuild TRAIN audit
            print(f"  Building train dataset audit...")
            _, _, _, _ = build_training_dataset(
                city=city,
                start_date=train_start,
                end_date=train_end,
                bracket_type=bracket,
                feature_set="elasticnet_rich",
                completeness_csv_path=csv_path,
                split="train"
            )

            # Rebuild TEST audit (append to same CSV)
            print(f"  Building test dataset audit...")
            _, _, _, _ = build_training_dataset(
                city=city,
                start_date=test_start,
                end_date=test_end,
                bracket_type=bracket,
                feature_set="elasticnet_rich",
                completeness_csv_path=csv_path,
                split="test"
            )

            print(f"  ✓ Wrote {csv_path}")

            # Display the audit results
            import pandas as pd
            audit_df = pd.read_csv(csv_path)
            print(f"  Audit summary:")
            for _, row in audit_df.iterrows():
                print(f"    {row['split']}: {row['rows_before']} → {row['rows_after_optional_impute']} "
                      f"({row['pct_kept_final']}, YES={row.get('pct_yes', 'N/A')})")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    print(f"\n✓ Completed rebuilding for {city}/{bracket}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Rebuild feature_completeness.csv for trained models"
    )
    ap.add_argument("--city", required=True, help="City name (e.g., chicago)")
    ap.add_argument(
        "--bracket",
        required=True,
        choices=["between", "greater", "less"],
        help="Bracket type"
    )
    ap.add_argument(
        "--models-dir",
        required=True,
        help="Path to models directory (e.g., models/production/chicago/less)"
    )

    args = ap.parse_args()
    rebuild(args.city, args.bracket, Path(args.models_dir))