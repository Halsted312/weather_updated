#!/usr/bin/env python3
"""
Step 3: Train per-city ordinal CatBoost with Optuna (uses cached parquets if present).

This script automatically handles train/test splitting:
- If train_data_full.parquet and test_data_full.parquet exist, use them
- If only full.parquet exists, auto-split with 80/20 ratio using day-grouped splits
- Day-grouped splits ensure ALL snapshots from a given day stay in same fold (no lookahead)

Defaults:
- include_market=True
- include_multi_horizon=True
- include_station_city=True (disable with --no-station-city)

Usage:
    # Fast machine with pre-built full.parquet
    python models/pipeline/03_train_ordinal.py --city chicago --trials 100 --workers 16 \
        --cache-dir data/training_cache

    # Slow machine (builds from DB if no cache)
    python models/pipeline/03_train_ordinal.py --city chicago --trials 100 --force-rebuild
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from models.data.splits import train_test_split_by_ratio
from scripts.train_city_ordinal_optuna import (  # noqa: E402
    VALID_CITIES,
    main as train_city_main,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_ordinal_pipeline")


def ensure_train_test_split(cache_dir: Path, city: str, test_ratio: float = 0.2) -> bool:
    """
    Ensure train/test split files exist. If only full.parquet exists, auto-split it.

    Uses day-grouped splitting to prevent lookahead leakage - all snapshots from
    a given day go to the same fold (train or test).

    Args:
        cache_dir: Directory containing parquet files
        city: City identifier
        test_ratio: Fraction of days for test set (default 0.2 = 20%)

    Returns:
        True if split files exist (or were created), False if no data available
    """
    city_dir = cache_dir / city
    train_path = city_dir / "train_data_full.parquet"
    test_path = city_dir / "test_data_full.parquet"
    full_path = city_dir / "full.parquet"

    # Check if split files already exist
    if train_path.exists() and test_path.exists():
        logger.info(f"Found existing split files in {city_dir}")
        return True

    # Check if full.parquet exists to split
    if not full_path.exists():
        logger.warning(f"No full.parquet found at {full_path}")
        return False

    # Auto-split full.parquet
    logger.info(f"Auto-splitting {full_path} with {test_ratio:.0%} test ratio...")

    df = pd.read_parquet(full_path)
    logger.info(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Determine date column (prefer 'day', fallback to 'event_date')
    date_col = "day"
    if date_col not in df.columns:
        if "event_date" in df.columns:
            df["day"] = pd.to_datetime(df["event_date"]).dt.date
            logger.info("  Created 'day' column from 'event_date'")
        else:
            logger.error("  No 'day' or 'event_date' column found!")
            return False

    # Get unique days for logging
    unique_days = sorted(df["day"].unique())
    n_days = len(unique_days)
    n_test_days = max(1, int(n_days * test_ratio))

    logger.info(f"  Total days: {n_days}")
    logger.info(f"  Train days: {n_days - n_test_days} ({unique_days[0]} to {unique_days[n_days - n_test_days - 1]})")
    logger.info(f"  Test days: {n_test_days} ({unique_days[n_days - n_test_days]} to {unique_days[-1]})")

    # Split using day-grouped function (prevents lookahead)
    df_train, df_test = train_test_split_by_ratio(df, test_ratio=test_ratio, date_col="day")

    logger.info(f"  Train set: {len(df_train):,} rows ({df_train['day'].nunique()} days)")
    logger.info(f"  Test set: {len(df_test):,} rows ({df_test['day'].nunique()} days)")

    # Save split files
    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)

    logger.info(f"  Saved: {train_path}")
    logger.info(f"  Saved: {test_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train ordinal CatBoost with Optuna (per city)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast machine with pre-built parquets
  python models/pipeline/03_train_ordinal.py --city chicago --trials 100 \\
      --cache-dir data/training_cache --workers 16

  # Slow machine (rebuild from DB)
  python models/pipeline/03_train_ordinal.py --city chicago --trials 100 --force-rebuild
        """,
    )
    parser.add_argument("--city", required=True, choices=VALID_CITIES, help="City to train")
    parser.add_argument("--trials", type=int, default=80, help="Optuna trials")
    parser.add_argument("--cv-splits", type=int, default=4, help="CV splits for Optuna")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 24) - 4),
        help="Parallel workers for dataset build (if rebuild needed)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="models/saved",
        help="Directory containing cached parquets (default: models/saved)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of days for test set when auto-splitting (default: 0.2)",
    )
    parser.add_argument(
        "--no-station-city",
        action="store_true",
        help="Disable station-city aggregate obs features",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore cached parquets and rebuild datasets from DB",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    # Step 1: Ensure train/test split exists (auto-split if needed)
    if not args.force_rebuild:
        logger.info(f"Checking for cached data in {cache_dir / args.city}...")
        if ensure_train_test_split(cache_dir, args.city, args.test_ratio):
            logger.info("Train/test split ready.")
        else:
            logger.warning("No cached data found. Will attempt to build from DB.")

    # Step 2: Build CLI args for underlying training script
    cli_args = [
        "--city",
        args.city,
        "--trials",
        str(args.trials),
        "--workers",
        str(args.workers),
        "--cv-splits",
        str(args.cv_splits),
        "--cache-dir",
        str(cache_dir),
    ]

    if not args.force_rebuild:
        cli_args.append("--use-cached")
    if args.no_station_city:
        cli_args.append("--no-station-city")

    logger.info(f"Invoking train_city_ordinal_optuna with args: {' '.join(cli_args)}")
    sys.argv = ["train_city_ordinal_optuna"] + cli_args
    return train_city_main()


if __name__ == "__main__":
    raise SystemExit(main())
