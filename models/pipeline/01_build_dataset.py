#!/usr/bin/env python3
"""
Step 1: Build per-city market-clock dataset with full features.

- 5-minute snapshots, market-clock window
- include_multi_horizon=True
- include_market=True
- include_station_city=True
- include_meteo=True

Outputs:
  models/saved/{city}/train_data_full.parquet
  models/saved/{city}/test_data_full.parquet
"""

import argparse
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.data.loader import get_available_date_range  # noqa: E402
from scripts.train_city_ordinal_optuna import (  # noqa: E402
    VALID_CITIES,
    build_dataset_parallel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_dataset_pipeline")


def parse_date(val: str) -> date:
    return datetime.strptime(val, "%Y-%m-%d").date()


def split_train_test(df: pd.DataFrame, holdout_pct: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split by unique days (last holdout_pct go to test)."""
    days = pd.to_datetime(df["day"]).dt.date.unique()
    days_sorted = sorted(days)
    n_days = len(days_sorted)
    n_holdout = max(1, int(n_days * holdout_pct))

    test_days = set(days_sorted[-n_holdout:])
    train_days = set(days_sorted[:-n_holdout]) if n_days > n_holdout else set()

    df_train = df[df["day"].isin(train_days)].copy()
    df_test = df[df["day"].isin(test_days)].copy()
    return df_train, df_test


def main():
    default_workers = max(1, (os.cpu_count() or 24) - 4)

    parser = argparse.ArgumentParser(description="Build dataset (market-clock, full features)")
    parser.add_argument("--city", required=True, choices=VALID_CITIES, help="City to process")
    parser.add_argument(
        "--start",
        type=parse_date,
        help="Start date (YYYY-MM-DD). Default: min available in DB",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        help="End date (YYYY-MM-DD). Default: max available in DB",
    )
    parser.add_argument(
        "--holdout-pct",
        type=float,
        default=0.20,
        help="Test holdout percentage by days (default 0.20)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help="Parallel workers (default: min(32, cpu_count))",
    )
    parser.add_argument(
        "--no-station-city",
        action="store_true",
        help="Disable station-city aggregate obs features",
    )
    args = parser.parse_args()

    city = args.city
    include_station_city = not args.no_station_city

    output_dir = Path(f"models/saved/{city}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_data_full.parquet"
    test_path = output_dir / "test_data_full.parquet"

    # Determine date range
    if args.start and args.end:
        start_date, end_date = args.start, args.end
    else:
        from src.db import get_db_session

        with get_db_session() as session:
            min_date, max_date = get_available_date_range(session, city)
        if min_date is None or max_date is None:
            raise SystemExit(f"No data available for {city}")
        start_date = args.start or min_date
        end_date = args.end or max_date

    logger.info(f"Building dataset for {city}: {start_date} to {end_date}")
    logger.info(f"Workers: {args.workers}, include_station_city={include_station_city}")

    df_full = build_dataset_parallel(
        city=city,
        start_date=start_date,
        end_date=end_date,
        n_workers=args.workers,
        include_station_city=include_station_city,
    )

    logger.info(f"Built {len(df_full):,} rows ({df_full['day'].nunique()} days, {len(df_full.columns)} cols)")

    df_train, df_test = split_train_test(df_full, args.holdout_pct)
    logger.info(f"Train: {len(df_train):,} rows ({df_train['day'].nunique()} days)")
    logger.info(f"Test:  {len(df_test):,} rows ({df_test['day'].nunique()} days)")

    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)
    logger.info(f"Saved train to {train_path}")
    logger.info(f"Saved test  to {test_path}")


if __name__ == "__main__":
    raise SystemExit(main())
