#!/usr/bin/env python3
"""
Step 2: Delta range sweep (Optuna) per city.

Loads train/test parquets, concatenates, and runs scripts/optuna_delta_range_sweep.py
to find the best asymmetric delta range. Saves results under models/saved/{city}/delta_range_sweep/.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.core.optuna_delta_range_sweep import run_optuna_sweep  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("delta_sweep_pipeline")

VALID_CITIES = ["chicago", "austin", "denver", "los_angeles", "miami", "philadelphia"]


def load_parquets(city: str) -> pd.DataFrame:
    base = Path(f"models/saved/{city}")
    train_path = base / "train_data_full.parquet"
    test_path = base / "test_data_full.parquet"

    dfs = []
    if train_path.exists():
        dfs.append(pd.read_parquet(train_path))
    if test_path.exists():
        dfs.append(pd.read_parquet(test_path))

    if not dfs:
        raise FileNotFoundError(f"Could not find train/test parquets for {city} in {base}")

    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Delta range sweep (Optuna)")
    parser.add_argument("--city", required=True, choices=VALID_CITIES, help="City to process")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument(
        "--test-days",
        type=int,
        default=66,
        help="Test days for time-based split (default 66 ~20% of a 331d span)",
    )
    args = parser.parse_args()

    city = args.city
    output_dir = Path(f"models/saved/{city}/delta_range_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading cached train/test parquets for {city}")
    df = load_parquets(city)
    logger.info(f"Data: {len(df):,} rows, {df['day'].nunique()} days, columns={len(df.columns)}")

    results = run_optuna_sweep(
        df=df,
        test_days=args.test_days,
        n_trials=args.trials,
        output_dir=output_dir,
    )

    logger.info(f"Best delta range: {results['best_range']}, within_2={results['best_within_2']:.1%}")
    logger.info(f"Best model + summary saved under {output_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
