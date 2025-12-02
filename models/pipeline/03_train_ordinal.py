#!/usr/bin/env python3
"""
Step 3: Train per-city ordinal CatBoost with Optuna (uses cached parquets if present).

Defaults:
- include_market=True
- include_multi_horizon=True
- include_station_city=True (disable with --no-station-city)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

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


def main():
    parser = argparse.ArgumentParser(description="Train ordinal CatBoost with Optuna (per city)")
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
        "--no-station-city",
        action="store_true",
        help="Disable station-city aggregate obs features",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore cached parquets and rebuild datasets",
    )
    args = parser.parse_args()

    cli_args = [
        "--city",
        args.city,
        "--trials",
        str(args.trials),
        "--workers",
        str(args.workers),
        "--cv-splits",
        str(args.cv_splits),
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
