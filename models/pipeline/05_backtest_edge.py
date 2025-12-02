#!/usr/bin/env python3
"""
Step 5: Backtest edge detection (sanity check).

Uses the ordinal model + cached test parquet + DB candles/settlements.
"""

import argparse
import logging
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_edge import main as backtest_main  # noqa: E402
from scripts.backtest_edge import CITY_CONFIG  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest_edge_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Edge detection backtest (per city)")
    parser.add_argument("--city", required=True, choices=list(CITY_CONFIG.keys()))
    parser.add_argument("--days", type=int, default=60, help="Number of recent days to test")
    parser.add_argument("--threshold", type=float, default=1.5, help="Edge threshold (deg F)")
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Snapshot interval in minutes (multiples of 5)",
    )
    args = parser.parse_args()

    cli_args = [
        "--city",
        args.city,
        "--days",
        str(args.days),
        "--threshold",
        str(args.threshold),
        "--interval",
        str(args.interval),
    ]

    logger.info(f"Invoking backtest_edge with args: {' '.join(cli_args)}")
    sys.argv = ["backtest_edge"] + cli_args
    return backtest_main()


if __name__ == "__main__":
    raise SystemExit(main())
