#!/usr/bin/env python3
"""
Step 4: Train edge classifier (CatBoost + Optuna) for a city.

Requires:
- models/saved/{city}/ordinal_catboost_optuna.pkl
- models/saved/{city}/train_data_full.parquet + test_data_full.parquet
- Kalshi candles + settlements in DB for the covered dates
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_edge_classifier import main as edge_main  # noqa: E402
from scripts.train_edge_classifier import CITY_CONFIG  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_edge_classifier_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Train edge classifier (per city)")
    parser.add_argument("--city", required=True, choices=list(CITY_CONFIG.keys()))
    parser.add_argument("--trials", type=int, default=80, help="Optuna trials")
    parser.add_argument("--workers", type=int, default=12, help="Optuna parallel jobs")
    parser.add_argument("--threshold", type=float, default=1.5, help="Edge threshold (deg F)")
    parser.add_argument("--sample-rate", type=int, default=6, help="Sample every Nth snapshot")
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Edge classifier decision threshold (probability)",
    )
    parser.add_argument(
        "--optuna-metric",
        type=str,
        default="filtered_precision",
        choices=["auc", "filtered_precision", "f1", "mean_pnl", "sharpe"],
        help="Optuna objective metric",
    )
    parser.add_argument(
        "--min-trades-for-metric",
        type=int,
        default=10,
        help="Minimum trades when optimizing precision/F1",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of CV splits for DayGroupedTimeSeriesSplit (default: 5)",
    )
    parser.add_argument(
        "--no-threshold-tuning",
        action="store_true",
        help="Disable validation-based tuning of decision threshold",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate edge data even if cached",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        help="Limit to last N days (debug/testing)",
    )
    args = parser.parse_args()

    cli_args = [
        "--city",
        args.city,
        "--trials",
        str(args.trials),
        "--workers",
        str(args.workers),
        "--threshold",
        str(args.threshold),
        "--sample-rate",
        str(args.sample_rate),
        "--decision-threshold",
        str(args.decision_threshold),
        "--optuna-metric",
        args.optuna_metric,
        "--min-trades-for-metric",
        str(args.min_trades_for_metric),
        "--cv-splits",
        str(args.cv_splits),
    ]
    if args.regenerate:
        cli_args.append("--regenerate")
    if args.max_days:
        cli_args.extend(["--max-days", str(args.max_days)])
    if args.no_threshold_tuning:
        cli_args.append("--no-threshold-tuning")

    logger.info(f"Invoking train_edge_classifier with args: {' '.join(cli_args)}")
    sys.argv = ["train_edge_classifier"] + cli_args
    return edge_main()


if __name__ == "__main__":
    raise SystemExit(main())
