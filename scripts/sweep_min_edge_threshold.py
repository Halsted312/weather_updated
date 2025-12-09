#!/usr/bin/env python3
"""Sweep min edge thresholds to find optimal P&L/Sharpe tradeoff.

This script evaluates different edge magnitude thresholds (in °F) against
P&L and Sharpe metrics to find the optimal threshold for each city.

The edge system has two thresholds:
1. Edge magnitude threshold (°F) - "How much model-market disagreement needed?"
   This is what we're sweeping here.
2. Classifier probability threshold (0-1) - tuned by Optuna in EdgeClassifier

Usage:
    # Basic sweep with default thresholds
    PYTHONPATH=. python scripts/sweep_min_edge_threshold.py --city austin

    # Custom thresholds
    PYTHONPATH=. python scripts/sweep_min_edge_threshold.py \\
        --city austin \\
        --thresholds 0.5,0.75,1.0,1.25,1.5,2.0,2.5

    # Regenerate edge data with low threshold first
    PYTHONPATH=. python scripts/train_edge_classifier.py \\
        --city austin --threshold 0.25 --regenerate
    PYTHONPATH=. python scripts/sweep_min_edge_threshold.py --city austin
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import sweep thresholds from config (0.5 to 11.0 by 0.5)
from config.edge_thresholds import SWEEP_THRESHOLDS

DEFAULT_THRESHOLDS = SWEEP_THRESHOLDS


def compute_metrics(df: pd.DataFrame, threshold: float) -> Dict:
    """Compute P&L metrics for a given edge threshold.

    Args:
        df: DataFrame with edge data (must have 'edge', 'pnl' columns)
        threshold: Min edge threshold in °F

    Returns:
        Dict with n_trades, mean_pnl, sharpe, hit_rate, total_pnl
    """
    # Filter to rows that pass threshold AND have valid P&L
    mask = (np.abs(df["edge"]) >= threshold) & (df["pnl"].notna())
    df_filtered = df[mask]

    n_trades = len(df_filtered)

    if n_trades == 0:
        return {
            "threshold": threshold,
            "n_trades": 0,
            "mean_pnl": np.nan,
            "std_pnl": np.nan,
            "sharpe": np.nan,
            "hit_rate": np.nan,
            "total_pnl": np.nan,
        }

    pnl_values = df_filtered["pnl"].values
    mean_pnl = float(np.mean(pnl_values))
    std_pnl = float(np.std(pnl_values))

    # Sharpe = mean / std (annualization not meaningful here)
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else np.nan

    # Hit rate = fraction of profitable trades
    hit_rate = float((pnl_values > 0).sum() / n_trades)

    # Total P&L
    total_pnl = float(np.sum(pnl_values))

    return {
        "threshold": threshold,
        "n_trades": n_trades,
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "total_pnl": total_pnl,
    }


def format_results_table(results: List[Dict]) -> str:
    """Format results as a nice ASCII table."""
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("MIN EDGE THRESHOLD SWEEP RESULTS")
    lines.append("=" * 80)
    lines.append("")

    # Header
    header = f"{'Threshold':>10} {'N Trades':>10} {'Mean PnL':>12} {'Sharpe':>10} {'Hit Rate':>10} {'Total PnL':>12}"
    lines.append(header)
    lines.append("-" * 80)

    # Data rows
    for r in results:
        if r["n_trades"] == 0:
            lines.append(f"{r['threshold']:>10.2f} {'---':>10} {'---':>12} {'---':>10} {'---':>10} {'---':>12}")
        else:
            lines.append(
                f"{r['threshold']:>10.2f} "
                f"{r['n_trades']:>10,} "
                f"${r['mean_pnl']:>11.4f} "
                f"{r['sharpe']:>10.3f} "
                f"{r['hit_rate']:>9.1%} "
                f"${r['total_pnl']:>11.2f}"
            )

    lines.append("-" * 80)
    lines.append("")

    return "\n".join(lines)


def find_optimal_threshold(
    results: List[Dict],
    metric: str = "sharpe",
    min_trades: int = 500,
) -> Tuple[float, Dict]:
    """Find the optimal threshold given constraints.

    Args:
        results: List of metrics dicts
        metric: Which metric to maximize ('sharpe', 'mean_pnl')
        min_trades: Minimum number of trades required

    Returns:
        (optimal_threshold, metrics_dict)
    """
    valid_results = [r for r in results if r["n_trades"] >= min_trades]

    if not valid_results:
        logger.warning(f"No thresholds have >= {min_trades} trades")
        return (np.nan, {})

    if metric == "sharpe":
        best = max(valid_results, key=lambda r: r["sharpe"] if not np.isnan(r["sharpe"]) else -np.inf)
    elif metric == "mean_pnl":
        best = max(valid_results, key=lambda r: r["mean_pnl"] if not np.isnan(r["mean_pnl"]) else -np.inf)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return (best["threshold"], best)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep min edge thresholds for P&L/Sharpe optimization"
    )
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        choices=["austin", "chicago", "denver", "los_angeles", "miami", "philadelphia"],
        help="City to analyze",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Comma-separated list of thresholds (e.g., 0.5,1.0,1.5,2.0)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=500,
        help="Minimum trades to consider a threshold valid (default: 500)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="sharpe",
        choices=["sharpe", "mean_pnl"],
        help="Metric to optimize (default: sharpe)",
    )
    parser.add_argument(
        "--pnl-mode",
        type=str,
        default="realistic",
        choices=["realistic", "simplified", "realistic_multi_mfp"],
        help="P&L mode used in edge data (default: realistic)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional: save results to JSON file",
    )
    args = parser.parse_args()

    # Parse thresholds
    if args.thresholds:
        thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
    else:
        thresholds = DEFAULT_THRESHOLDS

    thresholds = sorted(thresholds)

    print("=" * 60)
    print(f"MIN EDGE THRESHOLD SWEEP: {args.city.upper()}")
    print("=" * 60)
    print(f"Thresholds to test: {thresholds}")
    print(f"Min trades required: {args.min_trades}")
    print(f"Optimization metric: {args.metric}")
    print()

    # Load cached edge data
    cache_path = Path(f"models/saved/{args.city}/edge_training_data_{args.pnl_mode}.parquet")

    if not cache_path.exists():
        logger.error(f"Edge data not found: {cache_path}")
        logger.error("Generate it first with:")
        logger.error(f"  PYTHONPATH=. python scripts/train_edge_classifier.py \\")
        logger.error(f"      --city {args.city} --threshold 0.25 --regenerate")
        return 1

    logger.info(f"Loading edge data from {cache_path}")
    df = pd.read_parquet(cache_path)
    logger.info(f"Loaded {len(df):,} rows")

    # Check edge range
    edge_min = df["edge"].min()
    edge_max = df["edge"].max()
    edge_abs_min = np.abs(df["edge"]).min()
    logger.info(f"Edge range: [{edge_min:.2f}, {edge_max:.2f}]")
    logger.info(f"Min absolute edge in data: {edge_abs_min:.2f}")

    # Check how many rows have valid P&L
    n_valid_pnl = df["pnl"].notna().sum()
    logger.info(f"Rows with valid P&L: {n_valid_pnl:,} ({100*n_valid_pnl/len(df):.1f}%)")

    # Warn if min threshold is below data's minimum
    if min(thresholds) < edge_abs_min:
        logger.warning(
            f"Lowest threshold ({min(thresholds)}) is below data's min edge ({edge_abs_min:.2f}). "
            f"Consider regenerating with --threshold {min(thresholds) * 0.5:.2f}"
        )

    # Sweep thresholds
    results = []
    for threshold in thresholds:
        metrics = compute_metrics(df, threshold)
        results.append(metrics)

    # Print results table
    print(format_results_table(results))

    # Find optimal threshold
    optimal_threshold, optimal_metrics = find_optimal_threshold(
        results, metric=args.metric, min_trades=args.min_trades
    )

    if not np.isnan(optimal_threshold):
        print(f"OPTIMAL THRESHOLD (maximize {args.metric}, min {args.min_trades} trades): {optimal_threshold:.2f}°F")
        print(f"  → N trades: {optimal_metrics['n_trades']:,}")
        print(f"  → Mean P&L: ${optimal_metrics['mean_pnl']:.4f}")
        print(f"  → Sharpe: {optimal_metrics['sharpe']:.3f}")
        print(f"  → Hit rate: {optimal_metrics['hit_rate']:.1%}")
        print(f"  → Total P&L: ${optimal_metrics['total_pnl']:.2f}")
    else:
        print(f"No threshold meets the minimum trades requirement ({args.min_trades})")

    # Save to JSON if requested
    if args.output_json:
        import json
        output = {
            "city": args.city,
            "thresholds_tested": thresholds,
            "results": results,
            "optimal_threshold": optimal_threshold,
            "optimal_metrics": optimal_metrics,
            "min_trades_required": args.min_trades,
            "optimization_metric": args.metric,
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Saved results to {args.output_json}")

    # Suggest config update
    print()
    print("=" * 60)
    print("SUGGESTED CONFIG UPDATE")
    print("=" * 60)
    print()
    print("Add to config/edge_thresholds.py:")
    print()
    print(f'EDGE_MIN_THRESHOLD_F = {{')
    print(f'    "{args.city}": {optimal_threshold:.2f},')
    print(f'    # ... other cities')
    print(f'}}')
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
