#!/usr/bin/env python3
"""
A/B testing script for comparing ElasticNet vs CatBoost models.

This script:
1. Trains both models on the same walk-forward windows
2. Runs backtests for each model
3. Compares key metrics: Sharpe, Brier/ECE, fees, P&L
4. Generates a comparison report
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import subprocess
import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


def train_models(
    city: str,
    bracket: str,
    start: str,
    end: str,
    feature_set: str = "baseline",
    train_days: int = 90,
    models_dir: str = "models/trained"
) -> Dict[str, bool]:
    """
    Train both ElasticNet and CatBoost models.

    Returns:
        Dict mapping model_type to success status
    """
    results = {}

    # Train ElasticNet
    logger.info(f"\n{'='*70}")
    logger.info(f"Training ElasticNet model...")
    logger.info(f"{'='*70}")

    cmd_elasticnet = [
        "python", "ml/train_walkforward.py",
        "--city", city,
        "--bracket", bracket,
        "--start", start,
        "--end", end,
        "--feature-set", feature_set,
        "--train-days", str(train_days),
        "--model-type", "elasticnet",
        "--trials", "40",
        "--outdir", models_dir
    ]

    result = subprocess.run(cmd_elasticnet, capture_output=True, text=True)
    results["elasticnet"] = result.returncode == 0

    if results["elasticnet"]:
        logger.info("✓ ElasticNet training complete")
    else:
        logger.error(f"✗ ElasticNet training failed: {result.stderr}")

    # Train CatBoost
    logger.info(f"\n{'='*70}")
    logger.info(f"Training CatBoost model...")
    logger.info(f"{'='*70}")

    cmd_catboost = [
        "python", "ml/train_walkforward.py",
        "--city", city,
        "--bracket", bracket,
        "--start", start,
        "--end", end,
        "--feature-set", feature_set,
        "--train-days", str(train_days),
        "--model-type", "catboost",
        "--trials", "60",
        "--outdir", models_dir
    ]

    result = subprocess.run(cmd_catboost, capture_output=True, text=True)
    results["catboost"] = result.returncode == 0

    if results["catboost"]:
        logger.info("✓ CatBoost training complete")
    else:
        logger.error(f"✗ CatBoost training failed: {result.stderr}")

    return results


def run_backtest(
    city: str,
    bracket: str,
    model_type: str,
    models_dir: str = "models/trained",
    initial_cash: float = 10000.0,
    unified_head: bool = False
) -> Dict[str, Any]:
    """
    Run backtest for a specific model type.

    Returns:
        Dict with backtest metrics
    """
    output_file = f"results/ab_test_{city}_{bracket}_{model_type}.json"

    cmd = [
        "python", "backtest/run_backtest.py",
        "--city", city,
        "--bracket", bracket,
        "--strategy", "model_kelly",
        "--models-dir", models_dir,
        "--model-type", model_type,
        "--initial-cash", str(initial_cash),
        "--output-json", output_file
    ]

    if unified_head:
        cmd.append("--unified-head")

    logger.info(f"\nRunning {model_type} backtest...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Backtest failed: {result.stderr}")
        return {}

    # Load JSON results
    try:
        with open(output_file, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        logger.error(f"Failed to load backtest results: {e}")
        return {}


def compare_models(
    elasticnet_metrics: Dict[str, Any],
    catboost_metrics: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create comparison table between models.

    Returns:
        DataFrame with side-by-side comparison
    """
    comparison = {
        "Metric": [
            "Sharpe Ratio",
            "Max Drawdown (%)",
            "Total P&L ($)",
            "Gross P&L ($)",
            "Total Fees ($)",
            "Fee Ratio (%)",
            "Number of Trades",
            "ECE (trades)",
            "ECE (all minutes)"
        ],
        "ElasticNet": [
            elasticnet_metrics.get("sharpe", 0),
            elasticnet_metrics.get("max_drawdown", 0) * 100,
            elasticnet_metrics.get("total_pnl_cents", 0) / 100,
            elasticnet_metrics.get("gross_pnl_cents", 0) / 100,
            elasticnet_metrics.get("total_fees_cents", 0) / 100,
            elasticnet_metrics.get("fee_ratio", 0) * 100,
            elasticnet_metrics.get("n_trades", 0),
            elasticnet_metrics.get("ece_trades"),
            elasticnet_metrics.get("ece_all_minutes")
        ],
        "CatBoost": [
            catboost_metrics.get("sharpe", 0),
            catboost_metrics.get("max_drawdown", 0) * 100,
            catboost_metrics.get("total_pnl_cents", 0) / 100,
            catboost_metrics.get("gross_pnl_cents", 0) / 100,
            catboost_metrics.get("total_fees_cents", 0) / 100,
            catboost_metrics.get("fee_ratio", 0) * 100,
            catboost_metrics.get("n_trades", 0),
            catboost_metrics.get("ece_trades"),
            catboost_metrics.get("ece_all_minutes")
        ]
    }

    df = pd.DataFrame(comparison)

    # Add winner column
    winners = []
    for i, metric in enumerate(comparison["Metric"]):
        en_val = comparison["ElasticNet"][i]
        cb_val = comparison["CatBoost"][i]

        if en_val is None or cb_val is None:
            winners.append("N/A")
            continue

        # Higher is better for: Sharpe, P&L
        # Lower is better for: Drawdown, Fees, ECE
        if metric in ["Sharpe Ratio", "Total P&L ($)", "Gross P&L ($)", "Number of Trades"]:
            if en_val > cb_val:
                winners.append("ElasticNet ↑")
            elif cb_val > en_val:
                winners.append("CatBoost ↑")
            else:
                winners.append("Tie")
        else:  # Lower is better
            if en_val < cb_val:
                winners.append("ElasticNet ↓")
            elif cb_val < en_val:
                winners.append("CatBoost ↓")
            else:
                winners.append("Tie")

    df["Winner"] = winners

    return df


def main():
    parser = argparse.ArgumentParser(
        description="A/B test comparing ElasticNet vs CatBoost models"
    )
    parser.add_argument("--city", default="chicago", help="City name")
    parser.add_argument("--bracket", choices=["between", "greater", "less"],
                        default="between", help="Bracket type")
    parser.add_argument("--train-start", required=True, help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--train-end", required=True, help="Training end date (YYYY-MM-DD)")
    parser.add_argument("--feature-set", default="baseline",
                        choices=["baseline", "ridge_conservative", "elasticnet_rich"],
                        help="Feature set to use")
    parser.add_argument("--train-days", type=int, default=90,
                        help="Training window size in days")
    parser.add_argument("--models-dir", default="models/trained",
                        help="Output directory for models")
    parser.add_argument("--initial-cash", type=float, default=10000.0,
                        help="Initial cash for backtest ($)")
    parser.add_argument("--unified-head", action="store_true",
                        help="Use unified head coupling")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only run backtests")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*80)
    print("A/B TEST: ElasticNet vs CatBoost")
    print("="*80)
    print(f"City: {args.city}")
    print(f"Bracket: {args.bracket}")
    print(f"Training period: {args.train_start} to {args.train_end}")
    print(f"Feature set: {args.feature_set}")
    print(f"Training window: {args.train_days} days")
    print(f"Unified head: {args.unified_head}")
    print("="*80 + "\n")

    # Step 1: Train models (unless skipped)
    if not args.skip_training:
        train_results = train_models(
            city=args.city,
            bracket=args.bracket,
            start=args.train_start,
            end=args.train_end,
            feature_set=args.feature_set,
            train_days=args.train_days,
            models_dir=args.models_dir
        )

        if not all(train_results.values()):
            logger.error("Some models failed to train. Check logs above.")
            # Continue anyway to test what we have

    # Step 2: Run backtests
    elasticnet_metrics = run_backtest(
        city=args.city,
        bracket=args.bracket,
        model_type="elasticnet",
        models_dir=args.models_dir,
        initial_cash=args.initial_cash,
        unified_head=args.unified_head
    )

    catboost_metrics = run_backtest(
        city=args.city,
        bracket=args.bracket,
        model_type="catboost",
        models_dir=args.models_dir,
        initial_cash=args.initial_cash,
        unified_head=args.unified_head
    )

    # Step 3: Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80 + "\n")

    comparison_df = compare_models(elasticnet_metrics, catboost_metrics)
    print(comparison_df.to_string(index=False))

    # Save comparison to CSV
    output_path = f"results/ab_comparison_{args.city}_{args.bracket}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"\nComparison saved to: {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Count wins
    wins = comparison_df["Winner"].value_counts()
    for model in ["ElasticNet", "CatBoost"]:
        model_wins = sum(1 for w in comparison_df["Winner"] if w.startswith(model))
        print(f"{model} wins: {model_wins} metrics")

    # Key insights
    print("\nKey Insights:")

    # Sharpe comparison
    en_sharpe = elasticnet_metrics.get("sharpe", 0)
    cb_sharpe = catboost_metrics.get("sharpe", 0)
    if en_sharpe > 0 or cb_sharpe > 0:
        winner = "ElasticNet" if en_sharpe > cb_sharpe else "CatBoost"
        pct_diff = abs(en_sharpe - cb_sharpe) / max(abs(en_sharpe), abs(cb_sharpe)) * 100
        print(f"- {winner} has {pct_diff:.1f}% better Sharpe ratio")

    # Fee comparison
    en_fees = elasticnet_metrics.get("total_fees_cents", 0)
    cb_fees = catboost_metrics.get("total_fees_cents", 0)
    if en_fees > 0 or cb_fees > 0:
        winner = "ElasticNet" if en_fees < cb_fees else "CatBoost"
        fee_diff = abs(en_fees - cb_fees) / 100
        print(f"- {winner} has ${fee_diff:.2f} lower fees")

    # Trade count comparison
    en_trades = elasticnet_metrics.get("n_trades", 0)
    cb_trades = catboost_metrics.get("n_trades", 0)
    if en_trades > 0 or cb_trades > 0:
        more_active = "ElasticNet" if en_trades > cb_trades else "CatBoost"
        print(f"- {more_active} is more active with {abs(en_trades - cb_trades)} more trades")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()