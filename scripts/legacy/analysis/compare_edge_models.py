#!/usr/bin/env python3
"""Compare CatBoost vs Linear models for edge classification.

Trains all 4 models (CatBoost, ElasticNet, Lasso, Ridge) on same data and metrics.
Reports performance comparison and feature importance.

Usage:
    # Compare all models
    python scripts/compare_edge_models.py --city miami --threshold 1.5

    # Test on recent data only (last 5 months)
    python scripts/compare_edge_models.py --city miami --threshold 1.5 --recent-months 5

    # Compare specific models
    python scripts/compare_edge_models.py --city miami --models elastic,lasso,ridge
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import time

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.edge.linear_elastic import LinearElasticEdgeClassifier
from models.edge.linear_lasso import LinearLassoEdgeClassifier
from models.edge.linear_ridge import LinearRidgeEdgeClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Feature columns (same as EdgeClassifier)
FEATURE_COLS = [
    'forecast_temp', 'market_temp', 'edge', 'confidence',
    'forecast_uncertainty', 'market_uncertainty', 'base_temp',
    'predicted_delta', 'obs_fcst_max_gap', 'fcst_remaining_potential',
    'snapshot_hour', 'hours_to_event_close', 'minutes_since_market_open',
    'temp_volatility_30min', 'market_bid_ask_spread'
]


def load_and_prepare_data(city: str, threshold: float, recent_months: Optional[int] = None):
    """Load edge data and prepare for training.

    Args:
        city: City name
        threshold: Min edge threshold
        recent_months: If set, use only last N months

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    cache_path = Path(f"models/saved/{city}/edge_training_data_realistic.parquet")

    if not cache_path.exists():
        logger.error(f"Edge data not found: {cache_path}")
        logger.error(f"Run: python scripts/train_edge_classifier.py --city {city} --regenerate-only")
        sys.exit(1)

    logger.info(f"Loading edge data from {cache_path}")
    df = pd.read_parquet(cache_path)
    logger.info(f"  Loaded {len(df):,} rows")

    # Filter to threshold
    df = df[df['edge'].abs() >= threshold].copy()
    logger.info(f"  Filtered to >= {threshold}°F: {len(df):,} rows")

    # Filter to signals only (exclude no_trade)
    df = df[df['signal'] != 'no_trade'].copy()
    logger.info(f"  After signal filter: {len(df):,} rows")

    # Filter to recent months if specified
    if recent_months:
        df['date'] = pd.to_datetime(df['day'])
        cutoff = df['date'].max() - pd.DateOffset(months=recent_months)
        df_before = len(df)
        df = df[df['date'] >= cutoff].copy()
        logger.info(f"  Filtered to last {recent_months} months: {df_before:,} → {len(df):,} rows")

    # Prepare features and target
    X = df[FEATURE_COLS].fillna(0)
    y = (df['pnl'] > 0).astype(int)

    # Temporal split (80/20 by rows, which maintains temporal order)
    n_train = int(len(X) * 0.8)
    X_train = X.iloc[:n_train]
    X_test = X.iloc[n_train:]
    y_train = y.iloc[:n_train]
    y_test = y.iloc[n_train:]

    logger.info(f"\nTemporal split:")
    logger.info(f"  Train: {len(X_train):,} samples ({y_train.mean():.1%} positive)")
    logger.info(f"  Test: {len(X_test):,} samples ({y_test.mean():.1%} positive)")

    return X_train, X_test, y_train, y_test, FEATURE_COLS


def train_and_evaluate(model, X_train, y_train, X_test, y_test, name: str):
    """Train model and compute metrics.

    Args:
        model: Model instance with fit/predict_proba methods
        X_train, y_train: Training data
        X_test, y_test: Test data
        name: Model name for logging

    Returns:
        Dict with performance metrics
    """
    logger.info(f"\nTraining {name}...")

    # Train
    start = time.time()
    model.fit(X_train.values, y_train.values, feature_names=FEATURE_COLS)
    train_time = time.time() - start
    logger.info(f"  Training time: {train_time:.1f}s")

    # Predictions
    y_pred_proba = model.predict_proba(X_test.values)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    baseline_win = y_test.mean()
    filtered_win = y_test[y_pred == 1].mean() if y_pred.sum() > 0 else 0
    n_trades = y_pred.sum()

    logger.info(f"  Test AUC: {auc:.4f}")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  Baseline win rate: {baseline_win:.1%}")
    logger.info(f"  Filtered win rate: {filtered_win:.1%} ({n_trades} trades)")

    return {
        'name': name,
        'train_time': train_time,
        'test_auc': auc,
        'test_accuracy': acc,
        'baseline_win_rate': baseline_win,
        'filtered_win_rate': filtered_win,
        'n_trades': n_trades,
        'model': model,
    }


def print_comparison_table(results: list):
    """Print formatted comparison table."""
    print("\n" + "="*90)
    print("MODEL COMPARISON RESULTS")
    print("="*90)
    print(f"{'Model':<15} {'Train Time':<12} {'Test AUC':<10} {'Baseline Win':<13} {'Filtered Win':<13} {'N Trades':<10}")
    print("-"*90)

    for res in sorted(results, key=lambda x: x['test_auc'], reverse=True):
        print(f"{res['name']:<15} {res['train_time']:>10.1f}s "
              f"{res['test_auc']:>9.4f} {res['baseline_win_rate']:>11.1%} "
              f"{res['filtered_win_rate']:>11.1%} {res['n_trades']:>9,}")

    print("-"*90)


def print_feature_importance(results: list, top_n: int = 15):
    """Print feature importance from linear models."""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE (ElasticNet Coefficients)")
    print("="*70)

    # Get ElasticNet results
    elastic_res = [r for r in results if 'elastic' in r['name'].lower()]
    if not elastic_res:
        print("ElasticNet not trained - skipping feature importance")
        return

    model = elastic_res[0]['model']
    feat_imp = model.get_feature_coefficients()  # Get signed coefficients

    # Sort by absolute value
    sorted_feats = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Feature':<30} {'Coefficient':<12} {'Abs Value':<10}")
    print("-"*70)
    for feat, coef in sorted_feats[:top_n]:
        print(f"{feat:<30} {coef:>+11.4f} {abs(coef):>9.4f}")

    # Highlight time features
    time_features = ['snapshot_hour', 'hours_to_event_close', 'minutes_since_market_open']
    print("\n" + "="*70)
    print("TIME-OF-DAY FEATURES (Highlighting)")
    print("="*70)
    for feat in time_features:
        if feat in feat_imp:
            coef = feat_imp[feat]
            rank = sorted_feats.index((feat, coef)) + 1
            print(f"{feat:<30} {coef:>+11.4f} (rank {rank}/{len(feat_imp)})")


def main():
    parser = argparse.ArgumentParser(
        description="Compare edge classification models"
    )
    parser.add_argument(
        "--city",
        required=True,
        choices=["austin", "chicago", "denver", "los_angeles", "miami", "philadelphia"],
        help="City to analyze"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Min edge threshold in °F (default: 1.5)"
    )
    parser.add_argument(
        "--recent-months",
        type=int,
        help="Use only last N months of data (for stationarity check)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="elastic,lasso,ridge",
        help="Comma-separated list of models to train (elastic,lasso,ridge)"
    )
    args = parser.parse_args()

    print("="*70)
    print("EDGE MODEL COMPARISON")
    print("="*70)
    print(f"City: {args.city}")
    print(f"Threshold: {args.threshold}°F")
    if args.recent_months:
        print(f"Recent data only: Last {args.recent_months} months")
    print("="*70)

    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(
        args.city, args.threshold, args.recent_months
    )

    # Initialize models
    model_map = {
        'elastic': ('ElasticNet', LinearElasticEdgeClassifier()),
        'lasso': ('Lasso', LinearLassoEdgeClassifier()),
        'ridge': ('Ridge', LinearRidgeEdgeClassifier()),
    }

    # Filter to requested models
    requested = args.models.lower().split(',')
    models_to_train = {k: v for k, v in model_map.items() if k in requested}

    if not models_to_train:
        logger.error(f"No valid models specified: {args.models}")
        logger.error(f"Valid options: elastic, lasso, ridge")
        sys.exit(1)

    # Train all models
    results = []
    for model_key, (model_name, model_instance) in models_to_train.items():
        res = train_and_evaluate(model_instance, X_train, y_train, X_test, y_test, model_name)
        results.append(res)

    # Print comparison
    print_comparison_table(results)

    # Print feature importance
    print_feature_importance(results, top_n=15)

    # Save results summary
    output_dir = Path(f"models/saved/{args.city}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / "linear_model_comparison.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Model Comparison for {args.city}\n")
        f.write(f"Threshold: {args.threshold}°F\n")
        f.write(f"Date filter: Last {args.recent_months} months\n" if args.recent_months else "Full dataset\n")
        f.write("\n")
        for res in results:
            f.write(f"{res['name']}: AUC={res['test_auc']:.4f}, Win={res['filtered_win_rate']:.1%}, Trades={res['n_trades']}\n")

    print(f"\nSummary saved to: {summary_file}")

    # Return best model info
    best = max(results, key=lambda x: x['test_auc'])
    print(f"\n🏆 Best Model: {best['name']} (AUC={best['test_auc']:.4f})")


if __name__ == "__main__":
    main()
