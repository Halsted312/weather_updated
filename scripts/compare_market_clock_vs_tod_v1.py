#!/usr/bin/env python3
"""
Compare Market-Clock TOD v1 vs Per-City TOD v1 Models

This script compares predictions from:
1. Market-Clock TOD v1 (global model, all cities)
2. TOD v1 (per-city models)

For fair comparison, we:
- Filter market-clock to event day only (is_event_day == 1)
- Align to 15-min intervals that match TOD v1 snapshots
- Compare metrics per city

Usage:
    python scripts/compare_market_clock_vs_tod_v1.py
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

CITIES = ['austin', 'chicago', 'denver', 'los_angeles', 'miami', 'philadelphia']


def load_market_clock_model():
    """Load the global market-clock model."""
    model_path = MODEL_DIR / "market_clock_tod_v1" / "ordinal_catboost_market_clock_tod_v1.pkl"

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    logger.info(f"Loaded market-clock model: {len(model_data['classifiers'])} classifiers")
    return model_data


def load_tod_v1_model(city: str):
    """Load a per-city TOD v1 model."""
    model_path = MODEL_DIR / f"{city}_tod_v1" / "ordinal_catboost_tod_v1.pkl"

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    return model_data


def predict_ordinal(model_data: dict, features_df: pd.DataFrame) -> np.ndarray:
    """
    Run ordinal prediction through a model.

    Returns:
        Array of predicted deltas (rounded expected values)
    """
    classifiers = model_data['classifiers']
    thresholds = model_data['thresholds']
    delta_classes = np.array(model_data['delta_classes'])

    n_samples = len(features_df)
    cum_proba = np.ones((n_samples, len(thresholds) + 1))

    for i, k in enumerate(thresholds):
        clf = classifiers[k]
        pool = Pool(features_df)
        p_ge_k = clf.predict_proba(pool)[:, 1]
        cum_proba[:, i + 1] = p_ge_k

    # Convert cumulative to class probabilities
    class_proba = np.zeros((n_samples, len(delta_classes)))
    for i in range(len(delta_classes)):
        if i == 0:
            class_proba[:, i] = 1 - cum_proba[:, 1]
        elif i == len(delta_classes) - 1:
            class_proba[:, i] = cum_proba[:, i]
        else:
            class_proba[:, i] = cum_proba[:, i] - cum_proba[:, i + 1]

    # Expected delta
    expected_delta = (class_proba * delta_classes).sum(axis=1)
    return np.round(expected_delta).astype(int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute evaluation metrics."""
    errors = np.abs(y_true - y_pred)
    return {
        'mae': errors.mean(),
        'accuracy': (y_true == y_pred).mean() * 100,
        'within_1': (errors <= 1).mean() * 100,
        'within_2': (errors <= 2).mean() * 100,
        'n_samples': len(y_true),
    }


def run_comparison():
    """Run the full comparison."""
    logger.info("=" * 70)
    logger.info("MARKET-CLOCK VS TOD V1 COMPARISON")
    logger.info("=" * 70)

    # Load market-clock model and test data
    mc_model = load_market_clock_model()
    mc_test = pd.read_parquet(MODEL_DIR / "market_clock_tod_v1" / "test_data.parquet")
    mc_feature_cols = mc_model['feature_cols']

    logger.info(f"Market-clock test data: {len(mc_test):,} rows")

    # Filter to event day only
    mc_event_day = mc_test[mc_test['is_event_day'] == 1].copy()
    logger.info(f"After filtering to event day: {len(mc_event_day):,} rows")

    # Extract hour and minute from snapshot_datetime
    mc_event_day['local_hour'] = mc_event_day['snapshot_datetime'].dt.hour
    mc_event_day['minute'] = mc_event_day['snapshot_datetime'].dt.minute

    # Filter to TOD v1 time range (10:00 - 23:45)
    mc_tod_range = mc_event_day[
        (mc_event_day['local_hour'] >= 10) &
        (mc_event_day['local_hour'] <= 23)
    ].copy()
    logger.info(f"After filtering to 10:00-23:59: {len(mc_tod_range):,} rows")

    # Filter to 15-min aligned snapshots (minute in 0, 15, 30, 45)
    mc_aligned = mc_tod_range[mc_tod_range['minute'].isin([0, 15, 30, 45])].copy()
    logger.info(f"After 15-min alignment: {len(mc_aligned):,} rows")

    # Results storage
    results = []

    for city in CITIES:
        logger.info(f"\n--- {city.upper()} ---")

        # Load TOD v1 model and test data
        tod_model = load_tod_v1_model(city)
        tod_test = pd.read_parquet(MODEL_DIR / f"{city}_tod_v1" / "test_data.parquet")
        tod_feature_cols = tod_model['feature_cols']

        # Get market-clock data for this city
        mc_city = mc_aligned[mc_aligned['city'] == city].copy()

        logger.info(f"  TOD v1 test: {len(tod_test):,} rows")
        logger.info(f"  Market-clock (aligned): {len(mc_city):,} rows")

        # Run predictions on market-clock data
        mc_X = mc_city[mc_feature_cols]
        mc_y_true = mc_city['delta'].values
        mc_y_pred = predict_ordinal(mc_model, mc_X)
        mc_metrics = compute_metrics(mc_y_true, mc_y_pred)

        # Run predictions on TOD v1 data
        tod_X = tod_test[tod_feature_cols]
        tod_y_true = tod_test['delta'].values
        tod_y_pred = predict_ordinal(tod_model, tod_X)
        tod_metrics = compute_metrics(tod_y_true, tod_y_pred)

        logger.info(f"  TOD v1:        MAE={tod_metrics['mae']:.3f}, W1={tod_metrics['within_1']:.1f}%, W2={tod_metrics['within_2']:.1f}%")
        logger.info(f"  Market-Clock:  MAE={mc_metrics['mae']:.3f}, W1={mc_metrics['within_1']:.1f}%, W2={mc_metrics['within_2']:.1f}%")

        results.append({
            'city': city,
            'tod_v1_mae': tod_metrics['mae'],
            'tod_v1_w1': tod_metrics['within_1'],
            'tod_v1_w2': tod_metrics['within_2'],
            'mc_mae': mc_metrics['mae'],
            'mc_w1': mc_metrics['within_1'],
            'mc_w2': mc_metrics['within_2'],
            'mae_delta': mc_metrics['mae'] - tod_metrics['mae'],
            'tod_v1_n': tod_metrics['n_samples'],
            'mc_n': mc_metrics['n_samples'],
        })

    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY (Event Day, 15-min Aligned)")
    logger.info("=" * 70)

    df = pd.DataFrame(results)

    # Print formatted table
    print("\n" + "-" * 90)
    print(f"{'City':<15} | {'TOD v1 MAE':>10} | {'MC MAE':>10} | {'Delta':>8} | {'TOD v1 W2':>10} | {'MC W2':>10}")
    print("-" * 90)

    for _, row in df.iterrows():
        delta_str = f"+{row['mae_delta']:.3f}" if row['mae_delta'] > 0 else f"{row['mae_delta']:.3f}"
        print(f"{row['city']:<15} | {row['tod_v1_mae']:>10.3f} | {row['mc_mae']:>10.3f} | {delta_str:>8} | {row['tod_v1_w2']:>9.1f}% | {row['mc_w2']:>9.1f}%")

    print("-" * 90)

    # Aggregate metrics
    print(f"{'AVERAGE':<15} | {df['tod_v1_mae'].mean():>10.3f} | {df['mc_mae'].mean():>10.3f} | {df['mae_delta'].mean():>+8.3f} | {df['tod_v1_w2'].mean():>9.1f}% | {df['mc_w2'].mean():>9.1f}%")
    print("-" * 90)

    # Decision recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    avg_delta = df['mae_delta'].mean()
    if avg_delta < -0.1:
        print(f"Market-Clock is BETTER by {-avg_delta:.3f} MAE on average")
        print("→ Proceed with market-clock model for production")
    elif avg_delta > 0.1:
        print(f"TOD v1 is BETTER by {avg_delta:.3f} MAE on average")
        print("→ Consider hybrid approach: TOD v1 for event day, market-clock for D-1 only")
    else:
        print(f"Models are COMPARABLE (delta={avg_delta:.3f} MAE)")
        print("→ Prefer market-clock for simplicity (1 global model vs 6)")

    # Check per-city winners
    mc_wins = (df['mae_delta'] < 0).sum()
    tod_wins = (df['mae_delta'] > 0).sum()
    print(f"\nPer-city wins: Market-Clock={mc_wins}, TOD v1={tod_wins}")

    return df


if __name__ == "__main__":
    results_df = run_comparison()

    # Save results
    output_path = MODEL_DIR / "market_clock_tod_v1" / "comparison_results.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")
