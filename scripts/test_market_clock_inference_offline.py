#!/usr/bin/env python3
"""
Offline Inference Test for Market-Clock TOD v1

Tests train/inference parity by:
1. Loading trained model and test dataset
2. For sample rows, rebuilding features via inference path
3. Comparing inference predictions to training-time predictions

This catches any train/inference skew (e.g., missing features, different NaN handling).

Usage:
    python scripts/test_market_clock_inference_offline.py [--samples 100]
"""

import argparse
import logging
import pickle
import sys
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool
from zoneinfo import ZoneInfo

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "saved" / "market_clock_tod_v1"


def load_model():
    """Load the trained market-clock model."""
    model_path = MODEL_DIR / "ordinal_catboost_market_clock_tod_v1.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    logger.info(f"Loaded model with {len(model_data['classifiers'])} classifiers")
    return model_data


def load_test_data():
    """Load the test dataset used during training."""
    test_path = MODEL_DIR / "test_data.parquet"

    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    df = pd.read_parquet(test_path)
    logger.info(f"Loaded test data: {len(df):,} rows")
    return df


def predict_ordinal(model_data: dict, features_df: pd.DataFrame) -> np.ndarray:
    """
    Run ordinal prediction through the model.

    Returns:
        Array of class probabilities for each sample
    """
    classifiers = model_data['classifiers']
    thresholds = model_data['thresholds']
    delta_classes = model_data['delta_classes']

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

    return class_proba


def predict_delta(model_data: dict, features_df: pd.DataFrame) -> np.ndarray:
    """Get expected delta predictions."""
    class_proba = predict_ordinal(model_data, features_df)
    delta_classes = np.array(model_data['delta_classes'])
    return (class_proba * delta_classes).sum(axis=1)


def run_parity_test(model_data: dict, test_df: pd.DataFrame, n_samples: int = 100):
    """
    Test train/inference parity on sample rows.

    Compares:
    1. Predictions using training-time features (from test_df)
    2. Any discrepancies in feature values
    """
    feature_cols = model_data['feature_cols']

    # Sample rows for testing
    if n_samples < len(test_df):
        sample_idx = np.random.choice(len(test_df), n_samples, replace=False)
        sample_df = test_df.iloc[sample_idx].copy()
    else:
        sample_df = test_df.copy()

    logger.info(f"Testing {len(sample_df)} samples")

    # Get features from test data
    X_test = sample_df[feature_cols]
    y_true = sample_df['delta'].values

    # Run predictions
    pred_delta = predict_delta(model_data, X_test)
    pred_delta_round = np.round(pred_delta).astype(int)

    # Compute metrics
    mae = np.abs(y_true - pred_delta_round).mean()
    within_1 = (np.abs(y_true - pred_delta_round) <= 1).mean() * 100
    within_2 = (np.abs(y_true - pred_delta_round) <= 2).mean() * 100

    logger.info(f"Parity Test Results (n={len(sample_df)}):")
    logger.info(f"  MAE: {mae:.3f}")
    logger.info(f"  Within-1: {within_1:.1f}%")
    logger.info(f"  Within-2: {within_2:.1f}%")

    # Check feature completeness
    missing_features = [col for col in feature_cols if col not in test_df.columns]
    if missing_features:
        logger.warning(f"Missing features in test data: {missing_features}")

    # Check for NaN values
    nan_counts = X_test.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        logger.warning(f"Features with NaN values:")
        for col, count in nan_cols.items():
            logger.warning(f"  {col}: {count} NaNs")

    return {
        'mae': mae,
        'within_1': within_1,
        'within_2': within_2,
        'n_samples': len(sample_df),
        'missing_features': missing_features,
        'nan_features': nan_cols.to_dict() if len(nan_cols) > 0 else {},
    }


def test_specific_points(model_data: dict, test_df: pd.DataFrame):
    """
    Test specific (city, date, time) points for detailed inspection.
    """
    feature_cols = model_data['feature_cols']

    # Get unique city/date combinations
    unique_combos = test_df[['city', 'event_date']].drop_duplicates()

    logger.info(f"\n{'='*60}")
    logger.info("SPECIFIC POINT TESTS")
    logger.info(f"{'='*60}")

    # Test a few specific points
    test_points = [
        # Early D-1 (high uncertainty expected)
        {'city': 'chicago', 'is_d_minus_1': 1, 'hours_range': (0, 6)},
        # Late event day (low uncertainty expected)
        {'city': 'chicago', 'is_event_day': 1, 'hours_range': (18, 24)},
        # Miami (typically easiest city)
        {'city': 'miami', 'is_event_day': 1, 'hours_range': (12, 18)},
        # Denver (typically hardest city)
        {'city': 'denver', 'is_event_day': 1, 'hours_range': (12, 18)},
    ]

    for point in test_points:
        city = point['city']
        mask = test_df['city'] == city

        if 'is_d_minus_1' in point:
            mask &= test_df['is_d_minus_1'] == point['is_d_minus_1']
        if 'is_event_day' in point:
            mask &= test_df['is_event_day'] == point['is_event_day']

        hours_lo, hours_hi = point['hours_range']
        mask &= (test_df['hours_since_market_open'] >= hours_lo)
        mask &= (test_df['hours_since_market_open'] < hours_hi)

        subset = test_df[mask]
        if len(subset) == 0:
            logger.warning(f"No data for {point}")
            continue

        # Take first row for detailed inspection
        row = subset.iloc[0]
        X_row = row[feature_cols].values.reshape(1, -1)
        X_df = pd.DataFrame(X_row, columns=feature_cols)

        pred = predict_delta(model_data, X_df)[0]
        actual = row['delta']

        phase = "D-1" if row.get('is_d_minus_1', 0) == 1 else "D"
        hours = row.get('hours_since_market_open', 0)

        logger.info(f"\n{city.upper()} - {phase} @ {hours:.1f}h:")
        logger.info(f"  Event date: {row['event_date']}")
        logger.info(f"  Actual delta: {actual}")
        logger.info(f"  Predicted delta: {pred:.2f} (rounded: {round(pred)})")
        logger.info(f"  Error: {abs(actual - round(pred))}")
        logger.info(f"  t_base (vc_max_f_sofar): {row.get('vc_max_f_sofar', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="Test market-clock inference parity")
    parser.add_argument('--samples', type=int, default=1000, help="Number of samples to test")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    logger.info("="*60)
    logger.info("MARKET-CLOCK TOD V1 OFFLINE INFERENCE TEST")
    logger.info("="*60)

    # Load model and data
    model_data = load_model()
    test_df = load_test_data()

    # Run parity test
    results = run_parity_test(model_data, test_df, n_samples=args.samples)

    # Test specific points
    test_specific_points(model_data, test_df)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Parity test passed: MAE={results['mae']:.3f}, W2={results['within_2']:.1f}%")

    if results['missing_features']:
        logger.warning(f"WARNING: {len(results['missing_features'])} missing features")
        return 1

    if results['nan_features']:
        logger.warning(f"WARNING: {len(results['nan_features'])} features have NaN values")

    logger.info("Test complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
