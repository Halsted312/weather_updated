#!/usr/bin/env python3
"""
Market-Clock Model Health Check

Evaluates the global Market-Clock model performance by time bucket.
Answers the question: "How well does the model predict at different times
in the D-1 to event-day window?"

Buckets:
- [-38,-30): Very early D-1 (10:00-18:00)
- [-30,-24): Late D-1 afternoon (18:00-24:00)
- [-24,-18): D-1 night / early D morning (D 00:00-06:00)
- [-18,-12): D morning (06:00-12:00)
- [-12,-6): D afternoon (12:00-18:00)
- [-6,0]: D evening (18:00-23:55)

Usage:
    .venv/bin/python scripts/health_check_market_clock.py
"""

import logging
import sys
import pickle
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from catboost import Pool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = Path("models/saved/market_clock_tod_v1")
MODEL_PATH = MODEL_DIR / "ordinal_catboost_market_clock_tod_v1.pkl"
TEST_DATA_PATH = MODEL_DIR / "test_data.parquet"
TRAIN_DATA_PATH = MODEL_DIR / "train_data.parquet"

# Time buckets (hours to event close) - POSITIVE values
# Market close is D 23:55, market open is D-1 10:00, so:
# - D-1 10:00 = 37.92 hours to close
# - D-1 18:00 = 29.92 hours to close
# - D 00:00 = 23.92 hours to close
# - D 06:00 = 17.92 hours to close
# - D 12:00 = 11.92 hours to close
# - D 18:00 = 5.92 hours to close
# - D 23:55 = 0 hours to close
TIME_BUCKETS = [
    (30, 38, "D-1 morning (10:00-18:00)"),
    (24, 30, "D-1 evening (18:00-24:00)"),
    (18, 24, "D early morning (00:00-06:00)"),
    (12, 18, "D morning (06:00-12:00)"),
    (6, 12, "D afternoon (12:00-18:00)"),
    (0, 6, "D evening (18:00-23:55)"),
]


def compute_hours_to_event_close(df: pd.DataFrame) -> pd.DataFrame:
    """Add hours_to_event_close column to dataframe.

    Market close is always event_date 23:55 local time.
    """
    df = df.copy()

    # snapshot_datetime should be in the dataframe
    if 'snapshot_datetime' not in df.columns:
        logger.warning("No snapshot_datetime column - using hours_since_market_open")
        # Infer from hours_since_market_open
        # Market open is D-1 10:00, market close is D 23:55 = 37.917 hours after open
        df['hours_to_event_close'] = 37.917 - df['hours_since_market_open']
        return df

    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['snapshot_datetime']):
        df['snapshot_datetime'] = pd.to_datetime(df['snapshot_datetime'])

    # Compute market close for each event
    if 'event_date' in df.columns:
        df['market_close'] = pd.to_datetime(df['event_date']) + pd.Timedelta(hours=23, minutes=55)
    else:
        # Fall back to inferring from snapshot
        df['market_close'] = df['snapshot_datetime'].dt.normalize() + pd.Timedelta(hours=23, minutes=55)
        # Adjust for D-1 snapshots (where is_d_minus_1 == 1)
        if 'is_d_minus_1' in df.columns:
            mask = df['is_d_minus_1'] == 1
            df.loc[mask, 'market_close'] = df.loc[mask, 'market_close'] + pd.Timedelta(days=1)

    # Compute hours to close
    df['hours_to_event_close'] = (df['market_close'] - df['snapshot_datetime']).dt.total_seconds() / 3600.0

    # Drop helper column
    df = df.drop(columns=['market_close'], errors='ignore')

    return df


def bucket_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Assign time bucket to each row."""
    df = df.copy()

    def assign_bucket(hours):
        for low, high, label in TIME_BUCKETS:
            if low <= hours < high:
                return label
        if hours >= 38:
            return "D-1 morning (10:00-18:00)"  # Catch edge case (very early)
        return "D evening (18:00-23:55)"  # Catch edge case (very late)

    df['time_bucket'] = df['hours_to_event_close'].apply(assign_bucket)
    return df


def predict_with_model(df: pd.DataFrame, model_data: dict) -> np.ndarray:
    """Run predictions using loaded model.

    Returns array of predicted delta values.
    """
    feature_cols = model_data['feature_cols']
    classifiers = model_data['classifiers']
    thresholds = model_data['thresholds']
    delta_classes = model_data['delta_classes']

    # Filter to available features
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = set(feature_cols) - set(available_cols)
    if missing_cols:
        logger.warning(f"Missing {len(missing_cols)} features: {list(missing_cols)[:5]}...")

    X = df[available_cols].copy()

    # Fill NaN
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    n_samples = len(X)
    n_classes = len(delta_classes)

    # Compute cumulative probabilities P(delta >= k)
    cum_proba = np.ones((n_samples, len(thresholds) + 1))

    for i, k in enumerate(thresholds):
        clf_data = classifiers.get(k)
        if clf_data is None:
            continue

        if isinstance(clf_data, dict) and clf_data.get('type') == 'constant':
            # Constant predictor
            cum_proba[:, i + 1] = clf_data['prob']
        else:
            # CatBoost model
            pool = Pool(X)
            p_ge_k = clf_data.predict_proba(pool)[:, 1]
            cum_proba[:, i + 1] = p_ge_k

    # Convert cumulative to class probabilities
    class_proba = np.zeros((n_samples, n_classes))
    for i in range(n_classes):
        if i == 0:
            class_proba[:, i] = 1 - cum_proba[:, 1]
        elif i == n_classes - 1:
            class_proba[:, i] = cum_proba[:, i]
        else:
            class_proba[:, i] = cum_proba[:, i] - cum_proba[:, i + 1]

    # Predict most likely class
    predicted_idx = np.argmax(class_proba, axis=1)
    predicted_delta = np.array([delta_classes[i] for i in predicted_idx])

    return predicted_delta


def run_health_check():
    """Run health check on Market-Clock model."""

    logger.info("=" * 80)
    logger.info("MARKET-CLOCK MODEL HEALTH CHECK")
    logger.info("=" * 80)

    # Check files exist
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        return None

    if not TEST_DATA_PATH.exists():
        logger.error(f"Test data not found at {TEST_DATA_PATH}")
        return None

    # Load model
    logger.info(f"Loading model from {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)

    logger.info(f"  Feature columns: {len(model_data['feature_cols'])}")
    logger.info(f"  Classifiers: {len(model_data['classifiers'])}")
    logger.info(f"  Delta classes: {model_data['delta_classes']}")

    # Load test data
    logger.info(f"Loading test data from {TEST_DATA_PATH}")
    df_test = pd.read_parquet(TEST_DATA_PATH)
    logger.info(f"  Test rows: {len(df_test):,}")

    # Compute hours_to_event_close
    logger.info("Computing hours_to_event_close...")
    df_test = compute_hours_to_event_close(df_test)

    if 'hours_to_event_close' not in df_test.columns:
        logger.error("Failed to compute hours_to_event_close")
        return None

    # Assign buckets
    df_test = bucket_rows(df_test)

    # Run predictions
    logger.info("Running predictions...")
    df_test['predicted_delta'] = predict_with_model(df_test, model_data)

    # Compute metrics by bucket
    results = []

    for low, high, label in TIME_BUCKETS:
        mask = (df_test['hours_to_event_close'] >= low) & (df_test['hours_to_event_close'] < high)
        bucket_df = df_test[mask]

        if len(bucket_df) == 0:
            results.append({
                'bucket': label,
                'hours_range': f"[{low},{high})",
                'n_rows': 0,
                'accuracy': 0,
                'mae': float('inf'),
                'within_1': 0,
            })
            continue

        y_true = bucket_df['delta'].values
        y_pred = bucket_df['predicted_delta'].values

        accuracy = (y_true == y_pred).mean()
        mae = np.abs(y_true - y_pred).mean()
        within_1 = (np.abs(y_true - y_pred) <= 1).mean()

        results.append({
            'bucket': label,
            'hours_range': f"[{low},{high})",
            'n_rows': len(bucket_df),
            'accuracy': accuracy,
            'mae': mae,
            'within_1': within_1,
        })

    # Compute overall metrics
    y_true_all = df_test['delta'].values
    y_pred_all = df_test['predicted_delta'].values

    overall = {
        'bucket': 'OVERALL',
        'hours_range': '[0,38]',
        'n_rows': len(df_test),
        'accuracy': (y_true_all == y_pred_all).mean(),
        'mae': np.abs(y_true_all - y_pred_all).mean(),
        'within_1': (np.abs(y_true_all - y_pred_all) <= 1).mean(),
    }

    # Print results
    print("\n")
    print("=" * 80)
    print("MARKET-CLOCK HEALTH CHECK RESULTS")
    print("=" * 80)
    print("")
    print("## Performance by Time Bucket")
    print("")
    print("| Time Bucket | Hours Range | Rows | Accuracy | MAE | Within-1 |")
    print("|-------------|-------------|------|----------|-----|----------|")

    for r in results:
        print(f"| {r['bucket']:<28} | {r['hours_range']:<11} | {r['n_rows']:>5,} | "
              f"{r['accuracy']*100:>7.1f}% | {r['mae']:>4.2f} | {r['within_1']*100:>7.1f}% |")

    print("|-------------|-------------|------|----------|-----|----------|")
    print(f"| **{overall['bucket']}** | {overall['hours_range']:<11} | {overall['n_rows']:>5,} | "
          f"**{overall['accuracy']*100:.1f}%** | **{overall['mae']:.2f}** | **{overall['within_1']*100:.1f}%** |")

    print("")

    # Analysis
    print("## Analysis")
    print("")

    # Find best and worst buckets
    best_bucket = max(results, key=lambda x: x['accuracy'] if x['n_rows'] > 0 else 0)
    worst_bucket = min(results, key=lambda x: x['accuracy'] if x['n_rows'] > 0 else 1)

    print(f"- **Best bucket**: {best_bucket['bucket']} ({best_bucket['accuracy']*100:.1f}% accuracy)")
    print(f"- **Worst bucket**: {worst_bucket['bucket']} ({worst_bucket['accuracy']*100:.1f}% accuracy)")
    print("")

    # Check D-1 vs D performance
    d_minus_1_acc = []
    event_day_acc = []
    for r in results:
        if r['n_rows'] > 0:
            if 'D-1' in r['bucket']:
                d_minus_1_acc.append((r['accuracy'], r['n_rows']))
            else:
                event_day_acc.append((r['accuracy'], r['n_rows']))

    if d_minus_1_acc:
        d1_weighted = sum(a * n for a, n in d_minus_1_acc) / sum(n for _, n in d_minus_1_acc)
        print(f"- **D-1 weighted accuracy**: {d1_weighted*100:.1f}%")

    if event_day_acc:
        d_weighted = sum(a * n for a, n in event_day_acc) / sum(n for _, n in event_day_acc)
        print(f"- **Event day weighted accuracy**: {d_weighted*100:.1f}%")

    print("")

    # Recommendation
    print("## Recommendation")
    print("")
    if overall['accuracy'] < 0.40:
        print("**Model needs significant improvement** (accuracy < 40%).")
        print("Recommend: Increase Optuna trials to 150+, add `hours_to_event_close` feature.")
    elif overall['accuracy'] < 0.50:
        print("**Model needs improvement** (accuracy < 50%).")
        print("Recommend: Tune with more trials, consider time-bucket weighting.")
    else:
        print("**Model performance acceptable** for initial deployment.")

    print("")
    print("=" * 80)

    return {'results': results, 'overall': overall}


if __name__ == '__main__':
    result = run_health_check()
    sys.exit(0 if result else 1)
