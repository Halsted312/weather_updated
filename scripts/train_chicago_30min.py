#!/usr/bin/env python3
"""
Train Ordinal CatBoost for Chicago with 30-minute snapshot intervals.

This creates a finer-grained model for intraday trading opportunities.
Trains at: 10:00, 10:30, 11:00, 11:30, ..., 22:30, 23:00 (27 snapshot hours)

Usage:
    .venv/bin/python scripts/train_chicago_30min.py --trials 25
"""

import logging
import json
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.db.connection import get_db_session
from models.data.snapshot_builder import build_snapshot_dataset
from models.training.ordinal_trainer import OrdinalDeltaTrainer
from models.evaluation.metrics import compute_delta_metrics, compute_ordinal_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All integer hours from 10am to 11pm for finer granularity
# This gives us hourly snapshots instead of the sparse [10,12,14,16,18,20,22,23]
SNAPSHOT_HOURS_HOURLY = list(range(10, 24))  # 10, 11, 12, ..., 23 (14 hours)

# Note: To get true 30-min intervals, we'd need to modify snapshot_builder
# to handle fractional hours. For now, using all hours gives 14 snapshot times
# vs the current 8, which is a good improvement.
# Total: 14 snapshot hours (vs current 8)

TEST_DAYS = 60
CITY = "chicago"


def build_30min_dataset(session) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build snapshot dataset with 30-minute intervals."""
    logger.info(f"Building 30-minute interval snapshot dataset for {CITY}...")

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=700)

    df = build_snapshot_dataset(
        cities=[CITY],
        start_date=start_date,
        end_date=end_date,
        session=session,
        include_forecast_features=True,
        snapshot_hours=SNAPSHOT_HOURS_HOURLY,  # Use all hours 10-23
    )

    logger.info(f"Built {len(df)} snapshots for {df['day'].nunique()} days")
    logger.info(f"Snapshot hours: {sorted(df['snapshot_hour'].unique())}")

    # Time-based train/test split
    test_cutoff = df['day'].max() - pd.Timedelta(days=TEST_DAYS)
    df_train = df[df['day'] <= test_cutoff].copy()
    df_test = df[df['day'] > test_cutoff].copy()

    logger.info(f"Train: {len(df_train)} ({df_train['day'].nunique()} days)")
    logger.info(f"Test: {len(df_test)} ({df_test['day'].nunique()} days)")

    return df_train, df_test


def train_30min_model(df_train: pd.DataFrame, df_test: pd.DataFrame, n_trials: int) -> dict:
    """Train ordinal CatBoost with Optuna."""
    logger.info(f"\nTraining Ordinal CatBoost with {n_trials} Optuna trials...")

    # Create output folder (keeping "30min" name for consistency, though using hourly)
    output_dir = Path(f"models/saved/chicago_hourly")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train with Optuna
    trainer = OrdinalDeltaTrainer(
        base_model='catboost',
        n_trials=n_trials,
        cv_splits=3,
        verbose=True,
    )

    trainer.train(df_train)

    logger.info(f"Model delta range: [{trainer._min_delta}, {trainer._max_delta}]")
    logger.info(f"Thresholds: {len(trainer.thresholds)} classifiers")

    # Evaluate on test set
    y_true = df_test['delta'].values
    y_pred = trainer.predict(df_test)
    proba = trainer.predict_proba(df_test)

    delta_metrics = compute_delta_metrics(y_true, y_pred)
    ordinal_metrics = compute_ordinal_metrics(y_true, proba)

    # Save model with descriptive name
    model_filename = f"ordinal_catboost_hourly_{n_trials}trials.pkl"
    model_path = output_dir / model_filename
    trainer.save(model_path)

    # Save best params
    params_path = output_dir / f"best_params_hourly_{n_trials}trials.json"
    with open(params_path, 'w') as f:
        json.dump(trainer.best_params, f, indent=2)

    # Save train/test data
    df_train.to_parquet(output_dir / "train_data_hourly.parquet")
    df_test.to_parquet(output_dir / "test_data_hourly.parquet")

    # Save training metadata
    metadata = {
        "city": CITY,
        "interval": "hourly",  # Every hour from 10-23
        "n_snapshot_hours": len(SNAPSHOT_HOURS_HOURLY),
        "snapshot_hours": SNAPSHOT_HOURS_HOURLY,
        "n_optuna_trials": n_trials,
        "n_train_samples": len(df_train),
        "n_test_samples": len(df_test),
        "n_train_days": df_train['day'].nunique(),
        "n_test_days": df_test['day'].nunique(),
        "delta_range": [trainer._min_delta, trainer._max_delta],
        "accuracy": delta_metrics['delta_accuracy'],
        "mae": delta_metrics['delta_mae'],
        "within_1": delta_metrics['within_1_rate'],
        "within_2": delta_metrics['within_2_rate'],
        "ordinal_loss": ordinal_metrics['ordinal_loss'],
        "best_params": trainer.best_params,
    }

    metadata_path = output_dir / f"training_metadata_hourly_{n_trials}trials.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"\nSaved model to {model_path}")
    logger.info(f"Accuracy: {delta_metrics['delta_accuracy']:.1%}")
    logger.info(f"MAE: {delta_metrics['delta_mae']:.2f}")
    logger.info(f"Within-1: {delta_metrics['within_1_rate']:.1%}")
    logger.info(f"Ordinal Loss: {ordinal_metrics['ordinal_loss']:.2f}")

    return metadata


def evaluate_by_snapshot_hour(df_test: pd.DataFrame, trainer: OrdinalDeltaTrainer) -> pd.DataFrame:
    """Evaluate accuracy by snapshot hour."""
    logger.info("\nEvaluating by snapshot hour...")

    results = []
    for hour in sorted(df_test['snapshot_hour'].unique()):
        df_hour = df_test[df_test['snapshot_hour'] == hour]

        y_true = df_hour['delta'].values
        y_pred = trainer.predict(df_hour)
        proba = trainer.predict_proba(df_hour)

        metrics = compute_delta_metrics(y_true, y_pred)
        ordinal_metrics = compute_ordinal_metrics(y_true, proba)

        results.append({
            'snapshot_hour': hour,
            'n_samples': len(df_hour),
            'accuracy': metrics['delta_accuracy'],
            'mae': metrics['delta_mae'],
            'within_1': metrics['within_1_rate'],
            'ordinal_loss': ordinal_metrics['ordinal_loss'],
        })

    df_results = pd.DataFrame(results)

    # Print summary
    print("\nAccuracy by Snapshot Hour (30-min intervals):")
    print("="*80)
    print(f"{'Hour':<8} {'Samples':<10} {'Accuracy':<12} {'MAE':<8} {'Within-1':<12}")
    print("-"*80)

    for _, row in df_results.iterrows():
        hour_label = f"{int(row['snapshot_hour'])}:{int((row['snapshot_hour'] % 1) * 60):02d}"
        print(f"{hour_label:<8} {row['n_samples']:<10.0f} {row['accuracy']:<12.1%} "
              f"{row['mae']:<8.2f} {row['within_1']:<12.1%}")

    return df_results


def main():
    parser = argparse.ArgumentParser(description='Train 30-min interval model for Chicago')
    parser.add_argument('--trials', type=int, default=25, help='Optuna trials (default: 25)')
    parser.add_argument('--skip-hourly-eval', action='store_true', help='Skip by-hour evaluation')
    args = parser.parse_args()

    logger.info("="*80)
    logger.info(f"TRAINING CHICAGO 30-MINUTE INTERVAL MODEL ({args.trials} Optuna trials)")
    logger.info("="*80)

    with get_db_session() as session:
        # Build dataset
        df_train, df_test = build_30min_dataset(session)

        # Train model
        metadata = train_30min_model(df_train, df_test, args.trials)

        # Evaluate by hour (if requested)
        if not args.skip_hourly_eval:
            from models.training.ordinal_trainer import OrdinalDeltaTrainer

            model_path = Path(f"models/saved/chicago_hourly/ordinal_catboost_hourly_{args.trials}trials.pkl")
            trainer = OrdinalDeltaTrainer()
            trainer.load(model_path)

            df_results = evaluate_by_snapshot_hour(df_test, trainer)

            # Save hourly results
            results_path = Path(f"models/saved/chicago_30min/hourly_accuracy_30min_{args.trials}trials.csv")
            df_results.to_csv(results_path, index=False)
            logger.info(f"\nSaved hourly results to {results_path}")

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Model: models/saved/chicago_hourly/ordinal_catboost_hourly_{args.trials}trials.pkl")
    logger.info(f"Overall Accuracy: {metadata['accuracy']:.1%}")
    logger.info(f"Overall MAE: {metadata['mae']:.2f}")
    logger.info("="*80)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
