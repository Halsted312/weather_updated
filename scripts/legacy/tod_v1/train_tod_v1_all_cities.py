#!/usr/bin/env python3
"""
Train Time-of-Day (TOD) Ordinal CatBoost models for all 6 cities.

Uses 15-minute snapshot intervals (56 snapshots per day from 10:00 to 23:45).
Includes time-of-day features: hour, minute, cyclical encodings.

This creates tod_v1 models that can predict at arbitrary timestamps without
snapping to nearest hour.

Usage:
    # 15-minute intervals (default)
    .venv/bin/python scripts/train_tod_v1_all_cities.py --trials 80

    # 5-minute intervals (more granular)
    .venv/bin/python scripts/train_tod_v1_all_cities.py --trials 80 --interval 5
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

from src.db.connection import get_db_session
from models.data.tod_dataset_builder import build_tod_snapshot_dataset
from models.training.ordinal_trainer import OrdinalDeltaTrainer
from models.evaluation.metrics import compute_delta_metrics, compute_ordinal_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# City configurations
CITIES = {
    'chicago': {'code': 'CHI'},
    'austin': {'code': 'AUS'},
    'denver': {'code': 'DEN'},
    'los_angeles': {'code': 'LAX'},
    'miami': {'code': 'MIA'},
    'philadelphia': {'code': 'PHL'},
}

TEST_DAYS = 60


def prepare_tod_data(city: str, session, interval_min: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build tod snapshot dataset for a city."""
    logger.info(f"Building {interval_min}-min tod dataset for {city}...")

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=700)

    df = build_tod_snapshot_dataset(
        cities=[city],
        start_date=start_date,
        end_date=end_date,
        session=session,
        snapshot_interval_min=interval_min,
        include_forecast_features=True,
    )

    logger.info(f"  Built {len(df)} snapshots for {df['day'].nunique()} days")

    # Time-based train/test split
    test_cutoff = df['day'].max() - pd.Timedelta(days=TEST_DAYS)
    df_train = df[df['day'] <= test_cutoff].copy()
    df_test = df[df['day'] > test_cutoff].copy()

    logger.info(f"  Train: {len(df_train)} ({df_train['day'].nunique()} days), "
                f"Test: {len(df_test)} ({df_test['day'].nunique()} days)")

    return df_train, df_test


def train_tod_model(city: str, df_train: pd.DataFrame, df_test: pd.DataFrame,
                    n_trials: int, interval_min: int) -> dict:
    """Train TOD Ordinal CatBoost model for a city."""
    logger.info(f"\nTraining TOD Ordinal CatBoost for {city} with {n_trials} Optuna trials...")

    # Create city-specific tod_v1 folder
    city_folder = Path(f"models/saved/{city}_tod_v1")
    city_folder.mkdir(parents=True, exist_ok=True)

    # Train with Optuna
    trainer = OrdinalDeltaTrainer(
        base_model='catboost',
        n_trials=n_trials,
        cv_splits=3,
        verbose=False,  # Reduce logging for multi-city
    )
    trainer.train(df_train)

    logger.info(f"  Delta range: [{trainer._min_delta}, {trainer._max_delta}]")
    logger.info(f"  Classifiers: {len(trainer.classifiers)}")

    # Evaluate
    y_true = df_test['delta'].values
    y_pred = trainer.predict(df_test)
    proba = trainer.predict_proba(df_test)

    delta_metrics = compute_delta_metrics(y_true, y_pred)
    ordinal_metrics = compute_ordinal_metrics(y_true, proba)

    # Save model
    model_path = city_folder / "ordinal_catboost_tod_v1.pkl"
    trainer.save(model_path)

    # Save best params
    params_path = city_folder / "best_params.json"
    with open(params_path, 'w') as f:
        json.dump(trainer.best_params, f, indent=2)

    # Save train/test data
    df_train.to_parquet(city_folder / "train_data.parquet")
    df_test.to_parquet(city_folder / "test_data.parquet")

    # Save training metadata
    n_snapshots_per_day = ((23 - 10) * 60 + 45) // interval_min + 1  # 10:00 to 23:45

    metadata = {
        "model_variant": "tod_v1",
        "city": city,
        "snapshot_interval_min": interval_min,
        "n_snapshots_per_day": n_snapshots_per_day,
        "time_of_day_features": True,
        "delta_range": [trainer._min_delta, trainer._max_delta],
        "n_optuna_trials": n_trials,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "n_train_days": df_train['day'].nunique(),
        "n_test_days": df_test['day'].nunique(),
        "accuracy": delta_metrics['delta_accuracy'],
        "mae": delta_metrics['delta_mae'],
        "within_1": delta_metrics['within_1_rate'],
        "within_2": delta_metrics['within_2_rate'],
        "ordinal_loss": ordinal_metrics['ordinal_loss'],
        "best_optuna_auc": trainer.study.best_value if trainer.study else None,
        "best_params": trainer.best_params,
    }

    metadata_path = city_folder / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"  Saved to {model_path}")
    logger.info(f"  Accuracy: {delta_metrics['delta_accuracy']:.1%}, MAE: {delta_metrics['delta_mae']:.2f}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description='Train TOD v1 models for all cities')
    parser.add_argument('--trials', type=int, default=80, help='Optuna trials (default: 80)')
    parser.add_argument('--interval', type=int, default=15, help='Snapshot interval in minutes (default: 15)')
    parser.add_argument('--cities', nargs='+', default=None, help='Specific cities (default: all)')
    args = parser.parse_args()

    cities_to_train = args.cities if args.cities else list(CITIES.keys())

    logger.info("="*80)
    logger.info(f"TRAINING TOD V1 MODELS - {args.interval}-MIN INTERVALS ({args.trials} Optuna trials)")
    logger.info(f"Cities: {', '.join(cities_to_train)}")
    logger.info("="*80)

    results = []

    with get_db_session() as session:
        for city in cities_to_train:
            logger.info(f"\n{'='*80}")
            logger.info(f"PROCESSING {city.upper()}")
            logger.info(f"{'='*80}")

            try:
                # Build tod dataset
                df_train, df_test = prepare_tod_data(city, session, args.interval)

                # Train model
                result = train_tod_model(city, df_train, df_test, args.trials, args.interval)
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process {city}: {e}", exc_info=True)
                results.append({'city': city, 'error': str(e)})

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING SUMMARY - TOD V1 MODELS")
    logger.info(f"{'='*80}\n")

    print(f"{'City':<15} {'Accuracy':>10} {'MAE':>8} {'Within-1':>10} {'Snapshots/Day':>14}")
    print("-"*65)

    for r in results:
        if 'error' in r:
            print(f"{r['city']:<15} {'ERROR':<10} {r['error']}")
        else:
            print(f"{r['city']:<15} {r['accuracy']*100:>9.1f}% {r['mae']:>8.2f} "
                  f"{r['within_1']*100:>9.1f}% {r['n_snapshots_per_day']:>14}")

    # Save summary
    report_path = Path(f"models/reports/tod_v1_{args.interval}min_summary.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nSaved summary to {report_path}")
    logger.info("\n" + "="*80)
    logger.info("TOD V1 TRAINING COMPLETE!")
    logger.info("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
