#!/usr/bin/env python3
"""
Train Ordinal CatBoost for ALL 6 cities with hourly intervals and 80 Optuna trials.

Creates fine-grained models for intraday trading.
Snapshot hours: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

Usage:
    .venv/bin/python scripts/train_all_cities_hourly.py
"""

import logging
import json
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.db.connection import get_db_session
from models.data.snapshot_builder import build_snapshot_dataset
from models.training.ordinal_trainer import OrdinalDeltaTrainer
from models.evaluation.metrics import compute_delta_metrics, compute_ordinal_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# City configurations
CITIES = {
    'chicago': {'code': 'CHI', 'folder': 'chicago_hourly80'},
    'austin': {'code': 'AUS', 'folder': 'austin_hourly80'},
    'denver': {'code': 'DEN', 'folder': 'denver_hourly80'},
    'los_angeles': {'code': 'LAX', 'folder': 'los_angeles_hourly80'},
    'miami': {'code': 'MIA', 'folder': 'miami_hourly80'},
    'philadelphia': {'code': 'PHL', 'folder': 'philadelphia_hourly80'},
}

# Hourly intervals: [10, 11, 12, ..., 23] = 14 hours
SNAPSHOT_HOURS = list(range(10, 24))
N_OPTUNA_TRIALS = 80
TEST_DAYS = 60


def prepare_city_data(city: str, session) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build hourly snapshot dataset for a city."""
    logger.info(f"Building hourly snapshot dataset for {city}...")

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=700)

    df = build_snapshot_dataset(
        cities=[city],
        start_date=start_date,
        end_date=end_date,
        session=session,
        include_forecast_features=True,
        snapshot_hours=SNAPSHOT_HOURS,
    )

    logger.info(f"  Built {len(df)} snapshots for {df['day'].nunique()} days")

    # Time-based train/test split
    test_cutoff = df['day'].max() - pd.Timedelta(days=TEST_DAYS)
    df_train = df[df['day'] <= test_cutoff].copy()
    df_test = df[df['day'] > test_cutoff].copy()

    logger.info(f"  Train: {len(df_train)} ({df_train['day'].nunique()} days), "
                f"Test: {len(df_test)} ({df_test['day'].nunique()} days)")

    return df_train, df_test


def train_city_model(city: str, df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """Train Ordinal CatBoost with 80 Optuna trials for a city."""
    logger.info(f"\nTraining Ordinal CatBoost for {city} with {N_OPTUNA_TRIALS} Optuna trials...")

    city_folder = Path(f"models/saved/{CITIES[city]['folder']}")
    city_folder.mkdir(parents=True, exist_ok=True)

    # Train with Optuna
    trainer = OrdinalDeltaTrainer(
        base_model='catboost',
        n_trials=N_OPTUNA_TRIALS,
        cv_splits=3,
        verbose=False,  # Less logging for multi-city
    )
    trainer.train(df_train)

    logger.info(f"  Model delta range: [{trainer._min_delta}, {trainer._max_delta}]")
    logger.info(f"  Classifiers: {len(trainer.classifiers)}")

    # Evaluate
    y_true = df_test['delta'].values
    y_pred = trainer.predict(df_test)
    proba = trainer.predict_proba(df_test)

    delta_metrics = compute_delta_metrics(y_true, y_pred)
    ordinal_metrics = compute_ordinal_metrics(y_true, proba)

    # Save model
    model_path = city_folder / "ordinal_catboost_hourly_80trials.pkl"
    trainer.save(model_path)

    # Save best params
    params_path = city_folder / "best_params.json"
    with open(params_path, 'w') as f:
        json.dump(trainer.best_params, f, indent=2)

    # Save train/test data
    df_train.to_parquet(city_folder / "train_data.parquet")
    df_test.to_parquet(city_folder / "test_data.parquet")

    # Save metadata
    metadata = {
        "city": city,
        "interval": "hourly",
        "snapshot_hours": SNAPSHOT_HOURS,
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "delta_range": [trainer._min_delta, trainer._max_delta],
        "n_train": len(df_train),
        "n_test": len(df_test),
        "accuracy": delta_metrics['delta_accuracy'],
        "mae": delta_metrics['delta_mae'],
        "within_1": delta_metrics['within_1_rate'],
        "ordinal_loss": ordinal_metrics['ordinal_loss'],
        "best_optuna_auc": trainer.study.best_value if trainer.study else None,
    }

    metadata_path = city_folder / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"  Saved to {model_path}")
    logger.info(f"  Accuracy: {delta_metrics['delta_accuracy']:.1%}, MAE: {delta_metrics['delta_mae']:.2f}")

    return metadata


def main():
    """Train all cities with hourly intervals."""
    logger.info("="*80)
    logger.info(f"TRAINING ALL 6 CITIES - HOURLY INTERVALS ({N_OPTUNA_TRIALS} Optuna trials)")
    logger.info("="*80)

    results = []

    with get_db_session() as session:
        for city in CITIES:
            logger.info(f"\n{'='*80}")
            logger.info(f"PROCESSING {city.upper()}")
            logger.info(f"{'='*80}")

            try:
                df_train, df_test = prepare_city_data(city, session)
                result = train_city_model(city, df_train, df_test)
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process {city}: {e}", exc_info=True)
                results.append({'city': city, 'error': str(e)})

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING SUMMARY - ALL CITIES")
    logger.info(f"{'='*80}\n")

    print(f"{'City':<15} {'Accuracy':>10} {'MAE':>8} {'Within-1':>10} {'Folder':<25}")
    print("-"*75)

    for r in results:
        if 'error' in r:
            print(f"{r['city']:<15} {'ERROR':<10} {r['error']}")
        else:
            print(f"{r['city']:<15} {r['accuracy']*100:>9.1f}% {r['mae']:>8.2f} "
                  f"{r['within_1']*100:>9.1f}% {CITIES[r['city']]['folder']:<25}")

    # Save summary
    report_path = Path("models/reports/hourly_80trials_summary.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nSaved summary to {report_path}")
    logger.info("\n" + "="*80)
    logger.info("ALL CITIES TRAINING COMPLETE!")
    logger.info("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
