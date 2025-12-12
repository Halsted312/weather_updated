#!/usr/bin/env python3
"""
Train Ordinal CatBoost models for Los Angeles and Miami with fixed dynamic thresholds.

This script trains the two cities that previously failed due to hard-coded threshold issues.
They have delta range [-1, +10] instead of [-2, +10].
"""

import logging
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Add project root to path (scripts/training/ -> 2 levels up)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.db.connection import get_db_session
from models.data.snapshot_builder import build_snapshot_dataset
from models.training.ordinal_trainer import OrdinalDeltaTrainer
from models.evaluation.metrics import compute_delta_metrics, compute_ordinal_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cities to train
CITIES = {
    'los_angeles': {'code': 'LAX', 'folder': 'los_angeles'},
    'miami': {'code': 'MIA', 'folder': 'miami'},
}

# Training parameters
N_OPTUNA_TRIALS = 30
TEST_DAYS = 60


def prepare_city_data(city: str, session) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and split snapshot data for a city."""
    logger.info(f"Building snapshot data for {city}...")

    # Use last ~2 years of data
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=700)

    df = build_snapshot_dataset(
        cities=[city],
        start_date=start_date,
        end_date=end_date,
        session=session,
        include_forecast_features=True,
    )

    logger.info(f"  Built {len(df)} snapshots for {df['day'].nunique()} days")

    # Check delta distribution
    delta_dist = df['delta'].value_counts().sort_index()
    logger.info(f"  Delta distribution:")
    for delta, count in delta_dist.items():
        logger.info(f"    delta={delta:+d}: {count} ({count/len(df)*100:.1f}%)")

    # Time-based train/test split
    test_cutoff = df['day'].max() - pd.Timedelta(days=TEST_DAYS)
    df_train = df[df['day'] <= test_cutoff].copy()
    df_test = df[df['day'] > test_cutoff].copy()

    logger.info(f"  Train: {len(df_train)} ({df_train['day'].nunique()} days), "
                f"Test: {len(df_test)} ({df_test['day'].nunique()} days)")

    return df_train, df_test


def train_city_model(city: str, df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """Train Ordinal CatBoost with Optuna for a city."""
    logger.info(f"\nTraining Ordinal CatBoost for {city} with {N_OPTUNA_TRIALS} Optuna trials...")

    city_folder = Path(f"models/saved/{CITIES[city]['folder']}")
    city_folder.mkdir(parents=True, exist_ok=True)

    # Train with Optuna
    trainer = OrdinalDeltaTrainer(
        base_model='catboost',
        n_trials=N_OPTUNA_TRIALS,
        cv_splits=3,
        verbose=True,
    )
    trainer.train(df_train)

    # Log model details
    logger.info(f"  Model delta range: [{trainer._min_delta}, {trainer._max_delta}]")
    logger.info(f"  Model delta classes: {trainer._delta_classes}")
    logger.info(f"  Number of threshold classifiers: {len(trainer.classifiers)}")
    logger.info(f"  Thresholds: {trainer.thresholds}")

    # Count constant predictors
    n_constant = sum(1 for c in trainer.classifiers.values() if isinstance(c, dict))
    if n_constant > 0:
        logger.info(f"  Constant predictors: {n_constant}")

    # Evaluate
    y_true = df_test['delta'].values
    y_pred = trainer.predict(df_test)
    proba = trainer.predict_proba(df_test)

    # Check proba shape
    logger.info(f"  Proba shape: {proba.shape} (should be ({len(df_test)}, 13))")
    logger.info(f"  Proba sum check: min={proba.sum(axis=1).min():.4f}, max={proba.sum(axis=1).max():.4f} (should be ~1.0)")

    delta_metrics = compute_delta_metrics(y_true, y_pred)
    ordinal_metrics = compute_ordinal_metrics(y_true, proba)

    # Save model
    model_path = city_folder / "ordinal_catboost_optuna.pkl"
    trainer.save(model_path)

    # Save best params
    params_path = city_folder / "best_params.json"
    with open(params_path, 'w') as f:
        json.dump(trainer.best_params, f, indent=2)

    # Save train/test data
    df_train.to_parquet(city_folder / "train_data.parquet")
    df_test.to_parquet(city_folder / "test_data.parquet")

    logger.info(f"  Saved model to {model_path}")
    logger.info(f"  Accuracy: {delta_metrics['delta_accuracy']:.1%}, MAE: {delta_metrics['delta_mae']:.2f}")
    logger.info(f"  Within-1: {delta_metrics['within_1_rate']:.1%}, Within-2: {delta_metrics['within_2_rate']:.1%}")
    logger.info(f"  Ordinal Loss: {ordinal_metrics['ordinal_loss']:.2f}")

    return {
        'city': city,
        'delta_range': [trainer._min_delta, trainer._max_delta],
        'n_classifiers': len(trainer.classifiers),
        'n_constant_classifiers': n_constant,
        'n_train': len(df_train),
        'n_test': len(df_test),
        'accuracy': delta_metrics['delta_accuracy'],
        'mae': delta_metrics['delta_mae'],
        'within_1': delta_metrics['within_1_rate'],
        'within_2': delta_metrics['within_2_rate'],
        'ordinal_loss': ordinal_metrics['ordinal_loss'],
        'best_params': trainer.best_params,
    }


def main():
    """Train LA and Miami models."""
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
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*80}")

    for r in results:
        if 'error' in r:
            logger.error(f"{r['city']}: ERROR - {r['error']}")
        else:
            logger.info(f"\n{r['city'].upper()}:")
            logger.info(f"  Delta range: {r['delta_range']}")
            logger.info(f"  Classifiers: {r['n_classifiers']} ({r['n_constant_classifiers']} constant)")
            logger.info(f"  Accuracy: {r['accuracy']:.1%}, MAE: {r['mae']:.2f}")
            logger.info(f"  Within-1: {r['within_1']:.1%}, Within-2: {r['within_2']:.1%}")
            logger.info(f"  Ordinal Loss: {r['ordinal_loss']:.2f}")

    # Save results
    report_path = Path("models/reports/la_miami_training.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    json_results = []
    for r in results:
        r_copy = r.copy()
        if 'best_params' in r_copy:
            r_copy['best_params'] = r_copy['best_params'] if r_copy['best_params'] else {}
        json_results.append(r_copy)

    with open(report_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    logger.info(f"\nSaved report to {report_path}")


if __name__ == '__main__':
    main()
