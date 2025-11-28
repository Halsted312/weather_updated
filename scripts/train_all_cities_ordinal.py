#!/usr/bin/env python3
"""
Train Ordinal CatBoost models for all 6 cities with Optuna tuning.

Creates city-specific models in models/saved/{city}/ folders.
"""

import logging
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

from src.db.connection import get_db_session
from models.data.snapshot_builder import build_snapshot_dataset
from models.training.ordinal_trainer import OrdinalDeltaTrainer
from models.evaluation.metrics import compute_delta_metrics, compute_ordinal_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# City configurations
CITIES = {
    'chicago': {'code': 'CHI', 'folder': 'chicago'},
    'austin': {'code': 'AUS', 'folder': 'austin'},
    'denver': {'code': 'DEN', 'folder': 'denver'},
    'los_angeles': {'code': 'LAX', 'folder': 'los_angeles'},
    'miami': {'code': 'MIA', 'folder': 'miami'},
    'philadelphia': {'code': 'PHL', 'folder': 'philadelphia'},
}

# Training parameters
N_OPTUNA_TRIALS = 30  # Number of Optuna trials per city
TEST_DAYS = 60  # Last N days for test set


def prepare_city_data(city: str, session) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and split snapshot data for a city."""
    logger.info(f"Building snapshot data for {city}...")

    # Use last ~2 years of data
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=700)  # ~2 years

    df = build_snapshot_dataset(
        cities=[city],
        start_date=start_date,
        end_date=end_date,
        session=session,
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


def train_city_model(city: str, df_train: pd.DataFrame, df_test: pd.DataFrame,
                     n_trials: int = N_OPTUNA_TRIALS) -> dict:
    """Train Ordinal CatBoost with Optuna for a city."""
    logger.info(f"Training Ordinal CatBoost for {city} with {n_trials} Optuna trials...")

    city_folder = Path(f"models/saved/{CITIES[city]['folder']}")
    city_folder.mkdir(parents=True, exist_ok=True)

    # Train with Optuna
    trainer = OrdinalDeltaTrainer(
        base_model='catboost',
        n_trials=n_trials,
        cv_splits=3,
        verbose=True,
    )
    trainer.train(df_train)

    # Evaluate
    y_true = df_test['delta'].values
    y_pred = trainer.predict(df_test)
    proba = trainer.predict_proba(df_test)

    delta_metrics = compute_delta_metrics(y_true, y_pred)
    ordinal_metrics = compute_ordinal_metrics(y_true, proba)

    # Save model
    model_path = city_folder / "ordinal_catboost_optuna.pkl"
    trainer.save(model_path)

    # Save best params as separate JSON
    params_path = city_folder / "best_params.json"
    with open(params_path, 'w') as f:
        json.dump(trainer.best_params, f, indent=2)

    # Save train/test data
    df_train.to_parquet(city_folder / "train_data.parquet")
    df_test.to_parquet(city_folder / "test_data.parquet")

    logger.info(f"  Saved model to {model_path}")
    logger.info(f"  Accuracy: {delta_metrics['delta_accuracy']:.1%}, MAE: {delta_metrics['delta_mae']:.2f}")

    return {
        'city': city,
        'n_train': len(df_train),
        'n_test': len(df_test),
        'n_train_days': df_train['day'].nunique(),
        'n_test_days': df_test['day'].nunique(),
        'accuracy': delta_metrics['delta_accuracy'],
        'mae': delta_metrics['delta_mae'],
        'within_1': delta_metrics['within_1_rate'],
        'within_2': delta_metrics['within_2_rate'],
        'ordinal_loss': ordinal_metrics['ordinal_loss'],
        'best_params': trainer.best_params,
    }


def main():
    """Train all cities and generate comparison report."""
    results = []

    with get_db_session() as session:
        for city in CITIES:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {city.upper()}")
            logger.info(f"{'='*60}")

            try:
                # Prepare data
                df_train, df_test = prepare_city_data(city, session)

                # Train model
                result = train_city_model(city, df_train, df_test)
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process {city}: {e}")
                results.append({'city': city, 'error': str(e)})

    # Generate comparison report
    logger.info("\n" + "="*80)
    logger.info("CROSS-CITY COMPARISON REPORT")
    logger.info("="*80)

    print("\n")
    print(f"{'City':<15} {'Accuracy':>10} {'MAE':>8} {'Within 1':>10} {'Within 2':>10} {'Ord Loss':>10}")
    print("-"*70)

    for r in results:
        if 'error' in r:
            print(f"{r['city']:<15} ERROR: {r['error']}")
        else:
            print(f"{r['city']:<15} {r['accuracy']*100:>9.1f}% {r['mae']:>8.2f} "
                  f"{r['within_1']*100:>9.1f}% {r['within_2']*100:>9.1f}% "
                  f"{r['ordinal_loss']:>10.2f}")

    # Save full results
    report_path = Path("models/reports/cross_city_comparison.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter out non-serializable fields for JSON
    json_results = []
    for r in results:
        if 'best_params' in r:
            r_copy = r.copy()
            r_copy['best_params'] = r_copy['best_params'] if r_copy['best_params'] else {}
            json_results.append(r_copy)
        else:
            json_results.append(r)

    with open(report_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    logger.info(f"\nSaved report to {report_path}")


if __name__ == '__main__':
    main()
