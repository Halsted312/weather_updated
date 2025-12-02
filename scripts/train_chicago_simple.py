#!/usr/bin/env python3
"""Train Chicago ordinal CatBoost model (no Optuna).

Verifies that the 3 new multi-horizon features are working:
- fcst_multi_std: Std dev of T-1 through T-6 tempmax forecasts
- fcst_multi_mean: Mean/consensus forecast high
- fcst_multi_drift: T-1 minus T-6 (warming/cooling trend)

Usage:
    PYTHONPATH=. .venv/bin/python scripts/train_chicago_simple.py
"""

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import get_db_session
from models.data.dataset import DatasetConfig, build_dataset
from models.data.loader import get_available_date_range
from models.training.ordinal_trainer import OrdinalDeltaTrainer
from models.evaluation.metrics import compute_delta_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuration
CITY = "chicago"
START_DATE = date(2024, 1, 1)
HOLDOUT_DAYS = 60
OUTPUT_DIR = Path("models/saved/chicago")


def main():
    logger.info("=" * 60)
    logger.info("Chicago Simple CatBoost Training (no Optuna)")
    logger.info("=" * 60)

    with get_db_session() as session:
        # Get available date range
        min_date, max_date = get_available_date_range(session, CITY)
        logger.info(f"Available data: {min_date} to {max_date}")

        # Use max_date from DB
        end_date = max_date
        test_start = end_date - timedelta(days=HOLDOUT_DAYS)

        logger.info(f"Training: {START_DATE} to {test_start - timedelta(days=1)}")
        logger.info(f"Testing:  {test_start} to {end_date}")

        # Build dataset with multi-horizon features enabled
        config = DatasetConfig(
            time_window="market_clock",
            snapshot_interval_min=5,
            include_forecast=True,
            include_multi_horizon=True,  # <-- The 3 new features
            include_market=False,
            include_station_city=False,
            include_meteo=True,
        )

        logger.info("Building training dataset...")
        df_train = build_dataset(
            cities=[CITY],
            start_date=START_DATE,
            end_date=test_start - timedelta(days=1),
            config=config,
            session=session,
        )
        logger.info(f"Training samples: {len(df_train):,}")
        logger.info(f"Training days: {df_train['day'].nunique()}")

        logger.info("Building test dataset...")
        df_test = build_dataset(
            cities=[CITY],
            start_date=test_start,
            end_date=end_date,
            config=config,
            session=session,
        )
        logger.info(f"Test samples: {len(df_test):,}")
        logger.info(f"Test days: {df_test['day'].nunique()}")

    # Verify multi-horizon features are present
    print("\n" + "=" * 60)
    print("MULTI-HORIZON FEATURES CHECK")
    print("=" * 60)

    multi_cols = [c for c in df_train.columns if "fcst_multi" in c]
    logger.info(f"Multi-horizon features found: {multi_cols}")

    for col in multi_cols:
        non_null = df_train[col].notna().sum()
        pct = 100.0 * non_null / len(df_train)
        logger.info(f"  {col}: {non_null:,}/{len(df_train):,} non-null ({pct:.1f}%)")

    # Show sample values
    if multi_cols:
        print("\nSample multi-horizon values (first 10 rows):")
        print(df_train[multi_cols].head(10).to_string())

    # Train model (no Optuna - n_trials=0)
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    trainer = OrdinalDeltaTrainer(
        base_model="catboost",
        n_trials=0,  # No Optuna
        verbose=True,
    )
    trainer.train(df_train, df_val=df_test)

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    y_pred = trainer.predict(df_test)
    y_true = df_test["delta"].values

    metrics = compute_delta_metrics(y_true, y_pred)

    print("\nTest Set Metrics:")
    print("-" * 40)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "ordinal_catboost_simple.pkl"
    trainer.save(model_path)
    logger.info(f"Saved model to {model_path}")

    # Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    fi = trainer.get_feature_importance()
    if fi is not None:
        print("\nTop 25 Features:")
        print(fi.head(25).to_string())

        # Show multi-horizon features importance
        multi_fi = fi[fi["feature"].str.contains("fcst_multi", na=False)]
        if not multi_fi.empty:
            print("\n" + "-" * 40)
            print("Multi-horizon feature importance:")
            print(multi_fi.to_string())

            # Calculate rank
            for _, row in multi_fi.iterrows():
                rank = fi[fi["feature"] == row["feature"]].index[0] + 1
                print(f"  {row['feature']}: rank {rank} of {len(fi)}")
    else:
        logger.warning("Could not get feature importance")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
