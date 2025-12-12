#!/usr/bin/env python3
"""Train ordinal CatBoost model with Optuna tuning for any city.

Generic training script that works for all 6 cities. Uses parallel dataset
building, then runs Optuna hyperparameter optimization.

Usage:
    # Test run (3 trials)
    PYTHONPATH=. .venv/bin/python scripts/train_city_ordinal_optuna.py --city austin --trials 3

    # Full training (100 trials)
    PYTHONPATH=. .venv/bin/python scripts/train_city_ordinal_optuna.py --city austin --trials 100

    # Use cached dataset
    PYTHONPATH=. .venv/bin/python scripts/train_city_ordinal_optuna.py --city austin --trials 100 --use-cached
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Add project root to path (scripts/training/core/ -> 3 levels up)
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Supported cities
VALID_CITIES = ["chicago", "austin", "denver", "los_angeles", "miami", "philadelphia"]

# Default configuration
# Leave ~4 cores for the system by default
DEFAULT_WORKERS = max(1, (os.cpu_count() or 8) - 4)
DEFAULT_TRIALS = 80
DEFAULT_CV_SPLITS = 4
DEFAULT_HOLDOUT_PCT = 0.20  # 80/20 split


@dataclass
class ChunkResult:
    """Result from processing a date chunk."""
    chunk_id: int
    start_date: date
    end_date: date
    n_rows: int
    n_days: int
    df: Optional[pd.DataFrame] = None
    error: Optional[str] = None


def process_chunk(
    chunk_id: int,
    city: str,
    start_date: date,
    end_date: date,
    include_multi_horizon: bool = True,
    include_station_city: bool = True,
) -> ChunkResult:
    """Process a chunk of dates. Each worker creates its own DB connection."""
    try:
        import src.db.connection as db_conn
        from models.data.dataset import DatasetConfig, build_dataset

        # Reset global engine state for fresh connection
        db_conn._engine = None
        db_conn._SessionLocal = None

        config = DatasetConfig(
            time_window="market_clock",
            snapshot_interval_min=5,
            include_forecast=True,
            include_multi_horizon=include_multi_horizon,
            include_market=True,
            include_station_city=include_station_city,  # ENABLED
            include_meteo=True,
            include_more_apis=True,  # NOAA guidance (NBM, HRRR, NDFD)
        )

        with db_conn.get_db_session() as session:
            df = build_dataset(
                cities=[city],
                start_date=start_date,
                end_date=end_date,
                config=config,
                session=session,
            )

        return ChunkResult(
            chunk_id=chunk_id,
            start_date=start_date,
            end_date=end_date,
            n_rows=len(df),
            n_days=df['day'].nunique() if 'day' in df.columns and len(df) > 0 else 0,
            df=df,
        )
    except Exception as e:
        import traceback
        return ChunkResult(
            chunk_id=chunk_id,
            start_date=start_date,
            end_date=end_date,
            n_rows=0,
            n_days=0,
            error=f"{e}\n{traceback.format_exc()}",
        )


def split_date_range(start_date: date, end_date: date, n_chunks: int) -> list[tuple[date, date]]:
    """Split date range into n_chunks roughly equal parts."""
    total_days = (end_date - start_date).days + 1
    chunk_size = max(1, total_days // n_chunks)

    chunks = []
    current = start_date

    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_size - 1), end_date)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)

    return chunks


def build_dataset_parallel(
    city: str,
    start_date: date,
    end_date: date,
    n_workers: int = DEFAULT_WORKERS,
    include_station_city: bool = True,
) -> pd.DataFrame:
    """Build dataset using parallel processing."""
    total_days = (end_date - start_date).days + 1
    logger.info(f"Building dataset for {city}: {start_date} to {end_date} ({total_days} days)")
    logger.info(f"Using {n_workers} parallel workers")
    logger.info(f"include_station_city={include_station_city}")

    n_chunks = min(n_workers * 2, total_days)
    chunks = split_date_range(start_date, end_date, n_chunks)
    logger.info(f"Split into {len(chunks)} chunks")

    results = []
    completed = 0
    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                process_chunk, i, city, chunk_start, chunk_end,
                True, include_station_city
            ): i
            for i, (chunk_start, chunk_end) in enumerate(chunks)
        }

        for future in as_completed(futures):
            chunk_id = futures[future]
            result = future.result()
            results.append(result)
            completed += 1

            if result.error:
                logger.error(f"Chunk {chunk_id} failed: {result.error[:200]}...")
            else:
                logger.info(
                    f"Chunk {completed}/{len(chunks)}: "
                    f"{result.start_date} to {result.end_date}, "
                    f"{result.n_rows:,} rows, {result.n_days} days"
                )

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"All chunks completed in {elapsed:.1f}s")

    dfs = [r.df for r in results if r.df is not None and len(r.df) > 0]
    if not dfs:
        raise RuntimeError("No data collected from any chunk!")

    df_combined = pd.concat(dfs, ignore_index=True)

    if 'day' in df_combined.columns and 'cutoff_time' in df_combined.columns:
        df_combined = df_combined.sort_values(['day', 'cutoff_time']).reset_index(drop=True)

    logger.info(f"Combined dataset: {len(df_combined):,} rows, {df_combined['day'].nunique()} days")
    return df_combined


def main():
    parser = argparse.ArgumentParser(description='Train ordinal CatBoost model with Optuna for any city')
    parser.add_argument('--city', type=str, required=True, choices=VALID_CITIES,
                        help='City to train model for')
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS,
                        help='Number of Optuna trials')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help='Parallel workers for dataset building')
    parser.add_argument('--use-cached', action='store_true',
                        help='Use cached parquet files if available')
    parser.add_argument('--cv-splits', type=int, default=DEFAULT_CV_SPLITS,
                        help='Cross-validation splits')
    parser.add_argument('--holdout-pct', type=float, default=DEFAULT_HOLDOUT_PCT,
                        help='Holdout percentage for test set (default 0.20 = 20%%)')
    parser.add_argument('--no-station-city', action='store_true',
                        help='Disable station-city features')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Training start date (YYYY-MM-DD). Default: use all available data')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Training end date (YYYY-MM-DD). Default: use all available data')
    parser.add_argument('--objective', type=str, default='weighted_auc',
                        choices=['auc', 'within2', 'weighted_auc'],
                        help='Optuna optimization objective (default: weighted_auc)')
    parser.add_argument('--cache-dir', type=str, default='models/saved',
                        help='Directory containing cached parquet files (train_data_full.parquet + test_data_full.parquet per city)')
    args = parser.parse_args()

    city = args.city
    include_station_city = not args.no_station_city
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date() if args.start_date else None
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date() if args.end_date else None

    logger.info("=" * 60)
    logger.info(f"{city.upper()} Optuna Training ({args.trials} trials)")
    logger.info(f"Optimization objective: {args.objective}")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path(f"models/saved/{city}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Paths for pre-split parquets
    cache_dir = Path(args.cache_dir)
    train_parquet = cache_dir / city / "train_data_full.parquet"
    test_parquet = cache_dir / city / "test_data_full.parquet"

    # Load from pre-split parquets if --use-cached and files exist
    if args.use_cached and train_parquet.exists() and test_parquet.exists():
        logger.info(f"Loading cached train/test datasets from: {cache_dir / city}")
        logger.info(f"  Train: {train_parquet}")
        logger.info(f"  Test:  {test_parquet}")

        df_train = pd.read_parquet(train_parquet)
        df_test = pd.read_parquet(test_parquet)

        logger.info(f"Loaded train: {len(df_train):,} rows, {len(df_train.columns)} columns")
        logger.info(f"Loaded test:  {len(df_test):,} rows, {len(df_test.columns)} columns")

        # Debug: Check for NOAA feature columns
        noaa_cols = [
            "nbm_peak_window_max_f", "hrrr_peak_window_max_f",
            "nbm_t15_z_30d_f", "hrrr_t15_z_30d_f", "hrrr_minus_nbm_t15_z_30d_f",
        ]
        logger.info("\nNOAA feature columns check:")
        for col in noaa_cols:
            if col in df_train.columns:
                non_null = df_train[col].notna().sum()
                pct = 100.0 * non_null / len(df_train)
                logger.info(f"  ✓ {col}: {non_null:,}/{len(df_train):,} non-null ({pct:.1f}%)")
            else:
                logger.warning(f"  ✗ {col}: MISSING from parquet")

        # Ensure 'day' column exists
        date_col = 'event_date' if 'event_date' in df_train.columns else 'day'
        if date_col == 'event_date' and 'day' not in df_train.columns:
            df_train['day'] = pd.to_datetime(df_train['event_date']).dt.date
            df_test['day'] = pd.to_datetime(df_test['event_date']).dt.date
        elif 'day' in df_train.columns:
            df_train['day'] = pd.to_datetime(df_train['day']).dt.date
            df_test['day'] = pd.to_datetime(df_test['day']).dt.date

        # Apply requested date filters to cached data and re-split if needed
        if start_date or end_date:
            logger.info("Applying date filters to cached datasets")
            logger.info(f"  start_date: {start_date}, end_date: {end_date}")
            df_all = pd.concat([df_train, df_test], ignore_index=True)
            if start_date:
                df_all = df_all[df_all['day'] >= start_date]
            if end_date:
                df_all = df_all[df_all['day'] <= end_date]

            if df_all.empty:
                logger.error("No data remaining after applying date filters to cached parquets")
                return 1

            # Recompute split so cached path honors holdout pct and date bounds
            min_date = df_all['day'].min()
            max_date = df_all['day'].max()
            total_days = (max_date - min_date).days + 1
            holdout_days = int(total_days * args.holdout_pct)
            test_start = max_date - timedelta(days=holdout_days)

            df_train = df_all[df_all['day'] < test_start].reset_index(drop=True)
            df_test = df_all[df_all['day'] >= test_start].reset_index(drop=True)

            if df_train.empty or df_test.empty:
                logger.error("Insufficient data after splitting cached parquets with requested date range")
                logger.error(f"  Train rows: {len(df_train)}, Test rows: {len(df_test)}")
                return 1

        min_date = df_train['day'].min()
        max_date = df_test['day'].max()
        holdout_days = df_test['day'].nunique()
        total_days = df_train['day'].nunique() + holdout_days
        test_start = df_test['day'].min()

        logger.info(f"\nData range: {min_date} to {max_date}")
        logger.info(f"Training: {df_train['day'].min()} to {df_train['day'].max()} ({df_train['day'].nunique()} days)")
        logger.info(f"Testing:  {test_start} to {max_date} ({holdout_days} days)")

    elif args.use_cached:
        # --use-cached was specified but files don't exist
        logger.error(f"--use-cached specified but parquet files not found:")
        logger.error(f"  Expected: {train_parquet}")
        logger.error(f"  Expected: {test_parquet}")
        logger.error("Either build the dataset first or remove --use-cached to rebuild from DB")
        return 1

    else:
        # Build from database (original behavior)
        from src.db import get_db_session
        from models.data.loader import get_available_date_range

        with get_db_session() as session:
            min_date, max_date = get_available_date_range(session, city)

        if min_date is None or max_date is None:
            logger.error(f"No data available for {city}")
            return 1

        # Apply date filtering if specified
        if start_date:
            min_date = max(min_date, start_date)
        if end_date:
            max_date = min(max_date, end_date)

        logger.info(f"Available data: {min_date} to {max_date}")

        # Calculate train/test split (80/20 by days)
        total_days = (max_date - min_date).days + 1
        holdout_days = int(total_days * args.holdout_pct)
        test_start = max_date - timedelta(days=holdout_days)

        logger.info(f"Total days: {total_days}")
        logger.info(f"Training: {min_date} to {test_start - timedelta(days=1)} ({total_days - holdout_days} days)")
        logger.info(f"Testing:  {test_start} to {max_date} ({holdout_days} days)")

        logger.info("\nBuilding training dataset...")
        df_train = build_dataset_parallel(
            city, min_date, test_start - timedelta(days=1),
            n_workers=args.workers,
            include_station_city=include_station_city,
        )

        logger.info("\nBuilding test dataset...")
        df_test = build_dataset_parallel(
            city, test_start, max_date,
            n_workers=min(args.workers, 12),
            include_station_city=include_station_city,
        )

        # Cache datasets to cache_dir
        train_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_train.to_parquet(train_parquet, index=False)
        df_test.to_parquet(test_parquet, index=False)
        logger.info(f"Cached datasets to {cache_dir / city}")

    logger.info(f"\nTraining samples: {len(df_train):,}")
    logger.info(f"Training days: {df_train['day'].nunique()}")
    logger.info(f"Test samples: {len(df_test):,}")
    logger.info(f"Test days: {df_test['day'].nunique()}")

    # Verify station-city features (includes city_warmer_flag)
    sc_cols = [c for c in df_train.columns if "station_city" in c or c == "city_warmer_flag"]
    logger.info(f"\nStation-city features: {sc_cols}")
    for col in sc_cols:
        non_null = df_train[col].notna().sum()
        pct = 100.0 * non_null / len(df_train)
        logger.info(f"  {col}: {non_null:,}/{len(df_train):,} non-null ({pct:.1f}%)")

    # Verify multi-horizon features
    multi_cols = [c for c in df_train.columns if "fcst_multi" in c]
    logger.info(f"\nMulti-horizon features: {multi_cols}")
    for col in multi_cols:
        non_null = df_train[col].notna().sum()
        pct = 100.0 * non_null / len(df_train)
        logger.info(f"  {col}: {non_null:,}/{len(df_train):,} non-null ({pct:.1f}%)")

    # Train with Optuna
    from models.training.ordinal_trainer import OrdinalDeltaTrainer
    from models.evaluation.metrics import compute_delta_metrics

    print("\n" + "=" * 60)
    print(f"OPTUNA TRAINING ({args.trials} trials, objective={args.objective})")
    print("=" * 60)

    trainer = OrdinalDeltaTrainer(
        base_model="catboost",
        n_trials=args.trials,
        cv_splits=args.cv_splits,
        objective=args.objective,
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
    model_path = output_dir / "ordinal_catboost_optuna.pkl"
    trainer.save(model_path)
    logger.info(f"\nSaved model to {model_path}")

    # Save best params separately
    params_path = output_dir / "best_params.json"
    with open(params_path, 'w') as f:
        json.dump(trainer.best_params, f, indent=2)
    logger.info(f"Saved best params to {params_path}")

    # Save final metrics
    final_metrics = {
        "city": city,
        "trained_at": datetime.now().isoformat(),
        "n_train_samples": len(df_train),
        "n_train_days": int(df_train['day'].nunique()),
        "n_test_samples": len(df_test),
        "n_test_days": int(df_test['day'].nunique()),
        "date_range": {
            "min_date": str(min_date),
            "max_date": str(max_date),
            "train_start": str(min_date),
            "train_end": str(test_start - timedelta(days=1)),
            "test_start": str(test_start),
            "test_end": str(max_date),
        },
        "delta_range": trainer._metadata.get("delta_range", None),
        "optuna_trials": args.trials,
        "cv_splits": args.cv_splits,
        "include_station_city": include_station_city,
        "metrics": {
            "delta_accuracy": metrics.get("delta_accuracy", None),
            "delta_mae": metrics.get("delta_mae", None),
            "within_1_rate": metrics.get("within_1_rate", None),
            "within_2_rate": metrics.get("within_2_rate", None),
        },
        "best_params": trainer.best_params,
    }

    metrics_path = output_dir / f"final_metrics_{city}.json"
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2, default=str)
    logger.info(f"Saved final metrics to {metrics_path}")

    # Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    fi = trainer.get_feature_importance()
    if fi is not None:
        print("\nTop 30 Features:")
        print(fi.head(30).to_string())

        # Station-city features
        sc_fi = fi[fi["feature"].str.contains("station_city", na=False)]
        if not sc_fi.empty:
            print("\n" + "-" * 40)
            print("Station-city feature importance:")
            print(sc_fi.to_string())
            for _, row in sc_fi.iterrows():
                rank = fi[fi["feature"] == row["feature"]].index[0] + 1
                print(f"  {row['feature']}: rank {rank}/{len(fi)}")

        # Multi-horizon features
        multi_fi = fi[fi["feature"].str.contains("fcst_multi", na=False)]
        if not multi_fi.empty:
            print("\n" + "-" * 40)
            print("Multi-horizon feature importance:")
            print(multi_fi.to_string())
            for _, row in multi_fi.iterrows():
                rank = fi[fi["feature"] == row["feature"]].index[0] + 1
                print(f"  {row['feature']}: rank {rank}/{len(fi)}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"City: {city}")
    print(f"Model: {model_path}")
    print(f"Params: {params_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Training samples: {len(df_train):,}")
    print(f"Test samples: {len(df_test):,}")
    print(f"Optuna trials: {args.trials}")
    print(f"Best params: {trainer.best_params}")
    print()
    print("Key Metrics:")
    print(f"  Accuracy: {metrics.get('delta_accuracy', 0):.1%}")
    print(f"  MAE: {metrics.get('delta_mae', 0):.2f}")
    print(f"  Within 1: {metrics.get('within_1_rate', 0):.1%}")
    print(f"  Within 2: {metrics.get('within_2_rate', 0):.1%}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
