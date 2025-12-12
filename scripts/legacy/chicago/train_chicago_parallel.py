#!/usr/bin/env python3
"""Train Chicago ordinal CatBoost model using parallel dataset building.

Uses ProcessPoolExecutor with 24 workers to build dataset ~20x faster.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/train_chicago_parallel.py
    PYTHONPATH=. .venv/bin/python scripts/train_chicago_parallel.py --test-only  # Quick verification
"""

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
N_WORKERS = 24


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
) -> ChunkResult:
    """Process a chunk of dates. Each worker creates its own DB connection."""
    try:
        # Import inside worker to avoid pickling issues
        import src.db.connection as db_conn
        from models.data.dataset import DatasetConfig, build_dataset

        # CRITICAL: Reset global engine state so this worker creates fresh connection
        db_conn._engine = None
        db_conn._SessionLocal = None

        config = DatasetConfig(
            time_window="market_clock",
            snapshot_interval_min=5,
            include_forecast=True,
            include_multi_horizon=include_multi_horizon,
            include_market=False,
            include_station_city=False,
            include_meteo=True,
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
    n_workers: int = N_WORKERS,
) -> pd.DataFrame:
    """Build dataset using parallel processing."""
    total_days = (end_date - start_date).days + 1
    logger.info(f"Building dataset for {city}: {start_date} to {end_date} ({total_days} days)")
    logger.info(f"Using {n_workers} parallel workers")

    # Split into chunks - use more chunks than workers for better load balancing
    n_chunks = min(n_workers * 2, total_days)
    chunks = split_date_range(start_date, end_date, n_chunks)
    logger.info(f"Split into {len(chunks)} chunks")

    # Process chunks in parallel
    results = []
    completed = 0

    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all chunks
        futures = {
            executor.submit(
                process_chunk,
                i,
                city,
                chunk_start,
                chunk_end,
            ): i
            for i, (chunk_start, chunk_end) in enumerate(chunks)
        }

        # Collect results as they complete
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

    # Combine results
    dfs = [r.df for r in results if r.df is not None and len(r.df) > 0]

    if not dfs:
        raise RuntimeError("No data collected from any chunk!")

    df_combined = pd.concat(dfs, ignore_index=True)

    # Sort by event_date and cutoff_time
    if 'day' in df_combined.columns and 'cutoff_time' in df_combined.columns:
        df_combined = df_combined.sort_values(['day', 'cutoff_time']).reset_index(drop=True)

    logger.info(f"Combined dataset: {len(df_combined):,} rows, {df_combined['day'].nunique()} days")

    return df_combined


def verify_multi_horizon_features(df: pd.DataFrame) -> bool:
    """Check that multi-horizon features are present and populated."""
    print("\n" + "=" * 60)
    print("MULTI-HORIZON FEATURES CHECK")
    print("=" * 60)

    multi_cols = [c for c in df.columns if "fcst_multi" in c]
    logger.info(f"Multi-horizon features found: {multi_cols}")

    if not multi_cols:
        logger.error("NO MULTI-HORIZON FEATURES FOUND!")
        return False

    all_good = True
    for col in multi_cols:
        non_null = df[col].notna().sum()
        pct = 100.0 * non_null / len(df)
        logger.info(f"  {col}: {non_null:,}/{len(df):,} non-null ({pct:.1f}%)")
        if pct < 50:
            logger.warning(f"  WARNING: {col} has low coverage!")
            all_good = False

    # Show sample values
    if multi_cols:
        print("\nSample multi-horizon values (first 10 rows):")
        print(df[multi_cols].head(10).to_string())

    return all_good


def train_model(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    """Train CatBoost ordinal model (no Optuna)."""
    from models.training.ordinal_trainer import OrdinalDeltaTrainer
    from models.evaluation.metrics import compute_delta_metrics

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

    return trainer, metrics


def show_feature_importance(trainer, multi_cols: list[str]):
    """Display feature importance, highlighting multi-horizon features."""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    fi = trainer.get_feature_importance()
    if fi is None:
        logger.warning("Could not get feature importance")
        return

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


def main():
    parser = argparse.ArgumentParser(description='Train Chicago CatBoost with parallel dataset building')
    parser.add_argument('--test-only', action='store_true', help='Only test with 7 days of data')
    parser.add_argument('--workers', type=int, default=N_WORKERS, help='Number of parallel workers')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Chicago Parallel CatBoost Training")
    logger.info("=" * 60)

    # Get date range
    from src.db import get_db_session
    from models.data.loader import get_available_date_range

    with get_db_session() as session:
        min_date, max_date = get_available_date_range(session, CITY)

    logger.info(f"Available data: {min_date} to {max_date}")

    if args.test_only:
        # Quick test with just 7 days
        logger.info("\n*** TEST MODE: Using only 7 days of data ***\n")
        test_end = max_date
        test_start = test_end - timedelta(days=6)  # 7 days total
        train_start = test_start - timedelta(days=7)  # 7 days training

        logger.info(f"Test period: {test_start} to {test_end}")
        logger.info(f"Train period: {train_start} to {test_start - timedelta(days=1)}")

        # Build small datasets
        df_train = build_dataset_parallel(CITY, train_start, test_start - timedelta(days=1), n_workers=8)
        df_test = build_dataset_parallel(CITY, test_start, test_end, n_workers=4)

    else:
        # Full dataset
        end_date = max_date
        test_start = end_date - timedelta(days=HOLDOUT_DAYS)

        logger.info(f"Training: {START_DATE} to {test_start - timedelta(days=1)}")
        logger.info(f"Testing:  {test_start} to {end_date}")

        # Build datasets in parallel
        logger.info("\nBuilding training dataset...")
        df_train = build_dataset_parallel(CITY, START_DATE, test_start - timedelta(days=1), n_workers=args.workers)

        logger.info("\nBuilding test dataset...")
        df_test = build_dataset_parallel(CITY, test_start, end_date, n_workers=min(args.workers, 12))

    logger.info(f"\nTraining samples: {len(df_train):,}")
    logger.info(f"Training days: {df_train['day'].nunique()}")
    logger.info(f"Test samples: {len(df_test):,}")
    logger.info(f"Test days: {df_test['day'].nunique()}")

    # Verify multi-horizon features
    multi_cols = [c for c in df_train.columns if "fcst_multi" in c]
    features_ok = verify_multi_horizon_features(df_train)

    if not features_ok:
        logger.error("Multi-horizon features verification failed!")
        if not args.test_only:
            return 1

    # Train model
    trainer, metrics = train_model(df_train, df_test)

    # Show feature importance
    show_feature_importance(trainer, multi_cols)

    # Save model (only for full training)
    if not args.test_only:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = OUTPUT_DIR / "ordinal_catboost_simple.pkl"
        trainer.save(model_path)
        logger.info(f"\nSaved model to {model_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
