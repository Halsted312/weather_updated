#!/usr/bin/env python3
"""
Parallel dataset builder for market-clock models.

Splits date range into chunks and processes them in parallel using multiprocessing.
Each worker gets its own DB connection to avoid contention.

Usage:
    .venv/bin/python scripts/build_dataset_parallel.py \
        --city chicago \
        --start 2025-01-01 \
        --end 2025-11-27 \
        --output data/chicago/datasets/market_clock_v2.parquet \
        --workers 20
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    include_forecast: bool = True,
    include_market: bool = False,
) -> ChunkResult:
    """Process a chunk of dates. Each worker creates its own DB connection."""
    try:
        # Import inside worker to avoid pickling issues
        from src.db.connection import get_db_session
        from models.data.dataset import DatasetConfig, build_dataset

        config = DatasetConfig(
            time_window="market_clock",
            snapshot_interval_min=5,
            include_forecast=include_forecast,
            include_market=include_market,
            include_station_city=False,
            include_meteo=True,
        )

        with get_db_session() as session:
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
        return ChunkResult(
            chunk_id=chunk_id,
            start_date=start_date,
            end_date=end_date,
            n_rows=0,
            n_days=0,
            error=str(e),
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


def main():
    parser = argparse.ArgumentParser(description='Build dataset in parallel')
    parser.add_argument('--city', type=str, required=True, help='City to process')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, required=True, help='Output parquet path')
    parser.add_argument('--workers', type=int, default=20, help='Number of parallel workers')
    parser.add_argument('--include-market', action='store_true', help='Include market candle data')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end, '%Y-%m-%d').date()

    total_days = (end_date - start_date).days + 1
    logger.info(f"Building dataset for {args.city}: {start_date} to {end_date} ({total_days} days)")
    logger.info(f"Using {args.workers} parallel workers")

    # Split into chunks - use more chunks than workers for better load balancing
    n_chunks = min(args.workers * 2, total_days)  # At most 2 chunks per worker, or 1 per day
    chunks = split_date_range(start_date, end_date, n_chunks)
    logger.info(f"Split into {len(chunks)} chunks")

    # Process chunks in parallel
    results = []
    completed = 0

    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all chunks
        futures = {
            executor.submit(
                process_chunk,
                i,
                args.city,
                chunk_start,
                chunk_end,
                include_forecast=True,
                include_market=args.include_market,
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
                logger.error(f"Chunk {chunk_id} failed: {result.error}")
            else:
                logger.info(
                    f"Chunk {completed}/{len(chunks)} done: "
                    f"{result.start_date} to {result.end_date}, "
                    f"{result.n_rows:,} rows, {result.n_days} days"
                )

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"All chunks completed in {elapsed:.1f}s")

    # Combine results
    dfs = [r.df for r in results if r.df is not None and len(r.df) > 0]

    if not dfs:
        logger.error("No data collected!")
        return 1

    df_combined = pd.concat(dfs, ignore_index=True)

    # Sort by event_date and cutoff_time
    if 'event_date' in df_combined.columns and 'cutoff_time' in df_combined.columns:
        df_combined = df_combined.sort_values(['event_date', 'cutoff_time']).reset_index(drop=True)

    logger.info(f"Combined dataset: {len(df_combined):,} rows, {len(df_combined.columns)} columns")

    # Check for new time-confidence features
    new_features = ['obs_confidence', 'expected_delta_uncertainty',
                    'confidence_weighted_gap', 'fcst_importance_weight', 'remaining_upside']
    logger.info("New time-confidence features:")
    for feat in new_features:
        if feat in df_combined.columns:
            non_null = df_combined[feat].notna().sum()
            logger.info(f"  {feat}: present, {non_null:,} non-null values")
        else:
            logger.warning(f"  {feat}: MISSING!")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_parquet(output_path, index=False)
    logger.info(f"Saved to: {output_path}")

    # Summary
    if 'day' in df_combined.columns:
        n_days = df_combined['day'].nunique()
    elif 'event_date' in df_combined.columns:
        n_days = df_combined['event_date'].nunique()
    else:
        n_days = 0

    logger.info(f"\n{'='*60}")
    logger.info(f"DATASET SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"City: {args.city}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Total rows: {len(df_combined):,}")
    logger.info(f"Total days: {n_days}")
    logger.info(f"Columns: {len(df_combined.columns)}")
    logger.info(f"Build time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Output: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
