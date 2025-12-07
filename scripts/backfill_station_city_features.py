#!/usr/bin/env python3
"""
Backfill station-city features onto an existing training parquet - FAST VERSION.

Computes features once per unique (event_date, cutoff_time), then merges.
~1000x faster than row-by-row iteration.

Usage:
    # With defaults (reads/writes models/saved/{city}/full.parquet)
    PYTHONPATH=. python scripts/backfill_station_city_features.py --city chicago

    # Explicit paths
    PYTHONPATH=. python scripts/backfill_station_city_features.py \
        --city chicago \
        --input models/saved/chicago/full.parquet \
        --output models/saved/chicago/full.parquet
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def compute_features_for_cutoff(
    cutoff_time: datetime,
    station_obs_df: pd.DataFrame,
    city_obs_df: pd.DataFrame,
) -> dict:
    """Compute station-city features for a single cutoff time."""
    null_features = {
        "station_city_temp_gap": None,
        "station_city_max_gap_sofar": None,
        "station_city_mean_gap_sofar": None,
        "city_warmer_flag": None,
        "station_city_gap_std": None,
        "station_city_gap_trend": None,
    }

    if station_obs_df.empty or city_obs_df.empty:
        return null_features

    # Filter to cutoff (vectorized)
    station_filtered = station_obs_df[station_obs_df['datetime_local'] <= cutoff_time]
    city_filtered = city_obs_df[city_obs_df['datetime_local'] <= cutoff_time]

    if station_filtered.empty or city_filtered.empty:
        return null_features

    # Merge on datetime_local for aligned comparison
    merged = pd.merge(
        station_filtered[['datetime_local', 'temp_f']],
        city_filtered[['datetime_local', 'temp_f']],
        on='datetime_local',
        suffixes=('_station', '_city'),
        how='inner'
    )

    if merged.empty:
        return null_features

    # Compute gaps
    merged['gap'] = merged['temp_f_station'] - merged['temp_f_city']
    gaps = merged['gap'].values

    features = {}

    # Current gap (most recent)
    features["station_city_temp_gap"] = float(gaps[-1])

    # Max gap
    features["station_city_max_gap_sofar"] = float(
        merged['temp_f_station'].max() - merged['temp_f_city'].max()
    )

    # Mean gap
    features["station_city_mean_gap_sofar"] = float(np.mean(gaps))

    # Std of gap
    features["station_city_gap_std"] = float(np.std(gaps)) if len(gaps) > 1 else 0.0

    # City warmer flag
    features["city_warmer_flag"] = 1 if gaps[-1] < 0 else 0

    # Gap trend
    if len(gaps) >= 4:
        mid = len(gaps) // 2
        features["station_city_gap_trend"] = float(np.mean(gaps[mid:]) - np.mean(gaps[:mid]))
    else:
        features["station_city_gap_trend"] = None

    return features


def main():
    parser = argparse.ArgumentParser(description="Backfill station-city features (FAST)")
    parser.add_argument("--city", required=True, help="City code (e.g., chicago)")
    parser.add_argument("--input", help="Input parquet path (default: models/saved/{city}/full.parquet)")
    parser.add_argument("--output", help="Output parquet path (default: same as input)")
    parser.add_argument("--raw-data-dir", default="models/raw_data", help="Raw data directory")
    args = parser.parse_args()

    city = args.city

    # Default paths based on city
    default_path = Path(f"models/saved/{city}/full.parquet")
    input_path = Path(args.input) if args.input else default_path
    output_path = Path(args.output) if args.output else input_path
    raw_data_dir = Path(args.raw_data_dir) / city

    logger.info("=" * 60)
    logger.info(f"BACKFILL STATION-CITY FEATURES (FAST): {city.upper()}")
    logger.info("=" * 60)

    # Load existing parquet
    logger.info(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    logger.info(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")

    # Ensure datetime columns
    df['cutoff_time'] = pd.to_datetime(df['cutoff_time'])
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date']).dt.date

    # Load raw observation data
    logger.info(f"\nLoading raw data from {raw_data_dir}...")

    station_obs = pd.read_parquet(raw_data_dir / "vc_observations.parquet")
    station_obs['datetime_local'] = pd.to_datetime(station_obs['datetime_local'])
    station_obs['obs_date'] = station_obs['datetime_local'].dt.date
    logger.info(f"  Station observations: {len(station_obs):,} rows")

    city_obs = pd.read_parquet(raw_data_dir / "vc_city_observations.parquet")
    city_obs['datetime_local'] = pd.to_datetime(city_obs['datetime_local'])
    city_obs['obs_date'] = city_obs['datetime_local'].dt.date
    logger.info(f"  City observations: {len(city_obs):,} rows")

    # Pre-group by date
    logger.info("  Pre-grouping by date...")
    station_by_date = {d: g for d, g in station_obs.groupby('obs_date')}
    city_by_date = {d: g for d, g in city_obs.groupby('obs_date')}

    # Get unique (event_date, cutoff_time) pairs - this is the key optimization
    unique_snapshots = df[['event_date', 'cutoff_time']].drop_duplicates()
    logger.info(f"\nUnique snapshots to compute: {len(unique_snapshots):,}")

    # Compute features for each unique snapshot
    logger.info("Computing features for unique snapshots...")
    feature_records = []

    for _, row in tqdm(unique_snapshots.iterrows(), total=len(unique_snapshots), desc="Snapshots"):
        event_date = row['event_date']
        cutoff_time = row['cutoff_time']

        # Make cutoff naive if needed
        if hasattr(cutoff_time, 'tzinfo') and cutoff_time.tzinfo is not None:
            cutoff_time_naive = cutoff_time.replace(tzinfo=None)
        else:
            cutoff_time_naive = cutoff_time

        # Get obs for D and D-1
        d_minus_1 = event_date - timedelta(days=1)
        station_today = station_by_date.get(event_date, pd.DataFrame())
        station_yesterday = station_by_date.get(d_minus_1, pd.DataFrame())
        day_station = pd.concat([station_today, station_yesterday], ignore_index=True) if len(station_today) or len(station_yesterday) else pd.DataFrame()

        city_today = city_by_date.get(event_date, pd.DataFrame())
        city_yesterday = city_by_date.get(d_minus_1, pd.DataFrame())
        day_city = pd.concat([city_today, city_yesterday], ignore_index=True) if len(city_today) or len(city_yesterday) else pd.DataFrame()

        # Compute features
        features = compute_features_for_cutoff(cutoff_time_naive, day_station, day_city)
        features['event_date'] = event_date
        features['cutoff_time'] = cutoff_time
        feature_records.append(features)

    # Create features dataframe
    logger.info("\nMerging features onto main dataframe...")
    features_df = pd.DataFrame(feature_records)

    # Drop old columns if they exist (including city_warmer_flag)
    cols_to_drop = [c for c in df.columns if 'station_city' in c or c == 'city_warmer_flag']
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"  Dropped old columns: {cols_to_drop}")

    # Merge on (event_date, cutoff_time)
    df = df.merge(features_df, on=['event_date', 'cutoff_time'], how='left')

    # Add interaction feature if fcst_gap exists
    if 'fcst_gap' in df.columns:
        df['station_city_gap_x_fcst_gap'] = df['station_city_mean_gap_sofar'] * df['fcst_gap']
    else:
        df['station_city_gap_x_fcst_gap'] = None

    # Check results
    logger.info("\nResults:")
    sc_cols = [c for c in df.columns if 'station_city' in c or c == 'city_warmer_flag']
    for col in sc_cols:
        non_null = df[col].notna().sum()
        logger.info(f"  {col}: {non_null}/{len(df)} non-null ({100*non_null/len(df):.1f}%)")

    # Save
    logger.info(f"\nSaving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"  Done! {len(df):,} rows, {len(df.columns)} columns")

    return 0


if __name__ == "__main__":
    sys.exit(main())
