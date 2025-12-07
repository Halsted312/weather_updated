#!/usr/bin/env python3
"""
Patch cloudcover into existing parquet files via linear interpolation.

This script loads existing full.parquet files, interpolates hourly cloudcover
from wx.vc_forecast_hourly, and merges it back into the dataset.

Much faster than full rebuild (~2 min per city vs ~10 min).

Usage:
    python scripts/patch_cloudcover_all_cities.py
"""

import logging
import sys
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import get_db_session
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CITIES = ["austin", "chicago", "denver", "los_angeles", "miami", "philadelphia"]


def load_hourly_cloudcover(city: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Load hourly forecast cloudcover from database."""
    from models.data.loader import get_vc_location_id

    with get_db_session() as session:
        # Get vc_location_id for this city
        vc_location_id = get_vc_location_id(session, city, "city")

        if vc_location_id is None:
            logger.warning(f"No vc_location found for {city}")
            return pd.DataFrame()

        # Use DATE() to extract date from datetime column (no target_date column exists)
        query = text("""
            SELECT
                target_datetime_local,
                cloudcover
            FROM wx.vc_forecast_hourly
            WHERE vc_location_id = :vc_location_id
              AND DATE(target_datetime_local) BETWEEN :start_date AND :end_date
              AND cloudcover IS NOT NULL
              AND data_type = 'historical_forecast'
            ORDER BY target_datetime_local
        """)

        df = pd.read_sql(query, session.bind, params={
            "vc_location_id": vc_location_id,
            "start_date": start_date,
            "end_date": end_date
        })

    return df


def interpolate_cloudcover_linear(df: pd.DataFrame, hourly_cc_df: pd.DataFrame) -> pd.DataFrame:
    """Add interpolated cloudcover to dataset using linear interpolation.

    Args:
        df: Dataset with event_date and snapshot_hour columns
        hourly_cc_df: Hourly cloudcover with target_datetime_local and cloudcover

    Returns:
        df with cloudcover columns updated
    """
    if hourly_cc_df.empty:
        logger.warning("No hourly cloudcover data available")
        return df

    df = df.copy()

    # Create datetime from event_date and snapshot_hour
    df['snapshot_datetime'] = pd.to_datetime(df['event_date']) + pd.to_timedelta(df['snapshot_hour'], unit='h')

    # Ensure hourly cloudcover datetimes are datetime type
    hourly_cc_df = hourly_cc_df.copy()
    hourly_cc_df['target_datetime_local'] = pd.to_datetime(hourly_cc_df['target_datetime_local'])

    # Remove timezone if present
    if df['snapshot_datetime'].dt.tz is not None:
        df['snapshot_datetime'] = df['snapshot_datetime'].dt.tz_localize(None)
    if hourly_cc_df['target_datetime_local'].dt.tz is not None:
        hourly_cc_df['target_datetime_local'] = hourly_cc_df['target_datetime_local'].dt.tz_localize(None)

    # CRITICAL FIX: Handle duplicates by averaging (multiple forecasts for same hour)
    hourly_cc_df = hourly_cc_df.groupby('target_datetime_local', as_index=False)['cloudcover'].mean()

    # Create series for interpolation (now guaranteed no duplicates)
    hourly_series = pd.Series(
        hourly_cc_df['cloudcover'].values,
        index=hourly_cc_df['target_datetime_local']
    )

    # Combine all timestamps (dataset + hourly)
    all_times = pd.concat([
        pd.Series(df['snapshot_datetime']),
        pd.Series(hourly_series.index)
    ]).drop_duplicates().sort_values()

    # Reindex and interpolate
    interpolated = hourly_series.reindex(all_times).interpolate(method='linear', limit_direction='both')

    # Map back to dataset
    cloudcover_interpolated = df['snapshot_datetime'].map(interpolated)

    # Update cloudcover-related columns
    df['cloudcover_last_obs'] = cloudcover_interpolated
    df['cloudcover_mean_last_60min'] = cloudcover_interpolated  # Simplified - could compute rolling mean

    # Recompute derived features
    df['clear_sky_flag'] = (cloudcover_interpolated < 20).astype(float)
    df['high_cloud_flag'] = (cloudcover_interpolated > 70).astype(float)

    # Cloud regime: 0=clear, 1=partly, 2=overcast
    df['cloud_regime'] = pd.cut(
        cloudcover_interpolated,
        bins=[-1, 20, 70, 101],
        labels=[0, 1, 2],
        include_lowest=True
    ).astype(float)

    # Rate and volatility - set to None for now (need time-series window calculations)
    df['cloudcover_rate_last_30min'] = None
    df['cloudcover_volatility_60min'] = None
    df['clearing_trend_flag'] = 0.0
    df['clouding_trend_flag'] = 0.0
    df['cloud_stability_score'] = None

    # Interaction: cloudcover × hour
    if 'hour' in df.columns:
        df['cloudcover_x_hour'] = cloudcover_interpolated * df['hour']

    # Drop temp column
    df = df.drop(columns=['snapshot_datetime'])

    non_null = cloudcover_interpolated.notna().sum()
    logger.info(f"  Cloudcover interpolated: {non_null:,}/{len(df):,} ({100*non_null/len(df):.1f}%)")

    return df


def patch_city_cloudcover(city: str, base_dir: Path):
    """Patch cloudcover for a single city."""
    logger.info(f"\n{'='*80}")
    logger.info(f"PATCHING {city.upper()}")
    logger.info("="*80)

    parquet_file = base_dir / city / "full.parquet"

    if not parquet_file.exists():
        logger.error(f"File not found: {parquet_file}")
        return

    # Load existing dataset
    logger.info(f"Loading {parquet_file}")
    df = pd.read_parquet(parquet_file)
    logger.info(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")

    # Get date range
    min_date = df['event_date'].min()
    max_date = df['event_date'].max()
    logger.info(f"  Date range: {min_date} to {max_date}")

    # Load hourly cloudcover from database
    logger.info("Loading hourly cloudcover from database...")
    hourly_cc_df = load_hourly_cloudcover(city, min_date, max_date)
    logger.info(f"  Loaded {len(hourly_cc_df):,} hourly cloudcover records")

    if hourly_cc_df.empty:
        logger.warning(f"No cloudcover data available for {city}")
        return

    # Interpolate and merge
    logger.info("Interpolating cloudcover...")
    start_time = datetime.now()

    df = interpolate_cloudcover_linear(df, hourly_cc_df)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"  Interpolation completed in {elapsed:.1f}s")

    # Save back to parquet
    logger.info(f"Saving to {parquet_file}")
    df.to_parquet(parquet_file, index=False)

    logger.info(f"✓ {city} cloudcover patched successfully")

    # Verify
    cc_cols = ['cloudcover_last_obs', 'clear_sky_flag', 'high_cloud_flag', 'cloud_regime', 'cloudcover_x_hour']
    logger.info("Verification:")
    for col in cc_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = 100 * non_null / len(df)
            logger.info(f"  {col:30s}: {non_null:>8,}/{len(df):,} ({pct:>5.1f}%)")


def main():
    logger.info("="*80)
    logger.info("PATCHING CLOUDCOVER FOR ALL CITIES")
    logger.info("="*80)

    base_dir = Path("models/saved")

    for i, city in enumerate(CITIES, 1):
        logger.info(f"\n[{i}/6] Processing {city}...")
        try:
            patch_city_cloudcover(city, base_dir)
        except Exception as e:
            logger.error(f"Failed to patch {city}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("\n" + "="*80)
    logger.info("CLOUDCOVER PATCHING COMPLETE")
    logger.info("="*80)
    logger.info("\n✓ All cities ready for ordinal training!")
    logger.info("\nNext step: Train ordinal models")
    logger.info("  python scripts/train_city_ordinal_optuna.py --city austin --trials 100 --use-cached")


if __name__ == "__main__":
    main()
