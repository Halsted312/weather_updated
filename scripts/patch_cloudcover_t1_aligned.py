#!/usr/bin/env python3
"""
Patch cloudcover into parquet files using T-1 forecast (aligned with system).

Uses forecast_basis_date = event_date - 1 (same as temperature forecasts).
Linear interpolation from hourly to 5-min increments.

Usage:
    python scripts/patch_cloudcover_t1_aligned.py
"""

import logging
import sys
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import get_db_session
from models.data.loader import get_vc_location_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CITIES = ["austin", "chicago", "denver", "los_angeles", "miami", "philadelphia"]


def load_t1_hourly_cloudcover(
    session,
    vc_location_id: int,
    event_date: date
) -> pd.DataFrame:
    """Load T-1 hourly cloudcover for a specific event_date.

    Args:
        session: DB session
        vc_location_id: Location ID
        event_date: The event/settlement day

    Returns:
        DataFrame with target_datetime_local and cloudcover (~24 hourly values)
    """
    basis_date = event_date - timedelta(days=1)  # T-1 forecast

    query = text("""
        SELECT
            target_datetime_local,
            cloudcover
        FROM wx.vc_forecast_hourly
        WHERE vc_location_id = :vc_location_id
          AND forecast_basis_date = :basis_date
          AND DATE(target_datetime_local) = :event_date
          AND cloudcover IS NOT NULL
          AND data_type = 'historical_forecast'
        ORDER BY target_datetime_local
    """)

    result = session.execute(query, {
        "vc_location_id": vc_location_id,
        "basis_date": basis_date,
        "event_date": event_date,
    })

    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=['target_datetime_local', 'cloudcover'])


def patch_city_cloudcover(city: str, base_dir: Path):
    """Patch cloudcover for a city using T-1 forecast + linear interpolation."""
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

    unique_dates = sorted(df['event_date'].unique())
    logger.info(f"  Event dates: {len(unique_dates)} ({unique_dates[0]} to {unique_dates[-1]})")

    # Get vc_location_id
    with get_db_session() as session:
        vc_location_id = get_vc_location_id(session, city, "city")

        if vc_location_id is None:
            logger.error(f"No vc_location for {city}")
            return

        logger.info(f"  vc_location_id: {vc_location_id}")

        # Process each event_date
        total_populated = 0
        days_with_data = 0

        for i, event_date in enumerate(unique_dates):
            if i % 100 == 0:
                logger.info(f"  Processing day {i+1}/{len(unique_dates)}: {event_date}")

            # Load T-1 hourly cloudcover for this event
            hourly_cc = load_t1_hourly_cloudcover(session, vc_location_id, event_date)

            if hourly_cc.empty:
                continue  # No T-1 forecast available, leave as None

            days_with_data += 1

            # Ensure no timezone
            hourly_cc['target_datetime_local'] = pd.to_datetime(hourly_cc['target_datetime_local'])
            if hourly_cc['target_datetime_local'].dt.tz is not None:
                hourly_cc['target_datetime_local'] = hourly_cc['target_datetime_local'].dt.tz_localize(None)

            # Create series indexed by datetime
            hourly_series = hourly_cc.set_index('target_datetime_local')['cloudcover'].sort_index()

            # CRITICAL: Handle DST duplicates (fall-back hour occurs twice)
            # Average cloudcover for duplicate timestamps
            if hourly_series.index.duplicated().any():
                hourly_series = hourly_series.groupby(level=0).mean()

            # Get snapshots for this event_date
            day_mask = df['event_date'] == event_date
            day_indices = df[day_mask].index
            day_snapshot_hours = df.loc[day_mask, 'snapshot_hour'].values

            # Create snapshot datetimes
            snapshot_dts = pd.to_datetime(event_date) + pd.to_timedelta(day_snapshot_hours, unit='h')

            # Combine times for interpolation
            all_times = pd.concat([
                pd.Series(snapshot_dts),
                pd.Series(hourly_series.index)
            ]).drop_duplicates().sort_values()

            # Interpolate
            interpolated = hourly_series.reindex(all_times).interpolate(method='linear', limit_direction='both')

            # Map back to dataframe
            for idx, snapshot_dt in zip(day_indices, snapshot_dts):
                if snapshot_dt in interpolated.index:
                    cc_value = interpolated.loc[snapshot_dt]

                    if pd.notna(cc_value):
                        df.at[idx, 'cloudcover_last_obs'] = float(cc_value)
                        df.at[idx, 'cloudcover_mean_last_60min'] = float(cc_value)  # Simplified
                        df.at[idx, 'clear_sky_flag'] = 1.0 if cc_value < 20 else 0.0
                        df.at[idx, 'high_cloud_flag'] = 1.0 if cc_value > 70 else 0.0

                        # Cloud regime
                        if cc_value < 20:
                            df.at[idx, 'cloud_regime'] = 0.0
                        elif cc_value < 70:
                            df.at[idx, 'cloud_regime'] = 1.0
                        else:
                            df.at[idx, 'cloud_regime'] = 2.0

                        # Interaction
                        if 'hour' in df.columns:
                            df.at[idx, 'cloudcover_x_hour'] = cc_value * df.at[idx, 'hour']

                        total_populated += 1

    logger.info(f"  Days with T-1 cloudcover: {days_with_data}/{len(unique_dates)}")
    logger.info(f"  Snapshots populated: {total_populated:,}/{len(df):,} ({100*total_populated/len(df):.1f}%)")

    # Save
    logger.info(f"Saving to {parquet_file}")
    df.to_parquet(parquet_file, index=False)
    logger.info(f"✓ {city} cloudcover patched successfully")


def main():
    logger.info("="*80)
    logger.info("PATCHING CLOUDCOVER (T-1 ALIGNED)")
    logger.info("="*80)

    base_dir = Path("data/training_cache")

    for i, city in enumerate(CITIES, 1):
        logger.info(f"\n[{i}/6] Processing {city}...")
        try:
            start_time = datetime.now()
            patch_city_cloudcover(city, base_dir)
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"  Completed in {elapsed:.1f}s")
        except Exception as e:
            logger.error(f"Failed to patch {city}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("\n" + "="*80)
    logger.info("CLOUDCOVER PATCHING COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
