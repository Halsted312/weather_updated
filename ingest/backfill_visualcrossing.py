#!/usr/bin/env python3
"""
Visual Crossing 5-minute backfill script.

Fetches minute-level observations from Visual Crossing and loads to wx.minute_obs.
Optionally applies forward-fill to create complete 5-minute UTC grid (288 slots per day).

Usage:
    # Test backfill (1 week, Chicago only)
    python ingest/backfill_visualcrossing.py \
        --start-date 2024-01-10 \
        --end-date 2024-01-16 \
        --cities chicago \
        --ffill \
        --replace

    # Full backfill (all cities, full date range)
    python ingest/backfill_visualcrossing.py \
        --start-date 2024-01-01 \
        --end-date 2025-11-14 \
        --cities all \
        --ffill \
        --replace
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
import os
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from weather.visual_crossing import VisualCrossingClient, STATION_MAP
from db.connection import get_session
from db.loaders import bulk_upsert_wx_minutes

logger = logging.getLogger(__name__)


def forward_fill_5min_grid(df: pd.DataFrame, date_local: date) -> pd.DataFrame:
    """
    Forward-fill VC data to create complete 5-minute UTC grid for a day.

    Creates 288 rows (24h * 60min / 5min) per day in UTC.
    Marks forward-filled rows with ffilled=TRUE.

    Args:
        df: DataFrame with minute observations for a date
        date_local: The local date (for logging)

    Returns:
        DataFrame with 288 rows (5-min grid for 24h UTC) with ffilled column
    """
    if df.empty:
        logger.warning(f"No data to forward-fill for {date_local}")
        return df

    # Get UTC date range
    min_ts = df['ts_utc'].min()

    # Build grid from UTC midnight to next midnight
    start_utc = min_ts.floor('D')  # Floor to midnight UTC
    end_utc = start_utc + timedelta(days=1) - timedelta(minutes=5)  # Last 5-min slot

    # Create 5-minute grid (288 slots)
    utc_grid = pd.date_range(start=start_utc, end=end_utc, freq='5min', tz='UTC')
    grid_df = pd.DataFrame({'ts_utc': utc_grid})

    # Merge with actual observations
    merged = grid_df.merge(df, on='ts_utc', how='left')

    # Mark ffilled rows (where temp_f is missing from original VC data)
    merged['ffilled'] = merged['temp_f'].isna()

    # Forward-fill numeric columns
    numeric_cols = ['temp_f', 'humidity', 'dew_f', 'windspeed_mph', 'windgust_mph', 'pressure_mb', 'precip_in']
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill()

    # Forward-fill string columns
    if 'preciptype' in merged.columns:
        merged['preciptype'] = merged['preciptype'].ffill()

    # Fill raw_json with None for ffilled rows
    if 'raw_json' in merged.columns:
        merged.loc[merged['ffilled'], 'raw_json'] = None

    ffilled_count = merged['ffilled'].sum()
    ffilled_pct = 100.0 * ffilled_count / len(merged) if len(merged) > 0 else 0

    logger.info(
        f"  {date_local}: {len(merged)} total rows "
        f"({ffilled_count} ffilled, {ffilled_pct:.1f}%)"
    )

    return merged


def backfill_city(
    client: VisualCrossingClient,
    city: str,
    loc_id: str,
    vc_key: str,
    start_date: date,
    end_date: date,
    ffill: bool = False,
    replace: bool = False,
) -> dict:
    """
    Backfill Visual Crossing data for a single city.

    Args:
        client: VisualCrossingClient instance
        city: City name (e.g., "chicago")
        loc_id: Location ID (e.g., "KMDW")
        vc_key: Visual Crossing location key (e.g., "stn:KMDW")
        start_date: Start date
        end_date: End date
        ffill: If True, forward-fill to create complete 5-min grid
        replace: If True, delete existing records before loading

    Returns:
        Dict with stats: {total_rows, real_rows, ffilled_rows, errors}
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Backfilling {city.upper()} ({loc_id})")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Forward-fill: {ffill}")
    logger.info(f"Replace: {replace}")
    logger.info(f"{'='*60}\n")

    stats = {"total_rows": 0, "real_rows": 0, "ffilled_rows": 0, "errors": 0}

    try:
        # Fetch data from Visual Crossing
        df = client.fetch_range_for_station(
            loc_id,
            vc_key,
            start_date.isoformat(),
            end_date.isoformat()
        )

        if df.empty:
            logger.warning(f"No data fetched for {city}")
            return stats

        stats["real_rows"] = len(df)

        # Apply forward-fill if requested
        if ffill:
            # Process each date separately
            all_dfs = []
            current_date = start_date

            logger.info("Forward-filling per-day grids:")

            while current_date <= end_date:
                # Filter to current UTC date
                # Note: df['ts_utc'] is already timezone-aware (UTC)
                date_df = df[df['ts_utc'].dt.date == current_date]

                if not date_df.empty:
                    # Forward-fill this day
                    filled_df = forward_fill_5min_grid(date_df, current_date)
                    all_dfs.append(filled_df)
                else:
                    logger.warning(f"  {current_date}: No data from VC")

                current_date += timedelta(days=1)

            # Concatenate all days
            if all_dfs:
                df = pd.concat(all_dfs, ignore_index=True)
                stats["ffilled_rows"] = int(df['ffilled'].sum())
                stats["total_rows"] = len(df)

                logger.info(
                    f"\nTotal after forward-fill: {stats['total_rows']} rows "
                    f"({stats['real_rows']} real, {stats['ffilled_rows']} ffilled)"
                )
            else:
                logger.warning(f"No data after forward-fill for {city}")
                return stats
        else:
            # No forward-fill: mark all as real observations
            df['ffilled'] = False
            stats["total_rows"] = len(df)

        # Delete existing records if replace=True
        if replace:
            with get_session() as session:
                from sqlalchemy import delete
                from db.models import WxMinuteObs

                logger.info(f"Deleting existing records for {loc_id} from {start_date} to {end_date}...")

                # Delete by date range
                stmt = delete(WxMinuteObs).where(
                    WxMinuteObs.loc_id == loc_id,
                    WxMinuteObs.ts_utc >= pd.Timestamp(start_date, tz='UTC'),
                    WxMinuteObs.ts_utc < pd.Timestamp(end_date + timedelta(days=1), tz='UTC'),
                )
                result = session.execute(stmt)
                session.commit()
                logger.info(f"Deleted {result.rowcount} existing records")

        # Load to database
        with get_session() as session:
            bulk_upsert_wx_minutes(session, loc_id, df)
            session.commit()
            logger.info(f"✓ Loaded {len(df)} records for {city}")

        return stats

    except Exception as e:
        logger.error(f"Error backfilling {city}: {e}", exc_info=True)
        stats["errors"] += 1
        return stats


def main():
    """Main backfill entry point."""
    parser = argparse.ArgumentParser(
        description="Backfill Visual Crossing 5-minute weather observations"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--cities",
        type=str,
        nargs="+",
        default=["all"],
        help="Cities to backfill (default: all). Use 'all' or specific cities (chicago, new_york, etc.)"
    )
    parser.add_argument(
        "--ffill",
        action="store_true",
        help="Forward-fill to create complete 5-minute UTC grid (288 slots per day)"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing records in date range before loading"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    # Validate date range
    if start_date > end_date:
        logger.error(f"Invalid date range: {start_date} > {end_date}")
        return 1

    # Get cities list
    if "all" in args.cities:
        cities = list(STATION_MAP.keys())
    else:
        cities = args.cities

    # Validate cities
    for city in cities:
        if city not in STATION_MAP:
            logger.error(f"Unknown city: {city}. Available: {list(STATION_MAP.keys())}")
            return 1

    # Load API key
    load_dotenv()
    api_key = os.getenv("VC_API_KEY")
    if not api_key:
        logger.error("VC_API_KEY not found in environment. Please set it in .env file.")
        return 1

    # Initialize client
    client = VisualCrossingClient(api_key=api_key, minute_interval=5)

    # Backfill each city
    overall_stats = {"total_rows": 0, "real_rows": 0, "ffilled_rows": 0, "errors": 0}

    for city in cities:
        loc_id, vc_key = STATION_MAP[city]

        stats = backfill_city(
            client,
            city,
            loc_id,
            vc_key,
            start_date,
            end_date,
            ffill=args.ffill,
            replace=args.replace,
        )

        # Aggregate stats
        for key in stats:
            overall_stats[key] += stats[key]

    # Print summary
    print("\n" + "="*60)
    print("VISUAL CROSSING BACKFILL SUMMARY")
    print("="*60)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Cities: {', '.join(cities)}")
    print(f"Forward-fill: {args.ffill}")
    print(f"Replace: {args.replace}")
    print()
    print(f"Total rows loaded: {overall_stats['total_rows']}")
    print(f"  Real observations: {overall_stats['real_rows']}")
    print(f"  Forward-filled: {overall_stats['ffilled_rows']}")
    if overall_stats['total_rows'] > 0:
        ffill_pct = 100.0 * overall_stats['ffilled_rows'] / overall_stats['total_rows']
        print(f"  Forward-fill %: {ffill_pct:.1f}%")
    print(f"  Errors: {overall_stats['errors']}")
    print("="*60 + "\n")

    return 0 if overall_stats['errors'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
