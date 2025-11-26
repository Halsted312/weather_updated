#!/usr/bin/env python3
"""
Ingest Visual Crossing minute-level weather observations.

Fetches 5-minute interval data for all 6 cities and upserts into wx.minute_obs.

Supports resume-on-crash via checkpoints in meta.ingestion_checkpoint.

Usage:
    python scripts/ingest_vc_minutes.py --days 60
    python scripts/ingest_vc_minutes.py --start-date 2024-01-01 --end-date 2024-03-31
    python scripts/ingest_vc_minutes.py --city chicago --days 7
    python scripts/ingest_vc_minutes.py --all-history  # Fetch all available history
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import CITIES, EXCLUDED_VC_CITIES, get_city, get_settings
from src.db import get_db_session, WxMinuteObs
from src.db.checkpoint import (
    get_or_create_checkpoint,
    update_checkpoint,
    complete_checkpoint,
)
from src.weather.visual_crossing import VisualCrossingClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def df_to_db_records(df: pd.DataFrame, loc_id: str) -> List[dict]:
    """
    Convert Visual Crossing DataFrame to database records.

    Args:
        df: DataFrame from VisualCrossingClient
        loc_id: Station ICAO code (e.g., 'KMDW')

    Returns:
        List of dicts ready for upsert
    """
    records = []

    for _, row in df.iterrows():
        record = {
            "loc_id": loc_id,
            "ts_utc": row["ts_utc"],
            # Core weather
            "temp_f": row.get("temp_f"),
            "feelslike_f": row.get("feelslike_f"),
            "humidity": row.get("humidity"),
            "dew_f": row.get("dew_f"),
            # Precipitation
            "precip_in": row.get("precip_in"),
            "precip_prob": row.get("precip_prob"),
            "snow_in": row.get("snow_in"),
            "snow_depth_in": row.get("snow_depth_in"),
            # Wind
            "windspeed_mph": row.get("windspeed_mph"),
            "winddir": row.get("winddir_deg"),
            "windgust_mph": row.get("windgust_mph"),
            # Atmospheric
            "pressure_mb": row.get("pressure_mb"),
            "visibility_mi": row.get("visibility_mi"),
            "cloud_cover": row.get("cloud_cover"),
            # Solar
            "solar_radiation": row.get("solar_radiation"),
            "solar_energy": row.get("solar_energy"),
            "uv_index": row.get("uv_index"),
            # Conditions
            "conditions": row.get("conditions"),
            "icon": row.get("icon"),
            # Metadata
            "source": "visualcrossing",
            "stations": row.get("stations"),
            "ffilled": False,  # Real observations
            "raw_json": row.get("raw_json"),
        }
        records.append(record)

    return records


def upsert_minute_obs(session, records: List[dict]) -> int:
    """
    Upsert minute observations into wx.minute_obs.

    Returns:
        Number of rows affected
    """
    if not records:
        return 0

    stmt = insert(WxMinuteObs).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["loc_id", "ts_utc"],
        set_={
            "temp_f": stmt.excluded.temp_f,
            "feelslike_f": stmt.excluded.feelslike_f,
            "humidity": stmt.excluded.humidity,
            "dew_f": stmt.excluded.dew_f,
            "precip_in": stmt.excluded.precip_in,
            "precip_prob": stmt.excluded.precip_prob,
            "snow_in": stmt.excluded.snow_in,
            "snow_depth_in": stmt.excluded.snow_depth_in,
            "windspeed_mph": stmt.excluded.windspeed_mph,
            "winddir": stmt.excluded.winddir,
            "windgust_mph": stmt.excluded.windgust_mph,
            "pressure_mb": stmt.excluded.pressure_mb,
            "visibility_mi": stmt.excluded.visibility_mi,
            "cloud_cover": stmt.excluded.cloud_cover,
            "solar_radiation": stmt.excluded.solar_radiation,
            "solar_energy": stmt.excluded.solar_energy,
            "uv_index": stmt.excluded.uv_index,
            "conditions": stmt.excluded.conditions,
            "icon": stmt.excluded.icon,
            "source": stmt.excluded.source,
            "stations": stmt.excluded.stations,
            "ffilled": stmt.excluded.ffilled,
            "raw_json": stmt.excluded.raw_json,
        },
    )

    result = session.execute(stmt)
    return result.rowcount


def ingest_city(
    client: VisualCrossingClient,
    session,
    city_id: str,
    start_date: str,
    end_date: str,
) -> int:
    """
    Fetch and upsert minute observations for a single city.

    Returns:
        Number of records upserted
    """
    city = get_city(city_id)

    logger.info(f"Ingesting {city_id} ({city.icao}) from {start_date} to {end_date}")

    # Fetch from Visual Crossing (handles batching internally)
    df = client.fetch_range_for_city(city_id, start_date, end_date)

    if df.empty:
        logger.warning(f"No data returned for {city_id}")
        return 0

    logger.info(f"Fetched {len(df)} minute records for {city_id}")

    # Convert to DB format
    records = df_to_db_records(df, city.icao)

    # Upsert in batches
    batch_size = 1000
    total_upserted = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        rows = upsert_minute_obs(session, batch)
        total_upserted += rows

        if (i + batch_size) % 5000 == 0:
            logger.info(f"  Progress: {i + batch_size}/{len(records)} records")
            session.commit()

    return total_upserted


def main():
    parser = argparse.ArgumentParser(description="Ingest Visual Crossing minute observations")
    parser.add_argument("--days", type=int, default=60, help="Number of days to ingest (default: 60)")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD), overrides --days")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD), default: yesterday")
    parser.add_argument("--all-history", action="store_true", help="Fetch all available history (since 2022)")
    parser.add_argument("--city", type=str, help="Single city to ingest (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoint tracking")

    args = parser.parse_args()

    # Calculate date range (use yesterday as default end to ensure data is available)
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now(timezone.utc) - timedelta(days=1)

    if args.all_history:
        # Visual Crossing has historical data going back years
        start_date = datetime(2022, 1, 1)
        logger.info("Fetching ALL available history (since 2022)")
    elif args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=args.days - 1)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    logger.info(f"Ingesting Visual Crossing minute data from {start_str} to {end_str}")

    # Get settings and create client
    settings = get_settings()
    client = VisualCrossingClient(
        api_key=settings.vc_api_key,
        base_url=settings.vc_base_url,
        minute_interval=settings.wx_minute_interval,
    )

    # Determine which cities to process
    if args.city:
        if args.city in EXCLUDED_VC_CITIES:
            logger.error(f"{args.city} is excluded from Visual Crossing (high forward-fill)")
            return
        cities = [args.city]
    else:
        cities = [c for c in CITIES.keys() if c not in EXCLUDED_VC_CITIES]

    logger.info(f"Processing cities: {cities}")

    if args.dry_run:
        logger.info("DRY RUN - fetching but not writing to database")
        for city_id in cities:
            city = get_city(city_id)
            logger.info(f"Would ingest {city_id} ({city.icao}) from {start_str} to {end_str}")
        return

    # Ingest each city with checkpoint tracking
    total_records = 0
    with get_db_session() as session:
        for city_id in cities:
            # Get or create checkpoint for this city
            checkpoint = None
            if not args.no_checkpoint:
                checkpoint = get_or_create_checkpoint(
                    session=session,
                    pipeline_name="vc_minutes",
                    city=city_id,
                )
                session.commit()

            try:
                count = ingest_city(
                    client=client,
                    session=session,
                    city_id=city_id,
                    start_date=start_str,
                    end_date=end_str,
                )
                total_records += count
                logger.info(f"{city_id}: upserted {count} records")

                # Update checkpoint on success
                if checkpoint:
                    update_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint.id,
                        last_date=end_date.date() if hasattr(end_date, 'date') else end_date,
                        processed_count=count,
                    )
                session.commit()

                # Mark checkpoint completed for this city
                if checkpoint:
                    complete_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint.id,
                        status="completed",
                    )
                    session.commit()

            except Exception as e:
                logger.error(f"Error ingesting {city_id}: {e}")
                if checkpoint:
                    update_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint.id,
                        error=str(e),
                    )
                    complete_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint.id,
                        status="failed",
                    )
                    session.commit()
                session.rollback()
                raise

    logger.info(f"Ingestion complete: {total_records} total records upserted")


if __name__ == "__main__":
    main()
