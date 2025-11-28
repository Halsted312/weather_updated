#!/usr/bin/env python3
"""
PARALLEL Visual Crossing observation ingestion - uses multiprocessing for speed.

Fetches 5-minute data for all cities concurrently, utilizing multiple CPU cores
and fast network connection.

Usage:
    python scripts/ingest_vc_obs_parallel.py --start-date 2023-01-01 --end-date 2025-11-27
    python scripts/ingest_vc_obs_parallel.py --start-date 2023-01-01 --end-date 2025-11-27 --workers 32
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import get_settings
from src.db import get_db_session, VcLocation, VcMinuteWeather
from src.weather.visual_crossing import VisualCrossingClient
from zoneinfo import ZoneInfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_vc_minute_to_record(
    minute_data: Dict[str, Any],
    vc_location_id: int,
    iana_timezone: str,
) -> Dict[str, Any]:
    """Parse VC minute record with CORRECT timezone conversion."""
    epoch = minute_data.get("datetimeEpoch")
    if not epoch:
        return None

    # UTC datetime
    dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)

    # Convert to local using IANA timezone (don't trust VC's tzoffset!)
    tz_local = ZoneInfo(iana_timezone)
    dt_local_aware = dt_utc.astimezone(tz_local)
    dt_local_naive = dt_local_aware.replace(tzinfo=None)

    # Calculate offset
    offset_seconds = dt_local_aware.utcoffset().total_seconds()
    tzoffset_minutes = int(offset_seconds / 60)

    def _fmt_list(val):
        if val is None:
            return None
        if isinstance(val, list):
            return ",".join(str(v) for v in val)
        return str(val)

    return {
        "vc_location_id": vc_location_id,
        "data_type": "actual_obs",
        "forecast_basis_date": None,
        "forecast_basis_datetime_utc": None,
        "lead_hours": None,
        "datetime_epoch_utc": epoch,
        "datetime_utc": dt_utc,
        "datetime_local": dt_local_naive,
        "timezone": minute_data.get("timezone") or iana_timezone,
        "tzoffset_minutes": tzoffset_minutes,
        "temp_f": minute_data.get("temp"),
        "tempmax_f": minute_data.get("tempmax"),
        "tempmin_f": minute_data.get("tempmin"),
        "feelslike_f": minute_data.get("feelslike"),
        "dew_f": minute_data.get("dew"),
        "humidity": minute_data.get("humidity"),
        "precip_in": minute_data.get("precip"),
        "precipprob": minute_data.get("precipprob"),
        "preciptype": _fmt_list(minute_data.get("preciptype")),
        "snow_in": minute_data.get("snow"),
        "snowdepth_in": minute_data.get("snowdepth"),
        "windspeed_mph": minute_data.get("windspeed"),
        "windgust_mph": minute_data.get("windgust"),
        "winddir": minute_data.get("winddir"),
        "cloudcover": minute_data.get("cloudcover"),
        "visibility_miles": minute_data.get("visibility"),
        "pressure_mb": minute_data.get("pressure"),
        "uvindex": minute_data.get("uvindex"),
        "solarradiation": minute_data.get("solarradiation"),
        "cape": minute_data.get("cape"),
        "cin": minute_data.get("cin"),
        "conditions": minute_data.get("conditions"),
        "source_system": "vc_timeline",
        "raw_json": minute_data,
    }


def fetch_and_parse_batch(
    location_query: str,
    start_date: date,
    end_date: date,
    vc_location_id: int,
    iana_timezone: str,
) -> Tuple[str, int]:
    """Fetch one date range batch for one location.

    Returns (location_query, record_count) for logging.
    """
    settings = get_settings()
    client = VisualCrossingClient(api_key=settings.vc_api_key)

    try:
        # Extract station ID from location_query (e.g., "stn:KMDW" → "KMDW")
        station_id = location_query.replace("stn:", "")

        data = client.fetch_station_history_minutes(
            station_id=station_id,
            start_date=start_date,
            end_date=end_date,
        )

        # Parse all minutes
        records = []
        for day_data in data.get("days", []):
            for minute_data in day_data.get("hours", []):
                record = parse_vc_minute_to_record(minute_data, vc_location_id, iana_timezone)
                if record:
                    records.append(record)

        # Batch insert
        if records:
            with get_db_session() as session:
                stmt = insert(VcMinuteWeather).values(records)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["vc_location_id", "data_type", "forecast_basis_date", "datetime_utc"],
                    set_={"temp_f": stmt.excluded.temp_f, "humidity": stmt.excluded.humidity},
                )
                session.execute(stmt)
                session.commit()

        return (location_query, len(records))

    except Exception as e:
        logger.error(f"Error fetching {location_query} {start_date} to {end_date}: {e}")
        return (location_query, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--workers", type=int, default=28, help="Concurrent workers (default: 28)")
    parser.add_argument("--batch-days", type=int, default=7, help="Days per batch (default: 7)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)

    # Get all station locations and materialize data
    with get_db_session() as session:
        from sqlalchemy import select
        locs = session.execute(
            select(VcLocation).where(VcLocation.location_type == "station")
        ).scalars().all()

        # Materialize location data (detach from session for threading)
        locations = [
            {
                "query": loc.vc_location_query,
                "id": loc.id,
                "timezone": loc.iana_timezone,
                "city_code": loc.city_code,
            }
            for loc in locs
        ]

    logger.info(f"Found {len(locations)} stations to process")
    logger.info(f"Date range: {start} to {end}")
    logger.info(f"Workers: {args.workers}, Batch size: {args.batch_days} days")

    # Build work queue: (location_query, start_date, end_date, location_id, timezone)
    work_items = []
    for loc in locations:
        current = start
        while current <= end:
            batch_end = min(current + timedelta(days=args.batch_days - 1), end)
            work_items.append((loc["query"], current, batch_end, loc["id"], loc["timezone"]))
            current = batch_end + timedelta(days=1)

    logger.info(f"Total work items: {len(work_items)}")

    # Process in parallel
    completed = 0
    total_records = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_and_parse_batch, *item): item
            for item in work_items
        }

        for future in as_completed(futures):
            location_query, count = future.result()
            completed += 1
            total_records += count

            if completed % 10 == 0:
                progress_pct = 100 * completed / len(work_items)
                logger.info(f"Progress: {completed}/{len(work_items)} ({progress_pct:.1f}%) - {total_records:,} records")

    logger.info(f"COMPLETE: {total_records:,} total records from {len(work_items)} batches")


if __name__ == "__main__":
    main()
