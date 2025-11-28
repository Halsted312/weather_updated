#!/usr/bin/env python3
"""
Ingest Visual Crossing minute-level observations into wx.vc_minute_weather.

Fetches 5-minute interval data for all locations in wx.vc_location (both station
and city types) and upserts into the new canonical vc_minute_weather table.

Supports resume-on-crash via checkpoints in meta.ingestion_checkpoint.

Usage:
    python scripts/ingest_vc_obs_backfill.py --days 7
    python scripts/ingest_vc_obs_backfill.py --start-date 2024-01-01 --end-date 2024-03-31
    python scripts/ingest_vc_obs_backfill.py --location-type station --days 30
    python scripts/ingest_vc_obs_backfill.py --city-code CHI --days 7
    python scripts/ingest_vc_obs_backfill.py --all-history  # Fetch all available (since 2022)
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import get_settings
from src.db import get_db_session, VcLocation, VcMinuteWeather
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


def parse_vc_minute_to_record(
    minute_data: Dict[str, Any],
    vc_location_id: int,
    iana_timezone: str,
) -> Dict[str, Any]:
    """
    Parse a single VC minute record to database format.

    Args:
        minute_data: Raw minute data from VC API
        vc_location_id: Foreign key to vc_location
        iana_timezone: IANA timezone name from location

    Returns:
        Dict ready for insertion into vc_minute_weather
    """
    # Get epoch timestamp
    epoch = minute_data.get("datetimeEpoch")
    if not epoch:
        return None

    # Parse datetime
    dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)

    # Get timezone offset (VC returns hours, we store minutes)
    tzoffset_hours = minute_data.get("tzoffset", 0) or 0
    tzoffset_minutes = int(tzoffset_hours * 60)

    # Calculate local datetime
    dt_local = dt_utc + timedelta(minutes=tzoffset_minutes)
    dt_local_naive = dt_local.replace(tzinfo=None)

    record = {
        "vc_location_id": vc_location_id,
        "data_type": "actual_obs",
        "forecast_basis_date": None,
        "forecast_basis_datetime_utc": None,
        "lead_hours": None,
        # Time fields
        "datetime_epoch_utc": epoch,
        "datetime_utc": dt_utc,
        "datetime_local": dt_local_naive,
        "timezone": minute_data.get("timezone") or iana_timezone,
        "tzoffset_minutes": tzoffset_minutes,
        # Weather fields - map using VC_TO_DB_FIELD_MAP
        "temp_f": minute_data.get("temp"),
        "tempmax_f": minute_data.get("tempmax"),
        "tempmin_f": minute_data.get("tempmin"),
        "feelslike_f": minute_data.get("feelslike"),
        "feelslikemax_f": minute_data.get("feelslikemax"),
        "feelslikemin_f": minute_data.get("feelslikemin"),
        "dew_f": minute_data.get("dew"),
        "humidity": minute_data.get("humidity"),
        "precip_in": minute_data.get("precip"),
        "precipprob": minute_data.get("precipprob"),
        "preciptype": _format_list(minute_data.get("preciptype")),
        "precipcover": minute_data.get("precipcover"),
        "snow_in": minute_data.get("snow"),
        "snowdepth_in": minute_data.get("snowdepth"),
        "precipremote": minute_data.get("precipremote"),
        "windspeed_mph": minute_data.get("windspeed"),
        "windgust_mph": minute_data.get("windgust"),
        "winddir": minute_data.get("winddir"),
        "windspeedmean_mph": minute_data.get("windspeedmean"),
        "windspeedmin_mph": minute_data.get("windspeedmin"),
        "windspeedmax_mph": minute_data.get("windspeedmax"),
        "windspeed50_mph": minute_data.get("windspeed50"),
        "winddir50": minute_data.get("winddir50"),
        "windspeed80_mph": minute_data.get("windspeed80"),
        "winddir80": minute_data.get("winddir80"),
        "windspeed100_mph": minute_data.get("windspeed100"),
        "winddir100": minute_data.get("winddir100"),
        "cloudcover": minute_data.get("cloudcover"),
        "visibility_miles": minute_data.get("visibility"),
        "pressure_mb": minute_data.get("pressure"),
        "uvindex": minute_data.get("uvindex"),
        "solarradiation": minute_data.get("solarradiation"),
        "solarenergy": minute_data.get("solarenergy"),
        "dniradiation": minute_data.get("dniradiation"),
        "difradiation": minute_data.get("difradiation"),
        "ghiradiation": minute_data.get("ghiradiation"),
        "gtiradiation": minute_data.get("gtiradiation"),
        "sunelevation": minute_data.get("sunelevation"),
        "sunazimuth": minute_data.get("sunazimuth"),
        "cape": minute_data.get("cape"),
        "cin": minute_data.get("cin"),
        "deltat": minute_data.get("deltat"),
        "degreedays": minute_data.get("degreedays"),
        "accdegreedays": minute_data.get("accdegreedays"),
        "conditions": minute_data.get("conditions"),
        "icon": minute_data.get("icon"),
        "stations": _format_list(minute_data.get("stations")),
        "resolved_address": minute_data.get("resolvedAddress"),
        "source_system": "vc_timeline",
        "raw_json": minute_data,
    }

    return record


def _format_list(value: Any) -> Optional[str]:
    """Format list as comma-separated string."""
    if isinstance(value, list):
        return ",".join(str(v) for v in value) if value else None
    elif isinstance(value, str):
        return value
    return None


def flatten_vc_response_to_minutes(
    data: Dict[str, Any],
    vc_location_id: int,
    iana_timezone: str,
) -> List[Dict[str, Any]]:
    """
    Flatten VC API response to list of minute records.

    Args:
        data: Raw VC API response
        vc_location_id: Foreign key to vc_location
        iana_timezone: IANA timezone from location

    Returns:
        List of records ready for insertion
    """
    records = []

    for day in data.get("days", []):
        for hour in day.get("hours", []):
            for minute in hour.get("minutes", []):
                record = parse_vc_minute_to_record(minute, vc_location_id, iana_timezone)
                if record:
                    records.append(record)

    return records


def upsert_vc_minute_weather(session, records: List[Dict[str, Any]]) -> int:
    """
    Upsert minute weather records into wx.vc_minute_weather.

    Uses ON CONFLICT with the partial unique index `uq_vc_minute_obs`
    on (vc_location_id, data_type, datetime_utc) WHERE forecast_basis_date IS NULL.

    Returns:
        Number of rows affected
    """
    if not records:
        return 0

    stmt = insert(VcMinuteWeather).values(records)

    # On conflict, update all weather fields
    update_cols = {
        col.name: stmt.excluded[col.name]
        for col in VcMinuteWeather.__table__.columns
        if col.name not in ("id", "vc_location_id", "data_type", "forecast_basis_date", "datetime_utc", "created_at")
    }

    # Use partial index for obs (WHERE forecast_basis_date IS NULL)
    stmt = stmt.on_conflict_do_update(
        index_elements=["vc_location_id", "data_type", "datetime_utc"],
        index_where=VcMinuteWeather.forecast_basis_date.is_(None),
        set_=update_cols,
    )

    result = session.execute(stmt)
    return result.rowcount


def ingest_location(
    client: VisualCrossingClient,
    session,
    location: VcLocation,
    start_date: date,
    end_date: date,
    batch_days: int = 7,
) -> int:
    """
    Fetch and upsert minute observations for a single location.

    Args:
        client: Visual Crossing client
        session: Database session
        location: VcLocation record
        start_date: Start date
        end_date: End date
        batch_days: Days per API call (default: 7)

    Returns:
        Total records upserted
    """
    logger.info(
        f"Ingesting {location.city_code}/{location.location_type} "
        f"({location.vc_location_query}) from {start_date} to {end_date}"
    )

    total_upserted = 0
    current_start = start_date

    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=batch_days - 1), end_date)

        try:
            # Fetch based on location type
            if location.location_type == "station":
                data = client.fetch_station_history_minutes(
                    station_id=location.station_id,
                    start_date=current_start,
                    end_date=current_end,
                )
            else:  # city
                data = client.fetch_city_history_minutes(
                    city_query=location.vc_location_query,
                    start_date=current_start,
                    end_date=current_end,
                )

            # Parse to records
            records = flatten_vc_response_to_minutes(
                data, location.id, location.iana_timezone
            )

            if records:
                # Batch insert
                batch_size = 1000
                for i in range(0, len(records), batch_size):
                    batch = records[i : i + batch_size]
                    rows = upsert_vc_minute_weather(session, batch)
                    total_upserted += rows

                logger.info(
                    f"  {current_start} to {current_end}: {len(records)} records"
                )
            else:
                logger.warning(f"  {current_start} to {current_end}: no data")

        except Exception as e:
            logger.error(f"Error fetching {current_start} to {current_end}: {e}")
            # Continue with next batch

        current_start = current_end + timedelta(days=1)

    return total_upserted


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Visual Crossing minute observations into vc_minute_weather"
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Number of days to ingest (default: 7)"
    )
    parser.add_argument(
        "--start-date", type=str,
        help="Start date (YYYY-MM-DD), overrides --days"
    )
    parser.add_argument(
        "--end-date", type=str,
        help="End date (YYYY-MM-DD), default: yesterday"
    )
    parser.add_argument(
        "--all-history", action="store_true",
        help="Fetch all available history (since 2022-01-01)"
    )
    parser.add_argument(
        "--location-type", type=str, choices=["station", "city"],
        help="Filter by location type (default: both)"
    )
    parser.add_argument(
        "--city-code", type=str,
        help="Filter by city code (e.g., CHI, DEN)"
    )
    parser.add_argument(
        "--batch-days", type=int, default=7,
        help="Days per API call (default: 7)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Don't write to database"
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable checkpoint tracking"
    )

    args = parser.parse_args()

    # Calculate date range
    if args.end_date:
        end_date = date.fromisoformat(args.end_date)
    else:
        end_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()

    if args.all_history:
        start_date = date(2022, 1, 1)
        logger.info("Fetching ALL available history (since 2022-01-01)")
    elif args.start_date:
        start_date = date.fromisoformat(args.start_date)
    else:
        start_date = end_date - timedelta(days=args.days - 1)

    logger.info(f"Ingesting VC observations from {start_date} to {end_date}")

    # Get settings and create client
    settings = get_settings()
    client = VisualCrossingClient(
        api_key=settings.vc_api_key,
        base_url=settings.vc_base_url,
    )

    # Process each location
    total_records = 0

    with get_db_session() as session:
        # Query locations from database
        query = select(VcLocation)

        if args.location_type:
            query = query.where(VcLocation.location_type == args.location_type)
        if args.city_code:
            query = query.where(VcLocation.city_code == args.city_code.upper())

        locations = list(session.execute(query).scalars().all())

        if not locations:
            logger.error("No locations found matching criteria")
            return

        logger.info(f"Found {len(locations)} locations to process")

        if args.dry_run:
            logger.info("DRY RUN - would fetch but not write to database")
            for loc in locations:
                logger.info(f"  Would ingest: {loc.city_code}/{loc.location_type}")
            return

        for location in locations:
            checkpoint = None
            checkpoint_key = f"{location.city_code}_{location.location_type}"

            if not args.no_checkpoint:
                checkpoint = get_or_create_checkpoint(
                    session=session,
                    pipeline_name="vc_obs_backfill",
                    city=checkpoint_key,
                )
                session.commit()

            try:
                count = ingest_location(
                    client=client,
                    session=session,
                    location=location,
                    start_date=start_date,
                    end_date=end_date,
                    batch_days=args.batch_days,
                )
                total_records += count
                session.commit()

                logger.info(
                    f"{location.city_code}/{location.location_type}: "
                    f"upserted {count} records"
                )

                if checkpoint:
                    update_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint.id,
                        last_date=end_date,
                        processed_count=count,
                    )
                    complete_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint.id,
                        status="completed",
                    )
                    session.commit()

            except Exception as e:
                logger.error(f"Error processing {location.city_code}/{location.location_type}: {e}")
                session.rollback()

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

    logger.info(f"Ingestion complete: {total_records} total records upserted")


if __name__ == "__main__":
    main()
