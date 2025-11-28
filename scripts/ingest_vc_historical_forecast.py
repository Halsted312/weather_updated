#!/usr/bin/env python3
"""
Ingest Visual Crossing historical forecasts for backtesting.

Fetches what the forecast looked like on past dates using the `forecastBasisDate`
parameter. This enables ML model training on forecast accuracy (comparing
what was predicted vs what actually happened).

Stores into:
- wx.vc_forecast_daily (daily forecasts with data_type='historical_forecast')
- wx.vc_forecast_hourly (hourly forecasts with data_type='historical_forecast')

Usage:
    # Backfill last 30 days of forecast history
    python scripts/ingest_vc_historical_forecast_v2.py --days 30

    # Specific date range
    python scripts/ingest_vc_historical_forecast_v2.py --start-date 2024-01-01 --end-date 2024-06-30

    # Single city, station only
    python scripts/ingest_vc_historical_forecast_v2.py --city-code CHI --location-type station --days 7

    # Dry run
    python scripts/ingest_vc_historical_forecast_v2.py --dry-run --days 7
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
from src.db import get_db_session, VcLocation, VcForecastDaily, VcForecastHourly
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


def _format_list(value: Any) -> Optional[str]:
    """Format list as comma-separated string."""
    if isinstance(value, list):
        return ",".join(str(v) for v in value) if value else None
    elif isinstance(value, str):
        return value
    return None


def parse_historical_daily(
    day_data: Dict[str, Any],
    vc_location_id: int,
    basis_date: date,
    basis_datetime_utc: datetime,
) -> Dict[str, Any]:
    """Parse a historical daily forecast record."""
    target_date_str = day_data.get("datetime")
    if not target_date_str:
        return None

    target_date = date.fromisoformat(target_date_str)
    lead_days = (target_date - basis_date).days

    return {
        "vc_location_id": vc_location_id,
        "data_type": "historical_forecast",
        "forecast_basis_date": basis_date,
        "forecast_basis_datetime_utc": basis_datetime_utc,
        "target_date": target_date,
        "lead_days": lead_days,
        # Weather fields
        "tempmax_f": day_data.get("tempmax"),
        "tempmin_f": day_data.get("tempmin"),
        "temp_f": day_data.get("temp"),
        "feelslikemax_f": day_data.get("feelslikemax"),
        "feelslikemin_f": day_data.get("feelslikemin"),
        "feelslike_f": day_data.get("feelslike"),
        "dew_f": day_data.get("dew"),
        "humidity": day_data.get("humidity"),
        "precip_in": day_data.get("precip"),
        "precipprob": day_data.get("precipprob"),
        "preciptype": _format_list(day_data.get("preciptype")),
        "precipcover": day_data.get("precipcover"),
        "snow_in": day_data.get("snow"),
        "snowdepth_in": day_data.get("snowdepth"),
        "windspeed_mph": day_data.get("windspeed"),
        "windgust_mph": day_data.get("windgust"),
        "winddir": day_data.get("winddir"),
        "windspeedmean_mph": day_data.get("windspeedmean"),
        "windspeedmin_mph": day_data.get("windspeedmin"),
        "windspeedmax_mph": day_data.get("windspeedmax"),
        "windspeed50_mph": day_data.get("windspeed50"),
        "winddir50": day_data.get("winddir50"),
        "windspeed80_mph": day_data.get("windspeed80"),
        "winddir80": day_data.get("winddir80"),
        "windspeed100_mph": day_data.get("windspeed100"),
        "winddir100": day_data.get("winddir100"),
        "cloudcover": day_data.get("cloudcover"),
        "visibility_miles": day_data.get("visibility"),
        "pressure_mb": day_data.get("pressure"),
        "uvindex": day_data.get("uvindex"),
        "solarradiation": day_data.get("solarradiation"),
        "solarenergy": day_data.get("solarenergy"),
        "dniradiation": day_data.get("dniradiation"),
        "difradiation": day_data.get("difradiation"),
        "ghiradiation": day_data.get("ghiradiation"),
        "gtiradiation": day_data.get("gtiradiation"),
        "cape": day_data.get("cape"),
        "cin": day_data.get("cin"),
        "deltat": day_data.get("deltat"),
        "degreedays": day_data.get("degreedays"),
        "accdegreedays": day_data.get("accdegreedays"),
        "conditions": day_data.get("conditions"),
        "icon": day_data.get("icon"),
        "source_system": "vc_timeline",
        "raw_json": day_data,
    }


def parse_historical_hourly(
    hour_data: Dict[str, Any],
    vc_location_id: int,
    basis_date: date,
    basis_datetime_utc: datetime,
    iana_timezone: str,
) -> Dict[str, Any]:
    """Parse a historical hourly forecast record."""
    epoch = hour_data.get("datetimeEpoch")
    if not epoch:
        return None

    target_dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)

    # Get timezone offset
    tzoffset_hours = hour_data.get("tzoffset", 0) or 0
    tzoffset_minutes = int(tzoffset_hours * 60)

    # Calculate local datetime
    target_dt_local = target_dt_utc + timedelta(minutes=tzoffset_minutes)
    target_dt_local_naive = target_dt_local.replace(tzinfo=None)

    # Calculate lead hours
    lead_hours = int((target_dt_utc - basis_datetime_utc).total_seconds() / 3600)

    return {
        "vc_location_id": vc_location_id,
        "data_type": "historical_forecast",
        "forecast_basis_date": basis_date,
        "forecast_basis_datetime_utc": basis_datetime_utc,
        "target_datetime_epoch_utc": epoch,
        "target_datetime_utc": target_dt_utc,
        "target_datetime_local": target_dt_local_naive,
        "timezone": hour_data.get("timezone") or iana_timezone,
        "tzoffset_minutes": tzoffset_minutes,
        "lead_hours": lead_hours,
        # Weather fields
        "temp_f": hour_data.get("temp"),
        "feelslike_f": hour_data.get("feelslike"),
        "dew_f": hour_data.get("dew"),
        "humidity": hour_data.get("humidity"),
        "precip_in": hour_data.get("precip"),
        "precipprob": hour_data.get("precipprob"),
        "preciptype": _format_list(hour_data.get("preciptype")),
        "snow_in": hour_data.get("snow"),
        "windspeed_mph": hour_data.get("windspeed"),
        "windgust_mph": hour_data.get("windgust"),
        "winddir": hour_data.get("winddir"),
        "windspeed50_mph": hour_data.get("windspeed50"),
        "winddir50": hour_data.get("winddir50"),
        "windspeed80_mph": hour_data.get("windspeed80"),
        "winddir80": hour_data.get("winddir80"),
        "windspeed100_mph": hour_data.get("windspeed100"),
        "winddir100": hour_data.get("winddir100"),
        "cloudcover": hour_data.get("cloudcover"),
        "visibility_miles": hour_data.get("visibility"),
        "pressure_mb": hour_data.get("pressure"),
        "uvindex": hour_data.get("uvindex"),
        "solarradiation": hour_data.get("solarradiation"),
        "solarenergy": hour_data.get("solarenergy"),
        "dniradiation": hour_data.get("dniradiation"),
        "difradiation": hour_data.get("difradiation"),
        "ghiradiation": hour_data.get("ghiradiation"),
        "gtiradiation": hour_data.get("gtiradiation"),
        "sunelevation": hour_data.get("sunelevation"),
        "sunazimuth": hour_data.get("sunazimuth"),
        "cape": hour_data.get("cape"),
        "cin": hour_data.get("cin"),
        "conditions": hour_data.get("conditions"),
        "icon": hour_data.get("icon"),
        "source_system": "vc_timeline",
        "raw_json": hour_data,
    }


def upsert_historical_daily(session, records: List[Dict[str, Any]]) -> int:
    """Upsert historical daily forecast records."""
    if not records:
        return 0

    stmt = insert(VcForecastDaily).values(records)
    update_cols = {
        col.name: stmt.excluded[col.name]
        for col in VcForecastDaily.__table__.columns
        if col.name not in ("id", "vc_location_id", "target_date", "forecast_basis_date", "data_type", "created_at")
    }

    stmt = stmt.on_conflict_do_update(
        constraint="uq_vc_daily_row",
        set_=update_cols,
    )

    result = session.execute(stmt)
    return result.rowcount


def upsert_historical_hourly(session, records: List[Dict[str, Any]]) -> int:
    """Upsert historical hourly forecast records."""
    if not records:
        return 0

    stmt = insert(VcForecastHourly).values(records)
    update_cols = {
        col.name: stmt.excluded[col.name]
        for col in VcForecastHourly.__table__.columns
        if col.name not in ("id", "vc_location_id", "target_datetime_utc", "forecast_basis_date", "data_type", "created_at")
    }

    stmt = stmt.on_conflict_do_update(
        constraint="uq_vc_hourly_row",
        set_=update_cols,
    )

    result = session.execute(stmt)
    return result.rowcount


def ingest_location_historical_forecast(
    client: VisualCrossingClient,
    session,
    location: VcLocation,
    basis_date: date,
    horizon_days: int,
) -> Dict[str, int]:
    """
    Fetch and upsert historical forecast for a single location and basis date.

    Args:
        client: Visual Crossing client
        session: Database session
        location: VcLocation record
        basis_date: The date the forecast was made
        horizon_days: Forecast horizon in days

    Returns:
        Dict with counts for daily, hourly records
    """
    # Assume forecast was made at noon UTC on the basis date
    basis_datetime_utc = datetime.combine(basis_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    basis_datetime_utc = basis_datetime_utc.replace(hour=12)

    # Target range: from basis_date to basis_date + horizon_days
    target_start = basis_date
    target_end = basis_date + timedelta(days=horizon_days - 1)

    logger.debug(
        f"Fetching historical forecast for {location.city_code}/{location.location_type}, "
        f"basis={basis_date}, target={target_start} to {target_end}"
    )

    try:
        # Fetch based on location type
        if location.location_type == "station":
            data = client.fetch_station_historical_forecast(
                station_id=location.station_id,
                target_start=target_start,
                target_end=target_end,
                basis_date=basis_date,
            )
        else:  # city
            data = client.fetch_city_historical_forecast(
                city_query=location.vc_location_query,
                target_start=target_start,
                target_end=target_end,
                basis_date=basis_date,
            )

        # Parse records
        daily_records = []
        hourly_records = []

        for day in data.get("days", []):
            # Daily record
            daily_rec = parse_historical_daily(day, location.id, basis_date, basis_datetime_utc)
            if daily_rec:
                daily_records.append(daily_rec)

            # Hourly records
            for hour in day.get("hours", []):
                hourly_rec = parse_historical_hourly(
                    hour, location.id, basis_date, basis_datetime_utc, location.iana_timezone
                )
                if hourly_rec:
                    hourly_records.append(hourly_rec)

        # Upsert
        daily_count = upsert_historical_daily(session, daily_records)
        hourly_count = upsert_historical_hourly(session, hourly_records)

        return {
            "daily": daily_count,
            "hourly": hourly_count,
        }

    except Exception as e:
        logger.error(f"Error fetching historical forecast: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Visual Crossing historical forecasts for backtesting"
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Number of days of basis dates to backfill (default: 30)"
    )
    parser.add_argument(
        "--start-date", type=str,
        help="Start basis date (YYYY-MM-DD), overrides --days"
    )
    parser.add_argument(
        "--end-date", type=str,
        help="End basis date (YYYY-MM-DD), default: yesterday"
    )
    parser.add_argument(
        "--horizon-days", type=int, default=7,
        help="Forecast horizon per basis date (default: 7)"
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
        "--dry-run", action="store_true",
        help="Don't write to database"
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable checkpoint tracking"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Calculate basis date range
    if args.end_date:
        end_basis_date = date.fromisoformat(args.end_date)
    else:
        # Default: yesterday (can't get forecast for today's basis date)
        end_basis_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()

    if args.start_date:
        start_basis_date = date.fromisoformat(args.start_date)
    else:
        start_basis_date = end_basis_date - timedelta(days=args.days - 1)

    logger.info(
        f"Ingesting historical forecasts for basis dates {start_basis_date} to {end_basis_date}, "
        f"horizon={args.horizon_days} days"
    )

    # Get settings and create client
    settings = get_settings()
    client = VisualCrossingClient(
        api_key=settings.vc_api_key,
        base_url=settings.vc_base_url,
    )

    # Query locations from database
    with get_db_session() as session:
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
            logger.info(f"  Would fetch: {loc.city_code}/{loc.location_type}")
        logger.info(f"Basis dates: {start_basis_date} to {end_basis_date}")
        return

    # Process each location and basis date
    totals = {"daily": 0, "hourly": 0}
    total_basis_dates = (end_basis_date - start_basis_date).days + 1

    with get_db_session() as session:
        for location in locations:
            checkpoint = None
            checkpoint_key = f"{location.city_code}_{location.location_type}"

            if not args.no_checkpoint:
                checkpoint = get_or_create_checkpoint(
                    session=session,
                    pipeline_name="vc_historical_forecast",
                    city=checkpoint_key,
                )
                session.commit()

            location_totals = {"daily": 0, "hourly": 0}
            current_basis_date = start_basis_date

            logger.info(
                f"Processing {location.city_code}/{location.location_type}: "
                f"{total_basis_dates} basis dates"
            )

            try:
                while current_basis_date <= end_basis_date:
                    counts = ingest_location_historical_forecast(
                        client=client,
                        session=session,
                        location=location,
                        basis_date=current_basis_date,
                        horizon_days=args.horizon_days,
                    )

                    location_totals["daily"] += counts["daily"]
                    location_totals["hourly"] += counts["hourly"]

                    # Commit per basis date for resume-on-crash
                    session.commit()

                    current_basis_date += timedelta(days=1)

                totals["daily"] += location_totals["daily"]
                totals["hourly"] += location_totals["hourly"]

                logger.info(
                    f"{location.city_code}/{location.location_type}: "
                    f"{location_totals['daily']} daily, {location_totals['hourly']} hourly"
                )

                if checkpoint:
                    update_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint.id,
                        last_date=end_basis_date,
                        processed_count=sum(location_totals.values()),
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

    logger.info(
        f"Ingestion complete: {totals['daily']} daily, {totals['hourly']} hourly records"
    )


if __name__ == "__main__":
    main()
