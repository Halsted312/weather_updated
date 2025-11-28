#!/usr/bin/env python3
"""
Ingest Visual Crossing current + forecast snapshots.

Fetches current conditions and forecasts for all locations in wx.vc_location,
inserting into:
- wx.vc_minute_weather (15-min forecast minutes with data_type='forecast')
- wx.vc_forecast_hourly (hourly forecasts)
- wx.vc_forecast_daily (daily forecasts)

This script is intended to run nightly (or more frequently) to capture
forecast snapshots for ML training and backtesting.

Usage:
    python scripts/ingest_vc_forecast_snapshot.py
    python scripts/ingest_vc_forecast_snapshot.py --horizon-days 7
    python scripts/ingest_vc_forecast_snapshot.py --location-type station
    python scripts/ingest_vc_forecast_snapshot.py --city-code CHI
    python scripts/ingest_vc_forecast_snapshot.py --dry-run
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
from src.db import get_db_session, VcLocation, VcMinuteWeather, VcForecastDaily, VcForecastHourly
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


def parse_forecast_daily(
    day_data: Dict[str, Any],
    vc_location_id: int,
    basis_date: date,
    basis_datetime_utc: datetime,
) -> Dict[str, Any]:
    """Parse a daily forecast record."""
    target_date_str = day_data.get("datetime")
    if not target_date_str:
        return None

    target_date = date.fromisoformat(target_date_str)
    lead_days = (target_date - basis_date).days

    return {
        "vc_location_id": vc_location_id,
        "data_type": "forecast",
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


def parse_forecast_hourly(
    hour_data: Dict[str, Any],
    vc_location_id: int,
    basis_date: date,
    basis_datetime_utc: datetime,
    iana_timezone: str,
) -> Dict[str, Any]:
    """Parse an hourly forecast record."""
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
        "data_type": "forecast",
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


def parse_forecast_minute(
    minute_data: Dict[str, Any],
    vc_location_id: int,
    basis_date: date,
    basis_datetime_utc: datetime,
    iana_timezone: str,
) -> Dict[str, Any]:
    """Parse a minute-level forecast record."""
    epoch = minute_data.get("datetimeEpoch")
    if not epoch:
        return None

    dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)

    # Get timezone offset
    tzoffset_hours = minute_data.get("tzoffset", 0) or 0
    tzoffset_minutes = int(tzoffset_hours * 60)

    # Calculate local datetime
    dt_local = dt_utc + timedelta(minutes=tzoffset_minutes)
    dt_local_naive = dt_local.replace(tzinfo=None)

    # Calculate lead hours
    lead_hours = int((dt_utc - basis_datetime_utc).total_seconds() / 3600)

    return {
        "vc_location_id": vc_location_id,
        "data_type": "forecast",
        "forecast_basis_date": basis_date,
        "forecast_basis_datetime_utc": basis_datetime_utc,
        "lead_hours": lead_hours,
        # Time fields
        "datetime_epoch_utc": epoch,
        "datetime_utc": dt_utc,
        "datetime_local": dt_local_naive,
        "timezone": minute_data.get("timezone") or iana_timezone,
        "tzoffset_minutes": tzoffset_minutes,
        # Weather fields
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


def upsert_forecast_daily(session, records: List[Dict[str, Any]]) -> int:
    """Upsert daily forecast records."""
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


def upsert_forecast_hourly(session, records: List[Dict[str, Any]]) -> int:
    """Upsert hourly forecast records."""
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


def upsert_forecast_minutes(session, records: List[Dict[str, Any]]) -> int:
    """Upsert minute-level forecast records.

    Uses ON CONFLICT with the partial unique index `uq_vc_minute_fcst`
    on (vc_location_id, data_type, forecast_basis_date, datetime_utc)
    WHERE forecast_basis_date IS NOT NULL.
    """
    if not records:
        return 0

    stmt = insert(VcMinuteWeather).values(records)
    update_cols = {
        col.name: stmt.excluded[col.name]
        for col in VcMinuteWeather.__table__.columns
        if col.name not in ("id", "vc_location_id", "data_type", "forecast_basis_date", "datetime_utc", "created_at")
    }

    # Use partial index for forecasts (WHERE forecast_basis_date IS NOT NULL)
    stmt = stmt.on_conflict_do_update(
        index_elements=["vc_location_id", "data_type", "forecast_basis_date", "datetime_utc"],
        index_where=VcMinuteWeather.forecast_basis_date.isnot(None),
        set_=update_cols,
    )

    result = session.execute(stmt)
    return result.rowcount


def ingest_location_forecast(
    client: VisualCrossingClient,
    session,
    location: VcLocation,
    horizon_days: int,
) -> Dict[str, int]:
    """
    Fetch and upsert forecast snapshot for a single location.

    Returns:
        Dict with counts for daily, hourly, minute records
    """
    basis_datetime_utc = datetime.now(timezone.utc)
    basis_date = basis_datetime_utc.date()

    logger.info(
        f"Fetching forecast for {location.city_code}/{location.location_type} "
        f"({location.vc_location_query}), horizon={horizon_days} days"
    )

    try:
        # Fetch based on location type
        if location.location_type == "station":
            data = client.fetch_station_current_and_forecast(
                station_id=location.station_id,
                horizon_days=horizon_days,
            )
        else:  # city
            data = client.fetch_city_current_and_forecast(
                city_query=location.vc_location_query,
                horizon_days=horizon_days,
            )

        # Parse records
        daily_records = []
        hourly_records = []
        minute_records = []

        for day in data.get("days", []):
            # Daily record
            daily_rec = parse_forecast_daily(day, location.id, basis_date, basis_datetime_utc)
            if daily_rec:
                daily_records.append(daily_rec)

            # Hourly records
            for hour in day.get("hours", []):
                hourly_rec = parse_forecast_hourly(
                    hour, location.id, basis_date, basis_datetime_utc, location.iana_timezone
                )
                if hourly_rec:
                    hourly_records.append(hourly_rec)

                # Minute records
                for minute in hour.get("minutes", []):
                    minute_rec = parse_forecast_minute(
                        minute, location.id, basis_date, basis_datetime_utc, location.iana_timezone
                    )
                    if minute_rec:
                        minute_records.append(minute_rec)

        # Upsert
        daily_count = upsert_forecast_daily(session, daily_records)
        hourly_count = upsert_forecast_hourly(session, hourly_records)
        minute_count = upsert_forecast_minutes(session, minute_records)

        return {
            "daily": daily_count,
            "hourly": hourly_count,
            "minutes": minute_count,
        }

    except Exception as e:
        logger.error(f"Error fetching forecast: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Visual Crossing forecast snapshots"
    )
    parser.add_argument(
        "--horizon-days", type=int, default=7,
        help="Forecast horizon in days (default: 7)"
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

    args = parser.parse_args()

    logger.info(f"Ingesting VC forecast snapshots, horizon={args.horizon_days} days")

    # Get settings and create client
    settings = get_settings()
    client = VisualCrossingClient(
        api_key=settings.vc_api_key,
        base_url=settings.vc_base_url,
    )

    # Process each location
    totals = {"daily": 0, "hourly": 0, "minutes": 0}

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
                logger.info(f"  Would fetch: {loc.city_code}/{loc.location_type}")
            return

        for location in locations:
            checkpoint = None
            checkpoint_key = f"{location.city_code}_{location.location_type}"

            if not args.no_checkpoint:
                checkpoint = get_or_create_checkpoint(
                    session=session,
                    pipeline_name="vc_forecast_snapshot",
                    city=checkpoint_key,
                )
                session.commit()

            try:
                counts = ingest_location_forecast(
                    client=client,
                    session=session,
                    location=location,
                    horizon_days=args.horizon_days,
                )
                session.commit()

                totals["daily"] += counts["daily"]
                totals["hourly"] += counts["hourly"]
                totals["minutes"] += counts["minutes"]

                logger.info(
                    f"{location.city_code}/{location.location_type}: "
                    f"{counts['daily']} daily, {counts['hourly']} hourly, {counts['minutes']} minutes"
                )

                if checkpoint:
                    update_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint.id,
                        last_date=date.today(),
                        processed_count=sum(counts.values()),
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
        f"Ingestion complete: {totals['daily']} daily, "
        f"{totals['hourly']} hourly, {totals['minutes']} minute records"
    )


if __name__ == "__main__":
    main()
