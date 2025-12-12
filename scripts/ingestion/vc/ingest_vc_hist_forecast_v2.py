#!/usr/bin/env python3
"""
Ingest Visual Crossing historical forecasts using lat/lon + forecastBasisDay.

Supports TWO forecast sources per city:
  - station: lat/lon at settlement station (e.g., "30.18311,-97.67989")
  - city: city-aggregate query (e.g., "Austin,TX")

IMPORTANT: Do NOT use stn:KXXX queries for forecasts - they return observations!

Key features:
- Uses forecastBasisDay parameter for historical forecasts
- Validates source field is 'fcst' (not 'obs') before storing
- Parallel processing by month
- Supports 15-minute data with --include-minutes

Usage:
    # Station-anchored forecasts (default)
    python scripts/ingest_vc_hist_forecast_v2.py --city austin --start-date 2024-07-01

    # City-aggregate forecasts
    python scripts/ingest_vc_hist_forecast_v2.py --city austin --start-date 2024-07-01 --location-type city

    # With 15-minute data
    python scripts/ingest_vc_hist_forecast_v2.py --city austin --start-date 2024-07-01 --include-minutes

    # Parallel by month
    python scripts/ingest_vc_hist_forecast_v2.py --city austin --start-date 2023-01-01 --parallel
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import get_settings
from src.config.cities import CITIES, get_city
from src.db import get_db_session, VcLocation, VcForecastDaily, VcForecastHourly, VcMinuteWeather

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


def fetch_historical_forecast(
    session: requests.Session,
    api_key: str,
    latlon: str,
    target_date: str,
    lead_days: int,
    include_hours: bool = True,
    include_minutes: bool = False,
) -> Dict[str, Any]:
    """
    Fetch historical forecast for a single target date with specific lead.

    Uses forecastBasisDay parameter which returns the forecast that was
    made `lead_days` before the target date.

    Args:
        session: requests Session for connection pooling
        api_key: Visual Crossing API key
        latlon: Lat/lon string like "41.78412,-87.75514"
        target_date: Target date in YYYY-MM-DD format
        lead_days: How many days before target the forecast was made (0-14)
        include_hours: Whether to include hourly data
        include_minutes: Whether to include 15-minute data

    Returns:
        API response dict with days (and optionally hours, minutes) data
    """
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    url = f"{base_url}/{latlon}/{target_date}/{target_date}"

    # Build include string based on flags
    include_parts = ["days"]
    if include_hours:
        include_parts.append("hours")
    if include_minutes:
        include_parts.append("minutes")
    include = ",".join(include_parts)

    params = {
        "key": api_key,
        "unitGroup": "us",
        "include": include,
        "forecastBasisDay": lead_days,
        "contentType": "json",
    }

    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def parse_daily_forecast(
    day_data: Dict[str, Any],
    vc_location_id: int,
    target_date: date,
    lead_days: int,
) -> Optional[Dict[str, Any]]:
    """Parse daily forecast data into DB record format."""
    source = day_data.get("source", "")

    # Validate we got forecast data, not observations
    if source == "obs":
        logger.warning(f"Got observation data (source=obs) for {target_date} lead={lead_days}")
        return None

    # Calculate basis date from target and lead
    basis_date = target_date - timedelta(days=lead_days)
    basis_datetime_utc = datetime.combine(basis_date, datetime.min.time())

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
        "cloudcover": day_data.get("cloudcover"),
        "visibility_miles": day_data.get("visibility"),
        "pressure_mb": day_data.get("pressure"),
        "uvindex": day_data.get("uvindex"),
        "solarradiation": day_data.get("solarradiation"),
        "solarenergy": day_data.get("solarenergy"),
        "conditions": day_data.get("conditions"),
        "icon": day_data.get("icon"),
        "source_system": "vc_timeline_latlon",
        "raw_json": day_data,
    }


def parse_hourly_forecast(
    hour_data: Dict[str, Any],
    vc_location_id: int,
    target_date: date,
    lead_days: int,
    timezone_str: str,
) -> Optional[Dict[str, Any]]:
    """Parse hourly forecast data into DB record format."""
    import pytz

    epoch = hour_data.get("datetimeEpoch")
    if not epoch:
        return None

    target_dt_utc = datetime.utcfromtimestamp(epoch).replace(tzinfo=pytz.UTC)

    # Calculate local time
    try:
        tz = pytz.timezone(timezone_str)
        target_dt_local = target_dt_utc.astimezone(tz)
        tzoffset_minutes = int(target_dt_local.utcoffset().total_seconds() / 60)
    except Exception:
        target_dt_local = target_dt_utc
        tzoffset_minutes = 0

    # Calculate basis date from target and lead
    basis_date = target_date - timedelta(days=lead_days)
    basis_datetime_utc = datetime.combine(basis_date, datetime.min.time())

    # Calculate lead_hours from basis to target datetime
    lead_hours = int((target_dt_utc.replace(tzinfo=None) - basis_datetime_utc).total_seconds() / 3600)

    return {
        "vc_location_id": vc_location_id,
        "data_type": "historical_forecast",
        "forecast_basis_date": basis_date,
        "forecast_basis_datetime_utc": basis_datetime_utc,
        "target_datetime_utc": target_dt_utc.replace(tzinfo=None),
        "target_datetime_local": target_dt_local.replace(tzinfo=None),
        "target_datetime_epoch_utc": epoch,
        "lead_hours": lead_hours,
        "timezone": timezone_str,
        "tzoffset_minutes": tzoffset_minutes,
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
        "cloudcover": hour_data.get("cloudcover"),
        "visibility_miles": hour_data.get("visibility"),
        "pressure_mb": hour_data.get("pressure"),
        "uvindex": hour_data.get("uvindex"),
        "solarradiation": hour_data.get("solarradiation"),
        "solarenergy": hour_data.get("solarenergy"),
        "conditions": hour_data.get("conditions"),
        "icon": hour_data.get("icon"),
        "source_system": "vc_timeline_latlon",
        "raw_json": hour_data,
    }


def parse_minute_forecast(
    minute_data: Dict[str, Any],
    vc_location_id: int,
    basis_date: date,
    basis_datetime_utc: datetime,
    timezone_str: str,
) -> Optional[Dict[str, Any]]:
    """
    Parse minute-level forecast data into VcMinuteWeather record format.

    Args:
        minute_data: Minute node from VC API response
        vc_location_id: Foreign key to wx.vc_location
        basis_date: Date of forecast model run (e.g., target_date - 1 for T-1)
        basis_datetime_utc: Midnight UTC of basis date
        timezone_str: IANA timezone (e.g., 'America/Chicago')

    Returns:
        Dict ready for VcMinuteWeather upsert, or None if invalid
    """
    import pytz

    # Extract epoch - this is the MINUTE's timestamp, not the hour's!
    epoch = minute_data.get("datetimeEpoch")
    if not epoch:
        return None

    # Convert to timezone-aware UTC datetime
    minute_dt_utc = datetime.utcfromtimestamp(epoch).replace(tzinfo=pytz.UTC)

    # Calculate local time
    try:
        tz = pytz.timezone(timezone_str)
        minute_dt_local = minute_dt_utc.astimezone(tz)
        tzoffset_minutes = int(minute_dt_local.utcoffset().total_seconds() / 60)
    except Exception:
        minute_dt_local = minute_dt_utc
        tzoffset_minutes = 0

    # Calculate lead_hours from basis to this minute
    lead_hours = int((minute_dt_utc.replace(tzinfo=None) - basis_datetime_utc).total_seconds() / 3600)

    return {
        "vc_location_id": vc_location_id,
        "data_type": "historical_forecast",
        "forecast_basis_date": basis_date,
        "forecast_basis_datetime_utc": basis_datetime_utc,
        "lead_hours": lead_hours,
        "datetime_epoch_utc": epoch,
        "datetime_utc": minute_dt_utc.replace(tzinfo=None),
        "datetime_local": minute_dt_local.replace(tzinfo=None),
        "timezone": timezone_str,
        "tzoffset_minutes": tzoffset_minutes,
        # Required fields (always present in minute data)
        "temp_f": minute_data.get("temp"),
        "humidity": minute_data.get("humidity"),
        "dew_f": minute_data.get("dew"),
        # Common fields (usually present)
        "feelslike_f": minute_data.get("feelslike"),
        "pressure_mb": minute_data.get("pressure"),
        "windspeed_mph": minute_data.get("windspeed"),
        "winddir": minute_data.get("winddir"),
        "visibility_miles": minute_data.get("visibility"),
        "solarradiation": minute_data.get("solarradiation"),
        "solarenergy": minute_data.get("solarenergy"),
        # Optional fields (may be missing - graceful degradation)
        "precip_in": minute_data.get("precip"),
        "precipprob": minute_data.get("precipprob"),
        "preciptype": _format_list(minute_data.get("preciptype")),
        "windgust_mph": minute_data.get("windgust"),
        "cloudcover": minute_data.get("cloudcover"),
        "uvindex": minute_data.get("uvindex"),
        "snow_in": minute_data.get("snow"),
        "snowdepth_in": minute_data.get("snowdepth"),
        # Advanced fields (rare in minute data, but capture if present)
        "cape": minute_data.get("cape"),
        "cin": minute_data.get("cin"),
        # Metadata
        "conditions": minute_data.get("conditions"),
        "icon": minute_data.get("icon"),
        "source_system": "vc_timeline_latlon_minutes",
        "raw_json": minute_data,
        "is_forward_filled": False,  # This is raw forecast data, not forward-filled
    }


def process_month(
    city_id: str,
    year: int,
    month: int,
    lead_days_list: List[int],
    api_key: str,
    request_delay: float = 0.05,
    include_minutes: bool = False,
    location_type: str = "station",
) -> Tuple[int, int, int, int]:
    """
    Process one month of historical forecasts for a city.

    Args:
        city_id: City identifier (e.g., 'austin', 'chicago')
        year: Year to process
        month: Month to process
        lead_days_list: List of lead days (e.g., [0, 1, 2, 3])
        api_key: Visual Crossing API key
        request_delay: Delay between requests in seconds
        include_minutes: If True, fetch and store 15-minute forecast data
        location_type: 'station' (lat/lon at settlement) or 'city' (city-aggregate)

    Returns:
        Tuple of (daily_count, hourly_count, minute_count, error_count)
    """
    city_cfg = get_city(city_id)

    # Select query string based on location type
    if location_type == "station":
        query_string = city_cfg.vc_latlon_query  # e.g., "30.18311,-97.67989"
    else:
        query_string = city_cfg.vc_city_query  # e.g., "Austin,TX"

    # Calculate date range for this month
    start_date = date(year, month, 1)
    if month == 12:
        end_date = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(year, month + 1, 1) - timedelta(days=1)

    # Don't go past yesterday
    yesterday = date.today() - timedelta(days=1)
    if end_date > yesterday:
        end_date = yesterday

    if start_date > yesterday:
        logger.info(f"Skipping {year}-{month:02d} - future dates")
        return 0, 0, 0, 0

    logger.info(f"Processing {city_id} {year}-{month:02d} ({location_type}): {start_date} to {end_date}")

    daily_records = []
    hourly_records = []
    minute_records = []
    error_count = 0

    http_session = requests.Session()

    with get_db_session() as db_session:
        # Get vc_location_id for this city and location type
        loc_query = select(VcLocation).where(
            VcLocation.city_code == city_cfg.city_code,
            VcLocation.location_type == location_type,
        )
        location = db_session.execute(loc_query).scalar_one_or_none()

        if not location:
            logger.error(f"No VcLocation found for {city_cfg.city_code}/{location_type}")
            return 0, 0, 0, 1

        vc_location_id = location.id
        iana_timezone = location.iana_timezone or city_cfg.timezone

        current_date = start_date
        while current_date <= end_date:
            target_str = current_date.isoformat()

            for lead in lead_days_list:
                try:
                    data = fetch_historical_forecast(
                        session=http_session,
                        api_key=api_key,
                        latlon=query_string,  # lat/lon or city name
                        target_date=target_str,
                        lead_days=lead,
                        include_hours=True,
                        include_minutes=include_minutes,
                    )

                    # Parse daily
                    days = data.get("days", [])
                    if days:
                        day_data = days[0]
                        daily_rec = parse_daily_forecast(
                            day_data, vc_location_id, current_date, lead
                        )
                        if daily_rec:
                            daily_records.append(daily_rec)

                        # Parse hourly
                        for hour_data in day_data.get("hours", []):
                            hourly_rec = parse_hourly_forecast(
                                hour_data, vc_location_id, current_date, lead, iana_timezone
                            )
                            if hourly_rec:
                                hourly_records.append(hourly_rec)

                        # Parse minutes (if requested)
                        if include_minutes:
                            # Calculate basis date/time for this lead (constant across all minutes)
                            basis_date = current_date - timedelta(days=lead)
                            basis_datetime_utc = datetime.combine(basis_date, datetime.min.time())

                            for hour_data in day_data.get("hours", []):
                                for minute_data in hour_data.get("minutes", []):
                                    minute_rec = parse_minute_forecast(
                                        minute_data,
                                        vc_location_id,
                                        basis_date,
                                        basis_datetime_utc,
                                        iana_timezone,
                                    )
                                    if minute_rec:
                                        minute_records.append(minute_rec)

                    time.sleep(request_delay)

                except requests.exceptions.RequestException as e:
                    logger.warning(f"Error fetching {target_str} lead={lead}: {e}")
                    error_count += 1
                    time.sleep(0.5)  # Back off on errors

            current_date += timedelta(days=1)

        # Bulk upsert daily records
        if daily_records:
            stmt = insert(VcForecastDaily).values(daily_records)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_vc_daily_row",
                set_={col: stmt.excluded[col] for col in [
                    "tempmax_f", "tempmin_f", "temp_f",
                    "feelslikemax_f", "feelslikemin_f", "humidity",
                    "precip_in", "precipprob", "windspeed_mph",
                    "conditions", "icon", "source_system", "raw_json",
                ]},
            )
            db_session.execute(stmt)

        # Bulk upsert hourly records
        if hourly_records:
            stmt = insert(VcForecastHourly).values(hourly_records)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_vc_hourly_row",
                set_={col: stmt.excluded[col] for col in [
                    "temp_f", "feelslike_f", "humidity",
                    "precip_in", "precipprob", "windspeed_mph",
                    "conditions", "icon", "source_system", "raw_json",
                ]},
            )
            db_session.execute(stmt)

        # Bulk upsert minute records (if any)
        if minute_records:
            stmt = insert(VcMinuteWeather).values(minute_records)
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    VcMinuteWeather.vc_location_id,
                    VcMinuteWeather.data_type,
                    VcMinuteWeather.forecast_basis_date,
                    VcMinuteWeather.datetime_utc,
                ],
                index_where=text("forecast_basis_date IS NOT NULL"),
                set_={
                    "temp_f": stmt.excluded.temp_f,
                    "humidity": stmt.excluded.humidity,
                    "dew_f": stmt.excluded.dew_f,
                    "feelslike_f": stmt.excluded.feelslike_f,
                    "pressure_mb": stmt.excluded.pressure_mb,
                    "windspeed_mph": stmt.excluded.windspeed_mph,
                    "winddir": stmt.excluded.winddir,
                    "windgust_mph": stmt.excluded.windgust_mph,
                    "precip_in": stmt.excluded.precip_in,
                    "precipprob": stmt.excluded.precipprob,
                    "cloudcover": stmt.excluded.cloudcover,
                    "uvindex": stmt.excluded.uvindex,
                    "solarradiation": stmt.excluded.solarradiation,
                    "solarenergy": stmt.excluded.solarenergy,
                    "visibility_miles": stmt.excluded.visibility_miles,
                    "source_system": stmt.excluded.source_system,
                    "raw_json": stmt.excluded.raw_json,
                },
            )
            db_session.execute(stmt)

        db_session.commit()

    logger.info(
        f"Completed {city_id} {year}-{month:02d}: "
        f"{len(daily_records)} daily, {len(hourly_records)} hourly, {len(minute_records)} minute"
    )
    return len(daily_records), len(hourly_records), len(minute_records), error_count


def main():
    parser = argparse.ArgumentParser(description="Ingest VC historical forecasts")
    parser.add_argument("--city", required=True, help="City ID (e.g., chicago, austin)")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD), defaults to yesterday")
    parser.add_argument("--lead-days", default="0,1,2,3", help="Comma-separated lead days (default: 0,1,2,3)")
    parser.add_argument(
        "--location-type",
        choices=["station", "city"],
        default="station",
        help="Location type: 'station' (lat/lon at settlement) or 'city' (city-aggregate). Default: station"
    )
    parser.add_argument("--parallel", action="store_true", help="Process months in parallel")
    parser.add_argument("--workers", type=int, default=12, help="Number of parallel workers")
    parser.add_argument("--request-delay", type=float, default=0.05, help="Delay between requests (seconds)")
    parser.add_argument("--include-minutes", action="store_true", help="Include 15-minute forecast data (default: False)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done")

    args = parser.parse_args()

    # Validate city
    if args.city not in CITIES:
        logger.error(f"Unknown city: {args.city}. Available: {list(CITIES.keys())}")
        return

    # Parse dates
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date) if args.end_date else date.today() - timedelta(days=1)

    # Parse lead days
    lead_days_list = [int(x.strip()) for x in args.lead_days.split(",")]

    # Get API key
    settings = get_settings()
    api_key = settings.vc_api_key

    city_cfg = get_city(args.city)
    location_type = args.location_type

    # Select query string based on location type
    if location_type == "station":
        query_string = city_cfg.vc_latlon_query
    else:
        query_string = city_cfg.vc_city_query

    logger.info(f"City: {args.city} ({city_cfg.city_code})")
    logger.info(f"Location type: {location_type}")
    logger.info(f"Query string: {query_string}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Lead days: {lead_days_list}")

    if args.dry_run:
        logger.info("DRY RUN - would process the above")
        return

    # Generate list of (year, month) tuples
    months = []
    current = start_date.replace(day=1)
    while current <= end_date:
        months.append((current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    logger.info(f"Processing {len(months)} months")

    total_daily = 0
    total_hourly = 0
    total_minute = 0
    total_errors = 0

    if args.parallel and len(months) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_month,
                    args.city,
                    year,
                    month,
                    lead_days_list,
                    api_key,
                    args.request_delay,
                    args.include_minutes,
                    location_type,
                ): (year, month)
                for year, month in months
            }

            for future in as_completed(futures):
                year, month = futures[future]
                try:
                    daily, hourly, minute, errors = future.result()
                    total_daily += daily
                    total_hourly += hourly
                    total_minute += minute
                    total_errors += errors
                except Exception as e:
                    logger.error(f"Error processing {year}-{month:02d}: {e}")
                    total_errors += 1
    else:
        for year, month in months:
            daily, hourly, minute, errors = process_month(
                args.city,
                year,
                month,
                lead_days_list,
                api_key,
                args.request_delay,
                args.include_minutes,
                location_type,
            )
            total_daily += daily
            total_hourly += hourly
            total_minute += minute
            total_errors += errors

    logger.info(f"COMPLETE: {total_daily} daily, {total_hourly} hourly, {total_minute} minute, {total_errors} errors")


if __name__ == "__main__":
    main()
