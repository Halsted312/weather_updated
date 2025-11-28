#!/usr/bin/env python3
"""
Parallel ingestion of Visual Crossing historical forecasts.

Aggressive parallel processing with exponential backoff on errors.
Designed for unlimited VC API tokens with 24+ cores.

Usage:
    # Full backfill, 18 workers (6 cities × 3 years = 18 chunks)
    python scripts/ingest_vc_historical_forecast_parallel.py --workers 18

    # Maximum aggression (no rate limiting, 24 workers)
    python scripts/ingest_vc_historical_forecast_parallel.py --workers 24 --no-delay

    # Dry run to see chunks
    python scripts/ingest_vc_historical_forecast_parallel.py --dry-run
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import get_settings
from src.db import get_db_session, VcLocation, VcForecastDaily, VcForecastHourly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s",
)
logger = logging.getLogger(__name__)

# All 6 Kalshi weather cities
CITIES = ["CHI", "AUS", "DEN", "LAX", "MIA", "PHL"]

# Elements to request (comprehensive list)
ELEMENTS = [
    "datetime", "datetimeEpoch", "timezone", "tzoffset",
    "tempmax", "tempmin", "temp", "feelslikemax", "feelslikemin", "feelslike",
    "dew", "humidity", "precip", "precipprob", "preciptype", "precipcover",
    "snow", "snowdepth", "windspeed", "windgust", "winddir",
    "windspeedmean", "windspeedmin", "windspeedmax",
    "windspeed50", "winddir50", "windspeed80", "winddir80", "windspeed100", "winddir100",
    "cloudcover", "visibility", "pressure", "uvindex",
    "solarradiation", "solarenergy", "dniradiation", "difradiation", "ghiradiation", "gtiradiation",
    "sunelevation", "sunazimuth", "cape", "cin", "deltat", "degreedays", "accdegreedays",
    "conditions", "icon",
]


def _format_list(value: Any) -> Optional[str]:
    if isinstance(value, list):
        return ",".join(str(v) for v in value) if value else None
    elif isinstance(value, str):
        return value
    return None


def fetch_with_backoff(
    session: requests.Session,
    url: str,
    params: dict,
    max_retries: int = 5,
    base_delay: float = 0.1,
) -> Dict[str, Any]:
    """Fetch with exponential backoff on errors."""
    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=60)

            if response.status_code == 429:  # Rate limited
                delay = min(base_delay * (2 ** attempt), 2.0)  # Cap at 2s
                logger.warning(f"Rate limited, backing off {delay:.2f}s (attempt {attempt+1})")
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Request error: {e}, retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                raise

    raise Exception(f"Max retries exceeded for {url}")


def parse_daily_record(
    day_data: Dict[str, Any],
    vc_location_id: int,
    basis_date: date,
    basis_datetime_utc: datetime,
) -> Optional[Dict[str, Any]]:
    """Parse a daily forecast record."""
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


def parse_hourly_record(
    hour_data: Dict[str, Any],
    vc_location_id: int,
    basis_date: date,
    basis_datetime_utc: datetime,
    iana_timezone: str,
) -> Optional[Dict[str, Any]]:
    """Parse an hourly forecast record."""
    epoch = hour_data.get("datetimeEpoch")
    if not epoch:
        return None

    target_dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)
    tzoffset_hours = hour_data.get("tzoffset", 0) or 0
    tzoffset_minutes = int(tzoffset_hours * 60)
    target_dt_local = target_dt_utc + timedelta(minutes=tzoffset_minutes)
    target_dt_local_naive = target_dt_local.replace(tzinfo=None)
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


def process_chunk(
    api_key: str,
    base_url: str,
    location_id: int,
    location_query: str,
    location_type: str,
    city_code: str,
    iana_timezone: str,
    start_basis: date,
    end_basis: date,
    horizon_days: int,
    request_delay: float,
) -> Tuple[str, int, int, int]:
    """
    Process a chunk of basis dates for one location.

    Returns: (chunk_id, daily_count, hourly_count, errors)
    """
    chunk_id = f"{city_code}_{location_type}_{start_basis}_{end_basis}"

    http_session = requests.Session()
    http_session.headers.update({"Accept": "application/json"})

    daily_count = 0
    hourly_count = 0
    error_count = 0

    current_basis = start_basis

    while current_basis <= end_basis:
        basis_datetime_utc = datetime.combine(
            current_basis, datetime.min.time()
        ).replace(tzinfo=timezone.utc, hour=12)

        target_start = current_basis
        target_end = current_basis + timedelta(days=horizon_days - 1)

        # Build URL
        if location_type == "station":
            location_str = f"stn:{location_query.replace('stn:', '')}"
        else:
            location_str = location_query

        url = f"{base_url}/{location_str}/{target_start.isoformat()}/{target_end.isoformat()}"

        params = {
            "key": api_key,
            "unitGroup": "us",
            "include": "days,hours",
            "forecastBasisDate": current_basis.isoformat(),
            "elements": ",".join(ELEMENTS),
            "contentType": "json",
        }

        if location_type == "station":
            params.update({
                "maxStations": "1",
                "maxDistance": "1609",
                "elevationDifference": "50",
            })

        try:
            data = fetch_with_backoff(http_session, url, params)

            daily_records = []
            hourly_records = []

            for day in data.get("days", []):
                daily_rec = parse_daily_record(day, location_id, current_basis, basis_datetime_utc)
                if daily_rec:
                    daily_records.append(daily_rec)

                for hour in day.get("hours", []):
                    hourly_rec = parse_hourly_record(
                        hour, location_id, current_basis, basis_datetime_utc, iana_timezone
                    )
                    if hourly_rec:
                        hourly_records.append(hourly_rec)

            # Upsert to database
            if daily_records or hourly_records:
                with get_db_session() as db_session:
                    if daily_records:
                        stmt = insert(VcForecastDaily).values(daily_records)
                        update_cols = {
                            col.name: stmt.excluded[col.name]
                            for col in VcForecastDaily.__table__.columns
                            if col.name not in ("id", "vc_location_id", "target_date", "forecast_basis_date", "data_type", "created_at")
                        }
                        stmt = stmt.on_conflict_do_update(constraint="uq_vc_daily_row", set_=update_cols)
                        db_session.execute(stmt)
                        daily_count += len(daily_records)

                    if hourly_records:
                        stmt = insert(VcForecastHourly).values(hourly_records)
                        update_cols = {
                            col.name: stmt.excluded[col.name]
                            for col in VcForecastHourly.__table__.columns
                            if col.name not in ("id", "vc_location_id", "target_datetime_utc", "forecast_basis_date", "data_type", "created_at")
                        }
                        stmt = stmt.on_conflict_do_update(constraint="uq_vc_hourly_row", set_=update_cols)
                        db_session.execute(stmt)
                        hourly_count += len(hourly_records)

                    db_session.commit()

            if request_delay > 0:
                time.sleep(request_delay)

        except Exception as e:
            logger.error(f"{chunk_id} basis={current_basis}: {e}")
            error_count += 1

        current_basis += timedelta(days=1)

    return chunk_id, daily_count, hourly_count, error_count


def generate_chunks(
    locations: List[Tuple[int, str, str, str, str]],  # (id, query, type, city, tz)
    start_date: date,
    end_date: date,
    chunk_months: int = 6,
) -> List[Tuple]:
    """Generate (location_info, start, end) chunks."""
    chunks = []

    for loc_info in locations:
        current_start = start_date

        while current_start <= end_date:
            # Calculate chunk end
            next_month = current_start.month + chunk_months
            next_year = current_start.year + (next_month - 1) // 12
            next_month = (next_month - 1) % 12 + 1
            chunk_end = min(date(next_year, next_month, 1) - timedelta(days=1), end_date)

            chunks.append((loc_info, current_start, chunk_end))
            current_start = chunk_end + timedelta(days=1)

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Parallel ingestion of VC historical forecasts (aggressive)"
    )
    parser.add_argument("--start-date", type=str, default="2023-01-01")
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--horizon-days", type=int, default=4)
    parser.add_argument("--cities", type=str, nargs="+", default=CITIES)
    parser.add_argument("--location-type", type=str, choices=["station", "city"])
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--chunk-months", type=int, default=6)
    parser.add_argument("--request-delay", type=float, default=0.0,
                        help="Delay between requests (default: 0 = no delay)")
    parser.add_argument("--no-delay", action="store_true",
                        help="Explicitly set request delay to 0")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date) if args.end_date else date.today() - timedelta(days=1)

    # Adjust start for horizon
    adjusted_start = start_date - timedelta(days=args.horizon_days - 1)

    request_delay = 0.0 if args.no_delay else args.request_delay

    logger.info(f"Parallel historical forecast ingestion")
    logger.info(f"  Date range: {start_date} to {end_date} (basis: {adjusted_start})")
    logger.info(f"  Horizon: {args.horizon_days} days, Workers: {args.workers}")
    logger.info(f"  Request delay: {request_delay}s")

    # Get settings
    settings = get_settings()

    # Query locations
    with get_db_session() as session:
        query = select(VcLocation)
        if args.cities:
            query = query.where(VcLocation.city_code.in_([c.upper() for c in args.cities]))
        if args.location_type:
            query = query.where(VcLocation.location_type == args.location_type)

        locations = [
            (loc.id, loc.vc_location_query, loc.location_type, loc.city_code, loc.iana_timezone)
            for loc in session.execute(query).scalars().all()
        ]

    if not locations:
        logger.error("No locations found")
        return

    logger.info(f"Found {len(locations)} locations")

    # Generate chunks
    chunks = generate_chunks(locations, adjusted_start, end_date, args.chunk_months)
    logger.info(f"Generated {len(chunks)} chunks")

    if args.dry_run:
        for loc_info, chunk_start, chunk_end in chunks:
            days = (chunk_end - chunk_start).days + 1
            logger.info(f"  {loc_info[3]}/{loc_info[2]}: {chunk_start} to {chunk_end} ({days} days)")
        return

    # Process in parallel
    results = []
    total_daily = 0
    total_hourly = 0
    total_errors = 0

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="vc") as executor:
        futures = {
            executor.submit(
                process_chunk,
                settings.vc_api_key,
                settings.vc_base_url,
                loc_info[0],  # location_id
                loc_info[1],  # location_query
                loc_info[2],  # location_type
                loc_info[3],  # city_code
                loc_info[4],  # iana_timezone
                chunk_start,
                chunk_end,
                args.horizon_days,
                request_delay,
            ): (loc_info, chunk_start, chunk_end)
            for loc_info, chunk_start, chunk_end in chunks
        }

        for future in as_completed(futures):
            try:
                chunk_id, daily, hourly, errors = future.result()
                total_daily += daily
                total_hourly += hourly
                total_errors += errors

                status = "✓" if errors == 0 else f"⚠ ({errors} errors)"
                logger.info(f"{status} {chunk_id}: {daily} daily, {hourly} hourly")

            except Exception as e:
                chunk_info = futures[future]
                logger.error(f"✗ {chunk_info}: {e}")
                total_errors += 1

    elapsed = time.time() - start_time

    logger.info(f"\n{'='*60}")
    logger.info(f"Completed in {elapsed:.1f}s")
    logger.info(f"Total: {total_daily} daily, {total_hourly} hourly records")
    logger.info(f"Errors: {total_errors}")
    logger.info(f"Rate: {(total_daily + total_hourly) / elapsed:.1f} records/sec")


if __name__ == "__main__":
    main()
