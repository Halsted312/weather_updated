#!/usr/bin/env python3
# scripts/backfill_vc_historical_forecast_minutes_austin.py

"""
Backfill Visual Crossing 15-minute historical forecast minutes for Austin.

CRITICAL FIX: VC returns minutes[].datetime as time-only strings like "00:15:00",
NOT full ISO datetimes. This script combines day date + minute time to avoid
the 1900-01-01 bug.

Usage:
    python scripts/backfill_vc_historical_forecast_minutes_austin.py
"""

import os
import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any, Iterable, Optional

import requests
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.db.connection import get_db_session
from src.db.models import VcMinuteWeather
from src.config.cities import get_city
from models.data.loader import get_vc_location_id

# Load environment variables from .env file
load_dotenv()

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Set up logging to both file and console
log_file = LOGS_DIR / f"vc_backfill_austin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")

VC_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
VC_API_KEY = os.environ.get("VC_API_KEY")  # From .env file


@dataclass
class MinuteForecastRow:
    """Row for wx.vc_minute_weather table."""
    vc_location_id: int
    data_type: str
    forecast_basis_date: date
    datetime_epoch_utc: int  # Unix timestamp
    datetime_local: datetime
    datetime_utc: datetime
    timezone: str  # IANA timezone (e.g., 'America/Chicago')
    tzoffset_minutes: int  # UTC offset in minutes
    temp_f: Optional[float]
    humidity: Optional[float]
    cloudcover: Optional[float]
    windspeed_mph: Optional[float]
    windgust_mph: Optional[float]
    dew_f: Optional[float]
    pressure_mb: Optional[float]
    visibility_miles: Optional[float]
    conditions: Optional[str]
    raw_json: Dict[str, Any]


def fetch_historical_forecast_minutes(
    city_query: str,
    target_date: date,
    basis_date: date,
    elements: str = "datetime,datetimeEpoch,temp,humidity,dew,cloudcover,windspeed,windgust,pressure,visibility,conditions",
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> Dict[str, Any]:
    """
    Call VC Timeline API with forecastBasisDate for 15-min historical forecast.

    Includes exponential backoff for rate limiting and network errors.

    Args:
        city_query: VC location string (e.g., "Austin,TX")
        target_date: The day being forecast
        basis_date: When forecast was issued (usually target_date - 1)
        elements: Comma-separated list of weather elements to fetch
        max_retries: Maximum retry attempts (default: 5)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)

    Returns:
        API response dict with days/hours/minutes data

    Raises:
        RuntimeError: If API key not set
        requests.RequestException: If all retries exhausted
    """
    if not VC_API_KEY:
        raise RuntimeError("VC_API_KEY environment variable not set")

    url = f"{VC_BASE_URL}/{city_query}/{target_date}/{target_date}"
    params = {
        "key": VC_API_KEY,
        "unitGroup": "us",
        "include": "fcst,minutes",
        "forecastBasisDate": basis_date.isoformat(),
        "options": "minuteinterval_15,stnslevel1,usefcst",
        "elements": elements,
        "contentType": "json",
    }

    logger.debug(f"Fetching: {target_date} (basis={basis_date})")

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limited - exponential backoff
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Rate limited (429) on attempt {attempt+1}/{max_retries}, retrying in {delay:.1f}s...")
                time.sleep(delay)
            elif 500 <= e.response.status_code < 600:
                # Server error - retry with backoff
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Server error ({e.response.status_code}) on attempt {attempt+1}/{max_retries}, retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                # Client error (4xx) - don't retry
                logger.error(f"Client error ({e.response.status_code}): {e}")
                raise

        except requests.exceptions.RequestException as e:
            # Network error - retry with backoff
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Network error on attempt {attempt+1}/{max_retries}, retrying in {delay:.1f}s: {e}")
            time.sleep(delay)

    # All retries exhausted
    raise requests.exceptions.RetryError(f"Max retries ({max_retries}) exceeded for {target_date}")


def iter_minute_forecast_rows(
    vc_location_id: int,
    basis_date: date,
    payload: Dict[str, Any],
    tz_name: str,
) -> Iterable[MinuteForecastRow]:
    """
    Flatten VC payload into MinuteForecastRow entries.

    CRITICAL: VC returns minutes[].datetime as time-only strings like "00:15:00",
    NOT full ISO datetimes. Must combine with day date to avoid 1900-01-01 bug.

    Args:
        vc_location_id: FK to wx.vc_location
        basis_date: When forecast was issued
        payload: VC API response JSON
        tz_name: IANA timezone (e.g., "America/Chicago")

    Yields:
        MinuteForecastRow objects ready for database insertion
    """
    tz = ZoneInfo(tz_name)
    days = payload.get("days", [])

    if not days:
        logger.warning("No days[] in historical forecast payload")
        return

    for day in days:
        # Get the day's date (e.g., "2025-11-20")
        day_date_str = day.get("datetime")
        if not day_date_str:
            logger.warning("Missing datetime in day record")
            continue

        hours = day.get("hours") or []
        for hour in hours:
            minutes = hour.get("minutes") or []
            if not minutes:
                continue

            for minute in minutes:
                # CRITICAL FIX: VC minute datetime is often just "HH:MM:SS", not full ISO
                # Must combine with day_date_str to get correct datetime

                # Option 1: Use datetimeEpoch if available (most reliable)
                epoch = minute.get("datetimeEpoch")
                if epoch:
                    dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)
                    dt_local = dt_utc.astimezone(tz)
                else:
                    # Option 2: Combine day date + minute time string
                    minute_time_str = minute.get("datetime")
                    if not minute_time_str:
                        continue

                    # Build full local datetime string
                    if "T" in minute_time_str:
                        local_str = minute_time_str  # Already full ISO
                    else:
                        # Combine day + time: "2025-11-20" + "00:15:00" → "2025-11-20T00:15:00"
                        local_str = f"{day_date_str}T{minute_time_str}"

                    dt = datetime.fromisoformat(local_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=tz)

                    dt_utc = dt.astimezone(timezone.utc)
                    dt_local = dt.astimezone(tz)

                # Calculate epoch timestamp
                epoch_val = epoch if epoch else int(dt_utc.timestamp())

                # Calculate UTC offset in minutes
                offset_seconds = dt_local.utcoffset().total_seconds()
                tzoffset_min = int(offset_seconds / 60)

                yield MinuteForecastRow(
                    vc_location_id=vc_location_id,
                    data_type="historical_forecast",
                    forecast_basis_date=basis_date,
                    datetime_epoch_utc=epoch_val,
                    datetime_local=dt_local.replace(tzinfo=None),  # Store as naive
                    datetime_utc=dt_utc,
                    timezone=tz_name,
                    tzoffset_minutes=tzoffset_min,
                    temp_f=minute.get("temp"),
                    humidity=minute.get("humidity"),
                    cloudcover=minute.get("cloudcover"),
                    windspeed_mph=minute.get("windspeed"),
                    windgust_mph=minute.get("windgust"),
                    dew_f=minute.get("dew"),
                    pressure_mb=minute.get("pressure"),
                    visibility_miles=minute.get("visibility"),
                    conditions=minute.get("conditions"),
                    raw_json=minute,
                )


def upsert_vc_minute_weather(session: Session, rows: Iterable[MinuteForecastRow]) -> int:
    """
    Insert or update rows in wx.vc_minute_weather table.

    Uses ON CONFLICT with partial unique index `uq_vc_minute_fcst`
    on (vc_location_id, data_type, forecast_basis_date, datetime_utc)
    WHERE forecast_basis_date IS NOT NULL.

    Args:
        session: Database session
        rows: Iterable of MinuteForecastRow objects

    Returns:
        Number of rows affected
    """
    rows = list(rows)
    if not rows:
        return 0

    records = [
        {
            "vc_location_id": r.vc_location_id,
            "data_type": r.data_type,
            "forecast_basis_date": r.forecast_basis_date,
            "datetime_epoch_utc": r.datetime_epoch_utc,
            "datetime_local": r.datetime_local,
            "datetime_utc": r.datetime_utc,
            "timezone": r.timezone,
            "tzoffset_minutes": r.tzoffset_minutes,
            "temp_f": r.temp_f,
            "humidity": r.humidity,
            "cloudcover": r.cloudcover,
            "windspeed_mph": r.windspeed_mph,
            "windgust_mph": r.windgust_mph,
            "dew_f": r.dew_f,
            "pressure_mb": r.pressure_mb,
            "visibility_miles": r.visibility_miles,
            "conditions": r.conditions,
            "raw_json": r.raw_json,  # SQLAlchemy handles JSON conversion
        }
        for r in rows
    ]

    stmt = insert(VcMinuteWeather).values(records)

    # On conflict, update all weather fields (exclude keys and metadata)
    update_cols = {
        col.name: stmt.excluded[col.name]
        for col in VcMinuteWeather.__table__.columns
        if col.name not in ("id", "vc_location_id", "data_type", "forecast_basis_date",
                           "datetime_utc", "created_at")
    }

    # Use partial index for forecasts (WHERE forecast_basis_date IS NOT NULL)
    stmt = stmt.on_conflict_do_update(
        index_elements=["vc_location_id", "data_type", "forecast_basis_date", "datetime_utc"],
        index_where=VcMinuteWeather.forecast_basis_date.isnot(None),
        set_=update_cols,
    )

    result = session.execute(stmt)
    return result.rowcount


def backfill_city_historical_forecast_minutes(
    session: Session,
    city_id: str,
    start_event_date: date,
    end_event_date: date,
):
    """
    Backfill 15-minute historical forecasts for any city.

    Args:
        session: Database session
        city_id: City identifier ('austin', 'chicago', 'denver', etc.)
        start_event_date: First event date to backfill (inclusive)
        end_event_date: Last event date to backfill (inclusive)
    """
    city_cfg = get_city(city_id)
    tz_name = city_cfg.timezone  # IANA timezone like 'America/Chicago'
    city_query = city_cfg.vc_city_query  # City query like 'Austin,TX'

    logger.info("="*80)
    logger.info(f"VC 15-MIN FORECAST BACKFILL - {city_id.upper()}")
    logger.info("="*80)
    logger.info(f"Date range: {start_event_date} to {end_event_date}")
    logger.info(f"Timezone: {tz_name}")
    logger.info(f"City query: {city_query}")

    vc_location_id = get_vc_location_id(session, city_id, "city")
    if vc_location_id is None:
        raise RuntimeError(f"No vc_location_id for {city_id} city-level location")

    logger.info(f"Using vc_location_id: {vc_location_id}")

    current = start_event_date
    total_days = (end_event_date - start_event_date).days + 1
    processed = 0
    total_rows = 0
    failed_dates = []
    start_time = time.time()

    logger.info(f"Processing {total_days} days...")
    logger.info("")

    while current <= end_event_date:
        basis_date = current - timedelta(days=1)
        processed += 1

        # Progress indicator every 10 days
        if processed % 10 == 0 or processed == 1:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta_seconds = (total_days - processed) / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            logger.info(f"Progress: {processed}/{total_days} ({100*processed/total_days:.1f}%) | Rate: {rate:.1f} days/sec | ETA: {eta_minutes:.1f} min")

        try:
            payload = fetch_historical_forecast_minutes(
                city_query=city_query,
                target_date=current,
                basis_date=basis_date,
            )
        except Exception as e:
            logger.error(f"[{processed}/{total_days}] FAILED to fetch {current} (basis={basis_date}): {e}")
            failed_dates.append((current, str(e)))
            current += timedelta(days=1)
            continue

        rows = list(iter_minute_forecast_rows(
            vc_location_id=vc_location_id,
            basis_date=basis_date,
            payload=payload,
            tz_name=tz_name,
        ))

        num_rows = len(rows)

        # Sanity check: warn if we see 1900 dates (indicates parsing bug)
        if rows and rows[0].datetime_local.year == 1900:
            logger.error(f"⚠️  DATETIME PARSING BUG DETECTED: Got year 1900 for {current}")
            logger.error(f"First row datetime_local: {rows[0].datetime_local}")
            logger.error("Skipping this date to avoid bad data")
            failed_dates.append((current, "1900 datetime bug"))
            current += timedelta(days=1)
            continue

        affected = upsert_vc_minute_weather(session, rows)
        total_rows += affected

        logger.info(f"✓ [{processed}/{total_days}] {current} → {num_rows} rows inserted")

        current += timedelta(days=1)

    # Summary
    elapsed_total = time.time() - start_time
    logger.info("")
    logger.info("="*80)
    logger.info("BACKFILL COMPLETE")
    logger.info("="*80)
    logger.info(f"Processed: {processed} days in {elapsed_total/60:.1f} minutes")
    logger.info(f"Total rows: {total_rows}")
    logger.info(f"Success rate: {100*(processed-len(failed_dates))/processed:.1f}%")

    if failed_dates:
        logger.warning(f"Failed dates ({len(failed_dates)}):")
        for failed_date, error in failed_dates:
            logger.warning(f"  - {failed_date}: {error}")
    else:
        logger.info("All dates processed successfully!")

    logger.info(f"Log file: {log_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Backfill VC 15-min historical forecasts")
    parser.add_argument("--city", default="chicago", help="City ID (default: chicago)")
    parser.add_argument("--start-date", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                       default=date(2022, 12, 23), help="Start date (default: 2022-12-23)")
    parser.add_argument("--end-date", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                       default=date(2025, 12, 5), help="End date (default: 2025-12-05)")
    args = parser.parse_args()

    logger.info(f"Starting full historical backfill for {args.city}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Total days: {(args.end_date - args.start_date).days + 1}")

    with get_db_session() as session:
        backfill_city_historical_forecast_minutes(session, args.city, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
