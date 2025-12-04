#!/usr/bin/env python3
"""
Comprehensive backfill of Visual Crossing historical forecasts with 15-minute data.

Features:
- Backfills both station (lat/lon) and city (city-name) forecasts
- Skips dates that already have data in the database
- Adaptive rate limiting: starts at 0.007s, backs off to 0.015s on rate limit,
  settles at 0.01s after recovery
- Sequential processing to avoid hitting API rate limits
- Supports lead_days 0, 1, 2, 3 (configurable)
- Includes 15-minute resolution forecast data

IMPORTANT: Station forecasts use lat/lon queries (NOT stn:KXXX which returns obs!)

Usage:
    # Backfill all cities, both location types
    python scripts/backfill_vc_historical_forecasts.py

    # Backfill single city
    python scripts/backfill_vc_historical_forecasts.py --city austin

    # Backfill only station forecasts
    python scripts/backfill_vc_historical_forecasts.py --location-type station

    # Dry run (show what would be done)
    python scripts/backfill_vc_historical_forecasts.py --dry-run
"""

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from sqlalchemy import select, and_, func
from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import get_settings
from src.config.cities import CITIES, get_city
from src.db import (
    get_db_session,
    VcLocation,
    VcForecastDaily,
    VcForecastHourly,
    VcMinuteWeather,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/backfill_vc_forecasts.log"),
    ],
)
logger = logging.getLogger(__name__)


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts delay based on API responses.

    Starts at initial_delay (0.007s), backs off to max_delay (0.015s) on rate limit,
    then gradually settles to target_delay (0.01s) after recovery.
    """

    def __init__(
        self,
        initial_delay: float = 0.002,
        max_delay: float = 0.015,
        target_delay: float = 0.002,
        recovery_steps: int = 10,
    ):
        self.current_delay = initial_delay
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.target_delay = target_delay
        self.recovery_steps = recovery_steps
        self.steps_since_rate_limit = 0
        self.rate_limit_count = 0

    def wait(self):
        """Wait for the current delay period."""
        time.sleep(self.current_delay)

    def success(self):
        """Called after a successful request."""
        self.steps_since_rate_limit += 1

        # Gradually move toward target delay after recovery period
        if self.current_delay > self.target_delay and self.steps_since_rate_limit > self.recovery_steps:
            # Move 10% toward target
            self.current_delay = self.current_delay * 0.95 + self.target_delay * 0.05
            if self.current_delay < self.target_delay:
                self.current_delay = self.target_delay

    def rate_limited(self, wait_seconds: float = 60):
        """Called when rate limited - back off and wait."""
        self.rate_limit_count += 1
        self.steps_since_rate_limit = 0
        self.current_delay = self.max_delay

        logger.warning(
            f"Rate limited (count={self.rate_limit_count}), "
            f"waiting {wait_seconds}s, new delay={self.current_delay:.4f}s"
        )
        time.sleep(wait_seconds)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "current_delay": self.current_delay,
            "rate_limit_count": self.rate_limit_count,
            "steps_since_rate_limit": self.steps_since_rate_limit,
        }


def _format_list(value: Any) -> Optional[str]:
    """Format list as comma-separated string."""
    if isinstance(value, list):
        return ",".join(str(v) for v in value) if value else None
    elif isinstance(value, str):
        return value
    return None


def get_existing_dates(
    session,
    vc_location_id: int,
    lead_days: List[int],
    start_date: date,
    end_date: date,
) -> Dict[int, Set[date]]:
    """
    Get dates that already have minute-level historical forecast data.

    Returns:
        Dict mapping lead_day -> set of target dates that have data
    """
    from sqlalchemy import cast, Date

    existing = {lead: set() for lead in lead_days}

    # Query VcMinuteWeather for existing historical_forecast data
    # We consider a date "done" if it has any minute data for that lead
    for lead in lead_days:
        # Use cast to Date and distinct properly
        date_col = cast(VcMinuteWeather.datetime_utc, Date)
        result = session.execute(
            select(date_col)
            .where(
                and_(
                    VcMinuteWeather.vc_location_id == vc_location_id,
                    VcMinuteWeather.data_type == "historical_forecast",
                    VcMinuteWeather.lead_hours >= lead * 24,
                    VcMinuteWeather.lead_hours < (lead + 1) * 24,
                    date_col >= start_date,
                    date_col <= end_date,
                )
            )
            .distinct()
        ).scalars().all()

        existing[lead] = set(result)

    return existing


def fetch_historical_forecast(
    http_session: requests.Session,
    api_key: str,
    query_string: str,
    target_date: str,
    lead_days: int,
    rate_limiter: AdaptiveRateLimiter,
) -> Optional[Dict[str, Any]]:
    """
    Fetch historical forecast with 15-minute data.

    Uses forecastBasisDay parameter to get the forecast that was made
    `lead_days` before the target date.

    Returns:
        API response dict, or None on error
    """
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    url = f"{base_url}/{query_string}/{target_date}/{target_date}"

    params = {
        "key": api_key,
        "unitGroup": "us",
        "include": "days,hours,minutes",  # forecastBasisDay ensures forecast data
        "forecastBasisDay": lead_days,
        "contentType": "json",
    }

    try:
        rate_limiter.wait()
        response = http_session.get(url, params=params, timeout=60)

        if response.status_code == 429:
            # Rate limited - back off
            rate_limiter.rate_limited(wait_seconds=60)
            # Retry once
            response = http_session.get(url, params=params, timeout=60)

        response.raise_for_status()
        rate_limiter.success()

        data = response.json()

        # Validate we got forecast data
        if data.get("days"):
            source = data["days"][0].get("source", "unknown")
            if source == "obs":
                logger.warning(f"Got observations instead of forecast for {query_string} {target_date} T-{lead_days}")
                return None

        return data

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            rate_limiter.rate_limited(wait_seconds=120)
            return None
        logger.error(f"HTTP error fetching {query_string} {target_date} T-{lead_days}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching {query_string} {target_date} T-{lead_days}: {e}")
        time.sleep(1)  # Brief pause on error
        return None


def parse_minute_forecast(
    minute_data: Dict[str, Any],
    vc_location_id: int,
    basis_date: date,
    basis_datetime_utc: datetime,
    timezone_str: str,
) -> Optional[Dict[str, Any]]:
    """Parse minute-level forecast data into VcMinuteWeather record format."""
    import pytz

    epoch = minute_data.get("datetimeEpoch")
    if not epoch:
        return None

    minute_dt_utc = datetime.fromtimestamp(epoch, tz=pytz.UTC)

    try:
        tz = pytz.timezone(timezone_str)
        minute_dt_local = minute_dt_utc.astimezone(tz)
        tzoffset_minutes = int(minute_dt_local.utcoffset().total_seconds() / 60)
    except Exception:
        minute_dt_local = minute_dt_utc
        tzoffset_minutes = 0

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
        "temp_f": minute_data.get("temp"),
        "humidity": minute_data.get("humidity"),
        "dew_f": minute_data.get("dew"),
        "feelslike_f": minute_data.get("feelslike"),
        "pressure_mb": minute_data.get("pressure"),
        "windspeed_mph": minute_data.get("windspeed"),
        "winddir": minute_data.get("winddir"),
        "visibility_miles": minute_data.get("visibility"),
        "solarradiation": minute_data.get("solarradiation"),
        "solarenergy": minute_data.get("solarenergy"),
        "precip_in": minute_data.get("precip"),
        "precipprob": minute_data.get("precipprob"),
        "preciptype": _format_list(minute_data.get("preciptype")),
        "windgust_mph": minute_data.get("windgust"),
        "cloudcover": minute_data.get("cloudcover"),
        "uvindex": minute_data.get("uvindex"),
        "snow_in": minute_data.get("snow"),
        "snowdepth_in": minute_data.get("snowdepth"),
        "cape": minute_data.get("cape"),
        "cin": minute_data.get("cin"),
        "conditions": minute_data.get("conditions"),
        "icon": minute_data.get("icon"),
        "source_system": "vc_timeline_latlon_minutes",
        "raw_json": minute_data,
        "is_forward_filled": False,
    }


def parse_hourly_forecast(
    hour_data: Dict[str, Any],
    vc_location_id: int,
    target_date: date,
    lead_days: int,
    timezone_str: str,
) -> Optional[Dict[str, Any]]:
    """Parse hourly forecast data into VcForecastHourly record format."""
    import pytz

    epoch = hour_data.get("datetimeEpoch")
    if not epoch:
        return None

    target_dt_utc = datetime.fromtimestamp(epoch, tz=pytz.UTC)

    try:
        tz = pytz.timezone(timezone_str)
        target_dt_local = target_dt_utc.astimezone(tz)
        tzoffset_minutes = int(target_dt_local.utcoffset().total_seconds() / 60)
    except Exception:
        target_dt_local = target_dt_utc
        tzoffset_minutes = 0

    basis_date = target_date - timedelta(days=lead_days)
    basis_datetime_utc = datetime.combine(basis_date, datetime.min.time())
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


def parse_daily_forecast(
    day_data: Dict[str, Any],
    vc_location_id: int,
    target_date: date,
    lead_days: int,
) -> Optional[Dict[str, Any]]:
    """Parse daily forecast data into VcForecastDaily record format."""
    source = day_data.get("source", "")

    if source == "obs":
        return None

    basis_date = target_date - timedelta(days=lead_days)
    basis_datetime_utc = datetime.combine(basis_date, datetime.min.time())

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


def backfill_location(
    city_id: str,
    location_type: str,
    start_date: date,
    end_date: date,
    lead_days_list: List[int],
    api_key: str,
    rate_limiter: AdaptiveRateLimiter,
    dry_run: bool = False,
) -> Tuple[int, int, int, int]:
    """
    Backfill historical forecasts for a single location.

    Returns:
        Tuple of (requests_made, daily_inserted, hourly_inserted, minute_inserted)
    """
    city_cfg = get_city(city_id)

    # Select query string based on location type
    if location_type == "station":
        query_string = city_cfg.vc_latlon_query  # e.g., "30.18311,-97.67989"
    else:
        query_string = city_cfg.vc_city_query  # e.g., "Austin,TX"

    logger.info(f"Backfilling {city_id} ({location_type}): {query_string}")
    logger.info(f"  Date range: {start_date} to {end_date}")
    logger.info(f"  Lead days: {lead_days_list}")

    requests_made = 0
    daily_inserted = 0
    hourly_inserted = 0
    minute_inserted = 0

    http_session = requests.Session()

    with get_db_session() as db_session:
        # Get VcLocation ID
        loc_query = select(VcLocation).where(
            VcLocation.city_code == city_cfg.city_code,
            VcLocation.location_type == location_type,
        )
        location = db_session.execute(loc_query).scalar_one_or_none()

        if not location:
            logger.error(f"No VcLocation found for {city_cfg.city_code}/{location_type}")
            return 0, 0, 0, 0

        vc_location_id = location.id
        iana_timezone = location.iana_timezone or city_cfg.timezone

        # Get existing dates to skip
        existing_dates = get_existing_dates(
            db_session, vc_location_id, lead_days_list, start_date, end_date
        )

        total_existing = sum(len(dates) for dates in existing_dates.values())
        logger.info(f"  Found {total_existing} existing date/lead combinations to skip")

        # Calculate total work
        total_dates = (end_date - start_date).days + 1
        total_requests = total_dates * len(lead_days_list) - total_existing
        logger.info(f"  Total requests to make: ~{total_requests}")

        if dry_run:
            logger.info("  DRY RUN - would process above")
            return 0, 0, 0, 0

        # Process each date
        current_date = start_date
        batch_daily = []
        batch_hourly = []
        batch_minute = []
        batch_size = 50  # Commit every 50 requests

        while current_date <= end_date:
            for lead in lead_days_list:
                # Skip if we already have data
                if current_date in existing_dates.get(lead, set()):
                    continue

                target_str = current_date.isoformat()

                # Fetch from API
                data = fetch_historical_forecast(
                    http_session,
                    api_key,
                    query_string,
                    target_str,
                    lead,
                    rate_limiter,
                )
                requests_made += 1

                if not data:
                    continue

                days = data.get("days", [])
                if not days:
                    continue

                day_data = days[0]

                # Parse daily
                daily_rec = parse_daily_forecast(day_data, vc_location_id, current_date, lead)
                if daily_rec:
                    batch_daily.append(daily_rec)

                # Parse hourly
                for hour_data in day_data.get("hours", []):
                    hourly_rec = parse_hourly_forecast(
                        hour_data, vc_location_id, current_date, lead, iana_timezone
                    )
                    if hourly_rec:
                        batch_hourly.append(hourly_rec)

                # Parse minutes
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
                            batch_minute.append(minute_rec)

                # Log progress
                if requests_made % 100 == 0:
                    stats = rate_limiter.get_stats()
                    logger.info(
                        f"  Progress: {requests_made} requests, "
                        f"delay={stats['current_delay']:.4f}s, "
                        f"rate_limits={stats['rate_limit_count']}"
                    )

                # Batch commit
                if len(batch_minute) >= batch_size * 96:  # ~96 minutes per day
                    _commit_batch(db_session, batch_daily, batch_hourly, batch_minute)
                    daily_inserted += len(batch_daily)
                    hourly_inserted += len(batch_hourly)
                    minute_inserted += len(batch_minute)
                    batch_daily = []
                    batch_hourly = []
                    batch_minute = []

            current_date += timedelta(days=1)

        # Final commit
        if batch_daily or batch_hourly or batch_minute:
            _commit_batch(db_session, batch_daily, batch_hourly, batch_minute)
            daily_inserted += len(batch_daily)
            hourly_inserted += len(batch_hourly)
            minute_inserted += len(batch_minute)

    logger.info(
        f"  Completed: {requests_made} requests, "
        f"{daily_inserted} daily, {hourly_inserted} hourly, {minute_inserted} minute"
    )

    return requests_made, daily_inserted, hourly_inserted, minute_inserted


def _commit_batch(session, daily_records, hourly_records, minute_records):
    """Commit a batch of records with upsert."""
    from sqlalchemy import text

    if daily_records:
        stmt = insert(VcForecastDaily).values(daily_records)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_vc_daily_row",
            set_={col: stmt.excluded[col] for col in [
                "tempmax_f", "tempmin_f", "temp_f",
                "humidity", "conditions", "source_system", "raw_json",
            ]},
        )
        session.execute(stmt)

    if hourly_records:
        stmt = insert(VcForecastHourly).values(hourly_records)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_vc_hourly_row",
            set_={col: stmt.excluded[col] for col in [
                "temp_f", "humidity", "conditions", "source_system", "raw_json",
            ]},
        )
        session.execute(stmt)

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
                "source_system": stmt.excluded.source_system,
                "raw_json": stmt.excluded.raw_json,
            },
        )
        session.execute(stmt)

    session.commit()


def main():
    parser = argparse.ArgumentParser(
        description="Backfill VC historical forecasts with 15-minute data"
    )
    parser.add_argument(
        "--city",
        choices=list(CITIES.keys()),
        help="Specific city to backfill (default: all cities)",
    )
    parser.add_argument(
        "--location-type",
        choices=["station", "city", "both"],
        default="both",
        help="Location type to backfill (default: both)",
    )
    parser.add_argument(
        "--start-date",
        default="2022-12-23",
        help="Start date (default: 2022-12-23)",
    )
    parser.add_argument(
        "--end-date",
        help="End date (default: yesterday)",
    )
    parser.add_argument(
        "--lead-days",
        default="0,1,2,3",
        help="Comma-separated lead days (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--initial-delay",
        type=float,
        default=0.002,
        help="Initial delay between requests in seconds (default: 0.002)",
    )
    parser.add_argument(
        "--target-delay",
        type=float,
        default=0.002,
        help="Target steady-state delay between requests in seconds (default: 0.002)",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=0.015,
        help="Maximum delay when backing off after rate limits (default: 0.015)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making requests",
    )

    args = parser.parse_args()

    # Parse dates
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date) if args.end_date else date.today() - timedelta(days=1)

    # Parse lead days
    lead_days_list = [int(x.strip()) for x in args.lead_days.split(",")]

    # Determine cities and location types
    cities = [args.city] if args.city else list(CITIES.keys())
    if args.location_type == "both":
        location_types = ["station", "city"]
    else:
        location_types = [args.location_type]

    # Get API key
    settings = get_settings()
    api_key = settings.vc_api_key

    if not api_key:
        logger.error("VC_API_KEY not set in environment")
        return

    # Create rate limiter
    rate_limiter = AdaptiveRateLimiter(
        initial_delay=args.initial_delay,
        max_delay=args.max_delay,
        target_delay=args.target_delay,
        recovery_steps=10,
    )

    logger.info("=" * 80)
    logger.info("VISUAL CROSSING HISTORICAL FORECAST BACKFILL")
    logger.info("=" * 80)
    logger.info(f"Cities: {cities}")
    logger.info(f"Location types: {location_types}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Lead days: {lead_days_list}")
    logger.info(
        "Rate limiting: initial=%.4fs, max=%.4fs, target=%.4fs",
        rate_limiter.initial_delay,
        rate_limiter.max_delay,
        rate_limiter.target_delay,
    )
    logger.info("=" * 80)

    if args.dry_run:
        logger.info("DRY RUN MODE - No API requests will be made")

    total_requests = 0
    total_daily = 0
    total_hourly = 0
    total_minute = 0

    start_time = time.time()

    for city_id in cities:
        for location_type in location_types:
            requests, daily, hourly, minute = backfill_location(
                city_id=city_id,
                location_type=location_type,
                start_date=start_date,
                end_date=end_date,
                lead_days_list=lead_days_list,
                api_key=api_key,
                rate_limiter=rate_limiter,
                dry_run=args.dry_run,
            )
            total_requests += requests
            total_daily += daily
            total_hourly += hourly
            total_minute += minute

    elapsed = time.time() - start_time

    logger.info("=" * 80)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Total daily records: {total_daily}")
    logger.info(f"Total hourly records: {total_hourly}")
    logger.info(f"Total minute records: {total_minute}")
    logger.info(f"Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    if total_requests > 0:
        logger.info(f"Average rate: {total_requests/elapsed:.2f} req/s")
    logger.info(f"Rate limiter stats: {rate_limiter.get_stats()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
