#!/usr/bin/env python3
"""
24/7 Visual Crossing forecast polling daemon.

Runs continuously, checking each city at its local midnight (00:00-00:10)
to capture daily forecast snapshots (7 days) and hourly forecast curves (72 hours).

This daemon captures FORECAST data, not settlement verification.
Settlement verification remains on the NWS/IEM/NCEI pipeline.

Usage:
    python scripts/poll_vc_forecast_daemon.py
    python scripts/poll_vc_forecast_daemon.py --tick-interval 120
    python scripts/poll_vc_forecast_daemon.py --window-start 0 --window-end 15

Deployment:
    systemctl start vc-forecast-daemon
    docker-compose up -d vc-forecast-daemon
"""

import argparse
import logging
import signal
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import CITIES, EXCLUDED_VC_CITIES, get_city, get_settings
from src.db import get_db_session, WxForecastSnapshot, WxForecastSnapshotHourly
from src.db.checkpoint import get_or_create_checkpoint, update_checkpoint, complete_checkpoint
from src.weather.visual_crossing import VisualCrossingClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configurable horizons (per design decision)
MAX_HOUR_HORIZON = 72  # 3 days of hourly data
MAX_DAY_HORIZON = 7    # 7 days of daily data

# Shutdown flag for graceful termination
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, requesting graceful shutdown...")
    shutdown_requested = True


def get_city_local_time(city_id: str) -> datetime:
    """Get current local time for a city using its IANA timezone."""
    city = get_city(city_id)
    tz = ZoneInfo(city.timezone)
    return datetime.now(tz)


def is_in_snapshot_window(
    local_time: datetime,
    window_start_minute: int = -5,
    window_end_minute: int = 10,
) -> bool:
    """
    Check if local time is within the snapshot window (default: 23:55-00:10).

    The window spans midnight to catch the new day's forecast as it becomes available.
    Visual Crossing stores forecasts per forecastBasisDate (midnight UTC model runs),
    so by local midnight the relevant run is already stored.

    Args:
        local_time: Current local datetime for the city
        window_start_minute: Minutes relative to midnight to start (default: -5 = 23:55)
        window_end_minute: Minutes after midnight to end (default: 10 = 00:10)

    Returns:
        True if within the snapshot window
    """
    # Convert to minutes since midnight (can be negative for previous day)
    minutes_since_midnight = local_time.hour * 60 + local_time.minute

    # Handle the 23:55-00:10 window (spans midnight)
    # At 23:55, minutes_since_midnight = 1435, we want this to be -5
    if minutes_since_midnight >= 23 * 60:  # After 23:00
        minutes_since_midnight -= 24 * 60  # Make it negative

    return window_start_minute <= minutes_since_midnight < window_end_minute


def check_snapshot_exists(session, city_id: str, basis_date: date, table: str) -> bool:
    """
    Check if a forecast snapshot already exists for this city/basis_date.

    Args:
        session: Database session
        city_id: City ID
        basis_date: The basis date to check
        table: 'daily' or 'hourly'

    Returns:
        True if snapshot exists
    """
    if table == "daily":
        query = select(WxForecastSnapshot.city).where(
            WxForecastSnapshot.city == city_id,
            WxForecastSnapshot.basis_date == basis_date,
        ).limit(1)
    else:  # hourly
        query = select(WxForecastSnapshotHourly.city).where(
            WxForecastSnapshotHourly.city == city_id,
            WxForecastSnapshotHourly.basis_date == basis_date,
        ).limit(1)

    result = session.execute(query).scalar_one_or_none()
    return result is not None


def validate_72_hour_coverage(session, city_id: str, basis_date: date) -> tuple[bool, str]:
    """
    Validate that we have complete 72-hour coverage for a city/basis_date.

    Checks:
    - Count of rows with target_hour_local BETWEEN basis_date 00:00 AND basis_date+2 23:00 is exactly 72
    - MIN(target_hour_local) = basis_date 00:00
    - MAX(target_hour_local) = basis_date+2 23:00

    Args:
        session: Database session
        city_id: City ID
        basis_date: The basis date to validate

    Returns:
        Tuple of (is_valid, message)
    """
    from sqlalchemy import func

    expected_start = datetime.combine(basis_date, datetime.min.time())  # basis_date 00:00
    expected_end = datetime.combine(basis_date + timedelta(days=2), datetime.min.time().replace(hour=23))  # basis_date+2 23:00

    # Query count, min, max in one shot
    query = select(
        func.count(WxForecastSnapshotHourly.target_hour_local).label("cnt"),
        func.min(WxForecastSnapshotHourly.target_hour_local).label("min_hour"),
        func.max(WxForecastSnapshotHourly.target_hour_local).label("max_hour"),
    ).where(
        WxForecastSnapshotHourly.city == city_id,
        WxForecastSnapshotHourly.basis_date == basis_date,
        WxForecastSnapshotHourly.target_hour_local >= expected_start,
        WxForecastSnapshotHourly.target_hour_local <= expected_end,
    )

    result = session.execute(query).one()
    cnt, min_hour, max_hour = result.cnt, result.min_hour, result.max_hour

    if cnt != 72:
        return False, f"Expected 72 hours, got {cnt}"

    if min_hour != expected_start:
        return False, f"MIN mismatch: expected {expected_start}, got {min_hour}"

    if max_hour != expected_end:
        return False, f"MAX mismatch: expected {expected_end}, got {max_hour}"

    return True, "Valid 72-hour coverage"


def parse_daily_forecast_to_records(
    city_id: str,
    basis_date: date,
    payload: Dict[str, Any],
    horizon_days: int = MAX_DAY_HORIZON,
) -> List[dict]:
    """Convert Visual Crossing daily forecast payload to database records."""
    records = []

    for day in payload.get("days", []):
        target_str = day.get("datetime")
        if not target_str:
            continue

        target_date = date.fromisoformat(target_str)
        lead_days = (target_date - basis_date).days

        if lead_days < 0 or lead_days >= horizon_days:
            continue

        record = {
            "city": city_id,
            "target_date": target_date,
            "basis_date": basis_date,
            "lead_days": lead_days,
            "provider": "visualcrossing",
            "tempmax_fcst_f": day.get("tempmax"),
            "tempmin_fcst_f": day.get("tempmin"),
            "precip_fcst_in": day.get("precip"),
            "precip_prob_fcst": day.get("precipprob"),
            "humidity_fcst": day.get("humidity"),
            "windspeed_fcst_mph": day.get("windspeed"),
            "conditions_fcst": day.get("conditions"),
            "raw_json": day,
        }
        records.append(record)

    return records


def parse_hourly_forecast_to_records(
    city_id: str,
    tz_name: str,
    basis_date: date,
    payload: Dict[str, Any],
    max_hours: int = MAX_HOUR_HORIZON,
) -> List[dict]:
    """Convert Visual Crossing hourly forecast payload to database records."""
    records = []
    hour_count = 0

    # Track seen local hours to handle DST fall-back duplicates
    seen_local_hours: Set[datetime] = set()

    for day in payload.get("days", []):
        hours = day.get("hours", [])

        for hour in hours:
            if hour_count >= max_hours:
                break

            datetime_str = hour.get("datetime")
            epoch = hour.get("datetimeEpoch")

            if not datetime_str or not epoch:
                continue

            day_date = day.get("datetime")
            hour_time = datetime_str
            target_hour_local = datetime.fromisoformat(f"{day_date}T{hour_time}")

            # Skip duplicate local hour (DST fall-back)
            if target_hour_local in seen_local_hours:
                continue
            seen_local_hours.add(target_hour_local)

            record = {
                "city": city_id,
                "target_hour_local": target_hour_local,
                "target_hour_epoch": epoch,
                "basis_date": basis_date,
                "lead_hours": hour_count,
                "provider": "visualcrossing",
                "tz_name": tz_name,
                "temp_fcst_f": hour.get("temp"),
                "feelslike_fcst_f": hour.get("feelslike"),
                "humidity_fcst": hour.get("humidity"),
                "precip_fcst_in": hour.get("precip"),
                "precip_prob_fcst": hour.get("precipprob"),
                "windspeed_fcst_mph": hour.get("windspeed"),
                "conditions_fcst": hour.get("conditions"),
                "raw_json": hour,
            }
            records.append(record)
            hour_count += 1

        if hour_count >= max_hours:
            break

    return records


def upsert_daily_forecasts(session, records: List[dict]) -> int:
    """Upsert daily forecast snapshots into wx.forecast_snapshot."""
    if not records:
        return 0

    stmt = insert(WxForecastSnapshot).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["city", "target_date", "basis_date"],
        set_={
            "lead_days": stmt.excluded.lead_days,
            "provider": stmt.excluded.provider,
            "tempmax_fcst_f": stmt.excluded.tempmax_fcst_f,
            "tempmin_fcst_f": stmt.excluded.tempmin_fcst_f,
            "precip_fcst_in": stmt.excluded.precip_fcst_in,
            "precip_prob_fcst": stmt.excluded.precip_prob_fcst,
            "humidity_fcst": stmt.excluded.humidity_fcst,
            "windspeed_fcst_mph": stmt.excluded.windspeed_fcst_mph,
            "conditions_fcst": stmt.excluded.conditions_fcst,
            "raw_json": stmt.excluded.raw_json,
        },
    )

    result = session.execute(stmt)
    return result.rowcount


def upsert_hourly_forecasts(session, records: List[dict]) -> int:
    """Upsert hourly forecast snapshots into wx.forecast_snapshot_hourly."""
    if not records:
        return 0

    stmt = insert(WxForecastSnapshotHourly).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["city", "target_hour_local", "basis_date"],
        set_={
            "target_hour_epoch": stmt.excluded.target_hour_epoch,
            "lead_hours": stmt.excluded.lead_hours,
            "provider": stmt.excluded.provider,
            "tz_name": stmt.excluded.tz_name,
            "temp_fcst_f": stmt.excluded.temp_fcst_f,
            "feelslike_fcst_f": stmt.excluded.feelslike_fcst_f,
            "humidity_fcst": stmt.excluded.humidity_fcst,
            "precip_fcst_in": stmt.excluded.precip_fcst_in,
            "precip_prob_fcst": stmt.excluded.precip_prob_fcst,
            "windspeed_fcst_mph": stmt.excluded.windspeed_fcst_mph,
            "conditions_fcst": stmt.excluded.conditions_fcst,
            "raw_json": stmt.excluded.raw_json,
        },
    )

    result = session.execute(stmt)
    return result.rowcount


def fetch_and_store_city_snapshot(
    client: VisualCrossingClient,
    session,
    city_id: str,
    basis_date: date,
) -> dict:
    """
    Fetch and store both daily and hourly forecasts for a city.

    Args:
        client: Visual Crossing client
        session: Database session
        city_id: City ID
        basis_date: The basis date (local today)

    Returns:
        Dict with counts: {"daily": N, "hourly": M}
    """
    city = get_city(city_id)
    location = f"stn:{city.icao}"
    tz_name = city.timezone

    result = {"daily": 0, "hourly": 0}

    # Fetch hourly forecast (includes daily data in the response)
    try:
        payload = client.fetch_historical_hourly_forecast(
            location=location,
            basis_date=basis_date.isoformat(),
            horizon_hours=MAX_HOUR_HORIZON,
            horizon_days=MAX_DAY_HORIZON,
        )

        # Parse and upsert daily forecasts
        daily_records = parse_daily_forecast_to_records(
            city_id=city_id,
            basis_date=basis_date,
            payload=payload,
            horizon_days=MAX_DAY_HORIZON,
        )
        result["daily"] = upsert_daily_forecasts(session, daily_records)

        # Parse and upsert hourly forecasts
        hourly_records = parse_hourly_forecast_to_records(
            city_id=city_id,
            tz_name=tz_name,
            basis_date=basis_date,
            payload=payload,
            max_hours=MAX_HOUR_HORIZON,
        )
        result["hourly"] = upsert_hourly_forecasts(session, hourly_records)

        session.commit()

    except Exception as e:
        logger.error(f"Error fetching snapshot for {city_id} basis={basis_date}: {e}")
        session.rollback()
        raise

    return result


def get_target_basis_date(local_time: datetime) -> date:
    """
    Determine the target basis_date for the forecast snapshot.

    If we're before midnight (23:55-23:59), we want tomorrow's forecast.
    If we're after midnight (00:00-00:10), we want today's forecast.

    Args:
        local_time: Current local datetime for the city

    Returns:
        The basis_date to use for the forecast
    """
    if local_time.hour >= 23:
        # Before midnight - we want tomorrow's date as basis
        return local_time.date() + timedelta(days=1)
    else:
        # After midnight - we want today's date
        return local_time.date()


def process_cities_tick(
    client: VisualCrossingClient,
    session,
    cities: List[str],
    window_start: int,
    window_end: int,
    retry_counts: Optional[Dict[str, int]] = None,
    max_retries: int = 3,
) -> dict:
    """
    Process one tick of the daemon loop.

    For each city, check if it's in the snapshot window and if snapshot is needed.
    Uses proper 72-hour validation and retry logic with backoff.

    Args:
        client: Visual Crossing client
        session: Database session
        cities: List of city IDs to process
        window_start: Minutes relative to midnight to start window
        window_end: Minutes after midnight to end window
        retry_counts: Dict tracking retry attempts per city (mutated)
        max_retries: Maximum retries before giving up for the day

    Returns:
        Dict of cities processed this tick: {city_id: {"daily": N, "hourly": M}}
    """
    if retry_counts is None:
        retry_counts = {}

    processed = {}

    for city_id in cities:
        try:
            local_time = get_city_local_time(city_id)

            # Check if in snapshot window
            if not is_in_snapshot_window(local_time, window_start_minute=window_start, window_end_minute=window_end):
                # Reset retry count when outside window
                retry_counts.pop(city_id, None)
                continue

            # Determine the correct basis_date (handles 23:55-00:10 spanning midnight)
            basis_date = get_target_basis_date(local_time)

            # Check if we already have valid 72-hour coverage
            is_valid, msg = validate_72_hour_coverage(session, city_id, basis_date)
            if is_valid:
                logger.debug(f"{city_id}: Valid 72-hour coverage for {basis_date}")
                retry_counts.pop(city_id, None)  # Reset retry count
                continue

            # Check retry limit
            current_retries = retry_counts.get(city_id, 0)
            if current_retries >= max_retries:
                logger.warning(f"{city_id}: Max retries ({max_retries}) reached for {basis_date}, skipping until next day")
                continue

            # Take snapshot!
            logger.info(
                f"{city_id}: Taking forecast snapshot for basis_date={basis_date} "
                f"(local time: {local_time.strftime('%H:%M')}, retry: {current_retries})"
            )

            # Create checkpoint for tracking
            checkpoint = get_or_create_checkpoint(
                session=session,
                pipeline_name=f"vc_forecast_live/{city_id}",
                city=city_id,
            )
            session.commit()

            try:
                counts = fetch_and_store_city_snapshot(
                    client=client,
                    session=session,
                    city_id=city_id,
                    basis_date=basis_date,
                )

                # Validate the data we just wrote
                is_valid, validation_msg = validate_72_hour_coverage(session, city_id, basis_date)

                if is_valid:
                    # Success! Update and complete checkpoint
                    update_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint.id,
                        last_date=basis_date,
                        processed_count=counts["daily"] + counts["hourly"],
                    )
                    complete_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint.id,
                        status="completed",
                    )
                    session.commit()

                    processed[city_id] = counts
                    retry_counts.pop(city_id, None)  # Reset retry count
                    logger.info(
                        f"{city_id}: Snapshot complete and validated - {counts['daily']} daily, "
                        f"{counts['hourly']} hourly records"
                    )
                else:
                    # Data written but validation failed - will retry next tick
                    retry_counts[city_id] = current_retries + 1
                    logger.warning(
                        f"{city_id}: Snapshot written but validation failed: {validation_msg}. "
                        f"Will retry ({current_retries + 1}/{max_retries})"
                    )
                    update_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint.id,
                        error=f"Validation failed: {validation_msg}",
                    )
                    session.commit()

            except Exception as e:
                retry_counts[city_id] = current_retries + 1
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
                logger.error(f"{city_id}: Snapshot failed - {e}. Will retry ({current_retries + 1}/{max_retries})")

        except Exception as e:
            logger.error(f"Error processing {city_id}: {e}")
            # Continue to next city
            continue

    return processed


def run_daemon(
    tick_interval: int = 30,
    window_start: int = -5,
    window_end: int = 10,
    rate_limit: float = 0.05,
    max_retries: int = 3,
):
    """
    Main daemon loop.

    Args:
        tick_interval: Seconds between ticks (default: 30)
        window_start: Minutes relative to midnight to start window (default: -5 = 23:55)
        window_end: Minutes after midnight to end window (default: 10 = 00:10)
        rate_limit: API rate limit delay in seconds
        max_retries: Maximum retries per city before giving up
    """
    global shutdown_requested

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Get settings and create client
    settings = get_settings()
    client = VisualCrossingClient(
        api_key=settings.vc_api_key,
        base_url=settings.vc_base_url,
        rate_limit_delay=rate_limit,
    )

    # Get cities to process (exclude VC-excluded cities)
    cities = [c for c in CITIES.keys() if c not in EXCLUDED_VC_CITIES]

    # Format window for display
    if window_start < 0:
        window_start_str = f"23:{60 + window_start:02d}"
    else:
        window_start_str = f"00:{window_start:02d}"

    logger.info("=" * 60)
    logger.info("Visual Crossing Forecast Daemon Starting")
    logger.info(f"  Cities: {cities}")
    logger.info(f"  Tick interval: {tick_interval}s")
    logger.info(f"  Snapshot window: {window_start_str} - 00:{window_end:02d} local")
    logger.info(f"  Daily horizon: {MAX_DAY_HORIZON} days")
    logger.info(f"  Hourly horizon: {MAX_HOUR_HORIZON} hours")
    logger.info(f"  Max retries per city: {max_retries}")
    logger.info("=" * 60)

    tick_count = 0
    total_snapshots = {"daily": 0, "hourly": 0}
    retry_counts: Dict[str, int] = {}  # Track retries per city

    while not shutdown_requested:
        tick_count += 1
        tick_start = time.time()

        try:
            with get_db_session() as session:
                processed = process_cities_tick(
                    client=client,
                    session=session,
                    cities=cities,
                    window_start=window_start,
                    window_end=window_end,
                    retry_counts=retry_counts,
                    max_retries=max_retries,
                )

                # Accumulate totals
                for city_id, counts in processed.items():
                    total_snapshots["daily"] += counts["daily"]
                    total_snapshots["hourly"] += counts["hourly"]

        except Exception as e:
            logger.error(f"Tick {tick_count} error: {e}")

        # Log periodic status (every ~30 minutes at 30s interval)
        if tick_count % 60 == 0:
            logger.info(
                f"Status: tick={tick_count}, total_daily={total_snapshots['daily']}, "
                f"total_hourly={total_snapshots['hourly']}"
            )

        # Sleep until next tick (accounting for processing time)
        elapsed = time.time() - tick_start
        sleep_time = max(0, tick_interval - elapsed)

        if shutdown_requested:
            break

        time.sleep(sleep_time)

    logger.info("=" * 60)
    logger.info("Daemon shutting down gracefully")
    logger.info(f"  Total ticks: {tick_count}")
    logger.info(f"  Total daily records: {total_snapshots['daily']}")
    logger.info(f"  Total hourly records: {total_snapshots['hourly']}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="24/7 Visual Crossing forecast polling daemon"
    )
    parser.add_argument(
        "--tick-interval", type=int, default=30,
        help="Seconds between ticks (default: 30)"
    )
    parser.add_argument(
        "--window-start", type=int, default=-5,
        help="Minutes relative to midnight to start window (default: -5 = 23:55)"
    )
    parser.add_argument(
        "--window-end", type=int, default=10,
        help="Minutes after midnight to end window (default: 10 = 00:10)"
    )
    parser.add_argument(
        "--rate-limit", type=float, default=0.05,
        help="API rate limit delay in seconds (default: 0.05)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Maximum retries per city before giving up for the day (default: 3)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run once (check all cities) and exit, useful for testing"
    )

    args = parser.parse_args()

    if args.once:
        # Single-pass mode for testing
        logger.info("Running single-pass mode (--once)")
        settings = get_settings()
        client = VisualCrossingClient(
            api_key=settings.vc_api_key,
            base_url=settings.vc_base_url,
            rate_limit_delay=args.rate_limit,
        )
        cities = [c for c in CITIES.keys() if c not in EXCLUDED_VC_CITIES]

        with get_db_session() as session:
            for city_id in cities:
                local_time = get_city_local_time(city_id)
                basis_date = local_time.date()

                # Check if snapshot exists
                exists = check_snapshot_exists(session, city_id, basis_date, "hourly")
                logger.info(
                    f"{city_id}: local={local_time.strftime('%Y-%m-%d %H:%M %Z')}, "
                    f"basis={basis_date}, snapshot_exists={exists}"
                )

                if not exists:
                    logger.info(f"{city_id}: Taking snapshot...")
                    counts = fetch_and_store_city_snapshot(
                        client=client,
                        session=session,
                        city_id=city_id,
                        basis_date=basis_date,
                    )
                    logger.info(f"{city_id}: {counts['daily']} daily, {counts['hourly']} hourly")
        return

    # Normal daemon mode
    run_daemon(
        tick_interval=args.tick_interval,
        window_start=args.window_start,
        window_end=args.window_end,
        rate_limit=args.rate_limit,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
