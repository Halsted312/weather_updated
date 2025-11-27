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
from typing import Any, Dict, List, Optional, Set
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
    window_start_hour: int = 0,
    window_start_minute: int = 0,
    window_end_minute: int = 10,
) -> bool:
    """
    Check if local time is within the snapshot window (default: 00:00-00:10).

    Args:
        local_time: Current local datetime for the city
        window_start_hour: Hour to start window (default: 0 = midnight)
        window_start_minute: Minute to start window (default: 0)
        window_end_minute: Minute to end window (default: 10)

    Returns:
        True if within the snapshot window
    """
    if local_time.hour != window_start_hour:
        return False
    return window_start_minute <= local_time.minute < window_end_minute


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


def process_cities_tick(
    client: VisualCrossingClient,
    session,
    cities: List[str],
    window_start: int,
    window_end: int,
) -> dict:
    """
    Process one tick of the daemon loop.

    For each city, check if it's in the snapshot window and if snapshot is needed.

    Args:
        client: Visual Crossing client
        session: Database session
        cities: List of city IDs to process
        window_start: Minute to start window
        window_end: Minute to end window

    Returns:
        Dict of cities processed this tick: {city_id: {"daily": N, "hourly": M}}
    """
    processed = {}

    for city_id in cities:
        try:
            local_time = get_city_local_time(city_id)
            basis_date = local_time.date()

            # Check if in snapshot window
            if not is_in_snapshot_window(local_time, window_start_minute=window_start, window_end_minute=window_end):
                continue

            # Check if snapshot already exists (check hourly as proxy - both are fetched together)
            if check_snapshot_exists(session, city_id, basis_date, "hourly"):
                logger.debug(f"{city_id}: Snapshot already exists for {basis_date}")
                continue

            # Take snapshot!
            logger.info(f"{city_id}: Taking forecast snapshot for basis_date={basis_date} (local time: {local_time.strftime('%H:%M')})")

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

                # Update and complete checkpoint
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
                logger.info(
                    f"{city_id}: Snapshot complete - {counts['daily']} daily, "
                    f"{counts['hourly']} hourly records"
                )

            except Exception as e:
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
                logger.error(f"{city_id}: Snapshot failed - {e}")

        except Exception as e:
            logger.error(f"Error processing {city_id}: {e}")
            # Continue to next city
            continue

    return processed


def run_daemon(
    tick_interval: int = 60,
    window_start: int = 0,
    window_end: int = 10,
    rate_limit: float = 0.05,
):
    """
    Main daemon loop.

    Args:
        tick_interval: Seconds between ticks (default: 60)
        window_start: Minute to start snapshot window (default: 0)
        window_end: Minute to end snapshot window (default: 10)
        rate_limit: API rate limit delay in seconds
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

    logger.info("=" * 60)
    logger.info("Visual Crossing Forecast Daemon Starting")
    logger.info(f"  Cities: {cities}")
    logger.info(f"  Tick interval: {tick_interval}s")
    logger.info(f"  Snapshot window: 00:{window_start:02d} - 00:{window_end:02d} local")
    logger.info(f"  Daily horizon: {MAX_DAY_HORIZON} days")
    logger.info(f"  Hourly horizon: {MAX_HOUR_HORIZON} hours")
    logger.info("=" * 60)

    tick_count = 0
    total_snapshots = {"daily": 0, "hourly": 0}

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
                )

                # Accumulate totals
                for city_id, counts in processed.items():
                    total_snapshots["daily"] += counts["daily"]
                    total_snapshots["hourly"] += counts["hourly"]

        except Exception as e:
            logger.error(f"Tick {tick_count} error: {e}")

        # Log periodic status
        if tick_count % 60 == 0:  # Every ~60 ticks (1 hour at 60s interval)
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
        "--tick-interval", type=int, default=60,
        help="Seconds between ticks (default: 60)"
    )
    parser.add_argument(
        "--window-start", type=int, default=0,
        help="Minute to start snapshot window (default: 0)"
    )
    parser.add_argument(
        "--window-end", type=int, default=10,
        help="Minute to end snapshot window (default: 10)"
    )
    parser.add_argument(
        "--rate-limit", type=float, default=0.05,
        help="API rate limit delay in seconds (default: 0.05)"
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
    )


if __name__ == "__main__":
    main()
