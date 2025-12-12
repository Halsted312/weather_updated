#!/usr/bin/env python3
"""
Settlement polling daemon - keeps settlement data current.

Polls IEM/NCEI for settlement data at regular intervals to catch all timezone
publication times before their respective market opens.

Settlement data comes from IEM/NCEI and updates once per day when the previous
day's CLI report is published (typically around 6-7 AM local time for each city).

Cities span 4 US timezones:
- Eastern (Miami, Philadelphia): Data ~6-7 AM EST
- Central (Chicago, Austin): Data ~6-7 AM CST
- Mountain (Denver): Data ~6-7 AM MST
- Pacific (Los Angeles): Data ~6-7 AM PST

Default schedule: Start at 5:20 AM EST (10:20 UTC), poll every 30 min for 7.5 hours.
This covers all timezones before their 10 AM market open.

Usage:
    python scripts/poll_settlement_daemon.py
    python scripts/poll_settlement_daemon.py --once     # Run once and exit
    python scripts/poll_settlement_daemon.py --start 05:20 --interval 30 --duration 7.5
"""

import argparse
import logging
import signal
import sys
import time
from datetime import date, datetime, timedelta
from typing import List, Optional

from zoneinfo import ZoneInfo

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from sqlalchemy.dialects.postgresql import insert

from src.config import CITIES
from src.db import get_db_session
from src.db.models import WxSettlement, KalshiMarket
from src.weather.iem_cli import IEMCliClient
from src.weather.noaa_ncei import NCEIAccessClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    logger.info(f"Received signal {signum}, requesting graceful shutdown...")
    shutdown_requested = True


# Default schedule: 5:20 AM EST start, every 30 min for 7.5 hours
# This covers Eastern → Pacific timezones before their 10 AM market opens
DEFAULT_START = "05:20"  # EST
DEFAULT_INTERVAL = 30  # minutes
DEFAULT_DURATION = 7.5  # hours
SCHEDULE_TIMEZONE = ZoneInfo("America/New_York")  # EST/EDT


def get_kalshi_settlement_for_date(session, city: str, target_date: date) -> Optional[dict]:
    """Find the Kalshi market that settled YES for a given city and date."""
    from sqlalchemy import select, and_

    stmt = select(KalshiMarket).where(
        and_(
            KalshiMarket.city == city,
            KalshiMarket.event_date == target_date,
            KalshiMarket.result == "yes",
        )
    )
    result = session.execute(stmt).scalars().first()

    if not result:
        return None

    return {
        "settled_ticker": result.ticker,
        "settled_bucket_type": result.strike_type,
        "settled_floor_strike": result.floor_strike,
        "settled_cap_strike": result.cap_strike,
        "settled_bucket_label": f"{result.strike_type}_{result.floor_strike}_{result.cap_strike}",
    }


def ingest_settlement_for_city(
    session,
    iem_client: IEMCliClient,
    ncei_client: NCEIAccessClient,
    city: str,
    start_date: date,
    end_date: date,
) -> int:
    """Ingest settlement data for a single city. Returns count of records processed."""
    city_config = CITIES.get(city)
    if not city_config:
        logger.warning(f"Unknown city: {city}")
        return 0

    # Fetch from both sources (returns list of dicts)
    iem_data = iem_client.fetch_city_history(city, start_date, end_date)
    ncei_data = ncei_client.fetch_city_history(city, start_date, end_date)

    # Index by date
    iem_by_date = {r["date_local"]: r for r in iem_data}
    ncei_by_date = {r["date_local"]: r for r in ncei_data}

    # Get all dates
    all_dates = sorted(set(iem_by_date.keys()) | set(ncei_by_date.keys()))

    if not all_dates:
        return 0

    processed = 0
    for target_date in all_dates:
        iem = iem_by_date.get(target_date)
        ncei = ncei_by_date.get(target_date)

        tmax_iem = iem["tmax_f"] if iem else None
        tmax_ncei = ncei["tmax_f"] if ncei else None

        # Skip if no data from any source
        if tmax_iem is None and tmax_ncei is None:
            continue

        # Determine final TMAX (prefer IEM, fallback to NCEI)
        if tmax_iem is not None:
            tmax_final = tmax_iem
            source_final = "iem"
        else:
            tmax_final = tmax_ncei
            source_final = "ncei"

        # Get Kalshi settlement info
        kalshi_info = get_kalshi_settlement_for_date(session, city, target_date) or {}

        record = {
            "city": city,
            "date_local": target_date,
            "tmax_iem_f": tmax_iem,
            "tmax_ncei_f": tmax_ncei,
            "tmax_final": tmax_final,
            "source_final": source_final,
            "raw_payload_iem": iem["raw_json"] if iem else None,
            "raw_payload_ncei": ncei["raw_json"] if ncei else None,
            **kalshi_info,
        }

        # Upsert
        stmt = insert(WxSettlement).values(**record)
        stmt = stmt.on_conflict_do_update(
            index_elements=["city", "date_local"],
            set_={
                "tmax_iem_f": stmt.excluded.tmax_iem_f,
                "tmax_ncei_f": stmt.excluded.tmax_ncei_f,
                "tmax_final": stmt.excluded.tmax_final,
                "source_final": stmt.excluded.source_final,
                "raw_payload_iem": stmt.excluded.raw_payload_iem,
                "raw_payload_ncei": stmt.excluded.raw_payload_ncei,
                "settled_ticker": stmt.excluded.settled_ticker,
                "settled_bucket_type": stmt.excluded.settled_bucket_type,
                "settled_floor_strike": stmt.excluded.settled_floor_strike,
                "settled_cap_strike": stmt.excluded.settled_cap_strike,
                "settled_bucket_label": stmt.excluded.settled_bucket_label,
            },
        )
        session.execute(stmt)
        processed += 1

    session.commit()
    return processed


def run_settlement_update(lookback_days: int = 7) -> int:
    """Run settlement update for all cities, looking back N days."""
    logger.info(f"Running settlement update (lookback={lookback_days} days)")

    end_date = date.today() - timedelta(days=1)  # Yesterday (today hasn't settled)
    start_date = end_date - timedelta(days=lookback_days)

    iem_client = IEMCliClient()
    ncei_client = NCEIAccessClient()

    total = 0
    with get_db_session() as session:
        for city_id in CITIES.keys():
            try:
                count = ingest_settlement_for_city(
                    session, iem_client, ncei_client, city_id, start_date, end_date
                )
                if count > 0:
                    logger.info(f"  {city_id}: {count} records")
                total += count
            except Exception as e:
                logger.error(f"  {city_id}: ERROR - {e}")

    logger.info(f"Settlement update complete: {total} total records")
    return total


def is_in_polling_window(start_time: str, duration_hours: float, tz: ZoneInfo) -> bool:
    """Check if current time is within the daily polling window."""
    now = datetime.now(tz)
    hour, minute = map(int, start_time.split(":"))

    window_start = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    window_end = window_start + timedelta(hours=duration_hours)

    return window_start <= now <= window_end


def get_next_poll_time(
    start_time: str,
    interval_minutes: int,
    duration_hours: float,
    tz: ZoneInfo
) -> datetime:
    """Get the next poll time based on interval schedule."""
    now = datetime.now(tz)
    hour, minute = map(int, start_time.split(":"))

    # Today's window
    today_start = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    today_end = today_start + timedelta(hours=duration_hours)

    # If we're before today's window, return the start
    if now < today_start:
        return today_start

    # If we're in today's window, find next interval
    if now <= today_end:
        # Calculate which interval we're in
        elapsed = (now - today_start).total_seconds() / 60
        next_interval = ((int(elapsed) // interval_minutes) + 1) * interval_minutes
        next_time = today_start + timedelta(minutes=next_interval)

        # If next time is still in window, return it
        if next_time <= today_end:
            return next_time

    # Otherwise, return tomorrow's start
    tomorrow_start = today_start + timedelta(days=1)
    return tomorrow_start


def main():
    parser = argparse.ArgumentParser(
        description="Settlement polling daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 5:20 AM EST, every 30 min for 7.5 hours
  python scripts/poll_settlement_daemon.py

  # Custom schedule
  python scripts/poll_settlement_daemon.py --start 06:00 --interval 20 --duration 6

  # Run once and exit
  python scripts/poll_settlement_daemon.py --once
"""
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=DEFAULT_START,
        help=f"Start time HH:MM in EST (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Minutes between polls (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help=f"Hours to poll for each day (default: {DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=7,
        help="Number of days to look back (default: 7)",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Calculate number of polls per day
    polls_per_day = int((args.duration * 60) / args.interval) + 1

    # Parse end time for display
    start_h, start_m = map(int, args.start.split(":"))
    end_time = datetime(2000, 1, 1, start_h, start_m) + timedelta(hours=args.duration)

    logger.info("=" * 60)
    logger.info("Settlement Polling Daemon Starting")
    logger.info(f"  Schedule: {args.start} - {end_time.strftime('%H:%M')} EST")
    logger.info(f"  Interval: every {args.interval} minutes ({polls_per_day} polls/day)")
    logger.info(f"  Lookback: {args.lookback} days")
    logger.info("=" * 60)

    # Run immediately on startup
    logger.info("Running initial settlement update...")
    run_settlement_update(args.lookback)

    if args.once:
        logger.info("--once specified, exiting")
        return

    # Main loop
    while not shutdown_requested:
        next_run = get_next_poll_time(
            args.start, args.interval, args.duration, SCHEDULE_TIMEZONE
        )
        wait_seconds = (next_run - datetime.now(SCHEDULE_TIMEZONE)).total_seconds()

        # Ensure positive wait time
        if wait_seconds < 0:
            wait_seconds = 0

        logger.info(f"Next run at {next_run.strftime('%Y-%m-%d %H:%M %Z')} ({wait_seconds/60:.1f} min)")

        # Sleep until next run (check shutdown every 60s)
        while wait_seconds > 0 and not shutdown_requested:
            sleep_time = min(60, wait_seconds)
            time.sleep(sleep_time)
            wait_seconds -= sleep_time

        if shutdown_requested:
            break

        # Run update
        run_settlement_update(args.lookback)

    logger.info("=" * 60)
    logger.info("Settlement Polling Daemon Stopped")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
