#!/usr/bin/env python3
"""
High-frequency settlement poller (default 30-minute cadence).

Fetches the previous day's CLI (final) numbers from IEM and optionally refreshes
CF6 prelims to keep the wx.settlement table up-to-date. Intended to be run via
systemd/cron every 30 minutes so the "source of truth" Tmax is always fresh.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from weather.iem_cli import IEMCliClient, SETTLEMENT_STATIONS
from weather.iem_cf6 import IEMClient as IEMCF6Client
from db.connection import get_session
from db.loaders import upsert_settlement

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll CLI/CF6 settlements")
    parser.add_argument(
        "--cities",
        default="all",
        help="Comma-separated list or 'all' (default)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=2,
        help="How many days back to refresh (default: 2)",
    )
    parser.add_argument(
        "--refresh-cf6",
        action="store_true",
        help="Refresh CF6 prelim data for the same window",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously instead of once",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=1800,
        help="Loop interval when --loop enabled (default: 1800s = 30 minutes)",
    )
    return parser.parse_args()


def run_once(cities: List[str], days_back: int, refresh_cf6: bool) -> None:
    today = date.today()
    start_date = today - timedelta(days=days_back)
    end_date = today - timedelta(days=1)
    if end_date < start_date:
        logger.info("Nothing to do (window empty)")
        return

    logger.info(
        f"Polling settlements for {', '.join(cities)} from {start_date} to {end_date}"
    )

    cli_client = IEMCliClient()
    cf6_client = IEMCF6Client() if refresh_cf6 else None

    with get_session() as session:
        for city in cities:
            cli_records = cli_client.get_settlements_for_city(
                city, start_date, end_date, fetch_raw_text=False
            )
            for record in cli_records:
                upsert_settlement(session, record)
            logger.info(f"{city}: upserted {len(cli_records)} CLI rows")

            if cf6_client:
                cf6_records = cf6_client.get_settlements_for_city(
                    city, start_date, end_date, fetch_raw_text=False
                )
                for record in cf6_records:
                    upsert_settlement(session, record)
                logger.info(f"{city}: refreshed {len(cf6_records)} CF6 rows")

        session.commit()
    logger.info("Poll complete")


def main() -> int:
    args = parse_args()
    load_dotenv()

    if args.cities == "all":
        cities = list(SETTLEMENT_STATIONS.keys())
    else:
        cities = [c.strip() for c in args.cities.split(",")]

    for city in cities:
        if city not in SETTLEMENT_STATIONS:
            logger.error(f"Unknown city: {city}")
            return 1

    if args.loop:
        logger.info("Starting settlement poller loop")
        while True:
            start = time.time()
            try:
                run_once(cities, args.days_back, args.refresh_cf6)
            except Exception as exc:
                logger.error(f"Poll iteration failed: {exc}", exc_info=True)
            elapsed = time.time() - start
            sleep_for = max(0, args.interval_seconds - elapsed)
            logger.info(f"Sleeping {sleep_for:.1f}s before next poll")
            time.sleep(sleep_for)
    else:
        run_once(cities, args.days_back, args.refresh_cf6)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
