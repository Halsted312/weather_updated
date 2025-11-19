#!/usr/bin/env python3
"""Database health check covering schema, markets, and weather recency."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import get_session
from kalshi.city_config import CITY_CONFIG
from kalshi.date_utils import series_ticker_for_city
from weather.time_utils import coerce_datetime_to_utc, utc_now

logger = logging.getLogger(__name__)


def _check_connection(session) -> bool:
    session.execute(text("SELECT 1"))
    return True


def _check_alembic(session) -> bool:
    version = session.execute(text("SELECT version_num FROM alembic_version"))
    row = version.fetchone()
    if not row:
        logger.error("alembic_version table is empty")
        return False
    logger.info("Alembic head: %s", row.version_num)
    return True


def _check_series_recency(session, max_age: timedelta) -> bool:
    ok = True
    cutoff = utc_now() - max_age
    logger.info("Validating market recency since %s", cutoff.isoformat())

    for city in sorted(CITY_CONFIG):
        series_ticker = series_ticker_for_city(city)
        latest = session.execute(
            text("SELECT MAX(close_time) FROM markets WHERE series_ticker = :series"),
            {"series": series_ticker},
        ).scalar()
        if not latest:
            logger.error("No markets found for %s", series_ticker)
            ok = False
            continue
        latest_utc = coerce_datetime_to_utc(latest)
        logger.info("%s latest close_time: %s", series_ticker, latest_utc.isoformat())
        if latest_utc < cutoff:
            logger.error("%s stale by %.1f hours", series_ticker, (cutoff - latest_utc).total_seconds() / 3600)
            ok = False
    return ok


def _check_weather_recency(session, max_age: timedelta) -> bool:
    ok = True
    cutoff = utc_now() - max_age
    logger.info("Validating wx.minute_obs recency since %s", cutoff.isoformat())

    for city, cfg in sorted(CITY_CONFIG.items()):
        loc_id = cfg["loc_id"]
        latest = session.execute(
            text("SELECT MAX(ts_utc) FROM wx.minute_obs WHERE loc_id = :loc_id"),
            {"loc_id": loc_id},
        ).scalar()
        if not latest:
            logger.error("No weather minutes found for %s (%s)", city, loc_id)
            ok = False
            continue
        latest_utc = coerce_datetime_to_utc(latest)
        logger.info("%s latest ts_utc: %s", loc_id, latest_utc.isoformat())
        if latest_utc < cutoff:
            logger.error("%s (%s) stale by %.1f hours", loc_id, city, (cutoff - latest_utc).total_seconds() / 3600)
            ok = False
    return ok


def run_checks(max_age_hours: int) -> bool:
    max_age = timedelta(hours=max_age_hours)
    with get_session() as session:
        ok = True
        ok &= _check_connection(session)
        ok &= _check_alembic(session)
        ok &= _check_series_recency(session, max_age)
        ok &= _check_weather_recency(session, max_age)
        return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DB recency and schema state")
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=36,
        help="Fail if latest data is older than this many hours (default: 36)",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    success = run_checks(args.max_age_hours)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
