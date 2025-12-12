#!/usr/bin/env python3
"""
Audit data coverage for market-clock snapshots (D-1 10:00 -> D 23:55) at 5-minute resolution.

Checks (per target date):
- Observations: row counts vs expected 5-min slots, null rates for key meteo fields.
- Historical forecasts: lead_days coverage (0-14) for daily/hourly/minute feeds.
- Live forecasts: most recent basis timestamp and age.
- Kalshi candles: 1-minute coverage in the market-clock window.

Default city: austin. Fully parameterized to run on any supported city.

Usage:
    PYTHONPATH=. python scripts/audit_data_coverage.py --city austin --start-date 2024-10-01 --end-date 2024-10-07
"""

import argparse
import logging
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

# Local imports
from src.config.cities import get_city
from src.db import (
    VcForecastDaily,
    VcForecastHourly,
    VcLocation,
    VcMinuteWeather,
    KalshiCandle1m,
    get_db_session,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def daterange(start: date, end: date) -> Iterable[date]:
    """Inclusive date range generator."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def get_vc_location_id(session: Session, city_code: str, location_type: str = "station") -> Optional[int]:
    """Resolve vc_location id for a city code and location type."""
    row = (
        session.execute(
            select(VcLocation.id).where(
                VcLocation.city_code == city_code,
                VcLocation.location_type == location_type,
            )
        )
        .scalars()
        .first()
    )
    return row


def compute_expected_slots(window_start: datetime, window_end: datetime, step_minutes: int = 5) -> int:
    """Expected number of samples in a window at fixed step."""
    delta_minutes = int((window_end - window_start).total_seconds() // 60)
    return delta_minutes // step_minutes + 1


def print_header(title: str) -> None:
    logger.info("\n" + "=" * 80)
    logger.info(title)
    logger.info("=" * 80)


# -----------------------------------------------------------------------------
# Audits
# -----------------------------------------------------------------------------

def audit_observations(
    session: Session,
    city_id: str,
    start_date: date,
    end_date: date,
    location_type: str = "station",
) -> None:
    """Check obs coverage for each target date in window."""
    city = get_city(city_id)
    vc_location_id = get_vc_location_id(session, city.city_code, location_type)
    if vc_location_id is None:
        logger.warning(f"No vc_location for {city_id} ({location_type})")
        return

    print_header(f"OBS COVERAGE ({city_id}, {location_type})")
    tz = ZoneInfo(city.timezone)

    records = []
    for target_date in daterange(start_date, end_date):
        window_start = datetime.combine(target_date - timedelta(days=1), time(10, 0))
        window_end = datetime.combine(target_date, time(23, 55))

        q = (
            select(
                func.count().label("n_rows"),
                func.count(VcMinuteWeather.temp_f).label("n_temp"),
                func.count(VcMinuteWeather.humidity).label("n_humidity"),
                func.count(VcMinuteWeather.cloudcover).label("n_cloudcover"),
                func.count(VcMinuteWeather.windspeed_mph).label("n_wind"),
                func.min(VcMinuteWeather.datetime_local).label("min_ts"),
                func.max(VcMinuteWeather.datetime_local).label("max_ts"),
            )
            .where(
                VcMinuteWeather.vc_location_id == vc_location_id,
                VcMinuteWeather.data_type == "actual_obs",
                VcMinuteWeather.datetime_local >= window_start,
                VcMinuteWeather.datetime_local <= window_end,
            )
        )
        res = session.execute(q).one()
        expected = compute_expected_slots(window_start, window_end, step_minutes=5)
        coverage = (res.n_rows / expected) * 100 if expected else 0.0

        records.append(
            {
                "day": target_date,
                "expected_5min": expected,
                "rows": res.n_rows,
                "coverage_pct": round(coverage, 1),
                "temp_nonnull": res.n_temp,
                "humidity_nonnull": res.n_humidity,
                "cloudcover_nonnull": res.n_cloudcover,
                "wind_nonnull": res.n_wind,
                "min_ts": res.min_ts,
                "max_ts": res.max_ts,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        logger.info("No observation data found.")
        return

    logger.info(df.to_string(index=False))
    logger.info(
        "Obs summary: avg coverage=%.1f%%, temp null rate=%.1f%%",
        df["coverage_pct"].mean(),
        100 * (1 - df["temp_nonnull"].sum() / df["rows"].sum()) if df["rows"].sum() else 0,
    )


def audit_forecast_daily_hourly(
    session: Session,
    city_id: str,
    start_date: date,
    end_date: date,
    location_type: str = "station",
    max_lead_days: int = 14,
) -> None:
    """Check daily/hourly forecast lead coverage for each target date."""
    city = get_city(city_id)
    vc_location_id = get_vc_location_id(session, city.city_code, location_type)
    if vc_location_id is None:
        logger.warning(f"No vc_location for {city_id} ({location_type})")
        return

    print_header(f"HISTORICAL FORECAST DAILY/HOURLY ({city_id}, leads 0-{max_lead_days})")

    rows = []
    for target_date in daterange(start_date, end_date):
        # Daily leads
        daily_leads = (
            session.execute(
                select(VcForecastDaily.lead_days)
                .where(
                    VcForecastDaily.vc_location_id == vc_location_id,
                    VcForecastDaily.target_date == target_date,
                    VcForecastDaily.data_type == "historical_forecast",
                    VcForecastDaily.lead_days <= max_lead_days,
                )
                .distinct()
            )
            .scalars()
            .all()
        )
        missing_daily = sorted(set(range(0, max_lead_days + 1)) - set(daily_leads))

        # Hourly existence (derive lead_days from lead_hours)
        window_start = datetime.combine(target_date, time.min)
        window_end = datetime.combine(target_date, time.max)
        hourly_q = select(
            VcForecastHourly.lead_hours
        ).where(
            VcForecastHourly.vc_location_id == vc_location_id,
            VcForecastHourly.data_type == "historical_forecast",
            VcForecastHourly.target_datetime_local >= window_start,
            VcForecastHourly.target_datetime_local <= window_end,
        )
        hourly_df = pd.read_sql(hourly_q, session.bind)
        hourly_leads = sorted(
            set(
                int(h // 24)
                for h in hourly_df["lead_hours"].dropna().tolist()
                if h >= 0 and (h // 24) <= max_lead_days
            )
        )
        missing_hourly = sorted(set(range(0, max_lead_days + 1)) - set(hourly_leads))

        rows.append(
            {
                "day": target_date,
                "daily_leads_present": len(daily_leads),
                "daily_missing": missing_daily,
                "hourly_leads_present": len(hourly_leads),
                "hourly_missing": missing_hourly,
            }
        )

    df = pd.DataFrame(rows)
    logger.info(df.to_string(index=False))


def audit_forecast_minutes(
    session: Session,
    city_id: str,
    start_date: date,
    end_date: date,
    location_type: str = "station",
) -> None:
    """Check presence of minute-level historical forecasts (usually leads 0-1)."""
    city = get_city(city_id)
    vc_location_id = get_vc_location_id(session, city.city_code, location_type)
    if vc_location_id is None:
        logger.warning(f"No vc_location for {city_id} ({location_type})")
        return

    print_header(f"HISTORICAL FORECAST MINUTES ({city_id})")

    records = []
    for target_date in daterange(start_date, end_date):
        day_start = datetime.combine(target_date, time.min)
        day_end = datetime.combine(target_date, time.max)

        q = select(
            func.min(VcMinuteWeather.lead_hours).label("min_lead_hours"),
            func.max(VcMinuteWeather.lead_hours).label("max_lead_hours"),
            func.count().label("n_rows"),
        ).where(
            VcMinuteWeather.vc_location_id == vc_location_id,
            VcMinuteWeather.data_type == "historical_forecast",
            VcMinuteWeather.datetime_local >= day_start,
            VcMinuteWeather.datetime_local <= day_end,
        )

        res = session.execute(q).one()
        records.append(
            {
                "day": target_date,
                "rows": res.n_rows,
                "min_lead_hours": res.min_lead_hours,
                "max_lead_hours": res.max_lead_hours,
            }
        )

    df = pd.DataFrame(records)
    logger.info(df.to_string(index=False))


def audit_live_forecast_snapshot(session: Session, city_id: str, location_type: str = "station") -> None:
    """Report most recent live forecast basis timestamp for the city."""
    city = get_city(city_id)
    vc_location_id = get_vc_location_id(session, city.city_code, location_type)
    if vc_location_id is None:
        logger.warning(f"No vc_location for {city_id} ({location_type})")
        return

    print_header(f"LATEST LIVE FORECAST SNAPSHOT ({city_id})")

    q = (
        select(
            func.max(VcForecastDaily.forecast_basis_datetime_utc).label("latest_basis_utc"),
            func.max(VcForecastDaily.created_at).label("latest_ingest_ts"),
        )
        .where(
            VcForecastDaily.vc_location_id == vc_location_id,
            VcForecastDaily.data_type == "forecast",
        )
    )
    res = session.execute(q).one()
    logger.info("Latest basis UTC: %s | latest ingest_ts: %s", res.latest_basis_utc, res.latest_ingest_ts)


def audit_candles(
    session: Session,
    city_id: str,
    start_date: date,
    end_date: date,
    cutoff_minute: int = 55,
) -> None:
    """Check 1-minute Kalshi candle coverage for market-clock window."""
    city = get_city(city_id)
    print_header(f"KALSHI CANDLE COVERAGE ({city_id})")

    # Prefer dense table if it exists; fallback to sparse candles_1m
    conn = session.connection()
    has_dense = False
    try:
        res = conn.execute(text("SELECT to_regclass('kalshi.candles_1m_dense')")).scalar()
        has_dense = res is not None
    except Exception:
        has_dense = False

    rows = []
    for target_date in daterange(start_date, end_date):
        window_start = datetime.combine(target_date - timedelta(days=1), time(10, 0))
        window_end = datetime.combine(target_date, time(23, cutoff_minute))

        if has_dense:
            q = text(
                """
                SELECT
                    COUNT(*) AS n_rows,
                    MIN(bucket_start) AS min_ts,
                    MAX(bucket_start) AS max_ts
                FROM kalshi.candles_1m_dense
                WHERE ticker LIKE :ticker_pattern
                  AND bucket_start >= :window_start
                  AND bucket_start <= :window_end
                """
            )
            res = conn.execute(
                q,
                {
                    "ticker_pattern": f"%{city.kalshi_code}%",
                    "window_start": window_start,
                    "window_end": window_end,
                },
            ).one()
            n_rows, min_ts, max_ts = res.n_rows, res.min_ts, res.max_ts
        else:
            q = (
                select(
                    func.count().label("n_rows"),
                    func.min(KalshiCandle1m.bucket_start).label("min_ts"),
                    func.max(KalshiCandle1m.bucket_start).label("max_ts"),
                )
                .where(
                    KalshiCandle1m.bucket_start >= window_start,
                    KalshiCandle1m.bucket_start <= window_end,
                    KalshiCandle1m.ticker.like(f"%{city.kalshi_code}%"),
                )
            )
            res = session.execute(q).one()
            n_rows, min_ts, max_ts = res.n_rows, res.min_ts, res.max_ts

        expected = compute_expected_slots(window_start, window_end, step_minutes=1)
        rows.append(
            {
                "day": target_date,
                "expected_1min": expected,
                "rows": n_rows,
                "coverage_pct": round((n_rows / expected) * 100, 1) if expected else 0.0,
                "min_ts": min_ts,
                "max_ts": max_ts,
                "source": "dense" if has_dense else "sparse",
            }
        )

    df = pd.DataFrame(rows)
    logger.info(df.to_string(index=False))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit data coverage for market-clock snapshots.")
    parser.add_argument("--city", default="austin", help="city_id (e.g., austin, chicago, denver)")
    parser.add_argument("--start-date", required=True, type=lambda s: date.fromisoformat(s))
    parser.add_argument("--end-date", required=True, type=lambda s: date.fromisoformat(s))
    parser.add_argument("--location-type", default="station", choices=["station", "city"])
    parser.add_argument("--max-lead-days", type=int, default=14, help="max forecast lead days to audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with get_db_session() as session:
        audit_observations(session, args.city, args.start_date, args.end_date, args.location_type)
        audit_forecast_daily_hourly(
            session,
            args.city,
            args.start_date,
            args.end_date,
            args.location_type,
            max_lead_days=args.max_lead_days,
        )
        audit_forecast_minutes(session, args.city, args.start_date, args.end_date, args.location_type)
        audit_live_forecast_snapshot(session, args.city, args.location_type)
        audit_candles(session, args.city, args.start_date, args.end_date)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
