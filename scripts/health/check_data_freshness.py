#!/usr/bin/env python3
"""
Data freshness health-check script.

Checks all critical data tables and reports the latest data timestamp
for each city. Used to verify that all ingestion pipelines are working.

Checks:
- wx.settlement (TMAX settlement data)
- wx.forecast_snapshot (daily forecasts)
- wx.forecast_snapshot_hourly (hourly forecast curves)
- wx.minute_obs (5-minute observations)
- kalshi.markets (market metadata)
- kalshi.candles_1m (minute-level price data)
- kalshi.ws_raw (WebSocket raw data - if available)

Usage:
    python scripts/check_data_freshness.py
    python scripts/check_data_freshness.py --json  # Output as JSON
    python scripts/check_data_freshness.py --city chicago
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select, text

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import CITIES
from src.db import (
    get_db_session,
    KalshiCandle1m,
    KalshiMarket,
    KalshiWsRaw,
    WxForecastSnapshot,
    WxForecastSnapshotHourly,
    WxMinuteObs,
    WxSettlement,
)


def check_settlement_freshness(session, city_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Check wx.settlement freshness per city."""
    query = (
        select(
            WxSettlement.city,
            func.min(WxSettlement.date_local).label("min_date"),
            func.max(WxSettlement.date_local).label("max_date"),
            func.count(WxSettlement.city).label("row_count"),
        )
        .group_by(WxSettlement.city)
        .order_by(WxSettlement.city)
    )

    if city_filter:
        query = query.where(WxSettlement.city == city_filter)

    results = session.execute(query).all()
    return [
        {
            "city": r.city,
            "min_date": r.min_date.isoformat() if r.min_date else None,
            "max_date": r.max_date.isoformat() if r.max_date else None,
            "row_count": r.row_count,
        }
        for r in results
    ]


def check_daily_forecast_freshness(session, city_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Check wx.forecast_snapshot freshness per city."""
    query = (
        select(
            WxForecastSnapshot.city,
            func.min(WxForecastSnapshot.basis_date).label("min_basis"),
            func.max(WxForecastSnapshot.basis_date).label("max_basis"),
            func.min(WxForecastSnapshot.target_date).label("min_target"),
            func.max(WxForecastSnapshot.target_date).label("max_target"),
            func.count(WxForecastSnapshot.city).label("row_count"),
        )
        .group_by(WxForecastSnapshot.city)
        .order_by(WxForecastSnapshot.city)
    )

    if city_filter:
        query = query.where(WxForecastSnapshot.city == city_filter)

    results = session.execute(query).all()
    return [
        {
            "city": r.city,
            "min_basis": r.min_basis.isoformat() if r.min_basis else None,
            "max_basis": r.max_basis.isoformat() if r.max_basis else None,
            "min_target": r.min_target.isoformat() if r.min_target else None,
            "max_target": r.max_target.isoformat() if r.max_target else None,
            "row_count": r.row_count,
        }
        for r in results
    ]


def check_hourly_forecast_freshness(session, city_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Check wx.forecast_snapshot_hourly freshness per city."""
    query = (
        select(
            WxForecastSnapshotHourly.city,
            func.min(WxForecastSnapshotHourly.basis_date).label("min_basis"),
            func.max(WxForecastSnapshotHourly.basis_date).label("max_basis"),
            func.count(WxForecastSnapshotHourly.city).label("row_count"),
        )
        .group_by(WxForecastSnapshotHourly.city)
        .order_by(WxForecastSnapshotHourly.city)
    )

    if city_filter:
        query = query.where(WxForecastSnapshotHourly.city == city_filter)

    results = session.execute(query).all()
    return [
        {
            "city": r.city,
            "min_basis": r.min_basis.isoformat() if r.min_basis else None,
            "max_basis": r.max_basis.isoformat() if r.max_basis else None,
            "row_count": r.row_count,
        }
        for r in results
    ]


def check_minute_obs_freshness(session, city_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Check wx.minute_obs freshness per station."""
    query = (
        select(
            WxMinuteObs.loc_id,
            func.min(WxMinuteObs.ts_utc).label("min_ts"),
            func.max(WxMinuteObs.ts_utc).label("max_ts"),
            func.count(WxMinuteObs.loc_id).label("row_count"),
        )
        .group_by(WxMinuteObs.loc_id)
        .order_by(WxMinuteObs.loc_id)
    )

    # Filter by ICAO if city specified
    if city_filter and city_filter in CITIES:
        icao = CITIES[city_filter].icao
        query = query.where(WxMinuteObs.loc_id == icao)

    results = session.execute(query).all()
    return [
        {
            "loc_id": r.loc_id,
            "min_ts": r.min_ts.isoformat() if r.min_ts else None,
            "max_ts": r.max_ts.isoformat() if r.max_ts else None,
            "row_count": r.row_count,
        }
        for r in results
    ]


def check_markets_freshness(session, city_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Check kalshi.markets freshness per city."""
    query = (
        select(
            KalshiMarket.city,
            func.min(KalshiMarket.event_date).label("min_event_date"),
            func.max(KalshiMarket.event_date).label("max_event_date"),
            func.count(KalshiMarket.ticker).label("row_count"),
        )
        .where(KalshiMarket.city.isnot(None))
        .group_by(KalshiMarket.city)
        .order_by(KalshiMarket.city)
    )

    if city_filter:
        query = query.where(KalshiMarket.city == city_filter)

    results = session.execute(query).all()
    return [
        {
            "city": r.city,
            "min_event_date": r.min_event_date.isoformat() if r.min_event_date else None,
            "max_event_date": r.max_event_date.isoformat() if r.max_event_date else None,
            "row_count": r.row_count,
        }
        for r in results
    ]


def check_candles_freshness(session, city_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Check kalshi.candles_1m freshness per city (via markets join)."""
    # Use raw SQL for this complex join
    sql = """
    SELECT m.city,
           MIN(c.bucket_start) AS min_bucket,
           MAX(c.bucket_start) AS max_bucket,
           COUNT(*) AS row_count
    FROM kalshi.candles_1m c
    JOIN kalshi.markets m ON c.ticker = m.ticker
    WHERE m.city IS NOT NULL
    """

    if city_filter:
        sql += f" AND m.city = '{city_filter}'"

    sql += """
    GROUP BY m.city
    ORDER BY m.city
    """

    results = session.execute(text(sql)).all()
    return [
        {
            "city": r.city,
            "min_bucket": r.min_bucket.isoformat() if r.min_bucket else None,
            "max_bucket": r.max_bucket.isoformat() if r.max_bucket else None,
            "row_count": r.row_count,
        }
        for r in results
    ]


def check_ws_raw_freshness(session) -> Dict[str, Any]:
    """Check kalshi.ws_raw freshness (overall, not per city)."""
    query = select(
        func.min(KalshiWsRaw.ts_utc).label("min_ts"),
        func.max(KalshiWsRaw.ts_utc).label("max_ts"),
        func.count(KalshiWsRaw.id).label("row_count"),
    )

    result = session.execute(query).one()
    return {
        "min_ts": result.min_ts.isoformat() if result.min_ts else None,
        "max_ts": result.max_ts.isoformat() if result.max_ts else None,
        "row_count": result.row_count,
    }


def format_freshness_report(data: Dict[str, Any], today: date) -> str:
    """Format freshness data as human-readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"DATA FRESHNESS REPORT - {today.isoformat()}")
    lines.append("=" * 70)

    # Settlement data
    lines.append("\n--- wx.settlement (TMAX data) ---")
    for row in data.get("settlement", []):
        max_date = row.get("max_date", "N/A")
        days_old = "N/A"
        if max_date and max_date != "N/A":
            days_old = (today - date.fromisoformat(max_date)).days
        lines.append(
            f"  {row['city']:15} | {row.get('min_date', 'N/A')} to {max_date} | "
            f"{row['row_count']:,} rows | {days_old}d old"
        )

    # Daily forecasts
    lines.append("\n--- wx.forecast_snapshot (daily forecasts) ---")
    for row in data.get("daily_forecast", []):
        max_basis = row.get("max_basis", "N/A")
        days_old = "N/A"
        if max_basis and max_basis != "N/A":
            days_old = (today - date.fromisoformat(max_basis)).days
        lines.append(
            f"  {row['city']:15} | basis: {row.get('min_basis', 'N/A')} to {max_basis} | "
            f"{row['row_count']:,} rows | {days_old}d old"
        )

    # Hourly forecasts
    lines.append("\n--- wx.forecast_snapshot_hourly (72h curves) ---")
    for row in data.get("hourly_forecast", []):
        max_basis = row.get("max_basis", "N/A")
        days_old = "N/A"
        if max_basis and max_basis != "N/A":
            days_old = (today - date.fromisoformat(max_basis)).days
        lines.append(
            f"  {row['city']:15} | basis: {row.get('min_basis', 'N/A')} to {max_basis} | "
            f"{row['row_count']:,} rows | {days_old}d old"
        )

    # Minute observations
    lines.append("\n--- wx.minute_obs (5-min observations) ---")
    for row in data.get("minute_obs", []):
        max_ts = row.get("max_ts", "N/A")
        hours_old = "N/A"
        if max_ts and max_ts != "N/A":
            max_dt = datetime.fromisoformat(max_ts.replace("Z", "+00:00"))
            hours_old = int((datetime.now(timezone.utc) - max_dt).total_seconds() / 3600)
        lines.append(
            f"  {row['loc_id']:15} | {row.get('min_ts', 'N/A')[:10]} to {max_ts[:10] if max_ts else 'N/A'} | "
            f"{row['row_count']:,} rows | {hours_old}h old"
        )

    # Markets
    lines.append("\n--- kalshi.markets (market metadata) ---")
    for row in data.get("markets", []):
        max_date = row.get("max_event_date", "N/A")
        days_ahead = "N/A"
        if max_date and max_date != "N/A":
            days_ahead = (date.fromisoformat(max_date) - today).days
        lines.append(
            f"  {row['city']:15} | {row.get('min_event_date', 'N/A')} to {max_date} | "
            f"{row['row_count']:,} markets | {days_ahead}d ahead"
        )

    # Candles
    lines.append("\n--- kalshi.candles_1m (minute OHLC) ---")
    for row in data.get("candles", []):
        max_bucket = row.get("max_bucket", "N/A")
        hours_old = "N/A"
        if max_bucket and max_bucket != "N/A":
            max_dt = datetime.fromisoformat(max_bucket.replace("Z", "+00:00"))
            hours_old = int((datetime.now(timezone.utc) - max_dt).total_seconds() / 3600)
        lines.append(
            f"  {row['city']:15} | {row.get('min_bucket', 'N/A')[:10] if row.get('min_bucket') else 'N/A'} to "
            f"{max_bucket[:10] if max_bucket else 'N/A'} | {row['row_count']:,} candles | {hours_old}h old"
        )

    # WebSocket raw
    lines.append("\n--- kalshi.ws_raw (WebSocket data) ---")
    ws_data = data.get("ws_raw", {})
    if ws_data.get("row_count", 0) > 0:
        max_ts = ws_data.get("max_ts", "N/A")
        mins_old = "N/A"
        if max_ts and max_ts != "N/A":
            max_dt = datetime.fromisoformat(max_ts.replace("Z", "+00:00"))
            mins_old = int((datetime.now(timezone.utc) - max_dt).total_seconds() / 60)
        lines.append(
            f"  Total: {ws_data['row_count']:,} messages | last: {max_ts} | {mins_old}m old"
        )
    else:
        lines.append("  No WebSocket data recorded yet")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Check data freshness across all tables"
    )
    parser.add_argument(
        "--city", type=str,
        help="Filter to single city"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON instead of table"
    )

    args = parser.parse_args()

    today = date.today()

    with get_db_session() as session:
        data = {
            "check_time": datetime.now(timezone.utc).isoformat(),
            "settlement": check_settlement_freshness(session, args.city),
            "daily_forecast": check_daily_forecast_freshness(session, args.city),
            "hourly_forecast": check_hourly_forecast_freshness(session, args.city),
            "minute_obs": check_minute_obs_freshness(session, args.city),
            "markets": check_markets_freshness(session, args.city),
            "candles": check_candles_freshness(session, args.city),
            "ws_raw": check_ws_raw_freshness(session),
        }

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(format_freshness_report(data, today))


if __name__ == "__main__":
    main()
