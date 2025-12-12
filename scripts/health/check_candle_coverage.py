#!/usr/bin/env python3
"""
Diagnostics: validate 1-minute candle coverage for Kalshi weather markets.

Default usage checks Chicago for 2025 and reports missing minutes per ticker
without refetching anything. Exits with code 1 only if --fail-on-missing is
passed and gaps are detected.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Optional

import psycopg2


DSN_DEFAULT = "postgresql://kalshi:kalshi@localhost:5434/kalshi_weather"


@dataclass
class MarketRow:
    ticker: str
    event_date: date
    listed_at: Optional["datetime"]
    close_time: Optional["datetime"]


@dataclass
class CoverageReport:
    ticker: str
    event_date: date
    missing_minutes: int
    expected_minutes: int
    actual_minutes: int
    first_ts: Optional["datetime"]
    last_ts: Optional["datetime"]


def minute_gaps(timestamps: Iterable["datetime"]) -> int:
    """Count missing minutes between consecutive timestamps."""
    missing = 0
    prev = None
    for ts in sorted(timestamps):
        if prev is not None:
            delta = int((ts - prev).total_seconds() // 60) - 1
            if delta > 0:
                missing += delta
        prev = ts
    return missing


def analyze_coverage(
    dsn: str,
    city: str,
    start_date: date,
    end_date: date,
) -> list[CoverageReport]:
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT m.ticker, m.event_date, m.listed_at, m.close_time
        FROM kalshi.markets m
        WHERE m.city = %s AND m.event_date BETWEEN %s AND %s
        """,
        (city, start_date, end_date),
    )
    markets = [MarketRow(*row) for row in cur.fetchall()]

    cur.execute(
        """
        SELECT c.ticker, c.bucket_start
        FROM kalshi.candles_1m c
        JOIN kalshi.markets m ON c.ticker = m.ticker
        WHERE m.city = %s AND m.event_date BETWEEN %s AND %s
        """,
        (city, start_date, end_date),
    )
    candles_by_ticker: dict[str, list] = defaultdict(list)
    for ticker, ts in cur.fetchall():
        candles_by_ticker[ticker].append(ts)

    conn.close()

    reports: list[CoverageReport] = []
    for market in markets:
        ts_list = candles_by_ticker.get(market.ticker, [])
        if not ts_list:
            reports.append(
                CoverageReport(
                    ticker=market.ticker,
                    event_date=market.event_date,
                    missing_minutes=0,
                    expected_minutes=0,
                    actual_minutes=0,
                    first_ts=None,
                    last_ts=None,
                )
            )
            continue

        start_ts = (market.listed_at or ts_list[0]).replace(second=0, microsecond=0)
        end_ts = (market.close_time or ts_list[-1]).replace(second=0, microsecond=0)
        expected = int(((end_ts - start_ts).total_seconds() // 60) + 1)
        actual = len(set(ts_list))
        missing = minute_gaps(ts_list) + max(0, expected - actual)
        reports.append(
            CoverageReport(
                ticker=market.ticker,
                event_date=market.event_date,
                missing_minutes=missing,
                expected_minutes=expected,
                actual_minutes=actual,
                first_ts=min(ts_list),
                last_ts=max(ts_list),
            )
        )

    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Kalshi candle coverage.")
    parser.add_argument("--dsn", default=DSN_DEFAULT, help="Postgres DSN")
    parser.add_argument("--city", default="chicago", help="City id (default: chicago)")
    parser.add_argument(
        "--start-date",
        default="2025-01-01",
        help="Start date YYYY-MM-DD (default: 2025-01-01)",
    )
    parser.add_argument(
        "--end-date",
        default="2025-12-31",
        help="End date YYYY-MM-DD (default: 2025-12-31)",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit 1 if any missing minutes are detected.",
    )
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)

    reports = analyze_coverage(args.dsn, args.city, start_date, end_date)
    total_markets = len(reports)
    missing_markets = [r for r in reports if r.missing_minutes > 0]
    worst = sorted(missing_markets, key=lambda r: r.missing_minutes, reverse=True)[:5]

    print(f"City={args.city} window={start_date}..{end_date}")
    print(f"Markets analyzed: {total_markets}")
    print(f"Markets with missing minutes: {len(missing_markets)}")
    for r in worst:
        print(
            f"{r.ticker} {r.event_date}: missing {r.missing_minutes} "
            f"of {r.expected_minutes} (actual {r.actual_minutes}), "
            f"first={r.first_ts}, last={r.last_ts}"
        )

    if args.fail_on_missing and missing_markets:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
