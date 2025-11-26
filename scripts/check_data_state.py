#!/usr/bin/env python3
"""
Check current data state across all tables.

Usage:
    python scripts/check_data_state.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from src.db import get_engine


def check_data_state():
    """Print comprehensive data state summary."""
    engine = get_engine()

    with engine.connect() as conn:
        print("=" * 70)
        print("KALSHI WEATHER PIPELINE - DATA STATE SUMMARY")
        print("=" * 70)

        # Markets
        print("\n" + "-" * 70)
        print("KALSHI MARKETS BY CITY")
        print("-" * 70)
        result = conn.execute(
            text("""
            SELECT
                city,
                COUNT(*) as markets,
                MIN(event_date) as earliest_date,
                MAX(event_date) as latest_date,
                COUNT(DISTINCT event_date) as distinct_dates
            FROM kalshi.markets
            WHERE city IS NOT NULL
            GROUP BY city
            ORDER BY city
        """)
        )
        rows = list(result)
        if rows:
            print(f"{'City':<15} {'Markets':>8} {'Days':>6} {'Date Range':<25}")
            print("-" * 55)
            for row in rows:
                date_range = f"{row.earliest_date} to {row.latest_date}"
                print(
                    f"{row.city:<15} {row.markets:>8} {row.distinct_dates:>6} {date_range:<25}"
                )
        else:
            print("No markets data found.")

        # Candles by source
        print("\n" + "-" * 70)
        print("KALSHI CANDLES BY CITY AND SOURCE")
        print("-" * 70)
        result = conn.execute(
            text("""
            SELECT
                m.city,
                c.source,
                COUNT(*) as candle_count,
                MIN(c.bucket_start)::date as earliest,
                MAX(c.bucket_start)::date as latest
            FROM kalshi.candles_1m c
            JOIN kalshi.markets m ON c.ticker = m.ticker
            WHERE m.city IS NOT NULL
            GROUP BY m.city, c.source
            ORDER BY m.city, c.source
        """)
        )
        rows = list(result)
        if rows:
            print(f"{'City':<15} {'Source':<12} {'Count':>12} {'Date Range':<25}")
            print("-" * 65)
            for row in rows:
                date_range = f"{row.earliest} to {row.latest}"
                print(
                    f"{row.city:<15} {row.source:<12} {row.candle_count:>12,} {date_range:<25}"
                )
        else:
            print("No candles data found.")

        # Weather observations
        print("\n" + "-" * 70)
        print("WEATHER OBSERVATIONS BY STATION (5-min Visual Crossing)")
        print("-" * 70)
        result = conn.execute(
            text("""
            SELECT
                loc_id,
                COUNT(*) as obs_count,
                MIN(ts_utc)::date as earliest,
                MAX(ts_utc)::date as latest,
                COUNT(DISTINCT DATE(ts_utc)) as distinct_days
            FROM wx.minute_obs
            GROUP BY loc_id
            ORDER BY loc_id
        """)
        )
        rows = list(result)
        if rows:
            print(
                f"{'Station':<10} {'Observations':>12} {'Days':>6} {'Date Range':<25}"
            )
            print("-" * 55)
            for row in rows:
                date_range = f"{row.earliest} to {row.latest}"
                print(
                    f"{row.loc_id:<10} {row.obs_count:>12,} {row.distinct_days:>6} {date_range:<25}"
                )
        else:
            print("No weather observations found.")

        # Settlement data
        print("\n" + "-" * 70)
        print("SETTLEMENT DATA BY CITY (NWS Official Tmax)")
        print("-" * 70)
        result = conn.execute(
            text("""
            SELECT
                city,
                COUNT(*) as days,
                MIN(date_local) as earliest,
                MAX(date_local) as latest
            FROM wx.settlement
            GROUP BY city
            ORDER BY city
        """)
        )
        rows = list(result)
        if rows:
            print(f"{'City':<15} {'Days':>6} {'Date Range':<25}")
            print("-" * 45)
            for row in rows:
                date_range = f"{row.earliest} to {row.latest}"
                print(f"{row.city:<15} {row.days:>6} {date_range:<25}")
        else:
            print("No settlement data found.")

        # Totals
        print("\n" + "-" * 70)
        print("TOTALS")
        print("-" * 70)
        result = conn.execute(text("SELECT COUNT(*) FROM kalshi.markets"))
        print(f"Total markets:      {result.scalar():>12,}")

        result = conn.execute(text("SELECT COUNT(*) FROM kalshi.candles_1m"))
        print(f"Total candles:      {result.scalar():>12,}")

        result = conn.execute(text("SELECT COUNT(*) FROM wx.minute_obs"))
        print(f"Total weather obs:  {result.scalar():>12,}")

        result = conn.execute(text("SELECT COUNT(*) FROM wx.settlement"))
        print(f"Total settlements:  {result.scalar():>12,}")

        # Gap check for candles
        print("\n" + "-" * 70)
        print("DATA GAPS CHECK")
        print("-" * 70)

        # Check for cities with no data
        expected_cities = ["austin", "chicago", "denver", "los_angeles", "miami", "philadelphia"]
        result = conn.execute(
            text("""
            SELECT DISTINCT city FROM kalshi.markets WHERE city IS NOT NULL
        """)
        )
        cities_with_data = {row.city for row in result}
        missing_cities = set(expected_cities) - cities_with_data

        if missing_cities:
            print(f"Cities with NO market data: {', '.join(sorted(missing_cities))}")
        else:
            print("All 6 cities have market data.")

        # Check weather station coverage
        expected_stations = ["KAUS", "KMDW", "KDEN", "KLAX", "KMIA", "KPHL"]
        result = conn.execute(
            text("SELECT DISTINCT loc_id FROM wx.minute_obs")
        )
        stations_with_data = {row.loc_id for row in result}
        missing_stations = set(expected_stations) - stations_with_data

        if missing_stations:
            print(f"Stations with NO weather data: {', '.join(sorted(missing_stations))}")
        else:
            print("All 6 stations have weather data.")

        print("\n" + "=" * 70)
        print("END OF DATA STATE SUMMARY")
        print("=" * 70)


if __name__ == "__main__":
    check_data_state()
