#!/usr/bin/env python3
"""Check current database state for expansion planning."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

DB_URL = os.getenv("DB_URL", "postgresql://kalshi:kalshi@localhost:5444/kalshi")
engine = create_engine(DB_URL)

print("=" * 80)
print("MARKETS TABLE - Chicago High Temperature (KXHIGHCHI)")
print("=" * 80)

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT
            COUNT(*) as total_markets,
            MIN(close_time) as earliest_close,
            MAX(close_time) as latest_close,
            MIN(open_time) as earliest_open,
            MAX(open_time) as latest_open
        FROM markets
        WHERE series_ticker = 'KXHIGHCHI'
    """))
    row = result.fetchone()
    print(f"Total Markets: {row[0]}")
    print(f"Earliest Open: {row[3]}")
    print(f"Latest Open: {row[4]}")
    print(f"Earliest Close: {row[1]}")
    print(f"Latest Close: {row[2]}")

    if row[0] > 0:
        days_span = (row[2] - row[1]).days
        print(f"Date Span: {days_span} days")

print("\n" + "=" * 80)
print("CANDLES TABLE - Chicago Markets")
print("=" * 80)

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT
            COUNT(*) as total_candles,
            MIN(timestamp) as earliest_candle,
            MAX(timestamp) as latest_candle
        FROM candles
        WHERE market_ticker LIKE 'KXHIGHCHI%'
    """))
    row = result.fetchone()
    print(f"Total Candles: {row[0]}")
    print(f"Earliest Candle: {row[1]}")
    print(f"Latest Candle: {row[2]}")

print("\n" + "=" * 80)
print("CANDLES BY PERIOD")
print("=" * 80)

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT
            period_minutes,
            COUNT(*) as count,
            MIN(timestamp) as earliest,
            MAX(timestamp) as latest
        FROM candles
        WHERE market_ticker LIKE 'KXHIGHCHI%'
        GROUP BY period_minutes
        ORDER BY period_minutes
    """))
    for row in result:
        print(f"{row[0]}-minute: {row[1]:,} candles (range: {row[2]} to {row[3]})")

print("\n" + "=" * 80)
print("WEATHER OBSERVED - Chicago Midway (GHCND:USW00014819)")
print("=" * 80)

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT
            COUNT(*) as total_obs,
            MIN(date) as earliest_date,
            MAX(date) as latest_date
        FROM weather_observed
        WHERE station_id = 'GHCND:USW00014819'
    """))
    row = result.fetchone()
    print(f"Total Observations: {row[0]}")
    print(f"Earliest Date: {row[1]}")
    print(f"Latest Date: {row[2]}")

    if row[0] > 0:
        days_span = (row[2] - row[1]).days
        print(f"Date Span: {days_span} days")

print("\n" + "=" * 80)
print("VISUAL CROSSING WEATHER (wx schema)")
print("=" * 80)

with engine.connect() as conn:
    # Check if wx schema exists
    result = conn.execute(text("""
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name = 'wx'
    """))
    if result.fetchone():
        result = conn.execute(text("""
            SELECT
                loc_id,
                COUNT(*) as total_obs,
                MIN(ts_utc) as earliest,
                MAX(ts_utc) as latest
            FROM wx.minute_obs
            GROUP BY loc_id
            ORDER BY loc_id
        """))
        rows = result.fetchall()
        if rows:
            for row in rows:
                print(f"{row[0]}: {row[1]:,} observations (range: {row[2]} to {row[3]})")
        else:
            print("No Visual Crossing data found in wx.minute_obs")
    else:
        print("wx schema does not exist")

print("\n" + "=" * 80)
print("GAPS IN DATA")
print("=" * 80)

with engine.connect() as conn:
    # Find markets without candles
    result = conn.execute(text("""
        SELECT
            m.ticker,
            m.close_time
        FROM markets m
        LEFT JOIN candles c ON m.ticker = c.market_ticker
        WHERE m.series_ticker = 'KXHIGHCHI'
        AND c.market_ticker IS NULL
        ORDER BY m.close_time
        LIMIT 10
    """))
    rows = result.fetchall()
    if rows:
        print(f"Markets without candles: {len(rows)} (showing first 10)")
        for row in rows:
            print(f"  {row[0]}: closed {row[1]}")
    else:
        print("No markets without candles")
