#!/usr/bin/env python3
"""
Test SQL joins between Kalshi candles and Visual Crossing weather data.

Tests:
1. 5-minute joins (primary for modeling)
2. 1-minute joins (for granular tweaking)
3. Multi-city joins
4. Edge cases
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from sqlalchemy import text
from db.connection import get_session

# City to station mapping
CITY_MAP = {
    'KXHIGHCHI': ('chicago', 'KMDW'),
    'KXHIGHMIA': ('miami', 'KMIA'),
    'KXHIGHAUS': ('austin', 'KAUS'),
    'KXHIGHLAX': ('los_angeles', 'KLAX'),
    'KXHIGHDEN': ('denver', 'KDEN'),
    'KXHIGHPHIL': ('philadelphia', 'KPHL'),
}

def test_5min_joins(session):
    """Test 5-minute joins (primary for backtest/model)."""
    print("\n" + "="*60)
    print("TEST 1: 5-MINUTE JOINS (Kalshi ↔ Weather)")
    print("="*60 + "\n")

    # Test last 3 days for all cities
    three_days_ago = (datetime.utcnow() - timedelta(days=3)).strftime('%Y-%m-%d')

    for series, (city, loc_id) in CITY_MAP.items():
        result = session.execute(text(f"""
            WITH joined AS (
                SELECT c.market_ticker, c.timestamp, c.close as price_cents,
                       w.temp_f, w.humidity,
                       CASE WHEN w.temp_f IS NULL THEN 'NO_WX' ELSE 'MATCHED' END as status
                FROM candles c
                LEFT JOIN wx.minute_obs w ON w.loc_id = '{loc_id}' AND w.ts_utc = c.timestamp
                WHERE c.period_minutes = 5
                  AND c.market_ticker LIKE '{series}-%'
                  AND c.timestamp >= '{three_days_ago}'
            )
            SELECT status, COUNT(*) as count,
                   ROUND(AVG(temp_f)::numeric, 1) as avg_temp_f
            FROM joined
            GROUP BY status
        """)).fetchall()

        total = sum(r[1] for r in result)
        matched = next((r[1] for r in result if r[0] == 'MATCHED'), 0)
        match_rate = (matched / total * 100) if total > 0 else 0

        avg_temp = next((r[2] for r in result if r[0] == 'MATCHED'), None)

        status_icon = "✓" if match_rate >= 90 else "⚠" if match_rate >= 75 else "✗"
        print(f"  {status_icon} {city:15} ({loc_id}): {matched:>4}/{total:<4} = {match_rate:>5.1f}%  (avg temp: {avg_temp}°F)")

    print()

def test_1min_joins(session):
    """Test 1-minute joins (for granular analysis)."""
    print("\n" + "="*60)
    print("TEST 2: 1-MINUTE JOINS (Kalshi ↔ Weather Grid)")
    print("="*60 + "\n")

    # First check if grid has data
    grid_count = session.execute(text("SELECT COUNT(*) FROM wx.minute_obs_1m")).fetchone()[0]

    if grid_count == 0:
        print("  ⚠ 1-minute grid is EMPTY - materialized view not refreshed yet")
        print("    Run this test again after a few minutes\n")
        return

    print(f"  1-minute grid records: {grid_count:,}\n")

    # Test Chicago for last 3 days
    three_days_ago = (datetime.utcnow() - timedelta(days=3)).strftime('%Y-%m-%d')

    result = session.execute(text(f"""
        WITH joined AS (
            SELECT c.market_ticker, c.timestamp, c.close as price_cents,
                   w.temp_f,
                   CASE WHEN w.temp_f IS NULL THEN 'NO_WX' ELSE 'MATCHED' END as status
            FROM candles c
            LEFT JOIN wx.minute_obs_1m w ON w.loc_id = 'KMDW' AND w.ts_utc = c.timestamp
            WHERE c.period_minutes = 1
              AND c.market_ticker LIKE 'KXHIGHCHI-%'
              AND c.timestamp >= '{three_days_ago}'
        )
        SELECT status, COUNT(*) as count,
               ROUND(AVG(temp_f)::numeric, 1) as avg_temp_f
        FROM joined
        GROUP BY status
    """)).fetchall()

    if not result:
        print("  ⚠ No 1-minute candles found for Chicago in last 3 days\n")
        return

    total = sum(r[1] for r in result)
    matched = next((r[1] for r in result if r[0] == 'MATCHED'), 0)
    match_rate = (matched / total * 100) if total > 0 else 0

    avg_temp = next((r[2] for r in result if r[0] == 'MATCHED'), None)

    status_icon = "✓" if match_rate >= 90 else "⚠" if match_rate >= 75 else "✗"
    print(f"  {status_icon} Chicago (1-min): {matched:>5}/{total:<5} = {match_rate:>5.1f}%  (avg temp: {avg_temp}°F)\n")

def test_join_sample(session):
    """Show sample joined data."""
    print("\n" + "="*60)
    print("TEST 3: SAMPLE JOINED DATA (Chicago, last 10 records)")
    print("="*60 + "\n")

    result = session.execute(text("""
        SELECT c.timestamp, c.close as price_cents,
               w.temp_f, w.humidity, w.windspeed_mph
        FROM candles c
        LEFT JOIN wx.minute_obs w ON w.loc_id = 'KMDW' AND w.ts_utc = c.timestamp
        WHERE c.period_minutes = 5
          AND c.market_ticker LIKE 'KXHIGHCHI-%'
        ORDER BY c.timestamp DESC
        LIMIT 10
    """)).fetchall()

    print(f"  {'Timestamp':<20} {'Price':<8} {'Temp(°F)':<10} {'Humidity':<10} {'Wind(mph)':<10}")
    print("  " + "-"*70)

    for row in result:
        ts = row[0].strftime('%Y-%m-%d %H:%M')
        price = f"{row[1]}¢" if row[1] else "N/A"
        temp = f"{row[2]:.1f}" if row[2] else "NULL"
        humidity = f"{row[3]:.0f}%" if row[3] else "NULL"
        wind = f"{row[4]:.1f}" if row[4] else "NULL"

        print(f"  {ts:<20} {price:<8} {temp:<10} {humidity:<10} {wind:<10}")

    print()

def test_data_coverage(session):
    """Test data coverage and identify gaps."""
    print("\n" + "="*60)
    print("TEST 4: DATA COVERAGE & GAPS")
    print("="*60 + "\n")

    # Check weather data coverage
    result = session.execute(text("""
        SELECT loc_id,
               MIN(ts_utc)::date as earliest,
               MAX(ts_utc)::date as latest,
               COUNT(*) as total_records
        FROM wx.minute_obs
        GROUP BY loc_id
        ORDER BY loc_id
    """)).fetchall()

    print("  Weather data (5-min observations):")
    for row in result:
        print(f"    {row[0]}: {row[1]} to {row[2]}  ({row[3]:,} records)")

    # Check Kalshi candle coverage
    result = session.execute(text("""
        SELECT
            CASE
                WHEN market_ticker LIKE 'KXHIGHCHI-%' THEN 'KMDW'
                WHEN market_ticker LIKE 'KXHIGHMIA-%' THEN 'KMIA'
                WHEN market_ticker LIKE 'KXHIGHAUS-%' THEN 'KAUS'
                WHEN market_ticker LIKE 'KXHIGHLAX-%' THEN 'KLAX'
                WHEN market_ticker LIKE 'KXHIGHDEN-%' THEN 'KDEN'
                WHEN market_ticker LIKE 'KXHIGHPHIL-%' THEN 'KPHL'
            END as loc_id,
            MIN(timestamp)::date as earliest,
            MAX(timestamp)::date as latest,
            COUNT(*) as total_candles
        FROM candles
        WHERE market_ticker LIKE 'KXHIGH%'
          AND period_minutes = 5
        GROUP BY
            CASE
                WHEN market_ticker LIKE 'KXHIGHCHI-%' THEN 'KMDW'
                WHEN market_ticker LIKE 'KXHIGHMIA-%' THEN 'KMIA'
                WHEN market_ticker LIKE 'KXHIGHAUS-%' THEN 'KAUS'
                WHEN market_ticker LIKE 'KXHIGHLAX-%' THEN 'KLAX'
                WHEN market_ticker LIKE 'KXHIGHDEN-%' THEN 'KDEN'
                WHEN market_ticker LIKE 'KXHIGHPHIL-%' THEN 'KPHL'
            END
        ORDER BY loc_id
    """)).fetchall()

    print("\n  Kalshi candles (5-min):")
    for row in result:
        print(f"    {row[0]}: {row[1]} to {row[2]}  ({row[3]:,} candles)")

    print()

def test_edge_cases(session):
    """Test edge cases."""
    print("\n" + "="*60)
    print("TEST 5: EDGE CASES")
    print("="*60 + "\n")

    # 1. Timezone midnight crossing
    print("  1. Timezone boundary (UTC midnight):")
    result = session.execute(text("""
        SELECT DATE_TRUNC('hour', ts_utc) as hour, COUNT(*) as records
        FROM wx.minute_obs
        WHERE loc_id = 'KMDW'
          AND ts_utc >= '2025-11-10 23:00:00'
          AND ts_utc < '2025-11-11 01:00:00'
        GROUP BY DATE_TRUNC('hour', ts_utc)
        ORDER BY hour
    """)).fetchall()

    if result:
        for row in result:
            print(f"     {row[0]}: {row[1]} records")
        print(f"     ✓ Midnight crossing OK")
    else:
        print(f"     ⚠ No data for midnight test")

    # 2. Duplicate check
    print("\n  2. Duplicate prevention (composite PK):")
    result = session.execute(text("""
        SELECT loc_id, ts_utc, COUNT(*) as dupes
        FROM wx.minute_obs
        GROUP BY loc_id, ts_utc
        HAVING COUNT(*) > 1
        LIMIT 5
    """)).fetchall()

    if result:
        print(f"     ✗ FOUND {len(result)} DUPLICATES:")
        for row in result:
            print(f"       {row[0]} at {row[1]}: {row[2]} copies")
    else:
        print(f"     ✓ No duplicates (composite PK working)")

    # 3. NULL handling in joins
    print("\n  3. NULL weather handling (LEFT JOIN):")
    result = session.execute(text("""
        SELECT
            COUNT(*) as total_candles,
            COUNT(w.temp_f) as with_weather,
            COUNT(*) - COUNT(w.temp_f) as missing_weather
        FROM candles c
        LEFT JOIN wx.minute_obs w ON w.loc_id = 'KMDW' AND w.ts_utc = c.timestamp
        WHERE c.period_minutes = 5
          AND c.market_ticker LIKE 'KXHIGHCHI-%'
          AND c.timestamp >= CURRENT_DATE - INTERVAL '1 day'
    """)).fetchone()

    if result and result[0] > 0:
        print(f"     Total candles (last 24h): {result[0]}")
        print(f"     With weather: {result[1]}")
        print(f"     Missing weather: {result[2]}")
        if result[2] > 0:
            pct = (result[2] / result[0] * 100)
            print(f"     Missing rate: {pct:.1f}%")
            if pct < 10:
                print(f"     ✓ Low missing rate (< 10%)")
            else:
                print(f"     ⚠ High missing rate (weather poller may be catching up)")
        else:
            print(f"     ✓ Complete weather coverage")
    else:
        print(f"     (No candles in last 24h to test)")

    print()

def main():
    """Run all join tests."""
    print("\n" + "="*60)
    print("KALSHI ↔ WEATHER JOIN VALIDATION")
    print("="*60)
    print(f"Timestamp: {datetime.utcnow()} UTC\n")

    try:
        with get_session() as session:
            test_5min_joins(session)
            test_1min_joins(session)
            test_join_sample(session)
            test_data_coverage(session)
            test_edge_cases(session)

            print("\n" + "="*60)
            print("✓ ALL JOIN TESTS COMPLETE")
            print("="*60 + "\n")

    except Exception as e:
        print(f"\n✗ ERROR during join tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
