#!/usr/bin/env python3
"""
Verify complete data pipeline status.

Checks:
1. Weather data coverage (wx.minute_obs)
2. 1-minute grid status (wx.minute_obs_1m)
3. Kalshi market data
4. Join quality (5-min and 1-min)
5. Edge cases
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from sqlalchemy import text
from db.connection import get_session

def print_section(title):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")

def check_weather_data(session):
    """Check 5-minute raw weather observations."""
    print_section("1. WEATHER DATA (5-min observations)")

    # Overall stats
    result = session.execute(text("""
        SELECT COUNT(*) as total_records,
               COUNT(DISTINCT loc_id) as num_locations,
               MIN(ts_utc) as earliest,
               MAX(ts_utc) as latest
        FROM wx.minute_obs
    """)).fetchone()

    print(f"Total 5-min records: {result[0]:,}")
    print(f"Locations: {result[1]}")
    print(f"Date range: {result[2]} to {result[3]}")

    # Per-city breakdown
    results = session.execute(text("""
        SELECT l.city, l.loc_id, COUNT(*) as records,
               MIN(o.ts_utc)::date as earliest,
               MAX(o.ts_utc)::date as latest
        FROM wx.minute_obs o
        JOIN wx.location l USING (loc_id)
        GROUP BY l.city, l.loc_id
        ORDER BY l.city
    """)).fetchall()

    print(f"\nPer-city breakdown:")
    for row in results:
        print(f"  {row[0]:15} ({row[1]}): {row[2]:>6,} records  [{row[3]} to {row[4]}]")

    return result

def check_1min_grid(session):
    """Check 1-minute upsampled grid."""
    print_section("2. 1-MINUTE GRID (LOCF upsampled)")

    try:
        result = session.execute(text("""
            SELECT COUNT(*) as total_records,
                   COUNT(DISTINCT loc_id) as num_locations,
                   MIN(ts_utc) as earliest,
                   MAX(ts_utc) as latest
            FROM wx.minute_obs_1m
        """)).fetchone()

        if result[0] == 0:
            print("⚠ MATERIALIZED VIEW IS EMPTY - refresh may still be running")
            return None

        print(f"Total 1-min records: {result[0]:,}")
        print(f"Locations: {result[1]}")
        print(f"Date range: {result[2]} to {result[3]}")

        # Check upsampling quality
        sample = session.execute(text("""
            SELECT loc_id, COUNT(*) as count_1m,
                   (SELECT COUNT(*) FROM wx.minute_obs o WHERE o.loc_id = g.loc_id) as count_5m
            FROM wx.minute_obs_1m g
            WHERE loc_id = 'KMDW'
            GROUP BY loc_id
        """)).fetchone()

        if sample:
            ratio = sample[1] / sample[2] if sample[2] > 0 else 0
            print(f"\nUpsampling check (Chicago):")
            print(f"  1-min records: {sample[1]:,}")
            print(f"  5-min records: {sample[2]:,}")
            print(f"  Ratio: {ratio:.1f}x (expected: ~5x)")

        return result

    except Exception as e:
        print(f"⚠ ERROR checking 1-minute grid: {e}")
        print("  (Materialized view may still be refreshing)")
        return None

def check_kalshi_data(session):
    """Check Kalshi market and candle data."""
    print_section("3. KALSHI MARKET DATA")

    # Markets
    results = session.execute(text("""
        SELECT series_ticker,
               COUNT(DISTINCT ticker) as num_markets,
               MIN(close_time)::date as earliest_close,
               MAX(close_time)::date as latest_close
        FROM markets
        WHERE series_ticker IN (
            'KXHIGHCHI', 'KXHIGHMIA', 'KXHIGHAUST',
            'KXHIGHLA', 'KXHIGHDEN', 'KXHIGHPHL'
        )
        GROUP BY series_ticker
        ORDER BY series_ticker
    """)).fetchall()

    print(f"Markets by city:")
    total_markets = 0
    for row in results:
        city = row[0].replace('KXHIGH', '').upper()
        print(f"  {city:10}: {row[1]:>3} markets  [{row[2]} to {row[3]}]")
        total_markets += row[1]

    print(f"\nTotal markets: {total_markets}")

    # Candles
    result = session.execute(text("""
        SELECT COUNT(*) as total_candles,
               COUNT(DISTINCT market_ticker) as markets_with_candles,
               MIN(timestamp)::date as earliest,
               MAX(timestamp)::date as latest
        FROM candles
        WHERE market_ticker LIKE 'KXHIGH%'
    """)).fetchone()

    print(f"\nCandles:")
    print(f"  Total candles: {result[0]:,}")
    print(f"  Markets with candles: {result[1]}")
    print(f"  Date range: {result[2]} to {result[3]}")

    return result

def check_5min_joins(session):
    """Test 5-minute joins between candles and weather."""
    print_section("4. 5-MINUTE JOINS (Kalshi candles ↔ Weather)")

    # Test Chicago (KMDW) for last 3 days
    three_days_ago = (datetime.utcnow() - timedelta(days=3)).strftime('%Y-%m-%d')

    result = session.execute(text(f"""
        WITH joined AS (
            SELECT c.market_ticker, c.timestamp, c.close as price_cents,
                   w.temp_f, w.humidity,
                   CASE WHEN w.temp_f IS NULL THEN 'NO_WEATHER' ELSE 'MATCHED' END as status
            FROM candles c
            LEFT JOIN wx.minute_obs w ON w.loc_id = 'KMDW' AND w.ts_utc = c.timestamp
            WHERE c.period_minutes = 5
              AND c.market_ticker LIKE 'KXHIGHCHI-%'
              AND c.timestamp >= '{three_days_ago}'
        )
        SELECT status,
               COUNT(*) as count,
               ROUND(AVG(temp_f)::numeric, 1) as avg_temp_f,
               ROUND(AVG(price_cents)::numeric, 1) as avg_price_cents
        FROM joined
        GROUP BY status
        ORDER BY status
    """)).fetchall()

    print(f"Chicago 5-min joins (last 3 days):")
    total_candles = 0
    matched = 0
    for row in result:
        total_candles += row[1]
        if row[0] == 'MATCHED':
            matched = row[1]
        avg_temp = f"{row[2]}°F" if row[2] else "N/A"
        avg_price = f"{row[3]}¢" if row[3] else "N/A"
        print(f"  {row[0]:12}: {row[1]:>5} rows  (avg temp: {avg_temp:>8}, avg price: {avg_price:>8})")

    match_rate = (matched / total_candles * 100) if total_candles > 0 else 0
    print(f"\nMatch rate: {match_rate:.1f}%")

    if match_rate < 95:
        print(f"⚠ WARNING: Low match rate - weather data may be incomplete")
    else:
        print(f"✓ Good match rate")

    return match_rate

def check_1min_joins(session):
    """Test 1-minute joins."""
    print_section("5. 1-MINUTE JOINS (Kalshi candles ↔ Weather grid)")

    # First check if grid has data
    grid_count = session.execute(text("SELECT COUNT(*) FROM wx.minute_obs_1m")).fetchone()[0]

    if grid_count == 0:
        print("⚠ 1-minute grid is empty - skipping join test")
        print("  (Run this again after materialized view refresh completes)")
        return None

    # Test Chicago for last 3 days
    three_days_ago = (datetime.utcnow() - timedelta(days=3)).strftime('%Y-%m-%d')

    result = session.execute(text(f"""
        WITH joined AS (
            SELECT c.market_ticker, c.timestamp, c.close as price_cents,
                   w.temp_f,
                   CASE WHEN w.temp_f IS NULL THEN 'NO_WEATHER' ELSE 'MATCHED' END as status
            FROM candles c
            LEFT JOIN wx.minute_obs_1m w ON w.loc_id = 'KMDW' AND w.ts_utc = c.timestamp
            WHERE c.period_minutes = 1
              AND c.market_ticker LIKE 'KXHIGHCHI-%'
              AND c.timestamp >= '{three_days_ago}'
        )
        SELECT status,
               COUNT(*) as count,
               ROUND(AVG(temp_f)::numeric, 1) as avg_temp_f,
               ROUND(AVG(price_cents)::numeric, 1) as avg_price_cents
        FROM joined
        GROUP BY status
        ORDER BY status
    """)).fetchall()

    print(f"Chicago 1-min joins (last 3 days):")
    total_candles = 0
    matched = 0
    for row in result:
        total_candles += row[1]
        if row[0] == 'MATCHED':
            matched = row[1]
        avg_temp = f"{row[2]}°F" if row[2] else "N/A"
        avg_price = f"{row[3]}¢" if row[3] else "N/A"
        print(f"  {row[0]:12}: {row[1]:>5} rows  (avg temp: {avg_temp:>8}, avg price: {avg_price:>8})")

    match_rate = (matched / total_candles * 100) if total_candles > 0 else 0
    print(f"\nMatch rate: {match_rate:.1f}%")

    return match_rate

def check_edge_cases(session):
    """Test edge cases."""
    print_section("6. EDGE CASE TESTS")

    print("Test 1: Timezone boundaries (UTC midnight crossings)")
    result = session.execute(text("""
        SELECT DATE_TRUNC('hour', ts_utc) as hour,
               COUNT(*) as records
        FROM wx.minute_obs
        WHERE loc_id = 'KMDW'
          AND ts_utc >= '2025-11-10 23:00:00'
          AND ts_utc < '2025-11-11 01:00:00'
        GROUP BY DATE_TRUNC('hour', ts_utc)
        ORDER BY hour
    """)).fetchall()

    if result:
        for row in result:
            print(f"  {row[0]}: {row[1]} records")
        print(f"  ✓ Midnight crossing looks good")
    else:
        print(f"  ⚠ No data for midnight crossing test")

    print("\nTest 2: Missing weather data handling (LEFT JOIN returns NULL)")
    result = session.execute(text("""
        SELECT COUNT(*) as total,
               COUNT(w.temp_f) as with_weather,
               COUNT(*) - COUNT(w.temp_f) as missing_weather
        FROM candles c
        LEFT JOIN wx.minute_obs w ON w.loc_id = 'KMDW' AND w.ts_utc = c.timestamp
        WHERE c.period_minutes = 5
          AND c.market_ticker LIKE 'KXHIGHCHI-%'
          AND c.timestamp >= '2025-11-12'
        LIMIT 1
    """)).fetchone()

    if result and result[0] > 0:
        print(f"  Total candles today: {result[0]}")
        print(f"  With weather: {result[1]}")
        print(f"  Missing weather: {result[2]}")
        if result[2] > 0:
            print(f"  ⚠ {result[2]} candles missing weather (expected if weather poller not running)")
        else:
            print(f"  ✓ All candles have weather data")
    else:
        print(f"  No candles found for today yet")

    print("\nTest 3: Duplicate prevention (composite primary key)")
    result = session.execute(text("""
        SELECT loc_id, ts_utc, COUNT(*) as dupes
        FROM wx.minute_obs
        GROUP BY loc_id, ts_utc
        HAVING COUNT(*) > 1
        LIMIT 5
    """)).fetchall()

    if result:
        print(f"  ⚠ FOUND {len(result)} DUPLICATES:")
        for row in result:
            print(f"    {row[0]} at {row[1]}: {row[2]} copies")
    else:
        print(f"  ✓ No duplicates found (composite PK working)")

    print("\nTest 4: Concurrent ingestion check")
    # Check if continuous ingestion process is running
    import subprocess
    try:
        ps_result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )

        kalshi_running = "continuous_ingest.py" in ps_result.stdout
        wx_running = "poll_visualcrossing.py" in ps_result.stdout

        print(f"  Kalshi continuous ingest: {'✓ RUNNING' if kalshi_running else '✗ NOT RUNNING'}")
        print(f"  Weather poller: {'✓ RUNNING' if wx_running else '✗ NOT RUNNING'}")

        if not wx_running:
            print(f"\n  ⚠ Weather poller not running - start with: make poll-wx-live")
    except Exception as e:
        print(f"  Could not check process status: {e}")

def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("KALSHI WEATHER PIPELINE - DATA VERIFICATION")
    print("="*60)
    print(f"Timestamp: {datetime.utcnow()} UTC")

    try:
        with get_session() as session:
            # Run all checks
            wx_data = check_weather_data(session)
            grid_data = check_1min_grid(session)
            kalshi_data = check_kalshi_data(session)
            match_rate_5m = check_5min_joins(session)
            match_rate_1m = check_1min_joins(session)
            check_edge_cases(session)

            # Summary
            print_section("SUMMARY")

            if wx_data and wx_data[0] > 0:
                print(f"✓ Weather data: {wx_data[0]:,} records across {wx_data[1]} locations")
            else:
                print(f"✗ Weather data: MISSING or EMPTY")

            if grid_data and grid_data[0] > 0:
                print(f"✓ 1-minute grid: {grid_data[0]:,} records")
            else:
                print(f"⚠ 1-minute grid: Empty (refresh may be running)")

            if kalshi_data and kalshi_data[0] > 0:
                print(f"✓ Kalshi candles: {kalshi_data[0]:,} records")
            else:
                print(f"✗ Kalshi candles: MISSING or EMPTY")

            if match_rate_5m and match_rate_5m >= 95:
                print(f"✓ 5-minute joins: {match_rate_5m:.1f}% match rate")
            elif match_rate_5m:
                print(f"⚠ 5-minute joins: {match_rate_5m:.1f}% match rate (LOW)")
            else:
                print(f"✗ 5-minute joins: Could not test")

            if match_rate_1m and match_rate_1m >= 95:
                print(f"✓ 1-minute joins: {match_rate_1m:.1f}% match rate")
            elif match_rate_1m is None:
                print(f"⚠ 1-minute joins: Skipped (grid empty)")
            elif match_rate_1m:
                print(f"⚠ 1-minute joins: {match_rate_1m:.1f}% match rate (LOW)")

            print("\n" + "="*60)
            print("VERIFICATION COMPLETE")
            print("="*60 + "\n")

    except Exception as e:
        print(f"\n✗ ERROR during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
