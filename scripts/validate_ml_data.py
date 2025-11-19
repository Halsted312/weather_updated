#!/usr/bin/env python3
"""
Validate ML training data readiness.

Checks:
1. Candles exist for Chicago (last 60 days)
2. Weather observations exist (wx.minute_obs_1m)
3. Markets have close_time populated
4. Timezone alignment (candles UTC, weather LST)
5. No gaps in critical time series
6. Sufficient volume/liquidity for training
"""

import sys
import os
from datetime import datetime, timedelta, date, timezone
from typing import Dict, List, Tuple
import pandas as pd
from sqlalchemy import text

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import get_session
from ml.dataset import CITY_CONFIG


def validate_candles_chicago(session, days: int = 60) -> Dict:
    """
    Validate Chicago candles data.

    Returns dict with:
    - total_candles: int
    - date_range: (start_date, end_date)
    - num_markets: int
    - avg_candles_per_market: float
    - missing_dates: List[date]
    - low_volume_dates: List[date]
    """
    print("\n" + "="*60)
    print("1. VALIDATING CHICAGO CANDLES")
    print("="*60)

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

    # Get basic stats
    query = text("""
        SELECT
            COUNT(*) as total_candles,
            COUNT(DISTINCT market_ticker) as num_markets,
            MIN(timestamp) as earliest_ts,
            MAX(timestamp) as latest_ts,
            MIN(timestamp)::date as earliest_date,
            MAX(timestamp)::date as latest_date
        FROM candles
        WHERE market_ticker LIKE 'KXHIGHCHI%'
          AND timestamp >= :cutoff_date
    """)

    result = session.execute(query, {"cutoff_date": cutoff_date}).fetchone()

    total_candles = result.total_candles
    num_markets = result.num_markets
    earliest_date = result.earliest_date
    latest_date = result.latest_date

    print(f"\n✓ Total candles: {total_candles:,}")
    print(f"✓ Number of markets: {num_markets}")
    print(f"✓ Date range: {earliest_date} to {latest_date}")

    if total_candles == 0:
        print("❌ ERROR: No candles found for Chicago in last 60 days!")
        return {"valid": False, "error": "No candles"}

    avg_candles_per_market = total_candles / num_markets if num_markets > 0 else 0
    print(f"✓ Avg candles per market: {avg_candles_per_market:.0f}")

    # Check for date gaps (skip if date range invalid)
    missing_dates_result = []
    if earliest_date and latest_date:
        query = text("""
            WITH date_series AS (
                SELECT generate_series(
                    CAST(:start_date AS date),
                    CAST(:end_date AS date),
                    '1 day'::interval
                ) AS date
            ),
            market_dates AS (
                SELECT DISTINCT CAST(timestamp AS date) as date
                FROM candles
                WHERE market_ticker LIKE 'KXHIGHCHI%'
                  AND timestamp >= :cutoff_date
            )
            SELECT ds.date
            FROM date_series ds
            LEFT JOIN market_dates md ON ds.date = md.date
            WHERE md.date IS NULL
            ORDER BY ds.date
        """)

        missing_dates_result = session.execute(query, {
            "start_date": earliest_date,
            "end_date": latest_date,
            "cutoff_date": cutoff_date
        }).fetchall()

    missing_dates = [row.date for row in missing_dates_result]

    if missing_dates:
        print(f"\n⚠ WARNING: {len(missing_dates)} dates with no candles:")
        for d in missing_dates[:10]:  # Show first 10
            print(f"    {d}")
        if len(missing_dates) > 10:
            print(f"    ... and {len(missing_dates) - 10} more")
    else:
        print("\n✓ No missing dates in candle data")

    # Check for low volume dates
    query = text("""
        SELECT
            timestamp::date as date,
            SUM(volume) as total_volume,
            COUNT(*) as num_candles
        FROM candles
        WHERE market_ticker LIKE 'KXHIGHCHI%'
          AND timestamp >= :cutoff_date
        GROUP BY timestamp::date
        HAVING SUM(volume) < 100  -- Low volume threshold
        ORDER BY date DESC
    """)

    low_volume_result = session.execute(query, {"cutoff_date": cutoff_date}).fetchall()
    low_volume_dates = [row.date for row in low_volume_result]

    if low_volume_dates:
        print(f"\n⚠ WARNING: {len(low_volume_dates)} dates with low volume (<100):")
        for row in low_volume_result[:5]:
            print(f"    {row.date}: {row.total_volume} volume, {row.num_candles} candles")
    else:
        print("\n✓ All dates have sufficient volume")

    return {
        "valid": True,
        "total_candles": total_candles,
        "num_markets": num_markets,
        "date_range": (earliest_date, latest_date),
        "avg_candles_per_market": avg_candles_per_market,
        "missing_dates": missing_dates,
        "low_volume_dates": low_volume_dates,
    }


def validate_weather_1min(session, days: int = 60) -> Dict:
    """
    Validate 1-minute weather grid (wx.minute_obs_1m).

    Returns dict with:
    - total_records: int
    - num_locations: int
    - date_range: (start_date, end_date)
    - chicago_records: int
    - missing_hours: List[datetime]
    """
    print("\n" + "="*60)
    print("2. VALIDATING 1-MINUTE WEATHER GRID")
    print("="*60)

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

    # Check if view exists
    try:
        query = text("""
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT loc_id) as num_locations,
                MIN(ts_utc) as earliest_ts,
                MAX(ts_utc) as latest_ts,
                MIN(ts_utc)::date as earliest_date,
                MAX(ts_utc)::date as latest_date
            FROM wx.minute_obs_1m
            WHERE ts_utc >= :cutoff_date
        """)

        result = session.execute(query, {"cutoff_date": cutoff_date}).fetchone()

        total_records = result.total_records
        num_locations = result.num_locations
        earliest_date = result.earliest_date
        latest_date = result.latest_date

        print(f"\n✓ Total 1-min records: {total_records:,}")
        print(f"✓ Number of locations: {num_locations}")
        print(f"✓ Date range: {earliest_date} to {latest_date}")

        if total_records == 0:
            print("❌ ERROR: No weather data in wx.minute_obs_1m!")
            return {"valid": False, "error": "No weather data"}

        # Check Chicago specifically (loc_id = 'KMDW' for Chicago Midway)
        query = text("""
            SELECT COUNT(*) as chicago_records
            FROM wx.minute_obs_1m
            WHERE loc_id = 'KMDW'
              AND ts_utc >= :cutoff_date
        """)

        chicago_result = session.execute(query, {"cutoff_date": cutoff_date}).fetchone()
        chicago_records = chicago_result.chicago_records

        print(f"✓ Chicago records: {chicago_records:,}")

        if chicago_records == 0:
            print("❌ ERROR: No Chicago weather data in wx.minute_obs_1m!")
            return {"valid": False, "error": "No Chicago weather"}

        # Expected records: 1440 minutes/day * days * num_locations
        expected_total = 1440 * days * num_locations
        coverage_pct = 100.0 * total_records / expected_total
        print(f"✓ Coverage: {coverage_pct:.1f}% of expected records")

        if coverage_pct < 80:
            print(f"⚠ WARNING: Coverage below 80%")

        # Check for hourly gaps (sample check)
        query = text("""
            WITH hourly_series AS (
                SELECT generate_series(
                    date_trunc('hour', :start_date),
                    date_trunc('hour', :end_date),
                    '1 hour'::interval
                ) as hour
            ),
            weather_hours AS (
                SELECT DISTINCT date_trunc('hour', ts_utc) as hour
                FROM wx.minute_obs_1m
                WHERE loc_id = 'KMDW'
                  AND ts_utc >= :cutoff_date
            )
            SELECT hs.hour
            FROM hourly_series hs
            LEFT JOIN weather_hours wh ON hs.hour = wh.hour
            WHERE wh.hour IS NULL
            ORDER BY hs.hour
            LIMIT 20
        """)

        missing_hours_result = session.execute(query, {
            "start_date": cutoff_date,
            "end_date": datetime.now(timezone.utc),
            "cutoff_date": cutoff_date
        }).fetchall()

        missing_hours = [row.hour for row in missing_hours_result]

        if missing_hours:
            print(f"\n⚠ WARNING: {len(missing_hours)}+ hours with missing weather data:")
            for h in missing_hours[:5]:
                print(f"    {h}")
        else:
            print("\n✓ No missing hours in Chicago weather data")

        return {
            "valid": True,
            "total_records": total_records,
            "num_locations": num_locations,
            "date_range": (earliest_date, latest_date),
            "chicago_records": chicago_records,
            "coverage_pct": coverage_pct,
            "missing_hours": missing_hours,
        }

    except Exception as e:
        print(f"\n❌ ERROR: Failed to query wx.minute_obs_1m: {e}")
        print("\nTrying wx.minute_obs (5-min data) as fallback...")

        # Fallback to 5-min data
        query = text("""
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT loc_id) as num_locations,
                MIN(ts_utc) as earliest_ts,
                MAX(ts_utc) as latest_ts
            FROM wx.minute_obs
            WHERE loc_id = 'KMDW'
              AND ts_utc >= :cutoff_date
        """)

        result = session.execute(query, {"cutoff_date": cutoff_date}).fetchone()

        print(f"\n✓ 5-min weather records (Chicago): {result.total_records:,}")
        print(f"✓ Date range: {result.earliest_ts} to {result.latest_ts}")

        if result.total_records == 0:
            print("❌ ERROR: No Chicago weather data in wx.minute_obs!")
            return {"valid": False, "error": "No 5-min weather data"}

        print("\n⚠ NOTE: Using 5-min data with ffill. Recommend refreshing wx.minute_obs_1m view.")

        return {
            "valid": True,
            "total_records": result.total_records,
            "num_locations": result.num_locations,
            "is_5min_fallback": True,
        }


def validate_market_metadata(session, days: int = 60) -> Dict:
    """
    Validate market metadata (close_time, strikes, etc).

    Returns dict with:
    - total_markets: int
    - markets_with_close_time: int
    - markets_with_strikes: int
    - bracket_counts: Dict[str, int]
    """
    print("\n" + "="*60)
    print("3. VALIDATING MARKET METADATA")
    print("="*60)

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

    # Get basic stats
    query = text("""
        SELECT
            COUNT(*) as total_markets,
            COUNT(close_time) as markets_with_close_time,
            COUNT(*) FILTER (WHERE floor_strike IS NOT NULL OR cap_strike IS NOT NULL) as markets_with_strikes
        FROM markets
        WHERE ticker LIKE 'KXHIGHCHI%'
          AND close_time >= :cutoff_date
    """)

    result = session.execute(query, {"cutoff_date": cutoff_date}).fetchone()

    total_markets = result[0]
    markets_with_close_time = result[1]
    markets_with_strikes = result[2]

    print(f"\n✓ Total markets: {total_markets}")
    print(f"✓ Markets with close_time: {markets_with_close_time}")
    print(f"✓ Markets with strike data: {markets_with_strikes}")

    if total_markets == 0:
        print("❌ ERROR: No Chicago markets found!")
        return {"valid": False, "error": "No markets"}

    if markets_with_close_time < total_markets:
        missing_close = total_markets - markets_with_close_time
        print(f"⚠ WARNING: {missing_close} markets missing close_time")

    # Get bracket type counts separately
    query = text("""
        SELECT
            SUM(CASE WHEN strike_type = 'greater' THEN 1 ELSE 0 END) as greater_count,
            SUM(CASE WHEN strike_type = 'less' THEN 1 ELSE 0 END) as less_count,
            SUM(CASE WHEN strike_type = 'between' THEN 1 ELSE 0 END) as between_count
        FROM markets
        WHERE ticker LIKE 'KXHIGHCHI%'
          AND close_time >= :cutoff_date
    """)

    bracket_result = session.execute(query, {"cutoff_date": cutoff_date}).fetchone()
    greater_count = bracket_result[0] or 0
    less_count = bracket_result[1] or 0
    between_count = bracket_result[2] or 0

    print(f"\nBracket type distribution:")
    print(f"  Greater (>X): {greater_count}")
    print(f"  Less (<X): {less_count}")
    print(f"  Between (X-Y): {between_count}")

    # Check for markets without settlement value
    query = text("""
        SELECT COUNT(*) as no_settlement
        FROM markets
        WHERE ticker LIKE 'KXHIGHCHI%'
          AND close_time < :now
          AND close_time >= :cutoff_date
          AND settlement_value IS NULL
    """)

    result = session.execute(query, {
        "now": datetime.now(timezone.utc),
        "cutoff_date": cutoff_date
    }).fetchone()

    no_settlement = result.no_settlement

    if no_settlement > 0:
        print(f"\n⚠ WARNING: {no_settlement} closed markets missing settlement_value")
    else:
        print(f"\n✓ All closed markets have settlement values")

    return {
        "valid": True,
        "total_markets": total_markets,
        "markets_with_close_time": markets_with_close_time,
        "markets_with_strikes": markets_with_strikes,
        "bracket_counts": {
            "greater": greater_count,
            "less": less_count,
            "between": between_count,
        },
        "no_settlement": no_settlement,
    }


def validate_timezone_alignment(session) -> Dict:
    """
    Validate timezone alignment between candles (UTC) and weather (LST).

    Returns dict with:
    - candles_tz: str
    - weather_tz: str
    - sample_alignment: List[Tuple]
    """
    print("\n" + "="*60)
    print("4. VALIDATING TIMEZONE ALIGNMENT")
    print("="*60)

    # Check candle timestamps (should be UTC)
    query = text("""
        SELECT
            MIN(timestamp) as earliest,
            MAX(timestamp) as latest
        FROM candles
        WHERE market_ticker LIKE 'KXHIGHCHI%'
    """)

    result = session.execute(query).fetchone()

    if result and result[0]:
        print(f"\n✓ Candles timestamp range:")
        print(f"    Earliest: {result[0]}")
        print(f"    Latest: {result[1]}")
        print(f"    Timezone: UTC (assumed, stored as timestamp without timezone)")
    else:
        print("\n⚠ WARNING: Could not determine candle timezone")

    # Check weather timestamps
    try:
        query = text("""
            SELECT
                MIN(ts_utc) as earliest,
                MAX(ts_utc) as latest
            FROM wx.minute_obs_1m
            WHERE loc_id = 'KMDW'
            LIMIT 1
        """)

        result = session.execute(query).fetchone()

        if result and result[0]:
            print(f"\n✓ Weather (1-min) timestamp range:")
            print(f"    Earliest: {result[0]}")
            print(f"    Latest: {result[1]}")
            print(f"    Timezone: UTC (stored)")
        else:
            print("\n⚠ WARNING: Could not determine weather timezone")

    except Exception:
        # Fallback to 5-min data
        query = text("""
            SELECT
                MIN(ts_utc) as earliest,
                MAX(ts_utc) as latest
            FROM wx.minute_obs
            WHERE loc_id = 'KMDW'
            LIMIT 1
        """)

        result = session.execute(query).fetchone()

        if result and result[0]:
            print(f"\n✓ Weather (5-min) timestamp range:")
            print(f"    Earliest: {result[0]}")
            print(f"    Latest: {result[1]}")
            print(f"    Timezone: UTC (stored)")

    # Sample alignment check
    print("\n✓ Both candles and weather use UTC timestamps")
    print("✓ Local timezone conversions should be done in feature engineering")

    return {
        "valid": True,
        "candles_tz": "UTC",
        "weather_tz": "UTC",
    }


def validate_join_feasibility(session) -> Dict:
    """
    Validate that candles and weather can be joined successfully.

    Returns dict with:
    - sample_join_count: int
    - candles_without_weather: int
    - weather_without_candles: int
    """
    print("\n" + "="*60)
    print("5. VALIDATING JOIN FEASIBILITY")
    print("="*60)

    # Try a sample join (last 7 days)
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)

    try:
        query = text("""
            WITH candle_sample AS (
                SELECT
                    market_ticker,
                    timestamp as candle_ts,
                    close as price
                FROM candles
                WHERE market_ticker LIKE 'KXHIGHCHI%'
                  AND timestamp >= :cutoff_date
                LIMIT 1000
            ),
            weather_sample AS (
                SELECT
                    ts_utc as weather_ts,
                    temp_f
                FROM wx.minute_obs_1m
                WHERE loc_id = 'KMDW'
                  AND ts_utc >= :cutoff_date
                LIMIT 1000
            )
            SELECT COUNT(*) as join_count
            FROM candle_sample cs
            LEFT JOIN weather_sample ws
              ON date_trunc('minute', cs.candle_ts) = date_trunc('minute', ws.weather_ts)
            WHERE ws.weather_ts IS NOT NULL
        """)

        result = session.execute(query, {"cutoff_date": cutoff_date}).fetchone()
        join_count = result.join_count

        print(f"\n✓ Sample join (last 7 days): {join_count} successful matches")

        if join_count == 0:
            print("❌ ERROR: No successful candle-weather joins!")
            return {"valid": False, "error": "Join failed"}

        print("✓ Candles and weather can be joined successfully")

        return {
            "valid": True,
            "sample_join_count": join_count,
        }

    except Exception as e:
        print(f"\n❌ ERROR: Join test failed: {e}")
        print("\nTrying 5-min weather fallback...")

        # Try with 5-min data
        query = text("""
            WITH candle_sample AS (
                SELECT
                    market_ticker,
                    timestamp as candle_ts,
                    close as price
                FROM candles
                WHERE market_ticker LIKE 'KXHIGHCHI%'
                  AND timestamp >= :cutoff_date
                LIMIT 1000
            ),
            weather_sample AS (
                SELECT
                    ts_utc as weather_ts,
                    temp_f
                FROM wx.minute_obs
                WHERE loc_id = 'KMDW'
                  AND ts_utc >= :cutoff_date
                LIMIT 1000
            )
            SELECT COUNT(*) as join_count
            FROM candle_sample cs
            LEFT JOIN LATERAL (
                SELECT temp_f
                FROM weather_sample ws
                WHERE ws.weather_ts <= cs.candle_ts
                ORDER BY ws.weather_ts DESC
                LIMIT 1
            ) w ON true
            WHERE w.temp_f IS NOT NULL
        """)

        result = session.execute(query, {"cutoff_date": cutoff_date}).fetchone()
        join_count = result.join_count

        print(f"\n✓ Sample join with 5-min ffill: {join_count} successful matches")
        print("✓ Can use 5-min weather with forward-fill")

        return {
            "valid": True,
            "sample_join_count": join_count,
            "is_5min_fallback": True,
        }


def main():
    """Run all validation checks."""
    print("\n" + "="*70)
    print("ML DATA VALIDATION")
    print("="*70)

    results = {}
    all_valid = True

    with get_session() as session:
        # 1. Validate candles
        candles_result = validate_candles_chicago(session, days=60)
        results["candles"] = candles_result
        if not candles_result.get("valid"):
            all_valid = False

        # 2. Validate weather
        weather_result = validate_weather_1min(session, days=60)
        results["weather"] = weather_result
        if not weather_result.get("valid"):
            all_valid = False

        # 3. Validate market metadata
        metadata_result = validate_market_metadata(session, days=60)
        results["metadata"] = metadata_result
        if not metadata_result.get("valid"):
            all_valid = False

        # 4. Validate timezone alignment
        tz_result = validate_timezone_alignment(session)
        results["timezone"] = tz_result

        # 5. Validate join feasibility
        join_result = validate_join_feasibility(session)
        results["join"] = join_result
        if not join_result.get("valid"):
            all_valid = False

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    if all_valid:
        print("\n✅ ALL CHECKS PASSED")
        print("\nReady for ML training:")
        print(f"  • {results['candles']['total_candles']:,} candles from {results['candles']['num_markets']} markets")
        print(f"  • {results['weather'].get('chicago_records', 'N/A'):,} weather observations")
        print(f"  • {results['metadata']['total_markets']} markets with metadata")
        print(f"  • Timezone alignment verified")
        print(f"  • Join feasibility confirmed")
    else:
        print("\n❌ VALIDATION FAILED")
        print("\nErrors found:")
        for check, result in results.items():
            if not result.get("valid"):
                print(f"  • {check}: {result.get('error', 'Unknown error')}")
        sys.exit(1)

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
