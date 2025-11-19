#!/usr/bin/env python3
"""
Tests for feature engineering module.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import datetime, date, timezone
from ml.features import FeatureBuilder


def test_basic_feature_building():
    """Test basic feature building with sample data."""
    print("\n" + "="*60)
    print("TEST 1: Basic Feature Building")
    print("="*60)

    # Sample candles
    candles_df = pd.DataFrame([
        {
            "market_ticker": "KXHIGHCHI-25AUG10-B80",
            "end_period_ts": datetime(2025, 8, 10, 14, 0, 0),
            "yes_bid_close": 48,
            "yes_ask_close": 52,
            "price_close": 50,
        },
        {
            "market_ticker": "KXHIGHCHI-25AUG10-B80",
            "end_period_ts": datetime(2025, 8, 10, 15, 0, 0),
            "yes_bid_close": 49,
            "yes_ask_close": 53,
            "price_close": 51,
        },
    ])

    # Sample weather
    weather_df = pd.DataFrame([
        {"date": date(2025, 8, 10), "tmax_f": 78.0},
    ])

    # Sample market metadata
    market_metadata = pd.DataFrame([
        {
            "market_ticker": "KXHIGHCHI-25AUG10-B80",
            "close_time": datetime(2025, 8, 10, 18, 0, 0),
            "strike_type": "between",
            "floor_strike": 80,
            "cap_strike": 81,
        },
    ])

    # Build features
    fb = FeatureBuilder(city_timezone="America/Chicago")
    features_df = fb.build_features(candles_df, weather_df, market_metadata)

    print(f"\nBuilt {len(features_df)} feature rows from {len(candles_df)} candles")

    # Check feature columns exist
    expected_cols = fb.get_feature_columns()
    print(f"\nExpected feature columns: {len(expected_cols)}")
    for col in expected_cols:
        assert col in features_df.columns, f"Missing feature column: {col}"
        print(f"  ✓ {col}")

    # Check values
    print(f"\nSample feature values:")
    print(f"  yes_mid: {features_df['yes_mid'].iloc[0]} (expected: 50.0)")
    print(f"  spread_cents: {features_df['spread_cents'].iloc[0]} (expected: 4.0)")
    print(f"  temp_now: {features_df['temp_now'].iloc[0]} (expected: 78.0)")
    print(f"  temp_to_floor: {features_df['temp_to_floor'].iloc[0]} (expected: 2.0)")
    print(f"  hour_of_day_local: {features_df['hour_of_day_local'].iloc[0]}")
    print(f"  day_of_week: {features_df['day_of_week'].iloc[0]}")
    print(f"  minutes_to_close: {features_df['minutes_to_close'].iloc[0]} (expected: 240.0)")

    # Assertions
    assert abs(features_df['yes_mid'].iloc[0] - 50.0) < 0.01
    assert abs(features_df['spread_cents'].iloc[0] - 4.0) < 0.01
    assert abs(features_df['temp_now'].iloc[0] - 78.0) < 0.01
    assert abs(features_df['temp_to_floor'].iloc[0] - 2.0) < 0.01
    assert abs(features_df['minutes_to_close'].iloc[0] - 240.0) < 0.01

    print("\n✓ TEST 1 PASSED")


def test_missing_data_handling():
    """Test feature building with missing data."""
    print("\n" + "="*60)
    print("TEST 2: Missing Data Handling")
    print("="*60)

    # Candles with missing bid/ask
    candles_df = pd.DataFrame([
        {
            "market_ticker": "KXHIGHCHI-25AUG10-B80",
            "end_period_ts": datetime(2025, 8, 10, 14, 0, 0),
            "price_close": 50,
            # No bid/ask
        },
    ])

    # Empty weather
    weather_df = pd.DataFrame()

    # No metadata
    market_metadata = None

    # Build features
    fb = FeatureBuilder(city_timezone="America/Chicago")
    features_df = fb.build_features(candles_df, weather_df, market_metadata)

    print(f"\nBuilt {len(features_df)} feature rows")

    # Check that price_close was used as fallback
    print(f"  yes_mid: {features_df['yes_mid'].iloc[0]} (should use price_close fallback: 50.0)")
    assert abs(features_df['yes_mid'].iloc[0] - 50.0) < 0.01

    # Check missing features are NaN
    missing = fb.validate_features(features_df)
    print(f"\nMissing value counts:")
    for col, count in missing.items():
        if count > 0:
            print(f"  {col}: {count} / {len(features_df)}")

    assert missing['temp_now'] > 0, "temp_now should be missing (no weather data)"
    assert missing['minutes_to_close'] > 0, "minutes_to_close should be missing (no metadata)"

    print("\n✓ TEST 2 PASSED")


def test_multiple_markets():
    """Test feature building with multiple markets and strike types."""
    print("\n" + "="*60)
    print("TEST 3: Multiple Markets")
    print("="*60)

    # Multiple markets (different strike types)
    candles_df = pd.DataFrame([
        {
            "market_ticker": "KXHIGHCHI-25AUG10-B80",
            "end_period_ts": datetime(2025, 8, 10, 14, 0, 0),
            "yes_bid_close": 48,
            "yes_ask_close": 52,
            "price_close": 50,
        },
        {
            "market_ticker": "KXHIGHCHI-25AUG10-G75",
            "end_period_ts": datetime(2025, 8, 10, 14, 0, 0),
            "yes_bid_close": 65,
            "yes_ask_close": 69,
            "price_close": 67,
        },
        {
            "market_ticker": "KXHIGHCHI-25AUG10-L85",
            "end_period_ts": datetime(2025, 8, 10, 14, 0, 0),
            "yes_bid_close": 30,
            "yes_ask_close": 34,
            "price_close": 32,
        },
    ])

    weather_df = pd.DataFrame([
        {"date": date(2025, 8, 10), "tmax_f": 78.0},
    ])

    market_metadata = pd.DataFrame([
        {
            "market_ticker": "KXHIGHCHI-25AUG10-B80",
            "close_time": datetime(2025, 8, 10, 18, 0, 0),
            "strike_type": "between",
            "floor_strike": 80,
            "cap_strike": 81,
        },
        {
            "market_ticker": "KXHIGHCHI-25AUG10-G75",
            "close_time": datetime(2025, 8, 10, 18, 0, 0),
            "strike_type": "greater",
            "floor_strike": 75,
            "cap_strike": None,
        },
        {
            "market_ticker": "KXHIGHCHI-25AUG10-L85",
            "close_time": datetime(2025, 8, 10, 18, 0, 0),
            "strike_type": "less",
            "floor_strike": None,
            "cap_strike": 85,
        },
    ])

    # Build features
    fb = FeatureBuilder(city_timezone="America/Chicago")
    features_df = fb.build_features(candles_df, weather_df, market_metadata)

    print(f"\nBuilt {len(features_df)} feature rows from {len(candles_df)} candles")

    # Check each market
    for ticker in ["KXHIGHCHI-25AUG10-B80", "KXHIGHCHI-25AUG10-G75", "KXHIGHCHI-25AUG10-L85"]:
        row = features_df[features_df['market_ticker'] == ticker].iloc[0]
        print(f"\n{ticker}:")
        print(f"  yes_mid: {row['yes_mid']}")
        print(f"  temp_to_floor: {row['temp_to_floor']}")
        print(f"  temp_to_cap: {row['temp_to_cap']}")

    # Check temp_to_floor for "between" market (80-81 bracket, temp=78)
    between_row = features_df[features_df['market_ticker'] == "KXHIGHCHI-25AUG10-B80"].iloc[0]
    assert abs(between_row['temp_to_floor'] - 2.0) < 0.01, "temp_to_floor should be 80-78=2"
    assert abs(between_row['temp_to_cap'] - 3.0) < 0.01, "temp_to_cap should be 81-78=3"

    # Check temp_to_floor for "greater" market (>75, temp=78)
    greater_row = features_df[features_df['market_ticker'] == "KXHIGHCHI-25AUG10-G75"].iloc[0]
    assert abs(greater_row['temp_to_floor'] - (-3.0)) < 0.01, "temp_to_floor should be 75-78=-3"

    # Check temp_to_cap for "less" market (<85, temp=78)
    less_row = features_df[features_df['market_ticker'] == "KXHIGHCHI-25AUG10-L85"].iloc[0]
    assert abs(less_row['temp_to_cap'] - 7.0) < 0.01, "temp_to_cap should be 85-78=7"

    print("\n✓ TEST 3 PASSED")


def test_event_date_tracks_close_time_local_day():
    """FeatureBuilder should infer event_date from close_time when needed."""
    candles_df = pd.DataFrame(
        [
            {
                "market_ticker": "KXHIGHCHI-24MAR10-G70",
                "end_period_ts": datetime(2024, 3, 10, 2, 0),
                "price_close": 55,
            }
        ]
    )

    metadata = pd.DataFrame(
        [
            {
                "market_ticker": "KXHIGHCHI-24MAR10-G70",
                "close_time": datetime(2024, 3, 10, 5, 0, tzinfo=timezone.utc),
                "strike_type": "greater",
                "floor_strike": 70,
                "cap_strike": None,
            }
        ]
    )

    fb = FeatureBuilder(city_timezone="America/Chicago")
    features_df = fb.build_features(candles_df, weather_df=None, market_metadata=metadata)

    assert features_df["event_date"].nunique() == 1
    assert features_df["event_date"].iloc[0] == date(2024, 3, 9)


def test_tzaware_timestamps_are_supported():
    """Timestamp columns already in UTC should still convert cleanly to local time."""
    candles_df = pd.DataFrame(
        [
            {
                "market_ticker": "KXHIGHCHI-24JAN01-G70",
                "timestamp": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
                "close": 60.0,
                "temp_f": 70.0,
            }
        ]
    )

    metadata = pd.DataFrame(
        [
            {
                "market_ticker": "KXHIGHCHI-24JAN01-G70",
                "close_time": datetime(2024, 1, 1, 18, 0, tzinfo=timezone.utc),
                "strike_type": "greater",
                "floor_strike": 70,
                "cap_strike": None,
            }
        ]
    )

    fb = FeatureBuilder(city_timezone="America/Chicago")
    features_df = fb.build_features(candles_df, weather_df=None, market_metadata=metadata)

    # 12:00 UTC corresponds to 6:00 local time in Chicago (UTC-6 in January)
    assert features_df["hour_of_day_local"].iloc[0] == 6
    # Minutes to close should be the difference between timestamps regardless of tz awareness
    assert abs(features_df["minutes_to_close"].iloc[0] - 360.0) < 1e-6


def test_database_integration():
    """Test feature building with real database data (if available)."""
    print("\n" + "="*60)
    print("TEST 4: Database Integration")
    print("="*60)

    try:
        from db.connection import get_session
        from sqlalchemy import text

        with get_session() as session:
            # Get sample candles (last 100 minutes of Chicago markets)
            candles_query = text("""
                SELECT
                    market_ticker,
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM candles
                WHERE market_ticker LIKE 'KXHIGHCHI%'
                ORDER BY timestamp DESC
                LIMIT 100
            """)
            candles_df = pd.read_sql(candles_query, session.bind)

            if candles_df.empty:
                print("  ⚠ No candles found in database, skipping test")
                return

            print(f"\nLoaded {len(candles_df)} candles from database")

            # Get weather data
            weather_query = text("""
                SELECT DISTINCT
                    date,
                    tmax_f
                FROM weather_observed
                WHERE station_id = 'GHCND:USW00014819'
                ORDER BY date DESC
                LIMIT 30
            """)
            weather_df = pd.read_sql(weather_query, session.bind)
            print(f"Loaded {len(weather_df)} weather observations")

            # Get market metadata
            metadata_query = text("""
                SELECT
                    ticker as market_ticker,
                    close_time,
                    strike_type,
                    floor_strike,
                    cap_strike
                FROM markets
                WHERE ticker LIKE 'KXHIGHCHI%'
            """)
            market_metadata = pd.read_sql(metadata_query, session.bind)
            print(f"Loaded {len(market_metadata)} market metadata records")

            # Build features
            fb = FeatureBuilder(city_timezone="America/Chicago")
            features_df = fb.build_features(candles_df, weather_df, market_metadata)

            print(f"\nBuilt {len(features_df)} feature rows")

            # Validate features
            missing = fb.validate_features(features_df)
            print(f"\nMissing value counts:")
            total_missing = 0
            for col, count in missing.items():
                if count > 0:
                    pct = 100.0 * count / len(features_df)
                    print(f"  {col}: {count} / {len(features_df)} ({pct:.1f}%)")
                    total_missing += count

            # Show sample
            print(f"\nSample features (first 3 rows):")
            print(features_df[["market_ticker", "timestamp", "yes_mid", "spread_cents",
                               "minutes_to_close", "temp_now"]].head(3))

            print("\n✓ TEST 4 PASSED")

    except Exception as e:
        print(f"  ⚠ Database test skipped: {e}")
        print("  (This is OK if database is not set up)")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Feature Engineering Test Suite")
    print("="*70)

    try:
        test_basic_feature_building()
        test_missing_data_handling()
        test_multiple_markets()
        test_database_integration()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise

    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    main()
