"""Unit tests covering dataset + feature-building helpers."""

import sys
import os
from datetime import date, datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from zoneinfo import ZoneInfo

from ml.dataset import CITY_CONFIG
from ml.date_utils import event_date_from_close_time, series_ticker_for_city
from ml.features import FeatureBuilder


TARGET_CITIES = {"chicago", "miami", "austin", "la", "denver", "philadelphia"}


def test_city_config_has_only_target_cities():
    """CITY_CONFIG should only expose the six tradable cities."""
    assert set(CITY_CONFIG.keys()) == TARGET_CITIES


def test_market_features_leave_spread_nan_without_quotes():
    """When only OHLC data exists, spreads should remain NaN (no fake quotes)."""
    fb = FeatureBuilder()
    df = pd.DataFrame(
        {
            "market_ticker": ["KXHIGHCHI-TEST", "KXHIGHCHI-TEST"],
            "close": [60.0, 61.0],
            "high": [90.0, 95.0],
            "low": [10.0, 5.0],
        }
    )

    result = fb._add_market_features(df.copy())

    assert result["yes_mid"].tolist() == [60.0, 61.0]
    assert result["spread_cents"].isna().all()
    assert result["yes_bid"].isna().all()
    assert result["yes_ask"].isna().all()


def test_weather_features_use_minute_level_temp():
    """Minute-level Visual Crossing temps should map directly into temp_now."""
    fb = FeatureBuilder(city_timezone="America/Chicago")
    ts = pd.date_range("2025-08-01 12:00", periods=2, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "market_ticker": ["KXHIGHCHI-TEST", "KXHIGHCHI-TEST"],
            "timestamp": ts,
            "timestamp_local": ts.tz_convert(ZoneInfo("America/Chicago")),
            "temp_f": [70.0, 72.5],
        }
    )

    metadata = pd.DataFrame(
        {
            "market_ticker": ["KXHIGHCHI-TEST"],
            "strike_type": ["between"],
            "floor_strike": [65.0],
            "cap_strike": [85.0],
        }
    )

    result = fb._add_weather_features(df.copy(), weather_df=None, market_metadata=metadata)

    assert result["temp_now"].tolist() == [70.0, 72.5]
    assert result["temp_to_floor"].tolist() == [ -5.0, -7.5]
    assert result["temp_to_cap"].tolist() == [15.0, 12.5]


def test_event_date_from_close_time_handles_dst_offsets():
    """event_date helper should align UTC close times to city-local calendar days."""
    chicago_series = series_ticker_for_city("chicago")
    close_time = datetime(2024, 3, 10, 5, 0, tzinfo=timezone.utc)
    assert event_date_from_close_time(chicago_series, close_time) == date(2024, 3, 9)

    la_series = series_ticker_for_city("la")
    la_close = datetime(2024, 1, 2, 7, 0, tzinfo=timezone.utc)
    assert event_date_from_close_time(la_series, la_close) == date(2024, 1, 1)
