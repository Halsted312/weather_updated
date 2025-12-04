#!/usr/bin/env python3
"""
Test script for validating feature implementation before and after changes.

This script:
1. Tests that data is correctly populated in the database
2. Tests each feature group function with sample data
3. Validates the full pipeline from raw data → features

Run this script BEFORE and AFTER implementing features to verify correctness.

Usage:
    python scripts/test_feature_implementation.py --check-data
    python scripts/test_feature_implementation.py --test-features
    python scripts/test_feature_implementation.py --all
"""

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import get_db_session
from sqlalchemy import text


def check_data_availability():
    """Check what data is available in the database."""
    print("\n" + "=" * 70)
    print("DATA AVAILABILITY CHECK")
    print("=" * 70)

    with get_db_session() as session:
        # VcLocation entries
        result = session.execute(text("""
            SELECT city_code, location_type, vc_location_query
            FROM wx.vc_location
            ORDER BY city_code, location_type
        """)).fetchall()
        print("\n[VcLocation entries]")
        for row in result:
            print(f"  {row[0]:5} | {row[1]:8} | {row[2]}")

        # Minute data counts by location
        result = session.execute(text("""
            SELECT
                l.city_code,
                l.location_type,
                m.data_type,
                COUNT(*) as cnt,
                MIN(m.datetime_local::date) as min_date,
                MAX(m.datetime_local::date) as max_date
            FROM wx.vc_minute_weather m
            JOIN wx.vc_location l ON m.vc_location_id = l.id
            GROUP BY l.city_code, l.location_type, m.data_type
            ORDER BY l.city_code, l.location_type, m.data_type
        """)).fetchall()
        print("\n[VcMinuteWeather by location/type]")
        print(f"  {'City':5} | {'LocType':8} | {'DataType':20} | {'Count':>10} | Date Range")
        print("  " + "-" * 75)
        for row in result:
            print(f"  {row[0]:5} | {row[1]:8} | {row[2]:20} | {row[3]:>10,} | {row[4]} to {row[5]}")

        # Field population in minute forecasts
        result = session.execute(text("""
            SELECT
                COUNT(*) as total,
                COUNT(temp_f) as temp,
                COUNT(humidity) as humidity,
                COUNT(dew_f) as dew,
                COUNT(cloudcover) as cloudcover,
                COUNT(windspeed_mph) as wind,
                COUNT(solarradiation) as solar
            FROM wx.vc_minute_weather
            WHERE data_type = 'historical_forecast'
        """)).fetchone()
        print("\n[Minute historical_forecast field population]")
        print(f"  Total rows:  {result[0]:>10,}")
        print(f"  temp_f:      {result[1]:>10,} ({100*result[1]/result[0]:.1f}%)")
        print(f"  humidity:    {result[2]:>10,} ({100*result[2]/result[0]:.1f}%)")
        print(f"  dew_f:       {result[3]:>10,} ({100*result[3]/result[0]:.1f}%)")
        print(f"  cloudcover:  {result[4]:>10,} ({100*result[4]/result[0]:.1f}%) <- NOT in minute API!")
        print(f"  windspeed:   {result[5]:>10,} ({100*result[5]/result[0]:.1f}%)")
        print(f"  solar:       {result[6]:>10,} ({100*result[6]/result[0]:.1f}%)")

        # Field population in hourly forecasts
        result = session.execute(text("""
            SELECT
                COUNT(*) as total,
                COUNT(temp_f) as temp,
                COUNT(humidity) as humidity,
                COUNT(dew_f) as dew,
                COUNT(cloudcover) as cloudcover
            FROM wx.vc_forecast_hourly
            WHERE data_type = 'historical_forecast'
        """)).fetchone()
        print("\n[Hourly historical_forecast field population]")
        print(f"  Total rows:  {result[0]:>10,}")
        print(f"  temp_f:      {result[1]:>10,} ({100*result[1]/result[0]:.1f}%)")
        print(f"  humidity:    {result[2]:>10,} ({100*result[2]/result[0]:.1f}%)")
        print(f"  dew_f:       {result[3]:>10,} ({100*result[3]/result[0]:.1f}%)")
        print(f"  cloudcover:  {result[4]:>10,} ({100*result[4]/result[0]:.1f}%) <- Available here!")

        # Multi-lead coverage in daily forecasts
        result = session.execute(text("""
            SELECT
                lead_days,
                COUNT(*) as cnt
            FROM wx.vc_forecast_daily
            WHERE data_type = 'historical_forecast'
            GROUP BY lead_days
            ORDER BY lead_days
        """)).fetchall()
        print("\n[Daily historical_forecast lead_days coverage]")
        for row in result:
            print(f"  Lead {row[0]}: {row[1]:>8,} rows")

    print("\n" + "=" * 70)
    print("DATA CHECK COMPLETE")
    print("=" * 70 + "\n")


def test_feature_group_1_integer_boundary():
    """Test Feature Group 1: Integer boundary features."""
    print("\n[Feature Group 1: Integer Boundary]")

    # Import the function (will fail if not implemented yet)
    try:
        from models.features.forecast import compute_forecast_static_features
    except ImportError as e:
        print(f"  ERROR: Cannot import compute_forecast_static_features: {e}")
        return False

    # Test cases
    test_cases = [
        # (fcst_series, expected_max, expected_distance_to_int, expected_near_boundary)
        ([79.2, 82.7, 84.7, 84.7], 84.7, 0.3, False),  # 84.7 → distance 0.3, NOT near boundary (0.3 >= 0.25)
        ([80.0, 85.0, 90.0], 90.0, 0.0, True),         # 90.0 → distance 0.0, on boundary
        ([75.5, 80.5, 82.5], 82.5, 0.5, False),        # 82.5 → distance 0.5, not near boundary
        ([70.1, 75.1, 79.1], 79.1, 0.1, True),         # 79.1 → distance 0.1, near boundary
        ([70.9, 75.9, 79.9], 79.9, 0.1, True),         # 79.9 → distance 0.1, near boundary
    ]

    all_passed = True
    for fcst_series, exp_max, exp_dist, exp_near in test_cases:
        fs = compute_forecast_static_features(fcst_series)
        f = fs.to_dict()

        # Check max
        if abs(f["fcst_prev_max_f"] - exp_max) > 0.001:
            print(f"  FAIL: max expected {exp_max}, got {f['fcst_prev_max_f']}")
            all_passed = False

        # Check distance_to_int (new feature)
        if "fcst_prev_distance_to_int" not in f:
            print(f"  SKIP: fcst_prev_distance_to_int not implemented yet")
            return None  # Not implemented

        if abs(f["fcst_prev_distance_to_int"] - exp_dist) > 0.001:
            print(f"  FAIL: distance expected {exp_dist}, got {f['fcst_prev_distance_to_int']}")
            all_passed = False

        # Check near_boundary_flag
        if "fcst_prev_near_boundary_flag" not in f:
            print(f"  SKIP: fcst_prev_near_boundary_flag not implemented yet")
            return None

        actual_near = f["fcst_prev_near_boundary_flag"] == 1.0
        if actual_near != exp_near:
            print(f"  FAIL: near_boundary expected {exp_near}, got {actual_near}")
            all_passed = False

    if all_passed:
        print("  PASS: All integer boundary tests passed")
    return all_passed


def test_feature_group_2_peak_window():
    """Test Feature Group 2: Peak window features."""
    print("\n[Feature Group 2: Peak Window]")

    try:
        from models.features.forecast import compute_forecast_peak_window_features
    except ImportError:
        print("  SKIP: compute_forecast_peak_window_features not implemented yet")
        return None

    # Create test data: temps ramping up to peak at 2pm, then down
    base_date = datetime(2024, 12, 1, 6, 0, 0)
    temps = []
    timestamps = []

    # 6am-2pm: warming (60 → 90)
    for hour in range(8):  # 6-13
        temps.append(60 + hour * 3.75)  # 60, 63.75, 67.5, ... 86.25
        timestamps.append(base_date + timedelta(hours=hour))

    # 2pm-3pm: peak plateau at 90
    for minute in range(0, 60, 15):
        temps.append(90.0)
        timestamps.append(datetime(2024, 12, 1, 14, minute, 0))

    # 3pm-6pm: cooling
    for hour in range(3):
        temps.append(90 - (hour + 1) * 5)  # 85, 80, 75
        timestamps.append(datetime(2024, 12, 1, 15 + hour, 0, 0))

    fs = compute_forecast_peak_window_features(temps, timestamps, step_minutes=15)
    f = fs.to_dict()

    all_passed = True

    # Peak temp should be 90
    if f["fcst_peak_temp_f"] != 90.0:
        print(f"  FAIL: peak_temp expected 90.0, got {f['fcst_peak_temp_f']}")
        all_passed = False

    # Peak hour should be ~14.0 (2pm)
    if f["fcst_peak_hour_float"] is not None:
        if abs(f["fcst_peak_hour_float"] - 14.0) > 0.5:
            print(f"  FAIL: peak_hour expected ~14.0, got {f['fcst_peak_hour_float']}")
            all_passed = False

    # Peak band width should be ~60 min (4 x 15-min samples at 90°)
    if f["fcst_peak_band_width_min"] is not None:
        if f["fcst_peak_band_width_min"] < 45 or f["fcst_peak_band_width_min"] > 75:
            print(f"  FAIL: band_width expected ~60, got {f['fcst_peak_band_width_min']}")
            all_passed = False

    if all_passed:
        print("  PASS: All peak window tests passed")
    return all_passed


def test_feature_group_3_drift():
    """Test Feature Group 3: Forecast drift features."""
    print("\n[Feature Group 3: Forecast Drift]")

    try:
        from models.features.forecast import compute_forecast_drift_features
    except ImportError:
        print("  SKIP: compute_forecast_drift_features not implemented yet")
        return None

    # Test case: forecasts trending upward (older forecasts lower)
    df = pd.DataFrame({
        "lead_days": [6, 5, 4, 3, 2, 1, 0],
        "tempmax_f": [80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0],
    })

    fs = compute_forecast_drift_features(df)
    f = fs.to_dict()

    all_passed = True

    # Num leads = 7
    if f["fcst_drift_num_leads"] != 7.0:
        print(f"  FAIL: num_leads expected 7, got {f['fcst_drift_num_leads']}")
        all_passed = False

    # Std should be ~2.16 (std of 80-86)
    if f["fcst_drift_std_f"] is not None:
        if abs(f["fcst_drift_std_f"] - 2.16) > 0.1:
            print(f"  WARN: std expected ~2.16, got {f['fcst_drift_std_f']}")

    # Slope should be negative (forecasts decrease as lead increases)
    if f["fcst_drift_slope_f_per_lead"] is not None:
        if f["fcst_drift_slope_f_per_lead"] >= 0:
            print(f"  FAIL: slope should be negative, got {f['fcst_drift_slope_f_per_lead']}")
            all_passed = False

    if all_passed:
        print("  PASS: All drift tests passed")
    return all_passed


def test_feature_group_4_multivar():
    """Test Feature Group 4: Multivar static features."""
    print("\n[Feature Group 4: Multivar Static]")

    try:
        from models.features.forecast import compute_forecast_multivar_static_features
    except ImportError:
        print("  SKIP: compute_forecast_multivar_static_features not implemented yet")
        return None

    # Create test DataFrame
    timestamps = [datetime(2024, 12, 1, h, 0, 0) for h in range(6, 19)]  # 6am-6pm
    df = pd.DataFrame({
        "datetime_local": timestamps,
        "temp_f": [50 + i * 3 for i in range(13)],
        "humidity": [80 - i * 2 for i in range(13)],  # 80, 78, 76, ...
        "dew_f": [45 + i for i in range(13)],
        "cloudcover": [20 + i * 5 for i in range(13)],  # 20, 25, 30, ...
    })

    fs = compute_forecast_multivar_static_features(df)
    f = fs.to_dict()

    all_passed = True

    # Check humidity mean
    expected_hum_mean = np.mean([80 - i * 2 for i in range(13)])
    if f["fcst_humidity_mean"] is not None:
        if abs(f["fcst_humidity_mean"] - expected_hum_mean) > 0.1:
            print(f"  FAIL: humidity_mean expected {expected_hum_mean:.1f}, got {f['fcst_humidity_mean']}")
            all_passed = False

    # Check cloudcover range
    expected_cc_range = (20 + 12 * 5) - 20  # 60
    if f["fcst_cloudcover_range"] is not None:
        if abs(f["fcst_cloudcover_range"] - expected_cc_range) > 0.1:
            print(f"  FAIL: cloudcover_range expected {expected_cc_range}, got {f['fcst_cloudcover_range']}")
            all_passed = False

    if all_passed:
        print("  PASS: All multivar tests passed")
    return all_passed


def test_database_loader_functions():
    """Test new loader functions."""
    print("\n[Database Loader Functions]")

    try:
        from models.data.loader import load_historical_forecast_daily_multi
    except ImportError:
        print("  SKIP: load_historical_forecast_daily_multi not implemented yet")
        return None

    with get_db_session() as session:
        # Find a date that has multi-lead data for Austin (city location)
        result = session.execute(text("""
            SELECT d.target_date, COUNT(DISTINCT d.lead_days) as leads
            FROM wx.vc_forecast_daily d
            JOIN wx.vc_location l ON d.vc_location_id = l.id
            WHERE d.data_type = 'historical_forecast'
              AND l.city_code = 'AUS'
              AND l.location_type = 'city'
            GROUP BY d.target_date
            HAVING COUNT(DISTINCT d.lead_days) > 1
            ORDER BY d.target_date DESC
            LIMIT 1
        """)).fetchone()

        if not result:
            print("  SKIP: No multi-lead data found for Austin city")
            return None

        test_date = result[0]
        expected_leads = result[1]

        df = load_historical_forecast_daily_multi(session, "austin", test_date)

        if df.empty:
            print(f"  FAIL: No data returned for {test_date}")
            return False

        if len(df) < expected_leads:
            print(f"  WARN: Expected {expected_leads} leads, got {len(df)}")

        print(f"  PASS: Loaded {len(df)} leads for {test_date}")
        return True


def test_base_feature_cols():
    """Test that new features are registered in NUMERIC_FEATURE_COLS."""
    print("\n[Feature Registration Check]")

    from models.features.base import NUMERIC_FEATURE_COLS

    new_features = [
        # Group 1
        "fcst_prev_distance_to_int",
        "fcst_prev_near_boundary_flag",
        # Group 2
        "fcst_peak_temp_f",
        "fcst_peak_hour_float",
        "fcst_peak_band_width_min",
        "fcst_peak_step_minutes",
        # Group 3
        "fcst_drift_num_leads",
        "fcst_drift_std_f",
        "fcst_drift_max_upside_f",
        "fcst_drift_max_downside_f",
        "fcst_drift_mean_delta_f",
        "fcst_drift_slope_f_per_lead",
        # Group 4
        "fcst_humidity_mean",
        "fcst_humidity_min",
        "fcst_humidity_max",
        "fcst_humidity_range",
        "fcst_cloudcover_mean",
        "fcst_cloudcover_min",
        "fcst_cloudcover_max",
        "fcst_cloudcover_range",
        "fcst_dewpoint_mean",
        "fcst_dewpoint_min",
        "fcst_dewpoint_max",
        "fcst_dewpoint_range",
        "fcst_humidity_morning_mean",
        "fcst_humidity_afternoon_mean",
    ]

    missing = []
    present = []
    for feat in new_features:
        if feat in NUMERIC_FEATURE_COLS:
            present.append(feat)
        else:
            missing.append(feat)

    if present:
        print(f"  Present: {len(present)} features registered")
    if missing:
        print(f"  Missing: {len(missing)} features not yet registered:")
        for feat in missing[:5]:
            print(f"    - {feat}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")

    return len(missing) == 0


def run_all_tests():
    """Run all tests and summarize results."""
    print("\n" + "=" * 70)
    print("FEATURE IMPLEMENTATION TEST SUITE")
    print("=" * 70)

    results = {}

    results["Data Check"] = check_data_availability() is None  # Just prints info
    results["Group 1: Integer Boundary"] = test_feature_group_1_integer_boundary()
    results["Group 2: Peak Window"] = test_feature_group_2_peak_window()
    results["Group 3: Drift"] = test_feature_group_3_drift()
    results["Group 4: Multivar"] = test_feature_group_4_multivar()
    results["Loader Functions"] = test_database_loader_functions()
    results["Feature Registration"] = test_base_feature_cols()

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, result in results.items():
        if result is None:
            status = "SKIP"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {name:30} {status}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test feature implementation")
    parser.add_argument("--check-data", action="store_true", help="Check data availability")
    parser.add_argument("--test-features", action="store_true", help="Test feature functions")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    if args.check_data:
        check_data_availability()
    elif args.test_features:
        test_feature_group_1_integer_boundary()
        test_feature_group_2_peak_window()
        test_feature_group_3_drift()
        test_feature_group_4_multivar()
        test_base_feature_cols()
    elif args.all:
        run_all_tests()
    else:
        # Default: run all
        run_all_tests()


if __name__ == "__main__":
    main()
