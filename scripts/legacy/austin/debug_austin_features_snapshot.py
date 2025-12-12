#!/usr/bin/env python3
"""
Debug Austin feature computation for sample dates.

Tests loaders + feature computation for NOAA guidance + z-scores.
"""

from datetime import date, datetime

from src.db.connection import get_db_session
from models.data.loader import (
    load_weather_more_apis_guidance,
    load_obs_t15_stats_30d,
    load_historical_forecast_daily,
)
from models.features.more_apis import compute_more_apis_features

# Test dates across different seasons
TEST_DATES = [
    date(2023, 2, 15),  # Winter
    date(2023, 7, 15),  # Summer
    date(2024, 1, 10),  # Winter
    date(2024, 7, 15),  # Summer
    date(2025, 6, 15),  # Recent summer
]

def main():
    print("\n" + "="*80)
    print("AUSTIN FEATURE SNAPSHOT VALIDATION")
    print("="*80 + "\n")

    with get_db_session() as session:
        for test_date in TEST_DATES:
            print(f"\n{'Test Date: ' + str(test_date):=^80}\n")

            # Load NOAA guidance
            more_apis = load_weather_more_apis_guidance(
                session, "austin", test_date, snapshot_time_utc=None
            )

            # Load 30-day obs stats
            obs_mean, obs_std = load_obs_t15_stats_30d(
                session, "austin", test_date, lookback_days=30
            )

            # Load VC T-1 forecast
            basis_date = date(test_date.year, test_date.month, test_date.day - 1) if test_date.day > 1 else None
            vc_daily = load_historical_forecast_daily(session, "austin", test_date, basis_date) if basis_date else None
            vc_t1_tempmax = vc_daily.get("tempmax_f") if vc_daily else None

            # Print loaded data
            print("NOAA Guidance:")
            for model in ["nbm", "hrrr", "ndfd"]:
                latest = more_apis[model]["latest_run"]
                prev = more_apis[model]["prev_run"]
                if latest:
                    print(f"  {model.upper()} latest: {latest['peak_window_max_f']}°F (run={latest['run_datetime_utc']})")
                else:
                    print(f"  {model.upper()} latest: (none)")
                if prev:
                    print(f"  {model.upper()} prev:   {prev['peak_window_max_f']}°F")

            print(f"\n30-day obs stats at 15:00:")
            print(f"  Mean: {obs_mean}°F" if obs_mean else "  Mean: None")
            print(f"  Std:  {obs_std}°F" if obs_std else "  Std:  None")

            print(f"\nVC T-1 forecast:")
            print(f"  Tempmax: {vc_t1_tempmax}°F" if vc_t1_tempmax else "  Tempmax: None")

            # Compute features
            print(f"\nComputed Features:")
            fs = compute_more_apis_features(
                more_apis,
                vc_t1_tempmax,
                obs_mean,
                obs_std
            )

            for name, value in sorted(fs.features.items()):
                if value is not None:
                    if "z_30d" in name:
                        print(f"  {name:45s} = {value:.3f} (z-score)")
                    else:
                        print(f"  {name:45s} = {value:.2f}°F")
                else:
                    print(f"  {name:45s} = None")

            print("\n" + "-"*80)

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
