#!/usr/bin/env python3
"""
Debug script to inspect NOAA model guidance data and computed features.

Tests the full stack:
1. Load guidance from database
2. Compute features via feature module
3. Verify feature values are reasonable

Usage:
    python scripts/debug_weather_more_apis_guidance.py --city austin --date 2025-06-15
"""

import argparse
from datetime import date, datetime

from src.db.connection import get_db_session
from src.db.models import WeatherMoreApisGuidance
from models.data.loader import load_weather_more_apis_guidance
from models.features.more_apis import compute_more_apis_features


def main():
    parser = argparse.ArgumentParser(description="Debug NOAA guidance features")
    parser.add_argument("--city", required=True, help="City ID (e.g., austin)")
    parser.add_argument("--date", required=True, help="Target date (YYYY-MM-DD)")
    args = parser.parse_args()

    city_id = args.city
    target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    print(f"\n{'='*80}")
    print(f"NOAA GUIDANCE DEBUG - {city_id.upper()} on {target_date}")
    print(f"{'='*80}\n")

    with get_db_session() as session:
        # Query raw database rows
        print("Raw database rows:")
        print("-" * 80)
        rows = (
            session.query(WeatherMoreApisGuidance)
            .filter(
                WeatherMoreApisGuidance.city_id == city_id,
                WeatherMoreApisGuidance.target_date == target_date,
            )
            .order_by(
                WeatherMoreApisGuidance.model,
                WeatherMoreApisGuidance.run_datetime_utc.desc(),
            )
            .all()
        )

        if not rows:
            print(f"  (No data found for {city_id} {target_date})")
            return

        for row in rows:
            print(
                f"  {row.model.upper():5s} | run={row.run_datetime_utc} | "
                f"peak_window={row.peak_window_max_f}°F"
            )

        # Test loader function
        print(f"\n{'Loader Output':=^80}")
        guidance = load_weather_more_apis_guidance(
            session, city_id, target_date, snapshot_time_utc=None
        )

        for model in ["nbm", "hrrr", "ndfd"]:
            print(f"\n  {model.upper()}:")
            latest = guidance[model]["latest_run"]
            prev = guidance[model]["prev_run"]

            if latest:
                print(
                    f"    Latest: {latest['run_datetime_utc']} → {latest['peak_window_max_f']}°F"
                )
            else:
                print("    Latest: (none)")

            if prev:
                print(
                    f"    Prev:   {prev['run_datetime_utc']} → {prev['peak_window_max_f']}°F"
                )
            else:
                print("    Prev:   (none)")

        # Test feature computation
        print(f"\n{'Computed Features':=^80}\n")
        vc_t1_tempmax = 90.0  # Mock VC T-1 forecast for testing
        feature_set = compute_more_apis_features(guidance, vc_t1_tempmax)

        for name, value in sorted(feature_set.features.items()):
            value_str = f"{value:.2f}°F" if value is not None else "None"
            print(f"  {name:45s} = {value_str}")

        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
