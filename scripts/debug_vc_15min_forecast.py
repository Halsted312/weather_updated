#!/usr/bin/env python3
"""
Debug script: Visual Crossing 15-minute historical forecast API validation.

Purpose:
  - Fetch 1 day of 15-minute forecast data for Austin
  - Validate API response structure
  - Verify we get forecast data (not observations)
  - Check minute spacing and field availability

Usage:
  python scripts/debug_vc_15min_forecast.py [--target-date YYYY-MM-DD]
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from typing import Dict, Any, List

import requests

from src.config.cities import get_city


def fetch_historical_forecast_15min(
    latitude: float,
    longitude: float,
    target_date: date,
    forecast_basis_day: int = 1,
    api_key: str | None = None,
) -> Dict[str, Any]:
    """
    Fetch 15-minute historical forecast from Visual Crossing.

    Args:
        latitude: Station latitude
        longitude: Station longitude
        target_date: Target date to forecast
        forecast_basis_day: Days before target (1 = T-1 forecast)
        api_key: VC API key (from env if not provided)

    Returns:
        Raw JSON response from Visual Crossing
    """
    api_key = api_key or os.environ.get("VISUAL_CROSSING_API_KEY")
    if not api_key:
        raise RuntimeError(
            "VISUAL_CROSSING_API_KEY not set. "
            "Set via environment variable or pass as argument."
        )

    # Build location string (lat,lon format)
    location = f"{latitude},{longitude}"

    # Build URL for single target date
    target_str = target_date.strftime("%Y-%m-%d")
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    url = f"{base_url}/{location}/{target_str}/{target_str}"

    # Parameters - KEY: include minutes for 15-min data
    params = {
        "unitGroup": "us",
        "include": "days,hours,minutes",  # ← This is what we're testing!
        "forecastBasisDay": forecast_basis_day,  # T-1 forecast
        "key": api_key,
        "contentType": "json",
    }

    print(f"Fetching from: {url}")
    print(f"Parameters: {params}")
    print()

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


def validate_response(payload: Dict[str, Any], target_date: date) -> None:
    """
    Validate the API response structure and print summary.

    Checks:
      - minutes array exists and has ~96 entries (24 hours × 4 per hour)
      - Each minute has required fields (datetime, temp, humidity, etc.)
      - Timestamps are 15-min apart
      - source != "obs" (confirms forecast, not observation)
    """
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()

    # Basic structure
    print(f"📍 Location: {payload.get('resolvedAddress', 'N/A')}")
    print(f"🌐 Timezone: {payload.get('timezone', 'N/A')}")
    print(f"📅 Query cost: {payload.get('queryCost', 'N/A')} credits")
    print()

    # Days
    days = payload.get("days", [])
    if not days:
        print("❌ ERROR: No 'days' array in response")
        return

    if len(days) != 1:
        print(f"⚠️  WARNING: Expected 1 day, got {len(days)}")

    day = days[0]
    day_date = day.get("datetime", "N/A")
    print(f"📆 Day datetime: {day_date}")
    print(f"   Expected: {target_date}")
    print(f"   Match: {'✅' if day_date == str(target_date) else '❌'}")
    print()

    # Hours
    hours = day.get("hours", [])
    print(f"⏰ Hours array: {len(hours)} entries")
    if len(hours) != 24:
        print(f"   ⚠️  Expected 24 hours, got {len(hours)}")
    print()

    # Minutes (the key part we're testing!)
    total_minutes = 0
    minute_counts = []
    sample_minutes: List[Dict] = []

    for hour in hours:
        minutes = hour.get("minutes", [])
        minute_counts.append(len(minutes))
        total_minutes += len(minutes)

        # Collect sample from first hour
        if hour.get("datetime") == "00:00:00" and minutes:
            sample_minutes = minutes[:4]  # First 4 minutes (00:00, 00:15, 00:30, 00:45)

    print(f"⏱️  Total minutes: {total_minutes}")
    print(f"   Expected: ~96 (24 hours × 4 per hour for 15-min data)")
    print(f"   Match: {'✅' if 90 <= total_minutes <= 100 else '❌'}")
    print()

    print(f"📊 Minutes per hour: {set(minute_counts)}")
    if len(set(minute_counts)) == 1 and list(set(minute_counts))[0] == 4:
        print("   ✅ Consistent 4 minutes/hour (15-min intervals)")
    else:
        print(f"   ⚠️  Inconsistent spacing: {minute_counts}")
    print()

    # Sample minute data
    if sample_minutes:
        print("🔍 Sample minutes (first hour):")
        print()
        for i, minute in enumerate(sample_minutes, 1):
            dt = minute.get("datetime", "N/A")
            temp = minute.get("temp", "N/A")
            humidity = minute.get("humidity", "N/A")
            cloudcover = minute.get("cloudcover", "N/A")
            source = minute.get("source", "N/A")

            print(f"  Minute {i}:")
            print(f"    datetime:   {dt}")
            print(f"    temp:       {temp}°F")
            print(f"    humidity:   {humidity}%")
            print(f"    cloudcover: {cloudcover}%")
            print(f"    source:     {source}")

            # Validate source is NOT 'obs'
            if source == "obs":
                print(f"    ❌ ERROR: source='obs' (should be forecast data!)")
            elif source in ("fcst", "forecast", "comb"):
                print(f"    ✅ Correct: source indicates forecast")
            else:
                print(f"    ⚠️  Unexpected source value: '{source}'")
            print()
    else:
        print("❌ ERROR: No sample minutes found in first hour")
        print()

    # Verify 15-minute spacing
    if len(sample_minutes) >= 2:
        times = [minute.get("datetime") for minute in sample_minutes]
        print(f"🕐 Timestamp sequence (first 4): {times}")
        expected = ["00:00:00", "00:15:00", "00:30:00", "00:45:00"]
        if times == expected:
            print("   ✅ Perfect 15-minute spacing")
        else:
            print(f"   ❌ ERROR: Expected {expected}")
        print()

    # Check available fields in first minute
    if sample_minutes:
        first_minute = sample_minutes[0]
        print("📋 Available fields in first minute:")
        fields = sorted(first_minute.keys())
        for field in fields:
            value = first_minute[field]
            value_str = str(value)[:50]  # Truncate long values
            print(f"   - {field:20s} = {value_str}")
        print()

        # Check for key weather fields
        required_fields = ["datetime", "temp", "humidity", "cloudcover", "dew"]
        optional_fields = ["precip", "windspeed", "uvindex", "cape", "cin"]

        print("✅ Required fields:")
        for field in required_fields:
            status = "✅" if field in first_minute else "❌"
            print(f"   {status} {field}")
        print()

        print("📌 Optional fields:")
        for field in optional_fields:
            status = "✅" if field in first_minute else "⚠️ "
            print(f"   {status} {field}")
        print()

    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Debug VC 15-minute forecast API")
    parser.add_argument(
        "--target-date",
        type=str,
        default=None,
        help="Target date (YYYY-MM-DD). Default: yesterday",
    )
    parser.add_argument(
        "--city",
        type=str,
        default="austin",
        choices=["austin", "chicago", "denver", "los_angeles", "miami", "philadelphia"],
        help="City to test (default: austin)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save raw JSON response to file",
    )
    args = parser.parse_args()

    # Get city config
    city = get_city(args.city)
    print(f"Testing city: {city.city_id.upper()}")
    print(f"Station: {city.icao}")
    print(f"Coordinates: {city.latitude}, {city.longitude}")
    print()

    # Determine target date
    if args.target_date:
        target_date = datetime.strptime(args.target_date, "%Y-%m-%d").date()
    else:
        # Default: yesterday (ensures data is available)
        target_date = date.today() - timedelta(days=1)

    print(f"Target date: {target_date}")
    print(f"Forecast basis: T-1 (one day before target)")
    print()

    # Fetch data
    try:
        payload = fetch_historical_forecast_15min(
            latitude=city.latitude,
            longitude=city.longitude,
            target_date=target_date,
            forecast_basis_day=1,
        )
    except requests.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"Response: {e.response.text if e.response else 'N/A'}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    # Save raw JSON if requested
    if args.save_json:
        filename = f"vc_15min_debug_{args.city}_{target_date}.json"
        with open(filename, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"💾 Saved raw JSON to: {filename}")
        print()

    # Validate response
    validate_response(payload, target_date)

    print()
    print("✅ Debug script completed successfully!")
    print()
    print("Next step: Review output above and confirm:")
    print("  1. Total minutes ~96 (15-min spacing)")
    print("  2. source != 'obs' (actual forecast)")
    print("  3. All required fields present")
    print("  4. Temperature values are reasonable")


if __name__ == "__main__":
    main()
