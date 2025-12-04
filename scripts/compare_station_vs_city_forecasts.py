#!/usr/bin/env python3
"""
Compare Visual Crossing station vs city 15-minute historical forecasts.

Purpose:
  Compare station-anchored forecasts (lat/lon at settlement station) vs
  city-aggregate forecasts ("Austin,TX") for the same target dates.

  IMPORTANT: Uses `forecastBasisDay` parameter which returns forecasts.
  Previous versions using `stn:KXXX` were contaminated because that
  query type returns observations even with forecastBasisDay.

  The fix: Use lat/lon queries for station-anchored forecasts.

  This is CRITICAL because:
  - Kalshi settles on the STATION, not city average
  - Station-city forecast gaps may be predictive features
  - We need both series for robust modeling

Usage:
  python scripts/compare_station_vs_city_forecasts.py
"""

import sys
from datetime import date, datetime
from typing import Dict, Any, List, Tuple

import pandas as pd
import requests

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import get_settings
from src.config.cities import CITIES


def fetch_minute_forecast(
    location: str,
    target_date: date,
    forecast_basis_day: int,
    api_key: str,
) -> Dict[str, Any]:
    """
    Fetch 15-minute historical forecast from Visual Crossing.

    Args:
        location: VC location string (lat/lon, city, or stn:CODE)
        target_date: Target date
        forecast_basis_day: Days before target (1 = T-1)
        api_key: VC API key

    Returns:
        Raw JSON response
    """
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    target_str = target_date.strftime("%Y-%m-%d")
    url = f"{base_url}/{location}/{target_str}/{target_str}"

    params = {
        "unitGroup": "us",
        "include": "days,hours,minutes",  # forecastBasisDay ensures forecasts
        "forecastBasisDay": forecast_basis_day,
        "key": api_key,
        "contentType": "json",
    }

    print(f"Fetching {location} for {target_str} (T-{forecast_basis_day})...")
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    # Validate we got forecasts, not observations
    if data.get("days"):
        source = data["days"][0].get("source", "unknown")
        if source == "obs":
            print(f"  ⚠️ Warning: Got observations instead of forecasts for {location}")
        elif source == "fcst":
            pass  # Expected - this is a forecast
        else:
            print(f"  ℹ️ Source: {source} for {location}")

    return data


def extract_minute_temps(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract minute-level temperatures from VC response.

    Returns:
        DataFrame with columns: time (HH:MM), temp, humidity, dew
    """
    rows = []

    for day in payload.get("days", []):
        for hour in day.get("hours", []):
            for minute in hour.get("minutes", []):
                rows.append({
                    "time": minute.get("datetime", ""),
                    "temp": minute.get("temp"),
                    "humidity": minute.get("humidity"),
                    "dew": minute.get("dew"),
                    "pressure": minute.get("pressure"),
                    "windspeed": minute.get("windspeed"),
                })

    df = pd.DataFrame(rows)

    # Convert time to timedelta for easier analysis
    if not df.empty and "time" in df.columns:
        df["hour"] = df["time"].str.split(":").str[0].astype(int)
        df["minute"] = df["time"].str.split(":").str[1].astype(int)
        df["minutes_since_midnight"] = df["hour"] * 60 + df["minute"]

    return df


def compare_forecasts(
    station_df: pd.DataFrame,
    city_df: pd.DataFrame,
    target_date: date,
) -> Dict[str, Any]:
    """
    Compare station vs city forecast curves.

    Returns:
        Dict with comparison metrics
    """
    if station_df.empty or city_df.empty:
        return {
            "valid": False,
            "reason": "Missing data (station or city empty)",
        }

    # Merge on time
    merged = pd.merge(
        station_df[["time", "temp", "humidity", "dew"]].rename(
            columns={"temp": "temp_station", "humidity": "hum_station", "dew": "dew_station"}
        ),
        city_df[["time", "temp", "humidity", "dew"]].rename(
            columns={"temp": "temp_city", "humidity": "hum_city", "dew": "dew_city"}
        ),
        on="time",
        how="inner",
    )

    if merged.empty:
        return {
            "valid": False,
            "reason": "No matching timestamps",
        }

    # Calculate differences
    merged["temp_diff"] = merged["temp_station"] - merged["temp_city"]
    merged["hum_diff"] = merged["hum_station"] - merged["hum_city"]
    merged["dew_diff"] = merged["dew_station"] - merged["dew_city"]

    # Find forecast maxes
    station_max = merged["temp_station"].max()
    city_max = merged["temp_city"].max()

    station_max_time = merged.loc[merged["temp_station"].idxmax(), "time"]
    city_max_time = merged.loc[merged["temp_city"].idxmax(), "time"]

    return {
        "valid": True,
        "target_date": target_date,
        "n_points": len(merged),
        # Temperature differences
        "max_temp_station": station_max,
        "max_temp_city": city_max,
        "max_temp_diff": station_max - city_max,
        "time_of_max_station": station_max_time,
        "time_of_max_city": city_max_time,
        "temp_diff_mean": merged["temp_diff"].mean(),
        "temp_diff_std": merged["temp_diff"].std(),
        "temp_diff_max_abs": merged["temp_diff"].abs().max(),
        # Humidity differences
        "hum_diff_mean": merged["hum_diff"].mean(),
        "hum_diff_max_abs": merged["hum_diff"].abs().max(),
        # Dew point differences
        "dew_diff_mean": merged["dew_diff"].mean(),
        "dew_diff_max_abs": merged["dew_diff"].abs().max(),
    }


def main():
    """Compare station vs city forecasts for ALL 6 cities over 3 days each."""

    # Configuration
    test_dates = [date(2024, 11, 3), date(2024, 11, 4), date(2024, 11, 5)]

    settings = get_settings()
    api_key = settings.vc_api_key
    if not api_key:
        print("❌ vc_api_key not set in .env")
        return

    print("=" * 80)
    print("STATION VS CITY FORECAST COMPARISON - ALL 6 CITIES")
    print("=" * 80)
    print(f"Test dates: {test_dates}")
    print(f"Cities: {list(CITIES.keys())}")
    print("=" * 80)
    print()

    # Collect results per city
    all_city_results = {}

    for city_id, city in CITIES.items():
        print(f"\n{'#' * 80}")
        print(f"# CITY: {city_id.upper()} ({city.icao})")
        print(f"# Station query (lat/lon): {city.vc_latlon_query}")
        print(f"# City query: {city.vc_city_query}")
        print(f"{'#' * 80}")

        # Use lat/lon for station (NOT stn:KXXX which returns obs!)
        station_query = city.vc_latlon_query
        city_query = city.vc_city_query

        results = []

        for target_date in test_dates:
            print(f"\n  Testing: {target_date}")

            try:
                # Fetch station forecast
                station_payload = fetch_minute_forecast(
                    location=station_query,
                    target_date=target_date,
                    forecast_basis_day=1,
                    api_key=api_key,
                )
                station_df = extract_minute_temps(station_payload)

                # Fetch city forecast
                city_payload = fetch_minute_forecast(
                    location=city_query,
                    target_date=target_date,
                    forecast_basis_day=1,
                    api_key=api_key,
                )
                city_df = extract_minute_temps(city_payload)

                # Compare
                comparison = compare_forecasts(station_df, city_df, target_date)

                if comparison["valid"]:
                    comparison["city_id"] = city_id
                    results.append(comparison)
                    print(f"    Station max: {comparison['max_temp_station']:.1f}°F | "
                          f"City max: {comparison['max_temp_city']:.1f}°F | "
                          f"Diff: {comparison['max_temp_diff']:+.2f}°F")
                else:
                    print(f"    ❌ Comparison failed: {comparison.get('reason', 'Unknown')}")

            except Exception as e:
                print(f"    ❌ Error: {e}")

        # Store city results
        if results:
            all_city_results[city_id] = results

            # Per-city summary
            max_temp_diffs = [r["max_temp_diff"] for r in results]
            avg_diff = sum(max_temp_diffs) / len(max_temp_diffs)
            print(f"\n  📊 {city_id.upper()} Summary: Avg daily high diff = {avg_diff:+.2f}°F")

    # =========================================================================
    # CROSS-CITY SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("CROSS-CITY SUMMARY: STATION VS CITY FORECAST DIFFERENCES")
    print("=" * 80)
    print()

    print(f"{'City':<15} {'Avg High Diff':>15} {'Max High Diff':>15} {'Days':>6}")
    print("-" * 55)

    overall_diffs = []
    for city_id, results in all_city_results.items():
        max_temp_diffs = [r["max_temp_diff"] for r in results]
        avg_diff = sum(max_temp_diffs) / len(max_temp_diffs)
        max_diff = max(max_temp_diffs, key=abs)
        overall_diffs.extend(max_temp_diffs)
        print(f"{city_id:<15} {avg_diff:>+14.2f}°F {max_diff:>+14.2f}°F {len(results):>6}")

    print("-" * 55)

    if overall_diffs:
        overall_avg = sum(overall_diffs) / len(overall_diffs)
        overall_max = max(overall_diffs, key=abs)
        print(f"{'OVERALL':<15} {overall_avg:>+14.2f}°F {overall_max:>+14.2f}°F {len(overall_diffs):>6}")

        # Decision
        print("\n" + "=" * 80)
        print("DECISION:")
        print("=" * 80)

        avg_max_diff = abs(overall_avg)

        if avg_max_diff > 0.5:
            print("✅ STATION FORECASTS ADD MEANINGFUL INFORMATION")
            print(f"   • Overall average daily high difference: {avg_max_diff:.2f}°F")
            print("\n🎯 RECOMMENDATION:")
            print("   Station-city gaps are large enough to be valuable features.")
        else:
            print("⚠️  STATION AND CITY FORECASTS ARE VERY SIMILAR")
            print(f"   • Overall average daily high difference: {avg_max_diff:.2f}°F")
            print("\n💡 RECOMMENDATION:")
            print("   Station-city gap features likely provide minimal value.")
            print("   Focus on single-source forecasts (lat/lon or city query).")

    print()


if __name__ == "__main__":
    main()
