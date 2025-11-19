#!/usr/bin/env python3
"""
IEM CF6 (Preliminary Local Climatological Data) Fetcher.

Fetches official NWS CF6 climate data via IEM's JSON API.
CF6 is the "Preliminary Local Climatological Data" (WS Form F-6) product
that contains daily MAX/MIN temperatures from NWS climate observations.

Data Source: https://mesonet.agron.iastate.edu/json/cf6.py
Documentation: https://mesonet.agron.iastate.edu/json/cf6.py?help=

Key Differences from ADS:
- CF6 is an OFFICIAL NWS climate product (settlement-grade)
- ADS is COMPUTED from ASOS observations (analytics-grade)
- CF6 day definition: Midnight local standard time (1AM-1AM during DST)
- Temperatures are INTEGER °F (no decimals)

Settlement Precedence:
CLI (Daily Climate Report) > CF6 (this module) > IEM_CF6 > GHCND > ADS > VC

Station Mappings (ICAO 4-letter codes):
- Chicago: KMDW (Midway)
- New York: KNYC (Central Park)
- Los Angeles: KLAX
- Denver: KDEN
- Austin: KAUS
- Miami: KMIA
- Philadelphia: KPHL
"""

from __future__ import annotations
import datetime as dt
import io
import time
import logging
from typing import Dict, List
import requests
import pandas as pd

logger = logging.getLogger(__name__)

CF6_JSON_URL = "https://mesonet.agron.iastate.edu/json/cf6.py"

# Map cities to ICAO 4-letter station codes (used by CF6)
CITY_TO_CF6_STATION: Dict[str, str] = {
    "chicago":      "KMDW",
    "new_york":     "KNYC",
    "los_angeles":  "KLAX",
    "denver":       "KDEN",
    "austin":       "KAUS",
    "miami":        "KMIA",
    "philadelphia": "KPHL",
}


def fetch_cf6_year(
    station: str,
    year: int,
    fmt: str = "csv",
    pause_s: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch CF6 daily climate data for a station/year from IEM.

    Args:
        station: ICAO 4-letter station code (e.g., "KMDW")
        year: Year to fetch (e.g., 2024)
        fmt: Format ("csv" or "json")
        pause_s: Pause between retries (seconds)

    Returns:
        DataFrame with columns: [station, day, tmax_cf6_f]
        - day: date (datetime.date)
        - tmax_cf6_f: integer °F from CF6

    Raises:
        RuntimeError: If fetch fails after 3 attempts

    Notes:
        - CF6 JSON API returns columns: station, valid (date), high, low, etc.
        - "high" is the daily maximum temperature in °F (integer)
        - Missing values are returned as NaN
    """
    params = {"station": station, "year": year, "fmt": "csv"}

    for attempt in range(3):
        try:
            resp = requests.get(CF6_JSON_URL, params=params, timeout=30)
            resp.raise_for_status()

            text = resp.text.strip()
            if not text:
                raise RuntimeError(f"Empty CF6 response for {station}, {year}")

            # Parse CSV
            df = pd.read_csv(io.StringIO(text), index_col=False)

            # Expected columns: station, valid, high, low, precip, etc.
            if "valid" not in df.columns or "high" not in df.columns:
                raise RuntimeError(f"Unexpected CF6 columns: {df.columns.tolist()}")

            # Normalize columns
            df["valid"] = pd.to_datetime(df["valid"], errors="coerce").dt.date
            df["high"] = pd.to_numeric(df["high"], errors="coerce").round(0).astype("Int64")

            # Select and rename columns
            df = df[["station", "valid", "high"]].rename(
                columns={"valid": "day", "high": "tmax_cf6_f"}
            )

            logger.info(f"Fetched {len(df)} days from CF6: {station} {year}")

            return df

        except Exception as e:
            logger.warning(f"CF6 fetch attempt {attempt + 1}/3 failed for {station}/{year}: {e}")
            if attempt < 2:
                time.sleep(pause_s * (attempt + 1))

    raise RuntimeError(f"CF6 fetch failed for {station}, {year}")


def fetch_cf6_range(
    station: str,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """
    Fetch CF6 daily highs for [start, end] across year boundaries.

    Args:
        station: ICAO 4-letter station code (e.g., "KMDW")
        start: Start date (inclusive)
        end: End date (inclusive)

    Returns:
        DataFrame with columns: [station, day, tmax_cf6_f]

    Example:
        >>> df = fetch_cf6_range("KMDW", dt.date(2024, 1, 1), dt.date(2024, 12, 31))
        >>> len(df) > 300  # Should have most days of the year
        True
    """
    frames: List[pd.DataFrame] = []

    for year in range(start.year, end.year + 1):
        logger.info(f"Fetching CF6 for {station}, year {year}...")

        try:
            df = fetch_cf6_year(station, year, fmt="csv")
            frames.append(df)
        except Exception as e:
            logger.error(f"Failed to fetch {station}/{year}: {e}")
            # Continue to next year instead of failing completely
            continue

    if not frames:
        logger.warning(f"No CF6 data fetched for {station}")
        return pd.DataFrame(columns=["station", "day", "tmax_cf6_f"])

    # Combine all years
    out = pd.concat(frames, ignore_index=True)

    # Filter to requested date range
    out = out[(out["day"] >= start) & (out["day"] <= end)]

    # Sort by date
    out = out.sort_values("day").reset_index(drop=True)

    logger.info(f"Total CF6 records for {station}: {len(out)} days")

    return out


def fetch_cities(
    cities: list[str],
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """
    Fetch CF6 daily highs for multiple cities.

    Args:
        cities: City names (must be in CITY_TO_CF6_STATION mapping)
        start: Start date (inclusive)
        end: End date (inclusive)

    Returns:
        DataFrame with columns: [city, station, day, tmax_cf6_f]

    Example:
        >>> df = fetch_cities(["chicago"], dt.date(2024, 11, 12), dt.date(2024, 11, 13))
        >>> len(df) == 2  # Two days
        True
    """
    frames = []

    for city in cities:
        if city not in CITY_TO_CF6_STATION:
            logger.error(f"Unknown city: {city}. Must be one of {list(CITY_TO_CF6_STATION.keys())}")
            continue

        station = CITY_TO_CF6_STATION[city]
        logger.info(f"Fetching CF6 for {city} ({station})...")

        try:
            df = fetch_cf6_range(station, start, end)
            df.insert(0, "city", city)
            frames.append(df)
        except Exception as e:
            logger.error(f"Failed to fetch {city}: {e}")
            continue

    if not frames:
        logger.warning("No CF6 data fetched for any city")
        return pd.DataFrame(columns=["city", "station", "day", "tmax_cf6_f"])

    # Combine all cities
    out = pd.concat(frames, ignore_index=True)

    # Sort by city and date
    out = out.sort_values(["city", "day"]).reset_index(drop=True)

    logger.info(f"Total: {len(out)} CF6 records across {len(frames)} cities")

    return out


if __name__ == "__main__":
    import argparse

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Fetch IEM CF6 (NWS Preliminary Climate Data)"
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=list(CITY_TO_CF6_STATION.keys()),
        choices=list(CITY_TO_CF6_STATION.keys()),
        help="Cities to fetch (default: all)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file (optional)",
    )

    args = parser.parse_args()

    # Parse dates
    start = dt.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(args.end_date, "%Y-%m-%d").date()

    print(f"\nFetching IEM CF6 data:")
    print(f"  Cities: {', '.join(args.cities)}")
    print(f"  Date range: {start} to {end}")
    print(f"  Days: {(end - start).days + 1}")
    print()

    # Fetch data
    df = fetch_cities(args.cities, start, end)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total records: {len(df)}")
    print(f"\nBy city:")
    print(df.groupby("city").size())
    print(f"\nMissing values:")
    print(df["tmax_cf6_f"].isna().sum())
    print("\nSample data:")
    print(df.head(10))

    # Save if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to: {args.output}")

    print("="*60)
