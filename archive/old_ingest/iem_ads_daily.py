#!/usr/bin/env python3
"""
IEM ASOS Daily Summary (ADS) Fetcher.

Fetches daily maximum temperature data from Iowa Environmental Mesonet's
ASOS Daily Summary service (daily.py endpoint).

Data Source: https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py
Documentation: https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py?help=

Important Notes:
- ADS data is COMPUTED from ASOS observations (5-min or hourly)
- Day definition: Usually local calendar day, but some stations use "midnight standard time" (1AM-1AM during DST)
- For settlement parity, prefer NWS CLI/CF6 climate products over ADS
- Use ADS for: bulk history, intraday analytics, feature engineering
- Use CLI/CF6 for: settlement validation, backtest outcomes

Station Mappings (FAA 3-letter codes):
- Chicago: MDW (Midway) - IL_ASOS network
- New York: NYC (Central Park) - NY_ASOS
- Los Angeles: LAX - CA_ASOS
- Denver: DEN - CO_ASOS
- Austin: AUS - TX_ASOS
- Miami: MIA - FL_ASOS
- Philadelphia: PHL - PA_ASOS
"""

from __future__ import annotations
import datetime as dt
import io
import time
import logging
from typing import Dict, Tuple, Iterable
import requests
import pandas as pd

logger = logging.getLogger(__name__)

IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"

# Map cities to their ASOS network + station
# Format: (network, station_3letter)
CITY_TO_ASOS: Dict[str, Tuple[str, str]] = {
    "chicago":      ("IL_ASOS", "MDW"),
    "new_york":     ("NY_ASOS", "NYC"),
    "los_angeles":  ("CA_ASOS", "LAX"),
    "denver":       ("CO_ASOS", "DEN"),
    "austin":       ("TX_ASOS", "AUS"),
    "miami":        ("FL_ASOS", "MIA"),
    "philadelphia": ("PA_ASOS", "PHL"),
}


def build_iem_daily_url(network: str, station: str, start: dt.date, end: dt.date) -> str:
    """
    Build IEM daily.py URL for date range.

    Args:
        network: ASOS network (e.g., "IL_ASOS")
        station: 3-letter station code (e.g., "MDW")
        start: Start date (inclusive)
        end: End date (inclusive)

    Returns:
        Full URL for IEM daily.py request

    Example:
        >>> url = build_iem_daily_url("IL_ASOS", "MDW", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
        >>> "MDW" in url and "max_temp_f" in url
        True
    """
    return (
        f"{IEM_DAILY_URL}?network={network}&stations={station}"
        f"&year1={start.year}&month1={start.month}&day1={start.day}"
        f"&year2={end.year}&month2={end.month}&day2={end.day}"
        f"&var=max_temp_f&na=blank&format=csv"
    )


def fetch_iem_daily(
    network: str,
    station: str,
    start: dt.date,
    end: dt.date,
    pause_s: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch ASOS Daily Summary (ADS) daily highs from IEM.

    Args:
        network: ASOS network (e.g., "IL_ASOS")
        station: 3-letter station code (e.g., "MDW")
        start: Start date (inclusive)
        end: End date (inclusive)
        pause_s: Pause between retries (seconds)

    Returns:
        DataFrame with columns: [station, day, max_temp_f]
        - day: date (datetime.date)
        - max_temp_f: integer °F (rounded from ADS decimal values)

    Raises:
        RuntimeError: If fetch fails after 3 attempts

    Notes:
        - IEM returns CSV with header: station,day,max_temp_f
        - ADS values are decimal (e.g., 53.0), we round to integer to match NWS climate products
        - Missing values are returned as NaN
    """
    url = build_iem_daily_url(network, station, start, end)

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            if not resp.text.strip():
                logger.warning(f"Empty response from IEM for {network}/{station}")
                time.sleep(pause_s * (attempt + 1))
                continue

            # Parse CSV
            df = pd.read_csv(io.StringIO(resp.text))

            # Normalize columns
            df["day"] = pd.to_datetime(df["day"]).dt.date

            # Convert to integer °F (NWS climate values are integers)
            # pd.Int64 allows NaN values
            df["max_temp_f"] = pd.to_numeric(df["max_temp_f"], errors="coerce").round(0).astype("Int64")

            logger.info(
                f"Fetched {len(df)} days from IEM ADS: {network}/{station} "
                f"({start} to {end})"
            )

            return df

        except Exception as e:
            logger.warning(f"IEM fetch attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(pause_s * (attempt + 1))

    raise RuntimeError(f"IEM daily fetch failed for {network}/{station} {start}..{end}")


def fetch_cities(
    cities: Iterable[str],
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """
    Fetch ADS daily highs for multiple cities.

    Args:
        cities: City names (must be in CITY_TO_ASOS mapping)
        start: Start date (inclusive)
        end: End date (inclusive)

    Returns:
        DataFrame with columns: [city, station, day, tmax_ads_f]
        - city: City name (e.g., "chicago")
        - station: 3-letter station code (e.g., "MDW")
        - day: date
        - tmax_ads_f: integer °F from ADS

    Example:
        >>> df = fetch_cities(["chicago"], dt.date(2024, 11, 12), dt.date(2024, 11, 13))
        >>> len(df) == 2  # Two days
        True
    """
    frames = []

    for city in cities:
        if city not in CITY_TO_ASOS:
            logger.error(f"Unknown city: {city}. Must be one of {list(CITY_TO_ASOS.keys())}")
            continue

        network, station = CITY_TO_ASOS[city]
        logger.info(f"Fetching {city} ({network}/{station})...")

        try:
            df = fetch_iem_daily(network, station, start, end)
            df.insert(0, "city", city)
            frames.append(df)
        except Exception as e:
            logger.error(f"Failed to fetch {city}: {e}")
            continue

    if not frames:
        logger.warning("No data fetched for any city")
        return pd.DataFrame(columns=["city", "station", "day", "tmax_ads_f"])

    # Combine all cities
    out = pd.concat(frames, ignore_index=True)

    # Rename max_temp_f → tmax_ads_f to indicate source
    out = out.rename(columns={"max_temp_f": "tmax_ads_f"})

    # Sort by city and date
    out = out.sort_values(["city", "day"]).reset_index(drop=True)

    logger.info(f"Total: {len(out)} daily records across {len(frames)} cities")

    return out


if __name__ == "__main__":
    import argparse

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Fetch IEM ASOS Daily Summary (ADS) data"
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=list(CITY_TO_ASOS.keys()),
        choices=list(CITY_TO_ASOS.keys()),
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

    print(f"\nFetching IEM ADS data:")
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
    print(df["tmax_ads_f"].isna().sum())
    print("\nSample data:")
    print(df.head(10))

    # Save if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to: {args.output}")

    print("="*60)
