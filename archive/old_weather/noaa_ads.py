#!/usr/bin/env python3
"""
NOAA Access Data Service (ADS) client for fetching observed weather data.

Uses NCEI's Access Data Service (no token required) to fetch daily TMAX
for weather stations. Chicago markets settle to Chicago Midway (GHCND:USW00014819).

API Docs: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)


# Station IDs for each city (GHCND = Global Historical Climatology Network - Daily)
CITY_STATIONS = {
    "chicago": "GHCND:USW00014819",  # Chicago Midway
    "miami": "GHCND:USW00012839",  # Miami International
    "denver": "GHCND:USW00003017",  # Denver International
    "austin": "GHCND:USW00013958",  # Austin-Bergstrom International
    "los_angeles": "GHCND:USW00023174",  # Los Angeles International
    "philadelphia": "GHCND:USW00013739",  # Philadelphia International
}


class NOAAClient:
    """Client for NCEI Access Data Service."""

    def __init__(self, base_url: str = "https://www.ncei.noaa.gov/access/services/data/v1"):
        """
        Initialize NOAA client.

        Args:
            base_url: Base URL for NCEI Access Data Service
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get_daily_tmax(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        units: str = "standard",  # "standard" = Fahrenheit
    ) -> List[Dict[str, Any]]:
        """
        Fetch daily TMAX (high temperature) for a station and date range.

        Args:
            station_id: Station ID (e.g., "GHCND:USW00014819")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            units: "standard" (Fahrenheit) or "metric" (Celsius)

        Returns:
            List of dicts with keys: station, date, TMAX

        Example response:
            [
                {"station": "GHCND:USW00014819", "date": "2025-09-01", "TMAX": 78.0},
                {"station": "GHCND:USW00014819", "date": "2025-09-02", "TMAX": 82.0},
            ]
        """
        params = {
            "dataset": "daily-summaries",
            "stations": station_id.split(":")[-1] if ":" in station_id else station_id,
            "startDate": start_date,
            "endDate": end_date,
            "dataTypes": "TMAX",
            "units": units,
            "format": "json",
        }

        logger.info(f"Fetching TMAX for {station_id} from {start_date} to {end_date}...")

        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            for row in data:
                row["station"] = station_id
            logger.info(f"Fetched {len(data)} daily records")

            # Rate limiting: be nice to NOAA servers
            time.sleep(0.2)

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching NOAA data: {e}")
            raise

    def get_tmax_for_city(
        self,
        city_name: str,
        start_date: str,
        end_date: str,
        units: str = "standard",
    ) -> List[Dict[str, Any]]:
        """
        Fetch daily TMAX for a city by name.

        Args:
            city_name: City name (chicago, miami, denver, etc.)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            units: "standard" (Fahrenheit) or "metric" (Celsius)

        Returns:
            List of daily weather records with TMAX

        Raises:
            ValueError: If city_name is not recognized
        """
        station_id = CITY_STATIONS.get(city_name.lower())
        if not station_id:
            raise ValueError(
                f"Unknown city: {city_name}. Available: {list(CITY_STATIONS.keys())}"
            )

        return self.get_daily_tmax(station_id, start_date, end_date, units)

    def get_tmax_for_markets(
        self,
        city_name: str,
        market_dates: List[str],
        units: str = "standard",
    ) -> List[Dict[str, Any]]:
        """
        Fetch TMAX for specific market dates.

        Args:
            city_name: City name (chicago, miami, denver, etc.)
            market_dates: List of dates in YYYY-MM-DD format
            units: "standard" (Fahrenheit) or "metric" (Celsius)

        Returns:
            List of daily weather records for the specified dates
        """
        if not market_dates:
            return []

        # Sort dates and get min/max
        sorted_dates = sorted(market_dates)
        start_date = sorted_dates[0]
        end_date = sorted_dates[-1]

        # Fetch all data in range
        all_data = self.get_tmax_for_city(city_name, start_date, end_date, units)

        # Filter to only requested dates
        date_set = set(market_dates)
        filtered = [d for d in all_data if d.get("date") in date_set]

        logger.info(f"Filtered to {len(filtered)} dates out of {len(all_data)} total")

        return filtered

    def fetch_all_cities(
        self,
        start_date: str,
        end_date: str,
        cities: Optional[List[str]] = None,
        units: str = "standard",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch TMAX for all cities in a date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            cities: List of city names (default: all available cities)
            units: "standard" (Fahrenheit) or "metric" (Celsius)

        Returns:
            Dict mapping city name to list of weather records
        """
        if cities is None:
            cities = list(CITY_STATIONS.keys())

        results = {}

        for city in cities:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Fetching NOAA data for {city.upper()}")
                logger.info(f"{'='*60}\n")

                data = self.get_tmax_for_city(city, start_date, end_date, units)
                results[city] = data

            except Exception as e:
                logger.error(f"Error fetching {city}: {e}")
                results[city] = []

        return results


def main():
    """Demo: Fetch TMAX for Chicago last 7 days."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Fetch last 7 days for Chicago
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=7)

    client = NOAAClient()
    data = client.get_tmax_for_city(
        "chicago",
        start_date.isoformat(),
        end_date.isoformat(),
    )

    print("\n" + "=" * 60)
    print("Chicago TMAX (last 7 days)")
    print("=" * 60)
    for record in data:
        print(f"{record['date']}: {record.get('TMAX', 'N/A')}°F")


if __name__ == "__main__":
    main()
