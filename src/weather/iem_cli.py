#!/usr/bin/env python3
"""
IEM (Iowa Environmental Mesonet) CLI JSON API client.

Fetches parsed daily climate (CLI) data from IEM's JSON API.
This provides historical CLI data that the direct NWS scraper cannot.

API Documentation:
https://mesonet.agron.iastate.edu/json/cli.py

URL pattern: https://mesonet.agron.iastate.edu/json/cli.py?station={STATION}&year={YYYY}

One request per station+year returns the full year of daily CLI data.
This is the primary source for historical settlement TMAX data.
"""

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.weather.nws_cli import SETTLEMENT_STATIONS

logger = logging.getLogger(__name__)


IEM_CLI_BASE = "https://mesonet.agron.iastate.edu/json/cli.py"


@dataclass
class DailyCLI:
    """Parsed daily CLI record from IEM."""

    station: str
    date_local: date
    tmax_f: Optional[int]
    tmin_f: Optional[int]
    precip_in: Optional[float]
    snow_in: Optional[float]
    raw: Dict[str, Any]


class IEMCliClient:
    """Client for IEM CLI JSON API - historical daily climate data."""

    # Be polite - 1 request per second
    REQUEST_DELAY = 1.0

    def __init__(self, session: Optional[requests.Session] = None):
        """Initialize IEM CLI client."""
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "KalshiWeatherBot/1.0 (Educational Research)"
        })
        self._last_request = 0.0
        logger.info("IEM CLI client initialized")

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def fetch_cli_year(
        self,
        station: str,
        year: int,
    ) -> List[DailyCLI]:
        """
        Fetch daily CLI data for a station+year from IEM JSON API.

        Args:
            station: 4-letter ICAO station ID (e.g., "KMDW", "KDEN")
            year: Year to fetch (e.g., 2024)

        Returns:
            List of DailyCLI records for the year
        """
        self._rate_limit()

        params = {
            "station": station,
            "year": year,
        }

        logger.info(f"Fetching IEM CLI for {station} year {year}...")

        response = self.session.get(IEM_CLI_BASE, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        # Log payload structure for debugging
        logger.debug(f"IEM CLI payload keys: {list(payload.keys())}")

        # IEM JSON structure: {"station": "KMDW", "year": 2018, "results": [...]}
        data_rows = payload.get("results", [])

        if not data_rows:
            logger.warning(f"No CLI data for {station} year {year}")
            return []

        # Parse each row
        results: List[DailyCLI] = []
        for row in data_rows:
            try:
                # Parse date - format varies, try common patterns
                date_str = row.get("valid") or row.get("date")
                if not date_str:
                    continue

                # Handle date format: "2018-01-01" or "2018-01-01T00:00:00"
                if "T" in date_str:
                    date_str = date_str.split("T")[0]

                record_date = date.fromisoformat(date_str)

                # Extract temperature values - IEM uses various field names
                tmax = None
                tmin = None
                precip = None
                snow = None

                # Extract TMAX ("high" field)
                if "high" in row and row["high"] not in (None, "", "M", "m"):
                    try:
                        tmax = int(round(float(row["high"])))
                    except (ValueError, TypeError):
                        pass

                # Extract TMIN ("low" field)
                if "low" in row and row["low"] not in (None, "", "M", "m"):
                    try:
                        tmin = int(round(float(row["low"])))
                    except (ValueError, TypeError):
                        pass

                # Extract precipitation ("precip" field)
                if "precip" in row and row["precip"] not in (None, "", "M", "m"):
                    try:
                        val = row["precip"]
                        # Handle trace precipitation ("T")
                        if isinstance(val, str) and val.upper() == "T":
                            precip = 0.001  # Trace amount
                        else:
                            precip = float(val)
                    except (ValueError, TypeError):
                        pass

                # Extract snow ("snow" field)
                if "snow" in row and row["snow"] not in (None, "", "M", "m"):
                    try:
                        val = row["snow"]
                        if isinstance(val, str) and val.upper() == "T":
                            snow = 0.001  # Trace amount
                        else:
                            snow = float(val)
                    except (ValueError, TypeError):
                        pass

                results.append(DailyCLI(
                    station=station,
                    date_local=record_date,
                    tmax_f=tmax,
                    tmin_f=tmin,
                    precip_in=precip,
                    snow_in=snow,
                    raw=row,
                ))

            except Exception as e:
                logger.warning(f"Error parsing CLI row for {station}: {e}")
                continue

        logger.info(f"Fetched {len(results)} CLI records for {station} year {year}")
        return results

    def fetch_cli_range(
        self,
        station: str,
        start_date: date,
        end_date: date,
    ) -> List[DailyCLI]:
        """
        Fetch CLI data for a date range by fetching each year.

        Args:
            station: 4-letter ICAO station ID
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of DailyCLI records in date range
        """
        results: List[DailyCLI] = []

        # Fetch each year in range
        for year in range(start_date.year, end_date.year + 1):
            year_data = self.fetch_cli_year(station, year)

            # Filter to requested date range
            for record in year_data:
                if start_date <= record.date_local <= end_date:
                    results.append(record)

        results.sort(key=lambda x: x.date_local)
        logger.info(f"Fetched {len(results)} CLI records for {station} from {start_date} to {end_date}")
        return results

    def get_tmax_for_city(
        self,
        city_id: str,
        target_date: date,
    ) -> Optional[Dict[str, Any]]:
        """
        Get TMAX for a specific city and date from IEM CLI.

        Args:
            city_id: City ID (chicago, austin, etc.)
            target_date: Target date

        Returns:
            Dict with tmax_f and metadata, or None if not available
        """
        if city_id not in SETTLEMENT_STATIONS:
            raise ValueError(f"Unknown city: {city_id}")

        station = SETTLEMENT_STATIONS[city_id]
        icao = station["icao"]

        # Fetch the year containing target_date
        year_data = self.fetch_cli_year(icao, target_date.year)

        # Find the specific date
        for record in year_data:
            if record.date_local == target_date:
                return {
                    "city_id": city_id,
                    "icao": icao,
                    "date_local": record.date_local,
                    "tmax_f": record.tmax_f,
                    "tmin_f": record.tmin_f,
                    "precip_in": record.precip_in,
                    "snow_in": record.snow_in,
                    "source": "IEM_CLI",
                    "is_preliminary": False,
                    "raw_json": record.raw,
                }

        logger.warning(f"No IEM CLI data for {city_id} on {target_date}")
        return None

    def fetch_city_history(
        self,
        city_id: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical CLI TMAX for a city.

        Args:
            city_id: City ID
            start_date: Start date
            end_date: End date

        Returns:
            List of records with tmax_f and metadata
        """
        if city_id not in SETTLEMENT_STATIONS:
            raise ValueError(f"Unknown city: {city_id}")

        station = SETTLEMENT_STATIONS[city_id]
        icao = station["icao"]

        logger.info(f"Fetching IEM CLI history for {city_id} ({icao}) from {start_date} to {end_date}")

        records = self.fetch_cli_range(icao, start_date, end_date)

        results = []
        for record in records:
            results.append({
                "city_id": city_id,
                "icao": icao,
                "date_local": record.date_local,
                "tmax_f": record.tmax_f,
                "tmin_f": record.tmin_f,
                "precip_in": record.precip_in,
                "snow_in": record.snow_in,
                "source": "IEM_CLI",
                "is_preliminary": False,
                "raw_json": record.raw,
            })

        logger.info(f"Fetched {len(results)} IEM CLI records for {city_id}")
        return results

    def fetch_all_cities_history(
        self,
        start_date: date,
        end_date: date,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch historical CLI TMAX for all cities.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Dict mapping city_id to list of records
        """
        results = {}

        for city_id in SETTLEMENT_STATIONS:
            try:
                city_results = self.fetch_city_history(city_id, start_date, end_date)
                results[city_id] = city_results
            except Exception as e:
                logger.error(f"Error fetching IEM CLI for {city_id}: {e}")
                results[city_id] = []

        return results
