#!/usr/bin/env python3
"""
NOAA NCEI API clients for historical temperature data.

Provides historical TMAX (maximum temperature) data from the GHCN-Daily dataset,
which is the same source that underlies NWS CLI reports.

Two API options:
1. Access Data Service v1 (no token required) - DEFAULT
   URL: https://www.ncei.noaa.gov/access/services/data/v1
   Docs: https://www.ncei.noaa.gov/access/services/data/v1

2. CDO Web Services v2 (requires token) - LEGACY
   URL: https://www.ncdc.noaa.gov/cdo-web/api/v2
   Docs: https://www.ncdc.noaa.gov/cdo-web/webservices/v2

This is used for:
1. Backfilling historical settlement data (pre-CLI availability)
2. Cross-validation of CLI values
3. Audit trail for settlement disputes
"""

import logging
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.weather.nws_cli import SETTLEMENT_STATIONS

logger = logging.getLogger(__name__)


# GHCND station IDs (without the "GHCND:" prefix for v1 API)
NCEI_STATIONS = {
    "austin": "USW00013904",  # Austin Bergstrom (Kalshi official station, NOT Camp Mabry USW00013958)
    "chicago": "USW00014819",
    "los_angeles": "USW00023174",
    "miami": "USW00012839",
    "denver": "USW00003017",
    "philadelphia": "USW00013739",
}


class NCEIAccessClient:
    """Client for NCEI Access Data Service v1 (no token required).

    This is the preferred client for historical TMAX data - no API key needed.

    API Documentation:
    https://www.ncei.noaa.gov/access/services/data/v1
    """

    BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"

    # Be polite - 1 request per second
    REQUEST_DELAY = 1.0

    def __init__(self, session: Optional[requests.Session] = None):
        """Initialize NCEI Access client (no API key required)."""
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "KalshiWeatherBot/1.0 (Educational Research)"
        })
        self._last_request = 0.0
        logger.info("NCEI Access client initialized (no token required)")

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
    def fetch_tmax_range(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Fetch TMAX data for a date range using Access Data Service v1.

        Args:
            station_id: GHCND station ID (e.g., "USW00014819" for Chicago)
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of records with date and tmax_f
        """
        self._rate_limit()

        params = {
            "dataset": "daily-summaries",
            "stations": station_id,
            "dataTypes": "TMAX",
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "units": "standard",  # Fahrenheit
            "includeStationName": "1",
            "format": "json",
        }

        logger.info(f"Fetching NCEI TMAX for {station_id} from {start_date} to {end_date}")

        response = self.session.get(self.BASE_URL, params=params, timeout=60)

        if response.status_code == 429:
            logger.warning("Rate limited by NCEI, waiting 60 seconds...")
            time.sleep(60)
            raise requests.exceptions.RequestException("Rate limited")

        response.raise_for_status()

        # Response is a JSON array of records
        data = response.json()

        results = []
        for row in data:
            # Parse date - format is "YYYY-MM-DD"
            date_str = row.get("DATE")
            if not date_str:
                continue

            record_date = date.fromisoformat(date_str)

            # TMAX value - already in Fahrenheit with units=standard
            tmax_raw = row.get("TMAX")
            if tmax_raw is None or tmax_raw == "":
                continue

            try:
                tmax_f = int(round(float(tmax_raw)))
            except (ValueError, TypeError):
                continue

            results.append({
                "date_local": record_date,
                "tmax_f": tmax_f,
                "station_id": station_id,
                "station_name": row.get("NAME", ""),
                "source": "NCEI_ACCESS",
                "is_preliminary": False,
                "raw_json": row,
            })

        logger.info(f"Fetched {len(results)} NCEI records for {station_id}")
        return results

    def get_tmax_for_city(
        self,
        city_id: str,
        target_date: date,
    ) -> Optional[Dict[str, Any]]:
        """
        Get TMAX for a specific city and date.

        Args:
            city_id: City ID (chicago, austin, etc.)
            target_date: Target date

        Returns:
            Dict with tmax_f and metadata, or None if not available
        """
        if city_id not in NCEI_STATIONS:
            raise ValueError(f"Unknown city: {city_id}")

        station_id = NCEI_STATIONS[city_id]
        results = self.fetch_tmax_range(station_id, target_date, target_date)

        if not results:
            return None

        result = results[0]
        result["city_id"] = city_id
        return result

    def fetch_city_history(
        self,
        city_id: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical TMAX for a city.

        Args:
            city_id: City ID
            start_date: Start date
            end_date: End date

        Returns:
            List of TMAX records
        """
        if city_id not in NCEI_STATIONS:
            raise ValueError(f"Unknown city: {city_id}")

        station_id = NCEI_STATIONS[city_id]

        logger.info(f"Fetching NCEI history for {city_id} ({station_id})")

        # Fetch in 1-year chunks to avoid timeouts
        results = []
        current_start = start_date

        while current_start <= end_date:
            chunk_end = min(current_start + timedelta(days=364), end_date)

            try:
                chunk_results = self.fetch_tmax_range(station_id, current_start, chunk_end)
                for result in chunk_results:
                    result["city_id"] = city_id
                results.extend(chunk_results)
            except Exception as e:
                logger.error(f"Error fetching NCEI chunk {current_start} to {chunk_end}: {e}")

            current_start = chunk_end + timedelta(days=1)

        logger.info(f"Fetched {len(results)} NCEI TMAX records for {city_id}")
        return results

    def fetch_all_cities_history(
        self,
        start_date: date,
        end_date: date,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch historical TMAX for all cities.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Dict mapping city_id to list of records
        """
        results = {}

        for city_id in NCEI_STATIONS:
            try:
                city_results = self.fetch_city_history(city_id, start_date, end_date)
                results[city_id] = city_results
            except Exception as e:
                logger.error(f"Error fetching {city_id}: {e}")
                results[city_id] = []

        return results


class NOAANCEIClient:
    """Client for NOAA NCEI Climate Data Online (CDO) API."""

    BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
    DATASET = "GHCND"  # Global Historical Climatology Network - Daily
    DATATYPE = "TMAX"  # Maximum temperature (tenths of degree Celsius)

    # Rate limit: 5 req/sec
    REQUEST_DELAY = 0.25  # 4 requests per second to be safe

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NOAA NCEI client.

        Args:
            api_key: NOAA CDO API key (get free at https://www.ncdc.noaa.gov/cdo-web/token)
        """
        settings = get_settings()
        self.api_key = api_key or settings.noaa_api_key

        if not self.api_key:
            raise ValueError(
                "NOAA API key required. Get one at https://www.ncdc.noaa.gov/cdo-web/token "
                "and set NOAA_API_KEY environment variable."
            )

        self.session = requests.Session()
        self.session.headers.update({
            "token": self.api_key,
            "User-Agent": "KalshiWeatherBot/1.0 (Educational Research)"
        })

        self._last_request = 0.0
        logger.info("NOAA NCEI client initialized")

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
    def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make API request with retry logic.

        Args:
            endpoint: API endpoint (e.g., "/data")
            params: Query parameters

        Returns:
            JSON response dict
        """
        self._rate_limit()

        url = f"{self.BASE_URL}{endpoint}"
        response = self.session.get(url, params=params, timeout=30)

        if response.status_code == 429:
            logger.warning("Rate limited by NOAA, waiting 60 seconds...")
            time.sleep(60)
            raise requests.exceptions.RequestException("Rate limited")

        response.raise_for_status()
        return response.json()

    def fetch_tmax_range(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Fetch TMAX data for a date range.

        Args:
            station_id: GHCND station ID (e.g., "GHCND:USW00014819")
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of records with date and tmax_f
        """
        results = []

        # NOAA API returns max 1000 results per request
        # Split into 1-year chunks to be safe
        current_start = start_date

        while current_start <= end_date:
            chunk_end = min(current_start + timedelta(days=364), end_date)

            params = {
                "datasetid": self.DATASET,
                "datatypeid": self.DATATYPE,
                "stationid": station_id,
                "startdate": current_start.isoformat(),
                "enddate": chunk_end.isoformat(),
                "units": "standard",  # Fahrenheit
                "limit": 1000,
            }

            logger.info(f"Fetching TMAX for {station_id} from {current_start} to {chunk_end}")

            try:
                response = self._request("/data", params)

                if "results" not in response:
                    logger.warning(f"No results for {station_id} {current_start} to {chunk_end}")
                    current_start = chunk_end + timedelta(days=1)
                    continue

                for record in response["results"]:
                    # Parse date from "2024-01-15T00:00:00"
                    record_date = date.fromisoformat(record["date"][:10])
                    tmax_f = record["value"]  # Already in Fahrenheit with units=standard

                    results.append({
                        "date_local": record_date,
                        "tmax_f": int(round(tmax_f)),
                        "station_id": station_id,
                        "source": "NCEI",
                        "is_preliminary": False,
                        "raw_json": record,
                    })

                logger.info(f"  Fetched {len(response['results'])} records")

            except Exception as e:
                logger.error(f"Error fetching TMAX for {station_id}: {e}")
                raise

            current_start = chunk_end + timedelta(days=1)

        return results

    def get_tmax_for_city(
        self,
        city_id: str,
        target_date: date,
    ) -> Optional[Dict[str, Any]]:
        """
        Get TMAX for a specific city and date.

        Args:
            city_id: City ID (chicago, austin, etc.)
            target_date: Target date

        Returns:
            Dict with tmax_f and metadata, or None if not available
        """
        if city_id not in SETTLEMENT_STATIONS:
            raise ValueError(f"Unknown city: {city_id}")

        station = SETTLEMENT_STATIONS[city_id]
        station_id = station["ghcnd"]

        results = self.fetch_tmax_range(station_id, target_date, target_date)

        if not results:
            return None

        result = results[0]
        result["city_id"] = city_id
        result["icao"] = station["icao"]
        result["ghcnd"] = station_id

        return result

    def fetch_city_history(
        self,
        city_id: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical TMAX for a city.

        Args:
            city_id: City ID
            start_date: Start date
            end_date: End date

        Returns:
            List of TMAX records
        """
        if city_id not in SETTLEMENT_STATIONS:
            raise ValueError(f"Unknown city: {city_id}")

        station = SETTLEMENT_STATIONS[city_id]
        station_id = station["ghcnd"]

        logger.info(f"Fetching NCEI history for {city_id} ({station_id})")

        results = self.fetch_tmax_range(station_id, start_date, end_date)

        # Add city metadata
        for result in results:
            result["city_id"] = city_id
            result["icao"] = station["icao"]

        logger.info(f"Fetched {len(results)} TMAX records for {city_id}")

        return results

    def fetch_all_cities_history(
        self,
        start_date: date,
        end_date: date,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch historical TMAX for all cities.

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
                logger.error(f"Error fetching {city_id}: {e}")
                results[city_id] = []

        return results
