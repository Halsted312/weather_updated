#!/usr/bin/env python3
"""
Iowa Environmental Mesonet (IEM) CF6 API client.

IEM provides parsed CF6 (WS Form F-6) climate data via JSON API - much cleaner
than scraping HTML from weather.gov. This is the PRIMARY source for historical
settlement backfills.

API Documentation: https://mesonet.agron.iastate.edu/json/cf6.py?help=1

Key advantages:
- Structured JSON (no HTML parsing)
- Historical data going back years (tested 2023-2025)
- Links to raw NWS CF6 text products for audit
- Fast bulk fetching (1 request per city per year)
- All 7 cities supported via ICAO codes
"""

import logging
import time
from datetime import date, datetime
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

# Import station mapping from nws_cli
from weather.nws_cli import SETTLEMENT_STATIONS


class IEMClient:
    """Client for Iowa Environmental Mesonet CF6 JSON API."""

    BASE_URL = "https://mesonet.agron.iastate.edu/json/cf6.py"
    RAW_TEXT_BASE = "https://mesonet.agron.iastate.edu"

    def __init__(self):
        """Initialize IEM client."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "KalshiWeatherBot/1.0 (Educational Research)"
        })

    def fetch_cf6_year(
        self, station_icao: str, year: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch CF6 data for entire year from IEM.

        Args:
            station_icao: ICAO station code (e.g., "KMDW" for Chicago Midway)
            year: Year (e.g., 2025)

        Returns:
            List of daily records with parsed CF6 data

        Example response structure:
        [
            {
                "station": "KMDW",
                "valid": "2025-11-07",
                "state": "IL",
                "wfo": "LOT",
                "link": "/api/1/nwstext/202511120800-KLOT-CXUS53-CF6MDW",
                "product": "202511120800-KLOT-CXUS53-CF6MDW",
                "name": "CHICAGO",
                "high": 64,
                "low": 47,
                "avg_temp": 56.0,
                "precip": 0.1,
                ...
            }
        ]
        """
        params = {
            "station": station_icao,
            "year": year,
            "fmt": "json"
        }

        logger.info(f"Fetching IEM CF6 for {station_icao} year {year}...")

        try:
            response = self.session.get(
                self.BASE_URL, params=params, timeout=30
            )
            response.raise_for_status()

            data = response.json()

            # IEM returns dict with "results" key containing array of records
            if isinstance(data, dict) and "results" in data:
                records = data["results"]
            else:
                # Fallback: assume direct array
                records = data if isinstance(data, list) else []

            logger.info(f"Fetched {len(records)} daily records for {station_icao} {year}")

            # Rate limiting: be nice to IEM servers
            time.sleep(0.2)

            return records

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching IEM CF6 for {station_icao} {year}: {e}")
            return []

    def fetch_raw_cf6_text(self, link_path: str) -> Optional[str]:
        """
        Fetch raw NWS CF6 text product from IEM for audit trail.

        Args:
            link_path: Path from IEM record['link']
                      (e.g., "/api/1/nwstext/202511120800-KLOT-CXUS53-CF6MDW")

        Returns:
            Raw CF6 text product or None if fetch fails
        """
        url = f"{self.RAW_TEXT_BASE}{link_path}"

        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return response.text

        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not fetch raw CF6 text from {url}: {e}")
            return None

    def get_daily_records(
        self, station_icao: str, start_date: date, end_date: date
    ) -> List[Dict[str, Any]]:
        """
        Get daily CF6 records for a date range.

        Fetches all years needed and filters to date range.

        Args:
            station_icao: ICAO station code
            start_date: Start date
            end_date: End date

        Returns:
            List of daily records within date range
        """
        years = set()
        current = start_date
        while current <= end_date:
            years.add(current.year)
            # Move to next year
            try:
                current = date(current.year + 1, 1, 1)
            except ValueError:
                break

        all_records = []
        for year in sorted(years):
            records = self.fetch_cf6_year(station_icao, year)

            # Filter to date range
            for record in records:
                record_date = datetime.strptime(record["valid"], "%Y-%m-%d").date()
                if start_date <= record_date <= end_date:
                    all_records.append(record)

        logger.info(
            f"Filtered to {len(all_records)} records between "
            f"{start_date} and {end_date}"
        )

        return all_records

    def get_settlements_for_city(
        self, city: str, start_date: date, end_date: date,
        fetch_raw_text: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get settlement data for a city from IEM CF6.

        Args:
            city: City name (chicago, new_york, miami, etc.)
            start_date: Start date
            end_date: End date
            fetch_raw_text: Whether to fetch raw CF6 text for audit (slower)

        Returns:
            List of settlement dicts ready for database insertion

        Example return:
        [
            {
                "city": "chicago",
                "icao": "KMDW",
                "issuedby": "MDW",
                "date_local": date(2025, 11, 7),
                "tmax_f": 64.0,
                "source": "IEM_CF6",
                "is_preliminary": True,
                "raw_payload": "<raw CF6 text>"
            }
        ]
        """
        if city not in SETTLEMENT_STATIONS:
            raise ValueError(
                f"Unknown city: {city}. Available: {list(SETTLEMENT_STATIONS.keys())}"
            )

        station = SETTLEMENT_STATIONS[city]
        icao = station["icao"]
        issuedby = station["issuedby"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Fetching IEM CF6 for {city.upper()} ({icao})")
        logger.info(f"{'='*60}\n")

        # Fetch records from IEM
        records = self.get_daily_records(icao, start_date, end_date)

        settlements = []
        for record in records:
            record_date = datetime.strptime(record["valid"], "%Y-%m-%d").date()
            tmax_f = record.get("high")

            if tmax_f is None:
                logger.warning(f"Missing TMAX for {city} on {record_date}, skipping")
                continue

            # Optionally fetch raw CF6 text for audit trail
            raw_payload = None
            if fetch_raw_text and "link" in record:
                raw_payload = self.fetch_raw_cf6_text(record["link"])

            # Fallback: store JSON if raw text unavailable
            if raw_payload is None:
                raw_payload = str(record)

            settlement = {
                "city": city,
                "icao": icao,
                "issuedby": issuedby,
                "date_local": record_date,
                "tmax_f": float(tmax_f),
                "source": "IEM_CF6",
                "is_preliminary": True,  # CF6 is preliminary (CLI is final)
                "raw_payload": raw_payload,
            }

            settlements.append(settlement)
            logger.debug(f"  {record_date}: {tmax_f}°F")

        logger.info(f"✓ Fetched {len(settlements)} settlements for {city}")

        return settlements

    def backfill_all_cities(
        self, start_date: date, end_date: date, cities: Optional[List[str]] = None,
        fetch_raw_text: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Backfill settlements for all cities.

        Args:
            start_date: Start date
            end_date: End date
            cities: List of city names (default: all cities)
            fetch_raw_text: Whether to fetch raw CF6 text (slower but more audit data)

        Returns:
            Dict mapping city name to list of settlement records
        """
        if cities is None:
            cities = list(SETTLEMENT_STATIONS.keys())

        results = {}

        for city in cities:
            try:
                settlements = self.get_settlements_for_city(
                    city, start_date, end_date, fetch_raw_text
                )
                results[city] = settlements

            except Exception as e:
                logger.error(f"Error fetching IEM CF6 for {city}: {e}", exc_info=True)
                results[city] = []

        return results


def main():
    """Demo: Fetch IEM CF6 for Chicago for November 2025."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    client = IEMClient()

    # Fetch November 2025 for Chicago
    settlements = client.get_settlements_for_city(
        "chicago",
        date(2025, 11, 1),
        date(2025, 11, 11),
        fetch_raw_text=False  # Skip raw text for demo (faster)
    )

    print(f"\n{'='*60}")
    print(f"IEM CF6 RESULTS - CHICAGO NOVEMBER 2025")
    print(f"{'='*60}\n")

    for s in settlements:
        print(f"{s['date_local']}: {s['tmax_f']}°F (source: {s['source']})")

    print(f"\nTotal records: {len(settlements)}")


if __name__ == "__main__":
    main()
