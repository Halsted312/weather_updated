#!/usr/bin/env python3
"""
NWS Daily Climate Report (CLI) scraper.

The CLI (Daily Climate Report) is the official settlement source for Kalshi weather markets.
Posted by local NWS offices typically around 07:00 local time the next day.

Example URL:
https://forecast.weather.gov/product.php?issuedby=MDW&product=CLI&site=NWS

This scraper:
1. Fetches HTML from forecast.weather.gov
2. Parses the "CLIMATE SUMMARY FOR {DATE}" section
3. Extracts "MAXIMUM" temperature
4. Stores raw HTML payload for audit

Precedence: CLI (this) -> CF6 (preliminary) -> GHCND (audit)
"""

import logging
import re
from datetime import date, datetime
from typing import Any, Dict, Optional

import requests
from bs4 import BeautifulSoup

from src.config import CITIES, get_city

logger = logging.getLogger(__name__)


# Station mapping: city_id -> NWS issuedby code and GHCND station ID
SETTLEMENT_STATIONS = {
    "austin": {
        "icao": "KAUS",
        "issuedby": "AUS",
        "ghcnd": "GHCND:USW00013904",  # Austin Bergstrom (Kalshi official station)
    },
    "chicago": {
        "icao": "KMDW",
        "issuedby": "MDW",
        "ghcnd": "GHCND:USW00014819",
    },
    "los_angeles": {
        "icao": "KLAX",
        "issuedby": "LAX",
        "ghcnd": "GHCND:USW00023174",
    },
    "miami": {
        "icao": "KMIA",
        "issuedby": "MIA",
        "ghcnd": "GHCND:USW00012839",
    },
    "denver": {
        "icao": "KDEN",
        "issuedby": "DEN",
        "ghcnd": "GHCND:USW00003017",
    },
    "philadelphia": {
        "icao": "KPHL",
        "issuedby": "PHL",
        "ghcnd": "GHCND:USW00013739",
    },
}


class NWSCliClient:
    """Client for scraping NWS Daily Climate Reports (CLI)."""

    BASE_URL = "https://forecast.weather.gov/product.php"

    def __init__(self):
        """Initialize NWS CLI client."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "KalshiWeatherBot/1.0 (Educational Research)"
        })

    def fetch_cli_html(
        self,
        issuedby: str,
        target_date: Optional[date] = None,
    ) -> Optional[str]:
        """
        Fetch CLI HTML from forecast.weather.gov.

        Args:
            issuedby: NWS issuing office code (e.g., "MDW" for Chicago Midway)
            target_date: Target date (default: latest available)

        Returns:
            Raw HTML string, or None if not found
        """
        params = {
            "issuedby": issuedby,
            "product": "CLI",
            "site": "NWS",
        }

        logger.info(f"Fetching CLI for {issuedby}...")

        try:
            response = self.session.get(
                self.BASE_URL, params=params, timeout=15
            )
            response.raise_for_status()

            html = response.text

            # Verify we got a CLI product
            if "CLIMATE SUMMARY" not in html:
                logger.warning(f"No CLIMATE SUMMARY found in response for {issuedby}")
                return None

            logger.info(f"Fetched CLI HTML for {issuedby} ({len(html)} bytes)")
            return html

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching CLI for {issuedby}: {e}")
            return None

    def parse_cli_tmax(
        self,
        html: str,
        target_date: Optional[date] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Parse TMAX from CLI HTML.

        CLI format (example):
        ```
        CLIMATE SUMMARY FOR CHICAGO MIDWAY AIRPORT IL
        ...
        MAXIMUM TEMPERATURE FOR THE DATE               76
        ```
        OR in table format:
        ```
        MAXIMUM         42   4:55 PM  ...
        ```

        Args:
            html: Raw HTML from forecast.weather.gov
            target_date: Target date to validate against (optional)

        Returns:
            Dict with keys: date_local, tmax_f, raw_html
            Returns None if parsing fails
        """
        try:
            # Extract pre-formatted text block
            soup = BeautifulSoup(html, "html.parser")
            pre_tag = soup.find("pre", class_="glossaryProduct")

            if not pre_tag:
                logger.error("Could not find <pre> tag with climate data")
                return None

            text = pre_tag.get_text()

            # Extract date from "CLIMATE SUMMARY FOR NOVEMBER 11 2025"
            date_match = re.search(
                r"CLIMATE SUMMARY FOR (\w+ \d{1,2} \d{4})",
                text,
                re.IGNORECASE,
            )

            if not date_match:
                logger.error("Could not find date in CLI")
                return None

            date_str = date_match.group(1)
            cli_date = datetime.strptime(date_str, "%B %d %Y").date()

            # Try multiple patterns for MAXIMUM temperature
            # Pattern 1: Table format "MAXIMUM         42   4:55 PM"
            tmax_match = re.search(
                r"MAXIMUM\s+(\d+)\s+\d+:\d+\s+[AP]M",
                text,
                re.IGNORECASE,
            )

            # Pattern 2: Line format "MAXIMUM TEMPERATURE FOR THE DATE    76"
            if not tmax_match:
                tmax_match = re.search(
                    r"MAXIMUM TEMPERATURE\s+.*?\s+(\d+)",
                    text,
                    re.IGNORECASE,
                )

            # Pattern 3: Simple "MAXIMUM    76"
            if not tmax_match:
                tmax_match = re.search(
                    r"MAXIMUM\s+(\d+)",
                    text,
                    re.IGNORECASE,
                )

            if not tmax_match:
                logger.error("Could not find MAXIMUM TEMPERATURE in CLI")
                return None

            tmax_f = float(tmax_match.group(1))

            # Validate target date if provided
            if target_date and cli_date != target_date:
                logger.warning(
                    f"CLI date mismatch: expected {target_date}, got {cli_date} - rejecting"
                )
                return None

            logger.info(f"Parsed CLI for {cli_date}: TMAX = {tmax_f}F")

            return {
                "date_local": cli_date,
                "tmax_f": tmax_f,
                "is_preliminary": False,
                "source": "CLI",
                "raw_html": html,
            }

        except Exception as e:
            logger.error(f"Error parsing CLI HTML: {e}", exc_info=True)
            return None

    def get_tmax_for_city(
        self,
        city_id: str,
        target_date: Optional[date] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get TMAX from CLI for a city.

        Args:
            city_id: City ID (chicago, austin, miami, etc.)
            target_date: Target date (default: latest available)

        Returns:
            Dict with keys: city_id, icao, issuedby, date_local, tmax_f, raw_html
            Returns None if not found
        """
        if city_id not in SETTLEMENT_STATIONS:
            raise ValueError(
                f"Unknown city: {city_id}. Available: {list(SETTLEMENT_STATIONS.keys())}"
            )

        station = SETTLEMENT_STATIONS[city_id]
        issuedby = station["issuedby"]
        icao = station["icao"]

        # Fetch CLI HTML
        html = self.fetch_cli_html(issuedby, target_date)
        if not html:
            return None

        # Parse TMAX
        result = self.parse_cli_tmax(html, target_date)
        if not result:
            return None

        # Add city and station metadata
        result["city_id"] = city_id
        result["icao"] = icao
        result["issuedby"] = issuedby
        result["ghcnd"] = station["ghcnd"]

        return result

    def fetch_all_cities(
        self,
        target_date: Optional[date] = None,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Fetch CLI TMAX for all cities.

        Args:
            target_date: Target date (default: latest available)

        Returns:
            Dict mapping city ID to result (or None if failed)
        """
        results = {}

        for city_id in SETTLEMENT_STATIONS.keys():
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Fetching CLI for {city_id.upper()}")
                logger.info(f"{'='*60}\n")

                result = self.get_tmax_for_city(city_id, target_date)
                results[city_id] = result

                if result:
                    logger.info(
                        f"OK {city_id}: {result['tmax_f']}F on {result['date_local']}"
                    )
                else:
                    logger.warning(f"FAIL {city_id}: No CLI data available")

            except Exception as e:
                logger.error(f"Error fetching CLI for {city_id}: {e}", exc_info=True)
                results[city_id] = None

        return results


def get_settlement_station(city_id: str) -> Optional[Dict[str, str]]:
    """Get settlement station info for a city."""
    return SETTLEMENT_STATIONS.get(city_id)
