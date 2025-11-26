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

Precedence: CLI (this) → CF6 (preliminary) → GHCND (audit)
"""

import logging
import re
from datetime import datetime, date
from typing import Optional, Dict, Any, List
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Station mapping for all 7 cities
SETTLEMENT_STATIONS = {
    "austin": {
        "icao": "KAUS",
        "issuedby": "AUS",
        "ghcnd": "GHCND:USW00013958",
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
        self, issuedby: str, date_local: Optional[date] = None
    ) -> Optional[str]:
        """
        Fetch CLI HTML from forecast.weather.gov.

        Args:
            issuedby: NWS issuing office code (e.g., "MDW" for Chicago Midway)
            date_local: Target date (default: latest available)

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
        self, html: str, target_date: Optional[date] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse TMAX from CLI HTML.

        CLI format (example):
        ```
        CLIMATE SUMMARY FOR CHICAGO MIDWAY AIRPORT IL
        ...
        MAXIMUM TEMPERATURE FOR THE DATE               76
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

            # Extract MAXIMUM temperature from table
            # Format: "  MAXIMUM         42   4:55 PM  ..."
            # Look for "MAXIMUM" followed by temp and time (e.g., "3:49 PM")
            tmax_match = re.search(
                r"MAXIMUM\s+(\d+)\s+\d+:\d+\s+[AP]M",
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

            logger.info(f"Parsed CLI for {cli_date}: TMAX = {tmax_f}°F")

            return {
                "date_local": cli_date,
                "tmax_f": tmax_f,
                "raw_html": html,
            }

        except Exception as e:
            logger.error(f"Error parsing CLI HTML: {e}", exc_info=True)
            return None

    def get_tmax_for_city(
        self, city: str, target_date: Optional[date] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get TMAX from CLI for a city.

        Args:
            city: City name (chicago, new_york, miami, etc.)
            target_date: Target date (default: latest available)

        Returns:
            Dict with keys: city, icao, issuedby, date_local, tmax_f, raw_html
            Returns None if not found
        """
        if city not in SETTLEMENT_STATIONS:
            raise ValueError(
                f"Unknown city: {city}. Available: {list(SETTLEMENT_STATIONS.keys())}"
            )

        station = SETTLEMENT_STATIONS[city]
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
        result["city"] = city
        result["icao"] = icao
        result["issuedby"] = issuedby

        return result

    def fetch_all_cities(
        self, target_date: Optional[date] = None
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Fetch CLI TMAX for all cities.

        Args:
            target_date: Target date (default: latest available)

        Returns:
            Dict mapping city name to result (or None if failed)
        """
        results = {}

        for city in SETTLEMENT_STATIONS.keys():
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Fetching CLI for {city.upper()}")
                logger.info(f"{'='*60}\n")

                result = self.get_tmax_for_city(city, target_date)
                results[city] = result

                if result:
                    logger.info(
                        f"✓ {city}: {result['tmax_f']}°F on {result['date_local']}"
                    )
                else:
                    logger.warning(f"✗ {city}: No CLI data available")

            except Exception as e:
                logger.error(f"Error fetching CLI for {city}: {e}", exc_info=True)
                results[city] = None

        return results


def main():
    """Demo: Fetch latest CLI for Chicago."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    client = NWSCliClient()

    # Fetch Chicago CLI
    result = client.get_tmax_for_city("chicago")

    if result:
        print(f"\n{'='*60}")
        print(f"NWS CLI - {result['city'].upper()}")
        print(f"{'='*60}")
        print(f"Date: {result['date_local']}")
        print(f"TMAX: {result['tmax_f']}°F")
        print(f"Station: {result['icao']} (issued by {result['issuedby']})")
        print(f"Raw HTML: {len(result['raw_html'])} bytes")
    else:
        print("Failed to fetch CLI for Chicago")


if __name__ == "__main__":
    main()
