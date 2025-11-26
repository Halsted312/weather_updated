#!/usr/bin/env python3
"""
NWS CF6 (WS Form F-6) scraper.

CF6 is the NWS preliminary monthly climatological data table. It's available
sooner than CLI and provides preliminary TMAX values for the month.

Example URL:
https://forecast.weather.gov/product.php?issuedby=MDW&product=CF6&site=NWS

This scraper:
1. Fetches HTML from forecast.weather.gov
2. Parses the monthly table
3. Extracts MAX column for target day
4. Marks as preliminary (is_preliminary=true)

Precedence: CLI (final) → CF6 (this, preliminary) → GHCND (audit)
"""

import logging
import re
from datetime import datetime, date
from typing import Optional, Dict, Any, List
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Import station mapping from nws_cli
from weather.nws_cli import SETTLEMENT_STATIONS


class NWSCF6Client:
    """Client for scraping NWS CF6 preliminary climate data."""

    BASE_URL = "https://forecast.weather.gov/product.php"

    def __init__(self):
        """Initialize NWS CF6 client."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "KalshiWeatherBot/1.0 (Educational Research)"
        })

    def fetch_cf6_html(
        self, issuedby: str
    ) -> Optional[str]:
        """
        Fetch CF6 HTML from forecast.weather.gov.

        Args:
            issuedby: NWS issuing office code (e.g., "MDW" for Chicago Midway)

        Returns:
            Raw HTML string, or None if not found
        """
        params = {
            "issuedby": issuedby,
            "product": "CF6",
            "site": "NWS",
        }

        logger.info(f"Fetching CF6 for {issuedby}...")

        try:
            response = self.session.get(
                self.BASE_URL, params=params, timeout=15
            )
            response.raise_for_status()

            html = response.text

            # Verify we got a CF6 product
            if "PRELIMINARY" not in html or "MAX" not in html:
                logger.warning(f"No CF6 table found in response for {issuedby}")
                return None

            logger.info(f"Fetched CF6 HTML for {issuedby} ({len(html)} bytes)")
            return html

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching CF6 for {issuedby}: {e}")
            return None

    def parse_cf6_tmax(
        self, html: str, target_date: date
    ) -> Optional[Dict[str, Any]]:
        """
        Parse TMAX from CF6 HTML for a specific date.

        CF6 format (example):
        ```
        PRELIMINARY LOCAL CLIMATOLOGICAL DATA (WS FORM: F-6)

                                   STATION: CHICAGO MIDWAY AIRPORT IL
                                   MONTH:   NOVEMBER
                                   YEAR:    2025

        DAY  MAX  MIN  AVG ...
         01   52   41   47 ...
         02   55   43   49 ...
         11   42   25   34 ...
        ```

        Args:
            html: Raw HTML from forecast.weather.gov
            target_date: Target date to extract TMAX for

        Returns:
            Dict with keys: date_local, tmax_f, is_preliminary, raw_html
            Returns None if parsing fails
        """
        try:
            # Extract pre-formatted text block
            soup = BeautifulSoup(html, "html.parser")
            pre_tag = soup.find("pre", class_="glossaryProduct")

            if not pre_tag:
                logger.error("Could not find <pre> tag with CF6 data")
                return None

            text = pre_tag.get_text()

            # Extract month and year from header
            month_match = re.search(
                r"MONTH:\s+(\w+)",
                text,
                re.IGNORECASE,
            )
            year_match = re.search(
                r"YEAR:\s+(\d{4})",
                text,
                re.IGNORECASE,
            )

            if not month_match or not year_match:
                logger.error("Could not find MONTH/YEAR in CF6")
                return None

            month_str = month_match.group(1)
            year_str = year_match.group(1)

            # Validate this CF6 matches the target date
            cf6_month = datetime.strptime(month_str, "%B").month
            cf6_year = int(year_str)

            if cf6_month != target_date.month or cf6_year != target_date.year:
                logger.warning(
                    f"CF6 month/year mismatch: CF6 is {month_str} {year_str}, "
                    f"target is {target_date.strftime('%B %Y')}"
                )
                return None

            # Find the table and extract the row for target day
            # Format: " 1  53  43  48 ..." (DY MAX MIN AVG ...)
            day_num = target_date.day

            # First, find the data section (after the "DY MAX MIN" header)
            header_match = re.search(r"DY MAX MIN AVG", text)
            if not header_match:
                logger.error("Could not find CF6 table header")
                return None

            # Search only in text after the header
            text_after_header = text[header_match.end():]

            # Look for data row for specific day
            # Pattern: " 1  53  43  48..." with at least 2 spaces after day number
            day_pattern = rf"^\s*{day_num}\s\s+(\d+)\s+(\d+)\s+(\d+)"
            day_match = re.search(day_pattern, text_after_header, re.MULTILINE)

            if not day_match:
                logger.error(f"Could not find day {day_num} in CF6 table")
                return None

            tmax_f = float(day_match.group(1))

            logger.info(
                f"Parsed CF6 for {target_date}: TMAX = {tmax_f}°F (preliminary)"
            )

            return {
                "date_local": target_date,
                "tmax_f": tmax_f,
                "is_preliminary": True,
                "raw_html": html,
            }

        except Exception as e:
            logger.error(f"Error parsing CF6 HTML: {e}", exc_info=True)
            return None

    def get_tmax_for_city(
        self, city: str, target_date: date
    ) -> Optional[Dict[str, Any]]:
        """
        Get preliminary TMAX from CF6 for a city.

        Args:
            city: City name (chicago, new_york, miami, etc.)
            target_date: Target date

        Returns:
            Dict with keys: city, icao, issuedby, date_local, tmax_f,
                           is_preliminary, raw_html
            Returns None if not found
        """
        if city not in SETTLEMENT_STATIONS:
            raise ValueError(
                f"Unknown city: {city}. Available: {list(SETTLEMENT_STATIONS.keys())}"
            )

        station = SETTLEMENT_STATIONS[city]
        issuedby = station["issuedby"]
        icao = station["icao"]

        # Fetch CF6 HTML
        html = self.fetch_cf6_html(issuedby)
        if not html:
            return None

        # Parse TMAX for target date
        result = self.parse_cf6_tmax(html, target_date)
        if not result:
            return None

        # Add city and station metadata
        result["city"] = city
        result["icao"] = icao
        result["issuedby"] = issuedby

        return result

    def fetch_all_cities(
        self, target_date: date
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Fetch CF6 preliminary TMAX for all cities.

        Args:
            target_date: Target date

        Returns:
            Dict mapping city name to result (or None if failed)
        """
        results = {}

        for city in SETTLEMENT_STATIONS.keys():
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Fetching CF6 for {city.upper()}")
                logger.info(f"{'='*60}\n")

                result = self.get_tmax_for_city(city, target_date)
                results[city] = result

                if result:
                    logger.info(
                        f"✓ {city}: {result['tmax_f']}°F on {result['date_local']} (preliminary)"
                    )
                else:
                    logger.warning(f"✗ {city}: No CF6 data available")

            except Exception as e:
                logger.error(f"Error fetching CF6 for {city}: {e}", exc_info=True)
                results[city] = None

        return results


def main():
    """Demo: Fetch CF6 for Chicago for yesterday."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from datetime import timedelta

    client = NWSCF6Client()

    # Fetch CF6 for yesterday (most recent complete day)
    yesterday = date.today() - timedelta(days=1)
    result = client.get_tmax_for_city("chicago", yesterday)

    if result:
        print(f"\n{'='*60}")
        print(f"NWS CF6 (Preliminary) - {result['city'].upper()}")
        print(f"{'='*60}")
        print(f"Date: {result['date_local']}")
        print(f"TMAX: {result['tmax_f']}°F (PRELIMINARY)")
        print(f"Station: {result['icao']} (issued by {result['issuedby']})")
        print(f"Raw HTML: {len(result['raw_html'])} bytes")
    else:
        print(f"Failed to fetch CF6 for Chicago on {yesterday}")


if __name__ == "__main__":
    main()
