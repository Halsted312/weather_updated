#!/usr/bin/env python3
"""
Iowa Environmental Mesonet (IEM) CLI API client.

Provides historical Daily Climate Report (CLI) data in structured JSON.
Useful for both backfills and intraday polling once the previous day's
report is posted by the local NWS office (usually 06-08 local time).
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from typing import List, Dict, Any, Optional

import requests

from weather.nws_cli import SETTLEMENT_STATIONS

logger = logging.getLogger(__name__)


class IEMCliClient:
    """Client for the IEM CLI JSON endpoint."""

    BASE_URL = "https://mesonet.agron.iastate.edu/json/cli.py"
    RAW_TEXT_BASE = "https://mesonet.agron.iastate.edu"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "KalshiWeatherBot/1.0 (CLI Poller)"}
        )

    def fetch_cli_year(self, station_icao: str, year: int) -> List[Dict[str, Any]]:
        """Fetch CLI records for an entire calendar year."""
        params = {
            "station": station_icao,
            "year": year,
            "fmt": "json",
        }

        logger.info(f"Fetching IEM CLI for {station_icao} year {year}...")
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            records = data.get("results", []) if isinstance(data, dict) else data
            logger.info(f"Fetched {len(records)} CLI rows for {station_icao} {year}")
            time.sleep(0.2)
            return records or []
        except requests.RequestException as exc:
            logger.error(f"Error fetching CLI for {station_icao} {year}: {exc}")
            return []

    def fetch_raw_cli_text(self, link_path: str) -> Optional[str]:
        """Fetch the raw CLI AFOS product referenced by an IEM record."""
        if not link_path:
            return None

        url = f"{self.RAW_TEXT_BASE}{link_path}"
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            logger.warning(f"Unable to fetch CLI text {url}: {exc}")
            return None

    def get_daily_records(
        self,
        station_icao: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """Fetch CLI rows between two dates (inclusive)."""
        years = set()
        cur = start_date
        while cur <= end_date:
            years.add(cur.year)
            try:
                cur = date(cur.year + 1, 1, 1)
            except ValueError:
                break

        all_rows: List[Dict[str, Any]] = []
        for year in sorted(years):
            for record in self.fetch_cli_year(station_icao, year):
                rec_date = datetime.strptime(record["valid"], "%Y-%m-%d").date()
                if start_date <= rec_date <= end_date:
                    all_rows.append(record)

        logger.info(
            f"Filtered to {len(all_rows)} CLI rows for {station_icao} "
            f"between {start_date} and {end_date}"
        )
        return all_rows

    def get_settlements_for_city(
        self,
        city: str,
        start_date: date,
        end_date: date,
        fetch_raw_text: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return settlement dicts ready for upsert_settlement()."""
        if city not in SETTLEMENT_STATIONS:
            raise ValueError(f"Unknown city {city}")

        station = SETTLEMENT_STATIONS[city]
        icao = station["icao"]
        issuedby = station["issuedby"]

        rows = self.get_daily_records(icao, start_date, end_date)
        settlements: List[Dict[str, Any]] = []

        for row in rows:
            tmax = row.get("high")
            if tmax in (None, "M", "T"):
                continue

            try:
                tmax_f = float(tmax)
            except (TypeError, ValueError):
                continue

            rec_date = datetime.strptime(row["valid"], "%Y-%m-%d").date()
            payload = (
                self.fetch_raw_cli_text(row.get("link"))
                if fetch_raw_text
                else row
            )

            settlements.append(
                {
                    "city": city,
                    "icao": icao,
                    "issuedby": issuedby,
                    "date_local": rec_date,
                    "tmax_f": tmax_f,
                    "source": "CLI",
                    "is_preliminary": False,
                    "raw_payload": payload,
                }
            )

        return settlements


__all__ = ["IEMCliClient", "SETTLEMENT_STATIONS"]
