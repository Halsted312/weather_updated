#!/usr/bin/env python3
"""
Visual Crossing Weather API client for minute-level observations.

Uses Timeline API with specific parameters:
- include=minutes for sub-hourly data
- options=minuteinterval_5,nonulls,useobs for 5-min cadence with observations
- timezone=Z for UTC timestamps
- maxStations=1 & maxDistance=1609 to lock to specific ICAO station
- stn:<ICAO> location format to target exact station

API Docs: https://www.visualcrossing.com/resources/documentation/weather-api/timeline-weather-api/
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests
import pandas as pd

logger = logging.getLogger(__name__)


# Station mapping: city name -> (loc_id, vc_key)
STATION_MAP = {
    "austin": ("KAUS", "stn:KAUS"),
    "chicago": ("KMDW", "stn:KMDW"),
    "la": ("KLAX", "stn:KLAX"),
    "los_angeles": ("KLAX", "stn:KLAX"),  # Alias for legacy scripts
    "miami": ("KMIA", "stn:KMIA"),
    "denver": ("KDEN", "stn:KDEN"),
    "philadelphia": ("KPHL", "stn:KPHL"),
}


class VisualCrossingClient:
    """Client for Visual Crossing Timeline API with minute-level data."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline",
        minute_interval: int = 5,
    ):
        """
        Initialize Visual Crossing client.

        Args:
            api_key: Visual Crossing API key
            base_url: Base URL for Timeline API
            minute_interval: Minute interval for sub-hourly data (default: 5)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.minute_interval = minute_interval
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

        # Rate limiting (be nice to VC servers)
        self.rate_limit_delay = 0.2  # 200ms between requests (5 req/s max)

    def fetch_minutes(
        self,
        location: str,
        start_date: str,
        end_date: str,
        minute_interval: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch minute-level weather data for a date range.

        Args:
            location: Location string (e.g., "stn:KMDW" for Chicago Midway)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            minute_interval: Minute interval (default: 5)

        Returns:
            DataFrame with columns: ts_utc, temp_f, humidity, dew_f, windspeed_mph,
                                    stations (station ID used by VC for diagnostics), raw_json

            Note: windgust_mph, pressure_mb, precip_in, preciptype may be None with
                  current elements list (focused on core weather + station tracking)

        Example:
            >>> client = VisualCrossingClient(api_key="...")
            >>> df = client.fetch_minutes("stn:KMDW", "2025-11-01", "2025-11-01")
        """
        if minute_interval is None:
            minute_interval = self.minute_interval

        # Build URL: {base_url}/{location}/{start}/{end}
        url = f"{self.base_url}/{location}/{start_date}/{end_date}"

        # Build query params (exact as specified in VC station-pinned best practices)
        params = {
            "key": self.api_key,
            "unitGroup": "us",  # Fahrenheit, mph
            "include": "minutes",  # Include minute-level data
            "options": f"useobs,minuteinterval_{minute_interval},nonulls",  # Observed data, 5-min, no nulls
            "elements": "datetimeEpoch,temp,dew,humidity,windspeed,stations",  # Include stations for diagnostics
            "timezone": "Z",  # UTC
            "maxStations": "1",  # Lock to single station only
            "maxDistance": "0",  # No multi-station blending (strict station lock)
            "contentType": "json",
        }

        logger.info(f"Fetching minutes for {location} from {start_date} to {end_date}...")

        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()

            data = response.json()

            # Extract queryCost for billing tracking
            query_cost = data.get("queryCost", 0)
            logger.info(f"Query cost: {query_cost} records")

            # Flatten nested structure: days -> hours -> minutes
            rows = self._flatten_minutes(data)

            # Convert to DataFrame
            df = pd.DataFrame(rows)

            if df.empty:
                logger.warning(f"No minute data returned for {location}")
                return df

            # Remove duplicates and sort
            df = df.drop_duplicates(subset=["ts_utc"]).sort_values("ts_utc")

            logger.info(f"Fetched {len(df)} minute records")

            # Rate limiting: be nice to VC servers
            time.sleep(self.rate_limit_delay)

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Visual Crossing data: {e}")
            raise

    def _flatten_minutes(self, json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Flatten nested JSON structure (days -> hours -> minutes).

        Args:
            json_data: Response JSON from Timeline API

        Returns:
            List of dicts with minute-level observations
        """
        rows = []

        for day in json_data.get("days", []):
            for hour in day.get("hours", []):
                for minute in hour.get("minutes", []):
                    # Parse timestamp
                    ts_utc = pd.to_datetime(minute["datetimeEpoch"], unit="s", utc=True)

                    # Extract weather variables
                    row = {
                        "ts_utc": ts_utc,
                        "temp_f": minute.get("temp"),
                        "humidity": minute.get("humidity"),
                        "dew_f": minute.get("dew"),
                        "windspeed_mph": minute.get("windspeed"),
                        "windgust_mph": minute.get("windgust"),  # May be missing in new elements list
                        "pressure_mb": minute.get("pressure"),  # May be missing in new elements list
                        "precip_in": minute.get("precip"),  # May be missing in new elements list
                        "preciptype": self._format_preciptype(minute.get("preciptype")),  # May be missing
                        "stations": self._format_stations(minute.get("stations")),  # Station ID used for this minute
                        "raw_json": minute,
                    }

                    rows.append(row)

        return rows

    def _format_preciptype(self, preciptype: Any) -> Optional[str]:
        """Format preciptype as comma-separated string."""
        if isinstance(preciptype, list):
            return ",".join(preciptype) if preciptype else None
        elif isinstance(preciptype, str):
            return preciptype
        else:
            return None

    def _format_stations(self, stations: Any) -> Optional[str]:
        """Format stations as comma-separated string."""
        if isinstance(stations, list):
            return ",".join(stations) if stations else None
        elif isinstance(stations, str):
            return stations
        else:
            return None

    def fetch_day_for_station(
        self,
        loc_id: str,
        vc_key: str,
        date: str,
    ) -> pd.DataFrame:
        """
        Fetch minute data for a single day and station.

        Args:
            loc_id: Location ID (e.g., "KMDW")
            vc_key: Visual Crossing location key (e.g., "stn:KMDW")
            date: Date in YYYY-MM-DD format

        Returns:
            DataFrame with minute observations
        """
        return self.fetch_minutes(vc_key, date, date)

    def fetch_range_for_station(
        self,
        loc_id: str,
        vc_key: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch minute data for a date range and station.

        For large ranges, fetches per-day to avoid payload issues.

        Args:
            loc_id: Location ID (e.g., "KMDW")
            vc_key: Visual Crossing location key (e.g., "stn:KMDW")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with minute observations
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Calculate number of days
        num_days = (end_dt - start_dt).days + 1

        if num_days <= 7:
            # Small range: fetch all at once
            return self.fetch_minutes(vc_key, start_date, end_date)
        else:
            # Large range: fetch per-day
            logger.info(f"Fetching {num_days} days for {loc_id} (per-day batching)...")

            all_dfs = []
            current_date = start_dt

            while current_date <= end_dt:
                date_str = current_date.strftime("%Y-%m-%d")

                try:
                    df = self.fetch_day_for_station(loc_id, vc_key, date_str)
                    if not df.empty:
                        all_dfs.append(df)

                except Exception as e:
                    logger.error(f"Error fetching {date_str} for {loc_id}: {e}")

                current_date += timedelta(days=1)

            # Concatenate all DataFrames
            if all_dfs:
                result = pd.concat(all_dfs, ignore_index=True)
                result = result.drop_duplicates(subset=["ts_utc"]).sort_values("ts_utc")
                logger.info(f"Total: {len(result)} minute records for {loc_id}")
                return result
            else:
                logger.warning(f"No data fetched for {loc_id}")
                return pd.DataFrame()

    def fetch_all_stations(
        self,
        start_date: str,
        end_date: str,
        cities: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch minute data for all stations (or subset) in a date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            cities: List of city names (default: all stations)

        Returns:
            Dict mapping city name to DataFrame
        """
        if cities is None:
            cities = list(STATION_MAP.keys())

        results = {}

        for city in cities:
            if city not in STATION_MAP:
                logger.warning(f"Unknown city: {city}. Available: {list(STATION_MAP.keys())}")
                continue

            loc_id, vc_key = STATION_MAP[city]

            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Fetching Visual Crossing data for {city.upper()} ({loc_id})")
                logger.info(f"{'='*60}\n")

                df = self.fetch_range_for_station(loc_id, vc_key, start_date, end_date)
                results[city] = df

            except Exception as e:
                logger.error(f"Error fetching {city}: {e}")
                results[city] = pd.DataFrame()

        return results


def main():
    """Demo: Fetch minute data for Chicago last 3 days."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import os
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("VC_API_KEY")
    if not api_key:
        logger.error("VC_API_KEY not found in environment")
        return

    # Fetch last 3 days for Chicago
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=3)

    client = VisualCrossingClient(api_key=api_key)

    df = client.fetch_range_for_station(
        "KMDW",
        "stn:KMDW",
        start_date.isoformat(),
        end_date.isoformat(),
    )

    print("\n" + "=" * 60)
    print(f"Chicago Midway minute data (last 3 days)")
    print("=" * 60)
    print(f"\nFetched {len(df)} records")
    print(f"\nSample data:")
    print(df.head(10)[["ts_utc", "temp_f", "humidity", "windspeed_mph", "precip_in"]])


if __name__ == "__main__":
    main()
