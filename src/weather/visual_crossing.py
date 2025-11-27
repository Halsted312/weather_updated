#!/usr/bin/env python3
"""
Visual Crossing Weather API client for minute-level observations.

Uses Timeline API with specific parameters:
- include=minutes for sub-hourly data
- options=minuteinterval_5,nonulls,useobs for 5-min cadence with observations
- timezone=Z for UTC timestamps
- maxStations=1 & maxDistance=0 to lock to specific ICAO station
- stn:<ICAO> location format to target exact station

Stores ALL available weather fields for future-proofing:
- temp, feelslike, humidity, dew
- precip, precipprob, preciptype, snow, snowdepth
- windspeed, winddir, windgust
- pressure, visibility, cloudcover
- solarradiation, solarenergy, uvindex
- conditions, icon
- stations (for audit trail)

API Docs: https://www.visualcrossing.com/resources/documentation/weather-api/timeline-weather-api/
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from src.config import CITIES, EXCLUDED_VC_CITIES, get_city

logger = logging.getLogger(__name__)


class VisualCrossingClient:
    """Client for Visual Crossing Timeline API with minute-level data."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline",
        minute_interval: int = 5,
        rate_limit_delay: float = 0.2,
    ):
        """
        Initialize Visual Crossing client.

        Args:
            api_key: Visual Crossing API key
            base_url: Base URL for Timeline API
            minute_interval: Minute interval for sub-hourly data (default: 5)
            rate_limit_delay: Delay between requests in seconds (default: 0.2)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.minute_interval = minute_interval
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

        logger.info(f"Visual Crossing client initialized (interval={minute_interval}m)")

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
            minute_interval: Minute interval (default: uses instance default)

        Returns:
            DataFrame with ALL available weather fields
        """
        if minute_interval is None:
            minute_interval = self.minute_interval

        # Build URL: {base_url}/{location}/{start}/{end}
        url = f"{self.base_url}/{location}/{start_date}/{end_date}"

        # Request ALL available elements for future-proofing
        # This captures everything Visual Crossing provides
        elements = ",".join([
            "datetimeEpoch",
            "temp",
            "feelslike",
            "humidity",
            "dew",
            "precip",
            "precipprob",
            "preciptype",
            "snow",
            "snowdepth",
            "windspeed",
            "winddir",
            "windgust",
            "pressure",
            "visibility",
            "cloudcover",
            "solarradiation",
            "solarenergy",
            "uvindex",
            "conditions",
            "icon",
            "stations",
        ])

        params = {
            "key": self.api_key,
            "unitGroup": "us",  # Fahrenheit, mph, inches
            "include": "minutes",  # Include minute-level data
            "options": f"useobs,minuteinterval_{minute_interval},nonulls",
            "elements": elements,
            "timezone": "Z",  # UTC
            "maxStations": "1",  # Lock to single station
            "maxDistance": "0",  # No multi-station blending
            "contentType": "json",
        }

        logger.info(f"Fetching minutes for {location} from {start_date} to {end_date}...")

        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()

            data = response.json()

            # Track query cost for billing
            query_cost = data.get("queryCost", 0)
            logger.info(f"Query cost: {query_cost} records")

            # Flatten nested structure
            rows = self._flatten_minutes(data)

            df = pd.DataFrame(rows)

            if df.empty:
                logger.warning(f"No minute data returned for {location}")
                return df

            # Remove duplicates and sort
            df = df.drop_duplicates(subset=["ts_utc"]).sort_values("ts_utc")

            logger.info(f"Fetched {len(df)} minute records")

            # Rate limiting
            time.sleep(self.rate_limit_delay)

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Visual Crossing data: {e}")
            raise

    def _flatten_minutes(self, json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Flatten nested JSON structure (days -> hours -> minutes).

        Captures ALL available fields for future-proofing.
        """
        rows = []

        for day in json_data.get("days", []):
            for hour in day.get("hours", []):
                for minute in hour.get("minutes", []):
                    # Parse timestamp
                    ts_utc = pd.to_datetime(minute["datetimeEpoch"], unit="s", utc=True)

                    # Extract ALL weather variables (store everything!)
                    row = {
                        "ts_utc": ts_utc,
                        # Temperature
                        "temp_f": minute.get("temp"),
                        "feelslike_f": minute.get("feelslike"),
                        "dew_f": minute.get("dew"),
                        # Humidity
                        "humidity": minute.get("humidity"),
                        # Precipitation
                        "precip_in": minute.get("precip"),
                        "precip_prob": minute.get("precipprob"),
                        "precip_type": self._format_list(minute.get("preciptype")),
                        "snow_in": minute.get("snow"),
                        "snow_depth_in": minute.get("snowdepth"),
                        # Wind
                        "windspeed_mph": minute.get("windspeed"),
                        "winddir_deg": minute.get("winddir"),
                        "windgust_mph": minute.get("windgust"),
                        # Pressure and visibility
                        "pressure_mb": minute.get("pressure"),
                        "visibility_mi": minute.get("visibility"),
                        # Cloud and solar
                        "cloud_cover": minute.get("cloudcover"),
                        "solar_radiation": minute.get("solarradiation"),
                        "solar_energy": minute.get("solarenergy"),
                        "uv_index": minute.get("uvindex"),
                        # Conditions
                        "conditions": minute.get("conditions"),
                        "icon": minute.get("icon"),
                        # Station audit trail
                        "stations": self._format_list(minute.get("stations")),
                        # Raw JSON for future fields
                        "raw_json": minute,
                    }

                    rows.append(row)

        return rows

    def _format_list(self, value: Any) -> Optional[str]:
        """Format list as comma-separated string."""
        if isinstance(value, list):
            return ",".join(str(v) for v in value) if value else None
        elif isinstance(value, str):
            return value
        return None

    def fetch_day_for_city(self, city_id: str, date_str: str) -> pd.DataFrame:
        """
        Fetch minute data for a single day and city.

        Args:
            city_id: City ID (e.g., "chicago")
            date_str: Date in YYYY-MM-DD format

        Returns:
            DataFrame with minute observations
        """
        city = get_city(city_id)
        if not city:
            raise ValueError(f"Unknown city: {city_id}")

        if city_id in EXCLUDED_VC_CITIES:
            logger.warning(f"City {city_id} is excluded from Visual Crossing (high forward-fill)")
            return pd.DataFrame()

        location = f"stn:{city.icao}"
        return self.fetch_minutes(location, date_str, date_str)

    def fetch_range_for_city(
        self,
        city_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch minute data for a date range and city.

        For large ranges, fetches per-day to avoid payload issues.

        Args:
            city_id: City ID (e.g., "chicago")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with minute observations
        """
        city = get_city(city_id)
        if not city:
            raise ValueError(f"Unknown city: {city_id}")

        if city_id in EXCLUDED_VC_CITIES:
            logger.warning(f"City {city_id} is excluded from Visual Crossing")
            return pd.DataFrame()

        location = f"stn:{city.icao}"
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        num_days = (end_dt - start_dt).days + 1

        if num_days <= 7:
            # Small range: fetch all at once
            return self.fetch_minutes(location, start_date, end_date)
        else:
            # Large range: fetch per-day
            logger.info(f"Fetching {num_days} days for {city_id} (per-day batching)...")

            all_dfs = []
            current_date = start_dt

            while current_date <= end_dt:
                date_str = current_date.strftime("%Y-%m-%d")

                try:
                    df = self.fetch_minutes(location, date_str, date_str)
                    if not df.empty:
                        all_dfs.append(df)
                except Exception as e:
                    logger.error(f"Error fetching {date_str} for {city_id}: {e}")

                current_date += timedelta(days=1)

            if all_dfs:
                result = pd.concat(all_dfs, ignore_index=True)
                result = result.drop_duplicates(subset=["ts_utc"]).sort_values("ts_utc")
                logger.info(f"Total: {len(result)} minute records for {city_id}")
                return result
            else:
                return pd.DataFrame()

    def fetch_all_cities(
        self,
        start_date: str,
        end_date: str,
        cities: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch minute data for all cities (or subset) in a date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            cities: List of city IDs (default: all non-excluded cities)

        Returns:
            Dict mapping city ID to DataFrame
        """
        if cities is None:
            cities = [c for c in CITIES.keys() if c not in EXCLUDED_VC_CITIES]

        results = {}

        for city_id in cities:
            if city_id in EXCLUDED_VC_CITIES:
                logger.warning(f"Skipping {city_id} (excluded from Visual Crossing)")
                continue

            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Fetching Visual Crossing data for {city_id.upper()}")
                logger.info(f"{'='*60}\n")

                df = self.fetch_range_for_city(city_id, start_date, end_date)
                results[city_id] = df

            except Exception as e:
                logger.error(f"Error fetching {city_id}: {e}")
                results[city_id] = pd.DataFrame()

        return results

    def fetch_historical_daily_forecast(
        self,
        location: str,
        basis_date: str,
        horizon_days: int = 15,
    ) -> Dict[str, Any]:
        """
        Fetch the historical daily forecast as it looked on a specific past date.

        Uses the Visual Crossing Timeline API with `forecastBasisDate` parameter
        to retrieve what the forecast was for upcoming days as predicted on basis_date.

        API Docs: https://www.visualcrossing.com/resources/documentation/weather-data/
                  how-to-query-weather-forecasts-from-the-past-historical-forecasts/

        Args:
            location: Location string (e.g., "stn:KMDW" for Chicago Midway)
            basis_date: The date the forecast was made (YYYY-MM-DD format)
            horizon_days: Number of days in the forecast window (default: 15)

        Returns:
            Dict containing forecast data with "days" list, each day having:
            - datetime: Target date (YYYY-MM-DD)
            - tempmax: Forecast high temperature (°F)
            - tempmin: Forecast low temperature (°F)
            - precip: Forecast precipitation (inches)
            - precipprob: Precipitation probability (%)
            - humidity: Humidity (%)
            - windspeed: Wind speed (mph)
            - conditions: Conditions text
            - icon: Weather icon name
            - and more fields stored in raw response
        """
        # Calculate end date for the forecast horizon
        from datetime import datetime as dt

        basis_dt = dt.strptime(basis_date, "%Y-%m-%d")
        end_dt = basis_dt + timedelta(days=horizon_days - 1)
        end_date = end_dt.strftime("%Y-%m-%d")

        # Build URL: {base_url}/{location}/{start_date}/{end_date}
        url = f"{self.base_url}/{location}/{basis_date}/{end_date}"

        # Request daily forecast elements
        elements = ",".join([
            "datetime",
            "datetimeEpoch",
            "tempmax",
            "tempmin",
            "temp",
            "feelslikemax",
            "feelslikemin",
            "precip",
            "precipprob",
            "preciptype",
            "snow",
            "snowdepth",
            "humidity",
            "windspeed",
            "windgust",
            "winddir",
            "cloudcover",
            "visibility",
            "pressure",
            "conditions",
            "icon",
            "source",
        ])

        params = {
            "key": self.api_key,
            "unitGroup": "us",  # Fahrenheit, mph, inches
            "include": "days",  # Daily data only (not hours/minutes)
            "forecastBasisDate": basis_date,  # The key parameter for historical forecasts
            "elements": elements,
            "contentType": "json",
        }

        logger.debug(f"Fetching historical forecast for {location} as of {basis_date}...")

        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()

            data = response.json()

            # Track query cost for billing
            query_cost = data.get("queryCost", 0)
            days_count = len(data.get("days", []))
            logger.debug(f"Query cost: {query_cost}, days returned: {days_count}")

            # Rate limiting
            time.sleep(self.rate_limit_delay)

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical forecast: {e}")
            raise

    def fetch_historical_hourly_forecast(
        self,
        location: str,
        basis_date: str,
        horizon_hours: int = 72,
        horizon_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Fetch the historical hourly + daily forecast as it looked on a specific past date.

        Uses the Visual Crossing Timeline API with `forecastBasisDate` parameter
        and `include=days,hours` to get both daily summaries and hourly forecasts.

        Args:
            location: Location string (e.g., "stn:KMDW" for Chicago Midway)
            basis_date: The date the forecast was made (YYYY-MM-DD format)
            horizon_hours: Max hours of hourly data to return (default: 72 = 3 days)
            horizon_days: Number of days to fetch (default: 7)

        Returns:
            Dict containing forecast data with "days" list, each day having:
            - datetime: Target date (YYYY-MM-DD)
            - tempmax, tempmin: Daily high/low
            - hours: List of 24 hourly forecasts with:
              - datetime: Local time (YYYY-MM-DDTHH:00:00)
              - datetimeEpoch: UTC epoch timestamp
              - temp, feelslike, humidity, precip, precipprob, windspeed, conditions
        """
        from datetime import datetime as dt

        basis_dt = dt.strptime(basis_date, "%Y-%m-%d")
        end_dt = basis_dt + timedelta(days=horizon_days - 1)
        end_date = end_dt.strftime("%Y-%m-%d")

        # Build URL: {base_url}/{location}/{start_date}/{end_date}
        url = f"{self.base_url}/{location}/{basis_date}/{end_date}"

        # Request both daily and hourly elements
        elements = ",".join([
            "datetime",
            "datetimeEpoch",
            # Daily
            "tempmax",
            "tempmin",
            "temp",
            "feelslikemax",
            "feelslikemin",
            "feelslike",
            # Hourly & daily
            "humidity",
            "dew",
            "precip",
            "precipprob",
            "preciptype",
            "windspeed",
            "windgust",
            "winddir",
            "pressure",
            "visibility",
            "cloudcover",
            "conditions",
            "icon",
        ])

        params = {
            "key": self.api_key,
            "unitGroup": "us",  # Fahrenheit, mph, inches
            "include": "days,hours",  # Both daily summaries AND hourly data
            "forecastBasisDate": basis_date,
            "elements": elements,
            "contentType": "json",
            # NO timezone=Z - let VC return local times (per design decision)
        }

        logger.debug(f"Fetching hourly forecast for {location} as of {basis_date}...")

        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()

            data = response.json()

            # Track query cost
            query_cost = data.get("queryCost", 0)
            days_count = len(data.get("days", []))
            total_hours = sum(len(day.get("hours", [])) for day in data.get("days", []))
            logger.debug(
                f"Query cost: {query_cost}, days: {days_count}, hours: {total_hours}"
            )

            # Rate limiting
            time.sleep(self.rate_limit_delay)

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching hourly forecast: {e}")
            raise
