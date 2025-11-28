#!/usr/bin/env python3
"""
Visual Crossing Weather API client for minute-level observations and forecasts.

Supports two query modes:
1. Station-locked: Uses stn:{ICAO} with maxStations=1, maxDistance=1609 for exact station data
2. City-aggregate: Uses "City,State" format, lets VC interpolate from multiple stations

Uses Timeline API with specific parameters:
- include=obs,minutes for historical observations (5-min intervals)
- include=fcst,current,minutes for current+forecast (15-min intervals)
- include=days,hours for daily/hourly forecasts
- forecastBasisDate for historical forecasts (backtesting)

Stores ALL available weather fields including extended wind/solar via vc_elements.py.

API Docs: https://www.visualcrossing.com/resources/documentation/weather-api/timeline-weather-api/
"""

import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests

from src.config import CITIES, EXCLUDED_VC_CITIES, get_city
from src.config.vc_elements import build_elements_string, VC_TO_DB_FIELD_MAP

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

    # ==========================================================================
    # NEW METHODS FOR VC GREENFIELD SCHEMA
    # ==========================================================================

    def fetch_station_history_minutes(
        self,
        station_id: str,
        start_date: Union[str, date],
        end_date: Union[str, date],
        minute_interval: int = 5,
    ) -> Dict[str, Any]:
        """
        Fetch historical minute-level observations locked to a single weather station.

        Uses station-locking parameters to ensure data comes from exactly one station
        (e.g., KMDW) without any interpolation from nearby stations.

        Args:
            station_id: ICAO station code (e.g., 'KMDW', 'KDEN')
            start_date: Start date (YYYY-MM-DD or date object)
            end_date: End date (YYYY-MM-DD or date object)
            minute_interval: Minute interval (default: 5)

        Returns:
            Raw API response dict with 'days' containing 'hours' containing 'minutes'
        """
        if isinstance(start_date, date):
            start_date = start_date.isoformat()
        if isinstance(end_date, date):
            end_date = end_date.isoformat()

        location = f"stn:{station_id}"
        url = f"{self.base_url}/{location}/{start_date}/{end_date}"

        params = {
            "key": self.api_key,
            "unitGroup": "us",
            "include": "obs,minutes",
            "options": f"minuteinterval_{minute_interval},stnslevel1,useobs",
            "elements": build_elements_string(),
            "timezone": "Z",  # UTC
            # Station-locking parameters
            "maxStations": "1",
            "maxDistance": "1609",  # ~1 mile in meters
            "elevationDifference": "50",
            "contentType": "json",
        }

        logger.info(f"Fetching station history: {station_id} from {start_date} to {end_date}")

        try:
            response = self.session.get(url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()

            query_cost = data.get("queryCost", 0)
            logger.info(f"Query cost: {query_cost}")

            time.sleep(self.rate_limit_delay)
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching station history: {e}")
            raise

    def fetch_city_history_minutes(
        self,
        city_query: str,
        start_date: Union[str, date],
        end_date: Union[str, date],
        minute_interval: int = 5,
    ) -> Dict[str, Any]:
        """
        Fetch historical minute-level observations for a city (interpolated).

        Uses city name format (e.g., "Chicago,IL") which allows Visual Crossing
        to interpolate from multiple nearby stations for better coverage.

        Args:
            city_query: City query string (e.g., 'Chicago,IL', 'Denver,CO')
            start_date: Start date (YYYY-MM-DD or date object)
            end_date: End date (YYYY-MM-DD or date object)
            minute_interval: Minute interval (default: 5)

        Returns:
            Raw API response dict with 'days' containing 'hours' containing 'minutes'
        """
        if isinstance(start_date, date):
            start_date = start_date.isoformat()
        if isinstance(end_date, date):
            end_date = end_date.isoformat()

        url = f"{self.base_url}/{city_query}/{start_date}/{end_date}"

        params = {
            "key": self.api_key,
            "unitGroup": "us",
            "include": "obs,minutes",
            "options": f"minuteinterval_{minute_interval},useobs",
            "elements": build_elements_string(),
            "timezone": "Z",  # UTC
            # NO maxStations/maxDistance - let VC interpolate
            "contentType": "json",
        }

        logger.info(f"Fetching city history: {city_query} from {start_date} to {end_date}")

        try:
            response = self.session.get(url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()

            query_cost = data.get("queryCost", 0)
            logger.info(f"Query cost: {query_cost}")

            time.sleep(self.rate_limit_delay)
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching city history: {e}")
            raise

    def fetch_station_current_and_forecast(
        self,
        station_id: str,
        horizon_days: int = 7,
        minute_interval: int = 15,
    ) -> Dict[str, Any]:
        """
        Fetch current conditions + forecast for a station (15-min minute data).

        Used for nightly forecast snapshots. Includes current conditions,
        daily forecasts, hourly forecasts, and 15-minute minute-level forecasts.

        Args:
            station_id: ICAO station code (e.g., 'KMDW', 'KDEN')
            horizon_days: Number of forecast days (default: 7)
            minute_interval: Minute interval for forecast (default: 15, API minimum for fcst)

        Returns:
            Raw API response dict with days, hours, minutes
        """
        location = f"stn:{station_id}"
        url = f"{self.base_url}/{location}/next{horizon_days}days"

        params = {
            "key": self.api_key,
            "unitGroup": "us",
            "include": "fcst,current,days,hours,minutes",
            "options": f"minuteinterval_{minute_interval},stnslevel1,usefcst",
            "elements": build_elements_string(),
            "timezone": "Z",
            # Station-locking
            "maxStations": "1",
            "maxDistance": "1609",
            "elevationDifference": "50",
            "contentType": "json",
        }

        logger.info(f"Fetching station forecast: {station_id}, {horizon_days} days")

        try:
            response = self.session.get(url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()

            query_cost = data.get("queryCost", 0)
            logger.info(f"Query cost: {query_cost}")

            time.sleep(self.rate_limit_delay)
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching station forecast: {e}")
            raise

    def fetch_city_current_and_forecast(
        self,
        city_query: str,
        horizon_days: int = 7,
        minute_interval: int = 15,
    ) -> Dict[str, Any]:
        """
        Fetch current conditions + forecast for a city (interpolated, 15-min).

        Used for nightly forecast snapshots with city-aggregate data.

        Args:
            city_query: City query string (e.g., 'Chicago,IL')
            horizon_days: Number of forecast days (default: 7)
            minute_interval: Minute interval for forecast (default: 15)

        Returns:
            Raw API response dict with days, hours, minutes
        """
        url = f"{self.base_url}/{city_query}/next{horizon_days}days"

        params = {
            "key": self.api_key,
            "unitGroup": "us",
            "include": "fcst,current,days,hours,minutes",
            "options": f"minuteinterval_{minute_interval},usefcst",
            "elements": build_elements_string(),
            "timezone": "Z",
            "contentType": "json",
        }

        logger.info(f"Fetching city forecast: {city_query}, {horizon_days} days")

        try:
            response = self.session.get(url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()

            query_cost = data.get("queryCost", 0)
            logger.info(f"Query cost: {query_cost}")

            time.sleep(self.rate_limit_delay)
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching city forecast: {e}")
            raise

    def fetch_station_historical_forecast(
        self,
        station_id: str,
        target_start: Union[str, date],
        target_end: Union[str, date],
        basis_date: Union[str, date],
    ) -> Dict[str, Any]:
        """
        Fetch historical forecast as it looked on a past basis_date for a station.

        Used for backtesting: "What was the forecast for Dec 15 when made on Dec 10?"

        Args:
            station_id: ICAO station code (e.g., 'KMDW')
            target_start: Start of forecast target range
            target_end: End of forecast target range
            basis_date: The date the forecast was made

        Returns:
            Raw API response with days and hours arrays
        """
        if isinstance(target_start, date):
            target_start = target_start.isoformat()
        if isinstance(target_end, date):
            target_end = target_end.isoformat()
        if isinstance(basis_date, date):
            basis_date = basis_date.isoformat()

        location = f"stn:{station_id}"
        url = f"{self.base_url}/{location}/{target_start}/{target_end}"

        params = {
            "key": self.api_key,
            "unitGroup": "us",
            "include": "days,hours",
            "forecastBasisDate": basis_date,
            "elements": build_elements_string(),
            # Station-locking
            "maxStations": "1",
            "maxDistance": "1609",
            "elevationDifference": "50",
            "contentType": "json",
        }

        logger.debug(f"Fetching historical forecast: {station_id}, basis={basis_date}, target={target_start}/{target_end}")

        try:
            response = self.session.get(url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()

            query_cost = data.get("queryCost", 0)
            logger.debug(f"Query cost: {query_cost}")

            time.sleep(self.rate_limit_delay)
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical forecast: {e}")
            raise

    def fetch_city_historical_forecast(
        self,
        city_query: str,
        target_start: Union[str, date],
        target_end: Union[str, date],
        basis_date: Union[str, date],
    ) -> Dict[str, Any]:
        """
        Fetch historical forecast as it looked on a past basis_date for a city.

        Used for backtesting with city-aggregate data.

        Args:
            city_query: City query string (e.g., 'Chicago,IL')
            target_start: Start of forecast target range
            target_end: End of forecast target range
            basis_date: The date the forecast was made

        Returns:
            Raw API response with days and hours arrays
        """
        if isinstance(target_start, date):
            target_start = target_start.isoformat()
        if isinstance(target_end, date):
            target_end = target_end.isoformat()
        if isinstance(basis_date, date):
            basis_date = basis_date.isoformat()

        url = f"{self.base_url}/{city_query}/{target_start}/{target_end}"

        params = {
            "key": self.api_key,
            "unitGroup": "us",
            "include": "days,hours",
            "forecastBasisDate": basis_date,
            "elements": build_elements_string(),
            "contentType": "json",
        }

        logger.debug(f"Fetching historical forecast: {city_query}, basis={basis_date}, target={target_start}/{target_end}")

        try:
            response = self.session.get(url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()

            query_cost = data.get("queryCost", 0)
            logger.debug(f"Query cost: {query_cost}")

            time.sleep(self.rate_limit_delay)
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical forecast: {e}")
            raise

    @staticmethod
    def map_vc_to_db(vc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map Visual Crossing field names to database column names.

        Uses VC_TO_DB_FIELD_MAP from vc_elements.py.

        Args:
            vc_data: Dict with Visual Crossing field names

        Returns:
            Dict with database column names
        """
        result = {}
        for vc_key, value in vc_data.items():
            if vc_key in VC_TO_DB_FIELD_MAP:
                db_key = VC_TO_DB_FIELD_MAP[vc_key]
                # Special handling for tzoffset (VC returns hours, we store minutes)
                if vc_key == "tzoffset" and value is not None:
                    value = int(value * 60)
                result[db_key] = value
            else:
                # Keep unmapped fields as-is (like datetimeEpoch, datetime)
                result[vc_key] = value
        return result
