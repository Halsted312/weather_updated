"""
Utility functions for live trading.

Includes:
- Timezone handling per city
- Weather day calculations (NWS climate day rules)
- Event date helpers
"""

import logging
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# City timezone mappings (IANA timezone strings)
CITY_TIMEZONES: Dict[str, str] = {
    'chicago': 'America/Chicago',
    'austin': 'America/Chicago',  # Central time
    'denver': 'America/Denver',
    'los_angeles': 'America/Los_Angeles',
    'miami': 'America/New_York',  # Eastern time
    'philadelphia': 'America/New_York',
}


def get_city_timezone(city: str) -> ZoneInfo:
    """
    Get ZoneInfo object for city.

    Args:
        city: City identifier

    Returns:
        ZoneInfo instance for city's timezone

    Raises:
        KeyError: If city not recognized
    """
    if city not in CITY_TIMEZONES:
        raise KeyError(
            f"Unknown city '{city}'. "
            f"Valid cities: {list(CITY_TIMEZONES.keys())}"
        )

    return ZoneInfo(CITY_TIMEZONES[city])


def get_current_weather_day(city: str, utc_now: Optional[datetime] = None) -> date:
    """
    Get the current weather day for a city.

    NWS climate days use LOCAL STANDARD TIME:
    - Weather day runs from ~6am local to ~6am next day
    - If before 6am, we're still in "yesterday's" weather day

    Args:
        city: City identifier
        utc_now: Optional UTC timestamp (default: now)

    Returns:
        Local date representing current weather day
    """
    if utc_now is None:
        utc_now = datetime.now(ZoneInfo("UTC"))

    # Convert to city local time
    city_tz = get_city_timezone(city)
    local_now = utc_now.astimezone(city_tz)

    # Weather day typically ends around 6am local
    # If before 6am, we're still in "yesterday's" weather day
    if local_now.hour < 6:
        weather_day = (local_now - timedelta(days=1)).date()
    else:
        weather_day = local_now.date()

    return weather_day


def get_next_weather_day(city: str, utc_now: Optional[datetime] = None) -> date:
    """
    Get tomorrow's weather day for a city.

    Args:
        city: City identifier
        utc_now: Optional UTC timestamp (default: now)

    Returns:
        Tomorrow's local date
    """
    current_day = get_current_weather_day(city, utc_now)
    return current_day + timedelta(days=1)


def get_market_close_local(city: str, event_date: date) -> datetime:
    """
    Get market close time in city's local timezone.

    Weather markets typically close around midnight local time
    (end of the weather day).

    Args:
        city: City identifier
        event_date: Event date for the market

    Returns:
        Timestamp of market close in local timezone
    """
    city_tz = get_city_timezone(city)

    # Midnight local = end of weather day
    close_time = datetime.combine(
        event_date + timedelta(days=1),
        time(0, 0),
        tzinfo=city_tz
    )

    return close_time


def seconds_until_market_close(city: str, event_date: date, current_time: Optional[datetime] = None) -> int:
    """
    Get seconds remaining until market close.

    Args:
        city: City identifier
        event_date: Event date
        current_time: Optional current time (default: now)

    Returns:
        Seconds until market close (negative if market closed)
    """
    if current_time is None:
        current_time = datetime.now(ZoneInfo("UTC"))

    close_time = get_market_close_local(city, event_date)
    close_time_utc = close_time.astimezone(ZoneInfo("UTC"))

    delta = (close_time_utc - current_time).total_seconds()
    return int(delta)


def is_market_open(city: str, event_date: date, current_time: Optional[datetime] = None) -> bool:
    """
    Check if market for event_date is still open for trading.

    Args:
        city: City identifier
        event_date: Event date
        current_time: Optional current time (default: now)

    Returns:
        True if market is open, False if closed
    """
    seconds_left = seconds_until_market_close(city, event_date, current_time)
    return seconds_left > 0


def get_local_time(city: str, utc_time: Optional[datetime] = None) -> datetime:
    """
    Convert UTC time to city's local time.

    Args:
        city: City identifier
        utc_time: Optional UTC timestamp (default: now)

    Returns:
        Timestamp in city's local timezone
    """
    if utc_time is None:
        utc_time = datetime.now(ZoneInfo("UTC"))

    city_tz = get_city_timezone(city)
    return utc_time.astimezone(city_tz)


def format_temperature(temp_f: float, precision: int = 1) -> str:
    """
    Format temperature with degree symbol.

    Args:
        temp_f: Temperature in Fahrenheit
        precision: Decimal places

    Returns:
        Formatted string (e.g., "75.3°F")
    """
    return f"{temp_f:.{precision}f}°F"


def format_edge(edge_degf: float, precision: int = 2) -> str:
    """
    Format edge with sign and degree symbol.

    Args:
        edge_degf: Edge in degrees F (positive = market too low)
        precision: Decimal places

    Returns:
        Formatted string (e.g., "+2.3°F" or "-1.5°F")
    """
    sign = "+" if edge_degf >= 0 else ""
    return f"{sign}{edge_degf:.{precision}f}°F"
