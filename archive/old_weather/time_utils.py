"""Timezone utilities shared across weather + ingestion modules."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Union
from zoneinfo import ZoneInfo

# Canonical timezone mapping for all supported cities (plus historical aliases)
CITY_TIMEZONES: Dict[str, str] = {
    "austin": "America/Chicago",
    "chicago": "America/Chicago",
    "denver": "America/Denver",
    "la": "America/Los_Angeles",
    "los_angeles": "America/Los_Angeles",
    "miami": "America/New_York",
    "philadelphia": "America/New_York",
    # Legacy entries kept for compatibility with historical scripts/tests
    "new_york": "America/New_York",
}


def utc_now() -> datetime:
    """Return an explicit UTC timestamp."""

    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime instance is timezone-aware and normalized to UTC."""

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def coerce_datetime_to_utc(value: Union[str, datetime]) -> datetime:
    """Parse ISO8601 strings or datetimes into timezone-aware UTC datetimes."""

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        # Kalshi + VC timestamps are ISO strings that may end in "Z"
        cleaned = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
    else:
        raise TypeError(f"Unsupported datetime value: {type(value)!r}")

    return ensure_utc(dt)


def get_timezone_name(city: str) -> str:
    """Return the canonical timezone name for a city."""

    try:
        return CITY_TIMEZONES[city]
    except KeyError as exc:
        raise ValueError(
            f"No timezone mapping for city '{city}'. Known: {sorted(CITY_TIMEZONES)}"
        ) from exc


def get_timezone(city: str) -> ZoneInfo:
    """Return a ZoneInfo instance for the city."""

    return ZoneInfo(get_timezone_name(city))


def to_city_local(dt_utc: datetime, city: str) -> datetime:
    """Convert a UTC timestamp to the city's local timezone."""

    return ensure_utc(dt_utc).astimezone(get_timezone(city))


def local_date_from_utc(dt_utc: datetime, city: str):
    """Return the local calendar date for a UTC timestamp in the city's timezone."""

    return to_city_local(dt_utc, city).date()
