"""Helpers for deriving trading dates from Kalshi market metadata."""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict

from ml.city_config import CITY_CONFIG
from weather.time_utils import coerce_datetime_to_utc, local_date_from_utc

SERIES_TO_CITY: Dict[str, str] = {
    f"KXHIGH{cfg['series_code']}": city for city, cfg in CITY_CONFIG.items()
}


def series_ticker_for_city(city: str) -> str:
    """Return the canonical series ticker (e.g., KXHIGHCHI) for a city key."""

    if city not in CITY_CONFIG:
        raise ValueError(f"Unknown city '{city}'. Expected one of {sorted(CITY_CONFIG)}")
    return f"KXHIGH{CITY_CONFIG[city]['series_code']}"


def city_from_series(series_ticker: str) -> str:
    """Map a Kalshi series ticker back to the canonical city key."""

    try:
        return SERIES_TO_CITY[series_ticker]
    except KeyError as exc:
        raise ValueError(
            f"Unknown series_ticker '{series_ticker}'. Known: {sorted(SERIES_TO_CITY)}"
        ) from exc


def event_date_from_close_time(series_ticker: str, close_time: datetime | str) -> date:
    """Convert a UTC close_time into the trading-day date for a series."""

    city = city_from_series(series_ticker)
    close_time_utc = coerce_datetime_to_utc(close_time)
    return local_date_from_utc(close_time_utc, city)
