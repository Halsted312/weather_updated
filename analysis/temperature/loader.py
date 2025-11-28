"""
Database loading utilities for temperature reverse-engineering.

Handles:
- Converting local calendar days to UTC windows using city timezones
- Loading settlement data from wx.settlement
- Loading Visual Crossing 5-minute temps from wx.vc_minute_weather
- Combining into DaySeries objects for analysis
"""

from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.orm import Session

from analysis.temperature.datastructures import DaySeries
from src.config import CITIES


def get_cli_window_utc(city: str, day: date) -> Tuple[datetime, datetime]:
    """Convert local calendar day to UTC window for CLI data.

    The NWS Daily Climate Report (CLI) uses local midnight-to-midnight
    for the daily high temperature. We need to convert this local day
    to UTC timestamps to query the vc_minute_weather table.

    Args:
        city: City identifier (must exist in CITIES config)
        day: Local calendar date

    Returns:
        (start_utc, end_utc) tuple of timezone-aware datetimes

    Example:
        Chicago (America/Chicago) on 2025-11-20:
        - Local: 2025-11-20 00:00:00 CST to 2025-11-21 00:00:00 CST
        - UTC: 2025-11-20 06:00:00 UTC to 2025-11-21 06:00:00 UTC
    """
    city_config = CITIES[city]
    tz = ZoneInfo(city_config.timezone)

    # Local midnight to local midnight
    local_start = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=tz)
    local_end = local_start + timedelta(days=1)

    # Convert to UTC
    start_utc = local_start.astimezone(ZoneInfo("UTC"))
    end_utc = local_end.astimezone(ZoneInfo("UTC"))

    return start_utc, end_utc


def load_settlement(session: Session, city: str, day: date) -> Optional[int]:
    """Load settlement temperature for a city/day from wx.settlement.

    Args:
        session: SQLAlchemy session
        city: City identifier
        day: Local calendar date

    Returns:
        Integer °F daily high from NWS/IEM, or None if not found
    """
    query = text(
        """
        SELECT tmax_final
        FROM wx.settlement
        WHERE city = :city
          AND date_local = :day
        """
    )

    result = session.execute(query, {"city": city, "day": day}).fetchone()

    if result and result[0] is not None:
        return int(result[0])

    return None


def load_vc_temps_for_day(
    session: Session, city: str, day: date
) -> List[float]:
    """Load Visual Crossing 5-minute temperatures for a CLI day.

    Queries wx.vc_minute_weather for all temps in the local CLI window
    (midnight to midnight in the city's timezone, converted to UTC).

    Args:
        session: SQLAlchemy session
        city: City identifier
        day: Local calendar date

    Returns:
        List of float temperatures in °F, sorted by timestamp.
        Empty list if no data found.

    Note:
        Filters out NULL temps and returns only valid float values.
    """
    start_utc, end_utc = get_cli_window_utc(city, day)

    # Query for station-locked data (data_type='actual_obs')
    # We want historical observations, not forecasts
    query = text(
        """
        SELECT temp_f
        FROM wx.vc_minute_weather vmw
        JOIN wx.vc_location loc ON vmw.vc_location_id = loc.id
        WHERE loc.city_code = :city_code
          AND loc.location_type = 'station'
          AND vmw.data_type = 'actual_obs'
          AND vmw.datetime_utc >= :start_utc
          AND vmw.datetime_utc < :end_utc
          AND vmw.temp_f IS NOT NULL
        ORDER BY vmw.datetime_utc
        """
    )

    # Map city id to city_code (e.g., 'chicago' → 'CHI')
    city_config = CITIES[city]
    city_code = city_config.city_code

    results = session.execute(
        query, {"city_code": city_code, "start_utc": start_utc, "end_utc": end_utc}
    ).fetchall()

    temps = [float(r[0]) for r in results]
    return temps


def load_day_series(
    session: Session, city: str, day: date
) -> Optional[DaySeries]:
    """Load complete DaySeries for temperature analysis.

    Combines settlement high and VC 5-minute temps into a single
    analysis-ready object.

    Args:
        session: SQLAlchemy session
        city: City identifier
        day: Local calendar date

    Returns:
        DaySeries if both settlement and VC data exist, None otherwise

    Skip conditions:
        - No settlement data for this day
        - No VC temperature data for this day
        - Empty temperature series
    """
    # Load settlement (ground truth)
    settle_f = load_settlement(session, city, day)
    if settle_f is None:
        return None

    # Load VC temps
    temps_f = load_vc_temps_for_day(session, city, day)
    if not temps_f:
        return None

    return DaySeries(city=city, day=day, temps_f=temps_f, settle_f=settle_f)


def load_multiple_days(
    session: Session,
    city: str,
    start_day: date,
    end_day: date,
) -> List[DaySeries]:
    """Load DaySeries for a date range.

    Efficiently loads all days in range, skipping days with missing data.

    Args:
        session: SQLAlchemy session
        city: City identifier
        start_day: First day (inclusive)
        end_day: Last day (inclusive)

    Returns:
        List of DaySeries objects (may be less than total days if data missing)
    """
    series_list: List[DaySeries] = []

    current_day = start_day
    while current_day <= end_day:
        day_series = load_day_series(session, city, current_day)
        if day_series is not None:
            series_list.append(day_series)

        current_day += timedelta(days=1)

    return series_list
