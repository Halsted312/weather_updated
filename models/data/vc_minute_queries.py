"""
Query helpers for Visual Crossing minute-level forecast data.

This module provides DataFrame-based query functions for retrieving
15-minute historical forecast data from wx.vc_minute_weather.

Key use cases:
- Feature engineering: Get T-1 15-min forecast curves for a target date
- Station vs city comparison: Get both station and city forecasts
- Granularity detection: Infer step_minutes from timestamp spacing

Tables used:
    wx.vc_location: Location dimension (maps city_code to location_id)
    wx.vc_minute_weather: Minute-level data (obs + forecasts)

Design principles:
- Return pandas DataFrames, never raw SQL rows
- Handle missing data gracefully (return empty DataFrame, not error)
- Include robust type hints and documentation
- Follow existing loader.py patterns

Example:
    >>> from models.data.vc_minute_queries import fetch_tminus1_minute_forecast_df
    >>> from src.db.connection import get_db_session
    >>> from datetime import date
    >>>
    >>> with get_db_session() as session:
    ...     df = fetch_tminus1_minute_forecast_df(
    ...         session=session,
    ...         city_code='AUS',
    ...         target_date=date(2024, 11, 5),
    ...         location_type='city',
    ...     )
    ...     print(f"Got {len(df)} minute forecasts")
"""

import logging
from datetime import date, timedelta
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Type alias for location types
LocationType = Literal["station", "city"]


def get_vc_locations_for_city(
    session: Session,
    city_code: str,
) -> Dict[str, int]:
    """
    Get vc_location_ids for both station and city location types.

    Args:
        session: Database session
        city_code: 3-letter city code (e.g., 'AUS', 'CHI', 'DEN')

    Returns:
        Dict mapping location_type -> vc_location_id
        Example: {'station': 123, 'city': 124}

    Raises:
        ValueError: If city_code not found or required location types missing
    """
    query = text("""
        SELECT location_type, id
        FROM wx.vc_location
        WHERE city_code = :city_code
          AND location_type IN ('station', 'city')
        ORDER BY location_type
    """)

    result = session.execute(query, {"city_code": city_code}).fetchall()

    if not result:
        raise ValueError(
            f"No VcLocation rows found for city_code={city_code}. "
            "Ensure the city is configured in wx.vc_location."
        )

    loc_dict = {row[0]: row[1] for row in result}

    # Validate we have both types
    expected_types = {"station", "city"}
    missing = expected_types.difference(loc_dict.keys())
    if missing:
        raise ValueError(
            f"Missing VcLocation rows for city_code={city_code}, "
            f"missing types={missing}. "
            f"Found types={list(loc_dict.keys())}."
        )

    return loc_dict


def fetch_tminus1_minute_forecast_df(
    session: Session,
    city_code: str,
    target_date: date,
    location_type: LocationType,
    include_forward_filled: bool = True,
) -> pd.DataFrame:
    """
    Fetch T-1 15-minute historical forecast for a single target date.

    Queries wx.vc_minute_weather for:
    - data_type='historical_forecast'
    - forecast_basis_date = target_date - 1
    - DATE(datetime_local) = target_date

    Args:
        session: Database session
        city_code: 3-letter city code (e.g., 'AUS')
        target_date: Target date to get forecast for
        location_type: 'station' for station-locked, 'city' for city-aggregate
        include_forward_filled: If False, exclude forward-filled records

    Returns:
        DataFrame with columns:
            datetime_local, datetime_utc, datetime_epoch_utc,
            temp_f, humidity, dew_f, feelslike_f,
            pressure_mb, windspeed_mph, winddir, windgust_mph,
            precip_in, precipprob, cloudcover, uvindex,
            solarradiation, solarenergy, visibility_miles,
            is_forward_filled, lead_hours, forecast_basis_date

        Sorted by datetime_local ascending.
        Returns empty DataFrame if no data found.
    """
    # Calculate basis date (T-1)
    basis_date = target_date - timedelta(days=1)

    # Get location ID
    try:
        loc_dict = get_vc_locations_for_city(session, city_code)
        vc_location_id = loc_dict[location_type]
    except (ValueError, KeyError) as e:
        logger.warning(f"Location lookup failed: {e}. Returning empty DataFrame.")
        return pd.DataFrame()

    # Build query
    query = text("""
        SELECT
            datetime_local,
            datetime_utc,
            datetime_epoch_utc,
            temp_f,
            humidity,
            dew_f,
            feelslike_f,
            pressure_mb,
            windspeed_mph,
            winddir,
            windgust_mph,
            precip_in,
            precipprob,
            cloudcover,
            uvindex,
            solarradiation,
            solarenergy,
            visibility_miles,
            is_forward_filled,
            lead_hours,
            forecast_basis_date
        FROM wx.vc_minute_weather
        WHERE vc_location_id = :vc_location_id
          AND data_type = 'historical_forecast'
          AND forecast_basis_date = :basis_date
          AND DATE(datetime_local) = :target_date
        ORDER BY datetime_local ASC
    """)

    params = {
        "vc_location_id": vc_location_id,
        "basis_date": basis_date,
        "target_date": target_date,
    }

    # Execute and load into DataFrame
    result = session.execute(query, params)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

    # Filter out forward-filled if requested
    if not include_forward_filled and "is_forward_filled" in df.columns and not df.empty:
        df = df[(df["is_forward_filled"] == False) | (df["is_forward_filled"].isna())]

    # Ensure sorted by time
    if not df.empty:
        df = df.sort_values("datetime_local").reset_index(drop=True)

    logger.debug(
        f"Fetched {len(df)} minute forecasts for {city_code}/{location_type}, "
        f"target={target_date}, basis={basis_date}"
    )

    return df


def fetch_station_and_city_tminus1_minute_forecasts(
    session: Session,
    city_code: str,
    target_date: date,
    include_forward_filled: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch T-1 15-minute forecasts for both station and city locations.

    Convenience wrapper that calls fetch_tminus1_minute_forecast_df twice.

    Args:
        session: Database session
        city_code: 3-letter city code (e.g., 'AUS')
        target_date: Target date to get forecasts for
        include_forward_filled: If False, exclude forward-filled records

    Returns:
        Tuple of (station_df, city_df) where each DataFrame has the same
        structure as fetch_tminus1_minute_forecast_df.

        If either location is missing, returns empty DataFrame for that location.

    Example:
        >>> station_df, city_df = fetch_station_and_city_tminus1_minute_forecasts(
        ...     session, 'AUS', date(2024, 11, 5)
        ... )
        >>> print(f"Station: {len(station_df)} rows, City: {len(city_df)} rows")
    """
    station_df = fetch_tminus1_minute_forecast_df(
        session=session,
        city_code=city_code,
        target_date=target_date,
        location_type="station",
        include_forward_filled=include_forward_filled,
    )

    city_df = fetch_tminus1_minute_forecast_df(
        session=session,
        city_code=city_code,
        target_date=target_date,
        location_type="city",
        include_forward_filled=include_forward_filled,
    )

    return station_df, city_df


def infer_step_minutes(df: pd.DataFrame, datetime_col: str = "datetime_local") -> int:
    """
    Infer the step size (in minutes) from a datetime column.

    Computes the median time difference between consecutive rows.
    Useful for detecting whether data is 5-min, 15-min, hourly, etc.

    Args:
        df: DataFrame with datetime column
        datetime_col: Name of datetime column (default: 'datetime_local')

    Returns:
        Step size in minutes (int), or 15 as default if cannot infer

    Examples:
        >>> df = fetch_tminus1_minute_forecast_df(...)
        >>> step = infer_step_minutes(df)
        >>> print(f"Data granularity: {step} minutes")
        # Output: Data granularity: 15 minutes
    """
    if df.empty or datetime_col not in df.columns or len(df) < 2:
        logger.debug("Cannot infer step_minutes: insufficient data. Defaulting to 15.")
        return 15

    # Ensure datetime column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        try:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        except Exception as e:
            logger.warning(f"Could not convert {datetime_col} to datetime: {e}")
            return 15

    # Calculate differences in minutes
    diffs = df[datetime_col].diff().dropna().dt.total_seconds() / 60.0

    # Use median for robustness (handles occasional gaps)
    median_diff = diffs.median()

    if pd.isna(median_diff) or median_diff <= 0:
        logger.debug("Invalid median diff. Defaulting to 15.")
        return 15

    # Round to nearest integer
    step_minutes = int(round(median_diff))

    logger.debug(f"Inferred step_minutes={step_minutes} from {len(df)} rows")

    return step_minutes


def fetch_obs_minute_df(
    session: Session,
    city_code: str,
    target_date: date,
    location_type: LocationType,
    include_forward_filled: bool = True,
) -> pd.DataFrame:
    """
    Fetch minute-level observations for a single date.

    Queries wx.vc_minute_weather for:
    - data_type='actual_obs'
    - DATE(datetime_local) = target_date

    Args:
        session: Database session
        city_code: 3-letter city code (e.g., 'AUS')
        target_date: Date to get observations for
        location_type: 'station' for station-locked, 'city' for city-aggregate
        include_forward_filled: If False, exclude forward-filled records

    Returns:
        DataFrame with same structure as fetch_tminus1_minute_forecast_df
        (but without forecast_basis_date, lead_hours which will be NULL).

        Sorted by datetime_local ascending.
        Returns empty DataFrame if no data found.
    """
    # Get location ID
    try:
        loc_dict = get_vc_locations_for_city(session, city_code)
        vc_location_id = loc_dict[location_type]
    except (ValueError, KeyError) as e:
        logger.warning(f"Location lookup failed: {e}. Returning empty DataFrame.")
        return pd.DataFrame()

    # Build query for observations
    query = text("""
        SELECT
            datetime_local,
            datetime_utc,
            datetime_epoch_utc,
            temp_f,
            humidity,
            dew_f,
            feelslike_f,
            pressure_mb,
            windspeed_mph,
            winddir,
            windgust_mph,
            precip_in,
            precipprob,
            cloudcover,
            uvindex,
            solarradiation,
            solarenergy,
            visibility_miles,
            is_forward_filled
        FROM wx.vc_minute_weather
        WHERE vc_location_id = :vc_location_id
          AND data_type = 'actual_obs'
          AND DATE(datetime_local) = :target_date
        ORDER BY datetime_local ASC
    """)

    params = {
        "vc_location_id": vc_location_id,
        "target_date": target_date,
    }

    # Execute and load into DataFrame
    result = session.execute(query, params)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

    # Filter out forward-filled if requested
    if not include_forward_filled and "is_forward_filled" in df.columns and not df.empty:
        df = df[(df["is_forward_filled"] == False) | (df["is_forward_filled"].isna())]

    # Ensure sorted by time
    if not df.empty:
        df = df.sort_values("datetime_local").reset_index(drop=True)

    logger.debug(
        f"Fetched {len(df)} minute observations for {city_code}/{location_type}, "
        f"date={target_date}"
    )

    return df


def fetch_station_and_city_obs_minutes(
    session: Session,
    city_code: str,
    target_date: date,
    include_forward_filled: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch minute-level observations for both station and city locations.

    Convenience wrapper that calls fetch_obs_minute_df twice.

    Args:
        session: Database session
        city_code: 3-letter city code (e.g., 'AUS')
        target_date: Date to get observations for
        include_forward_filled: If False, exclude forward-filled records

    Returns:
        Tuple of (station_df, city_df) where each DataFrame has the same
        structure as fetch_obs_minute_df.

    Example:
        >>> station_df, city_df = fetch_station_and_city_obs_minutes(
        ...     session, 'AUS', date(2024, 11, 5)
        ... )
        >>> print(f"Station: {len(station_df)} obs, City: {len(city_df)} obs")
    """
    station_df = fetch_obs_minute_df(
        session=session,
        city_code=city_code,
        target_date=target_date,
        location_type="station",
        include_forward_filled=include_forward_filled,
    )

    city_df = fetch_obs_minute_df(
        session=session,
        city_code=city_code,
        target_date=target_date,
        location_type="city",
        include_forward_filled=include_forward_filled,
    )

    return station_df, city_df


def validate_minute_forecast_availability(
    session: Session,
    city_code: str,
    target_date: date,
    location_type: LocationType = "city",
) -> Dict[str, any]:
    """
    Check if T-1 15-minute forecast data is available for a date.

    Useful for feature engineering pipelines to determine if 15-min
    forecast features can be computed before attempting expensive operations.

    Args:
        session: Database session
        city_code: 3-letter city code
        target_date: Target date to check
        location_type: Which location to check

    Returns:
        Dict with:
            - available (bool): True if data exists
            - row_count (int): Number of minute records found
            - basis_date (date or None): The forecast basis date
            - expected_count (int): Expected rows (~96 for full day)
            - completeness (float): row_count / expected_count

    Example:
        >>> info = validate_minute_forecast_availability(session, 'AUS', date(2024,11,5))
        >>> if info['completeness'] > 0.9:
        ...     print("Good data coverage")
    """
    basis_date = target_date - timedelta(days=1)

    try:
        loc_dict = get_vc_locations_for_city(session, city_code)
        vc_location_id = loc_dict[location_type]
    except (ValueError, KeyError):
        return {
            "available": False,
            "row_count": 0,
            "basis_date": None,
            "expected_count": 96,
            "completeness": 0.0,
        }

    query = text("""
        SELECT COUNT(*) as n
        FROM wx.vc_minute_weather
        WHERE vc_location_id = :vc_location_id
          AND data_type = 'historical_forecast'
          AND forecast_basis_date = :basis_date
          AND DATE(datetime_local) = :target_date
    """)

    result = session.execute(query, {
        "vc_location_id": vc_location_id,
        "basis_date": basis_date,
        "target_date": target_date,
    }).fetchone()

    row_count = result[0] if result else 0
    expected = 96  # 24 hours × 4 per hour (15-min intervals)

    return {
        "available": row_count > 0,
        "row_count": row_count,
        "basis_date": basis_date,
        "expected_count": expected,
        "completeness": row_count / expected if expected > 0 else 0.0,
    }
