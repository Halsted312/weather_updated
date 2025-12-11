"""
Data loading from database for temperature Δ-models.

This module provides a unified interface for loading:
- Historical data (for training): VC obs + settlements + forecasts
- Current data (for inference): VC obs up to now + yesterday's forecast

Both use the same feature functions - only the data source differs.
The key principle is that feature code is pure and doesn't know about DB.

Tables used:
    wx.vc_location: Location dimension (maps city_code to vc_location_id)
    wx.vc_minute_weather: 5-minute observations (data_type='actual_obs')
    wx.settlement: Ground truth daily high (tmax_final)
    wx.vc_forecast_daily: T-1 daily forecasts (for forecast features)
    wx.vc_forecast_hourly: T-1 hourly forecasts (for forecast features)

Example:
    >>> from models.data.loader import load_training_data
    >>> from src.db.connection import get_db_session
    >>> with get_db_session() as session:
    ...     df = load_training_data(['chicago'], date(2024,1,1), date(2024,12,31), session)
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional, Dict, Any
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config.cities import get_city, CityConfig
from models.features.interpolation import interpolate_cloudcover_from_hourly

logger = logging.getLogger(__name__)


def get_vc_location_id(
    session: Session,
    city_id: str,
    location_type: str = "station",
) -> Optional[int]:
    """Get the vc_location_id for a city.

    Args:
        session: Database session
        city_id: City identifier (e.g., 'chicago')
        location_type: 'station' for KMDW-style, 'city' for Chicago,IL-style

    Returns:
        vc_location_id or None if not found
    """
    city_config = get_city(city_id)

    query = text("""
        SELECT id FROM wx.vc_location
        WHERE city_code = :city_code AND location_type = :location_type
        LIMIT 1
    """)

    result = session.execute(query, {
        "city_code": city_config.city_code,
        "location_type": location_type,
    }).fetchone()

    if result:
        return result[0]
    return None


def load_vc_observations(
    session: Session,
    city_id: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Load Visual Crossing 5-minute observations for a city and date range.

    Only loads actual observations (data_type='actual_obs'), not forecasts.

    Args:
        session: Database session
        city_id: City identifier (e.g., 'chicago')
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        DataFrame with columns:
            datetime_local, datetime_utc, temp_f, humidity, windspeed_mph,
            cloudcover, conditions, etc.
        Sorted by datetime_local ascending.
    """
    city_config = get_city(city_id)
    vc_location_id = get_vc_location_id(session, city_id, "station")

    if vc_location_id is None:
        logger.warning(f"No vc_location found for {city_id}, returning empty DataFrame")
        return pd.DataFrame()

    query = text("""
        SELECT
            datetime_local,
            datetime_utc,
            datetime_epoch_utc,
            temp_f,
            humidity,
            dew_f,
            windspeed_mph,
            windgust_mph,
            winddir,
            cloudcover,
            pressure_mb,
            visibility_miles,
            conditions,
            solarradiation
        FROM wx.vc_minute_weather
        WHERE vc_location_id = :vc_location_id
          AND data_type = 'actual_obs'
          AND datetime_local >= :start_date
          AND datetime_local < :end_date_plus_one
        ORDER BY datetime_local
    """)

    result = session.execute(query, {
        "vc_location_id": vc_location_id,
        "start_date": start_date,
        "end_date_plus_one": end_date + timedelta(days=1),
    })

    df = pd.DataFrame(result.fetchall(), columns=[
        "datetime_local", "datetime_utc", "datetime_epoch_utc",
        "temp_f", "humidity", "dew_f", "windspeed_mph", "windgust_mph",
        "winddir", "cloudcover", "pressure_mb", "visibility_miles",
        "conditions", "solarradiation"
    ])

    logger.info(f"Loaded {len(df)} VC observations for {city_id} from {start_date} to {end_date}")
    return df


def load_settlements(
    session: Session,
    city_id: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Load settlement (ground truth) data for a city and date range.

    Args:
        session: Database session
        city_id: City identifier (e.g., 'chicago')
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        DataFrame with columns:
            date_local, tmax_final, source_final
        One row per day.
    """
    query = text("""
        SELECT
            date_local,
            tmax_final,
            source_final,
            tmax_cli_f,
            tmax_iem_f,
            tmax_ncei_f
        FROM wx.settlement
        WHERE city = :city_id
          AND date_local >= :start_date
          AND date_local <= :end_date
        ORDER BY date_local
    """)

    result = session.execute(query, {
        "city_id": city_id,
        "start_date": start_date,
        "end_date": end_date,
    })

    df = pd.DataFrame(result.fetchall(), columns=[
        "date_local", "tmax_final", "source_final",
        "tmax_cli_f", "tmax_iem_f", "tmax_ncei_f"
    ])

    logger.info(f"Loaded {len(df)} settlement records for {city_id} from {start_date} to {end_date}")
    return df


def load_historical_forecast_daily(
    session: Session,
    city_id: str,
    target_date: date,
    basis_date: date,
) -> Optional[dict]:
    """Load T-1 daily forecast for a specific target date.

    Used to get "what did VC think yesterday about today's high?"

    Note: Forecasts are stored under 'city' locations (e.g., "Chicago,IL"),
    not 'station' locations (e.g., "KMDW").

    Args:
        session: Database session
        city_id: City identifier
        target_date: The day we're predicting
        basis_date: When the forecast was issued (typically target_date - 1)

    Returns:
        Dict with forecast fields or None if not found:
            tempmax_f, tempmin_f, temp_f, humidity, precip_in, etc.
    """
    # Forecasts are on city locations, not station locations
    vc_location_id = get_vc_location_id(session, city_id, "city")

    if vc_location_id is None:
        return None

    # Accept both 'forecast' (live/recent) and 'historical_forecast' (backfill)
    # Prefer 'historical_forecast' for backtesting consistency, but fall back to 'forecast'
    query = text("""
        SELECT
            tempmax_f, tempmin_f, temp_f,
            humidity, precip_in, precipprob,
            windspeed_mph, windgust_mph,
            cloudcover, conditions
        FROM wx.vc_forecast_daily
        WHERE vc_location_id = :vc_location_id
          AND target_date = :target_date
          AND forecast_basis_date = :basis_date
          AND data_type IN ('historical_forecast', 'forecast')
        ORDER BY CASE WHEN data_type = 'historical_forecast' THEN 0 ELSE 1 END
        LIMIT 1
    """)

    result = session.execute(query, {
        "vc_location_id": vc_location_id,
        "target_date": target_date,
        "basis_date": basis_date,
    }).fetchone()

    if result:
        return {
            "tempmax_f": result[0],
            "tempmin_f": result[1],
            "temp_f": result[2],
            "humidity": result[3],
            "precip_in": result[4],
            "precipprob": result[5],
            "windspeed_mph": result[6],
            "windgust_mph": result[7],
            "cloudcover": result[8],
            "conditions": result[9],
        }
    return None


def load_historical_forecast_hourly(
    session: Session,
    city_id: str,
    target_date: date,
    basis_date: date,
) -> pd.DataFrame:
    """Load T-1 hourly forecast curve for a specific target date.

    Returns the full hourly forecast series that was issued on basis_date
    for target_date. Used to compute forecast features.

    Note: Forecasts are stored under 'city' locations (e.g., "Chicago,IL"),
    not 'station' locations (e.g., "KMDW").

    Args:
        session: Database session
        city_id: City identifier
        target_date: The day we're predicting
        basis_date: When the forecast was issued

    Returns:
        DataFrame with hourly forecasts for target_date:
            target_datetime_local, temp_f, humidity, etc.
    """
    # Forecasts are on city locations, not station locations
    vc_location_id = get_vc_location_id(session, city_id, "city")

    if vc_location_id is None:
        return pd.DataFrame()

    # Accept both 'forecast' (live/recent) and 'historical_forecast' (backfill)
    query = text("""
        SELECT
            target_datetime_local,
            target_datetime_utc,
            lead_hours,
            temp_f,
            humidity,
            dew_f,
            precip_in,
            precipprob,
            windspeed_mph,
            cloudcover,
            conditions
        FROM wx.vc_forecast_hourly
        WHERE vc_location_id = :vc_location_id
          AND forecast_basis_date = :basis_date
          AND DATE(target_datetime_local) = :target_date
          AND data_type IN ('historical_forecast', 'forecast')
        ORDER BY target_datetime_local
    """)

    result = session.execute(query, {
        "vc_location_id": vc_location_id,
        "target_date": target_date,
        "basis_date": basis_date,
    })

    df = pd.DataFrame(result.fetchall(), columns=[
        "target_datetime_local", "target_datetime_utc", "lead_hours",
        "temp_f", "humidity", "dew_f", "precip_in", "precipprob",
        "windspeed_mph", "cloudcover", "conditions"
    ])

    return df


def load_historical_forecast_daily_multi(
    session: Session,
    city_id: str,
    target_date: date,
    max_lead_days: int = 6,
) -> pd.DataFrame:
    """Load multi-lead daily forecasts for a specific target date.

    Returns forecasts from multiple lead times (T-0, T-1, T-2, ..., T-6)
    for the same target date. Used for computing forecast drift features.

    Args:
        session: Database session
        city_id: City identifier
        target_date: The day we're predicting
        max_lead_days: Maximum lead days to include (default 6)

    Returns:
        DataFrame with columns: lead_days, tempmax_f, tempmin_f, humidity, cloudcover
        One row per lead_day that exists in the database.
    """
    # Forecasts are on city locations
    vc_location_id = get_vc_location_id(session, city_id, "city")

    if vc_location_id is None:
        return pd.DataFrame()

    query = text("""
        SELECT
            lead_days,
            tempmax_f,
            tempmin_f,
            humidity,
            cloudcover
        FROM wx.vc_forecast_daily
        WHERE vc_location_id = :vc_location_id
          AND data_type = 'historical_forecast'
          AND target_date = :target_date
          AND lead_days <= :max_lead
        ORDER BY lead_days ASC
    """)

    result = session.execute(query, {
        "vc_location_id": vc_location_id,
        "target_date": target_date,
        "max_lead": max_lead_days,
    })

    df = pd.DataFrame(result.fetchall(), columns=[
        "lead_days", "tempmax_f", "tempmin_f", "humidity", "cloudcover"
    ])

    return df


def load_historical_forecast_15min(
    session: Session,
    city_id: str,
    target_date: date,
    basis_date: date,
    location_type: str = "station",
) -> pd.DataFrame:
    """Load T-1 15-min forecast series for a day.

    Returns minute-level forecast data (15-min resolution) from Visual Crossing.
    Used for computing detailed multivar features from forecast curves.

    Note: Cloudcover is NOT available in minute-level API. Use hourly
    data for cloudcover features.

    Args:
        session: Database session
        city_id: City identifier
        target_date: The day we're predicting
        basis_date: When the forecast was issued (typically target_date - 1)
        location_type: 'station' or 'city'

    Returns:
        DataFrame with columns: datetime_local, temp_f, humidity, dew_f, cloudcover
    """
    vc_location_id = get_vc_location_id(session, city_id, location_type)

    if vc_location_id is None:
        return pd.DataFrame()

    query = text("""
        SELECT
            datetime_local,
            temp_f,
            humidity,
            dew_f,
            cloudcover
        FROM wx.vc_minute_weather
        WHERE vc_location_id = :vc_location_id
          AND data_type = 'historical_forecast'
          AND forecast_basis_date = :basis_date
          AND DATE(datetime_local) = :target_date
        ORDER BY datetime_local
    """)

    result = session.execute(query, {
        "vc_location_id": vc_location_id,
        "basis_date": basis_date,
        "target_date": target_date,
    })

    df = pd.DataFrame(result.fetchall(), columns=[
        "datetime_local", "temp_f", "humidity", "dew_f", "cloudcover"
    ])

    return df


def load_multi_horizon_forecasts(
    session: Session,
    city_id: str,
    target_date: date,
    lead_days: list[int] = None,
) -> dict[int, Optional[dict]]:
    """Load forecasts from multiple lead times for a single target date.

    Returns a dictionary mapping lead_day → forecast_dict, where each
    forecast_dict contains 'tempmax_f', 'tempmin_f', etc.

    Used by the unified pipeline to compute multi-horizon features that
    capture forecast drift/evolution.

    Args:
        session: Database session
        city_id: City identifier
        target_date: The day we're predicting
        lead_days: List of lead days to fetch (default [1,2,3,4,5,6])

    Returns:
        Dict mapping lead_day → forecast_dict or None if not available
    """
    if lead_days is None:
        lead_days = [1, 2, 3, 4, 5, 6]

    result = {}
    for lead in lead_days:
        basis_date = target_date - timedelta(days=lead)
        fcst = load_historical_forecast_daily(session, city_id, target_date, basis_date)
        result[lead] = fcst

    return result


def batch_load_multi_horizon_forecasts(
    session: Session,
    city_id: str,
    start_date: date,
    end_date: date,
    lead_days: list[int] = None,
) -> dict[tuple[date, int], Optional[dict]]:
    """Load multi-horizon forecasts for ALL target dates at once (batch mode).

    Instead of making 14 queries per day, this makes 14 queries total
    for the entire date range, dramatically reducing database overhead.

    Args:
        session: Database session
        city_id: City identifier
        start_date: First target date
        end_date: Last target date
        lead_days: List of lead days to fetch (default [1,2,3,4,5,6])

    Returns:
        Dict mapping (target_date, lead_day) → forecast_dict or None
        Example: {(date(2025,4,1), 1): {'tempmax_f': 85.0, ...}, ...}
    """
    if lead_days is None:
        lead_days = [1, 2, 3, 4, 5, 6]

    # Get location ID
    vc_location_id = get_vc_location_id(session, city_id, "city")
    if vc_location_id is None:
        logger.warning(f"No vc_location found for {city_id}")
        return {}

    all_forecasts = {}

    # Query once per lead_day for ALL target dates
    for lead in lead_days:
        query = text("""
            SELECT
                target_date,
                tempmax_f, tempmin_f, temp_f,
                humidity, precip_in, precipprob,
                windspeed_mph, windgust_mph,
                cloudcover, conditions
            FROM wx.vc_forecast_daily
            WHERE vc_location_id = :vc_location_id
              AND target_date BETWEEN :start_date AND :end_date
              AND lead_days = :lead_day
              AND data_type = 'historical_forecast'
            ORDER BY target_date
        """)

        rows = session.execute(query, {
            "vc_location_id": vc_location_id,
            "start_date": start_date,
            "end_date": end_date,
            "lead_day": lead,
        }).fetchall()

        # Store each forecast by (target_date, lead_day) key
        for row in rows:
            target_date = row.target_date
            all_forecasts[(target_date, lead)] = {
                "tempmax_f": float(row.tempmax_f) if row.tempmax_f is not None else None,
                "tempmin_f": float(row.tempmin_f) if row.tempmin_f is not None else None,
                "temp_f": float(row.temp_f) if row.temp_f is not None else None,
                "humidity": float(row.humidity) if row.humidity is not None else None,
                "precip_in": float(row.precip_in) if row.precip_in is not None else None,
                "precipprob": float(row.precipprob) if row.precipprob is not None else None,
                "windspeed_mph": float(row.windspeed_mph) if row.windspeed_mph is not None else None,
                "windgust_mph": float(row.windgust_mph) if row.windgust_mph is not None else None,
                "cloudcover": float(row.cloudcover) if row.cloudcover is not None else None,
                "conditions": row.conditions,
            }

    logger.info(f"Batch-loaded {len(all_forecasts)} multi-horizon forecasts for {city_id}")
    return all_forecasts


def load_training_data(
    cities: list[str],
    start_date: date,
    end_date: date,
    session: Session,
) -> pd.DataFrame:
    """Load all data needed for training snapshot dataset.

    Combines VC observations and settlements for the specified cities
    and date range. Returns a DataFrame with one row per (city, day)
    containing the raw data needed for feature engineering.

    Args:
        cities: List of city identifiers (e.g., ['chicago'])
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        session: Database session

    Returns:
        DataFrame with columns:
            city, day, settle_f, obs_df (nested DataFrame of 5-min temps)
    """
    all_data = []

    for city_id in cities:
        logger.info(f"Loading training data for {city_id}...")

        # Load settlements
        settlements_df = load_settlements(session, city_id, start_date, end_date)
        if settlements_df.empty:
            logger.warning(f"No settlements found for {city_id}")
            continue

        # Load observations
        obs_df = load_vc_observations(session, city_id, start_date, end_date)
        if obs_df.empty:
            logger.warning(f"No observations found for {city_id}")
            continue

        # Extract date from datetime_local
        obs_df["day"] = pd.to_datetime(obs_df["datetime_local"]).dt.date

        # Group observations by day
        for _, settle_row in settlements_df.iterrows():
            day = settle_row["date_local"]
            settle_f = settle_row["tmax_final"]

            # Get observations for this day
            day_obs = obs_df[obs_df["day"] == day].copy()

            if day_obs.empty:
                logger.debug(f"No observations for {city_id} on {day}")
                continue

            # Get T-1 forecast if available
            basis_date = day - timedelta(days=1)
            fcst_daily = load_historical_forecast_daily(session, city_id, day, basis_date)
            fcst_hourly = load_historical_forecast_hourly(session, city_id, day, basis_date)

            # Interpolate hourly cloudcover to 5-minute obs timestamps
            if fcst_hourly is not None and not fcst_hourly.empty:
                hourly_for_interp = fcst_hourly.rename(
                    columns={"target_datetime_local": "datetime_local"}
                )
                day_obs = interpolate_cloudcover_from_hourly(day_obs, hourly_for_interp)

            all_data.append({
                "city": city_id,
                "day": day,
                "settle_f": int(settle_f),
                "obs_df": day_obs,
                "fcst_daily": fcst_daily,
                "fcst_hourly_df": fcst_hourly if not fcst_hourly.empty else None,
            })

    logger.info(f"Loaded {len(all_data)} city-days for training")
    return pd.DataFrame(all_data)


def load_inference_data(
    city_id: str,
    target_date: date,
    cutoff_time: datetime,
    session: Session,
) -> dict:
    """Load current data for live inference.

    .. deprecated::
        Use :func:`load_full_inference_data` instead for full feature parity
        with the training pipeline. This function only loads temp_f and T-1
        forecast, missing 12 observation columns and 6 other data sources.

    Loads observations up to cutoff_time and yesterday's forecast
    for real-time prediction.

    Args:
        city_id: City identifier
        target_date: The day we're predicting
        cutoff_time: Local datetime - only use obs before this time
        session: Database session

    Returns:
        Dict with:
            temps_sofar: List of temps up to cutoff_time
            timestamps_sofar: List of timestamps
            fcst_daily: T-1 daily forecast dict
            fcst_hourly_df: T-1 hourly forecast DataFrame
    """
    import warnings
    warnings.warn(
        "load_inference_data() is deprecated. Use load_full_inference_data() "
        "for full feature parity with training pipeline.",
        DeprecationWarning,
        stacklevel=2,
    )
    city_config = get_city(city_id)
    vc_location_id = get_vc_location_id(session, city_id, "station")

    if vc_location_id is None:
        raise ValueError(f"No vc_location found for {city_id}")

    # Load observations up to cutoff
    query = text("""
        SELECT datetime_local, temp_f
        FROM wx.vc_minute_weather
        WHERE vc_location_id = :vc_location_id
          AND data_type = 'actual_obs'
          AND DATE(datetime_local) = :target_date
          AND datetime_local < :cutoff_time
        ORDER BY datetime_local
    """)

    result = session.execute(query, {
        "vc_location_id": vc_location_id,
        "target_date": target_date,
        "cutoff_time": cutoff_time,
    })

    rows = result.fetchall()
    timestamps_sofar = [row[0] for row in rows]
    temps_sofar = [row[1] for row in rows if row[1] is not None]

    # Load T-1 forecast
    basis_date = target_date - timedelta(days=1)
    fcst_daily = load_historical_forecast_daily(session, city_id, target_date, basis_date)
    fcst_hourly_df = load_historical_forecast_hourly(session, city_id, target_date, basis_date)

    return {
        "city": city_id,
        "target_date": target_date,
        "cutoff_time": cutoff_time,
        "temps_sofar": temps_sofar,
        "timestamps_sofar": timestamps_sofar,
        "fcst_daily": fcst_daily,
        "fcst_hourly_df": fcst_hourly_df if not fcst_hourly_df.empty else None,
    }


def get_available_date_range(
    session: Session,
    city_id: str,
) -> tuple[Optional[date], Optional[date]]:
    """Get the date range with both observations and settlements.

    Useful for determining valid training date ranges.

    Args:
        session: Database session
        city_id: City identifier

    Returns:
        Tuple of (min_date, max_date) or (None, None) if no data
    """
    vc_location_id = get_vc_location_id(session, city_id, "station")

    if vc_location_id is None:
        return None, None

    # Get observation date range
    obs_query = text("""
        SELECT MIN(DATE(datetime_local)), MAX(DATE(datetime_local))
        FROM wx.vc_minute_weather
        WHERE vc_location_id = :vc_location_id
          AND data_type = 'actual_obs'
    """)
    obs_result = session.execute(obs_query, {"vc_location_id": vc_location_id}).fetchone()

    # Get settlement date range
    settle_query = text("""
        SELECT MIN(date_local), MAX(date_local)
        FROM wx.settlement
        WHERE city = :city_id
    """)
    settle_result = session.execute(settle_query, {"city_id": city_id}).fetchone()

    if not obs_result or not settle_result:
        return None, None

    obs_min, obs_max = obs_result
    settle_min, settle_max = settle_result

    if not all([obs_min, obs_max, settle_min, settle_max]):
        return None, None

    # Return intersection
    min_date = max(obs_min, settle_min)
    max_date = min(obs_max, settle_max)

    return min_date, max_date


def load_weather_more_apis_guidance(
    session: Session,
    city_id: str,
    target_date: date,
    snapshot_time_utc: Optional[datetime] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load NOAA model guidance (NBM, HRRR, NDFD) for snapshot context.

    Returns: {'nbm': {'latest_run': {...}, 'prev_run': {...}}, ...}

    Args:
        session: Database session
        city_id: City ID
        target_date: Target date (local)
        snapshot_time_utc: Cutoff (None = all runs)

    Returns:
        Nested dict with latest/prev run per model
    """
    from src.db.models import WeatherMoreApisGuidance

    result = {
        "nbm": {"latest_run": None, "prev_run": None},
        "hrrr": {"latest_run": None, "prev_run": None},
        "ndfd": {"latest_run": None, "prev_run": None},
    }

    for model in ["nbm", "hrrr", "ndfd"]:
        query = session.query(WeatherMoreApisGuidance).filter(
            WeatherMoreApisGuidance.city_id == city_id,
            WeatherMoreApisGuidance.target_date == target_date,
            WeatherMoreApisGuidance.model == model,
        )

        if snapshot_time_utc:
            query = query.filter(WeatherMoreApisGuidance.run_datetime_utc <= snapshot_time_utc)

        runs = query.order_by(WeatherMoreApisGuidance.run_datetime_utc.desc()).limit(2).all()

        if runs:
            result[model]["latest_run"] = {
                "run_datetime_utc": runs[0].run_datetime_utc,
                "peak_window_max_f": float(runs[0].peak_window_max_f) if runs[0].peak_window_max_f else None,
            }
        if len(runs) >= 2:
            result[model]["prev_run"] = {
                "run_datetime_utc": runs[1].run_datetime_utc,
                "peak_window_max_f": float(runs[1].peak_window_max_f) if runs[1].peak_window_max_f else None,
            }

    return result


def load_obs_t15_stats_30d(
    session: Session,
    city_id: str,
    target_date: date,
    lookback_days: int = 30,
) -> tuple[Optional[float], Optional[float]]:
    """
    Compute rolling mean/std of observed temp at 15:00 local over previous N days.

    For z-score normalization of NOAA guidance features.

    Args:
        session: Database session
        city_id: City identifier
        target_date: Target date (stats computed for days BEFORE this)
        lookback_days: Lookback window (default: 30)

    Returns:
        (mean_f, std_f) or (None, None) if insufficient data
    """
    start_date = target_date - timedelta(days=lookback_days)
    end_date = target_date - timedelta(days=1)

    obs_df = load_vc_observations(session, city_id, start_date, end_date)

    if obs_df is None or obs_df.empty:
        return (None, None)

    # Filter to 15:00 local (±15 min window)
    obs_df['hour'] = obs_df['datetime_local'].dt.hour
    obs_df['minute'] = obs_df['datetime_local'].dt.minute

    t15_obs = obs_df[(obs_df['hour'] == 15) & (obs_df['minute'] <= 15)].copy()

    if len(t15_obs) < 10:
        return (None, None)

    # One temp per day (first obs near 15:00)
    daily_t15 = t15_obs.groupby(t15_obs['datetime_local'].dt.date)['temp_f'].first()

    if len(daily_t15) < 10:
        return (None, None)

    return (float(daily_t15.mean()), float(daily_t15.std(ddof=0)))


# =============================================================================
# Full Inference Data Loading (Unified Pipeline)
# =============================================================================


def _to_naive(dt: datetime) -> datetime:
    """Convert datetime to naive (strip tzinfo) for DB comparison.

    DB stores datetime_local as naive TIMESTAMP (per DATETIME_AND_API_REFERENCE.md).
    This ensures consistent comparison regardless of whether input is tz-aware.
    """
    if dt is None:
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def load_candles_for_inference(
    session: Session,
    city_id: str,
    event_date: date,
    window_start: datetime,
    cutoff_time: datetime,
) -> Optional[pd.DataFrame]:
    """Load market candles for inference window.

    Replicates logic from dataset._load_candles() for inference use.
    This is a public function to avoid brittle imports of private functions.

    Args:
        session: Database session
        city_id: City identifier
        event_date: Event date (not used directly, kept for API consistency)
        window_start: Start of observation window
        cutoff_time: End of observation window

    Returns:
        DataFrame with candle data, or None if empty
    """
    # Normalize datetimes to naive for DB comparison
    window_start_naive = _to_naive(window_start)
    cutoff_naive = _to_naive(cutoff_time)

    # Use centralized city config instead of hardcoded mapping
    city_config = get_city(city_id)
    ticker_pattern = city_config.city_code

    query = text("""
        SELECT bucket_start, ticker, yes_bid_close, yes_ask_close,
               volume, open_interest, has_trade, is_synthetic
        FROM kalshi.candles_1m_dense
        WHERE ticker LIKE :ticker_pattern
          AND bucket_start >= :window_start
          AND bucket_start <= :cutoff_time
        ORDER BY bucket_start
    """)

    try:
        result = session.execute(query, {
            "ticker_pattern": f"%{ticker_pattern}%",
            "window_start": window_start_naive,
            "cutoff_time": cutoff_naive,
        })
        df = pd.DataFrame(result.fetchall(), columns=[
            "bucket_start", "ticker", "yes_bid_close", "yes_ask_close",
            "volume", "open_interest", "has_trade", "is_synthetic"
        ])
        return df if not df.empty else None
    except Exception as e:
        logger.debug(f"Could not load candles for {city_id}: {e}")
        return None


def load_city_observations_for_inference(
    session: Session,
    city_id: str,
    window_start: datetime,
    cutoff_time: datetime,
) -> Optional[pd.DataFrame]:
    """Load city-aggregate observations for inference window.

    Replicates logic from dataset._load_city_observations() for inference use.
    This is a public function to avoid brittle imports of private functions.

    Args:
        session: Database session
        city_id: City identifier
        window_start: Start of observation window (D-1 10:00)
        cutoff_time: End of observation window

    Returns:
        DataFrame with city observation data, or None if empty
    """
    # Normalize datetimes to naive for DB comparison
    window_start_naive = _to_naive(window_start)
    cutoff_naive = _to_naive(cutoff_time)

    vc_location_id = get_vc_location_id(session, city_id, "city")
    if vc_location_id is None:
        logger.debug(f"No city location found for {city_id}")
        return None

    query = text("""
        SELECT datetime_local, datetime_utc, temp_f, humidity,
               windspeed_mph, cloudcover
        FROM wx.vc_minute_weather
        WHERE vc_location_id = :vc_location_id
          AND data_type = 'actual_obs'
          AND datetime_local >= :window_start
          AND datetime_local < :cutoff_time
        ORDER BY datetime_local
    """)

    result = session.execute(query, {
        "vc_location_id": vc_location_id,
        "window_start": window_start_naive,
        "cutoff_time": cutoff_naive,
    })

    df = pd.DataFrame(result.fetchall(), columns=[
        "datetime_local", "datetime_utc", "temp_f", "humidity",
        "windspeed_mph", "cloudcover"
    ])
    return df if not df.empty else None


def load_historical_lag_data(
    session: Session,
    city_id: str,
    event_date: date,
) -> pd.DataFrame:
    """Load historical data needed for compute_lag_features().

    Returns a DataFrame with ['city', 'day', 'settle_f', 'vc_max_f'] that can
    be passed to calendar.compute_lag_features().

    Args:
        session: Database session
        city_id: City identifier
        event_date: Target date (loads data for 7 days before this)

    Returns:
        DataFrame ready for compute_lag_features()
    """
    start_date = event_date - timedelta(days=7)
    end_date = event_date - timedelta(days=1)

    # Load settlements (reuse existing function)
    settlements_df = load_settlements(session, city_id, start_date, end_date)

    # Load VC obs and compute daily max (reuse existing function)
    obs_df = load_vc_observations(session, city_id, start_date, end_date)

    # Build historical DataFrame in format expected by compute_lag_features
    rows = []
    if not settlements_df.empty:
        for _, row in settlements_df.iterrows():
            day = row["date_local"]
            settle_f = int(row["tmax_final"]) if row["tmax_final"] else None

            # Get VC max for this day
            vc_max_f = None
            if obs_df is not None and not obs_df.empty:
                day_obs = obs_df[pd.to_datetime(obs_df["datetime_local"]).dt.date == day]
                if not day_obs.empty:
                    vc_max_f = float(day_obs["temp_f"].max())

            rows.append({
                "city": city_id,
                "day": day,
                "settle_f": settle_f,
                "vc_max_f": vc_max_f,
            })

    return pd.DataFrame(rows)


def load_full_inference_data(
    city_id: str,
    event_date: date,
    cutoff_time: datetime,
    session: Session,
) -> dict:
    """Load ALL data for inference matching training pipeline.

    This function replicates the data loading from dataset.build_dataset()
    to ensure feature parity between training and inference.

    STRICT MODE: Raises ValueError if critical data is missing.

    Args:
        city_id: City identifier
        event_date: The settlement date (D)
        cutoff_time: Local datetime cutoff for observations
        session: Database session

    Returns:
        Dict with all data needed for SnapshotContext:
            obs_df, temps_sofar, timestamps_sofar, fcst_daily, fcst_hourly_df,
            fcst_multi, candles_df, city_obs_df, more_apis,
            obs_t15_mean, obs_t15_std, window_start

    Raises:
        ValueError: If observations are empty or critical data missing
    """
    from models.data.snapshot import get_market_clock_window

    # Get market-clock window: D-1 10:00 to D 23:55
    window_start, window_end = get_market_clock_window(event_date)

    # 1. Load FULL observations (all 14 columns) from D-1 to D
    d_minus_1 = event_date - timedelta(days=1)
    obs_df = load_vc_observations(session, city_id, d_minus_1, event_date)

    # STRICT: Fail if no observations
    if obs_df is None or obs_df.empty:
        raise ValueError(f"No observations found for {city_id} {event_date}")

    # Normalize cutoff_time to naive for comparison with DB datetime_local
    # (DB stores local time as naive datetime)
    cutoff_naive = cutoff_time.replace(tzinfo=None) if cutoff_time.tzinfo else cutoff_time

    # Filter to [window_start, cutoff_time)
    obs_df = obs_df[
        (obs_df["datetime_local"] >= window_start) &
        (obs_df["datetime_local"] < cutoff_naive)
    ].copy()

    if obs_df.empty:
        raise ValueError(
            f"No observations in window [{window_start}, {cutoff_time}) for {city_id}"
        )

    # Extract temps_sofar and timestamps_sofar
    temps_sofar = []
    timestamps_sofar = []
    for _, row in obs_df.iterrows():
        if row["temp_f"] is not None:
            temps_sofar.append(float(row["temp_f"]))
            timestamps_sofar.append(row["datetime_local"])

    if len(temps_sofar) == 0:
        raise ValueError(f"No valid temperatures in observations for {city_id}")

    # 2. Load T-1 forecasts (daily + hourly)
    basis_date = event_date - timedelta(days=1)
    fcst_daily = load_historical_forecast_daily(session, city_id, event_date, basis_date)
    fcst_hourly_df = load_historical_forecast_hourly(session, city_id, event_date, basis_date)
    if fcst_hourly_df is not None and fcst_hourly_df.empty:
        fcst_hourly_df = None

    # 2a. Interpolate hourly cloudcover to 5-minute observation timestamps
    # VC minute obs often lack cloudcover; hourly forecasts have it
    if fcst_hourly_df is not None and not fcst_hourly_df.empty:
        # Rename column to match interpolation function expectation
        hourly_for_interp = fcst_hourly_df.rename(
            columns={"target_datetime_local": "datetime_local"}
        )
        obs_df = interpolate_cloudcover_from_hourly(obs_df, hourly_for_interp)
        logger.debug(
            f"Interpolated cloudcover from hourly forecast to {len(obs_df)} observations"
        )

    # 3. Load multi-horizon forecasts (T-1 to T-6)
    fcst_multi = load_multi_horizon_forecasts(session, city_id, event_date)

    # 4. Load market candles
    candles_df = load_candles_for_inference(
        session, city_id, event_date, window_start, cutoff_time
    )

    # 5. Load city observations for station-city gap
    city_obs_df = load_city_observations_for_inference(
        session, city_id, window_start, cutoff_time
    )

    # 6. Load NOAA guidance
    # Only convert to UTC if cutoff_time has tzinfo
    cutoff_utc = None
    if cutoff_time.tzinfo is not None:
        cutoff_utc = cutoff_time.astimezone(ZoneInfo("UTC"))
    more_apis = load_weather_more_apis_guidance(session, city_id, event_date, cutoff_utc)

    # 7. Load 30-day obs stats for z-score normalization
    obs_t15_mean, obs_t15_std = load_obs_t15_stats_30d(session, city_id, event_date)

    # 8. Load historical lag data (settlements + VC max for past 7 days)
    lag_data = load_historical_lag_data(session, city_id, event_date)

    logger.info(
        f"Loaded full inference data for {city_id} {event_date}: "
        f"{len(temps_sofar)} obs, fcst={'Yes' if fcst_daily else 'No'}, "
        f"candles={'Yes' if candles_df is not None else 'No'}, "
        f"city_obs={'Yes' if city_obs_df is not None else 'No'}, "
        f"lag_data={'Yes' if not lag_data.empty else 'No'}"
    )

    return {
        "city": city_id,
        "event_date": event_date,
        "cutoff_time": cutoff_time,
        "window_start": window_start,
        "obs_df": obs_df,
        "temps_sofar": temps_sofar,
        "timestamps_sofar": timestamps_sofar,
        "fcst_daily": fcst_daily,
        "fcst_hourly_df": fcst_hourly_df,
        "fcst_multi": fcst_multi,
        "candles_df": candles_df,
        "city_obs_df": city_obs_df,
        "more_apis": more_apis,
        "obs_t15_mean": obs_t15_mean,
        "obs_t15_std": obs_t15_std,
        "lag_data": lag_data,
    }
