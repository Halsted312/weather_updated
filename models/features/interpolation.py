"""
Interpolation utilities for filling gaps in observation data.

This module provides functions to interpolate hourly forecast data (e.g., cloudcover)
to match the finer-grained observation timestamps (5-minute intervals).

Physical rationale:
    - Cloudcover typically changes gradually over hours, not instantaneously
    - Linear interpolation provides a reasonable estimate between hourly values
    - Observations often have 5-min granularity, while some forecast variables are hourly

Functions:
    interpolate_cloudcover_from_hourly: Fill 5-min obs with interpolated hourly forecast cloudcover
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def interpolate_cloudcover_from_hourly(
    obs_df: pd.DataFrame,
    hourly_fcst_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Interpolate hourly forecast cloudcover to match observation timestamps.

    Args:
        obs_df: Observation DataFrame with datetime_local column (5-min intervals)
                May or may not have cloudcover column already.
        hourly_fcst_df: Hourly forecast DataFrame with datetime_local and cloudcover columns
                       (1-hour intervals). If None, returns obs_df unchanged.

    Returns:
        obs_df with cloudcover column added/updated via linear interpolation

    Example:
        >>> obs_df = load_observations(city, date)  # 5-min intervals, no cloudcover
        >>> hourly_fcst = load_hourly_forecast(city, date)  # Hourly cloudcover: [20, 50, 80, 60]
        >>> obs_df = interpolate_cloudcover_from_hourly(obs_df, hourly_fcst)
        >>> obs_df['cloudcover']  # Now has interpolated values at 5-min intervals
    """
    if hourly_fcst_df is None or hourly_fcst_df.empty:
        logger.debug("No hourly forecast data - cannot interpolate cloudcover")
        return obs_df

    if 'cloudcover' not in hourly_fcst_df.columns:
        logger.debug("No cloudcover column in hourly forecast data")
        return obs_df

    # Make a copy to avoid modifying original
    obs_df = obs_df.copy()

    # Ensure datetime columns are datetime type
    if not pd.api.types.is_datetime64_any_dtype(obs_df['datetime_local']):
        obs_df['datetime_local'] = pd.to_datetime(obs_df['datetime_local'])

    if not pd.api.types.is_datetime64_any_dtype(hourly_fcst_df['datetime_local']):
        hourly_fcst_df = hourly_fcst_df.copy()
        hourly_fcst_df['datetime_local'] = pd.to_datetime(hourly_fcst_df['datetime_local'])

    # Remove timezone info for merging (if present)
    if obs_df['datetime_local'].dt.tz is not None:
        obs_df['datetime_local'] = obs_df['datetime_local'].dt.tz_localize(None)
    if hourly_fcst_df['datetime_local'].dt.tz is not None:
        hourly_fcst_df = hourly_fcst_df.copy()
        hourly_fcst_df['datetime_local'] = hourly_fcst_df['datetime_local'].dt.tz_localize(None)

    # Extract hourly cloudcover values (drop nulls)
    hourly_cc = hourly_fcst_df[['datetime_local', 'cloudcover']].dropna()

    if hourly_cc.empty:
        logger.debug("No non-null cloudcover values in hourly forecast")
        return obs_df

    # Create a series indexed by datetime for interpolation
    hourly_series = pd.Series(
        hourly_cc['cloudcover'].values,
        index=hourly_cc['datetime_local']
    )

    # Combine observation timestamps with hourly timestamps
    all_times = pd.concat([
        pd.Series(obs_df['datetime_local']),
        pd.Series(hourly_series.index)
    ]).drop_duplicates().sort_values()

    # Reindex hourly series to all times and interpolate linearly
    interpolated_series = hourly_series.reindex(all_times)
    interpolated_series = interpolated_series.interpolate(method='linear', limit_direction='both')

    # Map interpolated values back to observation DataFrame
    obs_df['cloudcover'] = obs_df['datetime_local'].map(interpolated_series)

    # Log statistics
    non_null_before = 0 if 'cloudcover' not in obs_df.columns else obs_df['cloudcover'].notna().sum()
    non_null_after = obs_df['cloudcover'].notna().sum()
    logger.debug(
        f"Cloudcover interpolation: {non_null_before} → {non_null_after} non-null "
        f"({100 * non_null_after / len(obs_df):.1f}%)"
    )

    return obs_df


def interpolate_meteo_from_hourly(
    obs_df: pd.DataFrame,
    hourly_fcst_df: Optional[pd.DataFrame],
    variables: list[str] = None,
) -> pd.DataFrame:
    """Interpolate multiple meteo variables from hourly forecast to observation timestamps.

    More general version of interpolate_cloudcover_from_hourly that handles
    multiple variables (cloudcover, humidity, windspeed, etc.).

    Args:
        obs_df: Observation DataFrame with datetime_local
        hourly_fcst_df: Hourly forecast DataFrame
        variables: List of column names to interpolate (default: ['cloudcover'])

    Returns:
        obs_df with interpolated columns added/updated
    """
    if variables is None:
        variables = ['cloudcover']

    if hourly_fcst_df is None or hourly_fcst_df.empty:
        return obs_df

    obs_df = obs_df.copy()

    # Ensure datetime columns are datetime type
    if not pd.api.types.is_datetime64_any_dtype(obs_df['datetime_local']):
        obs_df['datetime_local'] = pd.to_datetime(obs_df['datetime_local'])

    if not pd.api.types.is_datetime64_any_dtype(hourly_fcst_df['datetime_local']):
        hourly_fcst_df = hourly_fcst_df.copy()
        hourly_fcst_df['datetime_local'] = pd.to_datetime(hourly_fcst_df['datetime_local'])

    # Remove timezone info
    if obs_df['datetime_local'].dt.tz is not None:
        obs_df['datetime_local'] = obs_df['datetime_local'].dt.tz_localize(None)
    if hourly_fcst_df['datetime_local'].dt.tz is not None:
        hourly_fcst_df = hourly_fcst_df.copy()
        hourly_fcst_df['datetime_local'] = hourly_fcst_df['datetime_local'].dt.tz_localize(None)

    for var in variables:
        if var not in hourly_fcst_df.columns:
            logger.debug(f"Variable '{var}' not in hourly forecast data")
            continue

        # Extract hourly values
        hourly_data = hourly_fcst_df[['datetime_local', var]].dropna()

        if hourly_data.empty:
            logger.debug(f"No non-null {var} values in hourly forecast")
            continue

        # Create series indexed by datetime
        hourly_series = pd.Series(
            hourly_data[var].values,
            index=hourly_data['datetime_local']
        )

        # Combine timestamps
        all_times = pd.concat([
            pd.Series(obs_df['datetime_local']),
            pd.Series(hourly_series.index)
        ]).drop_duplicates().sort_values()

        # Interpolate
        interpolated_series = hourly_series.reindex(all_times)
        interpolated_series = interpolated_series.interpolate(method='linear', limit_direction='both')

        # Map back to observations
        obs_df[var] = obs_df['datetime_local'].map(interpolated_series)

        non_null = obs_df[var].notna().sum()
        logger.debug(f"Interpolated {var}: {non_null}/{len(obs_df)} non-null ({100*non_null/len(obs_df):.1f}%)")

    return obs_df
