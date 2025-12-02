"""
Meteorological features for temperature delta-models.

This module computes features from non-temperature weather observations,
including humidity, wind speed, and cloud cover. These factors influence
how temperature evolves throughout the day and the final daily high.

Physical intuition:
    - Humidity: High humidity reduces diurnal temperature range (less cooling at night,
      less heating during day). Humid air has higher heat capacity.
    - Wind: Strong winds promote mixing, which can prevent temperature extremes.
      Wind can also advect different air masses into the region.
    - Cloud cover: Clouds block incoming solar radiation, reducing heating.
      High cloud cover typically means a cooler max temperature.

Features computed:
    Humidity:
        humidity_last_obs: Current relative humidity (%)
        humidity_mean_last_60min: Rolling mean humidity
        humidity_std_last_60min: Rolling std (variability)
        high_humidity_flag: 1 if humidity > 80%

    Wind:
        windspeed_last_obs: Current wind speed (mph)
        windspeed_max_last_60min: Max wind in last 60 min
        windgust_max_last_60min: Max gust in last 60 min
        strong_wind_flag: 1 if current windspeed > 15 mph

    Cloud cover:
        cloudcover_last_obs: Current cloud cover (%)
        cloudcover_mean_last_60min: Rolling mean cloud cover
        high_cloud_flag: 1 if cloudcover > 70% (suppresses heating)
        clear_sky_flag: 1 if cloudcover < 20% (max solar heating potential)

Example:
    >>> obs_df = load_observations(city, date)  # Has humidity, windspeed_mph, cloudcover
    >>> fs = compute_meteo_features(obs_df, snapshot_time)
    >>> fs['high_cloud_flag']
    1
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from models.features.base import FeatureSet, register_feature_group


def _get_obs_in_window(
    obs_df: pd.DataFrame,
    column: str,
    snapshot_time: datetime,
    window_minutes: int,
) -> pd.Series:
    """Extract observations within the last N minutes from snapshot time.

    Args:
        obs_df: DataFrame with datetime_local column
        column: Name of column to extract
        snapshot_time: Current time for filtering
        window_minutes: Size of lookback window in minutes

    Returns:
        Series of values within the window (may be empty)
    """
    if obs_df.empty or column not in obs_df.columns:
        return pd.Series(dtype=float)

    cutoff = snapshot_time - timedelta(minutes=window_minutes)
    mask = (obs_df["datetime_local"] >= cutoff) & (obs_df["datetime_local"] <= snapshot_time)
    return obs_df.loc[mask, column].dropna()


@register_feature_group("meteo")
def compute_meteo_features(
    obs_df: Optional[pd.DataFrame],
    snapshot_time: Optional[datetime] = None,
) -> FeatureSet:
    """Compute meteorological features from weather observations.

    Args:
        obs_df: DataFrame with columns:
                datetime_local, humidity, windspeed_mph, windgust_mph, cloudcover
                Sorted by datetime_local ascending.
        snapshot_time: Cutoff time (use observations up to this time).
                      If None, uses all observations.

    Returns:
        FeatureSet with meteorological features
    """
    null_features = {
        # Humidity
        "humidity_last_obs": None,
        "humidity_mean_last_60min": None,
        "humidity_std_last_60min": None,
        "high_humidity_flag": None,
        # Wind
        "windspeed_last_obs": None,
        "windspeed_max_last_60min": None,
        "windgust_max_last_60min": None,
        "strong_wind_flag": None,
        # Cloud cover
        "cloudcover_last_obs": None,
        "cloudcover_mean_last_60min": None,
        "high_cloud_flag": None,
        "clear_sky_flag": None,
    }

    if obs_df is None or obs_df.empty:
        return FeatureSet(name="meteo", features=null_features)

    df = obs_df.copy()

    # Ensure datetime_local is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["datetime_local"]):
        df["datetime_local"] = pd.to_datetime(df["datetime_local"])

    # Filter to snapshot time if provided
    if snapshot_time is not None:
        # Handle timezone comparison
        if df["datetime_local"].dt.tz is not None:
            if isinstance(snapshot_time, datetime) and snapshot_time.tzinfo is None:
                df["datetime_local"] = df["datetime_local"].dt.tz_localize(None)
        df = df[df["datetime_local"] <= snapshot_time]

    if df.empty:
        return FeatureSet(name="meteo", features=null_features)

    # Sort by time
    df = df.sort_values("datetime_local")

    features = {}

    # Use last row for "current" values
    last_row = df.iloc[-1]
    last_time = df["datetime_local"].iloc[-1]

    # If snapshot_time provided, use it for window calculations
    ref_time = snapshot_time if snapshot_time else last_time

    # -------------------------------------------------------------------------
    # Humidity features
    # -------------------------------------------------------------------------
    humidity = last_row.get("humidity")
    features["humidity_last_obs"] = float(humidity) if pd.notna(humidity) else None

    # Rolling humidity stats
    humidity_window = _get_obs_in_window(df, "humidity", ref_time, 60)
    if len(humidity_window) > 0:
        features["humidity_mean_last_60min"] = float(humidity_window.mean())
        features["humidity_std_last_60min"] = float(humidity_window.std(ddof=1)) if len(humidity_window) > 1 else 0.0
    else:
        features["humidity_mean_last_60min"] = None
        features["humidity_std_last_60min"] = None

    # High humidity flag (>80% typically feels muggy, affects heating)
    if pd.notna(humidity):
        features["high_humidity_flag"] = 1 if humidity > 80 else 0
    else:
        features["high_humidity_flag"] = None

    # -------------------------------------------------------------------------
    # Wind features
    # -------------------------------------------------------------------------
    windspeed = last_row.get("windspeed_mph")
    features["windspeed_last_obs"] = float(windspeed) if pd.notna(windspeed) else None

    # Rolling wind max
    wind_window = _get_obs_in_window(df, "windspeed_mph", ref_time, 60)
    if len(wind_window) > 0:
        features["windspeed_max_last_60min"] = float(wind_window.max())
    else:
        features["windspeed_max_last_60min"] = None

    # Rolling gust max
    gust_window = _get_obs_in_window(df, "windgust_mph", ref_time, 60)
    if len(gust_window) > 0:
        features["windgust_max_last_60min"] = float(gust_window.max())
    else:
        features["windgust_max_last_60min"] = None

    # Strong wind flag (>15 mph causes noticeable mixing)
    if pd.notna(windspeed):
        features["strong_wind_flag"] = 1 if windspeed > 15 else 0
    else:
        features["strong_wind_flag"] = None

    # -------------------------------------------------------------------------
    # Cloud cover features
    # -------------------------------------------------------------------------
    cloudcover = last_row.get("cloudcover")
    features["cloudcover_last_obs"] = float(cloudcover) if pd.notna(cloudcover) else None

    # Rolling cloud cover mean
    cloud_window = _get_obs_in_window(df, "cloudcover", ref_time, 60)
    if len(cloud_window) > 0:
        features["cloudcover_mean_last_60min"] = float(cloud_window.mean())
    else:
        features["cloudcover_mean_last_60min"] = None

    # Cloud flags
    if pd.notna(cloudcover):
        features["high_cloud_flag"] = 1 if cloudcover > 70 else 0
        features["clear_sky_flag"] = 1 if cloudcover < 20 else 0
    else:
        features["high_cloud_flag"] = None
        features["clear_sky_flag"] = None

    return FeatureSet(name="meteo", features=features)


def compute_meteo_trend_features(
    obs_df: Optional[pd.DataFrame],
    snapshot_time: Optional[datetime] = None,
) -> FeatureSet:
    """Compute trend features for meteorological variables.

    These features capture whether conditions are changing, which can
    signal upcoming weather changes that affect temperature.

    Args:
        obs_df: DataFrame with datetime_local and meteo columns
        snapshot_time: Cutoff time

    Returns:
        FeatureSet with trend features
    """
    null_features = {
        "cloudcover_change_last_60min": None,
        "humidity_change_last_60min": None,
        "wind_increasing_flag": None,
    }

    if obs_df is None or obs_df.empty:
        return FeatureSet(name="meteo_trends", features=null_features)

    df = obs_df.copy()

    # Ensure datetime_local is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["datetime_local"]):
        df["datetime_local"] = pd.to_datetime(df["datetime_local"])

    # Filter to snapshot time
    if snapshot_time is not None:
        if df["datetime_local"].dt.tz is not None:
            if isinstance(snapshot_time, datetime) and snapshot_time.tzinfo is None:
                df["datetime_local"] = df["datetime_local"].dt.tz_localize(None)
        df = df[df["datetime_local"] <= snapshot_time]

    if df.empty:
        return FeatureSet(name="meteo_trends", features=null_features)

    df = df.sort_values("datetime_local")
    last_time = df["datetime_local"].iloc[-1]
    ref_time = snapshot_time if snapshot_time else last_time

    features = {}

    # Cloud cover change
    cutoff = ref_time - timedelta(minutes=60)
    window_df = df[df["datetime_local"] >= cutoff]

    if len(window_df) >= 2 and "cloudcover" in window_df.columns:
        first_cloud = window_df["cloudcover"].dropna().iloc[0] if not window_df["cloudcover"].dropna().empty else None
        last_cloud = window_df["cloudcover"].dropna().iloc[-1] if not window_df["cloudcover"].dropna().empty else None
        if first_cloud is not None and last_cloud is not None:
            features["cloudcover_change_last_60min"] = float(last_cloud - first_cloud)
        else:
            features["cloudcover_change_last_60min"] = None
    else:
        features["cloudcover_change_last_60min"] = None

    # Humidity change
    if len(window_df) >= 2 and "humidity" in window_df.columns:
        first_hum = window_df["humidity"].dropna().iloc[0] if not window_df["humidity"].dropna().empty else None
        last_hum = window_df["humidity"].dropna().iloc[-1] if not window_df["humidity"].dropna().empty else None
        if first_hum is not None and last_hum is not None:
            features["humidity_change_last_60min"] = float(last_hum - first_hum)
        else:
            features["humidity_change_last_60min"] = None
    else:
        features["humidity_change_last_60min"] = None

    # Wind increasing flag
    if len(window_df) >= 2 and "windspeed_mph" in window_df.columns:
        first_wind = window_df["windspeed_mph"].dropna().iloc[0] if not window_df["windspeed_mph"].dropna().empty else None
        last_wind = window_df["windspeed_mph"].dropna().iloc[-1] if not window_df["windspeed_mph"].dropna().empty else None
        if first_wind is not None and last_wind is not None:
            features["wind_increasing_flag"] = 1 if last_wind > first_wind + 2 else 0
        else:
            features["wind_increasing_flag"] = None
    else:
        features["wind_increasing_flag"] = None

    return FeatureSet(name="meteo_trends", features=features)
