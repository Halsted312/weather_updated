"""
Advanced meteorological features: wet bulb, wind chill, cloud dynamics.

These features capture humidity stress, convective cooling, and radiative forcing
that affect daily high temperatures beyond simple humidity/wind/cloud measurements.

Physical intuition:
    - Wet bulb temperature: Captures the cooling effect of evaporation, which depends
      on both temperature and humidity. High wet bulb (>75°F) indicates oppressive
      humidity that limits maximum temperature potential.

    - Wind chill: Captures convective heat loss in cold/windy conditions. Only
      relevant when T <= 50°F (primarily Chicago, Denver, Philadelphia in winter).

    - Cloud dynamics: Static cloud cover alone doesn't capture transitions. Clearing
      sky (cloud cover dropping) signals incoming solar heating. Clouding up signals
      dimming and cooling.

Features computed:
    Wet Bulb (6 features):
        wetbulb_last_obs: Current wet bulb temperature (°F)
        wetbulb_mean_last_60min: Rolling mean wet bulb
        wetbulb_depression: Dry bulb - wet bulb (dryness indicator)
        wetbulb_depression_mean_60min: Rolling mean depression
        high_wetbulb_flag: 1 if wet bulb > 75°F
        wetbulb_rate_last_30min: Rate of wet bulb change (°F/hr)

    Wind Chill (5 features):
        windchill_last_obs: Current wind chill (°F), None if T > 50°F
        windchill_depression: Temp - wind chill (cooling effect)
        windchill_mean_last_60min: Rolling mean wind chill
        strong_windchill_flag: 1 if wind chill depression > 10°F
        windchill_warming_rate: Rate of wind chill increase (°F/hr)

    Cloud Dynamics (6 features):
        cloudcover_rate_last_30min: Rate of cloud cover change (%/hr)
        cloudcover_volatility_60min: Std dev of cloud cover (stability)
        clearing_trend_flag: 1 if cloud cover dropped > 20% in last 60min
        clouding_trend_flag: 1 if cloud cover increased > 20% in last 60min
        cloud_regime: 0=clear (<20%), 1=partly (20-70%), 2=overcast (>70%)
        cloud_stability_score: 1 - (volatility / 100), forecast reliability

Example:
    >>> obs_df = load_observations(city, date)  # Has temp_f, humidity, windspeed_mph, cloudcover
    >>> fs = compute_meteo_advanced_features(obs_df, snapshot_time)
    >>> fs['wetbulb_last_obs']
    72.3
    >>> fs['clearing_trend_flag']
    1  # Sky is clearing, expect warming
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from models.features.base import FeatureSet, register_feature_group


def _compute_wet_bulb_approx(temp_f: float, humidity_pct: float) -> float:
    """Stull wet bulb approximation (valid 0-50°C, RH 5-99%).

    Formula: Tw = T*arctan[0.151977*(RH + 8.313659)^0.5] +
             arctan(T + RH) - arctan(RH - 1.676331) +
             0.00391838*RH^1.5*arctan(0.023101*RH) - 4.686035

    Where T is in Celsius, RH is 0-100.

    Args:
        temp_f: Dry bulb temperature (°F)
        humidity_pct: Relative humidity (0-100%)

    Returns:
        Wet bulb temperature (°F)
    """
    # Convert to Celsius
    temp_c = (temp_f - 32) * 5/9
    rh = humidity_pct

    # Stull formula (simplified, good to ±1°C)
    tw_c = (temp_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) +
            np.arctan(temp_c + rh) -
            np.arctan(rh - 1.676331) +
            0.00391838 * (rh ** 1.5) * np.arctan(0.023101 * rh) -
            4.686035)

    # Convert back to Fahrenheit
    tw_f = tw_c * 9/5 + 32
    return float(tw_f)


def _compute_wind_chill(temp_f: float, windspeed_mph: float) -> float:
    """NWS wind chill formula (valid T <= 50°F, wind >= 3 mph).

    WC = 35.74 + 0.6215*T - 35.75*V^0.16 + 0.4275*T*V^0.16

    Args:
        temp_f: Air temperature (°F)
        windspeed_mph: Wind speed (mph)

    Returns:
        Wind chill temperature (°F), or input temp if formula doesn't apply
    """
    if temp_f > 50:
        return temp_f  # No wind chill effect above 50°F

    if windspeed_mph < 3:
        return temp_f  # Formula not valid below 3 mph

    wc = (35.74 + 0.6215 * temp_f -
          35.75 * (windspeed_mph ** 0.16) +
          0.4275 * temp_f * (windspeed_mph ** 0.16))

    return float(wc)


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


@register_feature_group("meteo_advanced")
def compute_meteo_advanced_features(
    obs_df: Optional[pd.DataFrame],
    snapshot_time: Optional[datetime] = None,
) -> FeatureSet:
    """Compute wet bulb, wind chill, and cloud dynamics features.

    Args:
        obs_df: DataFrame with columns: datetime_local, temp_f, humidity,
                windspeed_mph, cloudcover. Sorted by datetime_local ascending.
        snapshot_time: Cutoff time (use observations up to this time).
                      If None, uses all observations.

    Returns:
        FeatureSet with 17 advanced meteo features
    """
    null_features = {
        # Wet bulb (6 features)
        "wetbulb_last_obs": None,
        "wetbulb_mean_last_60min": None,
        "wetbulb_depression": None,
        "wetbulb_depression_mean_60min": None,
        "high_wetbulb_flag": None,
        "wetbulb_rate_last_30min": None,
        # Wind chill (5 features)
        "windchill_last_obs": None,
        "windchill_depression": None,
        "windchill_mean_last_60min": None,
        "strong_windchill_flag": None,
        "windchill_warming_rate": None,
        # Cloud dynamics (6 features)
        "cloudcover_rate_last_30min": None,
        "cloudcover_volatility_60min": None,
        "clearing_trend_flag": None,
        "clouding_trend_flag": None,
        "cloud_regime": None,
        "cloud_stability_score": None,
    }

    if obs_df is None or obs_df.empty:
        return FeatureSet(name="meteo_advanced", features=null_features)

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
        return FeatureSet(name="meteo_advanced", features=null_features)

    # Sort by time
    df = df.sort_values("datetime_local")

    features = {}

    # Use last row for "current" values
    last_row = df.iloc[-1]
    last_time = df["datetime_local"].iloc[-1]

    # If snapshot_time provided, use it for window calculations
    ref_time = snapshot_time if snapshot_time else last_time

    # =========================================================================
    # WET BULB FEATURES
    # =========================================================================

    # Compute wet bulb for all rows where we have temp and humidity
    if "temp_f" in df.columns and "humidity" in df.columns:
        wet_bulb_series = []
        for _, row in df.iterrows():
            temp = row.get("temp_f")
            hum = row.get("humidity")
            if pd.notna(temp) and pd.notna(hum) and 0 <= hum <= 100:
                try:
                    wb = _compute_wet_bulb_approx(temp, hum)
                    wet_bulb_series.append(wb)
                except:
                    wet_bulb_series.append(None)
            else:
                wet_bulb_series.append(None)

        df["wet_bulb_f"] = wet_bulb_series

        # Last observation
        wb_last = df["wet_bulb_f"].iloc[-1]
        features["wetbulb_last_obs"] = float(wb_last) if pd.notna(wb_last) else None

        # Rolling 60min mean
        wb_window = _get_obs_in_window(df, "wet_bulb_f", ref_time, 60)
        if len(wb_window) > 0:
            features["wetbulb_mean_last_60min"] = float(wb_window.mean())
        else:
            features["wetbulb_mean_last_60min"] = None

        # Wet bulb depression (dry bulb - wet bulb)
        temp_last = last_row.get("temp_f")
        if pd.notna(temp_last) and pd.notna(wb_last):
            depression = temp_last - wb_last
            features["wetbulb_depression"] = float(depression)
        else:
            features["wetbulb_depression"] = None

        # Rolling mean depression
        if "temp_f" in df.columns and len(wb_window) > 0:
            temp_window = _get_obs_in_window(df, "temp_f", ref_time, 60)
            if len(temp_window) > 0 and len(wb_window) == len(temp_window):
                depressions = temp_window.values - wb_window.values
                features["wetbulb_depression_mean_60min"] = float(np.mean(depressions))
            else:
                features["wetbulb_depression_mean_60min"] = None
        else:
            features["wetbulb_depression_mean_60min"] = None

        # High wet bulb flag (>75°F is oppressive)
        if pd.notna(wb_last):
            features["high_wetbulb_flag"] = 1 if wb_last > 75 else 0
        else:
            features["high_wetbulb_flag"] = None

        # Wet bulb rate of change (last 30min)
        wb_window_30 = _get_obs_in_window(df, "wet_bulb_f", ref_time, 30)
        if len(wb_window_30) >= 2:
            # Simple linear regression for rate
            times = pd.to_datetime(_get_obs_in_window(df, "datetime_local", ref_time, 30)).astype(np.int64) / 1e9  # seconds
            if len(times) == len(wb_window_30):
                # Rate in °F per second * 3600 = °F per hour
                rate_per_sec = (wb_window_30.iloc[-1] - wb_window_30.iloc[0]) / (times.iloc[-1] - times.iloc[0]) if len(times) > 1 and (times.iloc[-1] - times.iloc[0]) > 0 else 0
                features["wetbulb_rate_last_30min"] = float(rate_per_sec * 3600)
            else:
                features["wetbulb_rate_last_30min"] = None
        else:
            features["wetbulb_rate_last_30min"] = None
    else:
        # Missing temp or humidity columns
        features["wetbulb_last_obs"] = None
        features["wetbulb_mean_last_60min"] = None
        features["wetbulb_depression"] = None
        features["wetbulb_depression_mean_60min"] = None
        features["high_wetbulb_flag"] = None
        features["wetbulb_rate_last_30min"] = None

    # =========================================================================
    # WIND CHILL FEATURES
    # =========================================================================

    if "temp_f" in df.columns and "windspeed_mph" in df.columns:
        # Compute wind chill for all rows
        wind_chill_series = []
        for _, row in df.iterrows():
            temp = row.get("temp_f")
            wind = row.get("windspeed_mph")
            if pd.notna(temp) and pd.notna(wind):
                try:
                    wc = _compute_wind_chill(temp, wind)
                    wind_chill_series.append(wc)
                except:
                    wind_chill_series.append(None)
            else:
                wind_chill_series.append(None)

        df["wind_chill_f"] = wind_chill_series

        # Last observation
        wc_last = df["wind_chill_f"].iloc[-1]
        temp_last = last_row.get("temp_f")

        # Only report wind chill if temp <= 50°F
        if pd.notna(wc_last) and pd.notna(temp_last) and temp_last <= 50:
            features["windchill_last_obs"] = float(wc_last)

            # Wind chill depression
            depression = temp_last - wc_last
            features["windchill_depression"] = float(depression)

            # Strong wind chill flag (depression > 10°F)
            features["strong_windchill_flag"] = 1 if depression > 10 else 0
        else:
            # Too warm for wind chill
            features["windchill_last_obs"] = None
            features["windchill_depression"] = None
            features["strong_windchill_flag"] = None

        # Rolling 60min mean
        wc_window = _get_obs_in_window(df, "wind_chill_f", ref_time, 60)
        if len(wc_window) > 0:
            features["windchill_mean_last_60min"] = float(wc_window.mean())
        else:
            features["windchill_mean_last_60min"] = None

        # Wind chill warming rate (last 30min)
        wc_window_30 = _get_obs_in_window(df, "wind_chill_f", ref_time, 30)
        if len(wc_window_30) >= 2:
            times = pd.to_datetime(_get_obs_in_window(df, "datetime_local", ref_time, 30)).astype(np.int64) / 1e9
            if len(times) == len(wc_window_30) and (times.iloc[-1] - times.iloc[0]) > 0:
                rate_per_sec = (wc_window_30.iloc[-1] - wc_window_30.iloc[0]) / (times.iloc[-1] - times.iloc[0])
                features["windchill_warming_rate"] = float(rate_per_sec * 3600)
            else:
                features["windchill_warming_rate"] = None
        else:
            features["windchill_warming_rate"] = None
    else:
        # Missing temp or wind columns
        features["windchill_last_obs"] = None
        features["windchill_depression"] = None
        features["windchill_mean_last_60min"] = None
        features["strong_windchill_flag"] = None
        features["windchill_warming_rate"] = None

    # =========================================================================
    # CLOUD DYNAMICS FEATURES
    # =========================================================================

    if "cloudcover" in df.columns:
        cloudcover = last_row.get("cloudcover")

        # Cloud regime classification
        if pd.notna(cloudcover):
            if cloudcover < 20:
                features["cloud_regime"] = 0  # Clear
            elif cloudcover < 70:
                features["cloud_regime"] = 1  # Partly cloudy
            else:
                features["cloud_regime"] = 2  # Overcast
        else:
            features["cloud_regime"] = None

        # Rate of change (last 30min)
        cloud_window_30 = _get_obs_in_window(df, "cloudcover", ref_time, 30)
        if len(cloud_window_30) >= 2:
            # Change in % per hour
            delta_cloud = cloud_window_30.iloc[-1] - cloud_window_30.iloc[0]
            time_span_hours = 0.5  # 30 minutes
            features["cloudcover_rate_last_30min"] = float(delta_cloud / time_span_hours)
        else:
            features["cloudcover_rate_last_30min"] = None

        # Volatility (std dev over 60min)
        cloud_window_60 = _get_obs_in_window(df, "cloudcover", ref_time, 60)
        if len(cloud_window_60) > 1:
            volatility = float(cloud_window_60.std(ddof=1))
            features["cloudcover_volatility_60min"] = volatility

            # Stability score (1 - normalized volatility)
            features["cloud_stability_score"] = 1.0 - min(1.0, volatility / 100.0)
        else:
            features["cloudcover_volatility_60min"] = None
            features["cloud_stability_score"] = None

        # Clearing/clouding trend flags (change > 20% in 60min)
        if len(cloud_window_60) >= 2:
            delta_60min = cloud_window_60.iloc[-1] - cloud_window_60.iloc[0]
            features["clearing_trend_flag"] = 1 if delta_60min < -20 else 0
            features["clouding_trend_flag"] = 1 if delta_60min > 20 else 0
        else:
            features["clearing_trend_flag"] = None
            features["clouding_trend_flag"] = None
    else:
        # Missing cloudcover column
        features["cloudcover_rate_last_30min"] = None
        features["cloudcover_volatility_60min"] = None
        features["clearing_trend_flag"] = None
        features["clouding_trend_flag"] = None
        features["cloud_regime"] = None
        features["cloud_stability_score"] = None

    return FeatureSet(name="meteo_advanced", features=features)
