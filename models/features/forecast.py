"""
T-1 forecast and forecast-vs-actual error features.

This module computes features from the forecast issued on day D-1 for
target day D, plus real-time comparison with actual observations as
they come in throughout the day.

These features capture:
- What VC expected yesterday (prior knowledge)
- How wrong/right that forecast has been so far (forecast bias)
- Whether today is tracking hotter or cooler than expected

Static features (known at start of day):
    fcst_prev_max_f: Forecast daily high from T-1
    fcst_prev_min_f: Forecast daily low from T-1
    fcst_prev_mean_f: Mean of hourly forecast temps
    fcst_prev_std_f: Std of hourly forecast temps
    fcst_prev_q10_f through fcst_prev_q90_f: Percentiles
    fcst_prev_frac_part: Fractional part of forecast max
    fcst_prev_hour_of_max: Hour when forecast max occurs
    t_forecast_base: Rounded forecast high

Dynamic features (computed as obs come in):
    err_mean_sofar: Mean (obs - forecast) bias
    err_std_sofar: Std of forecast errors
    err_max_pos_sofar: Maximum positive error (obs warmer than fcst)
    err_max_neg_sofar: Maximum negative error (obs cooler than fcst)
    err_abs_mean_sofar: Mean absolute error
    err_last1h: Most recent error
    err_last3h_mean: Mean error over last 3 hours

Example:
    >>> fcst_temps = [70, 75, 82, 85, 83, 78]  # Hourly forecast
    >>> fs = compute_forecast_static_features(fcst_temps)
    >>> fs['fcst_prev_max_f']
    85.0
    >>> fs['t_forecast_base']
    85
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from models.features.base import FeatureSet, register_feature_group


@register_feature_group("forecast_static")
def compute_forecast_static_features(
    fcst_series: list[float],
) -> FeatureSet:
    """Compute features from T-1 forecast (no observations needed).

    These features represent what Visual Crossing predicted yesterday
    about today's temperatures. They're available at the start of the
    day and don't change as observations come in.

    Args:
        fcst_series: List of hourly (or sub-hourly) forecast temps for
                     the target day, from the T-1 forecast

    Returns:
        FeatureSet with forecast statistics
    """
    if not fcst_series:
        return FeatureSet(name="forecast_static", features={
            "fcst_prev_max_f": None,
            "fcst_prev_min_f": None,
            "fcst_prev_mean_f": None,
            "fcst_prev_std_f": None,
            "fcst_prev_q10_f": None,
            "fcst_prev_q25_f": None,
            "fcst_prev_q50_f": None,
            "fcst_prev_q75_f": None,
            "fcst_prev_q90_f": None,
            "fcst_prev_frac_part": None,
            "fcst_prev_hour_of_max": None,
            "t_forecast_base": None,
        })

    arr = np.asarray(fcst_series, dtype=np.float64)

    # Basic statistics
    max_f = float(arr.max())
    min_f = float(arr.min())
    mean_f = float(arr.mean())
    std_f = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    # Percentiles
    q10, q25, q50, q75, q90 = np.percentile(arr, [10, 25, 50, 75, 90])

    # Fractional part of max
    frac_part = max_f - round(max_f)

    # Hour of max (index in the series)
    hour_of_max = int(np.argmax(arr))

    # Baseline from forecast
    t_forecast_base = int(round(max_f))

    features = {
        "fcst_prev_max_f": max_f,
        "fcst_prev_min_f": min_f,
        "fcst_prev_mean_f": mean_f,
        "fcst_prev_std_f": std_f,
        "fcst_prev_q10_f": float(q10),
        "fcst_prev_q25_f": float(q25),
        "fcst_prev_q50_f": float(q50),
        "fcst_prev_q75_f": float(q75),
        "fcst_prev_q90_f": float(q90),
        "fcst_prev_frac_part": frac_part,
        "fcst_prev_hour_of_max": hour_of_max,
        "t_forecast_base": t_forecast_base,
    }

    return FeatureSet(name="forecast_static", features=features)


@register_feature_group("forecast_error")
def compute_forecast_error_features(
    fcst_series_sofar: list[float],
    obs_series_sofar: list[float],
) -> FeatureSet:
    """Compute forecast-vs-actual deltas up to snapshot time.

    These features capture how well the T-1 forecast has performed
    so far today. Positive errors mean observations are warmer than
    forecast; negative means cooler.

    Args:
        fcst_series_sofar: Forecast temps for hours up to τ
        obs_series_sofar: Actual observed temps for hours up to τ
                          (should be aligned with forecast by hour)

    Returns:
        FeatureSet with forecast error statistics
    """
    # Handle mismatched lengths by truncating to shorter
    n = min(len(fcst_series_sofar), len(obs_series_sofar))

    if n == 0:
        return FeatureSet(name="forecast_error", features={
            "err_mean_sofar": None,
            "err_std_sofar": None,
            "err_max_pos_sofar": None,
            "err_max_neg_sofar": None,
            "err_abs_mean_sofar": None,
            "err_last1h": None,
            "err_last3h_mean": None,
        })

    fcst = np.asarray(fcst_series_sofar[:n], dtype=np.float64)
    obs = np.asarray(obs_series_sofar[:n], dtype=np.float64)

    # Error = obs - forecast (positive = warmer than expected)
    err = obs - fcst

    features = {
        "err_mean_sofar": float(err.mean()),
        "err_std_sofar": float(err.std(ddof=1)) if n > 1 else 0.0,
        "err_max_pos_sofar": float(err.max()),
        "err_max_neg_sofar": float(err.min()),
        "err_abs_mean_sofar": float(np.abs(err).mean()),
    }

    # Recent errors
    if n >= 1:
        features["err_last1h"] = float(err[-1])
    else:
        features["err_last1h"] = None

    if n >= 3:
        features["err_last3h_mean"] = float(err[-3:].mean())
    else:
        features["err_last3h_mean"] = None

    return FeatureSet(name="forecast_error", features=features)


def align_forecast_to_observations(
    fcst_hourly_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    obs_datetime_col: str = "datetime_local",
    fcst_datetime_col: str = "target_datetime_local",
) -> tuple[list[float], list[float]]:
    """Align forecast and observations by hour for comparison.

    Visual Crossing forecasts are hourly while observations are 5-minute.
    This function aggregates observations to hourly means and aligns
    them with forecast values.

    Args:
        fcst_hourly_df: DataFrame with hourly forecasts (temp_f column)
        obs_df: DataFrame with 5-minute observations (temp_f column)
        obs_datetime_col: Column name for observation timestamps
        fcst_datetime_col: Column name for forecast timestamps

    Returns:
        Tuple of (forecast_temps, observed_temps) aligned by hour
    """
    if fcst_hourly_df is None or fcst_hourly_df.empty:
        return [], []

    if obs_df is None or obs_df.empty:
        return [], []

    # Ensure datetime columns are datetime type
    fcst_df = fcst_hourly_df.copy()
    obs_df = obs_df.copy()

    fcst_df[fcst_datetime_col] = pd.to_datetime(fcst_df[fcst_datetime_col])
    obs_df[obs_datetime_col] = pd.to_datetime(obs_df[obs_datetime_col])

    # Extract hour from both
    fcst_df["hour"] = fcst_df[fcst_datetime_col].dt.hour
    obs_df["hour"] = obs_df[obs_datetime_col].dt.hour

    # Aggregate observations to hourly mean
    obs_hourly = obs_df.groupby("hour")["temp_f"].mean().reset_index()
    obs_hourly.columns = ["hour", "obs_temp_f"]

    # Get forecast temps by hour
    fcst_hourly = fcst_df[["hour", "temp_f"]].copy()
    fcst_hourly.columns = ["hour", "fcst_temp_f"]

    # Merge on hour
    merged = pd.merge(fcst_hourly, obs_hourly, on="hour", how="inner")

    if merged.empty:
        return [], []

    # Sort by hour and return lists
    merged = merged.sort_values("hour")

    fcst_temps = merged["fcst_temp_f"].dropna().tolist()
    obs_temps = merged["obs_temp_f"].dropna().tolist()

    return fcst_temps, obs_temps


def compute_forecast_delta_features(
    fcst_max_f: Optional[float],
    obs_max_sofar: float,
) -> FeatureSet:
    """Compute simple delta between forecast max and observed max so far.

    Args:
        fcst_max_f: Forecast daily high from T-1
        obs_max_sofar: Maximum observed temperature so far

    Returns:
        FeatureSet with delta features
    """
    if fcst_max_f is None:
        return FeatureSet(name="forecast_delta", features={
            "delta_vcmax_fcstmax_sofar": None,
            "fcst_remaining_potential": None,
        })

    # How much has observed exceeded forecast so far?
    delta = obs_max_sofar - fcst_max_f

    # Remaining potential: how much higher could it still go if forecast is right?
    # (Only meaningful if obs is still below forecast max)
    remaining = max(0.0, fcst_max_f - obs_max_sofar)

    features = {
        "delta_vcmax_fcstmax_sofar": delta,
        "fcst_remaining_potential": remaining,
    }

    return FeatureSet(name="forecast_delta", features=features)
