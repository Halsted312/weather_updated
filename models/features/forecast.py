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
            # Feature Group 1
            "fcst_prev_distance_to_int": None,
            "fcst_prev_near_boundary_flag": None,
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

    # Fractional part of max (original: can be negative)
    frac_part = max_f - round(max_f)

    # === Feature Group 1: Integer boundary features ===
    # Raw fractional part in [0, 1)
    raw_frac = max_f - np.floor(max_f)
    # Distance to nearest integer (0 = on boundary, 0.5 = equidistant)
    distance_to_int = min(raw_frac, 1.0 - raw_frac)
    # Near boundary flag (within 0.25°F of integer)
    near_boundary_flag = float(distance_to_int < 0.25)

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
        # Feature Group 1
        "fcst_prev_distance_to_int": distance_to_int,
        "fcst_prev_near_boundary_flag": near_boundary_flag,
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
    """Compute forecast remaining potential feature.

    NOTE: delta_vcmax_fcstmax_sofar was removed - it duplicates obs_fcst_max_gap
    which is computed in compute_derived_forecast_features().

    Args:
        fcst_max_f: Forecast daily high from T-1
        obs_max_sofar: Maximum observed temperature so far

    Returns:
        FeatureSet with fcst_remaining_potential
    """
    if fcst_max_f is None:
        return FeatureSet(name="forecast_delta", features={
            "fcst_remaining_potential": None,
        })

    # Remaining potential: how much higher could it still go if forecast is right?
    # (Only meaningful if obs is still below forecast max)
    remaining = max(0.0, fcst_max_f - obs_max_sofar)

    features = {
        "fcst_remaining_potential": remaining,
    }

    return FeatureSet(name="forecast_delta", features=features)


# =============================================================================
# Feature Group 2: Peak Window Features
# =============================================================================

@register_feature_group("forecast_peak_window")
def compute_forecast_peak_window_features(
    temps_f: list[float],
    timestamps: list[datetime],
    step_minutes: int = 60,
    peak_band_width_f: float = 1.0,
) -> FeatureSet:
    """Compress forecast curve (hourly or 15-min) into peak-timing features.

    Extracts information about when the peak occurs and how long it lasts.

    Args:
        temps_f: List of forecast temperatures (hourly or 15-min)
        timestamps: List of corresponding datetime objects
        step_minutes: Time between samples (15 or 60)
        peak_band_width_f: Width of "near peak" band in °F (default 1.0)

    Returns:
        FeatureSet with:
            fcst_peak_temp_f: Max temp from forecast curve
            fcst_peak_hour_float: Hour of max as float (14.25 = 2:15pm)
            fcst_peak_band_width_min: Duration within 1°F of max (plateau detection)
            fcst_peak_step_minutes: Resolution of input curve (15 or 60)
    """
    null_features = {
        "fcst_peak_temp_f": None,
        "fcst_peak_hour_float": None,
        "fcst_peak_band_width_min": None,
        "fcst_peak_step_minutes": None,
    }

    if not temps_f or not timestamps or len(temps_f) != len(timestamps):
        return FeatureSet(name="forecast_peak_window", features=null_features)

    arr = np.asarray(temps_f, dtype=np.float64)
    tmax = float(arr.max())
    idx_max = int(np.argmax(arr))
    ts_max = timestamps[idx_max]

    # Hour as float (14.25 = 2:15pm)
    minutes_since_midnight = ts_max.hour * 60 + ts_max.minute + ts_max.second / 60.0
    hour_of_max_float = minutes_since_midnight / 60.0

    # Peak band duration (how long within peak_band_width_f of max)
    within_band = np.where(arr >= tmax - peak_band_width_f)[0]
    if within_band.size > 0:
        duration_minutes = (within_band[-1] - within_band[0] + 1) * step_minutes
    else:
        duration_minutes = step_minutes

    features = {
        "fcst_peak_temp_f": tmax,
        "fcst_peak_hour_float": hour_of_max_float,
        "fcst_peak_band_width_min": float(duration_minutes),
        "fcst_peak_step_minutes": float(step_minutes),
    }

    return FeatureSet(name="forecast_peak_window", features=features)


# =============================================================================
# Feature Group 3: Forecast Drift Features
# =============================================================================

@register_feature_group("forecast_drift")
def compute_forecast_drift_features(
    daily_multi_df: pd.DataFrame,
) -> FeatureSet:
    """Compress multi-lead daily highs into drift/volatility features.

    Measures how the forecast for a given day has changed across
    different lead times (T-6, T-5, ..., T-1, T-0). High volatility
    suggests uncertain forecasts.

    Args:
        daily_multi_df: DataFrame with columns ['lead_days', 'tempmax_f']
                        from multiple forecast leads for the same target date

    Returns:
        FeatureSet with:
            fcst_drift_num_leads: Number of lead forecasts available
            fcst_drift_std_f: Standard deviation of forecast highs across leads
            fcst_drift_max_upside_f: Max positive deviation from T-1 forecast
            fcst_drift_max_downside_f: Max negative deviation from T-1 forecast
            fcst_drift_mean_delta_f: Mean deviation from T-1 forecast
            fcst_drift_slope_f_per_lead: Linear trend (°F per lead day)
    """
    null_features = {
        "fcst_drift_num_leads": None,
        "fcst_drift_std_f": None,
        "fcst_drift_max_upside_f": None,
        "fcst_drift_max_downside_f": None,
        "fcst_drift_mean_delta_f": None,
        "fcst_drift_slope_f_per_lead": None,
    }

    if daily_multi_df is None or daily_multi_df.empty:
        return FeatureSet(name="forecast_drift", features=null_features)

    df = daily_multi_df.dropna(subset=["tempmax_f"]).copy()
    if df.empty:
        return FeatureSet(name="forecast_drift", features=null_features)

    df = df.sort_values("lead_days")
    leads = df["lead_days"].to_numpy(dtype=np.float64)
    highs = df["tempmax_f"].to_numpy(dtype=np.float64)

    # Anchor at T-1 (lead=1) if present, else closest-in lead
    mask_t1 = df["lead_days"] == 1
    if mask_t1.any():
        anchor_high = float(df.loc[mask_t1, "tempmax_f"].iloc[0])
    else:
        # Use the smallest lead (closest to target)
        anchor_high = float(highs[0])

    deltas = highs - anchor_high

    features = {
        "fcst_drift_num_leads": float(len(highs)),
        "fcst_drift_std_f": float(np.std(highs, ddof=1)) if len(highs) > 1 else 0.0,
        "fcst_drift_max_upside_f": float(np.max(highs) - anchor_high),
        "fcst_drift_max_downside_f": float(anchor_high - np.min(highs)),
        "fcst_drift_mean_delta_f": float(np.mean(deltas)),
    }

    # Linear slope (negative slope = forecasts decreasing as we get closer)
    if len(highs) >= 2:
        slope, _ = np.polyfit(leads, highs, deg=1)
        features["fcst_drift_slope_f_per_lead"] = float(slope)
    else:
        features["fcst_drift_slope_f_per_lead"] = 0.0

    return FeatureSet(name="forecast_drift", features=features)


# =============================================================================
# Feature Group 4: Multivar Static Features
# =============================================================================

@register_feature_group("forecast_multivar_static")
def compute_forecast_multivar_static_features(
    minute_df: pd.DataFrame,
) -> FeatureSet:
    """Day-level aggregates for humidity, cloudcover, dewpoint from forecast.

    Computes simple statistics from the 15-min or hourly forecast curve
    for multivariate weather features (not just temperature).

    Note: Cloudcover is NOT available in minute-level API, so pass hourly
    data if cloudcover features are needed.

    Args:
        minute_df: DataFrame with columns including:
            datetime_local, temp_f, humidity, dew_f, cloudcover

    Returns:
        FeatureSet with aggregated multivar features
    """
    null_features = {
        "fcst_humidity_mean": None,
        "fcst_humidity_min": None,
        "fcst_humidity_max": None,
        "fcst_humidity_range": None,
        "fcst_cloudcover_mean": None,
        "fcst_cloudcover_min": None,
        "fcst_cloudcover_max": None,
        "fcst_cloudcover_range": None,
        "fcst_dewpoint_mean": None,
        "fcst_dewpoint_min": None,
        "fcst_dewpoint_max": None,
        "fcst_dewpoint_range": None,
        "fcst_humidity_morning_mean": None,
        "fcst_humidity_afternoon_mean": None,
    }

    if minute_df is None or minute_df.empty:
        return FeatureSet(name="forecast_multivar_static", features=null_features)

    df = minute_df.copy()

    # Ensure datetime_local is datetime type
    if "datetime_local" in df.columns:
        df["datetime_local"] = pd.to_datetime(df["datetime_local"])
        df["hour"] = df["datetime_local"].dt.hour
    elif "target_datetime_local" in df.columns:
        # Support hourly forecast format
        df["datetime_local"] = pd.to_datetime(df["target_datetime_local"])
        df["hour"] = df["datetime_local"].dt.hour
    else:
        # No datetime column, can't compute AM/PM
        df["hour"] = None

    def _stats(series: pd.Series):
        """Compute mean, min, max, range for a series."""
        s = series.dropna()
        if s.empty:
            return None, None, None, None
        return float(s.mean()), float(s.min()), float(s.max()), float(s.max() - s.min())

    # Get series with fallback for missing columns
    humidity_series = df.get("humidity", pd.Series(dtype=float))
    cloudcover_series = df.get("cloudcover", pd.Series(dtype=float))
    dew_series = df.get("dew_f", pd.Series(dtype=float))

    hum_mean, hum_min, hum_max, hum_range = _stats(humidity_series)
    cc_mean, cc_min, cc_max, cc_range = _stats(cloudcover_series)
    dew_mean, dew_min, dew_max, dew_range = _stats(dew_series)

    # Morning vs afternoon humidity
    am_hum_mean = None
    pm_hum_mean = None
    if df["hour"] is not None and "humidity" in df.columns:
        am = df[df["hour"].between(6, 11)]
        pm = df[df["hour"].between(12, 18)]
        if not am.empty and "humidity" in am.columns:
            am_hum = am["humidity"].dropna()
            if not am_hum.empty:
                am_hum_mean = float(am_hum.mean())
        if not pm.empty and "humidity" in pm.columns:
            pm_hum = pm["humidity"].dropna()
            if not pm_hum.empty:
                pm_hum_mean = float(pm_hum.mean())

    features = {
        "fcst_humidity_mean": hum_mean,
        "fcst_humidity_min": hum_min,
        "fcst_humidity_max": hum_max,
        "fcst_humidity_range": hum_range,
        "fcst_cloudcover_mean": cc_mean,
        "fcst_cloudcover_min": cc_min,
        "fcst_cloudcover_max": cc_max,
        "fcst_cloudcover_range": cc_range,
        "fcst_dewpoint_mean": dew_mean,
        "fcst_dewpoint_min": dew_min,
        "fcst_dewpoint_max": dew_max,
        "fcst_dewpoint_range": dew_range,
        "fcst_humidity_morning_mean": am_hum_mean,
        "fcst_humidity_afternoon_mean": pm_hum_mean,
    }

    return FeatureSet(name="forecast_multivar_static", features=features)


# =============================================================================
# Multi-Horizon Features (from load_multi_horizon_forecasts)
# =============================================================================

@register_feature_group("forecast_multi_horizon")
def compute_multi_horizon_features(
    fcst_multi: dict[int, Optional[dict]],
) -> FeatureSet:
    """Compute features from multi-lead forecasts (T-1 to T-14).

    Captures forecast evolution, stability, and trend by comparing forecasts
    from different lead times for the same target day.

    Args:
        fcst_multi: Dict mapping lead_day → forecast_dict
                    Each forecast_dict has 'tempmax_f', etc.

    Returns:
        FeatureSet with 7 features:
            fcst_multi_mean: Simple average of tmax across T-1 to T-14
            fcst_multi_median: Median (robust to outliers)
            fcst_multi_ema: Exponential moving average weighted toward T-1
            fcst_multi_std: Std dev across T-1 to T-14
            fcst_multi_range: Max - Min (forecast uncertainty range)
            fcst_multi_t1_t2_diff: T-1 minus T-2 (most recent change)
            fcst_multi_drift: T-1 minus T-6 (long-term trend)
    """
    null_features = {
        "fcst_multi_mean": None,
        "fcst_multi_median": None,
        "fcst_multi_ema": None,
        "fcst_multi_std": None,
        "fcst_multi_range": None,
        "fcst_multi_t1_t2_diff": None,
        "fcst_multi_drift": None,
    }

    if not fcst_multi:
        return FeatureSet(name="forecast_multi_horizon", features=null_features)

    # Extract tempmax_f values from each lead (sorted)
    highs = []
    for lead in sorted(fcst_multi.keys()):
        fcst = fcst_multi[lead]
        if fcst is not None and fcst.get("tempmax_f") is not None:
            highs.append(fcst["tempmax_f"])

    if not highs:
        return FeatureSet(name="forecast_multi_horizon", features=null_features)

    arr = np.asarray(highs, dtype=np.float64)

    # Group 1: Central tendency
    mean_val = float(arr.mean())
    median_val = float(np.median(arr))

    # Exponential moving average (weighted toward T-1)
    leads_present = sorted([k for k in fcst_multi.keys() if fcst_multi[k] is not None and fcst_multi[k].get("tempmax_f") is not None])
    if leads_present:
        weights = np.exp(-0.15 * (np.array(leads_present) - 1))
        weights /= weights.sum()
        values = np.array([fcst_multi[k]["tempmax_f"] for k in leads_present])
        ema_val = float(np.sum(weights * values))
    else:
        ema_val = None

    # Group 2: Dispersion
    std_val = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    range_val = float(arr.max() - arr.min())

    # Group 3: Evolution
    t1_val = fcst_multi.get(1, {}).get("tempmax_f") if fcst_multi.get(1) else None
    t2_val = fcst_multi.get(2, {}).get("tempmax_f") if fcst_multi.get(2) else None
    t6_val = fcst_multi.get(6, {}).get("tempmax_f") if fcst_multi.get(6) else None

    t1_t2_diff = (t1_val - t2_val) if (t1_val is not None and t2_val is not None) else None
    drift = (t1_val - t6_val) if (t1_val is not None and t6_val is not None) else None

    features = {
        "fcst_multi_mean": mean_val,
        "fcst_multi_median": median_val,
        "fcst_multi_ema": ema_val,
        "fcst_multi_std": std_val,
        "fcst_multi_range": range_val,
        "fcst_multi_t1_t2_diff": t1_t2_diff,
        "fcst_multi_drift": drift,
    }

    return FeatureSet(name="forecast_multi_horizon", features=features)
