"""
Momentum and rate-of-change features for temperature Δ-models.

This module computes features that capture temperature trajectory and momentum,
including rolling windows, exponential decay weighted averages, and derivatives.

These features are critical for predicting whether temperature is still rising
or has plateaued - a key signal for final settlement.

Features computed:
    Rolling statistics (30min, 60min, 120min windows):
        temp_mean_last_30min, temp_std_last_30min, temp_max_last_30min
        temp_mean_last_60min, temp_std_last_60min, temp_max_last_60min
        temp_mean_last_120min, temp_std_last_120min

    Rate of change (°F per hour):
        temp_rate_last_30min: Slope over last 30 minutes
        temp_rate_last_60min: Slope over last 60 minutes
        temp_acceleration: Change in rate (second derivative)

    Exponential decay weighted:
        temp_ema_30min: Exponential moving average (30min half-life)
        temp_ema_60min: Exponential moving average (60min half-life)

    Time since max:
        minutes_since_max_observed: Minutes since daily max was hit

Example:
    >>> temps = [(t, T) for t, T in zip(timestamps, temperatures)]
    >>> fs = compute_momentum_features(temps, timestamps)
    >>> fs['temp_rate_last_30min']  # °F per hour
    2.5
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from models.features.base import FeatureSet, register_feature_group


def _get_temps_in_window(
    temps_with_times: list[tuple[datetime, float]],
    window_minutes: int,
) -> list[float]:
    """Extract temps within the last N minutes.

    Args:
        temps_with_times: List of (datetime, temp_f) tuples, sorted chronologically
        window_minutes: Size of lookback window in minutes

    Returns:
        List of temps within the window
    """
    if not temps_with_times:
        return []

    last_time = temps_with_times[-1][0]
    cutoff = last_time - timedelta(minutes=window_minutes)

    return [t for dt, t in temps_with_times if dt >= cutoff]


def _compute_rate(
    temps_with_times: list[tuple[datetime, float]],
    window_minutes: int,
) -> Optional[float]:
    """Compute temperature rate of change in °F per hour.

    Uses linear regression over the window for robustness.

    Args:
        temps_with_times: List of (datetime, temp_f) tuples
        window_minutes: Size of lookback window

    Returns:
        Rate in °F/hour, or None if insufficient data
    """
    if not temps_with_times or len(temps_with_times) < 2:
        return None

    last_time = temps_with_times[-1][0]
    cutoff = last_time - timedelta(minutes=window_minutes)

    # Get points in window
    window_points = [(dt, t) for dt, t in temps_with_times if dt >= cutoff]

    if len(window_points) < 2:
        return None

    # Convert to minutes from start and fit line
    start_time = window_points[0][0]
    x = np.array([(dt - start_time).total_seconds() / 60.0 for dt, _ in window_points])
    y = np.array([t for _, t in window_points])

    # Simple linear regression: slope = cov(x,y) / var(x)
    if x.std() < 0.001:  # No time variation
        return 0.0

    slope_per_min = np.cov(x, y)[0, 1] / np.var(x)
    slope_per_hour = slope_per_min * 60.0  # Convert to °F/hour

    return float(slope_per_hour)


def _compute_ema(
    temps_with_times: list[tuple[datetime, float]],
    halflife_minutes: float,
) -> Optional[float]:
    """Compute exponential moving average with given half-life.

    More recent observations get higher weight, with weight halving
    every halflife_minutes.

    Args:
        temps_with_times: List of (datetime, temp_f) tuples
        halflife_minutes: Half-life for decay in minutes

    Returns:
        EMA value, or None if no data
    """
    if not temps_with_times:
        return None

    last_time = temps_with_times[-1][0]
    decay_rate = np.log(2) / halflife_minutes

    weights = []
    temps = []

    for dt, temp in temps_with_times:
        age_minutes = (last_time - dt).total_seconds() / 60.0
        weight = np.exp(-decay_rate * age_minutes)
        weights.append(weight)
        temps.append(temp)

    weights = np.array(weights)
    temps = np.array(temps)

    if weights.sum() < 0.001:
        return None

    return float(np.average(temps, weights=weights))


@register_feature_group("momentum")
def compute_momentum_features(
    temps_with_times: list[tuple[datetime, float]],
) -> FeatureSet:
    """Compute momentum and rate-of-change features.

    Args:
        temps_with_times: List of (datetime_local, temp_f) tuples,
                         sorted chronologically. Timestamps should be
                         naive or local time.

    Returns:
        FeatureSet with momentum statistics
    """
    if not temps_with_times or len(temps_with_times) < 2:
        return FeatureSet(name="momentum", features={
            "temp_mean_last_30min": None,
            "temp_std_last_30min": None,
            "temp_max_last_30min": None,
            "temp_mean_last_60min": None,
            "temp_std_last_60min": None,
            "temp_max_last_60min": None,
            "temp_mean_last_120min": None,
            "temp_std_last_120min": None,
            "temp_rate_last_30min": None,
            "temp_rate_last_60min": None,
            "temp_acceleration": None,
            "temp_ema_30min": None,
            "temp_ema_60min": None,
            "minutes_since_max_observed": None,
        })

    features = {}

    # Rolling window statistics
    for window in [30, 60, 120]:
        window_temps = _get_temps_in_window(temps_with_times, window)

        if window_temps:
            arr = np.array(window_temps)
            features[f"temp_mean_last_{window}min"] = float(arr.mean())
            features[f"temp_std_last_{window}min"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            if window <= 60:  # Only compute max for shorter windows
                features[f"temp_max_last_{window}min"] = float(arr.max())
        else:
            features[f"temp_mean_last_{window}min"] = None
            features[f"temp_std_last_{window}min"] = None
            if window <= 60:
                features[f"temp_max_last_{window}min"] = None

    # Rate of change
    rate_30 = _compute_rate(temps_with_times, 30)
    rate_60 = _compute_rate(temps_with_times, 60)

    features["temp_rate_last_30min"] = rate_30
    features["temp_rate_last_60min"] = rate_60

    # Acceleration (change in rate)
    # Compare rate over recent 30min vs rate over prior 30min
    if rate_30 is not None and len(temps_with_times) >= 12:  # Need ~60 min of data
        # Get rate from 30-60 min ago
        midpoint_time = temps_with_times[-1][0] - timedelta(minutes=30)
        earlier_temps = [(dt, t) for dt, t in temps_with_times if dt <= midpoint_time]
        rate_prev_30 = _compute_rate(earlier_temps, 30) if earlier_temps else None

        if rate_prev_30 is not None:
            features["temp_acceleration"] = rate_30 - rate_prev_30
        else:
            features["temp_acceleration"] = None
    else:
        features["temp_acceleration"] = None

    # Exponential moving averages
    features["temp_ema_30min"] = _compute_ema(temps_with_times, 30)
    features["temp_ema_60min"] = _compute_ema(temps_with_times, 60)

    # Minutes since max
    if temps_with_times:
        temps = [t for _, t in temps_with_times]
        max_temp = max(temps)

        # Find last time max was hit
        for dt, t in reversed(temps_with_times):
            if abs(t - max_temp) < 0.01:  # Allow small float tolerance
                last_time = temps_with_times[-1][0]
                minutes_since = (last_time - dt).total_seconds() / 60.0
                features["minutes_since_max_observed"] = float(minutes_since)
                break
        else:
            features["minutes_since_max_observed"] = None
    else:
        features["minutes_since_max_observed"] = None

    return FeatureSet(name="momentum", features=features)


def compute_volatility_features(
    temps_with_times: list[tuple[datetime, float]],
) -> FeatureSet:
    """Compute volatility and range-based features.

    These features capture uncertainty and variability in the
    temperature signal.

    Args:
        temps_with_times: List of (datetime_local, temp_f) tuples

    Returns:
        FeatureSet with volatility statistics
    """
    if not temps_with_times or len(temps_with_times) < 2:
        return FeatureSet(name="volatility", features={
            "temp_volatility_30min": None,
            "temp_volatility_60min": None,
            "intraday_range_sofar": None,
            "temp_cv_sofar": None,
        })

    features = {}

    # Rolling volatility (std of rolling window)
    for window in [30, 60]:
        window_temps = _get_temps_in_window(temps_with_times, window)
        if len(window_temps) >= 2:
            features[f"temp_volatility_{window}min"] = float(np.std(window_temps, ddof=1))
        else:
            features[f"temp_volatility_{window}min"] = None

    # Full day range
    all_temps = [t for _, t in temps_with_times]
    features["intraday_range_sofar"] = float(max(all_temps) - min(all_temps))

    # Coefficient of variation
    mean_temp = np.mean(all_temps)
    if mean_temp > 0:
        features["temp_cv_sofar"] = float(np.std(all_temps, ddof=1) / mean_temp)
    else:
        features["temp_cv_sofar"] = None

    return FeatureSet(name="volatility", features=features)
