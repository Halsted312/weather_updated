"""
Shape-of-day features for spike vs plateau detection.

This module analyzes the temporal pattern of temperatures to distinguish:
- Sustained plateaus (high confidence in current max being final)
- Brief spikes (may see higher temps later in the day)

These features are particularly useful for identifying days where the
observed max may not be the final settlement temperature.

Features computed:
    minutes_ge_base: Total minutes >= t_base
    minutes_ge_base_p1: Total minutes >= t_base + 1
    minutes_ge_base_m1: Total minutes >= t_base - 1
    max_run_ge_base: Longest consecutive run >= t_base (minutes)
    max_run_ge_base_p1: Longest consecutive run >= t_base + 1
    max_run_ge_base_m1: Longest consecutive run >= t_base - 1
    max_minus_second_max: Difference between max and 2nd highest (spike indicator)
    max_morning_f_sofar: Max temp before noon
    max_afternoon_f_sofar: Max temp 12:00-17:00
    max_evening_f_sofar: Max temp after 17:00
    slope_max_30min_up_sofar: Maximum 30-min upward slope
    slope_max_30min_down_sofar: Maximum 30-min downward slope (negative)

Example:
    >>> temps = [72, 75, 80, 82, 82, 82, 82, 81, 79]  # Plateau at 82
    >>> ts = [datetime(2024,7,1,h) for h in range(9, 18)]
    >>> fs = compute_shape_features(temps, ts, t_base=82)
    >>> fs['max_run_ge_base']  # 4 consecutive samples = 20 minutes
    20
"""

from datetime import datetime
from typing import Optional

import numpy as np

from models.features.base import FeatureSet, register_feature_group


@register_feature_group("shape")
def compute_shape_features(
    temps_sofar: list[float],
    timestamps_local_sofar: list[datetime],
    t_base: int,
    step_minutes: int = 5,
) -> FeatureSet:
    """Compute plateau and shape features from partial-day temperatures.

    These features capture the "shape" of the temperature curve - whether
    temps are sustained at high levels (plateau) or show brief spikes.
    A plateau pattern suggests higher confidence that the current max
    will be the final settlement.

    Args:
        temps_sofar: List of temperatures in °F observed up to τ
        timestamps_local_sofar: Corresponding local timestamps
        t_base: Baseline temperature (rounded max) for comparison
        step_minutes: Time between samples (default 5 for VC data)

    Returns:
        FeatureSet with shape features. Empty if insufficient data.
    """
    if not temps_sofar or len(temps_sofar) < 2:
        return FeatureSet(name="shape", features={})

    arr = np.asarray(temps_sofar, dtype=np.float64)
    n = len(arr)

    features = {}

    # Minutes at or above various thresholds
    features["minutes_ge_base"] = _minutes_ge_threshold(arr, t_base, step_minutes)
    features["minutes_ge_base_p1"] = _minutes_ge_threshold(arr, t_base + 1, step_minutes)
    features["minutes_ge_base_m1"] = _minutes_ge_threshold(arr, t_base - 1, step_minutes)

    # Longest consecutive run at or above thresholds
    features["max_run_ge_base"] = _max_run_ge_threshold(arr, t_base, step_minutes)
    features["max_run_ge_base_p1"] = _max_run_ge_threshold(arr, t_base + 1, step_minutes)
    features["max_run_ge_base_m1"] = _max_run_ge_threshold(arr, t_base - 1, step_minutes)

    # Spike indicator: difference between max and second-highest value
    if n >= 2:
        sorted_temps = np.sort(arr)[::-1]  # Descending
        features["max_minus_second_max"] = float(sorted_temps[0] - sorted_temps[1])
    else:
        features["max_minus_second_max"] = 0.0

    # Time-of-day maximums (if timestamps available)
    if timestamps_local_sofar:
        hours = np.array([ts.hour for ts in timestamps_local_sofar])

        # Morning: before noon
        morning_mask = hours < 12
        if morning_mask.any():
            features["max_morning_f_sofar"] = float(arr[morning_mask].max())
        else:
            features["max_morning_f_sofar"] = None

        # Afternoon: 12:00 - 17:00
        afternoon_mask = (hours >= 12) & (hours < 17)
        if afternoon_mask.any():
            features["max_afternoon_f_sofar"] = float(arr[afternoon_mask].max())
        else:
            features["max_afternoon_f_sofar"] = None

        # Evening: after 17:00
        evening_mask = hours >= 17
        if evening_mask.any():
            features["max_evening_f_sofar"] = float(arr[evening_mask].max())
        else:
            features["max_evening_f_sofar"] = None
    else:
        features["max_morning_f_sofar"] = None
        features["max_afternoon_f_sofar"] = None
        features["max_evening_f_sofar"] = None

    # 30-minute slopes (momentum indicators)
    window = 30 // step_minutes  # Number of samples in 30 minutes
    if n >= window + 1:
        diffs = arr[window:] - arr[:-window]  # Change over 30-min window
        features["slope_max_30min_up_sofar"] = float(diffs.max())
        features["slope_max_30min_down_sofar"] = float(diffs.min())
    else:
        features["slope_max_30min_up_sofar"] = None
        features["slope_max_30min_down_sofar"] = None

    return FeatureSet(name="shape", features=features)


def _minutes_ge_threshold(
    arr: np.ndarray,
    threshold: float,
    step_minutes: int,
) -> int:
    """Count total minutes with temperature >= threshold."""
    count = int((arr >= threshold).sum())
    return count * step_minutes


def _max_run_ge_threshold(
    arr: np.ndarray,
    threshold: float,
    step_minutes: int,
) -> int:
    """Find longest consecutive run with temperature >= threshold."""
    best_run = 0
    current_run = 0

    for temp in arr:
        if temp >= threshold:
            current_run += 1
            best_run = max(best_run, current_run)
        else:
            current_run = 0

    return best_run * step_minutes


def compute_temperature_momentum(
    temps_sofar: list[float],
    step_minutes: int = 5,
) -> FeatureSet:
    """Compute momentum features showing temperature trajectory.

    These features indicate whether temperatures are rising, falling,
    or stable, which can be predictive of whether we'll see higher
    temps later in the day.

    Args:
        temps_sofar: List of temperatures
        step_minutes: Time between samples

    Returns:
        FeatureSet with momentum indicators
    """
    if len(temps_sofar) < 3:
        return FeatureSet(name="momentum", features={})

    arr = np.asarray(temps_sofar, dtype=np.float64)

    # Recent temperature change
    if len(arr) >= 12:  # 1 hour of data
        last_hour = arr[-12:]
        features = {
            "temp_change_last_1h": float(last_hour[-1] - last_hour[0]),
            "temp_max_last_1h": float(last_hour.max()),
            "temp_min_last_1h": float(last_hour.min()),
        }
    else:
        features = {
            "temp_change_last_1h": None,
            "temp_max_last_1h": None,
            "temp_min_last_1h": None,
        }

    # Trend direction (simple: compare first half to second half)
    mid = len(arr) // 2
    if mid > 0:
        first_half_mean = arr[:mid].mean()
        second_half_mean = arr[mid:].mean()
        features["trend_direction"] = float(second_half_mean - first_half_mean)
    else:
        features["trend_direction"] = None

    return FeatureSet(name="momentum", features=features)
