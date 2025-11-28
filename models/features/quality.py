"""
Data quality features for temperature Δ-models.

This module computes features that indicate potential data quality issues
which may affect prediction reliability. Low-quality data windows should
result in wider prediction uncertainty.

Features computed:
    missing_fraction_sofar: Fraction of expected samples missing
    max_gap_minutes: Largest gap between consecutive observations
    edge_max_flag: 1 if max temp is at start/end of window (incomplete)

Example:
    >>> temps = [72.1, 75.3, 80.2]  # 3 samples in a window expecting 12
    >>> ts = [datetime(2024,7,1,h) for h in [9, 10, 12]]  # 2-hour gap
    >>> fs = compute_quality_features(temps, ts, expected_samples=12)
    >>> fs['missing_fraction_sofar']
    0.75
    >>> fs['max_gap_minutes']
    120
"""

from datetime import datetime
from typing import Optional

import numpy as np

from models.features.base import FeatureSet, register_feature_group


@register_feature_group("quality")
def compute_quality_features(
    temps_sofar: list[float],
    timestamps_sofar: list[datetime],
    expected_samples: Optional[int] = None,
    step_minutes: int = 5,
) -> FeatureSet:
    """Compute data quality indicators.

    These features help the model understand when predictions may be
    less reliable due to missing or sparse data.

    Args:
        temps_sofar: List of observed temperatures
        timestamps_sofar: Corresponding timestamps
        expected_samples: Expected number of samples (if known)
        step_minutes: Expected interval between samples

    Returns:
        FeatureSet with quality indicators
    """
    if not temps_sofar or not timestamps_sofar:
        return FeatureSet(name="quality", features={
            "missing_fraction_sofar": 1.0,
            "max_gap_minutes": None,
            "edge_max_flag": None,
        })

    n = len(temps_sofar)
    features = {}

    # Missing fraction
    if expected_samples is not None and expected_samples > 0:
        features["missing_fraction_sofar"] = 1.0 - (n / expected_samples)
    else:
        # Estimate expected samples from time range
        if len(timestamps_sofar) >= 2:
            time_range_minutes = (timestamps_sofar[-1] - timestamps_sofar[0]).total_seconds() / 60
            expected_from_range = int(time_range_minutes / step_minutes) + 1
            features["missing_fraction_sofar"] = 1.0 - (n / max(expected_from_range, n))
        else:
            features["missing_fraction_sofar"] = 0.0

    # Maximum gap between observations
    if len(timestamps_sofar) >= 2:
        gaps = []
        for i in range(1, len(timestamps_sofar)):
            gap_seconds = (timestamps_sofar[i] - timestamps_sofar[i-1]).total_seconds()
            gaps.append(gap_seconds / 60)  # Convert to minutes
        features["max_gap_minutes"] = max(gaps)
    else:
        features["max_gap_minutes"] = None

    # Edge max flag: is the maximum at the very start or end?
    # This suggests the true max might be outside our observation window
    if n >= 3:
        arr = np.asarray(temps_sofar, dtype=np.float64)
        max_idx = int(np.argmax(arr))
        # Flag if max is in first or last 10% of samples
        edge_threshold = max(1, n // 10)
        is_edge = (max_idx < edge_threshold) or (max_idx >= n - edge_threshold)
        features["edge_max_flag"] = 1 if is_edge else 0
    else:
        features["edge_max_flag"] = None

    return FeatureSet(name="quality", features=features)


def compute_coverage_features(
    timestamps_sofar: list[datetime],
    day_start_hour: int = 6,
    day_end_hour: int = 23,
) -> FeatureSet:
    """Compute features about observation coverage across the day.

    Args:
        timestamps_sofar: List of observation timestamps
        day_start_hour: Expected start of observation window
        day_end_hour: Expected end of observation window

    Returns:
        FeatureSet with coverage indicators
    """
    if not timestamps_sofar:
        return FeatureSet(name="coverage", features={})

    hours_observed = set(ts.hour for ts in timestamps_sofar)

    # Expected hours
    expected_hours = set(range(day_start_hour, day_end_hour + 1))
    observed_expected = hours_observed & expected_hours

    features = {
        "hours_observed": len(hours_observed),
        "hours_coverage_fraction": len(observed_expected) / len(expected_hours),
        "earliest_hour_observed": min(ts.hour for ts in timestamps_sofar),
        "latest_hour_observed": max(ts.hour for ts in timestamps_sofar),
    }

    return FeatureSet(name="coverage", features=features)


def estimate_expected_samples(
    snapshot_hour: int,
    day_start_hour: int = 6,
    step_minutes: int = 5,
) -> int:
    """Estimate how many samples we should have by snapshot_hour.

    Assumes data collection starts at day_start_hour and runs continuously
    at step_minutes intervals.

    Args:
        snapshot_hour: Current hour (0-23)
        day_start_hour: When observation window starts
        step_minutes: Interval between samples

    Returns:
        Expected number of samples
    """
    if snapshot_hour <= day_start_hour:
        return 0

    hours_elapsed = snapshot_hour - day_start_hour
    minutes_elapsed = hours_elapsed * 60
    expected = minutes_elapsed // step_minutes

    return expected
