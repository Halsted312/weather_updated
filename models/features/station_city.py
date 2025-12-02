"""
Station vs City temperature gap features.

This module computes features from the difference between station-specific
observations (e.g., KMDW) and city-aggregate observations (e.g., Chicago,IL).

The city-aggregate data from Visual Crossing uses weighted averages across
multiple stations near the city center. This may correlate better with
NWS settlement temps than a single station.

Features computed:
    station_city_temp_gap: Current (station_temp - city_temp)
    station_city_max_gap_sofar: (station_max - city_max) for the day
    station_city_mean_gap_sofar: Mean gap throughout day
    station_city_gap_std: Standard deviation of gap (consistency)
    city_warmer_flag: 1 if city_temp > station_temp currently
    station_city_gap_trend: Change in gap (second half mean - first half mean)

Example:
    >>> station_temps = [(t1, 85.2), (t2, 86.1), ...]  # KMDW
    >>> city_temps = [(t1, 84.8), (t2, 85.9), ...]    # Chicago,IL
    >>> fs = compute_station_city_features(station_temps, city_temps)
    >>> fs['station_city_max_gap_sofar']
    0.4
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from models.features.base import FeatureSet, register_feature_group


def _align_temps_by_time(
    temps1: list[tuple[datetime, float]],
    temps2: list[tuple[datetime, float]],
    tolerance_minutes: int = 3,
) -> list[tuple[datetime, float, float]]:
    """Align two temperature series by timestamp.

    Uses O(n) dictionary lookup since VC station and city data have
    identical timestamps when queried with the same parameters.

    Args:
        temps1: List of (datetime, temp) tuples (e.g., station)
        temps2: List of (datetime, temp) tuples (e.g., city)
        tolerance_minutes: Max time difference to consider a match (unused, kept for API compat)

    Returns:
        List of (datetime, temp1, temp2) tuples where times align
    """
    if not temps1 or not temps2:
        return []

    # O(n) - build dict from temps2
    temps2_dict = {dt: t for dt, t in temps2}

    # O(n) - direct lookup (timestamps match exactly from VC)
    aligned = []
    for dt1, t1 in temps1:
        if dt1 in temps2_dict:
            aligned.append((dt1, t1, temps2_dict[dt1]))

    return aligned


@register_feature_group("station_city")
def compute_station_city_features(
    station_temps: list[tuple[datetime, float]],
    city_temps: list[tuple[datetime, float]],
) -> FeatureSet:
    """Compute features comparing station vs city temperatures.

    Args:
        station_temps: List of (datetime_local, temp_f) from station (e.g., KMDW)
        city_temps: List of (datetime_local, temp_f) from city aggregate

    Returns:
        FeatureSet with station-city comparison features
    """
    null_features = {
        "station_city_temp_gap": None,
        "station_city_max_gap_sofar": None,
        "station_city_mean_gap_sofar": None,
        "city_warmer_flag": None,
        "station_city_gap_std": None,
        "station_city_gap_trend": None,
    }

    if not station_temps or not city_temps:
        return FeatureSet(name="station_city", features=null_features)

    # Align the two series
    aligned = _align_temps_by_time(station_temps, city_temps)

    if not aligned:
        return FeatureSet(name="station_city", features=null_features)

    features = {}

    # Extract aligned values
    gaps = [stn - city for _, stn, city in aligned]
    station_vals = [stn for _, stn, _ in aligned]
    city_vals = [city for _, _, city in aligned]

    # Current gap (most recent)
    features["station_city_temp_gap"] = float(gaps[-1])

    # Max gap
    station_max = max(station_vals)
    city_max = max(city_vals)
    features["station_city_max_gap_sofar"] = float(station_max - city_max)

    # Mean gap throughout day
    features["station_city_mean_gap_sofar"] = float(np.mean(gaps))

    # Std of gap (consistency)
    features["station_city_gap_std"] = float(np.std(gaps)) if len(gaps) > 1 else 0.0

    # City warmer flag (current)
    features["city_warmer_flag"] = 1 if gaps[-1] < 0 else 0

    # Gap trend (is the gap increasing or decreasing?)
    # Positive trend = station warming faster than city
    # Negative trend = city warming faster than station
    if len(gaps) >= 4:  # Need at least ~20 min of data
        # Use last half vs first half of gaps
        mid = len(gaps) // 2
        first_half_mean = np.mean(gaps[:mid])
        second_half_mean = np.mean(gaps[mid:])
        features["station_city_gap_trend"] = float(second_half_mean - first_half_mean)
    else:
        features["station_city_gap_trend"] = None

    return FeatureSet(name="station_city", features=features)


def compute_multi_station_features(
    primary_station_temps: list[tuple[datetime, float]],
    secondary_temps_dict: dict[str, list[tuple[datetime, float]]],
) -> FeatureSet:
    """Compute features from multiple temperature sources.

    Useful when you have both station and city data, or multiple stations.

    Args:
        primary_station_temps: Main station temps (e.g., KMDW for settlement)
        secondary_temps_dict: Dict of source_name -> temps list
                             e.g., {"city": [...], "other_station": [...]}

    Returns:
        FeatureSet with cross-source comparison features
    """
    features = {}

    if not primary_station_temps:
        return FeatureSet(name="multi_station", features=features)

    primary_max = max(t for _, t in primary_station_temps)
    primary_mean = np.mean([t for _, t in primary_station_temps])

    for source_name, temps in secondary_temps_dict.items():
        if not temps:
            features[f"{source_name}_max_gap"] = None
            features[f"{source_name}_mean_gap"] = None
            continue

        secondary_max = max(t for _, t in temps)
        secondary_mean = np.mean([t for _, t in temps])

        features[f"{source_name}_max_gap"] = float(primary_max - secondary_max)
        features[f"{source_name}_mean_gap"] = float(primary_mean - secondary_mean)

    return FeatureSet(name="multi_station", features=features)
