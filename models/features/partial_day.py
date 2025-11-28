"""
Partial-day observation features for temperature Δ-models.

This module computes base statistics from Visual Crossing temperatures
observed up to snapshot time τ. These are the foundational features
that describe what we know about today's temperatures so far.

No future data is used - only observations with datetime_local < τ.

Features computed:
    vc_max_f_sofar: Maximum temperature observed so far
    vc_min_f_sofar: Minimum temperature observed so far
    vc_mean_f_sofar: Mean temperature observed so far
    vc_std_f_sofar: Standard deviation of temperatures
    vc_q10_f_sofar through vc_q90_f_sofar: Percentiles (10, 25, 50, 75, 90)
    vc_frac_part_sofar: Fractional part of max (for rounding behavior)
    num_samples_sofar: Number of observations (data quality indicator)
    t_base: Rounded max = baseline for Δ target

Example:
    >>> temps = [72.1, 75.3, 80.2, 82.5, 81.1]
    >>> fs = compute_partial_day_features(temps)
    >>> fs['vc_max_f_sofar']
    82.5
    >>> fs['t_base']
    82
"""

import numpy as np
from typing import Optional

from models.features.base import FeatureSet, register_feature_group


@register_feature_group("partial_day")
def compute_partial_day_features(temps_sofar: list[float]) -> FeatureSet:
    """Compute base statistics from temperatures observed up to snapshot time.

    These are the most fundamental features - simple statistics over the
    partial-day temperature series. They form the backbone of the Δ-model
    input, describing what we've observed so far without any lookahead.

    Args:
        temps_sofar: List of 5-minute VC temperatures in °F observed up to τ.
                     Should be ordered chronologically but order doesn't affect
                     these statistics.

    Returns:
        FeatureSet with partial-day statistics. Returns empty FeatureSet if
        temps_sofar is empty.

    Note:
        The t_base feature (rounded max) is computed here as it's the
        baseline for the Δ target: Δ = T_settle - t_base.
    """
    if not temps_sofar:
        return FeatureSet(name="partial_day", features={})

    arr = np.asarray(temps_sofar, dtype=np.float64)

    # Basic statistics
    max_f = float(np.max(arr))
    min_f = float(np.min(arr))
    mean_f = float(np.mean(arr))

    # Standard deviation (use ddof=1 for sample std)
    # Handle single-sample case
    if len(arr) > 1:
        std_f = float(np.std(arr, ddof=1))
    else:
        std_f = 0.0

    # Percentiles
    q10, q25, q50, q75, q90 = np.percentile(arr, [10, 25, 50, 75, 90])

    # Fractional part of max - useful for predicting rounding behavior
    # Positive if max is above its rounded value, negative if below
    frac_part = max_f - round(max_f)

    # Number of samples (data quality indicator)
    num_samples = len(arr)

    # Baseline temperature for Δ target
    t_base = int(round(max_f))

    features = {
        "vc_max_f_sofar": max_f,
        "vc_min_f_sofar": min_f,
        "vc_mean_f_sofar": mean_f,
        "vc_std_f_sofar": std_f,
        "vc_q10_f_sofar": float(q10),
        "vc_q25_f_sofar": float(q25),
        "vc_q50_f_sofar": float(q50),
        "vc_q75_f_sofar": float(q75),
        "vc_q90_f_sofar": float(q90),
        "vc_frac_part_sofar": frac_part,
        "num_samples_sofar": num_samples,
        "t_base": t_base,
    }

    return FeatureSet(name="partial_day", features=features)


def compute_delta_target(
    settle_f: int,
    vc_max_f_sofar: float,
    clip_range: tuple[int, int] = (-2, 10),
) -> dict[str, int]:
    """Compute the Δ target for training.

    The Δ target is the deviation between the final settlement temperature
    and the baseline (rounded partial-day max). This is what the model
    learns to predict.

    The model predicts discrete delta classes [-2, ..., 0, ..., +10] (13 classes).
    Low clip at -2 keeps tail ≤15% (2.1% actual). High clip at +10 captures
    early-morning snapshots where daily high hasn't occurred yet.
    The raw delta is preserved in 'delta_raw' for analysis.

    Args:
        settle_f: Final NWS/Kalshi settlement temperature (integer °F)
        vc_max_f_sofar: Maximum VC temperature observed up to snapshot time
        clip_range: Min/max delta values (default: [-2, +10])

    Returns:
        Dictionary with:
            t_base: Rounded max (baseline)
            delta: T_settle - t_base, clipped to [-2, +10] (target for model)
            delta_raw: Raw unclipped delta (for analysis)
            abs_delta: Absolute value of clipped delta

    Example:
        >>> compute_delta_target(settle_f=92, vc_max_f_sofar=91.4)
        {'t_base': 91, 'delta': 1, 'delta_raw': 1, 'abs_delta': 1}
        >>> compute_delta_target(settle_f=87, vc_max_f_sofar=92.2)
        {'t_base': 92, 'delta': -2, 'delta_raw': -5, 'abs_delta': 2}  # clipped at -2
    """
    t_base = int(round(vc_max_f_sofar))
    delta_raw = int(settle_f - t_base)

    # Clip delta to model's prediction range
    delta_clipped = max(clip_range[0], min(clip_range[1], delta_raw))
    abs_delta = abs(delta_clipped)

    return {
        "t_base": t_base,
        "delta": delta_clipped,
        "delta_raw": delta_raw,
        "abs_delta": abs_delta,
    }


def compute_range_features(temps_sofar: list[float]) -> FeatureSet:
    """Compute range-based features from partial-day temps.

    These features capture the spread and extremes of the temperature
    distribution, which can be predictive of final settlement behavior.

    Args:
        temps_sofar: List of 5-minute VC temperatures

    Returns:
        FeatureSet with range statistics
    """
    if not temps_sofar:
        return FeatureSet(name="range", features={})

    arr = np.asarray(temps_sofar, dtype=np.float64)

    max_f = float(np.max(arr))
    min_f = float(np.min(arr))
    range_f = max_f - min_f

    # Interquartile range
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = float(q75 - q25)

    features = {
        "temp_range_sofar": range_f,
        "temp_iqr_sofar": iqr,
    }

    return FeatureSet(name="range", features=features)


def get_baseline_prediction(temps_sofar: list[float]) -> Optional[int]:
    """Get the simplest baseline prediction: rounded max.

    This is the most naive predictor - just round the observed max.
    Useful as a sanity check baseline.

    Args:
        temps_sofar: List of 5-minute VC temperatures

    Returns:
        Rounded max temperature, or None if no data
    """
    if not temps_sofar:
        return None
    return int(round(max(temps_sofar)))
