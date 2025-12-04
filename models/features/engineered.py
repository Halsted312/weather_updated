"""
Engineered features: transforms and interactions of existing features.

This module computes derived features that capture non-linear relationships
and synergistic effects between base features. These are designed based on
feature importance analysis and domain knowledge.

Feature groups:
    Log Transforms (4 features):
        Compress outliers and capture diminishing returns / exponential effects

    Squared Features (3 features):
        Amplify strong signals (large temperature rates, large forecast errors)

    High-Value Interactions (6 features):
        Cross-feature products that capture conditional relationships

Total: 13 features

Example:
    >>> base_features = {...}  # Dict from other feature modules
    >>> fs = compute_engineered_features(base_features)
    >>> fs['log_abs_obs_fcst_gap']
    1.79  # log(1 + abs(obs_fcst_max_gap))
"""

import math
import numpy as np
from typing import Dict, Any, Optional

from models.features.base import FeatureSet, register_feature_group


@register_feature_group("engineered")
def compute_engineered_features(
    base_features: Dict[str, Any],
) -> FeatureSet:
    """Compute engineered transforms and interactions from existing features.

    Args:
        base_features: Dictionary of features from other modules
                      (partial_day, forecast, momentum, meteo, interactions, etc.)

    Returns:
        FeatureSet with 13 engineered features
    """
    features = {}

    # =========================================================================
    # LOG TRANSFORMS (4 features)
    # =========================================================================

    # Log of absolute obs-forecast gap (upside potential)
    obs_fcst_gap = base_features.get("obs_fcst_max_gap")
    if obs_fcst_gap is not None:
        features["log_abs_obs_fcst_gap"] = math.log(1 + abs(obs_fcst_gap))
    else:
        features["log_abs_obs_fcst_gap"] = None

    # Log of temperature std (compress outlier volatility)
    temp_std = base_features.get("temp_std_last_60min")
    if temp_std is not None and temp_std >= 0:
        features["log_temp_std_last_60min"] = math.log(0.1 + temp_std)
    else:
        features["log_temp_std_last_60min"] = None

    # Log of intraday range (compress large ranges)
    intraday_range = base_features.get("intraday_range_sofar")
    if intraday_range is not None and intraday_range >= 0:
        features["log_intraday_range"] = math.log(1 + intraday_range)
    else:
        features["log_intraday_range"] = None

    # Log of expected delta uncertainty (time-based uncertainty)
    delta_unc = base_features.get("expected_delta_uncertainty")
    if delta_unc is not None and delta_unc > 0:
        features["log_expected_delta_uncertainty"] = math.log(delta_unc)
    else:
        features["log_expected_delta_uncertainty"] = None

    # =========================================================================
    # SQUARED FEATURES (3 features)
    # =========================================================================

    # Temperature rate squared (amplify strong warming/cooling signals)
    temp_rate = base_features.get("temp_rate_last_30min")
    if temp_rate is not None:
        features["temp_rate_last_30min_squared"] = temp_rate ** 2
    else:
        features["temp_rate_last_30min_squared"] = None

    # Forecast error squared (emphasize large errors)
    err_mean = base_features.get("err_mean_sofar")
    if err_mean is not None:
        features["err_mean_sofar_squared"] = err_mean ** 2
    else:
        features["err_mean_sofar_squared"] = None

    # Obs-forecast gap squared (emphasize large gaps)
    if obs_fcst_gap is not None:
        features["obs_fcst_gap_squared"] = obs_fcst_gap ** 2
    else:
        features["obs_fcst_gap_squared"] = None

    # =========================================================================
    # HIGH-VALUE INTERACTIONS (6 features)
    # =========================================================================

    # Multi-horizon coefficient of variation (forecast uncertainty ratio)
    fcst_multi_mean = base_features.get("fcst_multi_mean")
    fcst_multi_std = base_features.get("fcst_multi_std")
    if fcst_multi_mean is not None and fcst_multi_std is not None and abs(fcst_multi_mean) > 0.01:
        features["fcst_multi_cv"] = fcst_multi_std / abs(fcst_multi_mean)
    else:
        features["fcst_multi_cv"] = None

    # Multi-horizon range as percentage of mean
    fcst_multi_range = base_features.get("fcst_multi_range")
    if fcst_multi_mean is not None and fcst_multi_range is not None and abs(fcst_multi_mean) > 0.01:
        features["fcst_multi_range_pct"] = (fcst_multi_range / abs(fcst_multi_mean)) * 100
    else:
        features["fcst_multi_range_pct"] = None

    # Humidity × Temperature rate (humid air heats/cools slower)
    humidity = base_features.get("humidity_last_obs")
    if humidity is not None and temp_rate is not None:
        features["humidity_x_temp_rate"] = humidity * temp_rate
    else:
        features["humidity_x_temp_rate"] = None

    # Cloud cover × Hour (cloud impact varies by time of day)
    cloudcover = base_features.get("cloudcover_last_obs")
    hour = base_features.get("hour")
    if cloudcover is not None and hour is not None:
        features["cloudcover_x_hour"] = cloudcover * hour
    else:
        features["cloudcover_x_hour"] = None

    # Temperature EMA × Day fraction (current temp weighted by progress through day)
    temp_ema = base_features.get("temp_ema_60min")
    day_fraction = base_features.get("day_fraction")
    if temp_ema is not None and day_fraction is not None:
        features["temp_ema_x_day_fraction"] = temp_ema * day_fraction
    else:
        features["temp_ema_x_day_fraction"] = None

    # Station-city gap × Forecast gap (double signal when both show same direction)
    station_city_gap = base_features.get("station_city_temp_gap")
    if station_city_gap is not None and obs_fcst_gap is not None:
        features["station_city_gap_x_fcst_gap"] = station_city_gap * obs_fcst_gap
    else:
        features["station_city_gap_x_fcst_gap"] = None

    return FeatureSet(name="engineered", features=features)
