"""
Interaction and combined features for temperature Δ-models.

This module computes features that combine multiple inputs to capture
non-linear relationships and synergistic effects.

Key insight: The importance of certain features changes based on context.
For example, the "gap to forecast max" matters more when there's less
time remaining until market close.

Features computed:
    Time × Temperature interactions:
        temp_x_hours_remaining: vc_max * hours_to_close (upside potential weighted by time)
        gap_x_hours_remaining: obs_fcst_gap * hours_to_close
        temp_x_day_fraction: vc_max * day_fraction

    Forecast × Observation ratios:
        fcst_obs_ratio: fcst_max / obs_max (>1 means upside expected)
        fcst_obs_diff_squared: (fcst - obs)^2 (emphasizes large gaps)

    Log transforms:
        log_minutes_since_market_open: Captures diminishing returns
        log_hours_to_close: Time decay

    Normalized features:
        temp_zscore_vs_forecast: (obs_max - fcst_mean) / fcst_std

Example:
    >>> fs = compute_interaction_features(
    ...     vc_max=85.2, fcst_max=87.0, hours_remaining=4.5
    ... )
    >>> fs['gap_x_hours_remaining']
    8.1  # 1.8 * 4.5
"""

import math
import numpy as np
from typing import Optional

from models.features.base import FeatureSet, register_feature_group


@register_feature_group("interactions")
def compute_interaction_features(
    vc_max_f_sofar: Optional[float] = None,
    fcst_prev_max_f: Optional[float] = None,
    fcst_prev_mean_f: Optional[float] = None,
    fcst_prev_std_f: Optional[float] = None,
    hours_to_event_close: Optional[float] = None,
    minutes_since_market_open: Optional[float] = None,
    day_fraction: Optional[float] = None,
    obs_fcst_max_gap: Optional[float] = None,
) -> FeatureSet:
    """Compute interaction and combined features.

    Args:
        vc_max_f_sofar: Max observed temp so far
        fcst_prev_max_f: Forecast max from T-1
        fcst_prev_mean_f: Forecast mean from T-1
        fcst_prev_std_f: Forecast std from T-1
        hours_to_event_close: Hours until market closes
        minutes_since_market_open: Minutes since market opened
        day_fraction: Fraction of day elapsed (0-1)
        obs_fcst_max_gap: fcst_max - obs_max (upside potential)

    Returns:
        FeatureSet with interaction features
    """
    features = {}

    # Time × Temperature interactions
    if vc_max_f_sofar is not None and hours_to_event_close is not None:
        features["temp_x_hours_remaining"] = vc_max_f_sofar * hours_to_event_close
    else:
        features["temp_x_hours_remaining"] = None

    if obs_fcst_max_gap is not None and hours_to_event_close is not None:
        # Gap weighted by time - upside potential matters more with more time
        features["gap_x_hours_remaining"] = obs_fcst_max_gap * hours_to_event_close
    else:
        features["gap_x_hours_remaining"] = None

    if vc_max_f_sofar is not None and day_fraction is not None:
        features["temp_x_day_fraction"] = vc_max_f_sofar * day_fraction
    else:
        features["temp_x_day_fraction"] = None

    # Forecast × Observation ratios
    if fcst_prev_max_f is not None and vc_max_f_sofar is not None and vc_max_f_sofar > 0:
        features["fcst_obs_ratio"] = fcst_prev_max_f / vc_max_f_sofar
    else:
        features["fcst_obs_ratio"] = None

    if fcst_prev_max_f is not None and vc_max_f_sofar is not None:
        diff = fcst_prev_max_f - vc_max_f_sofar
        features["fcst_obs_diff_squared"] = diff * diff
    else:
        features["fcst_obs_diff_squared"] = None

    # Log transforms
    if minutes_since_market_open is not None and minutes_since_market_open > 0:
        features["log_minutes_since_open"] = math.log(minutes_since_market_open)
    else:
        features["log_minutes_since_open"] = None

    if hours_to_event_close is not None and hours_to_event_close > 0:
        features["log_hours_to_close"] = math.log(hours_to_event_close)
    else:
        features["log_hours_to_close"] = None

    # Z-score: How unusual is current obs relative to forecast distribution
    if (vc_max_f_sofar is not None and fcst_prev_mean_f is not None
            and fcst_prev_std_f is not None and fcst_prev_std_f > 0):
        features["temp_zscore_vs_forecast"] = (
            (vc_max_f_sofar - fcst_prev_mean_f) / fcst_prev_std_f
        )
    else:
        features["temp_zscore_vs_forecast"] = None

    return FeatureSet(name="interactions", features=features)


def compute_lagged_forecast_error_features(
    fcst_errors_history: list[float],
) -> FeatureSet:
    """Compute features from historical forecast accuracy.

    These features capture how accurate forecasts have been recently,
    which helps calibrate confidence in today's forecast.

    Args:
        fcst_errors_history: List of (fcst_max - actual_settle) for recent days
                            Most recent first. Positive = forecast was too high.

    Returns:
        FeatureSet with forecast accuracy features
    """
    features = {}

    if not fcst_errors_history:
        features["fcst_error_lag1"] = None
        features["fcst_error_lag7_mean"] = None
        features["fcst_mae_recent"] = None
        features["fcst_bias_recent"] = None
        return FeatureSet(name="fcst_accuracy", features=features)

    # Lag 1 (yesterday's error)
    features["fcst_error_lag1"] = float(fcst_errors_history[0])

    # 7-day mean error
    if len(fcst_errors_history) >= 7:
        features["fcst_error_lag7_mean"] = float(np.mean(fcst_errors_history[:7]))
    else:
        features["fcst_error_lag7_mean"] = float(np.mean(fcst_errors_history))

    # MAE (mean absolute error)
    features["fcst_mae_recent"] = float(np.mean(np.abs(fcst_errors_history)))

    # Bias (systematic over/under prediction)
    features["fcst_bias_recent"] = float(np.mean(fcst_errors_history))

    return FeatureSet(name="fcst_accuracy", features=features)


def compute_derived_forecast_features(
    fcst_prev_max_f: Optional[float],
    vc_max_f_sofar: Optional[float],
    fcst_prev_hour_of_max: Optional[int],
    snapshot_hour: Optional[int],
) -> FeatureSet:
    """Compute derived features combining forecast and observation data.

    These are the "obs_fcst_*" features referenced in the existing codebase.

    Args:
        fcst_prev_max_f: Forecast max from T-1
        vc_max_f_sofar: Current observed max
        fcst_prev_hour_of_max: Hour when forecast expects max
        snapshot_hour: Current hour

    Returns:
        FeatureSet with derived forecast features
    """
    features = {}

    # Gap to forecast max (upside potential)
    if fcst_prev_max_f is not None and vc_max_f_sofar is not None:
        features["obs_fcst_max_gap"] = fcst_prev_max_f - vc_max_f_sofar
        features["above_fcst_flag"] = 1 if vc_max_f_sofar > fcst_prev_max_f else 0
    else:
        features["obs_fcst_max_gap"] = None
        features["above_fcst_flag"] = None

    # Hours until forecast max
    if fcst_prev_hour_of_max is not None and snapshot_hour is not None:
        features["hours_until_fcst_max"] = max(0, fcst_prev_hour_of_max - snapshot_hour)
    else:
        features["hours_until_fcst_max"] = None

    return FeatureSet(name="derived_forecast", features=features)


def compute_regime_features(
    temp_rate_last_30min: Optional[float] = None,
    minutes_since_max_observed: Optional[float] = None,
    snapshot_hour: Optional[int] = None,
) -> FeatureSet:
    """Compute temperature regime/phase flags.

    The daily temperature cycle typically follows:
    1. Heating phase (morning): Temperature rising
    2. Plateau phase (early afternoon): Temperature near max, relatively stable
    3. Cooling phase (evening): Temperature declining

    These phases are important because prediction strategies differ by phase.
    During heating, we care about how much more warming is possible.
    During plateau, we're estimating the max. During cooling, the max is likely set.

    Args:
        temp_rate_last_30min: Rate of temperature change (°F/hour)
        minutes_since_max_observed: Minutes since daily max was observed
        snapshot_hour: Current hour (0-23)

    Returns:
        FeatureSet with regime flags
    """
    features = {
        "is_heating_phase": None,
        "is_plateau_phase": None,
        "is_cooling_phase": None,
    }

    # Need at least one piece of information
    if temp_rate_last_30min is None and snapshot_hour is None:
        return FeatureSet(name="regime", features=features)

    # Thresholds for determining phase
    # Heating: temp rising at > 0.5°F/hour OR it's morning (before noon) with non-negative rate
    # Plateau: temp stable (rate between -0.5 and +0.5) and within 30 min of max
    # Cooling: temp falling at < -0.5°F/hour OR rate negative and > 30 min since max

    heating = 0
    plateau = 0
    cooling = 0

    if temp_rate_last_30min is not None:
        if temp_rate_last_30min > 0.5:
            # Clearly rising
            heating = 1
        elif temp_rate_last_30min < -0.5:
            # Clearly falling
            cooling = 1
        else:
            # Rate is relatively flat - could be plateau or transition
            if minutes_since_max_observed is not None:
                if minutes_since_max_observed < 30:
                    # Near max, flat rate = plateau
                    plateau = 1
                else:
                    # Been a while since max, flat or slightly falling = cooling
                    cooling = 1
            elif snapshot_hour is not None:
                # Use time of day as fallback
                if 11 <= snapshot_hour <= 15:
                    # Mid-day with flat rate = plateau
                    plateau = 1
                elif snapshot_hour < 11:
                    # Morning with flat rate = probably still heating
                    heating = 1
                else:
                    # Evening with flat rate = cooling
                    cooling = 1
    elif snapshot_hour is not None:
        # No rate available, use time-of-day heuristic
        if snapshot_hour < 11:
            heating = 1
        elif 11 <= snapshot_hour <= 15:
            plateau = 1
        else:
            cooling = 1

    features["is_heating_phase"] = heating
    features["is_plateau_phase"] = plateau
    features["is_cooling_phase"] = cooling

    return FeatureSet(name="regime", features=features)


@register_feature_group("time_confidence")
def compute_time_confidence_features(
    hours_since_market_open: Optional[float] = None,
    is_event_day: Optional[int] = None,
    vc_max_f_sofar: Optional[float] = None,
    fcst_prev_max_f: Optional[float] = None,
) -> FeatureSet:
    """Features that capture increasing certainty as time passes.

    Key insight: As time progresses through the trading day, our confidence
    in the observed max being the final settlement max increases. This should
    be explicitly encoded so the model learns to weight observations more
    heavily later in the day.

    Features:
        obs_confidence: 0 at market open → 1 near close (faster on event day)
        expected_delta_uncertainty: Expected magnitude of remaining error (shrinks with time)
        confidence_weighted_gap: fcst-obs gap weighted by (1 - confidence)
        fcst_importance_weight: How much to trust forecast (1 - confidence)
        remaining_upside: Upside potential weighted by remaining uncertainty

    Args:
        hours_since_market_open: Hours elapsed since D-1 10:00 AM
        is_event_day: 1 if on event day (D-0), 0 if D-1
        vc_max_f_sofar: Max observed temperature so far
        fcst_prev_max_f: Forecast max from T-1

    Returns:
        FeatureSet with time-confidence features
    """
    features = {
        "obs_confidence": None,
        "expected_delta_uncertainty": None,
        "confidence_weighted_gap": None,
        "fcst_importance_weight": None,
        "remaining_upside": None,
    }

    if hours_since_market_open is None:
        return FeatureSet(name="time_confidence", features=features)

    # 1. Observation confidence: 0 at market open → 1 near close
    #    Ramps faster on event day (D-0) since max is likely already observed
    if is_event_day == 1:
        # On D-0 (hours 24+), confidence ramps quickly
        # Full confidence by hour 34 (10 hours into event day)
        hours_into_event_day = max(0, hours_since_market_open - 24)
        obs_confidence = min(1.0, hours_into_event_day / 10)
    else:
        # On D-1 (hours 0-24), confidence ramps slowly
        # Only 80% confident by hour 24 (end of D-1)
        obs_confidence = min(0.8, hours_since_market_open / 30)

    features["obs_confidence"] = obs_confidence

    # 2. Expected remaining delta magnitude
    #    Early: expect ±8 degrees of uncertainty
    #    Late: expect ±2 degrees of uncertainty
    expected_delta_uncertainty = max(2.0, 8.0 - hours_since_market_open * 0.2)
    features["expected_delta_uncertainty"] = expected_delta_uncertainty

    # 3. Forecast importance weight (inverse of observation confidence)
    fcst_importance_weight = 1.0 - obs_confidence
    features["fcst_importance_weight"] = fcst_importance_weight

    # Features that need both temps
    if vc_max_f_sofar is not None and fcst_prev_max_f is not None:
        obs_fcst_gap = fcst_prev_max_f - vc_max_f_sofar

        # 4. Confidence-weighted gap
        #    Gap matters less when we're confident in observations
        features["confidence_weighted_gap"] = obs_fcst_gap * fcst_importance_weight

        # 5. Remaining upside potential
        #    Only meaningful when fcst > obs and we're early in day
        features["remaining_upside"] = max(0, obs_fcst_gap) * fcst_importance_weight

    return FeatureSet(name="time_confidence", features=features)
