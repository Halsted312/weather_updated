"""
Base utilities for feature engineering.

This module defines the foundational types and utilities used across all
feature modules. The key abstraction is FeatureSet, a named container for
a group of related features.

Design principles:
    - Features are pure functions: data in, features out
    - No database coupling inside feature code
    - All functions work identically for training and inference
    - Registry pattern allows easy iteration and composition

Example:
    >>> from models.features.base import FeatureSet, compose_features
    >>> fs1 = FeatureSet("partial_day", {"vc_max_f_sofar": 92.5, "vc_min_f_sofar": 68.2})
    >>> fs2 = FeatureSet("calendar", {"snapshot_hour": 14, "month": 7})
    >>> combined = compose_features(fs1, fs2)
    >>> combined
    {'vc_max_f_sofar': 92.5, 'vc_min_f_sofar': 68.2, 'snapshot_hour': 14, 'month': 7}
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class FeatureSet:
    """Container for a named group of features.

    Attributes:
        name: Identifier for this feature group (e.g., 'partial_day', 'shape')
        features: Dictionary mapping feature names to values
    """

    name: str
    features: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of features in this set."""
        return len(self.features)

    def __getitem__(self, key: str) -> Any:
        """Access feature by name."""
        return self.features[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get feature with default value."""
        return self.features.get(key, default)

    def keys(self) -> list[str]:
        """List of feature names."""
        return list(self.features.keys())

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dictionary."""
        return self.features.copy()


def compose_features(*feature_sets: FeatureSet) -> dict[str, Any]:
    """Merge multiple FeatureSets into one dictionary.

    Combines features from multiple FeatureSets into a single flat dictionary.
    Later feature sets override earlier ones if there are name collisions.

    Args:
        *feature_sets: Variable number of FeatureSet objects to merge

    Returns:
        Combined dictionary of all features

    Example:
        >>> fs1 = FeatureSet("a", {"x": 1, "y": 2})
        >>> fs2 = FeatureSet("b", {"z": 3})
        >>> compose_features(fs1, fs2)
        {'x': 1, 'y': 2, 'z': 3}
    """
    result: dict[str, Any] = {}
    for fs in feature_sets:
        if fs is not None:
            result.update(fs.features)
    return result


def prefix_features(fs: FeatureSet, prefix: str) -> FeatureSet:
    """Add prefix to all feature names in a FeatureSet.

    Useful for distinguishing features from different time windows
    or data sources.

    Args:
        fs: Original FeatureSet
        prefix: String to prepend to each feature name

    Returns:
        New FeatureSet with prefixed feature names

    Example:
        >>> fs = FeatureSet("stats", {"max": 92, "min": 68})
        >>> prefix_features(fs, "morning_")
        FeatureSet(name='stats', features={'morning_max': 92, 'morning_min': 68})
    """
    prefixed = {f"{prefix}{k}": v for k, v in fs.features.items()}
    return FeatureSet(name=fs.name, features=prefixed)


def filter_none_features(features: dict[str, Any]) -> dict[str, Any]:
    """Remove features with None values.

    Useful for cleaning up feature dictionaries before model input.

    Args:
        features: Dictionary potentially containing None values

    Returns:
        Dictionary with None values removed
    """
    return {k: v for k, v in features.items() if v is not None}


# Registry of all feature group computation functions
# Each module registers its compute function here
# Key: group name, Value: function that returns FeatureSet
ALL_FEATURE_GROUPS: dict[str, Callable[..., FeatureSet]] = {}


def register_feature_group(name: str):
    """Decorator to register a feature computation function.

    Example:
        @register_feature_group("partial_day")
        def compute_partial_day_features(temps_sofar: list[float]) -> FeatureSet:
            ...
    """
    def decorator(func: Callable[..., FeatureSet]) -> Callable[..., FeatureSet]:
        ALL_FEATURE_GROUPS[name] = func
        return func
    return decorator


# Standard feature column lists for model training
# These define which features go into the numeric vs categorical pipelines

NUMERIC_FEATURE_COLS: list[str] = [
    # Partial day stats (core features)
    "vc_max_f_sofar", "vc_min_f_sofar", "vc_mean_f_sofar", "vc_std_f_sofar",
    "vc_q10_f_sofar", "vc_q25_f_sofar", "vc_q50_f_sofar",
    "vc_q75_f_sofar", "vc_q90_f_sofar",
    "vc_frac_part_sofar", "num_samples_sofar",
    "t_base",
    # Shape features (removed constant features: minutes_ge_base_p1, max_run_ge_base_p1)
    "minutes_ge_base", "minutes_ge_base_m1",
    "max_run_ge_base", "max_run_ge_base_m1",
    "max_minus_second_max",
    # Time-of-day max features (structural nulls - filled with vc_max_f_sofar)
    "max_morning_f_sofar", "max_afternoon_f_sofar", "max_evening_f_sofar",
    "slope_max_30min_up_sofar", "slope_max_30min_down_sofar",
    # Rule predictions - only keep those that differ from t_base
    # REMOVED: pred_max_round_sofar, pred_max_of_rounded_sofar (identical to t_base)
    # REMOVED: pred_c_first_sofar, pred_ignore_singletons_sofar (low differentiation)
    "pred_ceil_max_sofar",       # differs 50% from t_base
    "pred_floor_max_sofar",      # differs 44% from t_base
    "pred_plateau_20min_sofar",  # differs 51% from t_base
    # Note: err_{rule}_sofar features are EXCLUDED (target leakage)
    # Note: disagree_flag_sofar EXCLUDED (99.4% constant at 1)
    "range_pred_rules_sofar", "num_distinct_preds_sofar",
    # T-1 Forecast features
    "fcst_prev_max_f", "fcst_prev_min_f", "fcst_prev_mean_f", "fcst_prev_std_f",
    "fcst_prev_q10_f", "fcst_prev_q25_f", "fcst_prev_q50_f",
    "fcst_prev_q75_f", "fcst_prev_q90_f",
    "fcst_prev_frac_part", "fcst_prev_hour_of_max", "t_forecast_base",
    # Feature Group 1: Integer boundary features
    "fcst_prev_distance_to_int", "fcst_prev_near_boundary_flag",
    # Feature Group 2: Peak window features (from hourly/15-min curve)
    "fcst_peak_temp_f", "fcst_peak_hour_float",
    "fcst_peak_band_width_min", "fcst_peak_step_minutes",
    # Feature Group 3: Forecast drift features (multi-lead)
    "fcst_drift_num_leads", "fcst_drift_std_f",
    "fcst_drift_max_upside_f", "fcst_drift_max_downside_f",
    "fcst_drift_mean_delta_f", "fcst_drift_slope_f_per_lead",
    # Multi-horizon features (T-1 to T-6 forecast evolution)
    "fcst_multi_mean", "fcst_multi_ema",
    "fcst_multi_std", "fcst_multi_range",
    "fcst_multi_drift",
    # Feature Group 4: Multivar static features (humidity, cloudcover, dewpoint)
    "fcst_humidity_mean", "fcst_humidity_min", "fcst_humidity_max", "fcst_humidity_range",
    "fcst_cloudcover_mean", "fcst_cloudcover_min", "fcst_cloudcover_max", "fcst_cloudcover_range",
    "fcst_dewpoint_mean", "fcst_dewpoint_min", "fcst_dewpoint_max", "fcst_dewpoint_range",
    "fcst_humidity_morning_mean", "fcst_humidity_afternoon_mean",
    # Forecast vs observation errors (NOT target leakage - compares fcst to obs)
    "err_mean_sofar", "err_std_sofar",
    "err_max_pos_sofar", "err_max_neg_sofar", "err_abs_mean_sofar",
    "err_last1h", "err_last3h_mean",
    # Derived forecast features (high correlation with delta, no leakage)
    "obs_fcst_max_gap",       # fcst_max - vc_max_sofar (upside potential)
    "hours_until_fcst_max",   # fcst_hour_of_max - snapshot_hour
    "above_fcst_flag",        # 1 if vc_max > fcst_max
    "day_fraction",           # (snapshot_hour - 6) / 18
    # Calendar features (time-of-day)
    # New time-of-day features (tod_v1)
    "hour", "minute", "minutes_since_midnight",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "time_of_day_sin", "time_of_day_cos",
    # Day-level features
    "doy_sin", "doy_cos", "week_sin", "week_cos",
    "month", "is_weekend",
    # Lag features (small % nulls - median imputation OK)
    "settle_f_lag1", "settle_f_lag2", "settle_f_lag7",
    "vc_max_f_lag1", "vc_max_f_lag7", "delta_vcmax_lag1",
    # Quality features (removed max_gap_minutes - 99.6% constant at 5)
    "missing_fraction_sofar", "edge_max_flag",
    # Station-city gap features
    "station_city_temp_gap", "station_city_max_gap_sofar",
    "station_city_mean_gap_sofar", "station_city_gap_std", "station_city_gap_trend",
    # Temperature momentum and volatility
    "temp_mean_last_30min", "temp_std_last_30min", "temp_max_last_30min",
    "temp_mean_last_60min", "temp_std_last_60min", "temp_max_last_60min",
    "temp_mean_last_120min", "temp_std_last_120min",
    "temp_rate_last_30min", "temp_rate_last_60min", "temp_acceleration",
    "temp_ema_30min", "temp_ema_60min",
    "intraday_range_sofar",
    "minutes_since_max_observed",
    # Meteo observations (humidity, wind, cloud)
    "humidity_last_obs", "humidity_mean_last_60min", "humidity_std_last_60min", "high_humidity_flag",
    "windspeed_last_obs", "windspeed_max_last_60min", "windgust_max_last_60min", "strong_wind_flag",
    "cloudcover_last_obs", "cloudcover_mean_last_60min", "clear_sky_flag", "high_cloud_flag",
    # Advanced meteo features (wet bulb, wind chill, cloud dynamics)
    "wetbulb_last_obs", "wetbulb_mean_last_60min", "wetbulb_depression",
    "wetbulb_depression_mean_60min", "high_wetbulb_flag", "wetbulb_rate_last_30min",
    "windchill_last_obs", "windchill_depression", "windchill_mean_last_60min",
    "strong_windchill_flag", "windchill_warming_rate",
    "cloudcover_rate_last_30min", "cloudcover_volatility_60min",
    "clearing_trend_flag", "clouding_trend_flag", "cloud_regime", "cloud_stability_score",
    # Engineered features (transforms and interactions)
    "log_abs_obs_fcst_gap", "log_temp_std_last_60min", "log_intraday_range",
    "log_expected_delta_uncertainty", "temp_rate_last_30min_squared",
    "err_mean_sofar_squared", "obs_fcst_gap_squared", "fcst_multi_cv",
    "fcst_multi_range_pct", "humidity_x_temp_rate", "cloudcover_x_hour",
    "temp_ema_x_day_fraction", "station_city_gap_x_fcst_gap",
    # Quality and interaction features
    "confidence_weighted_gap", "remaining_upside", "expected_delta_uncertainty",
    "fcst_obs_ratio",
    "gap_x_hours_remaining", "obs_confidence", "fcst_remaining_potential",
    # Time context
    "hours_since_market_open",
    "log_hours_to_close", "log_minutes_since_open",
    # Phase indicators
    "is_heating_phase", "is_cooling_phase", "is_plateau_phase",
    "is_event_day",
    # Market features (bid/ask, volume, momentum) - toggled via include_market
    "market_yes_bid", "market_yes_ask",
    "market_bid_ask_spread", "market_mid_price",
    "bid_change_last_30min", "bid_change_last_60min", "bid_momentum_30min",
    "volume_last_30min", "volume_last_60min", "cumulative_volume_today",
    "has_recent_trade", "open_interest",
    # Additional engineered features from parquet
    "city_warmer_flag", "fcst_importance_weight",
    "fcst_multi_median", "fcst_multi_t1_t2_diff",
    "temp_cv_sofar", "temp_x_day_fraction", "temp_x_hours_remaining", "temp_zscore_vs_forecast",
    # NOAA model guidance features (NBM, HRRR, NDFD)
    "nbm_peak_window_max_f", "nbm_peak_window_revision_1h_f",
    "hrrr_peak_window_max_f", "hrrr_peak_window_revision_1h_f",
    "ndfd_tmax_T1_f", "ndfd_drift_T2_to_T1_f",
    "hrrr_minus_nbm_peak_window_max_f", "ndfd_minus_vc_T1_f",
    # Z-score features (dimensionless, normalized by 30d obs at 15:00)
    "nbm_t15_z_30d_f", "hrrr_t15_z_30d_f", "hrrr_minus_nbm_t15_z_30d_f",
    # Candle microstructure features (logit-based, 15-min window)
    "c_logit_mid_last", "c_logit_mom_15m", "c_logit_vol_15m", "c_logit_surprise_15m",
    "c_spread_pct_mean_15m", "c_mid_range_pct_15m", "c_trade_frac_15m", "c_synth_frac_15m",
]

# Market features list (for toggle logic in get_feature_columns)
MARKET_FEATURE_COLS: list[str] = [
    "market_yes_bid", "market_yes_ask",
    "market_bid_ask_spread", "market_mid_price",
    "bid_change_last_30min", "bid_change_last_60min", "bid_momentum_30min",
    "volume_last_30min", "volume_last_60min", "cumulative_volume_today",
    "has_recent_trade", "open_interest",
]

CATEGORICAL_FEATURE_COLS: list[str] = [
    "city",
]

# Delta classes for the Δ-model
# Range [-12, +12] chosen for 38-hour market-clock window:
# - Covers 90.4% of data (clips 5.2% below, 4.4% above)
# - Symmetric range handles overnight cooling and daytime warming
# - Extreme events (< -12 or > +12) are clipped to boundaries
# - 25 ordinal classes with good representation across range
DELTA_CLASSES = list(range(-12, 13))  # -12, -11, ..., 0, ..., +11, +12 (25 classes)


def get_feature_columns(
    include_forecast: bool = True,
    include_lags: bool = True,
    include_market: bool = True,
    include_momentum: bool = True,
    include_meteo: bool = True,
) -> tuple[list[str], list[str]]:
    """Get lists of numeric and categorical feature columns.

    Args:
        include_forecast: Whether to include forecast-related features
        include_lags: Whether to include lag features
        include_market: Whether to include Kalshi market features (bid/ask, volume)
        include_momentum: Whether to include temperature momentum features (rate, ema, std)
        include_meteo: Whether to include weather obs features (humidity, wind, cloud)

    Returns:
        Tuple of (numeric_cols, categorical_cols)
    """
    num_cols = NUMERIC_FEATURE_COLS.copy()

    if not include_forecast:
        forecast_cols = [
            "fcst_prev_max_f", "fcst_prev_min_f", "fcst_prev_mean_f",
            "fcst_prev_std_f", "fcst_prev_q10_f", "fcst_prev_q25_f",
            "fcst_prev_q50_f", "fcst_prev_q75_f", "fcst_prev_q90_f",
            "fcst_prev_frac_part", "fcst_prev_hour_of_max", "t_forecast_base",
            # Feature Group 1
            "fcst_prev_distance_to_int", "fcst_prev_near_boundary_flag",
            # Feature Group 2
            "fcst_peak_temp_f", "fcst_peak_hour_float",
            "fcst_peak_band_width_min", "fcst_peak_step_minutes",
            # Feature Group 3
            "fcst_drift_num_leads", "fcst_drift_std_f",
            "fcst_drift_max_upside_f", "fcst_drift_max_downside_f",
            "fcst_drift_mean_delta_f", "fcst_drift_slope_f_per_lead",
            # Feature Group 4
            "fcst_humidity_mean", "fcst_humidity_min", "fcst_humidity_max", "fcst_humidity_range",
            "fcst_cloudcover_mean", "fcst_cloudcover_min", "fcst_cloudcover_max", "fcst_cloudcover_range",
            "fcst_dewpoint_mean", "fcst_dewpoint_min", "fcst_dewpoint_max", "fcst_dewpoint_range",
            "fcst_humidity_morning_mean", "fcst_humidity_afternoon_mean",
            # Error features
            "err_mean_sofar", "err_std_sofar", "err_max_pos_sofar",
            "err_max_neg_sofar", "err_abs_mean_sofar", "err_last1h",
            "err_last3h_mean",
        ]
        num_cols = [c for c in num_cols if c not in forecast_cols]

    if not include_lags:
        lag_cols = [
            "settle_f_lag1", "settle_f_lag2", "settle_f_lag7",
            "vc_max_f_lag1", "vc_max_f_lag7", "delta_vcmax_lag1",
        ]
        num_cols = [c for c in num_cols if c not in lag_cols]

    if not include_market:
        # Remove all market features from MARKET_FEATURE_COLS
        num_cols = [c for c in num_cols if c not in MARKET_FEATURE_COLS]

    if not include_momentum:
        # Remove temperature momentum features
        momentum_keywords = [
            'temp_mean_last', 'temp_std_last', 'temp_max_last',
            'temp_rate_', 'temp_acceleration', 'temp_ema_',
            'intraday_range', 'minutes_since_max_observed'
        ]
        num_cols = [c for c in num_cols if not any(kw in c for kw in momentum_keywords)]

    if not include_meteo:
        # Remove meteorological observation features
        meteo_keywords = [
            'humidity_', 'windspeed_', 'windgust_', 'cloudcover_',
            'wetbulb_', 'windchill_', 'cloud_regime', 'cloud_stability',
            'clearing_trend', 'clouding_trend', 'high_wetbulb',
            'strong_windchill', 'high_cloud_flag', 'clear_sky_flag',
            'high_humidity_flag', 'strong_wind_flag'
        ]
        num_cols = [c for c in num_cols if not any(kw in c for kw in meteo_keywords)]

    return num_cols, CATEGORICAL_FEATURE_COLS.copy()
