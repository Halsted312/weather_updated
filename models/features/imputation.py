"""
Consolidated null-filling and imputation for feature pipeline.

This module provides all the null-filling functions needed when optional
data sources (forecasts, candles, city obs, meteo) are not available.

The imputation strategy is simple: fill with None (let the model handle it).
CatBoost handles None/NaN natively, so we don't need numeric imputation.
"""

from typing import Any


# =============================================================================
# Forecast Feature Null-Filling
# =============================================================================

FORECAST_STATIC_COLS = [
    "fcst_prev_max_f",
    "fcst_prev_min_f",
    "fcst_prev_mean_f",
    "fcst_prev_std_f",
    "fcst_prev_q10_f",
    "fcst_prev_q25_f",
    "fcst_prev_q50_f",
    "fcst_prev_q75_f",
    "fcst_prev_q90_f",
    "fcst_prev_frac_part",
    "fcst_prev_hour_of_max",
    "t_forecast_base",
]

FORECAST_ERROR_COLS = [
    "err_mean_sofar",
    "err_std_sofar",
    "err_max_pos_sofar",
    "err_max_neg_sofar",
    "err_abs_mean_sofar",
    "err_last1h",
    "err_last3h_mean",
    "delta_vcmax_fcstmax_sofar",
    "fcst_remaining_potential",
]


def fill_forecast_static_nulls(row: dict[str, Any]) -> None:
    """Fill forecast static features with None."""
    for col in FORECAST_STATIC_COLS:
        if col not in row:
            row[col] = None


def fill_forecast_error_nulls(row: dict[str, Any]) -> None:
    """Fill forecast error features with None."""
    for col in FORECAST_ERROR_COLS:
        if col not in row:
            row[col] = None


def fill_all_forecast_nulls(row: dict[str, Any]) -> None:
    """Fill all forecast features (static + error) with None."""
    fill_forecast_static_nulls(row)
    fill_forecast_error_nulls(row)


# =============================================================================
# Market Feature Null-Filling
# =============================================================================

MARKET_COLS = [
    "market_yes_bid",
    "market_yes_ask",
    "market_bid_ask_spread",
    "market_mid_price",
    "bid_change_last_30min",
    "bid_change_last_60min",
    "bid_momentum_30min",
    "volume_last_30min",
    "volume_last_60min",
    "cumulative_volume_today",
    "has_recent_trade",
    "open_interest",
]


def fill_market_nulls(row: dict[str, Any]) -> None:
    """Fill market-derived features with None.

    These require candle data which may not always be available.
    """
    for col in MARKET_COLS:
        if col not in row:
            row[col] = None


# =============================================================================
# Station-City Feature Null-Filling
# =============================================================================

STATION_CITY_COLS = [
    "station_city_temp_gap",
    "station_city_max_gap_sofar",
    "station_city_mean_gap_sofar",
    "city_warmer_flag",
    "station_city_gap_std",
    "station_city_gap_trend",
]


def fill_station_city_nulls(row: dict[str, Any]) -> None:
    """Fill station-city gap features with None.

    These require city-level observations which may not always be available.
    """
    for col in STATION_CITY_COLS:
        if col not in row:
            row[col] = None


# =============================================================================
# Meteo Feature Null-Filling
# =============================================================================

METEO_COLS = [
    "humidity_last_obs",
    "humidity_mean_last_60min",
    "humidity_std_last_60min",
    "high_humidity_flag",
    "windspeed_last_obs",
    "windspeed_max_last_60min",
    "windgust_max_last_60min",
    "strong_wind_flag",
    "cloudcover_last_obs",
    "cloudcover_mean_last_60min",
    "high_cloud_flag",
    "clear_sky_flag",
]


def fill_meteo_nulls(row: dict[str, Any]) -> None:
    """Fill meteorological features with None.

    These require meteo columns (humidity, windspeed, cloudcover) from obs_df
    which may not always be available.
    """
    for col in METEO_COLS:
        if col not in row:
            row[col] = None


# =============================================================================
# Advanced Meteo Feature Null-Filling
# =============================================================================

METEO_ADVANCED_COLS = [
    # Wet bulb
    "wetbulb_last_obs", "wetbulb_mean_last_60min", "wetbulb_depression",
    "wetbulb_depression_mean_60min", "high_wetbulb_flag", "wetbulb_rate_last_30min",
    # Wind chill
    "windchill_last_obs", "windchill_depression", "windchill_mean_last_60min",
    "strong_windchill_flag", "windchill_warming_rate",
    # Cloud dynamics
    "cloudcover_rate_last_30min", "cloudcover_volatility_60min",
    "clearing_trend_flag", "clouding_trend_flag", "cloud_regime", "cloud_stability_score",
]


def fill_meteo_advanced_nulls(row: dict[str, Any]) -> None:
    """Fill advanced meteorological features with None.

    These include wet bulb, wind chill, and cloud dynamics features.
    """
    for col in METEO_ADVANCED_COLS:
        if col not in row:
            row[col] = None


# =============================================================================
# Engineered Feature Null-Filling
# =============================================================================

ENGINEERED_COLS = [
    "log_abs_obs_fcst_gap", "log_temp_std_last_60min", "log_intraday_range",
    "log_expected_delta_uncertainty", "temp_rate_last_30min_squared",
    "err_mean_sofar_squared", "obs_fcst_gap_squared", "fcst_multi_cv",
    "fcst_multi_range_pct", "humidity_x_temp_rate", "cloudcover_x_hour",
    "temp_ema_x_day_fraction", "station_city_gap_x_fcst_gap",
]


def fill_engineered_nulls(row: dict[str, Any]) -> None:
    """Fill engineered transform and interaction features with None."""
    for col in ENGINEERED_COLS:
        if col not in row:
            row[col] = None


# =============================================================================
# Regime Feature Null-Filling
# =============================================================================

REGIME_COLS = [
    "is_heating_phase",
    "is_plateau_phase",
    "is_cooling_phase",
]


def fill_regime_nulls(row: dict[str, Any]) -> None:
    """Fill regime/phase features with None."""
    for col in REGIME_COLS:
        if col not in row:
            row[col] = None


# =============================================================================
# Interaction Feature Null-Filling
# =============================================================================

INTERACTION_COLS = [
    "temp_x_hours_remaining",
    "gap_x_hours_remaining",
    "temp_x_day_fraction",
    "fcst_obs_ratio",
    "fcst_obs_diff_squared",
    "log_minutes_since_open",
    "log_hours_to_close",
    "temp_zscore_vs_forecast",
]


def fill_interaction_nulls(row: dict[str, Any]) -> None:
    """Fill interaction features with None."""
    for col in INTERACTION_COLS:
        if col not in row:
            row[col] = None


# =============================================================================
# Derived Feature Null-Filling
# =============================================================================

DERIVED_COLS = [
    "obs_fcst_max_gap",
    "hours_until_fcst_max",
    "above_fcst_flag",
    "day_fraction",
]


def fill_derived_nulls(row: dict[str, Any]) -> None:
    """Fill derived features with None."""
    for col in DERIVED_COLS:
        if col not in row:
            row[col] = None


# =============================================================================
# Master Imputation Function
# =============================================================================

def apply_imputation(row: dict[str, Any]) -> None:
    """Apply final imputation to ensure all expected columns exist.

    This is the last step in feature computation. It ensures that all
    expected feature columns exist (even if None) so the model receives
    a consistent feature set.

    The imputation strategy is simple: fill missing values with None.
    CatBoost handles None/NaN natively, treating them as missing values.
    """
    # Ensure all optional feature groups have their columns
    fill_all_forecast_nulls(row)
    fill_market_nulls(row)
    fill_station_city_nulls(row)
    fill_meteo_nulls(row)
    fill_regime_nulls(row)
    fill_interaction_nulls(row)
    fill_derived_nulls(row)


# =============================================================================
# Column Lists for Export
# =============================================================================

ALL_IMPUTED_COLS = (
    FORECAST_STATIC_COLS
    + FORECAST_ERROR_COLS
    + MARKET_COLS
    + STATION_CITY_COLS
    + METEO_COLS
    + REGIME_COLS
    + INTERACTION_COLS
    + DERIVED_COLS
)
