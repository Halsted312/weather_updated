"""
Unified feature engineering pipeline for training and inference.

This module provides a single code path for all feature computation,
ensuring training/test/inference parity.

The core abstraction is `SnapshotContext` which holds all inputs needed
to compute features for a single point-in-time snapshot. The main function
`compute_snapshot_features()` orchestrates all feature modules.

Example:
    >>> from models.features.pipeline import SnapshotContext, compute_snapshot_features
    >>> ctx = SnapshotContext(
    ...     city="chicago",
    ...     event_date=date(2024, 7, 15),
    ...     cutoff_time=datetime(2024, 7, 15, 14, 30),
    ...     window_start=datetime(2024, 7, 14, 10, 0),  # D-1 10:00
    ...     temps_sofar=[72.1, 75.3, 80.2, 82.5],
    ...     timestamps_sofar=[...],
    ... )
    >>> features = compute_snapshot_features(ctx, include_labels=False)
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

# Import all feature modules
from models.features.partial_day import compute_partial_day_features
from models.features.shape import compute_shape_features
from models.features.rules import compute_rule_features
from models.features.calendar import compute_calendar_features
from models.features.quality import compute_quality_features
from models.features.forecast import (
    compute_forecast_static_features,
    compute_forecast_error_features,
    compute_forecast_delta_features,
    compute_multi_horizon_features,
    align_forecast_to_observations,
    # Feature Groups 2-4
    compute_forecast_peak_window_features,
    compute_forecast_drift_features,
    compute_forecast_multivar_static_features,
)
from models.features.momentum import compute_momentum_features, compute_volatility_features
from models.features.interactions import (
    compute_interaction_features,
    compute_regime_features,
    compute_derived_forecast_features,
    compute_time_confidence_features,
)
from models.features.station_city import compute_station_city_features
from models.features.meteo import compute_meteo_features
from models.features.meteo_advanced import compute_meteo_advanced_features
from models.features.engineered import compute_engineered_features
from models.features.market import compute_market_features

from models.features.imputation import (
    fill_all_forecast_nulls,
    fill_market_nulls,
    fill_station_city_nulls,
    fill_meteo_nulls,
    fill_meteo_advanced_nulls,
    fill_engineered_nulls,
    fill_regime_nulls,
    fill_interaction_nulls,
    fill_derived_nulls,
    apply_imputation,
)


# =============================================================================
# Core Data Classes
# =============================================================================

@dataclass
class SnapshotContext:
    """All inputs needed to compute features for one snapshot.

    This is the single input type for compute_snapshot_features().
    It contains all the data needed to compute any feature, but individual
    feature groups gracefully handle missing data (e.g., no candles = null market features).

    Attributes:
        city: City identifier (e.g., 'chicago', 'austin')
        event_date: The settlement date (D)
        cutoff_time: The snapshot timestamp (must be <= event close)
        window_start: Start of the observation window (e.g., D-1 10:00 for market-clock)

        temps_sofar: List of temperatures observed up to cutoff_time
        timestamps_sofar: List of corresponding timestamps (local time)
        obs_df: Full observation DataFrame with meteo columns (optional)

        fcst_daily: T-1 daily forecast dict with tempmax_f, etc. (optional)
        fcst_hourly_df: T-1 hourly forecast DataFrame (optional)

        candles_df: Market candle DataFrame (optional)
        city_obs_df: City-aggregate observation DataFrame for station-city features (optional)

        settle_f: Settlement temperature (training only)
    """
    city: str
    event_date: date
    cutoff_time: datetime
    window_start: datetime

    # Observations (required)
    temps_sofar: list[float]
    timestamps_sofar: list[datetime]
    obs_df: Optional[pd.DataFrame] = None

    # Forecast (optional)
    fcst_daily: Optional[dict] = None  # City-level T-1 forecast
    fcst_daily_station: Optional[dict] = None  # Station-level T-1 forecast
    fcst_hourly_df: Optional[pd.DataFrame] = None
    fcst_multi: Optional[dict[int, Optional[dict]]] = None  # Multi-horizon {lead_day: forecast}

    # Market data (optional)
    candles_df: Optional[pd.DataFrame] = None

    # Station-city comparison (optional)
    city_obs_df: Optional[pd.DataFrame] = None

    # NOAA model guidance (NBM, HRRR, NDFD) - optional
    more_apis: Optional[dict[str, dict[str, Any]]] = None

    # 30-day obs stats at 15:00 local (for z-score normalization)
    obs_t15_mean_30d_f: Optional[float] = None
    obs_t15_std_30d_f: Optional[float] = None

    # Labels (training only)
    settle_f: Optional[int] = None


# =============================================================================
# City One-Hot Encoding
# =============================================================================

CITIES = ["chicago", "austin", "denver", "los_angeles", "miami", "philadelphia"]


def _city_one_hot(city: str) -> dict[str, int]:
    """Create one-hot encoding for city.

    Args:
        city: City identifier

    Returns:
        Dictionary with one-hot city features
    """
    return {f"city_{c}": int(city == c) for c in CITIES}


# =============================================================================
# Market Clock Features
# =============================================================================

def _compute_market_clock_features(ctx: SnapshotContext) -> dict[str, Any]:
    """Compute market-clock specific features.

    These features measure time relative to market open and close,
    which is more relevant for trading than calendar time.

    Args:
        ctx: SnapshotContext with cutoff_time, event_date, window_start

    Returns:
        Dictionary with market-clock features
    """
    cutoff_time = ctx.cutoff_time
    event_date = ctx.event_date
    market_open = ctx.window_start

    # Normalize to naive datetimes to avoid mixing aware/naive
    if hasattr(cutoff_time, 'tzinfo') and cutoff_time.tzinfo is not None:
        cutoff_time = cutoff_time.replace(tzinfo=None)
    if hasattr(market_open, 'tzinfo') and market_open.tzinfo is not None:
        market_open = market_open.replace(tzinfo=None)

    # Minutes since market open (using window_start as market open)
    delta_seconds = (cutoff_time - market_open).total_seconds()
    minutes_since_market_open = max(0, int(delta_seconds // 60))
    hours_since_market_open = minutes_since_market_open / 60.0

    # Market close is event_date 23:55
    market_close = datetime.combine(event_date, datetime.min.time()).replace(hour=23, minute=55)
    hours_to_event_close = max(0.0, (market_close - cutoff_time).total_seconds() / 3600.0)

    # Is this D-1 or D?
    is_d_minus_1 = int(cutoff_time.date() == (event_date - timedelta(days=1)))
    is_event_day = 1 - is_d_minus_1

    return {
        "minutes_since_market_open": minutes_since_market_open,
        "hours_since_market_open": hours_since_market_open,
        "hours_to_event_close": hours_to_event_close,
        "is_d_minus_1": is_d_minus_1,
        "is_event_day": is_event_day,
    }


# =============================================================================
# Identity Features
# =============================================================================

def _compute_identity_features(ctx: SnapshotContext) -> dict[str, Any]:
    """Compute identity and timing features.

    These are metadata columns useful for grouping/filtering but not for modeling.
    Note: 'day' is an alias for 'event_date' for backward compatibility.
    """
    return {
        "city": ctx.city,
        "event_date": ctx.event_date,
        "day": ctx.event_date,  # Alias for backward compatibility
        "cutoff_time": ctx.cutoff_time,
    }


# =============================================================================
# Forecast Features
# =============================================================================

def _add_forecast_features(features: dict[str, Any], ctx: SnapshotContext) -> None:
    """Add all forecast-related features.

    Modifies features dict in place.
    """
    # Static forecast features (from T-1 daily forecast)
    if ctx.fcst_daily:
        tempmax_f = ctx.fcst_daily.get("tempmax_f")
        tempmin_f = ctx.fcst_daily.get("tempmin_f")

        features["fcst_prev_max_f"] = tempmax_f  # City-level forecast
        features["fcst_prev_min_f"] = tempmin_f

        # Station-level forecast features (if available)
        if ctx.fcst_daily_station:
            station_tempmax_f = ctx.fcst_daily_station.get("tempmax_f")
            features["fcst_station_max_f"] = station_tempmax_f
            # Compute city-station gap (valuable signal - varies by city)
            if tempmax_f is not None and station_tempmax_f is not None:
                features["fcst_city_station_gap"] = station_tempmax_f - tempmax_f
            else:
                features["fcst_city_station_gap"] = None
        else:
            features["fcst_station_max_f"] = None
            features["fcst_city_station_gap"] = None

        # Compute other static features from hourly
        if ctx.fcst_hourly_df is not None and not ctx.fcst_hourly_df.empty:
            hourly_temps = ctx.fcst_hourly_df["temp_f"].dropna().tolist()
            static_fs = compute_forecast_static_features(hourly_temps)
            # Don't overwrite fcst_prev_max_f/min_f from daily forecast with hourly-derived values
            # Daily tempmax_f is the official forecast high, max(hourly_temps) is different
            static_dict = static_fs.to_dict()
            static_dict.pop("fcst_prev_max_f", None)
            static_dict.pop("fcst_prev_min_f", None)
            features.update(static_dict)
        else:
            # Use daily values if no hourly
            if tempmax_f is not None:
                features["fcst_prev_mean_f"] = tempmax_f  # Approximate
                features["fcst_prev_std_f"] = None
                features["fcst_prev_q10_f"] = None
                features["fcst_prev_q25_f"] = None
                features["fcst_prev_q50_f"] = tempmax_f
                features["fcst_prev_q75_f"] = None
                features["fcst_prev_q90_f"] = None
                features["fcst_prev_frac_part"] = tempmax_f - round(tempmax_f) if tempmax_f else None
                features["fcst_prev_hour_of_max"] = None
                features["t_forecast_base"] = int(round(tempmax_f)) if tempmax_f else None
    else:
        fill_all_forecast_nulls(features)
        return

    # Forecast error features (comparing forecast to observations)
    if ctx.fcst_hourly_df is not None and not ctx.fcst_hourly_df.empty and ctx.obs_df is not None:
        fcst_temps, obs_temps = align_forecast_to_observations(
            ctx.fcst_hourly_df,
            ctx.obs_df,
            obs_datetime_col="datetime_local",
            fcst_datetime_col="target_datetime_local",
        )
        error_fs = compute_forecast_error_features(fcst_temps, obs_temps)
        features.update(error_fs.to_dict())
    else:
        # Fill error features with None
        features["err_mean_sofar"] = None
        features["err_std_sofar"] = None
        features["err_max_pos_sofar"] = None
        features["err_max_neg_sofar"] = None
        features["err_abs_mean_sofar"] = None
        features["err_last1h"] = None
        features["err_last3h_mean"] = None

    # Forecast delta features
    fcst_max = features.get("fcst_prev_max_f")
    obs_max = features.get("vc_max_f_sofar")
    delta_fs = compute_forecast_delta_features(fcst_max, obs_max if obs_max else 0)
    features.update(delta_fs.to_dict())


# =============================================================================
# Derived Features
# =============================================================================

def _add_derived_features(features: dict[str, Any]) -> None:
    """Add derived features computed from existing features.

    These are combinations of other features that are useful for modeling.
    Modifies features dict in place.
    """
    # obs_fcst_max_gap: upside potential = fcst_max - vc_max_sofar
    fcst_max = features.get("fcst_prev_max_f")
    vc_max = features.get("vc_max_f_sofar")
    if fcst_max is not None and vc_max is not None:
        features["obs_fcst_max_gap"] = fcst_max - vc_max
    else:
        features["obs_fcst_max_gap"] = None

    # hours_until_fcst_max: fcst_hour_of_max - current_hour
    fcst_hour_of_max = features.get("fcst_prev_hour_of_max")
    hour = features.get("hour")
    if fcst_hour_of_max is not None and hour is not None:
        features["hours_until_fcst_max"] = fcst_hour_of_max - hour
    else:
        features["hours_until_fcst_max"] = None

    # above_fcst_flag: 1 if vc_max > fcst_max
    if fcst_max is not None and vc_max is not None:
        features["above_fcst_flag"] = int(vc_max > fcst_max)
    else:
        features["above_fcst_flag"] = None

    # day_fraction: (hour - 6) / 18  (fraction of heating day elapsed)
    if hour is not None:
        features["day_fraction"] = max(0.0, (hour - 6) / 18.0)
    else:
        features["day_fraction"] = None


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_temps_with_times(obs_df: Optional[pd.DataFrame]) -> list[tuple[datetime, float]]:
    """Extract (datetime, temp) tuples from observation DataFrame."""
    if obs_df is None or obs_df.empty:
        return []

    result = []
    for _, row in obs_df.iterrows():
        dt = row.get("datetime_local")
        temp = row.get("temp_f")
        if dt is not None and temp is not None and not pd.isna(temp):
            # Ensure datetime is naive or convert
            if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            result.append((dt, float(temp)))

    return sorted(result, key=lambda x: x[0])


def _has_meteo_columns(obs_df: pd.DataFrame) -> bool:
    """Check if obs_df has meteo columns."""
    meteo_cols = ["humidity", "windspeed_mph", "cloudcover"]
    return any(col in obs_df.columns for col in meteo_cols)


# =============================================================================
# Main Feature Computation Function
# =============================================================================

def compute_snapshot_features(
    ctx: SnapshotContext,
    include_labels: bool = False,
) -> dict[str, Any]:
    """Compute all features for a single snapshot.

    This is THE unified feature computation function used by both
    training and inference. All features from V1 + V2 are computed here.

    Args:
        ctx: SnapshotContext with all input data
        include_labels: If True, include delta and settle_f labels

    Returns:
        Dictionary with all features
    """
    features: dict[str, Any] = {}

    # ==========================================================================
    # 1. Identity & timing (metadata, not for modeling)
    # ==========================================================================
    features.update(_compute_identity_features(ctx))

    # ==========================================================================
    # 2. Market clock features
    # ==========================================================================
    features.update(_compute_market_clock_features(ctx))

    # ==========================================================================
    # 3. City one-hot encoding
    # ==========================================================================
    features.update(_city_one_hot(ctx.city))

    # ==========================================================================
    # 4. Partial day features (core temperature statistics)
    # ==========================================================================
    partial_fs = compute_partial_day_features(ctx.temps_sofar)
    features.update(partial_fs.to_dict())

    # Extract t_base for shape features
    t_base = features.get("t_base", 0)
    if t_base is None:
        t_base = 0

    # ==========================================================================
    # 5. Shape features (plateau vs spike detection)
    # ==========================================================================
    shape_fs = compute_shape_features(
        ctx.temps_sofar,
        ctx.timestamps_sofar,
        t_base,
        step_minutes=5,
    )
    features.update(shape_fs.to_dict())

    # ==========================================================================
    # 6. Rule features (predictions from deterministic rules)
    # ==========================================================================
    rules_fs = compute_rule_features(
        ctx.temps_sofar,
        settle_f=ctx.settle_f if include_labels else None,
    )
    features.update(rules_fs.to_dict())

    # ==========================================================================
    # 7. Calendar features (time encoding)
    # ==========================================================================
    calendar_fs = compute_calendar_features(
        ctx.event_date,
        cutoff_time=ctx.cutoff_time,
    )
    features.update(calendar_fs.to_dict())

    # ==========================================================================
    # 8. Quality features (data completeness)
    # ==========================================================================
    quality_fs = compute_quality_features(
        ctx.temps_sofar,
        ctx.timestamps_sofar,
    )
    features.update(quality_fs.to_dict())

    # ==========================================================================
    # 9. Forecast features (T-1 forecast and forecast errors)
    # ==========================================================================
    if ctx.fcst_daily or ctx.fcst_hourly_df is not None:
        _add_forecast_features(features, ctx)
    else:
        fill_all_forecast_nulls(features)

    # ==========================================================================
    # 9b. Multi-horizon forecast features (T-6 through T-1 statistics)
    # ==========================================================================
    if ctx.fcst_multi:
        multi_fs = compute_multi_horizon_features(ctx.fcst_multi)
        features.update(multi_fs.to_dict())
    else:
        # Fill with None if no multi-horizon data
        features["fcst_multi_std"] = None
        features["fcst_multi_mean"] = None
        features["fcst_multi_drift"] = None

    # ==========================================================================
    # 9c. Peak window features (Feature Group 2: timing and duration of forecast peak)
    # ==========================================================================
    if ctx.fcst_hourly_df is not None and not ctx.fcst_hourly_df.empty:
        temps_f = ctx.fcst_hourly_df["temp_f"].dropna().tolist()
        # Get timestamps from target_datetime_local
        ts_col = "target_datetime_local"
        if ts_col in ctx.fcst_hourly_df.columns:
            timestamps = pd.to_datetime(ctx.fcst_hourly_df[ts_col]).tolist()
            step_minutes = 60  # Hourly data
            peak_fs = compute_forecast_peak_window_features(temps_f, timestamps, step_minutes)
            features.update(peak_fs.to_dict())
        else:
            features["fcst_peak_temp_f"] = None
            features["fcst_peak_hour_float"] = None
            features["fcst_peak_band_width_min"] = None
            features["fcst_peak_step_minutes"] = None
    else:
        features["fcst_peak_temp_f"] = None
        features["fcst_peak_hour_float"] = None
        features["fcst_peak_band_width_min"] = None
        features["fcst_peak_step_minutes"] = None

    # ==========================================================================
    # 9d. Forecast drift features (Feature Group 3: how forecast evolved over leads)
    # ==========================================================================
    if ctx.fcst_multi:
        # Convert fcst_multi dict to DataFrame format expected by drift features
        drift_rows = []
        for lead_day, fcst in ctx.fcst_multi.items():
            if fcst is not None and fcst.get("tempmax_f") is not None:
                drift_rows.append({
                    "lead_days": lead_day,
                    "tempmax_f": fcst.get("tempmax_f"),
                    "tempmin_f": fcst.get("tempmin_f"),
                    "humidity": fcst.get("humidity"),
                    "cloudcover": fcst.get("cloudcover"),
                })
        if drift_rows:
            drift_df = pd.DataFrame(drift_rows)
            drift_fs = compute_forecast_drift_features(drift_df)
            features.update(drift_fs.to_dict())
        else:
            features["fcst_drift_num_leads"] = None
            features["fcst_drift_std_f"] = None
            features["fcst_drift_max_upside_f"] = None
            features["fcst_drift_max_downside_f"] = None
            features["fcst_drift_mean_delta_f"] = None
            features["fcst_drift_slope_f_per_lead"] = None
    else:
        features["fcst_drift_num_leads"] = None
        features["fcst_drift_std_f"] = None
        features["fcst_drift_max_upside_f"] = None
        features["fcst_drift_max_downside_f"] = None
        features["fcst_drift_mean_delta_f"] = None
        features["fcst_drift_slope_f_per_lead"] = None

    # ==========================================================================
    # 9e. Multivar static features (Feature Group 4: humidity/cloudcover/dewpoint)
    # ==========================================================================
    if ctx.fcst_hourly_df is not None and not ctx.fcst_hourly_df.empty:
        # Use hourly data for multivar (has cloudcover, unlike minute data)
        multivar_fs = compute_forecast_multivar_static_features(ctx.fcst_hourly_df)
        features.update(multivar_fs.to_dict())
    else:
        features["fcst_humidity_mean"] = None
        features["fcst_humidity_min"] = None
        features["fcst_humidity_max"] = None
        features["fcst_humidity_range"] = None
        features["fcst_cloudcover_mean"] = None
        features["fcst_cloudcover_min"] = None
        features["fcst_cloudcover_max"] = None
        features["fcst_cloudcover_range"] = None
        features["fcst_dewpoint_mean"] = None
        features["fcst_dewpoint_min"] = None
        features["fcst_dewpoint_max"] = None
        features["fcst_dewpoint_range"] = None
        features["fcst_humidity_morning_mean"] = None
        features["fcst_humidity_afternoon_mean"] = None

    # ==========================================================================
    # 10. Momentum features (temperature trajectory)
    # ==========================================================================
    temps_with_times = list(zip(ctx.timestamps_sofar, ctx.temps_sofar))
    momentum_fs = compute_momentum_features(temps_with_times)
    features.update(momentum_fs.to_dict())

    volatility_fs = compute_volatility_features(temps_with_times)
    features.update(volatility_fs.to_dict())

    # ==========================================================================
    # 11. Derived features (computed from other features)
    # ==========================================================================
    _add_derived_features(features)

    # ==========================================================================
    # 12. Interaction features (cross-feature combinations)
    # ==========================================================================
    interaction_fs = compute_interaction_features(
        vc_max_f_sofar=features.get("vc_max_f_sofar"),
        fcst_prev_max_f=features.get("fcst_prev_max_f"),
        fcst_prev_mean_f=features.get("fcst_prev_mean_f"),
        fcst_prev_std_f=features.get("fcst_prev_std_f"),
        hours_to_event_close=features.get("hours_to_event_close"),
        minutes_since_market_open=features.get("minutes_since_market_open"),
        day_fraction=features.get("day_fraction"),
        obs_fcst_max_gap=features.get("obs_fcst_max_gap"),
    )
    features.update(interaction_fs.to_dict())

    # ==========================================================================
    # 13. Regime features (heating/plateau/cooling phase)
    # ==========================================================================
    regime_fs = compute_regime_features(
        temp_rate_last_30min=features.get("temp_rate_last_30min"),
        minutes_since_max_observed=features.get("minutes_since_max_observed"),
        snapshot_hour=features.get("hour"),
    )
    features.update(regime_fs.to_dict())

    # ==========================================================================
    # 13b. Time-confidence features (observation confidence increases with time)
    # ==========================================================================
    time_conf_fs = compute_time_confidence_features(
        hours_since_market_open=features.get("hours_since_market_open"),
        is_event_day=features.get("is_event_day"),
        vc_max_f_sofar=features.get("vc_max_f_sofar"),
        fcst_prev_max_f=features.get("fcst_prev_max_f"),
    )
    features.update(time_conf_fs.to_dict())

    # ==========================================================================
    # 14. Market features (from candle data)
    # ==========================================================================
    if ctx.candles_df is not None and not ctx.candles_df.empty:
        market_fs = compute_market_features(ctx.candles_df, ctx.cutoff_time)
        features.update(market_fs.to_dict())

        # 14b. Candle microstructure (logit-based, 15-min aggregates)
        from models.features.candles_micro import compute_candles_micro_features
        candles_micro_fs = compute_candles_micro_features(ctx.candles_df, ctx.cutoff_time)
        features.update(candles_micro_fs.to_dict())
    else:
        fill_market_nulls(features)
        from models.features.imputation import fill_candles_micro_nulls
        fill_candles_micro_nulls(features)

    # ==========================================================================
    # 15. Station-city features (comparing station vs city aggregate)
    # ==========================================================================
    if ctx.city_obs_df is not None and not ctx.city_obs_df.empty and ctx.obs_df is not None:
        station_temps = _extract_temps_with_times(ctx.obs_df)
        city_temps = _extract_temps_with_times(ctx.city_obs_df)
        sc_fs = compute_station_city_features(station_temps, city_temps)
        features.update(sc_fs.to_dict())
    else:
        fill_station_city_nulls(features)

    # ==========================================================================
    # 16. Meteo features (humidity, wind, cloud cover)
    # ==========================================================================
    if ctx.obs_df is not None and _has_meteo_columns(ctx.obs_df):
        meteo_fs = compute_meteo_features(ctx.obs_df, ctx.cutoff_time)
        features.update(meteo_fs.to_dict())
    else:
        fill_meteo_nulls(features)

    # ==========================================================================
    # 16b. Advanced meteo features (wet bulb, wind chill, cloud dynamics)
    # ==========================================================================
    if ctx.obs_df is not None and _has_meteo_columns(ctx.obs_df):
        meteo_adv_fs = compute_meteo_advanced_features(ctx.obs_df, ctx.cutoff_time)
        features.update(meteo_adv_fs.to_dict())
    else:
        fill_meteo_advanced_nulls(features)

    # ==========================================================================
    # 16c. Engineered features (transforms and interactions)
    # ==========================================================================
    engineered_fs = compute_engineered_features(features)
    features.update(engineered_fs.to_dict())
    # No null-filling needed here since engineered features handle None gracefully

    # ==========================================================================
    # 16d. NOAA model guidance (NBM, HRRR, NDFD)
    # ==========================================================================
    from models.features.more_apis import compute_more_apis_features
    vc_t1_tempmax = ctx.fcst_daily.get("tempmax_f") if ctx.fcst_daily else None
    more_apis_fs = compute_more_apis_features(
        ctx.more_apis,
        vc_t1_tempmax,
        ctx.obs_t15_mean_30d_f,
        ctx.obs_t15_std_30d_f
    )
    features.update(more_apis_fs.to_dict())

    # ==========================================================================
    # 17. Labels (training only)
    # ==========================================================================
    if include_labels and ctx.settle_f is not None:
        # Delta = settlement - T-1 STATION forecast (settlement comes from station obs)
        # Fall back to city forecast if station not available
        fcst_max = features.get("fcst_station_max_f") or features.get("fcst_prev_max_f")
        if fcst_max is not None:
            delta = ctx.settle_f - fcst_max
            features["delta"] = int(round(delta))
        else:
            # Fallback if no forecast available - use 0 delta (will be filtered)
            features["delta"] = 0
        features["settle_f"] = ctx.settle_f

    # ==========================================================================
    # 18. Final imputation (ensure all columns exist)
    # ==========================================================================
    apply_imputation(features)

    return features


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_snapshot_features_simple(
    city: str,
    event_date: date,
    cutoff_time: datetime,
    temps_sofar: list[float],
    timestamps_sofar: list[datetime],
    settle_f: Optional[int] = None,
    fcst_daily: Optional[dict] = None,
    include_labels: bool = False,
) -> dict[str, Any]:
    """Simplified interface for basic feature computation.

    For cases where you don't need market/station-city/meteo features.

    Args:
        city: City identifier
        event_date: Settlement date
        cutoff_time: Snapshot timestamp
        temps_sofar: Temperature observations
        timestamps_sofar: Timestamp for each observation
        settle_f: Settlement temperature (training only)
        fcst_daily: T-1 daily forecast dict
        include_labels: Include delta/settle_f in output

    Returns:
        Feature dictionary
    """
    # Default window_start to D-1 10:00 (market open)
    window_start = datetime.combine(
        event_date - timedelta(days=1),
        datetime.min.time()
    ).replace(hour=10, minute=0)

    ctx = SnapshotContext(
        city=city,
        event_date=event_date,
        cutoff_time=cutoff_time,
        window_start=window_start,
        temps_sofar=temps_sofar,
        timestamps_sofar=timestamps_sofar,
        fcst_daily=fcst_daily,
        settle_f=settle_f,
    )

    return compute_snapshot_features(ctx, include_labels=include_labels)
