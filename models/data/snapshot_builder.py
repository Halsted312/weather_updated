"""
Snapshot dataset construction for temperature Δ-models.

This module builds the training dataset of (city, day, snapshot_hour) rows
with all features computed from partial-day data. The key constraint is
that features at snapshot τ use ONLY data with datetime_local < τ.

A "snapshot" represents the model's view at a specific time during the day:
- What temperatures have been observed so far?
- What did yesterday's forecast predict?
- How is today tracking vs forecast?

The dataset is built by iterating over all (city, day, snapshot_hour)
combinations and computing features at each snapshot time.

Snapshot hours: [10, 12, 14, 16, 18, 20, 22, 23] local time
- Earlier snapshots have more uncertainty (less data)
- Later snapshots are more confident (more observed temps)

Example:
    >>> from models.data.snapshot_builder import build_snapshot_dataset
    >>> from src.db.connection import get_db_session
    >>> with get_db_session() as session:
    ...     df = build_snapshot_dataset(
    ...         cities=['chicago'],
    ...         start_date=date(2024, 6, 1),
    ...         end_date=date(2024, 8, 31),
    ...         session=session,
    ...     )
    >>> df.columns  # city, day, snapshot_hour, settle_f, delta, features...
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from models.data.loader import (
    load_vc_observations,
    load_settlements,
    load_historical_forecast_daily,
    load_historical_forecast_hourly,
    load_historical_forecast_daily_multi,
    load_historical_forecast_15min,
)
from models.features.partial_day import compute_partial_day_features, compute_delta_target
from models.features.shape import compute_shape_features
from models.features.rules import compute_rule_features
from models.features.calendar import compute_calendar_features, add_lag_features_to_dataframe
from models.features.quality import compute_quality_features, estimate_expected_samples
from models.features.forecast import (
    compute_forecast_static_features,
    compute_forecast_error_features,
    compute_forecast_delta_features,
    align_forecast_to_observations,
    compute_forecast_peak_window_features,
    compute_forecast_drift_features,
    compute_forecast_multivar_static_features,
)
from models.features.base import compose_features, FeatureSet

logger = logging.getLogger(__name__)

# Standard snapshot hours (local time)
SNAPSHOT_HOURS = [10, 12, 14, 16, 18, 20, 22, 23]

# Minimum samples required at a snapshot to include it
MIN_SAMPLES = 12  # ~1 hour of 5-min data

# Maximum consecutive NaN values to interpolate
MAX_INTERPOLATE_GAP = 3  # ~15 minutes at 5-min intervals


def interpolate_small_gaps(series: pd.Series, max_gap: int = MAX_INTERPOLATE_GAP) -> pd.Series:
    """Interpolate small gaps in temperature data.

    Uses linear interpolation for gaps up to max_gap consecutive NaN values.
    Larger gaps are left as NaN.

    Args:
        series: Temperature series with potential NaN values
        max_gap: Maximum consecutive NaN values to interpolate (default: 3)

    Returns:
        Series with small gaps filled via linear interpolation
    """
    if series.isna().sum() == 0:
        return series

    # Use pandas interpolate with limit to only fill small gaps
    result = series.interpolate(method='linear', limit=max_gap, limit_direction='both')
    return result


def build_snapshot_dataset(
    cities: list[str],
    start_date: date,
    end_date: date,
    session: Session,
    snapshot_hours: Optional[list[int]] = None,
    output_path: Optional[Path] = None,
    include_forecast_features: bool = True,
) -> pd.DataFrame:
    """Build full snapshot-level feature table for training.

    Iterates over all (city, day, snapshot_hour) combinations and computes
    features at each snapshot time using only data available up to that time.

    Args:
        cities: List of city identifiers (e.g., ['chicago'])
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        session: Database session
        snapshot_hours: List of snapshot hours (default: SNAPSHOT_HOURS)
        output_path: Optional path to save parquet file
        include_forecast_features: Whether to compute forecast features

    Returns:
        DataFrame with one row per (city, day, snapshot_hour) containing:
            - city, day, snapshot_hour: Identifiers
            - settle_f: Ground truth settlement
            - t_base: Baseline (rounded max so far)
            - delta: Target variable (settle_f - t_base)
            - All feature columns
    """
    if snapshot_hours is None:
        snapshot_hours = SNAPSHOT_HOURS

    all_rows = []

    for city_id in cities:
        logger.info(f"Building snapshots for {city_id} from {start_date} to {end_date}")

        # Load all observations and settlements for the date range
        settlements_df = load_settlements(session, city_id, start_date, end_date)
        if settlements_df.empty:
            logger.warning(f"No settlements for {city_id}, skipping")
            continue

        obs_df = load_vc_observations(session, city_id, start_date, end_date)
        if obs_df.empty:
            logger.warning(f"No observations for {city_id}, skipping")
            continue

        # Add date column to observations
        obs_df["day"] = pd.to_datetime(obs_df["datetime_local"]).dt.date

        # Load hourly forecast for cloudcover interpolation
        # (cloudcover not available in 5-min obs, only in hourly forecasts)
        from models.features.interpolation import interpolate_cloudcover_from_hourly

        # We'll interpolate cloudcover per day when we have fcst_hourly_df
        # (see below in the daily loop after loading fcst_hourly_df)

        # Process each day
        for _, settle_row in settlements_df.iterrows():
            day = settle_row["date_local"]
            settle_f = int(settle_row["tmax_final"])

            # Get observations for this day
            day_obs = obs_df[obs_df["day"] == day].copy()
            if day_obs.empty:
                continue

            # Load T-1 forecast (once per day)
            basis_date = day - timedelta(days=1)
            fcst_daily = None
            fcst_hourly_df = None
            fcst_peak_fs = None
            fcst_drift_fs = None
            fcst_multivar_fs = None

            if include_forecast_features:
                fcst_daily = load_historical_forecast_daily(session, city_id, day, basis_date)
                fcst_hourly_df = load_historical_forecast_hourly(session, city_id, day, basis_date)

                # Interpolate cloudcover from hourly forecast to observation timestamps
                # (cloudcover only available hourly, not in 5-min observations)
                if fcst_hourly_df is not None and not fcst_hourly_df.empty:
                    day_obs = interpolate_cloudcover_from_hourly(day_obs, fcst_hourly_df)

                # Compute per-day forecast features (reused across all snapshots)

                # Feature Group 2: Peak window (from hourly curve)
                if fcst_hourly_df is not None and not fcst_hourly_df.empty:
                    tmp = fcst_hourly_df.sort_values("target_datetime_local").copy()
                    temps = tmp["temp_f"].dropna().tolist()
                    times = pd.to_datetime(tmp["target_datetime_local"]).tolist()
                    step_min = int((times[1] - times[0]).total_seconds() / 60) if len(times) > 1 else 60
                    fcst_peak_fs = compute_forecast_peak_window_features(temps, times, step_min)

                # Feature Group 3: Forecast drift (from multi-lead daily)
                fcst_daily_multi = load_historical_forecast_daily_multi(session, city_id, day, max_lead_days=6)
                fcst_drift_fs = compute_forecast_drift_features(fcst_daily_multi)

                # Feature Group 4: Multivar static (from hourly data - has cloudcover)
                # Note: Minute data does NOT have cloudcover, so we use hourly
                if fcst_hourly_df is not None and not fcst_hourly_df.empty:
                    fcst_multivar_fs = compute_forecast_multivar_static_features(fcst_hourly_df)
                else:
                    fcst_multivar_fs = None

            # Build snapshot for each hour
            for snapshot_hour in snapshot_hours:
                row = build_single_snapshot(
                    city=city_id,
                    day=day,
                    snapshot_hour=snapshot_hour,
                    day_obs_df=day_obs,
                    settle_f=settle_f,
                    fcst_daily=fcst_daily,
                    fcst_hourly_df=fcst_hourly_df,
                    include_forecast=include_forecast_features,
                    fcst_peak_fs=fcst_peak_fs,
                    fcst_drift_fs=fcst_drift_fs,
                    fcst_multivar_fs=fcst_multivar_fs,
                )

                if row is not None:
                    all_rows.append(row)

        logger.info(f"Built {len([r for r in all_rows if r['city'] == city_id])} snapshots for {city_id}")

    if not all_rows:
        logger.warning("No snapshots built!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Add lag features (requires full dataset)
    df = add_lag_features_to_dataframe(df)

    # Sort by city, day, snapshot_hour
    df = df.sort_values(["city", "day", "snapshot_hour"]).reset_index(drop=True)

    logger.info(f"Built {len(df)} total snapshots")

    # Optionally save to parquet
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved snapshot dataset to {output_path}")

    return df


def build_single_snapshot(
    city: str,
    day: date,
    snapshot_hour: int,
    day_obs_df: pd.DataFrame,
    settle_f: int,
    fcst_daily: Optional[dict] = None,
    fcst_hourly_df: Optional[pd.DataFrame] = None,
    include_forecast: bool = True,
    fcst_peak_fs: Optional[FeatureSet] = None,
    fcst_drift_fs: Optional[FeatureSet] = None,
    fcst_multivar_fs: Optional[FeatureSet] = None,
) -> Optional[dict]:
    """Build feature row for one snapshot.

    Args:
        city: City identifier
        day: Target date
        snapshot_hour: Local hour of snapshot (0-23)
        day_obs_df: DataFrame with day's observations (datetime_local, temp_f, ...)
        settle_f: Ground truth settlement
        fcst_daily: T-1 daily forecast dict (optional)
        fcst_hourly_df: T-1 hourly forecast DataFrame (optional)
        include_forecast: Whether to compute forecast features
        fcst_peak_fs: Feature Group 2 - peak window features (optional)
        fcst_drift_fs: Feature Group 3 - forecast drift features (optional)
        fcst_multivar_fs: Feature Group 4 - multivar static features (optional)

    Returns:
        Dictionary with all features, or None if insufficient data
    """
    # Create cutoff datetime
    cutoff = datetime(day.year, day.month, day.day, snapshot_hour, 0, 0)

    # Filter observations to before cutoff
    day_obs_df = day_obs_df.copy()
    day_obs_df["datetime_local"] = pd.to_datetime(day_obs_df["datetime_local"])
    obs_sofar = day_obs_df[day_obs_df["datetime_local"] < cutoff]

    if len(obs_sofar) < MIN_SAMPLES:
        return None

    # Interpolate small gaps in temperature data (up to 3 consecutive NaN)
    obs_sofar = obs_sofar.copy()
    obs_sofar["temp_f"] = interpolate_small_gaps(obs_sofar["temp_f"])

    # Filter to rows with valid temp (after interpolation)
    valid_mask = obs_sofar["temp_f"].notna()
    temps_sofar = obs_sofar.loc[valid_mask, "temp_f"].tolist()
    timestamps_sofar = obs_sofar.loc[valid_mask, "datetime_local"].tolist()

    if not temps_sofar:
        return None

    # Compute partial-day features
    partial_day_fs = compute_partial_day_features(temps_sofar)
    if not partial_day_fs.features:
        return None

    t_base = partial_day_fs["t_base"]
    vc_max_f_sofar = partial_day_fs["vc_max_f_sofar"]

    # Compute delta target
    delta_info = compute_delta_target(settle_f, vc_max_f_sofar)

    # Compute shape features
    shape_fs = compute_shape_features(temps_sofar, timestamps_sofar, t_base)

    # Compute rule features
    rules_fs = compute_rule_features(temps_sofar, settle_f)

    # Compute calendar features
    calendar_fs = compute_calendar_features(day, snapshot_hour=snapshot_hour)

    # Compute quality features
    expected_samples = estimate_expected_samples(snapshot_hour=snapshot_hour)
    quality_fs = compute_quality_features(temps_sofar, timestamps_sofar, expected_samples)

    # Start building the row
    row = {
        "city": city,
        "day": day,
        "snapshot_hour": snapshot_hour,
        "settle_f": settle_f,
        **delta_info,
        **partial_day_fs.to_dict(),
        **shape_fs.to_dict(),
        **rules_fs.to_dict(),
        **calendar_fs.to_dict(),
        **quality_fs.to_dict(),
    }

    # Optionally compute forecast features
    if include_forecast and fcst_daily is not None:
        # Static forecast features
        fcst_max_f = fcst_daily.get("tempmax_f")

        if fcst_hourly_df is not None and not fcst_hourly_df.empty:
            fcst_temps = fcst_hourly_df["temp_f"].dropna().tolist()
        elif fcst_max_f is not None:
            # If no hourly, create synthetic series from daily
            fcst_temps = [fcst_max_f]
        else:
            fcst_temps = []

        fcst_static_fs = compute_forecast_static_features(fcst_temps)
        row.update(fcst_static_fs.to_dict())

        # Forecast error features (need hourly alignment)
        if fcst_hourly_df is not None and not fcst_hourly_df.empty:
            # Filter forecast to hours up to snapshot
            fcst_hourly_sofar = fcst_hourly_df[
                pd.to_datetime(fcst_hourly_df["target_datetime_local"]).dt.hour < snapshot_hour
            ]

            fcst_series, obs_series = align_forecast_to_observations(
                fcst_hourly_sofar,
                obs_sofar,
            )

            fcst_error_fs = compute_forecast_error_features(fcst_series, obs_series)
            row.update(fcst_error_fs.to_dict())

            # Forecast delta features
            fcst_delta_fs = compute_forecast_delta_features(fcst_max_f, vc_max_f_sofar)
            row.update(fcst_delta_fs.to_dict())
        else:
            # No forecast data - fill with None
            row.update({
                "err_mean_sofar": None,
                "err_std_sofar": None,
                "err_max_pos_sofar": None,
                "err_max_neg_sofar": None,
                "err_abs_mean_sofar": None,
                "err_last1h": None,
                "err_last3h_mean": None,
                "delta_vcmax_fcstmax_sofar": None,
                "fcst_remaining_potential": None,
            })
        # Add new feature groups (2, 3, 4)
        if fcst_peak_fs is not None:
            row.update(fcst_peak_fs.to_dict())
        if fcst_drift_fs is not None:
            row.update(fcst_drift_fs.to_dict())
        if fcst_multivar_fs is not None:
            row.update(fcst_multivar_fs.to_dict())
    else:
        # Fill forecast columns with None if not including
        forecast_cols = [
            "fcst_prev_max_f", "fcst_prev_min_f", "fcst_prev_mean_f", "fcst_prev_std_f",
            "fcst_prev_q10_f", "fcst_prev_q25_f", "fcst_prev_q50_f",
            "fcst_prev_q75_f", "fcst_prev_q90_f",
            "fcst_prev_frac_part", "fcst_prev_hour_of_max", "t_forecast_base",
            # Feature Group 1 (integer boundary)
            "fcst_prev_distance_to_int", "fcst_prev_near_boundary_flag",
            # Error features
            "err_mean_sofar", "err_std_sofar", "err_max_pos_sofar", "err_max_neg_sofar",
            "err_abs_mean_sofar", "err_last1h", "err_last3h_mean",
            "delta_vcmax_fcstmax_sofar", "fcst_remaining_potential",
            # Feature Group 2 (peak window)
            "fcst_peak_temp_f", "fcst_peak_hour_float",
            "fcst_peak_band_width_min", "fcst_peak_step_minutes",
            # Feature Group 3 (forecast drift)
            "fcst_drift_num_leads", "fcst_drift_std_f",
            "fcst_drift_max_upside_f", "fcst_drift_max_downside_f",
            "fcst_drift_mean_delta_f", "fcst_drift_slope_f_per_lead",
            # Feature Group 4 (multivar static)
            "fcst_humidity_mean", "fcst_humidity_min", "fcst_humidity_max", "fcst_humidity_range",
            "fcst_cloudcover_mean", "fcst_cloudcover_min", "fcst_cloudcover_max", "fcst_cloudcover_range",
            "fcst_dewpoint_mean", "fcst_dewpoint_min", "fcst_dewpoint_max", "fcst_dewpoint_range",
            "fcst_humidity_morning_mean", "fcst_humidity_afternoon_mean",
        ]
        for col in forecast_cols:
            row[col] = None

    return row


def build_snapshot_for_inference(
    city: str,
    day: date,
    temps_sofar: list[float],
    timestamps_sofar: list[datetime],
    cutoff_time: Optional[datetime] = None,
    snapshot_hour: Optional[int] = None,
    fcst_daily: Optional[dict] = None,
    fcst_hourly_df: Optional[pd.DataFrame] = None,
    fcst_peak_fs: Optional[FeatureSet] = None,
    fcst_drift_fs: Optional[FeatureSet] = None,
    fcst_multivar_fs: Optional[FeatureSet] = None,
) -> dict:
    """Build a single snapshot row for inference (no settle_f needed).

    Similar to build_single_snapshot but:
    - Doesn't require settlement (we're predicting it)
    - Returns dict suitable for model input
    - Supports both legacy snapshot_hour and new cutoff_time for tod_v1

    Args:
        city: City identifier
        day: Target date
        temps_sofar: Observed temperatures up to now
        timestamps_sofar: Corresponding timestamps
        cutoff_time: Full datetime for snapshot (tod_v1 models) - PRIMARY
        snapshot_hour: Legacy integer hour (baseline models) - DEPRECATED
        fcst_daily: T-1 forecast dict
        fcst_hourly_df: T-1 hourly forecast DataFrame
        fcst_peak_fs: Feature Group 2 - peak window features (optional)
        fcst_drift_fs: Feature Group 3 - forecast drift features (optional)
        fcst_multivar_fs: Feature Group 4 - multivar static features (optional)

    Returns:
        Feature dictionary ready for model inference
    """
    # Backward compatibility: support old snapshot_hour parameter
    if cutoff_time is None and snapshot_hour is not None:
        cutoff_time = datetime.combine(day, datetime.min.time()).replace(hour=snapshot_hour, minute=0)
    elif cutoff_time is None:
        raise ValueError("Must provide either cutoff_time or snapshot_hour")

    # Extract snapshot_hour for backward compat (some features may still use it)
    if snapshot_hour is None:
        snapshot_hour = cutoff_time.hour

    if not temps_sofar:
        raise ValueError("No temperature observations provided")

    # Compute features (no settle_f)
    partial_day_fs = compute_partial_day_features(temps_sofar)
    t_base = partial_day_fs["t_base"]

    shape_fs = compute_shape_features(temps_sofar, timestamps_sofar, t_base)
    rules_fs = compute_rule_features(temps_sofar, settle_f=None)  # No settlement
    calendar_fs = compute_calendar_features(day, cutoff_time=cutoff_time)

    expected_samples = estimate_expected_samples(cutoff_time=cutoff_time)
    quality_fs = compute_quality_features(temps_sofar, timestamps_sofar, expected_samples)

    row = {
        "city": city,
        "day": day,
        "snapshot_hour": snapshot_hour,
        "t_base": t_base,
        **partial_day_fs.to_dict(),
        **shape_fs.to_dict(),
        **rules_fs.to_dict(),
        **calendar_fs.to_dict(),
        **quality_fs.to_dict(),
    }

    # Forecast features
    if fcst_daily is not None:
        fcst_max_f = fcst_daily.get("tempmax_f")
        fcst_temps = []

        if fcst_hourly_df is not None and not fcst_hourly_df.empty:
            fcst_temps = fcst_hourly_df["temp_f"].dropna().tolist()

        fcst_static_fs = compute_forecast_static_features(fcst_temps)
        row.update(fcst_static_fs.to_dict())

        # Forecast delta
        fcst_delta_fs = compute_forecast_delta_features(
            fcst_max_f, partial_day_fs["vc_max_f_sofar"]
        )
        row.update(fcst_delta_fs.to_dict())

    # Add new feature groups (2, 3, 4)
    if fcst_peak_fs is not None:
        row.update(fcst_peak_fs.to_dict())
    if fcst_drift_fs is not None:
        row.update(fcst_drift_fs.to_dict())
    if fcst_multivar_fs is not None:
        row.update(fcst_multivar_fs.to_dict())

    return row
