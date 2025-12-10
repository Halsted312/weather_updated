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


# =============================================================================
# UNIFIED SNAPSHOT BUILDER - Single source of truth for training AND inference
# =============================================================================

def build_snapshot(
    city: str,
    day: date,
    temps_sofar: list[float],
    timestamps_sofar: list[datetime],
    cutoff_time: datetime,
    fcst_daily: Optional[dict] = None,
    fcst_hourly_df: Optional[pd.DataFrame] = None,
    fcst_daily_multi: Optional[pd.DataFrame] = None,
    settle_f: Optional[int] = None,
) -> Optional[dict]:
    """
    UNIFIED snapshot builder for both training and inference.

    This is the SINGLE SOURCE OF TRUTH for feature computation.
    Training and inference use identical code paths.

    Args:
        city: City identifier
        day: Target date
        temps_sofar: Observed temperatures up to cutoff_time
        timestamps_sofar: Corresponding timestamps
        cutoff_time: Snapshot datetime (only use data before this)
        fcst_daily: T-1 daily forecast dict (optional)
        fcst_hourly_df: T-1 hourly forecast DataFrame (optional)
        fcst_daily_multi: Multi-lead daily forecasts DataFrame (optional)
        settle_f: Ground truth settlement (only for training, None for inference)

    Returns:
        Feature dictionary, or None if insufficient data

    Note:
        - If settle_f is provided: computes delta target (training mode)
        - If settle_f is None: skips delta computation (inference mode)
    """
    if not temps_sofar:
        return None

    if len(temps_sofar) < MIN_SAMPLES:
        return None

    # Extract snapshot_hour from cutoff_time
    snapshot_hour = cutoff_time.hour

    # === CORE FEATURES (always computed) ===

    # Partial-day features from observations
    partial_day_fs = compute_partial_day_features(temps_sofar)
    if not partial_day_fs.features:
        return None

    t_base = partial_day_fs["t_base"]
    vc_max_f_sofar = partial_day_fs["vc_max_f_sofar"]

    # Shape features
    shape_fs = compute_shape_features(temps_sofar, timestamps_sofar, t_base)

    # Rule features (settle_f can be None for inference)
    rules_fs = compute_rule_features(temps_sofar, settle_f)

    # Calendar features
    calendar_fs = compute_calendar_features(day, cutoff_time=cutoff_time)

    # Quality features
    expected_samples = estimate_expected_samples(cutoff_time=cutoff_time)
    quality_fs = compute_quality_features(temps_sofar, timestamps_sofar, expected_samples)

    # Build base row
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

    # === DELTA TARGET (only for training) ===
    if settle_f is not None:
        row["settle_f"] = settle_f
        delta_info = compute_delta_target(settle_f, vc_max_f_sofar)
        row.update(delta_info)

    # === FORECAST FEATURES ===
    if fcst_daily is not None:
        fcst_max_f = fcst_daily.get("tempmax_f")

        # Static forecast features
        if fcst_hourly_df is not None and not fcst_hourly_df.empty:
            fcst_temps = fcst_hourly_df["temp_f"].dropna().tolist()
        elif fcst_max_f is not None:
            fcst_temps = [fcst_max_f]
        else:
            fcst_temps = []

        fcst_static_fs = compute_forecast_static_features(fcst_temps)
        row.update(fcst_static_fs.to_dict())

        # Forecast delta features
        fcst_delta_fs = compute_forecast_delta_features(fcst_max_f, vc_max_f_sofar)
        row.update(fcst_delta_fs.to_dict())

        # Forecast error features (need hourly alignment)
        if fcst_hourly_df is not None and not fcst_hourly_df.empty:
            # Filter forecast to hours before snapshot
            fcst_hourly_sofar = fcst_hourly_df[
                pd.to_datetime(fcst_hourly_df["target_datetime_local"]).dt.hour < snapshot_hour
            ]

            # Build obs_df for alignment
            obs_df = pd.DataFrame({
                "datetime_local": timestamps_sofar,
                "temp_f": temps_sofar,
            })

            fcst_series, obs_series = align_forecast_to_observations(
                fcst_hourly_sofar, obs_df
            )
            fcst_error_fs = compute_forecast_error_features(fcst_series, obs_series)
            row.update(fcst_error_fs.to_dict())

        # Feature Group 2: Peak window (from hourly curve)
        if fcst_hourly_df is not None and not fcst_hourly_df.empty:
            try:
                tmp = fcst_hourly_df.sort_values("target_datetime_local").copy()
                temps = tmp["temp_f"].dropna().tolist()
                times = pd.to_datetime(tmp["target_datetime_local"]).tolist()
                step_min = int((times[1] - times[0]).total_seconds() / 60) if len(times) > 1 else 60
                fcst_peak_fs = compute_forecast_peak_window_features(temps, times, step_min)
                row.update(fcst_peak_fs.to_dict())
            except Exception as e:
                logger.debug(f"Could not compute peak window features: {e}")

        # Feature Group 3: Forecast drift (from multi-lead daily)
        if fcst_daily_multi is not None and not fcst_daily_multi.empty:
            fcst_drift_fs = compute_forecast_drift_features(fcst_daily_multi)
            row.update(fcst_drift_fs.to_dict())

        # Feature Group 4: Multivar static (humidity, cloudcover, dewpoint)
        if fcst_hourly_df is not None and not fcst_hourly_df.empty:
            try:
                fcst_multivar_fs = compute_forecast_multivar_static_features(fcst_hourly_df)
                row.update(fcst_multivar_fs.to_dict())
            except Exception as e:
                logger.debug(f"Could not compute multivar features: {e}")

    return row


# =============================================================================
# DATASET BUILDER (for training) - Uses unified build_snapshot
# =============================================================================

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
        from models.features.interpolation import interpolate_cloudcover_from_hourly

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
            fcst_daily_multi = None

            if include_forecast_features:
                fcst_daily = load_historical_forecast_daily(session, city_id, day, basis_date)
                fcst_hourly_df = load_historical_forecast_hourly(session, city_id, day, basis_date)
                fcst_daily_multi = load_historical_forecast_daily_multi(session, city_id, day, max_lead_days=6)

                # Interpolate cloudcover from hourly forecast to observation timestamps
                if fcst_hourly_df is not None and not fcst_hourly_df.empty:
                    day_obs = interpolate_cloudcover_from_hourly(day_obs, fcst_hourly_df)

            # Build snapshot for each hour
            for snapshot_hour in snapshot_hours:
                # Create cutoff datetime
                cutoff_time = datetime(day.year, day.month, day.day, snapshot_hour, 0, 0)

                # Filter observations to before cutoff
                day_obs_copy = day_obs.copy()
                day_obs_copy["datetime_local"] = pd.to_datetime(day_obs_copy["datetime_local"])
                obs_sofar = day_obs_copy[day_obs_copy["datetime_local"] < cutoff_time]

                if len(obs_sofar) < MIN_SAMPLES:
                    continue

                # Interpolate small gaps
                obs_sofar = obs_sofar.copy()
                obs_sofar["temp_f"] = interpolate_small_gaps(obs_sofar["temp_f"])

                # Extract temps and timestamps
                valid_mask = obs_sofar["temp_f"].notna()
                temps_sofar = obs_sofar.loc[valid_mask, "temp_f"].tolist()
                timestamps_sofar = obs_sofar.loc[valid_mask, "datetime_local"].tolist()

                if not temps_sofar:
                    continue

                # Use UNIFIED build_snapshot
                row = build_snapshot(
                    city=city_id,
                    day=day,
                    temps_sofar=temps_sofar,
                    timestamps_sofar=timestamps_sofar,
                    cutoff_time=cutoff_time,
                    fcst_daily=fcst_daily,
                    fcst_hourly_df=fcst_hourly_df,
                    fcst_daily_multi=fcst_daily_multi,
                    settle_f=settle_f,  # Training mode: provide settlement
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


# =============================================================================
# DEPRECATED - Keep for backward compatibility but redirect to unified function
# =============================================================================

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
    """DEPRECATED: Use build_snapshot() instead.

    This function is kept for backward compatibility only.
    """
    import warnings
    warnings.warn(
        "build_single_snapshot is deprecated, use build_snapshot instead",
        DeprecationWarning,
        stacklevel=2
    )

    # Convert DataFrame to lists
    cutoff = datetime(day.year, day.month, day.day, snapshot_hour, 0, 0)
    day_obs_df = day_obs_df.copy()
    day_obs_df["datetime_local"] = pd.to_datetime(day_obs_df["datetime_local"])
    obs_sofar = day_obs_df[day_obs_df["datetime_local"] < cutoff]

    if len(obs_sofar) < MIN_SAMPLES:
        return None

    obs_sofar = obs_sofar.copy()
    obs_sofar["temp_f"] = interpolate_small_gaps(obs_sofar["temp_f"])

    valid_mask = obs_sofar["temp_f"].notna()
    temps_sofar = obs_sofar.loc[valid_mask, "temp_f"].tolist()
    timestamps_sofar = obs_sofar.loc[valid_mask, "datetime_local"].tolist()

    return build_snapshot(
        city=city,
        day=day,
        temps_sofar=temps_sofar,
        timestamps_sofar=timestamps_sofar,
        cutoff_time=cutoff,
        fcst_daily=fcst_daily if include_forecast else None,
        fcst_hourly_df=fcst_hourly_df if include_forecast else None,
        fcst_daily_multi=None,  # Not supported in old API
        settle_f=settle_f,
    )


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
    """DEPRECATED: Use build_snapshot() instead.

    This function is kept for backward compatibility only.
    """
    import warnings
    warnings.warn(
        "build_snapshot_for_inference is deprecated, use build_snapshot instead",
        DeprecationWarning,
        stacklevel=2
    )

    # Handle legacy parameters
    if cutoff_time is None and snapshot_hour is not None:
        cutoff_time = datetime.combine(day, datetime.min.time()).replace(hour=snapshot_hour, minute=0)
    elif cutoff_time is None:
        raise ValueError("Must provide either cutoff_time or snapshot_hour")

    result = build_snapshot(
        city=city,
        day=day,
        temps_sofar=temps_sofar,
        timestamps_sofar=timestamps_sofar,
        cutoff_time=cutoff_time,
        fcst_daily=fcst_daily,
        fcst_hourly_df=fcst_hourly_df,
        fcst_daily_multi=None,  # Not supported in old API
        settle_f=None,  # Inference mode: no settlement
    )

    if result is None:
        raise ValueError("Insufficient data for snapshot")

    return result
