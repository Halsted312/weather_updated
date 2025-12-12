"""
Market-Clock Snapshot Dataset Builder

Generates training datasets for the Market-Clock TOD v1 model, which spans
from market open (D-1 10:00 local) to market close (D 23:55 local).

Key differences from tod_dataset_builder:
- Time window spans D-1 10:00 to D 23:55 (~38 hours, ~456 snapshots per event)
- Observations loaded from both D-1 and D
- Includes market-clock features: minutes_since_market_open, is_d_minus_1, etc.
- City encoded as one-hot numeric features (not categorical)

Usage:
    from models.data.market_clock_dataset_builder import build_market_clock_snapshot_dataset

    df = build_market_clock_snapshot_dataset(
        cities=['chicago', 'austin', ...],
        start_date=date(2023, 1, 1),
        end_date=date(2025, 11, 27),
        session=db_session,
        snapshot_interval_min=5,  # 5-minute snapshots (default)
    )
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd
from sqlalchemy.orm import Session

from models.data.loader import (
    load_vc_observations,
    load_historical_forecast_daily,
    load_historical_forecast_hourly,
)
from models.features.partial_day import compute_partial_day_features, compute_delta_target
from models.features.shape import compute_shape_features
from models.features.rules import compute_rule_features
from models.features.calendar import compute_calendar_features
from models.features.forecast import (
    compute_forecast_static_features,
    compute_forecast_error_features,
    compute_forecast_delta_features,
    align_forecast_to_observations,
)
from models.features.base import compose_features

# NEW v2 features
from models.features.momentum import compute_momentum_features, compute_volatility_features
from models.features.market import compute_market_features
from models.features.station_city import compute_station_city_features
from models.features.interactions import compute_interaction_features, compute_regime_features
from models.features.meteo import compute_meteo_features


logger = logging.getLogger(__name__)

# All cities supported by the model
ALL_CITIES = ["chicago", "austin", "denver", "los_angeles", "miami", "philadelphia"]

# Minimum samples required to build a valid snapshot
MIN_SAMPLES = 6  # ~30 minutes of 5-min data


def build_market_clock_snapshot_dataset(
    cities: list[str],
    start_date: date,
    end_date: date,
    session: Session,
    snapshot_interval_min: int = 5,
    market_open_hour: int = 10,
    include_forecast_features: bool = True,
) -> pd.DataFrame:
    """Build training dataset with market-clock aware snapshots.

    For each (city, event_date D), creates snapshots from D-1 10:00 (market open)
    to D 23:55 (market close) at snapshot_interval_min intervals.

    Example with 5-min intervals:
        D-1 10:00, 10:05, ..., 23:55, D 00:00, ..., D 23:55
        = ~456 snapshots per event

    Args:
        cities: List of city identifiers
        start_date: First event date for training
        end_date: Last event date for training
        session: Database session
        snapshot_interval_min: Minutes between snapshots (default 5)
        market_open_hour: Hour when market opens on D-1 (default 10)
        include_forecast_features: Whether to include T-1 forecast features

    Returns:
        DataFrame with one row per (city, event_date, snapshot_datetime)
        Columns include: city, event_date, snapshot_datetime, market-clock features,
                         city one-hot, all feature columns, settle_f, t_base, delta
    """
    logger.info(
        f"Building market-clock dataset: {len(cities)} cities, "
        f"{(end_date - start_date).days + 1} event days, {snapshot_interval_min}-min intervals"
    )

    # Load settlement data for all cities/dates
    settlement_df = _load_settlement_data(session, cities, start_date, end_date)
    logger.info(f"Loaded {len(settlement_df)} settlement records")

    all_snapshots = []
    total_snapshots = 0

    for city in cities:
        logger.info(f"Building snapshots for {city}...")

        # Get settlement for this city
        city_settlement = settlement_df[settlement_df["city"] == city]
        if city_settlement.empty:
            logger.warning(f"No settlement data for {city}, skipping")
            continue

        # Load observations for this city (from start_date - 1 to end_date + 1 for buffer)
        obs_df = load_vc_observations(
            session,
            city_id=city,
            start_date=start_date - timedelta(days=1),
            end_date=end_date + timedelta(days=1),
        )

        if obs_df.empty:
            logger.warning(f"No observation data for {city}, skipping")
            continue

        city_snapshots = 0

        # Process each event date
        for single_day in pd.date_range(start_date, end_date, freq="D"):
            event_date = single_day.date()

            # Get settlement for this day
            day_settlement = city_settlement[city_settlement["date_local"] == event_date]
            if day_settlement.empty:
                continue  # No settlement data for this day

            settle_f = int(day_settlement.iloc[0]["tmax_final"])

            # Define market window
            market_open = datetime.combine(
                event_date - timedelta(days=1),
                datetime.min.time()
            ).replace(hour=market_open_hour, minute=0)

            market_close = datetime.combine(
                event_date,
                datetime.min.time()
            ).replace(hour=23, minute=55)

            # Filter observations for market window
            window_obs = obs_df[
                (obs_df["datetime_local"] >= market_open) &
                (obs_df["datetime_local"] <= market_close)
            ].copy()

            if window_obs.empty:
                continue

            # Load T-1 forecast (once per event)
            basis_date = event_date - timedelta(days=1)
            fcst_daily = None
            fcst_hourly_df = None

            if include_forecast_features:
                fcst_daily = load_historical_forecast_daily(session, city, event_date, basis_date)
                fcst_hourly_df = load_historical_forecast_hourly(session, city, event_date, basis_date)

            # Generate snapshot timestamps
            snapshot_times = _generate_snapshot_times(
                market_open, market_close, snapshot_interval_min
            )

            # Build snapshot for each timestamp
            for snapshot_ts in snapshot_times:
                # Filter observations up to this snapshot time
                obs_sofar = window_obs[window_obs["datetime_local"] <= snapshot_ts].copy()

                if obs_sofar.empty:
                    continue

                temps_sofar = obs_sofar["temp_f"].dropna().tolist()
                timestamps_sofar = obs_sofar.loc[
                    obs_sofar["temp_f"].notna(), "datetime_local"
                ].tolist()

                if len(temps_sofar) < MIN_SAMPLES:
                    continue

                # Build snapshot features
                try:
                    snapshot_row = build_market_clock_snapshot_for_training(
                        city=city,
                        event_date=event_date,
                        cutoff_time=snapshot_ts,
                        temps_sofar=temps_sofar,
                        timestamps_sofar=timestamps_sofar,
                        fcst_daily=fcst_daily,
                        fcst_hourly_df=fcst_hourly_df,
                        settle_f=settle_f,
                        market_open=market_open,
                    )

                    all_snapshots.append(snapshot_row)
                    city_snapshots += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to build snapshot for {city} {event_date} {snapshot_ts}: {e}"
                    )
                    continue

        logger.info(f"  Built {city_snapshots} snapshots for {city}")
        total_snapshots += city_snapshots

    logger.info(f"Total snapshots built: {total_snapshots}")

    if not all_snapshots:
        logger.warning("No snapshots built!")
        return pd.DataFrame()

    df = pd.DataFrame(all_snapshots)

    # Add lag features (operates on full DataFrame)
    if not df.empty:
        from models.features.calendar import add_lag_features_to_dataframe
        # Need to rename event_date to day for lag function compatibility
        df["day"] = df["event_date"]
        df = add_lag_features_to_dataframe(df)

    logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")

    return df


def build_market_clock_snapshot_for_training(
    city: str,
    event_date: date,
    cutoff_time: datetime,
    temps_sofar: list[float],
    timestamps_sofar: list[datetime],
    fcst_daily: Optional[dict],
    fcst_hourly_df: Optional[pd.DataFrame],
    settle_f: int,
    market_open: datetime,
) -> dict[str, Any]:
    """Build a single snapshot row for training (with settle_f label).

    This computes all features for one snapshot including:
    - Market-clock features (minutes_since_market_open, is_d_minus_1, etc.)
    - Partial-day features (vc_max_f_sofar, t_base, etc.)
    - Shape features (plateau, slopes)
    - Rule features (heuristic predictions)
    - Forecast features (T-1 forecast and errors)
    - Calendar features (time-of-day, day-of-year)
    - Quality features (missing fraction, gaps)
    - City one-hot encoding
    - Delta target

    Note on event_date vs cutoff_time:
        event_date is always the settlement date D (the day we're predicting for),
        even when cutoff_time is on D-1. Day-level calendar features (month, doy)
        are tied to event_date, not cutoff_time. This is intentional - we want
        the model to learn seasonal patterns for the target day.

    Args:
        city: City identifier
        event_date: Settlement date (D) - used for day-level calendar features
        cutoff_time: Snapshot timestamp (could be D-1 or D) - used for time-of-day features
        temps_sofar: Temperature observations up to cutoff
        timestamps_sofar: Corresponding timestamps
        fcst_daily: T-1 daily forecast dict
        fcst_hourly_df: T-1 hourly forecast DataFrame
        settle_f: Ground truth settlement temperature
        market_open: Market open time (D-1 10:00)

    Returns:
        Dictionary with all features for one training row
    """
    # 1. Market-clock features
    market_clock = _compute_market_clock_features(
        cutoff_time=cutoff_time,
        event_date=event_date,
        market_open=market_open,
    )

    # 2. Partial-day features
    partial_day_fs = compute_partial_day_features(temps_sofar)
    if not partial_day_fs.features:
        raise ValueError("Cannot compute partial-day features - empty temps_sofar")

    t_base = partial_day_fs["t_base"]
    vc_max_f_sofar = partial_day_fs["vc_max_f_sofar"]

    # 3. Delta target
    delta_info = compute_delta_target(settle_f, vc_max_f_sofar)

    # 4. Shape features
    shape_fs = compute_shape_features(temps_sofar, timestamps_sofar, t_base)

    # 5. Rule features
    rules_fs = compute_rule_features(temps_sofar, settle_f)

    # 6. Calendar features (physical time-of-day)
    # Note: day is event_date (D), but cutoff_time may be D-1 or D
    calendar_fs = compute_calendar_features(day=event_date, cutoff_time=cutoff_time)

    # 7. Quality features (market-clock aware)
    quality_fs = _compute_quality_features_market_clock(
        timestamps_sofar=timestamps_sofar,
        market_open=market_open,
        step_minutes=5,
    )

    # 8. City one-hot
    city_one_hot = _city_one_hot(city)

    # Build row
    row: dict[str, Any] = {
        # Identity
        "city": city,
        "event_date": event_date,
        "snapshot_datetime": cutoff_time,
        # Market-clock features
        **market_clock,
        # City one-hot
        **city_one_hot,
        # Labels
        "settle_f": settle_f,
        **delta_info,
        # All other features
        **partial_day_fs.to_dict(),
        **shape_fs.to_dict(),
        **rules_fs.to_dict(),
        **calendar_fs.to_dict(),
        **quality_fs,
    }

    # 9. Forecast features (if available)
    # Professor's requirement: training and inference must handle forecast features identically
    # - Always compute static + delta features when fcst_daily exists
    # - Only compute error features when hourly data is present
    # - Otherwise explicitly fill error fields with None
    if fcst_daily is not None:
        fcst_max_f = fcst_daily.get("tempmax_f")

        # Static forecast features (always compute when fcst_daily exists)
        if fcst_hourly_df is not None and not fcst_hourly_df.empty:
            fcst_temps = fcst_hourly_df["temp_f"].dropna().tolist()
        elif fcst_max_f is not None:
            fcst_temps = [fcst_max_f]
        else:
            fcst_temps = []

        fcst_static_fs = compute_forecast_static_features(fcst_temps)
        row.update(fcst_static_fs.to_dict())

        # Forecast delta features (always compute when fcst_daily exists)
        fcst_delta_fs = compute_forecast_delta_features(fcst_max_f, vc_max_f_sofar)
        row.update(fcst_delta_fs.to_dict())

        # Forecast error features (only when hourly data is present)
        if fcst_hourly_df is not None and not fcst_hourly_df.empty:
            # Create obs DataFrame for alignment
            obs_df = pd.DataFrame({
                "datetime_local": timestamps_sofar,
                "temp_f": temps_sofar,
            })

            # Filter forecast to times up to cutoff
            fcst_hourly_sofar = fcst_hourly_df[
                pd.to_datetime(fcst_hourly_df["target_datetime_local"]) <= cutoff_time
            ]

            fcst_series, obs_series = align_forecast_to_observations(
                fcst_hourly_sofar, obs_df
            )

            fcst_error_fs = compute_forecast_error_features(fcst_series, obs_series)
            row.update(fcst_error_fs.to_dict())
        else:
            # No hourly forecast - fill error features with None
            _fill_forecast_error_nulls(row)
    else:
        # No forecast at all - fill all forecast columns with None
        _fill_all_forecast_nulls(row)

    # 10. Derived features for correlation with delta
    _add_derived_features(row)

    # =========================================================================
    # 11. NEW v2 Features: Momentum, Interactions (Market and Station-City
    #     require additional data loading done at dataset level)
    # =========================================================================

    # 11a. Momentum features (rolling windows, rate of change, EMA)
    temps_with_times = list(zip(timestamps_sofar, temps_sofar))
    momentum_fs = compute_momentum_features(temps_with_times)
    row.update(momentum_fs.to_dict())

    # 11b. Volatility features
    volatility_fs = compute_volatility_features(temps_with_times)
    row.update(volatility_fs.to_dict())

    # 11c. Interaction features
    interaction_fs = compute_interaction_features(
        vc_max_f_sofar=row.get("vc_max_f_sofar"),
        fcst_prev_max_f=row.get("fcst_prev_max_f"),
        fcst_prev_mean_f=row.get("fcst_prev_mean_f"),
        fcst_prev_std_f=row.get("fcst_prev_std_f"),
        hours_to_event_close=row.get("hours_to_event_close"),
        minutes_since_market_open=row.get("minutes_since_market_open"),
        day_fraction=row.get("day_fraction"),
        obs_fcst_max_gap=row.get("obs_fcst_max_gap"),
    )
    row.update(interaction_fs.to_dict())

    # 11d. Market features - placeholder (requires candle data loaded separately)
    # These will be filled with None here and populated in build_v2_dataset()
    _fill_market_feature_nulls(row)

    # 11e. Station-City features - placeholder (requires city obs loaded separately)
    # These will be filled with None here and populated in build_v2_dataset()
    _fill_station_city_feature_nulls(row)

    # 11f. Regime/phase features (computed from momentum + time)
    regime_fs = compute_regime_features(
        temp_rate_last_30min=row.get("temp_rate_last_30min"),
        minutes_since_max_observed=row.get("minutes_since_max_observed"),
        snapshot_hour=row.get("hour"),
    )
    row.update(regime_fs.to_dict())

    # 11g. Meteo features - placeholder (requires obs_df with meteo columns)
    # These will be filled with None here and populated in build_v2_dataset()
    _fill_meteo_feature_nulls(row)

    return row


def build_market_clock_snapshot_for_inference(
    city: str,
    event_date: date,
    cutoff_time: datetime,
    temps_sofar: list[float],
    timestamps_sofar: list[datetime],
    fcst_daily: Optional[dict],
    fcst_hourly_df: Optional[pd.DataFrame],
    market_open: datetime,
) -> dict[str, Any]:
    """Build a single snapshot for inference (no settle_f needed).

    Same as training version but without labels.

    Args:
        city: City identifier
        event_date: Settlement date (D)
        cutoff_time: Snapshot timestamp
        temps_sofar: Temperature observations up to cutoff
        timestamps_sofar: Corresponding timestamps
        fcst_daily: T-1 daily forecast dict
        fcst_hourly_df: T-1 hourly forecast DataFrame
        market_open: Market open time (D-1 10:00)

    Returns:
        Dictionary with all features for inference
    """
    # 1. Market-clock features
    market_clock = _compute_market_clock_features(
        cutoff_time=cutoff_time,
        event_date=event_date,
        market_open=market_open,
    )

    # 2. Partial-day features
    partial_day_fs = compute_partial_day_features(temps_sofar)
    if not partial_day_fs.features:
        raise ValueError("Cannot compute partial-day features - empty temps_sofar")

    t_base = partial_day_fs["t_base"]
    vc_max_f_sofar = partial_day_fs["vc_max_f_sofar"]

    # 3. Shape features
    shape_fs = compute_shape_features(temps_sofar, timestamps_sofar, t_base)

    # 4. Rule features (no settle_f for inference)
    rules_fs = compute_rule_features(temps_sofar, settle_f=None)

    # 5. Calendar features
    calendar_fs = compute_calendar_features(day=event_date, cutoff_time=cutoff_time)

    # 6. Quality features
    quality_fs = _compute_quality_features_market_clock(
        timestamps_sofar=timestamps_sofar,
        market_open=market_open,
        step_minutes=5,
    )

    # 7. City one-hot
    city_one_hot = _city_one_hot(city)

    # Build row
    row: dict[str, Any] = {
        "city": city,
        "event_date": event_date,
        "snapshot_datetime": cutoff_time,
        "snapshot_hour": cutoff_time.hour,  # For backward compat
        **market_clock,
        **city_one_hot,
        "t_base": t_base,
        **partial_day_fs.to_dict(),
        **shape_fs.to_dict(),
        **rules_fs.to_dict(),
        **calendar_fs.to_dict(),
        **quality_fs,
    }

    # 8. Forecast features
    # Professor's requirement: training and inference must handle forecast features identically
    # - Always compute static + delta features when fcst_daily exists
    # - Only compute error features when hourly data is present
    # - Otherwise explicitly fill error fields with None
    if fcst_daily is not None:
        fcst_max_f = fcst_daily.get("tempmax_f")

        # Static forecast features (always compute when fcst_daily exists)
        if fcst_hourly_df is not None and not fcst_hourly_df.empty:
            fcst_temps = fcst_hourly_df["temp_f"].dropna().tolist()
        elif fcst_max_f is not None:
            fcst_temps = [fcst_max_f]
        else:
            fcst_temps = []

        fcst_static_fs = compute_forecast_static_features(fcst_temps)
        row.update(fcst_static_fs.to_dict())

        # Forecast delta features (always compute when fcst_daily exists)
        fcst_delta_fs = compute_forecast_delta_features(fcst_max_f, vc_max_f_sofar)
        row.update(fcst_delta_fs.to_dict())

        # Error features only when hourly data is present
        if fcst_hourly_df is None or fcst_hourly_df.empty:
            _fill_forecast_error_nulls(row)
    else:
        _fill_all_forecast_nulls(row)

    # 9. Derived features
    _add_derived_features(row)

    # =========================================================================
    # 10. NEW v2 Features: Momentum, Interactions
    # =========================================================================

    # 10a. Momentum features
    temps_with_times = list(zip(timestamps_sofar, temps_sofar))
    momentum_fs = compute_momentum_features(temps_with_times)
    row.update(momentum_fs.to_dict())

    # 10b. Volatility features
    volatility_fs = compute_volatility_features(temps_with_times)
    row.update(volatility_fs.to_dict())

    # 10c. Interaction features
    interaction_fs = compute_interaction_features(
        vc_max_f_sofar=row.get("vc_max_f_sofar"),
        fcst_prev_max_f=row.get("fcst_prev_max_f"),
        fcst_prev_mean_f=row.get("fcst_prev_mean_f"),
        fcst_prev_std_f=row.get("fcst_prev_std_f"),
        hours_to_event_close=row.get("hours_to_event_close"),
        minutes_since_market_open=row.get("minutes_since_market_open"),
        day_fraction=row.get("day_fraction"),
        obs_fcst_max_gap=row.get("obs_fcst_max_gap"),
    )
    row.update(interaction_fs.to_dict())

    # 10d. Market features placeholder
    _fill_market_feature_nulls(row)

    # 10e. Station-City features placeholder
    _fill_station_city_feature_nulls(row)

    # 10f. Regime/phase features
    regime_fs = compute_regime_features(
        temp_rate_last_30min=row.get("temp_rate_last_30min"),
        minutes_since_max_observed=row.get("minutes_since_max_observed"),
        snapshot_hour=row.get("hour"),
    )
    row.update(regime_fs.to_dict())

    # 10g. Meteo features placeholder
    _fill_meteo_feature_nulls(row)

    return row


def _compute_market_clock_features(
    cutoff_time: datetime,
    event_date: date,
    market_open: datetime,
) -> dict[str, Any]:
    """Compute market-clock specific features.

    Args:
        cutoff_time: Current snapshot timestamp
        event_date: Settlement date (D)
        market_open: Market open time (D-1 10:00)

    Returns:
        Dictionary with market-clock features
    """
    # Minutes since market open
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


def _compute_quality_features_market_clock(
    timestamps_sofar: list[datetime],
    market_open: datetime,
    step_minutes: int = 5,  # VC observation cadence, NOT snapshot interval
) -> dict[str, Any]:
    """Compute data quality features using market-clock semantics.

    Expected samples based on minutes since market open, not calendar day.

    Note: step_minutes is the Visual Crossing observation cadence (5 minutes),
    NOT the snapshot interval used for training. This is intentional - we want
    to measure data quality relative to the underlying data source's granularity.

    Args:
        timestamps_sofar: Observation timestamps
        market_open: Market open time
        step_minutes: VC observation cadence (default 5 min), not snapshot interval

    Returns:
        Dictionary with quality features
    """
    if not timestamps_sofar:
        return {
            "missing_fraction_sofar": 1.0,
            "max_gap_minutes": 24 * 60,  # Max possible
            "edge_max_flag": 0,
        }

    n = len(timestamps_sofar)

    # Expected samples since market open
    last_ts = max(timestamps_sofar)
    minutes_since_open = max(0, int((last_ts - market_open).total_seconds() // 60))
    expected_samples = max(1, minutes_since_open // step_minutes)

    # Missing fraction (clamped to [0, 1])
    mf = 1.0 - (n / expected_samples)
    missing_fraction_sofar = max(0.0, min(1.0, mf))

    # Max gap between observations
    if n >= 2:
        timestamps_sorted = sorted(timestamps_sofar)
        gaps = [
            (t2 - t1).total_seconds() / 60.0
            for t1, t2 in zip(timestamps_sorted[:-1], timestamps_sorted[1:])
        ]
        max_gap_minutes = max(gaps) if gaps else 0.0
    else:
        max_gap_minutes = 0.0

    # Edge max flag: set to 0 for v1 (can implement later if needed)
    edge_max_flag = 0

    return {
        "missing_fraction_sofar": missing_fraction_sofar,
        "max_gap_minutes": max_gap_minutes,
        "edge_max_flag": edge_max_flag,
    }


def _city_one_hot(city: str) -> dict[str, int]:
    """Create one-hot encoding for city.

    Args:
        city: City identifier

    Returns:
        Dictionary with one-hot city features
    """
    return {
        "city_chicago": int(city == "chicago"),
        "city_austin": int(city == "austin"),
        "city_denver": int(city == "denver"),
        "city_los_angeles": int(city == "los_angeles"),
        "city_miami": int(city == "miami"),
        "city_philadelphia": int(city == "philadelphia"),
    }


def _generate_snapshot_times(
    market_open: datetime,
    market_close: datetime,
    interval_min: int,
) -> list[datetime]:
    """Generate snapshot timestamps from market open to close.

    Args:
        market_open: Market open time (D-1 10:00)
        market_close: Market close time (D 23:55)
        interval_min: Minutes between snapshots

    Returns:
        List of datetime objects at interval_min spacing
    """
    snapshots = []
    current = market_open

    while current <= market_close:
        snapshots.append(current)
        current += timedelta(minutes=interval_min)

    return snapshots


def _load_settlement_data(
    session: Session,
    cities: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Load settlement data for multiple cities.

    Args:
        session: Database session
        cities: List of city identifiers
        start_date: First date
        end_date: Last date

    Returns:
        DataFrame with columns: city, date_local, tmax_final
    """
    from sqlalchemy import select
    from src.db.models import WxSettlement

    query = select(
        WxSettlement.city,
        WxSettlement.date_local,
        WxSettlement.tmax_final,
    ).where(
        WxSettlement.city.in_(cities),
        WxSettlement.date_local >= start_date,
        WxSettlement.date_local <= end_date,
    )

    result = session.execute(query).fetchall()
    df = pd.DataFrame(result, columns=["city", "date_local", "tmax_final"])

    return df


def _fill_forecast_error_nulls(row: dict) -> None:
    """Fill forecast error features with None."""
    error_cols = [
        "err_mean_sofar", "err_std_sofar", "err_max_pos_sofar",
        "err_max_neg_sofar", "err_abs_mean_sofar", "err_last1h",
        "err_last3h_mean", "delta_vcmax_fcstmax_sofar", "fcst_remaining_potential",
    ]
    for col in error_cols:
        row[col] = None


def _fill_all_forecast_nulls(row: dict) -> None:
    """Fill all forecast features with None."""
    forecast_cols = [
        "fcst_prev_max_f", "fcst_prev_min_f", "fcst_prev_mean_f", "fcst_prev_std_f",
        "fcst_prev_q10_f", "fcst_prev_q25_f", "fcst_prev_q50_f",
        "fcst_prev_q75_f", "fcst_prev_q90_f",
        "fcst_prev_frac_part", "fcst_prev_hour_of_max", "t_forecast_base",
        "err_mean_sofar", "err_std_sofar", "err_max_pos_sofar", "err_max_neg_sofar",
        "err_abs_mean_sofar", "err_last1h", "err_last3h_mean",
        "delta_vcmax_fcstmax_sofar", "fcst_remaining_potential",
    ]
    for col in forecast_cols:
        row[col] = None


def _add_derived_features(row: dict) -> None:
    """Add derived features that correlate with delta.

    These are computed from existing features and help the model.
    """
    # obs_fcst_max_gap: upside potential = fcst_max - vc_max_sofar
    fcst_max = row.get("fcst_prev_max_f")
    vc_max = row.get("vc_max_f_sofar")
    if fcst_max is not None and vc_max is not None:
        row["obs_fcst_max_gap"] = fcst_max - vc_max
    else:
        row["obs_fcst_max_gap"] = None

    # hours_until_fcst_max: fcst_hour_of_max - current_hour
    fcst_hour_of_max = row.get("fcst_prev_hour_of_max")
    hour = row.get("hour")
    if fcst_hour_of_max is not None and hour is not None:
        row["hours_until_fcst_max"] = fcst_hour_of_max - hour
    else:
        row["hours_until_fcst_max"] = None

    # above_fcst_flag: 1 if vc_max > fcst_max
    if fcst_max is not None and vc_max is not None:
        row["above_fcst_flag"] = int(vc_max > fcst_max)
    else:
        row["above_fcst_flag"] = None

    # day_fraction: (snapshot_hour - 6) / 18
    if hour is not None:
        row["day_fraction"] = max(0.0, (hour - 6) / 18.0)
    else:
        row["day_fraction"] = None


def _fill_market_feature_nulls(row: dict) -> None:
    """Fill market-derived features with None (placeholder).

    These require candle data which is loaded at the dataset level,
    not per-snapshot. Use build_v2_dataset() to populate these.
    """
    market_cols = [
        "market_yes_bid", "market_yes_ask",
        "market_bid_ask_spread", "market_mid_price",
        "bid_change_last_30min", "bid_change_last_60min",
        "bid_momentum_30min",
        "volume_last_30min", "volume_last_60min",
        "cumulative_volume_today",
        "has_recent_trade", "open_interest",
    ]
    for col in market_cols:
        if col not in row:
            row[col] = None


def _fill_station_city_feature_nulls(row: dict) -> None:
    """Fill station-city gap features with None (placeholder).

    These require city-level observations which are loaded at the
    dataset level, not per-snapshot. Use build_v2_dataset() to populate these.
    """
    station_city_cols = [
        "station_city_temp_gap",
        "station_city_max_gap_sofar",
        "station_city_mean_gap_sofar",
        "city_warmer_flag",
        "station_city_gap_std",
        "station_city_gap_trend",
    ]
    for col in station_city_cols:
        if col not in row:
            row[col] = None


def _fill_meteo_feature_nulls(row: dict) -> None:
    """Fill meteorological features with None (placeholder).

    These require meteo columns from obs_df which may not always be available.
    """
    meteo_cols = [
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
    for col in meteo_cols:
        if col not in row:
            row[col] = None


def _fill_regime_feature_nulls(row: dict) -> None:
    """Fill regime/phase features with None (placeholder)."""
    regime_cols = [
        "is_heating_phase",
        "is_plateau_phase",
        "is_cooling_phase",
    ]
    for col in regime_cols:
        if col not in row:
            row[col] = None


# =============================================================================
# V2 Dataset Builder - Production Ready with Candles + City Obs
# =============================================================================


def load_city_observations(
    session: Session,
    city_id: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Load Visual Crossing city-aggregate observations (not station-locked).

    City observations use multi-station interpolation and may correlate
    better with NWS settlement than single-station data.

    Args:
        session: Database session
        city_id: City identifier (e.g., 'chicago')
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        DataFrame with columns: datetime_local, temp_f
    """
    from sqlalchemy import text as sql_text
    from models.data.loader import get_vc_location_id

    vc_location_id = get_vc_location_id(session, city_id, "city")

    if vc_location_id is None:
        logger.warning(f"No city-level vc_location found for {city_id}")
        return pd.DataFrame()

    query = sql_text("""
        SELECT datetime_local, temp_f
        FROM wx.vc_minute_weather
        WHERE vc_location_id = :vc_location_id
          AND data_type = 'actual_obs'
          AND datetime_local >= :start_date
          AND datetime_local < :end_date_plus_one
        ORDER BY datetime_local
    """)

    result = session.execute(query, {
        "vc_location_id": vc_location_id,
        "start_date": start_date,
        "end_date_plus_one": end_date + timedelta(days=1),
    })

    df = pd.DataFrame(result.fetchall(), columns=["datetime_local", "temp_f"])
    logger.info(f"Loaded {len(df)} city observations for {city_id}")
    return df


def load_dense_candles_for_ticker(
    session: Session,
    ticker: str,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    """Load dense 1-minute candles for a single ticker.

    Args:
        session: Database session
        ticker: Kalshi market ticker
        start_time: Start of window (timezone-aware)
        end_time: End of window (timezone-aware)

    Returns:
        DataFrame with candle data sorted by bucket_start
    """
    from sqlalchemy import text as sql_text

    query = sql_text("""
        SELECT
            bucket_start,
            yes_bid_close,
            yes_ask_close,
            trade_close,
            volume,
            open_interest,
            has_trade
        FROM kalshi.candles_1m_dense
        WHERE ticker = :ticker
          AND bucket_start >= :start_time
          AND bucket_start <= :end_time
        ORDER BY bucket_start
    """)

    result = session.execute(query, {
        "ticker": ticker,
        "start_time": start_time,
        "end_time": end_time,
    })

    df = pd.DataFrame(result.fetchall(), columns=[
        "bucket_start", "yes_bid_close", "yes_ask_close",
        "trade_close", "volume", "open_interest", "has_trade"
    ])

    return df


def get_atm_ticker_for_event(
    session: Session,
    city: str,
    event_date: date,
    fcst_max_f: Optional[float],
) -> Optional[str]:
    """Get the at-the-money (ATM) bracket ticker for an event.

    ATM is determined by the forecast max temperature. If no forecast,
    we skip candle features for this event.

    Args:
        session: Database session
        city: City identifier
        event_date: Event date
        fcst_max_f: Forecast max temperature (from T-1 forecast)

    Returns:
        Ticker string or None if not found
    """
    if fcst_max_f is None:
        return None

    from sqlalchemy import text as sql_text

    # Find bracket containing forecast max
    # Brackets are (floor_strike, cap_strike] - temp > floor AND temp <= cap
    query = sql_text("""
        SELECT ticker
        FROM kalshi.markets
        WHERE city = :city
          AND event_date = :event_date
          AND strike_type = 'between'
          AND floor_strike < :fcst_max
          AND cap_strike >= :fcst_max
        LIMIT 1
    """)

    result = session.execute(query, {
        "city": city,
        "event_date": event_date,
        "fcst_max": fcst_max_f,
    }).fetchone()

    if result:
        return result[0]
    return None


# =============================================================================
# Null Handling Strategy
# =============================================================================

"""
NULL HANDLING STRATEGY FOR V2 FEATURES
======================================

Different null strategies for different feature types:

1. STRUCTURAL NULLS (keep as None → model handles as missing)
   - Features that genuinely don't exist (e.g., no candle data before market open)
   - Features from missing data sources (e.g., city obs not available)
   - The model (CatBoost/XGBoost) handles these natively

2. ZERO IMPUTATION (replace None with 0.0)
   - Volume features: no volume = 0 volume
   - Rate/change features at start of day: no history = no change
   - Binary flags with no data: treated as "not present"

3. FORWARD FILL (use last known value)
   - Market prices when no trade: use last bid/ask
   - This is already done in dense candles table

4. MEDIAN/MEAN IMPUTATION (use historical median)
   - For features where "typical" value is better than None/0
   - Used sparingly - only when we have strong priors

The key insight: For tree-based models (CatBoost, XGBoost, LightGBM),
None/NaN is often the BEST choice - the model learns to route missing
values appropriately. Don't over-impute!

FEATURE-SPECIFIC RULES:
-----------------------

Market Features:
- market_yes_bid/ask/spread/mid: None if no candle data (structural)
- bid_change_*: None if insufficient history (structural)
- volume_*: 0.0 if no trades (zero imputation - absence = zero)
- has_recent_trade: 0 if no trades (zero imputation)

Station-City Features:
- All None if city obs not available (structural)
- Compute from aligned data when available

Momentum Features:
- Rolling stats: None if insufficient data points (structural)
- Rate features: None if < 2 data points (structural)
- EMA: Computed from available data, None only if empty

Interaction Features:
- Computed from component features
- None if any component is None (propagate nulls)

"""


def apply_v2_imputation(row: dict[str, Any]) -> None:
    """Apply production imputation rules to v2 features.

    Modifies row in-place.

    Strategy:
    - Volume features: None → 0.0 (no volume = zero volume)
    - Binary flags: None → 0 (no signal = not present)
    - Price/rate features: Keep None (let model handle as missing)
    """
    # Volume features: None → 0.0
    volume_cols = [
        "volume_last_30min",
        "volume_last_60min",
        "cumulative_volume_today",
    ]
    for col in volume_cols:
        if col in row and row[col] is None:
            row[col] = 0.0

    # Binary flags: None → 0
    flag_cols = [
        "has_recent_trade",
        "city_warmer_flag",
        "above_fcst_flag",
    ]
    for col in flag_cols:
        if col in row and row[col] is None:
            row[col] = 0


def build_v2_dataset(
    cities: list[str],
    start_date: date,
    end_date: date,
    session: Session,
    snapshot_interval_min: int = 5,
    market_open_hour: int = 10,
    include_forecast_features: bool = True,
    include_market_features: bool = True,
    include_station_city_features: bool = True,
    include_meteo_features: bool = True,
) -> pd.DataFrame:
    """Build production-ready v2 dataset with all feature categories.

    This is the main entry point for training Market-Clock TOD v2 models.
    It extends build_market_clock_snapshot_dataset with:
    - Market features from dense candles
    - Station-city temperature gap features
    - Meteorological features (humidity, wind, cloud cover)
    - Regime/phase flags
    - Proper null handling and imputation

    Args:
        cities: List of city identifiers
        start_date: First event date
        end_date: Last event date
        session: Database session
        snapshot_interval_min: Minutes between snapshots (default 5)
        market_open_hour: Hour when market opens on D-1 (default 10)
        include_forecast_features: Include T-1 forecast features
        include_market_features: Include Kalshi candle features
        include_station_city_features: Include station vs city gap features
        include_meteo_features: Include meteorological features (humidity, wind, clouds)

    Returns:
        DataFrame with one row per (city, event_date, snapshot_datetime)
    """
    logger.info(
        f"Building v2 dataset: {len(cities)} cities, "
        f"{(end_date - start_date).days + 1} event days, {snapshot_interval_min}-min intervals"
    )

    # Load settlement data for all cities/dates
    settlement_df = _load_settlement_data(session, cities, start_date, end_date)
    logger.info(f"Loaded {len(settlement_df)} settlement records")

    all_snapshots = []
    total_snapshots = 0

    for city in cities:
        logger.info(f"Building v2 snapshots for {city}...")

        city_settlement = settlement_df[settlement_df["city"] == city]
        if city_settlement.empty:
            logger.warning(f"No settlement data for {city}, skipping")
            continue

        # Load station observations (primary)
        station_obs_df = load_vc_observations(
            session,
            city_id=city,
            start_date=start_date - timedelta(days=1),
            end_date=end_date + timedelta(days=1),
        )

        if station_obs_df.empty:
            logger.warning(f"No station observation data for {city}, skipping")
            continue

        # Load city observations (for station-city features)
        city_obs_df = pd.DataFrame()
        if include_station_city_features:
            city_obs_df = load_city_observations(
                session, city, start_date - timedelta(days=1), end_date + timedelta(days=1)
            )
            if city_obs_df.empty:
                logger.warning(f"No city observations for {city}, station-city features will be None")

        city_snapshots = 0

        # Process each event date
        for single_day in pd.date_range(start_date, end_date, freq="D"):
            event_date = single_day.date()

            day_settlement = city_settlement[city_settlement["date_local"] == event_date]
            if day_settlement.empty:
                continue

            settle_f = int(day_settlement.iloc[0]["tmax_final"])

            # Define market window
            market_open = datetime.combine(
                event_date - timedelta(days=1),
                datetime.min.time()
            ).replace(hour=market_open_hour, minute=0)

            market_close = datetime.combine(
                event_date,
                datetime.min.time()
            ).replace(hour=23, minute=55)

            # Filter station observations for market window
            window_station_obs = station_obs_df[
                (station_obs_df["datetime_local"] >= market_open) &
                (station_obs_df["datetime_local"] <= market_close)
            ].copy()

            if window_station_obs.empty:
                continue

            # Filter city observations for market window (if available)
            window_city_obs = pd.DataFrame()
            if not city_obs_df.empty:
                window_city_obs = city_obs_df[
                    (city_obs_df["datetime_local"] >= market_open) &
                    (city_obs_df["datetime_local"] <= market_close)
                ].copy()

            # Load T-1 forecast
            basis_date = event_date - timedelta(days=1)
            fcst_daily = None
            fcst_hourly_df = None

            if include_forecast_features:
                fcst_daily = load_historical_forecast_daily(session, city, event_date, basis_date)
                fcst_hourly_df = load_historical_forecast_hourly(session, city, event_date, basis_date)

            # Get ATM ticker for candle features
            atm_ticker = None
            candles_df = pd.DataFrame()

            if include_market_features and fcst_daily is not None:
                fcst_max_f = fcst_daily.get("tempmax_f")
                atm_ticker = get_atm_ticker_for_event(session, city, event_date, fcst_max_f)

                if atm_ticker:
                    # Load dense candles for full market window
                    # Note: bucket_start is timezone-aware (UTC), need to handle properly
                    candles_df = load_dense_candles_for_ticker(
                        session, atm_ticker, market_open, market_close
                    )

            # Generate snapshot timestamps
            snapshot_times = _generate_snapshot_times(
                market_open, market_close, snapshot_interval_min
            )

            # Build snapshot for each timestamp
            for snapshot_ts in snapshot_times:
                # Filter station observations up to this snapshot time
                station_obs_sofar = window_station_obs[
                    window_station_obs["datetime_local"] <= snapshot_ts
                ].copy()

                if station_obs_sofar.empty:
                    continue

                temps_sofar = station_obs_sofar["temp_f"].dropna().tolist()
                timestamps_sofar = station_obs_sofar.loc[
                    station_obs_sofar["temp_f"].notna(), "datetime_local"
                ].tolist()

                if len(temps_sofar) < MIN_SAMPLES:
                    continue

                try:
                    # Build base snapshot (v1 features + momentum + interactions)
                    snapshot_row = build_market_clock_snapshot_for_training(
                        city=city,
                        event_date=event_date,
                        cutoff_time=snapshot_ts,
                        temps_sofar=temps_sofar,
                        timestamps_sofar=timestamps_sofar,
                        fcst_daily=fcst_daily,
                        fcst_hourly_df=fcst_hourly_df,
                        settle_f=settle_f,
                        market_open=market_open,
                    )

                    # Add market features from candles
                    if include_market_features and not candles_df.empty:
                        market_fs = compute_market_features(candles_df, snapshot_ts)
                        snapshot_row.update(market_fs.to_dict())

                    # Add station-city features
                    if include_station_city_features and not window_city_obs.empty:
                        # Get city obs up to this snapshot
                        city_obs_sofar = window_city_obs[
                            window_city_obs["datetime_local"] <= snapshot_ts
                        ].copy()

                        if not city_obs_sofar.empty:
                            # Build aligned temp lists
                            station_temps = list(zip(timestamps_sofar, temps_sofar))
                            city_temps = list(zip(
                                city_obs_sofar["datetime_local"].tolist(),
                                city_obs_sofar["temp_f"].dropna().tolist()
                            ))

                            station_city_fs = compute_station_city_features(
                                station_temps, city_temps
                            )
                            snapshot_row.update(station_city_fs.to_dict())

                    # Add meteo features from station obs
                    if include_meteo_features:
                        meteo_fs = compute_meteo_features(station_obs_sofar, snapshot_ts)
                        snapshot_row.update(meteo_fs.to_dict())

                    # Apply imputation rules
                    apply_v2_imputation(snapshot_row)

                    all_snapshots.append(snapshot_row)
                    city_snapshots += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to build v2 snapshot for {city} {event_date} {snapshot_ts}: {e}"
                    )
                    continue

        logger.info(f"  Built {city_snapshots} v2 snapshots for {city}")
        total_snapshots += city_snapshots

    logger.info(f"Total v2 snapshots built: {total_snapshots}")

    if not all_snapshots:
        logger.warning("No v2 snapshots built!")
        return pd.DataFrame()

    df = pd.DataFrame(all_snapshots)

    # Add lag features
    if not df.empty:
        from models.features.calendar import add_lag_features_to_dataframe
        df["day"] = df["event_date"]
        df = add_lag_features_to_dataframe(df)

    logger.info(f"Final v2 dataset: {len(df)} rows, {len(df.columns)} columns")

    return df


def build_v2_inference_snapshot(
    city: str,
    event_date: date,
    cutoff_time: datetime,
    session: Session,
    market_open: Optional[datetime] = None,
    ticker: Optional[str] = None,
) -> dict[str, Any]:
    """Build a single v2 snapshot for live inference.

    This is the inference-time equivalent of build_v2_dataset().
    Loads current observations, candles, and computes all v2 features.

    Args:
        city: City identifier
        event_date: Settlement date (D)
        cutoff_time: Current time (snapshot time)
        session: Database session
        market_open: Market open time (if None, defaults to D-1 10:00)
        ticker: ATM bracket ticker (if None, will try to determine from forecast)

    Returns:
        Dictionary with all v2 features for inference
    """
    from models.data.loader import load_inference_data

    if market_open is None:
        market_open = datetime.combine(
            event_date - timedelta(days=1),
            datetime.min.time()
        ).replace(hour=10, minute=0)

    # Load base inference data
    inference_data = load_inference_data(city, event_date, cutoff_time, session)

    temps_sofar = inference_data["temps_sofar"]
    timestamps_sofar = inference_data["timestamps_sofar"]
    fcst_daily = inference_data["fcst_daily"]
    fcst_hourly_df = inference_data["fcst_hourly_df"]

    if len(temps_sofar) < MIN_SAMPLES:
        raise ValueError(f"Insufficient observations: {len(temps_sofar)} < {MIN_SAMPLES}")

    # Build base inference snapshot
    row = build_market_clock_snapshot_for_inference(
        city=city,
        event_date=event_date,
        cutoff_time=cutoff_time,
        temps_sofar=temps_sofar,
        timestamps_sofar=timestamps_sofar,
        fcst_daily=fcst_daily,
        fcst_hourly_df=fcst_hourly_df,
        market_open=market_open,
    )

    # Add market features
    if ticker:
        candles_df = load_dense_candles_for_ticker(
            session, ticker, market_open, cutoff_time
        )
        if not candles_df.empty:
            market_fs = compute_market_features(candles_df, cutoff_time)
            row.update(market_fs.to_dict())

    # Add station-city features
    city_obs_df = load_city_observations(
        session, city,
        event_date - timedelta(days=1),
        event_date + timedelta(days=1)
    )

    if not city_obs_df.empty:
        city_obs_sofar = city_obs_df[city_obs_df["datetime_local"] <= cutoff_time].copy()

        if not city_obs_sofar.empty:
            station_temps = list(zip(timestamps_sofar, temps_sofar))
            city_temps = list(zip(
                city_obs_sofar["datetime_local"].tolist(),
                city_obs_sofar["temp_f"].dropna().tolist()
            ))

            station_city_fs = compute_station_city_features(station_temps, city_temps)
            row.update(station_city_fs.to_dict())

    # Apply imputation
    apply_v2_imputation(row)

    return row
