"""
Time-of-Day (TOD) Snapshot Dataset Builder

Generates training datasets with arbitrary-timestamp snapshots (15-min or 5-min intervals)
for time-of-day aware ordinal CatBoost models.

This replaces the fixed hourly snapshot approach with continuous time features,
enabling predictions at any minute of the day.

Usage:
    from models.data.tod_dataset_builder import build_tod_snapshot_dataset

    df = build_tod_snapshot_dataset(
        cities=['chicago'],
        start_date=date(2023, 12, 28),
        end_date=date(2025, 11, 27),
        session=db_session,
        snapshot_interval_min=15,  # 15-minute snapshots
    )
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from models.data.loader import (
    load_vc_observations,
    load_historical_forecast_daily,
    load_historical_forecast_hourly,
)
from models.data.snapshot_builder import build_snapshot_for_inference
from models.features.partial_day import compute_partial_day_features
from models.features.shape import compute_shape_features
from models.features.rules import compute_rule_features
from models.features.calendar import compute_calendar_features
from models.features.quality import compute_quality_features, estimate_expected_samples

logger = logging.getLogger(__name__)


def build_tod_snapshot_dataset(
    cities: list[str],
    start_date: date,
    end_date: date,
    session: Session,
    snapshot_interval_min: int = 15,
    include_forecast_features: bool = True,
    day_start_hour: int = 10,
    day_end_hour: int = 23,
    day_end_minute: int = 45,
) -> pd.DataFrame:
    """Build training dataset with time-of-day aware snapshots.

    For each (city, day), creates snapshots at snapshot_interval_min intervals
    from day_start_hour to day_end_hour.

    Example with 15-min intervals:
        10:00, 10:15, 10:30, 10:45, 11:00, ..., 23:30, 23:45
        = 56 snapshots per day

    Args:
        cities: List of city identifiers
        start_date: First event date for training
        end_date: Last event date for training
        session: Database session
        snapshot_interval_min: Minutes between snapshots (15, 5, etc.)
        include_forecast_features: Whether to include T-1 forecast features
        day_start_hour: First snapshot hour (default 10am)
        day_end_hour: Last snapshot hour (default 11pm)
        day_end_minute: Last snapshot minute (default 45)

    Returns:
        DataFrame with one row per (city, day, snapshot_timestamp)
        Columns: city, day, snapshot_timestamp, snapshot_hour, minute,
                 t_base, delta, settle_f, [all feature columns]
    """
    logger.info(f"Building TOD snapshot dataset: {len(cities)} cities, "
                f"{(end_date - start_date).days} days, {snapshot_interval_min}-min intervals")

    # Load settlement data for all cities/dates
    settlement_df = load_settlement_data(session, cities, start_date, end_date)
    logger.info(f"Loaded {len(settlement_df)} settlement records")

    snapshots = []
    total_snapshots = 0

    for city in cities:
        logger.info(f"Building snapshots for {city}...")

        # Load all observations for this city (once)
        obs_df = load_vc_observations(
            session,
            city_id=city,
            start_date=start_date,
            end_date=end_date + timedelta(days=1),  # Include full last day
        )

        city_settlement = settlement_df[settlement_df["city"] == city]
        city_snapshots = 0

        # Process each day
        for single_day in pd.date_range(start_date, end_date, freq='D'):
            single_day = single_day.date()

            # Get settlement for this day
            day_settlement = city_settlement[city_settlement["date_local"] == single_day]
            if day_settlement.empty:
                continue  # No settlement data

            settle_f = float(day_settlement.iloc[0]["tmax_final"])

            # Get observations for this day
            day_obs = obs_df[obs_df["datetime_local"].dt.date == single_day].copy()
            if day_obs.empty:
                continue  # No observations

            # Load T-1 forecast (once per day)
            basis_date = single_day - timedelta(days=1)
            fcst_daily = None
            fcst_hourly_df = None

            if include_forecast_features:
                fcst_daily = load_historical_forecast_daily(session, city, single_day, basis_date)
                fcst_hourly_df = load_historical_forecast_hourly(session, city, single_day, basis_date)

            # Generate snapshot timestamps for this day
            snapshot_times = _generate_snapshot_times(
                single_day,
                snapshot_interval_min,
                day_start_hour,
                day_end_hour,
                day_end_minute,
            )

            # Build snapshot for each timestamp
            for snapshot_ts in snapshot_times:
                # Filter observations up to this snapshot time
                obs_sofar = day_obs[day_obs["datetime_local"] <= snapshot_ts].copy()

                if obs_sofar.empty:
                    continue  # No obs yet

                temps_sofar = obs_sofar["temp_f"].dropna().tolist()
                timestamps_sofar = obs_sofar.loc[obs_sofar["temp_f"].notna(), "datetime_local"].tolist()

                if not temps_sofar:
                    continue

                # Build snapshot features
                try:
                    snapshot_row = build_snapshot_for_inference(
                        city=city,
                        day=single_day,
                        temps_sofar=temps_sofar,
                        timestamps_sofar=timestamps_sofar,
                        cutoff_time=snapshot_ts,
                        fcst_daily=fcst_daily,  # FIXED: Load T-1 forecast
                        fcst_hourly_df=fcst_hourly_df,  # FIXED: Load T-1 hourly
                    )

                    # Add labels
                    t_base = snapshot_row["t_base"]
                    delta = round(settle_f) - t_base

                    snapshot_row.update({
                        "settle_f": settle_f,
                        "delta": delta,
                        "snapshot_timestamp": snapshot_ts,  # Full datetime
                    })

                    snapshots.append(snapshot_row)
                    city_snapshots += 1

                except Exception as e:
                    logger.warning(f"Failed to build snapshot for {city} {single_day} {snapshot_ts}: {e}")
                    continue

        logger.info(f"  Built {city_snapshots} snapshots for {city}")
        total_snapshots += city_snapshots

    logger.info(f"Total snapshots built: {total_snapshots}")

    df = pd.DataFrame(snapshots)

    # Add lag features (operates on full DataFrame)
    if not df.empty:
        from models.features.calendar import add_lag_features_to_dataframe
        df = add_lag_features_to_dataframe(df)

    logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")

    return df


def _generate_snapshot_times(
    day: date,
    interval_min: int,
    start_hour: int,
    end_hour: int,
    end_minute: int,
) -> list[datetime]:
    """Generate snapshot timestamps for a single day.

    Args:
        day: The event date
        interval_min: Minutes between snapshots (15, 5, etc.)
        start_hour: First snapshot hour (e.g., 10)
        end_hour: Last snapshot hour (e.g., 23)
        end_minute: Last snapshot minute (e.g., 45)

    Returns:
        List of datetime objects at interval_min spacing

    Example with 15-min intervals:
        10:00, 10:15, 10:30, 10:45, 11:00, ..., 23:30, 23:45
    """
    snapshots = []

    # Start at day_start_hour:00
    current = datetime.combine(day, datetime.min.time()).replace(hour=start_hour, minute=0)

    # End at day_end_hour:end_minute
    end = datetime.combine(day, datetime.min.time()).replace(hour=end_hour, minute=end_minute)

    while current <= end:
        snapshots.append(current)
        current += timedelta(minutes=interval_min)

    return snapshots


def load_settlement_data(
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
