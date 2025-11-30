"""
Calendar and temporal features for temperature Δ-models.

This module encodes time-of-day, day-of-year, and computes lag features
from historical settlement data. These features capture:
- Intraday timing (snapshot_hour) - early vs late day predictions
- Seasonality (doy, week) - summer vs winter temperature patterns
- Autocorrelation (lags) - recent settlement history

Cyclical features (hour, day-of-year, week) use sin/cos encoding to
preserve their circular nature (e.g., hour 23 is close to hour 0).

Features computed:
    snapshot_hour: Hour of the snapshot (integer 0-23)
    snapshot_hour_sin, snapshot_hour_cos: Cyclical encoding
    doy_sin, doy_cos: Day-of-year cyclical encoding
    week_sin, week_cos: Week-of-year cyclical encoding
    month: Month of year (1-12)
    is_weekend: 1 if Saturday/Sunday, 0 otherwise
    settle_f_lag1, lag2, lag7: Prior settlement temperatures
    vc_max_f_lag1, lag7: Prior VC max temperatures
    delta_vcmax_lag1: Change in VC max from yesterday
"""

import math
from datetime import date, datetime
from typing import Optional

import pandas as pd

from models.features.base import FeatureSet, register_feature_group


@register_feature_group("calendar")
def compute_calendar_features(
    day: date,
    cutoff_time: Optional[datetime] = None,
    snapshot_hour: Optional[int] = None,
) -> FeatureSet:
    """Compute calendar and time encoding features.

    These features capture temporal patterns in temperature settlements.
    Cyclical encoding (sin/cos) preserves the circular nature of time
    features, so hour 23 is recognized as close to hour 0.

    Supports both legacy snapshot_hour (integer) and new cutoff_time (datetime)
    for time-of-day aware models.

    Args:
        day: The target date for settlement
        cutoff_time: Full datetime for snapshot (tod_v1 models)
        snapshot_hour: Legacy integer hour (baseline models) - DEPRECATED

    Returns:
        FeatureSet with calendar and time features

    Example:
        >>> from datetime import date, datetime
        >>> # Legacy usage (backward compat)
        >>> fs = compute_calendar_features(date(2024, 7, 15), snapshot_hour=14)
        >>> # New usage (tod_v1)
        >>> fs = compute_calendar_features(date(2024, 7, 15), cutoff_time=datetime(2024,7,15,14,30))
    """
    # Backward compatibility: if only snapshot_hour provided, construct cutoff_time
    if cutoff_time is None and snapshot_hour is not None:
        cutoff_time = datetime.combine(day, datetime.min.time()).replace(hour=snapshot_hour, minute=0)
    elif cutoff_time is None and snapshot_hour is None:
        raise ValueError("Must provide either cutoff_time or snapshot_hour")

    # Extract time components
    hour = cutoff_time.hour
    minute = cutoff_time.minute
    minutes_since_midnight = hour * 60 + minute
    # Day of year (1-366)
    doy = day.timetuple().tm_yday

    # Week of year (1-53)
    week = day.isocalendar()[1]

    # Month (1-12)
    month = day.month

    # Weekend flag
    is_weekend = 1 if day.weekday() >= 5 else 0

    # Cyclical encodings using sin/cos
    # This ensures that e.g., hour 23 is close to hour 0
    snapshot_hour_sin = math.sin(2 * math.pi * hour / 24.0)
    snapshot_hour_cos = math.cos(2 * math.pi * hour / 24.0)

    # New time-of-day features (tod_v1)
    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)
    minute_sin = math.sin(2 * math.pi * minute / 60.0)
    minute_cos = math.cos(2 * math.pi * minute / 60.0)
    time_of_day_sin = math.sin(2 * math.pi * minutes_since_midnight / (24.0 * 60.0))
    time_of_day_cos = math.cos(2 * math.pi * minutes_since_midnight / (24.0 * 60.0))

    # Day-level cyclical encodings
    doy_sin = math.sin(2 * math.pi * doy / 365.25)
    doy_cos = math.cos(2 * math.pi * doy / 365.25)

    week_sin = math.sin(2 * math.pi * week / 52.0)
    week_cos = math.cos(2 * math.pi * week / 52.0)

    features = {
        # Legacy features (backward compatibility)
        "snapshot_hour": hour,  # Keep for baseline models
        "snapshot_hour_sin": snapshot_hour_sin,
        "snapshot_hour_cos": snapshot_hour_cos,

        # New time-of-day features (tod_v1 models)
        "hour": hour,
        "minute": minute,
        "minutes_since_midnight": minutes_since_midnight,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "minute_sin": minute_sin,
        "minute_cos": minute_cos,
        "time_of_day_sin": time_of_day_sin,
        "time_of_day_cos": time_of_day_cos,

        # Day-level features (unchanged)
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "week_sin": week_sin,
        "week_cos": week_cos,
        "month": month,
        "is_weekend": is_weekend,
    }

    return FeatureSet(name="calendar", features=features)


def compute_lag_features(
    df: pd.DataFrame,
    city: str,
    day: date,
) -> FeatureSet:
    """Compute lag features from historical data.

    Lag features capture autocorrelation in temperature settlements.
    Yesterday's high is predictive of today's high, especially in
    stable weather patterns.

    Args:
        df: DataFrame with columns ['city', 'day', 'settle_f', 'vc_max_f']
            containing historical settlement data. Must be sorted by day.
        city: City identifier to filter data
        day: Target date - lags are computed relative to this date

    Returns:
        FeatureSet with lag features. Values are None if insufficient
        historical data exists.

    Note:
        This function requires the historical DataFrame to be available.
        During snapshot dataset construction, lags are computed after
        grouping by (city, day).
    """
    # Filter to this city and days before target
    city_df = df[(df["city"] == city) & (df["day"] < day)].copy()

    if city_df.empty:
        return FeatureSet(name="lags", features={
            "settle_f_lag1": None,
            "settle_f_lag2": None,
            "settle_f_lag7": None,
            "vc_max_f_lag1": None,
            "vc_max_f_lag7": None,
            "delta_vcmax_lag1": None,
        })

    # Sort by day descending to get most recent first
    city_df = city_df.sort_values("day", ascending=False)

    # Get lag values
    settle_f_lag1 = _get_lag_value(city_df, "settle_f", day, 1)
    settle_f_lag2 = _get_lag_value(city_df, "settle_f", day, 2)
    settle_f_lag7 = _get_lag_value(city_df, "settle_f", day, 7)

    vc_max_f_lag1 = _get_lag_value(city_df, "vc_max_f", day, 1)
    vc_max_f_lag7 = _get_lag_value(city_df, "vc_max_f", day, 7)

    # Delta from yesterday (if we have today's max and yesterday's max)
    # This is computed later when we have today's partial-day max
    delta_vcmax_lag1 = None  # Placeholder - computed in snapshot builder

    features = {
        "settle_f_lag1": settle_f_lag1,
        "settle_f_lag2": settle_f_lag2,
        "settle_f_lag7": settle_f_lag7,
        "vc_max_f_lag1": vc_max_f_lag1,
        "vc_max_f_lag7": vc_max_f_lag7,
        "delta_vcmax_lag1": delta_vcmax_lag1,
    }

    return FeatureSet(name="lags", features=features)


def _get_lag_value(
    df: pd.DataFrame,
    col: str,
    target_day: date,
    lag_days: int,
) -> Optional[float]:
    """Get a lagged value from the DataFrame.

    Args:
        df: DataFrame sorted by day (descending)
        col: Column name to get value from
        target_day: Reference date
        lag_days: Number of days to look back

    Returns:
        Value from lag_days ago, or None if not available
    """
    from datetime import timedelta

    lag_date = target_day - timedelta(days=lag_days)

    row = df[df["day"] == lag_date]
    if row.empty:
        return None

    value = row[col].iloc[0]
    if pd.isna(value):
        return None

    return float(value)


def add_lag_features_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features to a snapshot DataFrame.

    This is a convenience function for adding lags during dataset
    construction. It groups by (city, day) to get daily values,
    computes lags, and merges back.

    Args:
        df: Snapshot DataFrame with columns ['city', 'day', 'settle_f',
            'vc_max_f_sofar', ...]. May have multiple rows per (city, day)
            for different snapshot hours.

    Returns:
        DataFrame with lag columns added

    Note:
        The delta_vcmax_lag1 feature compares today's partial-day max
        (vc_max_f_sofar) to yesterday's final VC max (vc_max_f_lag1).
    """
    df = df.copy()

    # Get daily values (one row per city-day)
    # Use first row since settle_f and day are same for all snapshots of a day
    daily = df.groupby(["city", "day"]).agg({
        "settle_f": "first",
        "vc_max_f_sofar": "max",  # Use max across all snapshots as "final"
    }).reset_index()
    daily = daily.rename(columns={"vc_max_f_sofar": "vc_max_f"})
    daily = daily.sort_values(["city", "day"])

    # Compute lags within each city
    for lag in [1, 2, 7]:
        daily[f"settle_f_lag{lag}"] = daily.groupby("city")["settle_f"].shift(lag)
        if lag in [1, 7]:
            daily[f"vc_max_f_lag{lag}"] = daily.groupby("city")["vc_max_f"].shift(lag)

    # Select lag columns to merge
    lag_cols = [
        "city", "day",
        "settle_f_lag1", "settle_f_lag2", "settle_f_lag7",
        "vc_max_f_lag1", "vc_max_f_lag7",
    ]
    daily_lags = daily[lag_cols]

    # Merge back to snapshot-level data
    df = df.merge(daily_lags, on=["city", "day"], how="left")

    # Compute delta_vcmax_lag1 = today's max so far - yesterday's final max
    df["delta_vcmax_lag1"] = df["vc_max_f_sofar"] - df["vc_max_f_lag1"]

    return df
