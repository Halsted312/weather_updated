"""
Unified snapshot building for training and inference.

This module provides functions to create SnapshotContext objects from
raw data (database rows or live data) and compute features.

Example:
    >>> from models.data.snapshot import build_snapshot
    >>> ctx = build_snapshot(
    ...     city="chicago",
    ...     event_date=date(2024, 7, 15),
    ...     cutoff_time=datetime(2024, 7, 15, 14, 30),
    ...     obs_df=obs_df,
    ...     fcst_daily=fcst_daily,
    ...     settle_f=92,  # Training only
    ... )
"""

from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd

from models.features.pipeline import SnapshotContext, compute_snapshot_features


# =============================================================================
# Snapshot Building from DataFrames
# =============================================================================

def build_snapshot(
    city: str,
    event_date: date,
    cutoff_time: datetime,
    obs_df: pd.DataFrame,
    window_start: Optional[datetime] = None,
    fcst_daily: Optional[dict] = None,
    fcst_daily_station: Optional[dict] = None,
    fcst_hourly_df: Optional[pd.DataFrame] = None,
    fcst_multi: Optional[dict[int, Optional[dict]]] = None,
    candles_df: Optional[pd.DataFrame] = None,
    city_obs_df: Optional[pd.DataFrame] = None,
    more_apis: Optional[dict] = None,
    obs_t15_mean_30d_f: Optional[float] = None,
    obs_t15_std_30d_f: Optional[float] = None,
    settle_f: Optional[int] = None,
    include_labels: bool = False,
) -> dict[str, Any]:
    """Build features for a single snapshot.

    This is the main entry point for snapshot building. It creates a
    SnapshotContext from the input data and computes all features.

    Args:
        city: City identifier (e.g., 'chicago')
        event_date: Settlement date (D)
        cutoff_time: Snapshot timestamp (observations up to this time)
        obs_df: DataFrame with station observations (datetime_local, temp_f, etc.)
        window_start: Start of observation window (default: D-1 10:00)
        fcst_daily: T-1 daily forecast dict (city-level)
        fcst_daily_station: T-1 daily forecast dict (station-level)
        fcst_hourly_df: T-1 hourly forecast DataFrame
        fcst_multi: Multi-horizon forecasts {lead_day: forecast_dict}
        candles_df: Market candle DataFrame
        city_obs_df: City-aggregate observations DataFrame
        settle_f: Settlement temperature (training only)
        include_labels: Whether to include delta/settle_f in output

    Returns:
        Feature dictionary
    """
    # Default window_start to D-1 10:00 (market open)
    if window_start is None:
        window_start = datetime.combine(
            event_date - timedelta(days=1),
            datetime.min.time()
        ).replace(hour=10, minute=0)

    # Filter observations to cutoff_time
    obs_filtered = _filter_obs_to_cutoff(obs_df, cutoff_time)

    # Extract temps and timestamps
    temps_sofar, timestamps_sofar = _extract_temps_and_times(obs_filtered)

    # Create context
    ctx = SnapshotContext(
        city=city,
        event_date=event_date,
        cutoff_time=cutoff_time,
        window_start=window_start,
        temps_sofar=temps_sofar,
        timestamps_sofar=timestamps_sofar,
        obs_df=obs_filtered,
        fcst_daily=fcst_daily,
        fcst_daily_station=fcst_daily_station,
        fcst_hourly_df=fcst_hourly_df,
        fcst_multi=fcst_multi,
        candles_df=_filter_candles_to_cutoff(candles_df, cutoff_time) if candles_df is not None else None,
        city_obs_df=_filter_obs_to_cutoff(city_obs_df, cutoff_time) if city_obs_df is not None else None,
        more_apis=more_apis,
        obs_t15_mean_30d_f=obs_t15_mean_30d_f,
        obs_t15_std_30d_f=obs_t15_std_30d_f,
        settle_f=settle_f,
    )

    return compute_snapshot_features(ctx, include_labels=include_labels)


def build_snapshot_for_inference(
    city: str,
    event_date: date,
    cutoff_time: datetime,
    obs_df: pd.DataFrame,
    fcst_daily: Optional[dict] = None,
    fcst_hourly_df: Optional[pd.DataFrame] = None,
    candles_df: Optional[pd.DataFrame] = None,
    city_obs_df: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Build features for inference (no labels).

    Convenience wrapper around build_snapshot() for inference use cases.
    """
    return build_snapshot(
        city=city,
        event_date=event_date,
        cutoff_time=cutoff_time,
        obs_df=obs_df,
        fcst_daily=fcst_daily,
        fcst_hourly_df=fcst_hourly_df,
        candles_df=candles_df,
        city_obs_df=city_obs_df,
        settle_f=None,
        include_labels=False,
    )


def build_snapshot_for_training(
    city: str,
    event_date: date,
    cutoff_time: datetime,
    obs_df: pd.DataFrame,
    settle_f: int,
    fcst_daily: Optional[dict] = None,
    fcst_hourly_df: Optional[pd.DataFrame] = None,
    candles_df: Optional[pd.DataFrame] = None,
    city_obs_df: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Build features for training (with labels).

    Convenience wrapper around build_snapshot() for training use cases.
    """
    return build_snapshot(
        city=city,
        event_date=event_date,
        cutoff_time=cutoff_time,
        obs_df=obs_df,
        fcst_daily=fcst_daily,
        fcst_hourly_df=fcst_hourly_df,
        candles_df=candles_df,
        city_obs_df=city_obs_df,
        settle_f=settle_f,
        include_labels=True,
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _filter_obs_to_cutoff(
    obs_df: Optional[pd.DataFrame],
    cutoff_time: datetime,
) -> pd.DataFrame:
    """Filter observations to only include those before cutoff_time."""
    if obs_df is None or obs_df.empty:
        return pd.DataFrame()

    df = obs_df.copy()

    # Ensure datetime_local is datetime type
    if "datetime_local" not in df.columns:
        return pd.DataFrame()

    if not pd.api.types.is_datetime64_any_dtype(df["datetime_local"]):
        df["datetime_local"] = pd.to_datetime(df["datetime_local"])

    # Handle timezone - make naive for comparison
    if df["datetime_local"].dt.tz is not None:
        df["datetime_local"] = df["datetime_local"].dt.tz_localize(None)

    # Ensure cutoff_time is naive
    cutoff_naive = cutoff_time
    if hasattr(cutoff_time, 'tzinfo') and cutoff_time.tzinfo is not None:
        cutoff_naive = cutoff_time.replace(tzinfo=None)

    # Filter
    df = df[df["datetime_local"] <= cutoff_naive]
    df = df.sort_values("datetime_local")

    return df


def _filter_candles_to_cutoff(
    candles_df: Optional[pd.DataFrame],
    cutoff_time: datetime,
) -> pd.DataFrame:
    """Filter candles to only include those before cutoff_time."""
    if candles_df is None or candles_df.empty:
        return pd.DataFrame()

    df = candles_df.copy()

    # Try different timestamp column names
    ts_col = None
    for col in ["bucket_start", "ts_local", "timestamp_local", "datetime_local", "ts"]:
        if col in df.columns:
            ts_col = col
            break

    if ts_col is None:
        return pd.DataFrame()

    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col])

    # Handle timezone
    if df[ts_col].dt.tz is not None:
        df[ts_col] = df[ts_col].dt.tz_localize(None)

    cutoff_naive = cutoff_time
    if hasattr(cutoff_time, 'tzinfo') and cutoff_time.tzinfo is not None:
        cutoff_naive = cutoff_time.replace(tzinfo=None)

    df = df[df[ts_col] <= cutoff_naive]
    df = df.sort_values(ts_col)

    return df


def _extract_temps_and_times(
    obs_df: pd.DataFrame,
) -> tuple[list[float], list[datetime]]:
    """Extract temperature and timestamp lists from observation DataFrame."""
    if obs_df.empty:
        return [], []

    temps = []
    timestamps = []

    for _, row in obs_df.iterrows():
        dt = row.get("datetime_local")
        temp = row.get("temp_f")

        if dt is not None and temp is not None and not pd.isna(temp):
            # Ensure datetime is naive
            if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)

            # Convert numpy datetime to python datetime if needed
            if isinstance(dt, pd.Timestamp):
                dt = dt.to_pydatetime()

            timestamps.append(dt)
            temps.append(float(temp))

    return temps, timestamps


# =============================================================================
# Time Window Generation
# =============================================================================

def generate_snapshot_times(
    window_start: datetime,
    window_end: datetime,
    interval_min: int = 5,
) -> list[datetime]:
    """Generate snapshot timestamps from window start to end.

    Args:
        window_start: First snapshot time
        window_end: Last snapshot time
        interval_min: Minutes between snapshots

    Returns:
        List of datetime objects at interval_min spacing
    """
    times = []
    current = window_start

    while current <= window_end:
        times.append(current)
        current += timedelta(minutes=interval_min)

    return times


def get_market_clock_window(event_date: date) -> tuple[datetime, datetime]:
    """Get market-clock window: D-1 10:00 to D 23:55.

    Args:
        event_date: The settlement date (D)

    Returns:
        Tuple of (window_start, window_end)
    """
    d_minus_1 = event_date - timedelta(days=1)
    window_start = datetime.combine(d_minus_1, datetime.min.time()).replace(hour=10, minute=0)
    window_end = datetime.combine(event_date, datetime.min.time()).replace(hour=23, minute=55)
    return window_start, window_end


def get_event_day_window(event_date: date) -> tuple[datetime, datetime]:
    """Get event-day window: D 10:00 to D 23:45.

    Args:
        event_date: The settlement date (D)

    Returns:
        Tuple of (window_start, window_end)
    """
    window_start = datetime.combine(event_date, datetime.min.time()).replace(hour=10, minute=0)
    window_end = datetime.combine(event_date, datetime.min.time()).replace(hour=23, minute=45)
    return window_start, window_end
