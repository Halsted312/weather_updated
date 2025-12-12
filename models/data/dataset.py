"""
Unified dataset builder for training and testing.

This module provides a generic dataset builder that works with any time window
configuration. It loops over cities, dates, and snapshot times to build a
complete dataset using the unified feature pipeline.

Example:
    >>> from models.data.dataset import DatasetConfig, build_dataset
    >>> from src.db.connection import get_db_session
    >>>
    >>> config = DatasetConfig(
    ...     time_window="market_clock",  # D-1 10:00 to D 23:55
    ...     snapshot_interval_min=5,
    ...     include_forecast=True,
    ...     include_market=True,
    ... )
    >>>
    >>> with get_db_session() as session:
    ...     df = build_dataset(
    ...         cities=["chicago", "austin"],
    ...         start_date=date(2024, 1, 1),
    ...         end_date=date(2024, 12, 31),
    ...         config=config,
    ...         session=session,
    ...     )
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterator, Literal, Optional

import pandas as pd
from sqlalchemy.orm import Session

from models.data.loader import (
    load_vc_observations,
    load_settlements,
    load_historical_forecast_daily,
    load_historical_forecast_hourly,
    load_multi_horizon_forecasts,
    batch_load_multi_horizon_forecasts,
    get_vc_location_id,
)
from models.data.snapshot import (
    build_snapshot,
    generate_snapshot_times,
    get_market_clock_window,
    get_event_day_window,
)
from models.features.calendar import add_lag_features_to_dataframe

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset building.

    Attributes:
        time_window: Predefined window type or "custom"
            - "market_clock": D-1 10:00 to D 23:55
            - "event_day": D 10:00 to D 23:45
            - "custom": Use custom_start_offset_hours and custom_end_hour

        snapshot_interval_min: Minutes between snapshots (default 5)

        custom_start_offset_hours: For custom window, hours before D midnight
            (e.g., 14 for D-1 10:00, or 0 for D midnight)
        custom_end_hour: For custom window, end hour on event day
        custom_end_minute: For custom window, end minute

        include_forecast: Whether to load and use forecast data
        include_market: Whether to load and use candle data
        include_station_city: Whether to load city-aggregate observations
        include_meteo: Whether to include meteo features (requires obs columns)

        min_obs_per_snapshot: Minimum observations required per snapshot
    """
    time_window: Literal["market_clock", "event_day", "custom"] = "market_clock"
    snapshot_interval_min: int = 5

    # Custom window settings (used when time_window="custom")
    custom_start_offset_hours: float = 14.0  # Hours before D midnight (14 = D-1 10:00)
    custom_end_hour: int = 23
    custom_end_minute: int = 55

    # Optional features - ALL TRUE by default for full feature set
    include_forecast: bool = True
    include_multi_horizon: bool = True  # T-6 through T-1 forecast evolution
    include_market: bool = True  # Kalshi candle data
    include_station_city: bool = True  # City-aggregate obs for station-city gap
    include_meteo: bool = True  # Humidity, wind, cloud cover
    include_lags: bool = True  # Lag features (settle_f_lag1, etc.)
    include_more_apis: bool = True  # NOAA guidance (NBM, HRRR, NDFD)

    # Quality filters
    min_obs_per_snapshot: int = 1


# =============================================================================
# Dataset Building
# =============================================================================

def build_dataset(
    cities: list[str],
    start_date: date,
    end_date: date,
    config: DatasetConfig,
    session: Session,
    progress_callback: Optional[callable] = None,
) -> pd.DataFrame:
    """Build a training/test dataset with unified feature pipeline.

    Args:
        cities: List of city identifiers
        start_date: First event date to include
        end_date: Last event date to include
        config: Dataset configuration
        session: Database session
        progress_callback: Optional callback(city, event_date, n_snapshots)

    Returns:
        DataFrame with one row per snapshot, all features computed
    """
    all_rows = []

    for city in cities:
        logger.info(f"Building dataset for {city}...")

        # Load settlements
        settlements_df = load_settlements(session, city, start_date, end_date)
        if settlements_df.empty:
            logger.warning(f"No settlements found for {city}")
            continue

        # Load observations for the entire date range
        # Extend range to include D-1 for market-clock window
        obs_start = start_date - timedelta(days=1)
        obs_df = load_vc_observations(session, city, obs_start, end_date)
        if obs_df.empty:
            logger.warning(f"No observations found for {city}")
            continue

        # Add date column for filtering
        obs_df["obs_date"] = pd.to_datetime(obs_df["datetime_local"]).dt.date

        # Load city observations if needed
        city_obs_df = None
        if config.include_station_city:
            city_obs_df = _load_city_observations(session, city, obs_start, end_date)

        # OPTIMIZATION: Batch-load multi-horizon forecasts for ALL dates at once
        all_fcst_multi = None
        if config.include_multi_horizon:
            logger.info(f"Batch-loading multi-horizon forecasts for {city}...")
            all_fcst_multi = batch_load_multi_horizon_forecasts(
                session, city, start_date, end_date, lead_days=[1, 2, 3, 4, 5, 6]  # T-1 to T-6 only
            )

        # OPTIMIZATION: Batch-load candles for ALL dates at once
        all_candles_df = None
        if config.include_market:
            logger.info(f"Batch-loading candles for {city}...")
            all_candles_df = _batch_load_candles(session, city, start_date, end_date)

        # Process each event date
        for _, settle_row in settlements_df.iterrows():
            event_date = settle_row["date_local"]
            settle_f = int(settle_row["tmax_final"])

            # Get time window
            window_start, window_end = _get_window(event_date, config)

            # Generate snapshot times
            snapshot_times = generate_snapshot_times(
                window_start, window_end, config.snapshot_interval_min
            )

            # Load forecast for this day (both city and station levels)
            fcst_daily = None
            fcst_daily_station = None
            fcst_hourly_df = None
            fcst_multi = None
            if config.include_forecast:
                basis_date = event_date - timedelta(days=1)
                fcst_daily = load_historical_forecast_daily(session, city, event_date, basis_date, location_type="city")
                fcst_daily_station = load_historical_forecast_daily(session, city, event_date, basis_date, location_type="station")
                fcst_hourly_df = load_historical_forecast_hourly(session, city, event_date, basis_date)
                if fcst_hourly_df is not None and fcst_hourly_df.empty:
                    fcst_hourly_df = None

            # Extract multi-horizon forecasts for this day from pre-loaded data
            fcst_multi = None
            if all_fcst_multi is not None:
                fcst_multi = {
                    lead: all_fcst_multi.get((event_date, lead))
                    for lead in range(1, 7)  # T-1 to T-6 only
                }

            # Filter candles for this day from pre-loaded data
            candles_df = None
            if all_candles_df is not None:
                day_candles = all_candles_df[all_candles_df['event_date'] == event_date]
                candles_df = day_candles if not day_candles.empty else None

            # Load NOAA model guidance for this day
            more_apis = None
            obs_t15_mean = None
            obs_t15_std = None
            if config.include_more_apis:
                from models.data.loader import load_weather_more_apis_guidance, load_obs_t15_stats_30d
                more_apis = load_weather_more_apis_guidance(session, city, event_date)
                obs_t15_mean, obs_t15_std = load_obs_t15_stats_30d(session, city, event_date)

            # Filter observations to relevant dates
            d_minus_1 = event_date - timedelta(days=1)
            day_obs_df = obs_df[
                (obs_df["obs_date"] == event_date) | (obs_df["obs_date"] == d_minus_1)
            ].copy()

            day_city_obs_df = None
            if city_obs_df is not None:
                day_city_obs_df = city_obs_df[
                    (city_obs_df["obs_date"] == event_date) | (city_obs_df["obs_date"] == d_minus_1)
                ].copy()

            # Build snapshot for each time
            n_snapshots = 0
            for cutoff_time in snapshot_times:
                try:
                    features = build_snapshot(
                        city=city,
                        event_date=event_date,
                        cutoff_time=cutoff_time,
                        obs_df=day_obs_df,
                        window_start=window_start,
                        fcst_daily=fcst_daily,
                        fcst_daily_station=fcst_daily_station,
                        fcst_hourly_df=fcst_hourly_df,
                        fcst_multi=fcst_multi,
                        candles_df=candles_df,
                        city_obs_df=day_city_obs_df,
                        more_apis=more_apis,
                        obs_t15_mean_30d_f=obs_t15_mean,
                        obs_t15_std_30d_f=obs_t15_std,
                        settle_f=settle_f,
                        include_labels=True,
                    )

                    # Skip if insufficient observations
                    n_obs = features.get("num_samples_sofar", 0) or 0
                    if n_obs < config.min_obs_per_snapshot:
                        continue

                    # Add day column for splitting (use event_date, not cutoff date)
                    features["day"] = event_date

                    all_rows.append(features)
                    n_snapshots += 1

                except Exception as e:
                    logger.warning(f"Error building snapshot for {city}/{event_date}/{cutoff_time}: {e}")
                    continue

            if progress_callback:
                progress_callback(city, event_date, n_snapshots)

    logger.info(f"Built {len(all_rows)} snapshots total")
    df = pd.DataFrame(all_rows)

    # Add lag features if enabled and we have enough data
    if config.include_lags and len(df) > 0 and "settle_f" in df.columns:
        logger.info("Adding lag features...")
        df = add_lag_features_to_dataframe(df)
        logger.info(f"Lag features added: {[c for c in df.columns if 'lag' in c.lower()]}")

    return df


# =============================================================================
# Helper Functions
# =============================================================================

def _get_window(event_date: date, config: DatasetConfig) -> tuple[datetime, datetime]:
    """Get window start and end times based on config."""
    if config.time_window == "market_clock":
        return get_market_clock_window(event_date)
    elif config.time_window == "event_day":
        return get_event_day_window(event_date)
    else:  # custom
        # Calculate start from offset
        d_midnight = datetime.combine(event_date, datetime.min.time())
        window_start = d_midnight - timedelta(hours=config.custom_start_offset_hours)
        window_end = d_midnight.replace(
            hour=config.custom_end_hour,
            minute=config.custom_end_minute
        )
        return window_start, window_end


def _load_city_observations(
    session: Session,
    city: str,
    start_date: date,
    end_date: date,
) -> Optional[pd.DataFrame]:
    """Load city-aggregate observations.

    City observations use location_type='city' instead of 'station'.
    """
    from sqlalchemy import text
    from src.config.cities import get_city

    city_config = get_city(city)
    vc_location_id = get_vc_location_id(session, city, "city")

    if vc_location_id is None:
        logger.warning(f"No city location found for {city}")
        return None

    query = text("""
        SELECT
            datetime_local,
            datetime_utc,
            temp_f,
            humidity,
            windspeed_mph,
            cloudcover
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

    df = pd.DataFrame(result.fetchall(), columns=[
        "datetime_local", "datetime_utc", "temp_f",
        "humidity", "windspeed_mph", "cloudcover"
    ])

    # Always add obs_date column (even for empty DataFrames) to avoid KeyError
    if not df.empty:
        df["obs_date"] = pd.to_datetime(df["datetime_local"]).dt.date
    else:
        df["obs_date"] = pd.Series(dtype="object")

    return df


def _batch_load_candles(
    session: Session,
    city: str,
    start_date: date,
    end_date: date,
) -> Optional[pd.DataFrame]:
    """Load ALL market candles for entire date range at once (batch mode).

    Instead of querying once per day, load all candles for the date range
    in a single query, then filter by day in memory.

    Args:
        session: Database session
        city: City identifier
        start_date: First date
        end_date: Last date

    Returns:
        DataFrame with candles for all days, with 'event_date' column added
        for filtering. Returns None if no candles found.
    """
    from sqlalchemy import text

    # Map city names to ticker patterns
    city_ticker_map = {
        "chicago": "CHI",
        "austin": "AUS",
        "denver": "DEN",
        "los_angeles": "LAX",
        "miami": "MIA",
        "philadelphia": "PHL",
    }
    ticker_pattern = city_ticker_map.get(city.lower(), city.upper()[:3])

    # Compute window for entire date range
    # Market clock: D-1 10:00 to D 23:55, so we need D-1 for first date
    first_window_start, _ = get_market_clock_window(start_date)
    _, last_window_end = get_market_clock_window(end_date)

    query = text("""
        SELECT
            bucket_start,
            ticker,
            yes_bid_close,
            yes_ask_close,
            volume,
            open_interest,
            has_trade,
            is_synthetic
        FROM kalshi.candles_1m_dense
        WHERE ticker LIKE :ticker_pattern
          AND bucket_start >= :window_start
          AND bucket_start <= :window_end
        ORDER BY bucket_start
    """)

    try:
        result = session.execute(query, {
            "ticker_pattern": f"%{ticker_pattern}%",
            "window_start": first_window_start,
            "window_end": last_window_end,
        })

        df = pd.DataFrame(result.fetchall(), columns=[
            "bucket_start", "ticker", "yes_bid_close", "yes_ask_close", "volume", "open_interest",
            "has_trade", "is_synthetic"
        ])

        if df.empty:
            return None

        # Add event_date column for filtering
        # Extract date from ticker (e.g., "HIGHAUS-23AUG01-B100.5" -> 2023-08-01)
        # Ticker format: HIGHXXX-YYMMMDD-bracket (e.g., HIGHAUS-23AUG01-B100.5)
        def extract_event_date(ticker_str):
            try:
                # Format: HIGHAUS-23AUG01-B100.5 (3 parts) or HIGHCHI-25APR01 (2 parts)
                parts = ticker_str.split('-')
                if len(parts) < 2:
                    return None
                date_str = parts[1]  # Date is always the second part
                # Try parsing YYMMMDD format first (23AUG01)
                try:
                    return pd.to_datetime(date_str, format='%y%b%d').date()
                except:
                    # Try parsing YYMMDD format (230801)
                    try:
                        return pd.to_datetime(date_str, format='%y%m%d').date()
                    except:
                        return None
            except:
                return None

        df['event_date'] = df['ticker'].apply(extract_event_date)

        logger.info(f"Batch-loaded {len(df)} candles for {city} ({start_date} to {end_date})")
        return df

    except Exception as e:
        logger.warning(f"Could not batch-load candles: {e}")
        return None


def _load_candles(
    session: Session,
    city: str,
    event_date: date,
    window_start: datetime,
    window_end: datetime,
) -> Optional[pd.DataFrame]:
    """Load market candles for the given window.

    Delegates to the public load_candles_for_inference() in loader.py
    to ensure parity between training and inference pipelines.
    """
    from models.data.loader import load_candles_for_inference

    return load_candles_for_inference(
        session=session,
        city_id=city,
        event_date=event_date,
        window_start=window_start,
        cutoff_time=window_end,
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def build_market_clock_dataset(
    cities: list[str],
    start_date: date,
    end_date: date,
    session: Session,
    snapshot_interval_min: int = 5,
    include_forecast: bool = True,
    include_market: bool = False,
    include_station_city: bool = False,
) -> pd.DataFrame:
    """Build dataset with market-clock window (D-1 10:00 to D 23:55).

    Convenience function with sensible defaults for market-clock models.
    """
    config = DatasetConfig(
        time_window="market_clock",
        snapshot_interval_min=snapshot_interval_min,
        include_forecast=include_forecast,
        include_market=include_market,
        include_station_city=include_station_city,
    )
    return build_dataset(cities, start_date, end_date, config, session)


def build_event_day_dataset(
    cities: list[str],
    start_date: date,
    end_date: date,
    session: Session,
    snapshot_interval_min: int = 15,
    include_forecast: bool = True,
) -> pd.DataFrame:
    """Build dataset with event-day window (D 10:00 to D 23:45).

    Convenience function with sensible defaults for TOD v1 models.
    """
    config = DatasetConfig(
        time_window="event_day",
        snapshot_interval_min=snapshot_interval_min,
        include_forecast=include_forecast,
        include_market=False,
        include_station_city=False,
    )
    return build_dataset(cities, start_date, end_date, config, session)
