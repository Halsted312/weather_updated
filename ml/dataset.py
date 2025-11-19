#!/usr/bin/env python3
"""
Dataset builder for Kalshi weather ML training.

Joins 1-min candles to 1-min weather grid and market metadata,
extracts features, and generates labels for Ridge baseline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import date, datetime, timezone, timedelta
from typing import Dict, List, Literal, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import text
from zoneinfo import ZoneInfo

from db.connection import get_session
from ml.features import FeatureBuilder
from pathlib import Path
from ml.city_config import CITY_CONFIG as _CITY_CONFIG, EXCLUDED_VC_CITIES as _EXCLUDED_VC_CITIES
from ml.date_utils import event_date_from_close_time, series_ticker_for_city

logger = logging.getLogger(__name__)

# Re-export canonical config for backwards compatibility
CITY_CONFIG = _CITY_CONFIG
EXCLUDED_VC_CITIES = _EXCLUDED_VC_CITIES


def _write_feature_completeness(
    csv_path: Path,
    df_before: pd.DataFrame,
    df_after_critical: Optional[pd.DataFrame],
    df_after_impute: pd.DataFrame,
    critical_features: List[str],
    optional_features: List[str],
    split: str,
):
    """
    Write feature completeness audit to CSV.

    Logs:
    - Row counts before/after critical drop/impute
    - Missing counts for CRITICAL and OPTIONAL features
    - Class balance (pct_yes)

    Args:
        csv_path: Path to output CSV
        df_before: DataFrame after feature building, before NA policy
        df_after_critical: DataFrame after dropping critical NAs (None if no drop)
        df_after_impute: Final DataFrame after imputing optional features
        critical_features: List of CRITICAL feature names
        optional_features: List of OPTIONAL feature names
        split: "train" or "test"
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows_before = len(df_before)
    rows_after_critical = len(df_after_critical) if df_after_critical is not None else rows_before
    rows_after_impute = len(df_after_impute)

    pct_kept_critical = 100.0 * rows_after_critical / rows_before if rows_before > 0 else 0.0
    pct_kept_final = 100.0 * rows_after_impute / rows_before if rows_before > 0 else 0.0

    # Count missing values in df_before
    row = {
        "split": split,
        "rows_before": rows_before,
        "rows_after_critical_drop": rows_after_critical,
        "rows_after_optional_impute": rows_after_impute,
        "pct_kept_after_critical": f"{pct_kept_critical:.1f}%",
        "pct_kept_final": f"{pct_kept_final:.1f}%",
    }

    # Count missing for CRITICAL features
    for col in critical_features:
        if col in df_before.columns:
            row[f"missing_{col}"] = int(df_before[col].isna().sum())
        else:
            row[f"missing_{col}"] = -1  # Column doesn't exist

    # Count missing for OPTIONAL features
    for col in optional_features:
        if col in df_before.columns:
            row[f"missing_{col}"] = int(df_before[col].isna().sum())
        else:
            row[f"missing_{col}"] = -1

    # Class balance if label present
    if "settlement_value" in df_after_impute.columns:
        pct_yes = 100.0 * (df_after_impute["settlement_value"] == 100.0).mean()
        row["pct_yes"] = f"{pct_yes:.1f}%"
    else:
        row["pct_yes"] = "N/A"

    # Write to CSV (append mode)
    pd.DataFrame([row]).to_csv(csv_path, mode="a", index=False, header=not csv_path.exists())
    logger.info(f"Feature completeness logged to {csv_path}")


def load_candles_with_weather_and_metadata(
    city: str,
    start_date: date,
    end_date: date,
    bracket_type: Optional[Literal["greater", "less", "between"]] = None,
) -> pd.DataFrame:
    """
    Load 1-min candles joined to weather and market metadata.

    Args:
        city: City name (e.g., "chicago")
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        bracket_type: Filter to specific bracket type (optional)

    Returns:
        DataFrame with columns:
            - market_ticker
            - timestamp (candle end time, UTC)
            - open, high, low, close (prices in cents)
            - volume
            - temp_f (weather observation at candle time)
            - close_time (market close time)
            - settlement_value (0 or 1 for NO/YES, or NULL if not settled)
            - strike_type, floor_strike, cap_strike
    """
    if city not in CITY_CONFIG:
        raise ValueError(f"Unknown city: {city}. Valid cities: {list(CITY_CONFIG.keys())}")

    loc_id = CITY_CONFIG[city]["loc_id"]
    series_ticker = series_ticker_for_city(city)

    logger.info(f"Loading candles for {city} ({start_date} to {end_date})")

    with get_session() as session:
        # Convert requested local dates to UTC ranges so markets closing after midnight
        # (local) are still included in the requested window.
        city_tz = ZoneInfo(CITY_CONFIG[city]["timezone"])
        start_local = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=city_tz)
        end_local = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=city_tz)
        start_dt = start_local.astimezone(timezone.utc)
        end_dt = end_local.astimezone(timezone.utc)

        # SQL query to join candles, weather, and metadata
        query = text("""
            WITH candle_data AS (
                SELECT
                    c.market_ticker,
                    c.timestamp,
                    c.open,
                    c.high,
                    c.low,
                    c.close,
                    c.volume
                FROM candles c
                WHERE c.market_ticker LIKE :series_pattern
                  AND c.timestamp >= :start_dt
                  AND c.timestamp <= :end_dt
            ),
            market_meta AS (
                SELECT
                    m.ticker as market_ticker,
                    m.close_time,
                    m.settlement_value,
                    m.strike_type,
                    m.floor_strike,
                    m.cap_strike
                FROM markets m
                WHERE m.ticker LIKE :series_pattern
                  AND m.close_time >= :start_dt
                  AND m.close_time <= :end_dt
            )
            SELECT
                cd.market_ticker,
                cd.timestamp,
                cd.open,
                cd.high,
                cd.low,
                cd.close,
                cd.volume,
                wd.temp_f,
                wd.dew_f,
                wd.humidity_pct,
                wd.wind_mph,
                mm.close_time,
                mm.settlement_value,
                mm.strike_type,
                mm.floor_strike,
                mm.cap_strike
            FROM candle_data cd
            LEFT JOIN LATERAL (
                SELECT
                    w.temp_f,
                    w.dew_f,
                    w.humidity as humidity_pct,
                    w.windspeed_mph as wind_mph
                FROM wx.minute_obs w
                WHERE w.loc_id = :loc_id
                  AND w.ts_utc <= cd.timestamp
                ORDER BY w.ts_utc DESC
                LIMIT 1
            ) wd ON true
            LEFT JOIN market_meta mm
              ON cd.market_ticker = mm.market_ticker
            ORDER BY cd.market_ticker, cd.timestamp
        """)

        params = {
            "series_pattern": f"{series_ticker}%",
            "loc_id": loc_id,
            "start_dt": start_dt,
            "end_dt": end_dt,
        }

        df = pd.read_sql(query, session.bind, params=params)

    if df.empty:
        return df

    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    if df["close_time"].isna().any():
        missing = df["close_time"].isna().sum()
        logger.warning("Dropping %s rows with NULL close_time for %s", missing, city)
        df = df.dropna(subset=["close_time"]).copy()

    df["event_date"] = df["close_time"].apply(
        lambda ct: event_date_from_close_time(series_ticker, ct)
    )

    logger.info(f"Loaded {len(df)} candle rows")

    # Filter by bracket type if specified
    if bracket_type:
        df = df[df["strike_type"] == bracket_type].copy()
        logger.info(f"Filtered to {len(df)} rows for bracket_type={bracket_type}")

    # Drop rows without settlement_value (not yet settled)
    df = df[df["settlement_value"].notna()].copy()
    logger.info(f"Filtered to {len(df)} rows with settlement values")

    return df


def build_training_dataset(
    city: str,
    start_date: date,
    end_date: date,
    bracket_type: Literal["greater", "less", "between"],
    feature_set: str = "baseline",
    completeness_csv_path: Optional[Path] = None,
    split: str = "train",
    persist_feature_snapshots: bool = False,
    max_minutes_to_close: Optional[float] = None,
    prior_peak_back_minutes: Optional[float] = None,
    prior_peak_lookup_days: int = 3,
    prior_peak_default_minutes: Optional[float] = None,
    return_feature_names: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build ML training dataset for a specific bracket type.

    Args:
        city: City name (e.g., "chicago")
        start_date: Training start date (inclusive)
        end_date: Training end date (inclusive)
        bracket_type: Bracket type to train on
        feature_set: Feature set to use ("baseline", "ridge_conservative", "elasticnet_rich")
        completeness_csv_path: Optional path to save feature completeness audit CSV
        split: "train" or "test" label for completeness logging
        persist_feature_snapshots: If True, snaps engineered features to Postgres for reuse
        max_minutes_to_close: If set, drop rows with minutes_to_close greater than this value
        prior_peak_back_minutes: If set, keep only rows after (yesterday's peak time minus this many minutes)
        prior_peak_lookup_days: How many days back to search for a valid peak time
        prior_peak_default_minutes: Fallback start minute (since midnight local) if no peak found

    Returns:
        Tuple of (X, y, groups, metadata):
            - X: numpy array of features (N x F) where F depends on feature_set
            - y: numpy array of binary labels (N,) - 1 if YES, 0 if NO
            - groups: numpy array of day indices (N,) for GroupKFold
            - metadata: DataFrame with [market_ticker, timestamp, strike_type, ...]
    """
    # Load data
    df = load_candles_with_weather_and_metadata(
        city=city,
        start_date=start_date,
        end_date=end_date,
        bracket_type=bracket_type,
    )

    if df.empty:
        logger.warning(f"No data for {city} {bracket_type} in date range")
        return (
            np.array([]).reshape(0, 10),
            np.array([]),
            np.array([]),
            pd.DataFrame(),
        )

    # Convert to FeatureBuilder input format
    # Include VC weather columns (temp_f, dew_f, humidity_pct, wind_mph)
    # These will be NULL for excluded cities (e.g., NYC)
    candle_cols = [
        "market_ticker",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "temp_f",
        "dew_f",
        "humidity_pct",
        "wind_mph",
    ]
    if "event_date" in df.columns:
        candle_cols.append("event_date")

    candles_df = df[candle_cols].copy()

    # Rename columns for FeatureBuilder
    candles_df["end_period_ts"] = candles_df["timestamp"]

    # Market metadata
    market_metadata = df[[
        "market_ticker",
        "close_time",
        "strike_type",
        "floor_strike",
        "cap_strike",
    ]].drop_duplicates(subset=["market_ticker"]).copy()

    # Build features using FeatureBuilder
    fb = FeatureBuilder(city_timezone=CITY_CONFIG[city]["timezone"])
    features_df = fb.build_features(
        candles_df,
        weather_df=None,
        market_metadata=market_metadata,
        feature_set=feature_set,
    )

    # DEBUG: Check if event_date exists
    logger.info(f"Features columns after build_features: {list(features_df.columns)}")
    if "event_date" not in features_df.columns:
        logger.error("EVENT_DATE MISSING! Creating it now...")
        if "timestamp_local" in features_df.columns:
            features_df["event_date"] = features_df["timestamp_local"].dt.date
        elif "timestamp" in features_df.columns:
            features_df["event_date"] = pd.to_datetime(features_df["timestamp"]).dt.tz_localize("UTC").dt.tz_convert(CITY_CONFIG[city]["timezone"]).dt.date

    if features_df.empty:
        logger.warning(f"No features built for {city} {bracket_type}")
        return (
            np.array([]).reshape(0, 10),
            np.array([]),
            np.array([]),
            pd.DataFrame(),
        )

    if max_minutes_to_close is not None and "minutes_to_close" in features_df.columns:
        before = len(features_df)
        features_df = features_df[features_df["minutes_to_close"] <= max_minutes_to_close].copy()
        after = len(features_df)
        logger.info(
            f"Filtered rows by minutes_to_close <= {max_minutes_to_close}: kept {after}/{before} "
            f"({after / before * 100 if before else 0:.1f}%)"
        )
        if features_df.empty:
            logger.warning("All rows removed by minutes_to_close filter")
            return (
                np.array([]).reshape(0, 10),
                np.array([]),
                np.array([]),
                pd.DataFrame(),
            )

    if prior_peak_back_minutes is not None:
        tz = ZoneInfo(CITY_CONFIG[city]["timezone"])
        if "timestamp_local" not in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["timestamp_local"] = df["timestamp"].dt.tz_convert(tz)
        temp_df = df.dropna(subset=["temp_f"]).copy()
        peak_map: Dict[date, float] = {}
        if not temp_df.empty:
            temp_df["minutes_since_midnight_local"] = (
                temp_df["timestamp_local"].dt.hour * 60
                + temp_df["timestamp_local"].dt.minute
            )
            temp_df["date_local"] = temp_df["timestamp_local"].dt.date
            idx = temp_df.groupby("date_local")["temp_f"].idxmax()
            peak_map = dict(
                zip(
                    temp_df.loc[idx, "date_local"],
                    temp_df.loc[idx, "minutes_since_midnight_local"],
                )
            )

        if "minutes_since_midnight_local" not in features_df.columns:
            features_df["minutes_since_midnight_local"] = (
                features_df["timestamp_local"].dt.hour * 60
                + features_df["timestamp_local"].dt.minute
            )

        def lookup_peak_start(event_dt: date) -> Optional[float]:
            prev = event_dt - timedelta(days=1)
            for _ in range(max(1, prior_peak_lookup_days)):
                minutes = peak_map.get(prev)
                if minutes is not None:
                    return max(0.0, minutes - prior_peak_back_minutes)
                prev -= timedelta(days=1)
            return prior_peak_default_minutes

        features_df["peak_start_minutes"] = features_df["event_date"].apply(lookup_peak_start)

        before = len(features_df)
        features_df = features_df[
            features_df["peak_start_minutes"].notna()
            & (
                features_df["minutes_since_midnight_local"]
                >= features_df["peak_start_minutes"]
            )
        ].copy()
        after = len(features_df)
        logger.info(
            f"Filtered rows by prior peak window: kept {after}/{before} "
            f"({after / before * 100 if before else 0:.1f}%)"
        )
        if features_df.empty:
            logger.warning("All rows removed by prior peak filter")
            return (
                np.array([]).reshape(0, 10),
                np.array([]),
                np.array([]),
                pd.DataFrame(),
            )

    # Merge with original df to get settlement_value
    if "timestamp" in features_df.columns:
        features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], utc=True).dt.tz_convert(None)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)

    features_df = features_df.merge(
        df[["market_ticker", "timestamp", "settlement_value"]],
        on=["market_ticker", "timestamp"],
        how="left",
    )

    # Fill NaN values for bracket-type-specific features
    # For "greater" markets: temp_to_cap is NaN (no cap), fill with large value
    # For "less" markets: temp_to_floor is NaN (no floor), fill with large negative value
    # For "between" markets: both should be present
    if bracket_type == "greater":
        features_df["temp_to_cap"] = features_df["temp_to_cap"].fillna(999.0)
    elif bracket_type == "less":
        features_df["temp_to_floor"] = features_df["temp_to_floor"].fillna(-999.0)
    # "between" markets should have both; any NaN is a real missing value

    # Two-tier NA policy: CRITICAL vs OPTIONAL features
    # CRITICAL: price features, temp_now, strike distances, time-to-close (must exist)
    # OPTIONAL: VC-derived features (humidity, dewpoint, windspeed, deltas, rolling stats)

    CRITICAL_FEATURES = [
        "yes_mid",
        "minutes_to_close",
        "temp_now",
        "temp_to_floor",
        "temp_to_cap",
    ]

    feature_cols = fb.get_feature_columns(feature_set=feature_set)

    # Identify optional features (not in critical list)
    optional_features = [f for f in feature_cols if f not in CRITICAL_FEATURES]

    # Capture DataFrame before NA policy (for completeness logging)
    df_before_na_policy = features_df.copy()

    # Drop rows with missing CRITICAL features or settlement_value
    df_after_critical = features_df.dropna(subset=CRITICAL_FEATURES + ["settlement_value"]).copy()

    if df_after_critical.empty:
        logger.warning(f"No rows with complete CRITICAL features for {city} {bracket_type}")
        # Log completeness even if empty
        if completeness_csv_path:
            _write_feature_completeness(
                csv_path=completeness_csv_path,
                df_before=df_before_na_policy,
                df_after_critical=df_after_critical,
                df_after_impute=df_after_critical,  # Same as after_critical since empty
                critical_features=CRITICAL_FEATURES,
                optional_features=optional_features,
                split=split,
            )
        return (
            np.array([]).reshape(0, 10),
            np.array([]),
            np.array([]),
            pd.DataFrame(),
        )

    # Add NA indicator columns for OPTIONAL features BEFORE imputation
    # This allows ElasticNet to learn "data missing" as a signal
    for col in optional_features:
        if col in df_after_critical.columns:
            na_flag = f"{col}_is_na"
            df_after_critical[na_flag] = df_after_critical[col].isna().astype("int8")

    # Impute OPTIONAL features with 0 (dtype-stable to avoid FutureWarning)
    for col in optional_features:
        if col in df_after_critical.columns:
            # Ensure float dtype before fillna to avoid pandas FutureWarning
            if not pd.api.types.is_float_dtype(df_after_critical[col]):
                df_after_critical[col] = pd.to_numeric(df_after_critical[col], errors="coerce")
            df_after_critical[col] = df_after_critical[col].astype("float32").fillna(0.0)

    features_df = df_after_critical

    logger.info(f"Kept {len(features_df)} rows after two-tier NA policy (critical+impute optional)")

    # Write feature completeness audit if path provided
    if completeness_csv_path:
        _write_feature_completeness(
            csv_path=completeness_csv_path,
            df_before=df_before_na_policy,
            df_after_critical=df_after_critical,
            df_after_impute=features_df,
            critical_features=CRITICAL_FEATURES,
            optional_features=optional_features,
            split=split,
        )

    # Extract X, y
    # Include both original feature columns AND NA indicator columns for optional features
    na_indicator_cols = [f"{col}_is_na" for col in optional_features if f"{col}_is_na" in features_df.columns]
    all_feature_cols = list(dict.fromkeys(list(feature_cols) + na_indicator_cols))

    if persist_feature_snapshots:
        try:
            from ml.feature_store import persist_feature_snapshots as store_snapshots

            snapshot_cols = ["market_ticker", "timestamp"] + all_feature_cols
            missing_snapshot_cols = [col for col in snapshot_cols if col not in features_df.columns]
            if missing_snapshot_cols:
                logger.warning(
                    "Cannot persist features for %s %s; missing columns %s",
                    city,
                    bracket_type,
                    missing_snapshot_cols,
                )
            else:
                snapshot_df = features_df[snapshot_cols].copy()
                store_snapshots(
                    city=city,
                    feature_set=feature_set,
                    features_df=snapshot_df,
                    feature_cols=all_feature_cols,
                )
        except Exception as exc:
            logger.error("Feature snapshot persistence failed for %s/%s: %s", city, bracket_type, exc)

    # Convert settlement_value to binary: 1 if YES (100), 0 if NO (0)
    X = features_df[all_feature_cols].values
    y = (features_df["settlement_value"] == 100.0).astype(int).values

    # Create groups (day index) for GroupKFold
    # Group by event date to prevent temporal leakage
    # event_date should already be in features_df from build_features
    # Only merge if it's missing (defensive code)
    if "event_date" not in features_df.columns:
        logger.warning("event_date missing from features_df, merging from original df")
        features_df = features_df.merge(
            df[["market_ticker", "event_date"]].drop_duplicates(),
            on="market_ticker",
            how="left"
        )

    # Convert dates to integer indices
    unique_dates = sorted(features_df["event_date"].unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    groups = features_df["event_date"].map(date_to_idx).values

    # Metadata
    metadata = features_df[[
        "market_ticker",
        "timestamp",
        "event_date",
    ]].copy()
    metadata["bracket_type"] = bracket_type

    # Add bracket_key for unified head (derive from strike_type and strikes)
    if "floor_strike" in features_df.columns and "cap_strike" in features_df.columns:
        metadata["floor_strike"] = features_df["floor_strike"]
        metadata["cap_strike"] = features_df["cap_strike"]

        # Create bracket_key for identification across 6 brackets
        def make_bracket_key(row):
            if row["bracket_type"] == "less":
                return f"less_{int(row['cap_strike'])}"
            elif row["bracket_type"] == "greater":
                return f"greater_{int(row['floor_strike'])}"
            elif row["bracket_type"] == "between":
                return f"between_{int(row['floor_strike'])}_{int(row['cap_strike'])}"
            else:
                return f"unknown_{row['bracket_type']}"

        metadata["bracket_key"] = metadata.apply(make_bracket_key, axis=1)

        # Bracket coverage validation (for unified head)
        bracket_coverage = (
            metadata[["event_date", "bracket_key"]]
            .drop_duplicates()
            .groupby("event_date")
            .size()
        )
        incomplete_days = bracket_coverage[bracket_coverage != 6]
        if len(incomplete_days) > 0:
            logger.warning(
                f"Incomplete bracket coverage on {len(incomplete_days)} days "
                f"(expected 6 brackets per day): {list(incomplete_days.index[:5])}..."
            )

    logger.info(
        f"Built dataset: {len(X)} rows, {X.shape[1]} features, "
        f"{len(unique_dates)} days, {len(np.unique(y))} classes"
    )
    logger.info(f"Label distribution: YES={np.sum(y)}, NO={len(y) - np.sum(y)}")

    if return_feature_names:
        return X, y, groups, metadata, all_feature_cols

    if return_feature_names:
        return X, y, groups, metadata, all_feature_cols

    return X, y, groups, metadata


def load_training_datasets_for_city(
    city: str,
    start_date: date,
    end_date: date,
    feature_set: str = "baseline",
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]]:
    """
    Load training datasets for all bracket types for a city.

    Args:
        city: City name (e.g., "chicago")
        start_date: Training start date (inclusive)
        end_date: Training end date (inclusive)
        feature_set: Feature set to use ("baseline", "ridge_conservative", "elasticnet_rich")

    Returns:
        Dict mapping bracket_type -> (X, y, groups, metadata)
        with keys: "greater", "less", "between"
    """
    datasets = {}

    for bracket_type in ["greater", "less", "between"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading {city} {bracket_type} dataset")
        logger.info(f"{'='*60}")

        X, y, groups, metadata = build_training_dataset(
            city=city,
            start_date=start_date,
            end_date=end_date,
            bracket_type=bracket_type,
            feature_set=feature_set,
        )

        datasets[bracket_type] = (X, y, groups, metadata)

    return datasets


def main():
    """Demo: Build training dataset for Chicago."""
    from datetime import date

    print("\n" + "="*60)
    print("Dataset Builder Demo")
    print("="*60 + "\n")

    # Load Chicago data for last 42 days (one training window)
    end_date = date(2025, 11, 10)  # Recent date with settled markets
    start_date = date(2025, 9, 29)  # 42 days before

    print(f"Building dataset: {start_date} to {end_date}")

    datasets = load_training_datasets_for_city(
        city="chicago",
        start_date=start_date,
        end_date=end_date,
    )

    # Print summaries
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60 + "\n")

    for bracket_type, (X, y, groups, metadata) in datasets.items():
        if len(X) == 0:
            print(f"{bracket_type}: No data")
            continue

        print(f"\n{bracket_type}:")
        print(f"  Rows: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Days: {len(np.unique(groups))}")
        print(f"  Label distribution: YES={np.sum(y)}/{len(y)} ({100*np.sum(y)/len(y):.1f}%)")

        # Show feature stats
        print(f"  Feature ranges:")
        for i, col in enumerate(["yes_mid", "yes_bid", "yes_ask", "spread_cents",
                                  "minutes_to_close", "temp_now", "temp_to_floor",
                                  "temp_to_cap", "hour_of_day_local", "day_of_week"]):
            col_data = X[:, i]
            print(f"    {col}: [{np.min(col_data):.1f}, {np.max(col_data):.1f}]")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
