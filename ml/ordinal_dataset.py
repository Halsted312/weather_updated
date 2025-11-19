#!/usr/bin/env python3
"""
Ordinal dataset utilities for unified multi-bracket modeling.

Builds feature frames that are independent of individual markets so that a
single model can emit a coherent temperature distribution across brackets.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dataset import CITY_CONFIG, load_candles_with_weather_and_metadata
from ml.features import FeatureBuilder

logger = logging.getLogger(__name__)

# Ordinal model focuses on weather/time context rather than bracket-specific offsets
ORDINAL_FEATURE_COLUMNS = [
    "temp_now",
    "temp_delta_1h",
    "temp_rollmax_3h",
    "temp_rollmin_3h",
    "dewpoint",
    "humidity",
    "windspeed",
    "minutes_to_close",
    "log_minutes_to_close",
    "hour_of_day_local",
    "day_of_week",
    "minutes_since_midnight_local",
    "doy_sin",
    "doy_cos",
    "is_weekend",
]


@dataclass
class OrdinalDataset:
    features: pd.DataFrame
    feature_cols: List[str]
    target: np.ndarray
    bin_indices: np.ndarray
    metadata: pd.DataFrame

    @property
    def taus(self) -> List[float]:
        """Return default threshold values between ordinal bins."""
        max_bin = int(self.bin_indices.max()) if len(self.bin_indices) else -1
        return [i + 0.5 for i in range(max_bin)]  # e.g., 5 bins -> [0.5, 1.5, ...]


def _bracket_key(row: pd.Series) -> str:
    if row["strike_type"] == "less":
        return f"less_{row['cap_strike']}"
    if row["strike_type"] == "greater":
        return f"greater_{row['floor_strike']}"
    if row["strike_type"] == "between":
        return f"between_{row['floor_strike']}_{row['cap_strike']}"
    return f"unknown_{row['strike_type']}"


def _sort_value(row: pd.Series) -> float:
    """Deterministic ordering of brackets regardless of strike mix."""
    if row["strike_type"] == "less":
        cap = row.get("cap_strike")
        cap_val = float(cap) if pd.notna(cap) else -999.0
        return cap_val - 1000.0
    if row["strike_type"] == "between":
        floor = row.get("floor_strike")
        return float(floor) if pd.notna(floor) else 0.0
    if row["strike_type"] == "greater":
        floor = row.get("floor_strike")
        floor_val = float(floor) if pd.notna(floor) else 999.0
        return floor_val + 1000.0
    return 0.0


def _assign_bin_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Assign ordinal bin index per (event_date, bracket)."""
    mapping: Dict[Tuple[pd.Timestamp, str], int] = {}

    for event_date, group in df.groupby("event_date"):
        unique = group[["bracket_key", "strike_type", "floor_strike", "cap_strike"]].drop_duplicates("bracket_key")
        unique = unique.copy()
        unique["sort_value"] = unique.apply(_sort_value, axis=1)
        unique = unique.sort_values("sort_value").reset_index(drop=True)
        for idx, row in unique.iterrows():
            mapping[(event_date, row["bracket_key"])] = idx

    df["ordinal_bin"] = df.apply(
        lambda row: mapping.get((row["event_date"], row["bracket_key"])),
        axis=1,
    )
    df = df.dropna(subset=["ordinal_bin"])
    df["ordinal_bin"] = df["ordinal_bin"].astype(int)
    return df


def build_ordinal_dataset(
    city: str,
    start_date: date,
    end_date: date,
    feature_set: str = "nextgen",
    minutes_to_close_cutoff: int = 180,
) -> OrdinalDataset:
    """
    Build dataframe suitable for ordinal unified modeling.

    Returns snapshots for all brackets but uses only strike-agnostic
    features so the model can produce a single distribution per timestamp.
    """
    if city not in CITY_CONFIG:
        raise ValueError(f"Unknown city: {city}")

    raw = load_candles_with_weather_and_metadata(
        city=city,
        start_date=start_date,
        end_date=end_date,
        bracket_type=None,
    )
    if raw.empty:
        logger.warning("No raw data for %s between %s and %s", city, start_date, end_date)
        return OrdinalDataset(
            features=pd.DataFrame(columns=ORDINAL_FEATURE_COLUMNS),
            feature_cols=ORDINAL_FEATURE_COLUMNS,
            target=np.array([]),
            bin_indices=np.array([]),
            metadata=pd.DataFrame(),
        )

    fb = FeatureBuilder(city_timezone=CITY_CONFIG[city]["timezone"])
    features_df = fb.build_features(
        raw[[
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
        ]],
        weather_df=None,
        market_metadata=raw[[
            "market_ticker",
            "close_time",
            "strike_type",
            "floor_strike",
            "cap_strike",
        ]].drop_duplicates("market_ticker"),
        feature_set=feature_set,
    )

    if "timestamp" in features_df.columns:
        features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], utc=True).dt.tz_convert(None)
    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(None)

    features_df = features_df.merge(
        raw[[
            "market_ticker",
            "timestamp",
            "settlement_value",
            "strike_type",
            "floor_strike",
            "cap_strike",
        ]],
        on=["market_ticker", "timestamp"],
        how="left",
        suffixes=("", "_raw"),
    )

    if "event_date" not in features_df.columns and "timestamp_local" in features_df.columns:
        features_df["event_date"] = features_df["timestamp_local"].dt.date

    features_df["bracket_key"] = features_df.apply(_bracket_key, axis=1)
    features_df = features_df.dropna(subset=["event_date"])

    # Determine winning bracket per day using settlement value
    winners = (
        features_df[["event_date", "bracket_key", "settlement_value"]]
        .drop_duplicates(["event_date", "bracket_key"])
    )
    winners = winners[winners["settlement_value"] == 100.0]
    winners = winners.rename(columns={"bracket_key": "winner_bracket"})

    features_df = features_df.merge(winners[["event_date", "winner_bracket"]], on="event_date", how="left")
    features_df = features_df.dropna(subset=["winner_bracket"])

    features_df = _assign_bin_indices(features_df)
    features_df["ordinal_target"] = features_df["ordinal_bin"].astype(float)

    if minutes_to_close_cutoff is not None and "minutes_to_close" in features_df.columns:
        features_df = features_df[features_df["minutes_to_close"] <= minutes_to_close_cutoff]

    # Keep strike-agnostic feature columns
    feature_cols = [col for col in ORDINAL_FEATURE_COLUMNS if col in features_df.columns]
    missing = set(ORDINAL_FEATURE_COLUMNS) - set(feature_cols)
    if missing:
        logger.warning("Ordinal feature columns missing for %s: %s", city, sorted(missing))

    features_only = features_df[feature_cols].fillna(0.0)

    metadata_cols = [
        "market_ticker",
        "timestamp",
        "event_date",
        "bracket_key",
        "strike_type",
        "floor_strike",
        "cap_strike",
        "settlement_value",
        "ordinal_bin",
    ]
    metadata_cols = [col for col in metadata_cols if col in features_df.columns]
    metadata = features_df[metadata_cols].copy()

    return OrdinalDataset(
        features=features_only.reset_index(drop=True),
        feature_cols=feature_cols,
        target=features_df["ordinal_target"].to_numpy(),
        bin_indices=features_df["ordinal_bin"].to_numpy(),
        metadata=metadata.reset_index(drop=True),
    )
