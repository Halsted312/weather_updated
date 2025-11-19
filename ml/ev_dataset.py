#!/usr/bin/env python3
"""
Expected-value dataset builder for minute-level price deltas.

For each 1-minute candle we compute the future mid-price horizon minutes ahead
and label the row with the delta (future_mid - current_mid) in cents. This is
used for near-term trading decisions rather than end-of-day settlement.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional, Literal, Tuple

import numpy as np
import pandas as pd

from ml.dataset import (
    CITY_CONFIG,
    load_candles_with_weather_and_metadata,
    build_training_dataset,
)

logger = logging.getLogger(__name__)


def build_ev_dataset(
    city: str,
    start_date: date,
    end_date: date,
    bracket_type: Literal["greater", "less", "between"],
    horizon_minutes: int = 60,
    feature_set: str = "nextgen",
    max_minutes_to_close: Optional[float] = None,
    prior_peak_back_minutes: Optional[float] = None,
    prior_peak_lookup_days: int = 3,
    prior_peak_default_minutes: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """
    Build dataset for predicting near-term price deltas.

    Args:
        city: City name
        start_date: Inclusive start date
        end_date: Inclusive end date
        bracket_type: Market bracket type
        horizon_minutes: Future horizon in minutes (default 60)
        feature_set: Feature set name for FeatureBuilder
        max_minutes_to_close: Optional minutes_to_close filter
        prior_peak_back_minutes: Optional start window based on prior-day peak
        prior_peak_lookup_days: Days to search for prior peak
        prior_peak_default_minutes: Default start minute if no peak found

    Returns:
        (X, y, groups, metadata, feature_cols)
    """
    X, _, groups, metadata, feature_cols = build_training_dataset(
        city=city,
        start_date=start_date,
        end_date=end_date,
        bracket_type=bracket_type,
        feature_set=feature_set,
        max_minutes_to_close=max_minutes_to_close,
        prior_peak_back_minutes=prior_peak_back_minutes,
        prior_peak_lookup_days=prior_peak_lookup_days,
        prior_peak_default_minutes=prior_peak_default_minutes,
        persist_feature_snapshots=False,
        split="train",
        return_feature_names=True,
    )

    if len(X) == 0:
        logger.warning("EV dataset empty for %s %s", city, bracket_type)
        return X, np.array([]), np.array([]), metadata, feature_cols

    features_df = pd.DataFrame(X, columns=feature_cols)
    features_df["market_ticker"] = metadata["market_ticker"].values
    features_df["timestamp"] = pd.to_datetime(metadata["timestamp"])

    if "yes_mid" not in features_df.columns:
        raise ValueError("Feature set must include yes_mid to compute price delta")

    features_df = features_df.sort_values(["market_ticker", "timestamp"])
    target_future = (
        features_df.groupby("market_ticker")["yes_mid"]
        .shift(-horizon_minutes)
        .astype(float)
    )
    delta = target_future - features_df["yes_mid"]

    valid_mask = delta.notna()
    features_df = features_df[valid_mask]
    delta = delta[valid_mask]
    metadata = metadata.loc[valid_mask].reset_index(drop=True)
    groups = groups[valid_mask.to_numpy()]
    X = features_df[feature_cols].to_numpy()

    metadata = metadata.copy()
    metadata["future_mid_cents"] = target_future[valid_mask].values
    metadata["current_mid_cents"] = features_df["yes_mid"].values

    return X, delta.to_numpy(), groups, metadata, feature_cols


__all__ = ["build_ev_dataset"]
