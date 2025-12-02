"""
Market-derived features from Kalshi candle data.

This module computes features from dense 1-minute candles, capturing
market sentiment, implied probabilities, and trading momentum.

These features are unique alpha - they represent what the market
"thinks" about the final settlement, independent of weather data.

Features computed:
    Price/implied features:
        market_yes_bid: Current yes bid price (0-99)
        market_yes_ask: Current yes ask price (1-100)
        market_bid_ask_spread: Spread (uncertainty proxy)
        market_mid_price: Midpoint price

    Momentum features:
        bid_change_last_30min: Change in yes_bid over 30 min
        bid_change_last_60min: Change in yes_bid over 60 min
        bid_momentum_30min: Rate of bid change (per hour)

    Volume features:
        volume_last_30min: Trading volume in last 30 min
        volume_last_60min: Trading volume in last 60 min
        cumulative_volume_today: Total volume since market open

    Liquidity/activity:
        has_recent_trade: 1 if trade in last 30 min
        open_interest: Current open interest

Example:
    >>> candles = load_dense_candles(ticker, from_time, to_time)
    >>> fs = compute_market_features(candles)
    >>> fs['market_bid_ask_spread']
    3.0
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from models.features.base import FeatureSet, register_feature_group


@register_feature_group("market")
def compute_market_features(
    candles_df: Optional[pd.DataFrame],
    snapshot_time: Optional[datetime] = None,
) -> FeatureSet:
    """Compute features from Kalshi dense candles.

    Args:
        candles_df: DataFrame with dense candle data, columns:
                    bucket_start, yes_bid_close, yes_ask_close,
                    trade_close, volume, open_interest, has_trade
                    Sorted by bucket_start ascending.
        snapshot_time: Cutoff time (use candles up to this time).
                      If None, uses all candles.

    Returns:
        FeatureSet with market-derived features
    """
    null_features = {
        "market_yes_bid": None,
        "market_yes_ask": None,
        "market_bid_ask_spread": None,
        "market_mid_price": None,
        "bid_change_last_30min": None,
        "bid_change_last_60min": None,
        "bid_momentum_30min": None,
        "volume_last_30min": None,
        "volume_last_60min": None,
        "cumulative_volume_today": None,
        "has_recent_trade": None,
        "open_interest": None,
    }

    if candles_df is None or candles_df.empty:
        return FeatureSet(name="market", features=null_features)

    df = candles_df.copy()

    # Ensure bucket_start is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["bucket_start"]):
        df["bucket_start"] = pd.to_datetime(df["bucket_start"])

    # Filter to snapshot time if provided
    if snapshot_time is not None:
        # Handle timezone comparison carefully:
        # - bucket_start from DB is typically timezone-aware (UTC)
        # - snapshot_time from training may be naive (local time)
        if df["bucket_start"].dt.tz is not None:
            # Candles are tz-aware (UTC)
            if isinstance(snapshot_time, datetime) and snapshot_time.tzinfo is None:
                # Snapshot is naive - remove timezone from candles for comparison
                df["bucket_start"] = df["bucket_start"].dt.tz_localize(None)
        else:
            # Candles are naive
            if isinstance(snapshot_time, datetime) and snapshot_time.tzinfo is not None:
                df["bucket_start"] = df["bucket_start"].dt.tz_localize("UTC")
        df = df[df["bucket_start"] <= snapshot_time]

    if df.empty:
        return FeatureSet(name="market", features=null_features)

    # Sort by time
    df = df.sort_values("bucket_start")

    features = {}

    # Current prices (most recent candle)
    last_row = df.iloc[-1]

    yes_bid = last_row.get("yes_bid_close")
    yes_ask = last_row.get("yes_ask_close")

    features["market_yes_bid"] = float(yes_bid) if pd.notna(yes_bid) else None
    features["market_yes_ask"] = float(yes_ask) if pd.notna(yes_ask) else None

    if pd.notna(yes_bid) and pd.notna(yes_ask):
        features["market_bid_ask_spread"] = float(yes_ask - yes_bid)
        features["market_mid_price"] = float((yes_bid + yes_ask) / 2)
    else:
        features["market_bid_ask_spread"] = None
        features["market_mid_price"] = None

    # Open interest
    oi = last_row.get("open_interest")
    features["open_interest"] = float(oi) if pd.notna(oi) else None

    # Time-based features
    last_time = df["bucket_start"].iloc[-1]

    # Price momentum (change in bid)
    for window in [30, 60]:
        cutoff = last_time - timedelta(minutes=window)
        window_df = df[df["bucket_start"] >= cutoff]

        if len(window_df) >= 2:
            first_bid = window_df["yes_bid_close"].iloc[0]
            last_bid = window_df["yes_bid_close"].iloc[-1]

            if pd.notna(first_bid) and pd.notna(last_bid):
                features[f"bid_change_last_{window}min"] = float(last_bid - first_bid)
            else:
                features[f"bid_change_last_{window}min"] = None
        else:
            features[f"bid_change_last_{window}min"] = None

    # Bid momentum (rate per hour)
    if features.get("bid_change_last_30min") is not None:
        features["bid_momentum_30min"] = features["bid_change_last_30min"] * 2  # Convert to per hour
    else:
        features["bid_momentum_30min"] = None

    # Volume features
    for window in [30, 60]:
        cutoff = last_time - timedelta(minutes=window)
        window_df = df[df["bucket_start"] >= cutoff]

        if "volume" in window_df.columns:
            vol = window_df["volume"].sum()
            features[f"volume_last_{window}min"] = float(vol) if pd.notna(vol) else 0.0
        else:
            features[f"volume_last_{window}min"] = 0.0

    # Cumulative volume
    if "volume" in df.columns:
        features["cumulative_volume_today"] = float(df["volume"].sum())
    else:
        features["cumulative_volume_today"] = 0.0

    # Recent trade flag
    cutoff_30 = last_time - timedelta(minutes=30)
    recent_df = df[df["bucket_start"] >= cutoff_30]

    if "has_trade" in recent_df.columns:
        features["has_recent_trade"] = 1 if recent_df["has_trade"].any() else 0
    elif "volume" in recent_df.columns:
        features["has_recent_trade"] = 1 if recent_df["volume"].sum() > 0 else 0
    else:
        features["has_recent_trade"] = None

    return FeatureSet(name="market", features=features)


def compute_market_bracket_features(
    bracket_candles: dict[str, pd.DataFrame],
    snapshot_time: Optional[datetime] = None,
) -> FeatureSet:
    """Compute features across multiple bracket markets.

    When we have candles for multiple brackets (e.g., 85-86, 86-87, 87-88),
    we can compute features about the probability distribution implied
    by the market.

    Args:
        bracket_candles: Dict mapping bracket label to candle DataFrame
        snapshot_time: Cutoff time for candles

    Returns:
        FeatureSet with cross-bracket features
    """
    null_features = {
        "market_implied_bracket_idx": None,
        "market_prob_entropy": None,
        "market_max_prob": None,
        "market_prob_above_current": None,
    }

    if not bracket_candles:
        return FeatureSet(name="market_brackets", features=null_features)

    # Get latest yes_bid for each bracket
    bracket_probs = {}
    for bracket_label, df in bracket_candles.items():
        if df is None or df.empty:
            continue

        if snapshot_time is not None:
            df = df[df["bucket_start"] <= snapshot_time]

        if df.empty:
            continue

        last_bid = df.sort_values("bucket_start")["yes_bid_close"].iloc[-1]
        if pd.notna(last_bid):
            # Convert cents to probability (0-100 -> 0-1)
            bracket_probs[bracket_label] = last_bid / 100.0

    if not bracket_probs:
        return FeatureSet(name="market_brackets", features=null_features)

    probs = np.array(list(bracket_probs.values()))
    labels = list(bracket_probs.keys())

    # Normalize to sum to 1 (they may not due to bid prices)
    probs_norm = probs / probs.sum() if probs.sum() > 0 else probs

    features = {}

    # Index of highest probability bracket
    max_idx = int(np.argmax(probs_norm))
    features["market_implied_bracket_idx"] = max_idx
    features["market_max_prob"] = float(probs_norm.max())

    # Entropy (uncertainty measure)
    # Higher entropy = more uncertain, lower = more confident
    entropy = -np.sum(probs_norm * np.log(probs_norm + 1e-10))
    features["market_prob_entropy"] = float(entropy)

    # Probability mass above median bracket (upside bias)
    mid_idx = len(probs_norm) // 2
    features["market_prob_above_current"] = float(probs_norm[mid_idx:].sum())

    return FeatureSet(name="market_brackets", features=features)
