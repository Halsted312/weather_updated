"""
Kalshi 1-minute candle microstructure features.

Computes 8 aggregated features over last 15 fully-closed minutes:
- Logit-based price features (dimensionless, information scale)
- Spread and volatility (liquidity proxies)
- Activity indicators (trade/synthetic fractions)

CRITICAL LEAKAGE PREVENTION:
- bucket_start is START of minute bucket (computed from end_period_ts - 60)
- At snapshot_time T, candle with bucket_start=T is NOT yet closed
- Use effective = floor(snapshot_time to minute)
- Filter: bucket_start < effective (strict inequality)
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from models.features.base import FeatureSet

import logging
logger = logging.getLogger(__name__)


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Ensure timestamp is UTC-aware."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute logit transform: log(p / (1-p))."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _window(df: pd.DataFrame, end: pd.Timestamp, minutes: int) -> pd.DataFrame:
    """Extract candles in [end - minutes, end) interval."""
    start = end - pd.Timedelta(minutes=minutes)
    return df[(df["bucket_start"] >= start) & (df["bucket_start"] < end)]


def compute_candles_micro_features(
    candles_df: Optional[pd.DataFrame],
    snapshot_time: Optional[pd.Timestamp],
    window_minutes: int = 15,
) -> FeatureSet:
    """
    Compute microstructure features from 1-minute Kalshi candles.

    Args:
        candles_df: DataFrame with bucket_start, yes_bid_close, yes_ask_close, etc.
        snapshot_time: Snapshot timestamp (observations up to this time)
        window_minutes: Lookback window (default: 15 minutes)

    Returns:
        FeatureSet with 8 candle microstructure features

    Features:
        c_logit_mid_last: Logit of last mid-price (dimensionless)
        c_logit_mom_15m: Logit momentum over window (directional signal)
        c_logit_vol_15m: Std dev of logit returns (volatility)
        c_logit_surprise_15m: |momentum| / volatility (dimensionless)
        c_spread_pct_mean_15m: Mean bid-ask spread % (liquidity)
        c_mid_range_pct_15m: (max-min)/mean mid-price % (range)
        c_trade_frac_15m: Fraction of minutes with trades (activity)
        c_synth_frac_15m: Fraction of synthetic candles (data quality)
    """
    feature_names = [
        "c_logit_mid_last",
        "c_logit_mom_15m",
        "c_logit_vol_15m",
        "c_logit_surprise_15m",
        "c_spread_pct_mean_15m",
        "c_mid_range_pct_15m",
        "c_trade_frac_15m",
        "c_synth_frac_15m",
    ]
    none_out = {k: None for k in feature_names}

    if candles_df is None or len(candles_df) == 0 or snapshot_time is None:
        logger.debug(f"candles_micro: no data (candles={candles_df is not None if candles_df is not None else 'None'}, len={len(candles_df) if candles_df is not None else 0}, snapshot={snapshot_time is not None})")
        return FeatureSet(name="candles_micro", features=none_out)

    logger.debug(f"candles_micro: processing {len(candles_df)} candles, snapshot={snapshot_time}")
    df = candles_df.copy()

    # Normalize timestamps to UTC
    df["bucket_start"] = pd.to_datetime(df["bucket_start"], utc=True, errors="coerce")
    df = df.dropna(subset=["bucket_start"]).sort_values("bucket_start")

    if df.empty:
        return FeatureSet(name="candles_micro", features=none_out)

    snap = _ensure_utc(snapshot_time)

    # CRITICAL: Use last fully-closed minute
    # bucket_start is START of 1-min bucket [bucket_start, bucket_start+60s)
    # At snapshot_time T, bucket with bucket_start=T is NOT closed yet
    effective = snap.floor("min")
    df = df[df["bucket_start"] < effective]

    if df.empty:
        return FeatureSet(name="candles_micro", features=none_out)

    # Extract window (last 15 fully-closed minutes)
    w = _window(df, end=effective, minutes=window_minutes)

    if len(w) < 3:  # Need at least 3 candles for meaningful stats
        logger.debug(f"candles_micro: insufficient candles in window ({len(w)} < 3), effective={effective}, window={window_minutes}min")
        return FeatureSet(name="candles_micro", features=none_out)

    # Extract price data (in cents)
    bid = w["yes_bid_close"].astype(float).to_numpy()
    ask = w["yes_ask_close"].astype(float).to_numpy()
    mid = 0.5 * (bid + ask)
    spread = ask - bid

    # Convert to probability and logit
    p = mid / 100.0  # Cents → probability
    lmid = _logit(p)

    # Logit returns (1-min changes)
    dlmid = np.diff(lmid)

    # Feature 1: Last logit mid-price
    c_logit_mid_last = float(lmid[-1])

    # Feature 2: Logit momentum (first → last)
    c_logit_mom_15m = float(lmid[-1] - lmid[0])

    # Feature 3: Logit volatility (std of 1-min returns)
    c_logit_vol_15m = float(np.std(dlmid, ddof=1)) if len(dlmid) >= 2 else 0.0

    # Feature 4: Logit surprise (|momentum| / volatility)
    c_logit_surprise_15m = float(abs(c_logit_mom_15m) / (c_logit_vol_15m + 1e-6))

    # Feature 5: Mean spread % (liquidity)
    mid_mean = float(np.mean(mid))
    c_spread_pct_mean_15m = float(np.mean(spread / np.maximum(mid, 1e-6)))

    # Feature 6: Mid-price range % (another volatility proxy)
    mid_range = np.max(mid) - np.min(mid)
    c_mid_range_pct_15m = float(mid_range / max(mid_mean, 1e-6))

    # Feature 7: Trade fraction (activity indicator)
    if "has_trade" in w.columns:
        c_trade_frac_15m = float(np.mean(w["has_trade"].astype(bool).to_numpy()))
    elif "volume" in w.columns:
        c_trade_frac_15m = float(np.mean((w["volume"].astype(float) > 0).to_numpy()))
    else:
        c_trade_frac_15m = None

    # Feature 8: Synthetic fraction (data quality indicator)
    if "is_synthetic" in w.columns:
        c_synth_frac_15m = float(np.mean(w["is_synthetic"].astype(bool).to_numpy()))
    else:
        c_synth_frac_15m = None

    features = {
        "c_logit_mid_last": c_logit_mid_last,
        "c_logit_mom_15m": c_logit_mom_15m,
        "c_logit_vol_15m": c_logit_vol_15m,
        "c_logit_surprise_15m": c_logit_surprise_15m,
        "c_spread_pct_mean_15m": c_spread_pct_mean_15m,
        "c_mid_range_pct_15m": c_mid_range_pct_15m,
        "c_trade_frac_15m": c_trade_frac_15m,
        "c_synth_frac_15m": c_synth_frac_15m,
    }

    return FeatureSet(name="candles_micro", features=features)
