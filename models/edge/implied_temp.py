"""
Temperature inference from probabilities and market prices.

This module computes "implied temperatures" - the temperature that is implied by:
1. Forecast model: ordinal delta probabilities → expected settlement temperature
2. Market prices: Kalshi bracket prices → market's implied temperature

These are compared to detect trading edges.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ForecastImpliedResult:
    """Result from forecast-implied temperature calculation.

    Attributes:
        implied_temp: Expected settlement temperature (°F)
        uncertainty: Standard deviation of temperature distribution (°F)
        predicted_delta: Most likely delta class (mode)
        valid: Whether the result is valid (always True for forecast)
    """
    implied_temp: float
    uncertainty: float
    predicted_delta: int
    valid: bool = True


@dataclass
class MarketImpliedResult:
    """Result from market-implied temperature calculation.

    Attributes:
        implied_temp: Market's implied temperature (°F)
        uncertainty: Market uncertainty from bid/ask spreads (°F)
        valid: Whether result is valid (False if no market data)
    """
    implied_temp: float
    uncertainty: float
    valid: bool = True


def compute_forecast_implied_temp(
    delta_probs: Dict[int, float],
    base_temp: float,
    delta_classes: Optional[list] = None,
) -> ForecastImpliedResult:
    """Compute forecast-implied temperature from ordinal model delta probabilities.

    Args:
        delta_probs: Dict mapping delta class → probability
                     e.g. {-2: 0.01, -1: 0.05, 0: 0.80, 1: 0.12, 2: 0.02}
        base_temp: Base temperature (typically round(max_temp_observed_so_far))
        delta_classes: List of all possible delta classes (default: [-10, ..., 10])

    Returns:
        ForecastImpliedResult with:
        - implied_temp: Expected settlement temperature = base_temp + E[delta]
        - uncertainty: Standard deviation of temperature distribution
        - predicted_delta: Most likely delta class (mode)
        - valid: Always True

    Example:
        >>> delta_probs = {-1: 0.1, 0: 0.7, 1: 0.2}
        >>> result = compute_forecast_implied_temp(delta_probs, base_temp=85.0)
        >>> result.implied_temp  # 85.1°F
        >>> result.uncertainty   # 0.5°F
    """
    if delta_classes is None:
        delta_classes = list(range(-10, 11))  # [-10, -9, ..., 9, 10]

    # Ensure probabilities sum to 1 (handle rounding errors)
    probs_array = np.array([delta_probs.get(d, 0.0) for d in delta_classes])
    probs_array = probs_array / probs_array.sum()  # Normalize

    # Compute expected value: E[delta] = sum(delta * P(delta))
    deltas_array = np.array(delta_classes)
    expected_delta = float(np.dot(deltas_array, probs_array))

    # Compute standard deviation: sqrt(E[delta^2] - E[delta]^2)
    variance = float(np.dot(deltas_array ** 2, probs_array)) - expected_delta ** 2
    std_delta = float(np.sqrt(max(0, variance)))

    # Find mode (most likely delta)
    mode_idx = np.argmax(probs_array)
    predicted_delta = int(delta_classes[mode_idx])

    # Implied temperature = base + expected delta
    implied_temp = float(base_temp + expected_delta)

    return ForecastImpliedResult(
        implied_temp=implied_temp,
        uncertainty=std_delta,
        predicted_delta=predicted_delta,
        valid=True,
    )


def compute_market_implied_temp(
    bracket_candles: Dict[str, pd.DataFrame],
    bracket_thresholds: Optional[Dict[str, float]] = None,
    snapshot_time=None,  # Optional, for compatibility with training script
) -> MarketImpliedResult:
    """Compute market-implied temperature from Kalshi bracket prices.

    Uses bid/ask midpoint prices to infer what temperature the market is pricing.
    Weighted average of bracket midpoints by market probability.

    Args:
        bracket_candles: Dict mapping bracket_label → DataFrame with candles
                        Each DataFrame should have columns:
                        - yes_bid_close, yes_ask_close (prices in cents, 0-100)
                        - bucket_start (timestamp)
        bracket_thresholds: Dict mapping bracket_label → temperature threshold
                           e.g. {"T_ge_85": 85.0, "T_ge_86": 86.0}
                           If None, will parse from bracket labels
        snapshot_time: Optional timestamp (for filtering candles, currently unused)

    Returns:
        MarketImpliedResult with:
        - implied_temp: Market's implied expected temperature
        - uncertainty: Spread-based uncertainty measure
        - valid: False if no market data available

    Example:
        >>> candles = {
        ...     "T_ge_85": df_with_bid_ask,  # Price = 70 cents
        ...     "T_ge_86": df_with_bid_ask,  # Price = 40 cents
        ... }
        >>> result = compute_market_implied_temp(candles)
        >>> result.implied_temp  # 85.4°F
        >>> result.uncertainty   # 0.3°F
    """
    if not bracket_candles:
        # No market data available
        return MarketImpliedResult(
            implied_temp=np.nan,
            uncertainty=np.nan,
            valid=False,
        )

    # Parse bracket thresholds from labels if not provided
    if bracket_thresholds is None:
        bracket_thresholds = {}
        for label in bracket_candles.keys():
            # Expected format: "T_ge_85" or "85-86" or similar
            if "T_ge_" in label:
                try:
                    thresh = float(label.split("T_ge_")[1])
                    bracket_thresholds[label] = thresh
                except (IndexError, ValueError):
                    logger.warning(f"Could not parse threshold from label: {label}")
            elif "-" in label:
                try:
                    parts = label.split("-")
                    thresh = float(parts[0])
                    bracket_thresholds[label] = thresh
                except (IndexError, ValueError):
                    logger.warning(f"Could not parse threshold from label: {label}")

    if not bracket_thresholds:
        logger.warning("Could not determine bracket thresholds")
        return MarketImpliedResult(
            implied_temp=np.nan,
            uncertainty=np.nan,
            valid=False,
        )

    # Collect latest prices and thresholds
    prices = []
    thresholds = []
    spreads = []

    for label, df in bracket_candles.items():
        if df.empty:
            continue

        thresh = bracket_thresholds.get(label)
        if thresh is None:
            continue

        # Get latest candle
        latest = df.iloc[-1]

        # Compute midpoint price (in probability units, 0-1)
        bid = latest.get("yes_bid_close", np.nan)
        ask = latest.get("yes_ask_close", np.nan)

        if pd.isna(bid) or pd.isna(ask):
            continue

        # Convert cents to probability
        bid_prob = bid / 100.0
        ask_prob = ask / 100.0
        mid_prob = (bid_prob + ask_prob) / 2.0

        prices.append(mid_prob)
        thresholds.append(thresh)
        spreads.append(ask_prob - bid_prob)

    if not prices:
        logger.warning("No valid bracket prices found")
        return MarketImpliedResult(
            implied_temp=np.nan,
            uncertainty=np.nan,
            valid=False,
        )

    prices = np.array(prices)
    thresholds = np.array(thresholds)
    spreads = np.array(spreads)

    # Sort by threshold (ascending)
    sort_idx = np.argsort(thresholds)
    prices = prices[sort_idx]
    thresholds = thresholds[sort_idx]
    spreads = spreads[sort_idx]

    # Compute cumulative probabilities P(T >= threshold)
    # These should be monotonically decreasing, but market prices may violate this
    # For simplicity, use prices as-is (could add isotonic regression here)

    # Compute bracket probabilities: P(threshold[i] <= T < threshold[i+1])
    bracket_probs = []
    for i in range(len(prices) - 1):
        # P(T in [thresh[i], thresh[i+1])) = P(T >= thresh[i]) - P(T >= thresh[i+1])
        prob = prices[i] - prices[i + 1]
        bracket_probs.append(max(0, prob))  # Ensure non-negative

    # Add tails
    # P(T < thresh[0]) = 1 - P(T >= thresh[0])
    prob_below = 1.0 - prices[0]
    bracket_probs.insert(0, max(0, prob_below))

    # P(T >= thresh[-1])
    prob_above = prices[-1]
    bracket_probs.append(max(0, prob_above))

    # Normalize probabilities
    bracket_probs = np.array(bracket_probs)
    bracket_probs = bracket_probs / bracket_probs.sum()

    # Compute bracket midpoints for expected value calculation
    bracket_midpoints = []
    # Below first threshold
    bracket_midpoints.append(thresholds[0] - 1.0)
    # Between thresholds
    for i in range(len(thresholds) - 1):
        midpoint = (thresholds[i] + thresholds[i + 1]) / 2.0
        bracket_midpoints.append(midpoint)
    # Above last threshold
    bracket_midpoints.append(thresholds[-1] + 1.0)

    bracket_midpoints = np.array(bracket_midpoints)

    # Compute expected temperature
    implied_temp = float(np.dot(bracket_midpoints, bracket_probs))

    # Compute uncertainty from spreads (average spread as proxy for uncertainty)
    avg_spread = float(spreads.mean())
    uncertainty = avg_spread * 2.0  # Rough conversion: spread → temp uncertainty

    return MarketImpliedResult(
        implied_temp=implied_temp,
        uncertainty=uncertainty,
        valid=True,
    )


def compare_implied_temps(
    forecast_temp: float,
    market_temp: float,
    forecast_uncertainty: float,
    market_uncertainty: float,
) -> Dict[str, float]:
    """Compare forecast-implied and market-implied temperatures.

    Args:
        forecast_temp: Forecast model's implied temperature
        market_temp: Market's implied temperature
        forecast_uncertainty: Forecast uncertainty (std dev)
        market_uncertainty: Market uncertainty (spread-based)

    Returns:
        Dict with comparison metrics:
        - edge: forecast_temp - market_temp (in °F)
        - abs_edge: |edge|
        - edge_in_stds: edge / sqrt(forecast_unc^2 + market_unc^2)
        - confidence: Bayesian confidence in edge direction

    Example:
        >>> compare_implied_temps(86.0, 84.5, 0.5, 0.3)
        {'edge': 1.5, 'abs_edge': 1.5, 'edge_in_stds': 2.56, 'confidence': 0.99}
    """
    edge = forecast_temp - market_temp
    abs_edge = abs(edge)

    # Combined uncertainty (assuming independent)
    combined_unc = np.sqrt(forecast_uncertainty ** 2 + market_uncertainty ** 2)

    # Edge in standard deviations
    if combined_unc > 0:
        edge_in_stds = edge / combined_unc
    else:
        edge_in_stds = 0.0

    # Confidence: P(edge is real) using normal approximation
    # confidence = 1 - P(|Z| < edge_in_stds) where Z ~ N(0,1)
    from scipy.stats import norm

    if combined_unc > 0:
        confidence = 1.0 - 2.0 * norm.cdf(-abs_edge / combined_unc)
    else:
        confidence = 0.5  # No information

    return {
        "edge": float(edge),
        "abs_edge": float(abs_edge),
        "edge_in_stds": float(edge_in_stds),
        "confidence": float(confidence),
    }
