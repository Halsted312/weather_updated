"""
Probability conversion utilities for temperature Δ-models.

This module translates model output (Δ probabilities) into actionable
trading signals (bracket probabilities) for Kalshi markets.

Key conversions:
    Δ probs → Temperature probs: P(Δ=d) → P(T=t) for all temps
    Temperature probs → Bracket probs: P(T=t) → P(T >= K) or P(a <= T < b)

Example:
    >>> delta_probs = {-2: 0.01, -1: 0.05, 0: 0.80, 1: 0.12, 2: 0.02}
    >>> t_base = 92
    >>> bracket_probs = compute_bracket_probabilities(delta_probs, t_base)
    >>> print(bracket_probs['T_ge_90'])  # P(T >= 90) = P(Δ >= -2)
    0.99
"""

import logging
from typing import Optional

import numpy as np

from models.features.base import DELTA_CLASSES

logger = logging.getLogger(__name__)


def delta_probs_to_dict(
    proba: np.ndarray,
    delta_classes: Optional[list[int]] = None,
) -> dict[int, float]:
    """Convert probability array to dict mapping Δ to probability.

    Args:
        proba: Array of probabilities (length = n_classes)
        delta_classes: List of Δ class values (default: DELTA_CLASSES)

    Returns:
        Dict mapping each Δ value to its probability
    """
    if delta_classes is None:
        delta_classes = DELTA_CLASSES

    return {d: float(p) for d, p in zip(delta_classes, proba)}


def delta_probs_to_temp_probs(
    delta_probs: dict[int, float],
    t_base: int,
) -> dict[int, float]:
    """Convert P(Δ=d) to P(T=t) for each temperature.

    Since T = t_base + Δ, we can map each Δ probability to a
    temperature probability.

    Args:
        delta_probs: Dict mapping Δ to probability
        t_base: Baseline temperature (rounded partial-day max)

    Returns:
        Dict mapping each temperature to its probability

    Example:
        >>> delta_probs = {-1: 0.1, 0: 0.8, 1: 0.1}
        >>> t_base = 90
        >>> delta_probs_to_temp_probs(delta_probs, t_base)
        {89: 0.1, 90: 0.8, 91: 0.1}
    """
    temp_probs = {}
    for delta, prob in delta_probs.items():
        temp = t_base + delta
        temp_probs[temp] = prob
    return temp_probs


def temp_probs_to_bracket_prob(
    temp_probs: dict[int, float],
    bracket_floor: int,
    bracket_cap: Optional[int] = None,
) -> float:
    """Compute P(floor <= T < cap) from temperature distribution.

    If bracket_cap is None, computes P(T >= floor).

    Args:
        temp_probs: Dict mapping temperature to probability
        bracket_floor: Lower bound (inclusive)
        bracket_cap: Upper bound (exclusive), or None for open-ended

    Returns:
        Probability that temperature falls in the bracket

    Example:
        >>> temp_probs = {88: 0.1, 89: 0.2, 90: 0.3, 91: 0.3, 92: 0.1}
        >>> temp_probs_to_bracket_prob(temp_probs, 90, None)  # P(T >= 90)
        0.7
        >>> temp_probs_to_bracket_prob(temp_probs, 90, 92)  # P(90 <= T < 92)
        0.6
    """
    prob = 0.0
    for temp, p in temp_probs.items():
        if temp >= bracket_floor:
            if bracket_cap is None or temp < bracket_cap:
                prob += p
    return prob


def compute_bracket_probabilities(
    delta_probs: dict[int, float],
    t_base: int,
    thresholds: list[int] = [80, 85, 90, 95],
) -> dict[str, float]:
    """Compute P(T >= K) for multiple thresholds.

    This is the main function for translating model output into
    trading signals for Kalshi high-temperature markets.

    Args:
        delta_probs: Dict mapping Δ to probability
        t_base: Baseline temperature
        thresholds: List of temperature thresholds

    Returns:
        Dict with keys like 'T_ge_90' and probability values

    Example:
        >>> delta_probs = {-2: 0.01, -1: 0.05, 0: 0.80, 1: 0.12, 2: 0.02}
        >>> t_base = 92
        >>> probs = compute_bracket_probabilities(delta_probs, t_base)
        >>> probs['T_ge_90']  # P(T >= 90) where t_base=92
        0.99  # All deltas >= -2 give T >= 90
    """
    temp_probs = delta_probs_to_temp_probs(delta_probs, t_base)

    bracket_probs = {}
    for threshold in thresholds:
        prob = temp_probs_to_bracket_prob(temp_probs, threshold, None)
        bracket_probs[f"T_ge_{threshold}"] = prob

    return bracket_probs


def compute_all_bracket_probs(
    delta_probs: dict[int, float],
    t_base: int,
    bracket_width: int = 5,
    min_temp: int = 60,
    max_temp: int = 110,
) -> dict[str, float]:
    """Compute probabilities for all standard Kalshi brackets.

    Kalshi high-temp markets typically have 5°F wide brackets like:
    T < 80, 80 <= T < 85, 85 <= T < 90, ..., T >= 105

    Args:
        delta_probs: Dict mapping Δ to probability
        t_base: Baseline temperature
        bracket_width: Width of each bracket (default 5°F)
        min_temp: Minimum temperature bracket
        max_temp: Maximum temperature for open-ended bracket

    Returns:
        Dict with bracket names and probabilities
    """
    temp_probs = delta_probs_to_temp_probs(delta_probs, t_base)

    bracket_probs = {}

    # Lower tail: T < min_temp
    bracket_probs[f"T_lt_{min_temp}"] = temp_probs_to_bracket_prob(
        temp_probs, -999, min_temp
    )

    # Middle brackets: [floor, floor+width)
    for floor in range(min_temp, max_temp, bracket_width):
        cap = floor + bracket_width
        prob = temp_probs_to_bracket_prob(temp_probs, floor, cap)
        bracket_probs[f"T_{floor}_{cap}"] = prob

    # Upper tail: T >= max_temp
    bracket_probs[f"T_ge_{max_temp}"] = temp_probs_to_bracket_prob(
        temp_probs, max_temp, None
    )

    return bracket_probs


def expected_settlement(
    delta_probs: dict[int, float],
    t_base: int,
) -> float:
    """Compute expected settlement temperature from Δ distribution.

    E[T] = t_base + E[Δ] = t_base + Σ(d * P(Δ=d))

    Args:
        delta_probs: Dict mapping Δ to probability
        t_base: Baseline temperature

    Returns:
        Expected settlement temperature
    """
    expected_delta = sum(d * p for d, p in delta_probs.items())
    return t_base + expected_delta


def settlement_std(delta_probs: dict[int, float]) -> float:
    """Compute standard deviation of settlement prediction.

    Useful for sizing positions - higher std = less confidence.

    Args:
        delta_probs: Dict mapping Δ to probability

    Returns:
        Standard deviation of Δ distribution
    """
    expected_delta = sum(d * p for d, p in delta_probs.items())
    variance = sum((d - expected_delta) ** 2 * p for d, p in delta_probs.items())
    return np.sqrt(variance)


def confidence_interval(
    delta_probs: dict[int, float],
    t_base: int,
    level: float = 0.9,
) -> tuple[int, int]:
    """Compute confidence interval for settlement temperature.

    Finds the smallest interval [low, high] containing at least
    `level` probability mass.

    Args:
        delta_probs: Dict mapping Δ to probability
        t_base: Baseline temperature
        level: Confidence level (default 0.9 = 90%)

    Returns:
        Tuple of (low_temp, high_temp)
    """
    # Sort deltas by value
    sorted_deltas = sorted(delta_probs.keys())
    probs = [delta_probs[d] for d in sorted_deltas]

    # Find shortest interval containing `level` probability
    n = len(sorted_deltas)
    best_interval = (sorted_deltas[0], sorted_deltas[-1])
    best_width = best_interval[1] - best_interval[0]

    for i in range(n):
        cumsum = 0.0
        for j in range(i, n):
            cumsum += probs[j]
            if cumsum >= level:
                width = sorted_deltas[j] - sorted_deltas[i]
                if width < best_width:
                    best_width = width
                    best_interval = (sorted_deltas[i], sorted_deltas[j])
                break

    return (t_base + best_interval[0], t_base + best_interval[1])


def compare_to_market_prob(
    model_prob: float,
    market_price_cents: int,
) -> dict:
    """Compare model probability to Kalshi market price.

    Kalshi prices are in cents (0-100), representing the market's
    implied probability for the event.

    Args:
        model_prob: Model's probability for the event
        market_price_cents: Kalshi price in cents (0-100)

    Returns:
        Dict with edge calculation and signal
    """
    market_prob = market_price_cents / 100.0

    edge = model_prob - market_prob
    edge_cents = edge * 100

    # Determine signal
    if edge > 0.05:  # 5% edge threshold
        signal = "BUY"
    elif edge < -0.05:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "model_prob": model_prob,
        "market_prob": market_prob,
        "edge": edge,
        "edge_cents": edge_cents,
        "signal": signal,
    }
