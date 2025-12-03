"""
Edge detection logic for Kalshi weather trading.

This module identifies trading opportunities by comparing forecast-implied
and market-implied temperatures. An "edge" exists when the model's temperature
prediction differs significantly from what the market is pricing.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EdgeSignal(Enum):
    """Edge trading signal enum.

    Attributes:
        BUY_HIGH: Buy the "Yes" on T >= K bracket (forecast predicts higher than market)
        BUY_LOW: Buy the "Yes" on T < K bracket (forecast predicts lower than market)
        NO_TRADE: No significant edge detected
    """

    BUY_HIGH = "buy_high"
    BUY_LOW = "buy_low"
    NO_TRADE = "no_trade"

    def __str__(self):
        return self.value


@dataclass
class EdgeResult:
    """Result of edge detection.

    Attributes:
        signal: Trading signal (BUY_HIGH, BUY_LOW, or NO_TRADE)
        edge: Temperature edge in °F (forecast - market)
        abs_edge: Absolute magnitude of edge
        confidence: Confidence score (0-1) that edge is real
        forecast_temp: Forecast model's implied temperature
        market_temp: Market's implied temperature
        threshold_f: Edge detection threshold used (in °F)
    """

    signal: EdgeSignal
    edge: float
    abs_edge: float
    confidence: float
    forecast_temp: float
    market_temp: float
    threshold_f: float

    def __post_init__(self):
        """Validate edge result."""
        if not isinstance(self.signal, EdgeSignal):
            raise TypeError(f"signal must be EdgeSignal, got {type(self.signal)}")

        if not 0 <= self.confidence <= 1:
            logger.warning(
                f"Confidence {self.confidence:.2f} outside [0,1] range, clipping"
            )
            self.confidence = np.clip(self.confidence, 0.0, 1.0)

    def to_dict(self):
        """Convert to dictionary for storage/logging."""
        return {
            "signal": self.signal.value,
            "edge": float(self.edge),
            "abs_edge": float(self.abs_edge),
            "confidence": float(self.confidence),
            "forecast_temp": float(self.forecast_temp),
            "market_temp": float(self.market_temp),
            "threshold_f": float(self.threshold_f),
        }


def detect_edge(
    forecast_temp: float = None,
    market_temp: float = None,
    threshold_f: float = 1.5,
    forecast_uncertainty: float = 0.5,
    market_uncertainty: float = 0.3,
    min_confidence: float = 0.6,
    # Aliases for compatibility with training script
    forecast_implied: float = None,
    market_implied: float = None,
    threshold: float = None,
) -> EdgeResult:
    """Detect edge trading opportunity.

    Compares forecast-implied and market-implied temperatures to identify
    significant mispricings.

    Args:
        forecast_temp: Model's forecast-implied temperature (°F)
        market_temp: Market's implied temperature from bracket prices (°F)
        threshold_f: Minimum edge magnitude to trade (default 1.5°F)
        forecast_uncertainty: Forecast uncertainty (std dev, °F)
        market_uncertainty: Market uncertainty (spread-based, °F)
        min_confidence: Minimum confidence score to generate signal (default 0.6)
        forecast_implied: Alias for forecast_temp (for script compatibility)
        market_implied: Alias for market_temp (for script compatibility)
        threshold: Alias for threshold_f (for script compatibility)

    Returns:
        EdgeResult with trading signal and edge details

    Edge Detection Logic:
        - If forecast_temp > market_temp + threshold_f: BUY_HIGH signal
          (Forecast predicts higher temp → buy "Yes" on high brackets)

        - If forecast_temp < market_temp - threshold_f: BUY_LOW signal
          (Forecast predicts lower temp → buy "Yes" on low brackets)

        - Otherwise: NO_TRADE

    Example:
        >>> detect_edge(forecast_temp=86.5, market_temp=84.0, threshold_f=1.5)
        EdgeResult(signal=BUY_HIGH, edge=2.5, abs_edge=2.5, confidence=0.95, ...)

        # Forecast is 2.5°F higher than market → BUY_HIGH signal
    """
    # Handle parameter aliases (for backward compatibility with training script)
    if forecast_implied is not None:
        forecast_temp = forecast_implied
    if market_implied is not None:
        market_temp = market_implied
    if threshold is not None:
        threshold_f = threshold

    if forecast_temp is None or market_temp is None:
        raise ValueError("Must provide forecast_temp and market_temp (or aliases)")
    # Handle missing values
    if np.isnan(forecast_temp) or np.isnan(market_temp):
        return EdgeResult(
            signal=EdgeSignal.NO_TRADE,
            edge=0.0,
            abs_edge=0.0,
            confidence=0.0,
            forecast_temp=forecast_temp if not np.isnan(forecast_temp) else 0.0,
            market_temp=market_temp if not np.isnan(market_temp) else 0.0,
            threshold_f=threshold_f,
        )

    # Compute edge
    edge = forecast_temp - market_temp
    abs_edge = abs(edge)

    # Compute confidence using statistical significance
    # Confidence = P(|edge| > 0 | observed difference)
    # Use normal approximation with combined uncertainty
    combined_unc = np.sqrt(forecast_uncertainty ** 2 + market_uncertainty ** 2)

    if combined_unc > 0:
        # Z-score: how many standard deviations is the edge
        z_score = abs_edge / combined_unc

        # Confidence using normal CDF
        # P(edge is real) = 1 - 2*P(Z < -|edge|/sigma) where Z ~ N(0,1)
        from scipy.stats import norm

        confidence = float(1.0 - 2.0 * norm.cdf(-z_score))
    else:
        # No uncertainty information → moderate confidence
        confidence = 0.5

    # Determine signal
    signal = EdgeSignal.NO_TRADE

    if abs_edge >= threshold_f and confidence >= min_confidence:
        if edge > 0:
            # Forecast > Market → Expect higher temp → BUY_HIGH
            signal = EdgeSignal.BUY_HIGH
        else:
            # Forecast < Market → Expect lower temp → BUY_LOW
            signal = EdgeSignal.BUY_LOW

    return EdgeResult(
        signal=signal,
        edge=float(edge),
        abs_edge=float(abs_edge),
        confidence=float(confidence),
        forecast_temp=float(forecast_temp),
        market_temp=float(market_temp),
        threshold_f=float(threshold_f),
    )


def select_bracket_for_signal(
    signal: EdgeSignal,
    forecast_temp: float,
    available_brackets: list,
    bracket_thresholds: dict,
) -> Optional[str]:
    """Select which bracket to trade based on signal.

    Args:
        signal: Edge trading signal (BUY_HIGH or BUY_LOW)
        forecast_temp: Forecast-implied temperature
        available_brackets: List of available bracket labels (e.g., ["T_ge_85", "T_ge_86"])
        bracket_thresholds: Dict mapping bracket_label → threshold temp

    Returns:
        Bracket label to trade, or None if no suitable bracket

    Selection Logic:
        - BUY_HIGH: Buy bracket with threshold just below forecast_temp
          Example: forecast=86.5 → buy "T_ge_86" (not "T_ge_87")

        - BUY_LOW: Buy bracket with threshold just above forecast_temp
          Example: forecast=84.5 → buy "T_ge_85" (not "T_ge_84")

    Example:
        >>> select_bracket_for_signal(
        ...     EdgeSignal.BUY_HIGH,
        ...     forecast_temp=86.5,
        ...     available_brackets=["T_ge_85", "T_ge_86", "T_ge_87"],
        ...     bracket_thresholds={"T_ge_85": 85, "T_ge_86": 86, "T_ge_87": 87}
        ... )
        "T_ge_86"
    """
    if signal == EdgeSignal.NO_TRADE:
        return None

    # Get thresholds for available brackets
    valid_brackets = [
        (label, bracket_thresholds[label])
        for label in available_brackets
        if label in bracket_thresholds
    ]

    if not valid_brackets:
        logger.warning("No valid brackets with thresholds")
        return None

    # Sort by threshold
    valid_brackets.sort(key=lambda x: x[1])

    if signal == EdgeSignal.BUY_HIGH:
        # Buy bracket just below forecast temp
        # Find largest threshold <= forecast_temp
        candidates = [
            (label, thresh) for label, thresh in valid_brackets if thresh <= forecast_temp
        ]

        if not candidates:
            # Forecast is below all brackets → buy lowest bracket
            return valid_brackets[0][0]

        # Return bracket with highest threshold <= forecast
        return max(candidates, key=lambda x: x[1])[0]

    elif signal == EdgeSignal.BUY_LOW:
        # Buy bracket just above forecast temp
        # Find smallest threshold >= forecast_temp
        candidates = [
            (label, thresh) for label, thresh in valid_brackets if thresh >= forecast_temp
        ]

        if not candidates:
            # Forecast is above all brackets → buy highest bracket
            return valid_brackets[-1][0]

        # Return bracket with lowest threshold >= forecast
        return min(candidates, key=lambda x: x[1])[0]

    return None
