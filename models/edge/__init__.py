"""
Edge detection and classification for Kalshi weather trading.

This module provides tools for:
1. Computing forecast-implied and market-implied temperatures
2. Detecting arbitrage/edge opportunities
3. Classifying edge quality with ML

Key components:
- implied_temp: Temperature inference from probabilities and market prices
- detector: Edge signal detection logic
- classifier: ML-based edge quality classifier (CatBoost + calibration)
"""

from models.edge.detector import EdgeSignal, EdgeResult, detect_edge
from models.edge.implied_temp import (
    compute_forecast_implied_temp,
    compute_market_implied_temp,
    ForecastImpliedResult,
    MarketImpliedResult,
)

__all__ = [
    # Detector
    "EdgeSignal",
    "EdgeResult",
    "detect_edge",
    # Implied temperature
    "compute_forecast_implied_temp",
    "compute_market_implied_temp",
    "ForecastImpliedResult",
    "MarketImpliedResult",
]
