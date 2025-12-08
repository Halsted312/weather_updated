"""Edge detection threshold configuration.

This module stores the optimal min edge thresholds (in °F) for each city,
determined by sweeping against P&L/Sharpe metrics.

The edge system has two thresholds:
1. Edge magnitude threshold (°F) - defined here
   "How much model-market disagreement do I need before considering a trade?"

2. Classifier probability threshold (0-1) - tuned by Optuna in EdgeClassifier
   "Among edge candidates, which ones does the ML filter green-light?"

To update these values:
    PYTHONPATH=. python scripts/sweep_min_edge_threshold.py --city <city>

Usage:
    from config.edge_thresholds import get_min_edge_threshold

    threshold = get_min_edge_threshold("austin")  # Returns 1.5 (default)
"""

from typing import Dict

# Optimal min edge thresholds by city (in °F)
# These values should be updated based on sweep results
# Run: PYTHONPATH=. python scripts/sweep_min_edge_threshold.py --city <city>
EDGE_MIN_THRESHOLD_F: Dict[str, float] = {
    "austin": 1.5,        # Default - update after sweep
    "chicago": 1.5,       # Default - update after sweep
    "denver": 10.0,        # Default - update after sweep
    "los_angeles": 1.5,   # Default - update after sweep
    "miami": 7.5,         # Updated 2025-12-07: Sweep found 7.5°F optimal (1,955 trades, 56% win, Sharpe 0.48)
    "philadelphia": 1.5,  # Default - update after sweep
}

# Default threshold if city not found
DEFAULT_MIN_EDGE_THRESHOLD_F = 1.5


def get_min_edge_threshold(city: str) -> float:
    """Get the min edge threshold for a city.

    Args:
        city: City name (lowercase)

    Returns:
        Min edge threshold in °F
    """
    return EDGE_MIN_THRESHOLD_F.get(city.lower(), DEFAULT_MIN_EDGE_THRESHOLD_F)


def get_all_thresholds() -> Dict[str, float]:
    """Get all city thresholds.

    Returns:
        Dict mapping city -> threshold
    """
    return EDGE_MIN_THRESHOLD_F.copy()
