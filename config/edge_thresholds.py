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
    "austin": 9.5,         # Sweep: 85% baseline win rate
    "chicago": 10.0,       # Classifier: Sharpe 2.77, 94% win, 36 trades
    "denver": 10.0,        # Classifier: Sharpe 0.78, 77% win, 73 trades
    "los_angeles": 10.0,   # Sweep 2025-12-09: Sharpe 0.54, 76% win, 11,140 trades
    "miami": 9.0,          # Sweep 2025-12-09: Sharpe 0.55, 74% win, 782 trades
    "philadelphia": 10.0,  # Sweep 2025-12-09: Sharpe 0.48, 82% win, 2,563 trades
}

# Default threshold if city not found
DEFAULT_MIN_EDGE_THRESHOLD_F = 1.5

# Sweep configuration: thresholds to test when finding optimal edge threshold
# Range: 0.5 to 11.0 by 0.5 increments
SWEEP_THRESHOLDS = [x / 2 for x in range(1, 23)]  # [0.5, 1.0, 1.5, ..., 11.0]


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
