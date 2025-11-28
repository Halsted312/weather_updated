"""
Core data structures for temperature reverse-engineering.

This module defines the foundational types used to evaluate temperature
rounding rules against historical Kalshi/NWS settlements.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional


@dataclass
class DaySeries:
    """One day's 5-minute Visual Crossing temps + actual settlement high.

    Attributes:
        city: City identifier (e.g., 'chicago', 'austin')
        day: Local calendar date for the CLI day
        temps_f: List of 5-minute VC temperatures in °F for the CLI window
        settle_f: Actual NWS/Kalshi integer daily high (ground truth)
        vc_max_f: Raw maximum from VC series (for debugging)
        num_samples: Number of 5-minute samples in the day
    """

    city: str
    day: date
    temps_f: List[float]
    settle_f: int
    vc_max_f: float = field(init=False)
    num_samples: int = field(init=False)

    def __post_init__(self):
        """Compute derived fields."""
        if self.temps_f:
            self.vc_max_f = max(self.temps_f)
            self.num_samples = len(self.temps_f)
        else:
            self.vc_max_f = 0.0
            self.num_samples = 0


@dataclass
class RuleStats:
    """Performance tracking for a single temperature rounding rule.

    Tracks exact matches, off-by-1, off-by-2+, and mean absolute error
    to quantify rule accuracy against historical settlements.

    Attributes:
        name: Rule identifier
        total: Total days evaluated
        exact_matches: Days where prediction = settlement
        off_by_1: Days where |prediction - settlement| = 1
        off_by_2plus: Days where |prediction - settlement| >= 2
        sum_abs_error: Sum of absolute errors (for MAE calculation)
    """

    name: str
    total: int = 0
    exact_matches: int = 0
    off_by_1: int = 0
    off_by_2plus: int = 0
    sum_abs_error: int = 0

    def update(self, pred: Optional[int], actual: int) -> None:
        """Update stats with a new prediction.

        Args:
            pred: Predicted integer °F (None = skip)
            actual: Actual settled integer °F
        """
        if pred is None:
            return

        self.total += 1
        diff = abs(pred - actual)

        if diff == 0:
            self.exact_matches += 1
        elif diff == 1:
            self.off_by_1 += 1
        else:
            self.off_by_2plus += 1

        self.sum_abs_error += diff

    @property
    def accuracy(self) -> float:
        """Fraction of exact matches (0.0 to 1.0)."""
        return self.exact_matches / self.total if self.total > 0 else 0.0

    @property
    def mae(self) -> float:
        """Mean absolute error in °F."""
        return self.sum_abs_error / self.total if self.total > 0 else 0.0

    @property
    def off_by_1_rate(self) -> float:
        """Fraction of predictions off by exactly 1°F."""
        return self.off_by_1 / self.total if self.total > 0 else 0.0

    @property
    def off_by_2plus_rate(self) -> float:
        """Fraction of predictions off by 2+ °F."""
        return self.off_by_2plus / self.total if self.total > 0 else 0.0

    def __str__(self) -> str:
        """Human-readable summary."""
        if self.total == 0:
            return f"{self.name}: No data"

        return (
            f"{self.name:20s} | "
            f"Accuracy: {self.accuracy:6.2%} | "
            f"MAE: {self.mae:.3f}°F | "
            f"Exact: {self.exact_matches:4d}/{self.total:4d} | "
            f"Off±1: {self.off_by_1:4d} | "
            f"Off≥2: {self.off_by_2plus:4d}"
        )
