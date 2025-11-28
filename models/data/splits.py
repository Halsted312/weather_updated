"""
Train/test splitting utilities for temperature Δ-models.

This module implements day-based temporal splits to avoid lookahead leakage.
The key constraint is that ALL snapshots from a given day must go to the
same fold (train or test) - we never train on future days.

Temporal splits:
    - Train/test split by calendar date
    - TimeSeriesSplit for cross-validation (expanding window)
    - Optional gap between train and test to reduce autocorrelation

Example:
    >>> from models.data.splits import train_test_split_by_date
    >>> df_train, df_test = train_test_split_by_date(df, cutoff_date=date(2025, 1, 1))
    >>> # All snapshots from days < 2025-01-01 in train
    >>> # All snapshots from days >= 2025-01-01 in test
"""

from datetime import date, timedelta
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def train_test_split_by_date(
    df: pd.DataFrame,
    cutoff_date: date,
    date_col: str = "day",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by calendar date.

    All rows with day < cutoff go to train, day >= cutoff go to test.
    This ensures no future information leaks into training.

    Args:
        df: DataFrame with snapshot data
        cutoff_date: Split date (test includes this date and later)
        date_col: Column name containing dates

    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.copy()

    # Ensure date column is proper date type
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col]).dt.date

    train_mask = df[date_col] < cutoff_date
    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[~train_mask].reset_index(drop=True)

    return df_train, df_test


def train_test_split_by_ratio(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    date_col: str = "day",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by date ratio (e.g., last 20% of days for test).

    Args:
        df: DataFrame with snapshot data
        test_ratio: Fraction of days for test set (default 0.2)
        date_col: Column name containing dates

    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.copy()

    # Get unique days sorted
    unique_days = sorted(df[date_col].unique())
    n_days = len(unique_days)

    # Calculate split point
    n_test_days = max(1, int(n_days * test_ratio))
    cutoff_idx = n_days - n_test_days
    cutoff_date = unique_days[cutoff_idx]

    return train_test_split_by_date(df, cutoff_date, date_col)


class DayGroupedTimeSeriesSplit:
    """Time series cross-validator that keeps all snapshots from a day together.

    Similar to sklearn's TimeSeriesSplit but ensures that all rows from
    the same day stay in the same fold. This is crucial for avoiding
    lookahead bias when multiple snapshot hours exist per day.

    Args:
        n_splits: Number of CV folds
        gap_days: Number of days gap between train and validation
                  (helps reduce autocorrelation effects)
    """

    def __init__(self, n_splits: int = 5, gap_days: int = 0):
        self.n_splits = n_splits
        self.gap_days = gap_days

    def split(
        self,
        X: pd.DataFrame,
        y=None,
        groups=None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation indices for CV.

        Args:
            X: DataFrame with 'day' column
            y: Ignored (for sklearn compatibility)
            groups: Ignored (days are used as groups)

        Yields:
            Tuple of (train_indices, val_indices) for each fold
        """
        if "day" not in X.columns:
            raise ValueError("DataFrame must have 'day' column")

        # Get unique days sorted
        unique_days = sorted(X["day"].unique())
        n_days = len(unique_days)

        # Create mapping from day to row indices
        day_to_indices = {}
        for day in unique_days:
            day_to_indices[day] = X[X["day"] == day].index.tolist()

        # Calculate fold sizes
        # Using expanding window: each fold has more training data
        min_train_days = n_days // (self.n_splits + 1)

        for fold in range(self.n_splits):
            # Training days: expanding window
            n_train_days = min_train_days + (fold * n_days // (self.n_splits + 1))

            # Validation days: fixed size chunk
            val_start = n_train_days + self.gap_days
            val_end = val_start + (n_days - n_train_days - self.gap_days) // self.n_splits

            if val_end > n_days:
                val_end = n_days
            if val_start >= n_days:
                continue

            train_days = unique_days[:n_train_days]
            val_days = unique_days[val_start:val_end]

            # Collect indices
            train_indices = []
            for day in train_days:
                train_indices.extend(day_to_indices[day])

            val_indices = []
            for day in val_days:
                val_indices.extend(day_to_indices[day])

            if train_indices and val_indices:
                yield np.array(train_indices), np.array(val_indices)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits


def make_time_series_cv(
    n_splits: int = 5,
    gap_days: int = 1,
) -> DayGroupedTimeSeriesSplit:
    """Create a time-series CV splitter with day grouping.

    Args:
        n_splits: Number of CV folds
        gap_days: Days of gap between train and validation sets

    Returns:
        DayGroupedTimeSeriesSplit instance
    """
    return DayGroupedTimeSeriesSplit(n_splits=n_splits, gap_days=gap_days)


def get_cv_dates(
    df: pd.DataFrame,
    n_splits: int = 5,
    date_col: str = "day",
) -> list[dict]:
    """Get the date ranges for each CV fold.

    Useful for logging and debugging which dates are in each fold.

    Args:
        df: DataFrame with date column
        n_splits: Number of CV folds
        date_col: Column name containing dates

    Returns:
        List of dicts with train_start, train_end, val_start, val_end for each fold
    """
    cv = DayGroupedTimeSeriesSplit(n_splits=n_splits)
    fold_info = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(df)):
        train_dates = df.iloc[train_idx][date_col]
        val_dates = df.iloc[val_idx][date_col]

        fold_info.append({
            "fold": fold,
            "train_start": train_dates.min(),
            "train_end": train_dates.max(),
            "train_days": len(train_dates.unique()),
            "val_start": val_dates.min(),
            "val_end": val_dates.max(),
            "val_days": len(val_dates.unique()),
        })

    return fold_info
