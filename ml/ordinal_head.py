#!/usr/bin/env python3
"""
Ordinal unified head for temperature bracket prediction.

Trains K cumulative binary models for thresholds, then computes bracket
probabilities as CDF differences. This provides a coherent probability
distribution across all brackets while respecting the ordinal structure.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrdinalThresholds:
    """City/day-specific temperature thresholds."""
    taus: List[float]  # e.g., [69, 71, 73, 75, 76] for 6 bins
    city: str
    date: Optional[str] = None


def fit_cdf_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_temp_col: str,
    taus: List[float],
    l1_ratio: float = 0.5,
    C: float = 1.0,
    max_iter: int = 5000
) -> Tuple[List, List]:
    """
    Fit binary models for each temperature threshold.

    Args:
        df: Training data
        feature_cols: Feature column names
        label_temp_col: Column with actual temperature
        taus: Temperature thresholds
        l1_ratio: ElasticNet mixing parameter
        C: Regularization strength
        max_iter: Maximum iterations

    Returns:
        Tuple of (models, calibrators)
    """
    models = []
    calibrators = []

    X = df[feature_cols].values

    for i, tau in enumerate(taus):
        logger.info(f"Training CDF model for tau={tau} ({i+1}/{len(taus)})")

        # Binary label: temperature <= tau
        y_tau = (df[label_temp_col] <= tau).astype(int)

        # Skip if all same class
        if len(np.unique(y_tau)) < 2:
            logger.warning(f"Single class for tau={tau}, using dummy model")
            # Create dummy model that always predicts the single class
            class DummyModel:
                def __init__(self, pred_class):
                    self.pred_class = pred_class
                def predict_proba(self, X):
                    n = len(X)
                    if self.pred_class == 1:
                        return np.column_stack([np.zeros(n), np.ones(n)])
                    else:
                        return np.column_stack([np.ones(n), np.zeros(n)])

            models.append(DummyModel(y_tau[0]))
            calibrators.append(None)
            continue

        # Train ElasticNet logistic regression
        model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=l1_ratio,
            C=C,
            max_iter=max_iter,
            random_state=42
        )
        model.fit(X, y_tau)

        # Isotonic calibration
        y_pred = model.predict_proba(X)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_pred, y_tau)

        models.append(model)
        calibrators.append(calibrator)

    return models, calibrators


def predict_brackets(
    X: np.ndarray,
    models: List,
    calibrators: List,
    taus: List[float]
) -> np.ndarray:
    """
    Predict bracket probabilities using CDF differences.

    Args:
        X: Feature matrix (N x D)
        models: List of binary models
        calibrators: List of calibrators
        taus: Temperature thresholds

    Returns:
        Array of bracket probabilities (N x K+1)
    """
    n_samples = len(X)
    n_thresholds = len(taus)

    # Get CDF predictions for each threshold
    cdf = np.zeros((n_thresholds, n_samples))

    for i, (model, cal) in enumerate(zip(models, calibrators)):
        # Get probability of T <= tau_i
        p = model.predict_proba(X)[:, 1]

        # Apply calibration if available
        if cal is not None:
            p = cal.transform(p.reshape(-1, 1)).ravel()

        cdf[i] = p

    # Clip to [0, 1]
    cdf = np.clip(cdf, 0, 1)

    # Enforce monotonicity (CDF must be non-decreasing)
    cdf = np.maximum.accumulate(cdf, axis=0)

    # Convert CDF to bracket probabilities
    n_brackets = n_thresholds + 1
    p_brackets = np.zeros((n_samples, n_brackets))

    # First bracket: P(T <= tau_0)
    p_brackets[:, 0] = cdf[0]

    # Middle brackets: P(tau_{i-1} < T <= tau_i)
    for i in range(1, n_thresholds):
        p_brackets[:, i] = np.maximum(0.0, cdf[i] - cdf[i-1])

    # Last bracket: P(T > tau_{n-1})
    p_brackets[:, n_brackets-1] = np.maximum(0.0, 1.0 - cdf[-1])

    # Renormalize to ensure sum to 1
    row_sums = p_brackets.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)  # Avoid division by zero
    p_brackets = p_brackets / row_sums

    return p_brackets


def map_to_bracket_probs(
    df: pd.DataFrame,
    p_brackets: np.ndarray,
    bracket_mapping: Optional[dict] = None,
    bin_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Map ordinal predictions to specific bracket probabilities.

    Args:
        df: DataFrame with bracket_type, floor_strike, cap_strike
        p_brackets: Ordinal probability matrix (N x K+1)
        bracket_mapping: Maps (bracket_type, floor, cap) to ordinal bin index
        bin_col: Optional column containing per-row bin indices (overrides mapping)

    Returns:
        DataFrame with added p_ordinal column
    """
    df = df.copy()
    df['p_ordinal'] = 0.0

    if bin_col is not None and bin_col in df.columns:
        bins = df[bin_col].astype(int).tolist()
        for i, bin_idx in enumerate(bins):
            if 0 <= bin_idx < p_brackets.shape[1]:
                df.loc[df.index[i], 'p_ordinal'] = p_brackets[i, bin_idx]
            else:
                logger.warning("Ordinal bin %s out of range", bin_idx)
        return df

    if bracket_mapping is None:
        raise ValueError("Either bracket_mapping or bin_col must be provided")

    for i, row in df.iterrows():
        key = (row['bracket_type'], row.get('floor_strike'), row.get('cap_strike'))
        if key in bracket_mapping:
            bin_idx = bracket_mapping[key]
            df.loc[i, 'p_ordinal'] = p_brackets[i, bin_idx]
        else:
            logger.warning(f"No mapping for bracket {key}")

    return df


def create_bracket_mapping(city: str, date: str) -> Tuple[List[float], dict]:
    """
    Create temperature thresholds and bracket mapping for a city/date.

    Returns:
        Tuple of (thresholds, bracket_mapping)
    """
    # Example for Chicago (adjust based on actual brackets)
    if city.lower() == 'chicago':
        # Common Chicago temperature brackets
        # Assuming 6 brackets: <68, 68-73, 73-78, 78-83, 83-88, >88
        taus = [68, 73, 78, 83, 88]

        bracket_mapping = {
            ('less', None, 68): 0,
            ('between', 68, 73): 1,
            ('between', 73, 78): 2,
            ('between', 78, 83): 3,
            ('between', 83, 88): 4,
            ('greater', 88, None): 5,
        }
    else:
        raise ValueError(f"Unknown city: {city}")

    return taus, bracket_mapping


def main():
    """Demo of ordinal unified head."""
    print("Ordinal Unified Head Demo")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000

    # Features
    X = np.random.randn(n_samples, 10)

    # True temperature (correlated with features)
    temp = 75 + 5 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n_samples) * 2

    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['temperature'] = temp

    # Define thresholds
    taus = [70, 75, 80]

    # Fit models
    feature_cols = [f'feature_{i}' for i in range(10)]
    models, calibrators = fit_cdf_models(
        df, feature_cols, 'temperature', taus
    )

    # Predict
    p_brackets = predict_brackets(X, models, calibrators, taus)

    print(f"\nPredicted bracket probabilities (first 5 samples):")
    print(p_brackets[:5])

    print(f"\nSum check (should all be ~1.0):")
    print(p_brackets[:5].sum(axis=1))

    print("\nDone!")


if __name__ == "__main__":
    main()
