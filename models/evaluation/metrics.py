"""
Evaluation metrics for temperature Δ-models.

This module provides comprehensive metrics for evaluating model performance:
- Classification metrics: accuracy, off-by-1 rate, off-by-2+ rate
- Regression metrics: MAE, RMSE on settlement temperature
- Probabilistic metrics: log loss, Brier score, calibration
- Bracket-level metrics: P(T >= K) accuracy for key thresholds

Example:
    >>> from models.evaluation.metrics import compute_all_metrics
    >>> metrics = compute_all_metrics(y_true, y_pred, proba, t_base, t_settle)
    >>> print(f"Delta accuracy: {metrics['delta_accuracy']:.2%}")
    >>> print(f"Brier score (T>=90): {metrics['brier_90']:.4f}")
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.calibration import calibration_curve

from models.features.base import DELTA_CLASSES

logger = logging.getLogger(__name__)


def compute_delta_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute classification metrics on delta predictions.

    Args:
        y_true: True delta values
        y_pred: Predicted delta values

    Returns:
        Dict with delta_accuracy, delta_mae, off_by_1_rate, off_by_2plus_rate
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Off-by-1 and off-by-2+ rates
    diff = np.abs(y_true - y_pred)
    off_by_1 = np.mean(diff == 1)
    off_by_2plus = np.mean(diff >= 2)

    return {
        "delta_accuracy": float(accuracy),
        "delta_mae": float(mae),
        "off_by_1_rate": float(off_by_1),
        "off_by_2plus_rate": float(off_by_2plus),
    }


def compute_settlement_metrics(
    t_settle: np.ndarray,
    t_pred: np.ndarray,
) -> dict:
    """Compute metrics on settlement temperature predictions.

    Args:
        t_settle: True settlement temperatures
        t_pred: Predicted settlement temperatures (t_base + delta_pred)

    Returns:
        Dict with settlement_accuracy, settlement_mae, settlement_rmse
    """
    t_settle = np.asarray(t_settle)
    t_pred = np.asarray(t_pred)

    accuracy = np.mean(t_settle == t_pred)
    mae = mean_absolute_error(t_settle, t_pred)
    rmse = np.sqrt(mean_squared_error(t_settle, t_pred))

    return {
        "settlement_accuracy": float(accuracy),
        "settlement_mae": float(mae),
        "settlement_rmse": float(rmse),
    }


def compute_probabilistic_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: Optional[np.ndarray] = None,
) -> dict:
    """Compute probabilistic calibration metrics.

    Args:
        y_true: True delta values
        proba: Predicted probabilities (shape: n_samples x n_classes)
        classes: Class labels (default: DELTA_CLASSES)

    Returns:
        Dict with log_loss, expected_calibration_error
    """
    if classes is None:
        classes = np.array(DELTA_CLASSES)

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    # Log loss
    try:
        ll = log_loss(y_true, proba, labels=classes)
    except Exception as e:
        logger.warning(f"Could not compute log loss: {e}")
        ll = np.nan

    # Expected Calibration Error (ECE)
    # For multiclass, we compute ECE as average across classes
    ece = _compute_multiclass_ece(y_true, proba, classes)

    return {
        "log_loss": float(ll),
        "expected_calibration_error": float(ece),
    }


def _compute_multiclass_ece(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error for multiclass.

    ECE measures how well predicted probabilities match observed frequencies.
    """
    ece_sum = 0.0
    total_weight = 0.0

    for i, c in enumerate(classes):
        # Binary: is this class vs not
        y_binary = (y_true == c).astype(int)
        p_class = proba[:, i]

        # Bin by predicted probability
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(p_class, bins[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() == 0:
                continue

            # Average predicted prob in bin
            avg_pred = p_class[mask].mean()
            # Observed frequency in bin
            avg_true = y_binary[mask].mean()

            weight = mask.sum()
            ece_sum += weight * abs(avg_pred - avg_true)
            total_weight += weight

    if total_weight > 0:
        return ece_sum / total_weight
    return 0.0


def compute_bracket_brier_score(
    proba: np.ndarray,
    t_base: np.ndarray,
    t_settle: np.ndarray,
    threshold: int,
    delta_classes: Optional[np.ndarray] = None,
) -> float:
    """Compute Brier score for bracket event P(T >= threshold).

    The Brier score measures how well calibrated the bracket probabilities are.
    Lower is better (0 = perfect, 0.25 = random for 50/50 event).

    Args:
        proba: Delta probability matrix (n_samples x n_classes)
        t_base: Baseline temperatures
        t_settle: Actual settlement temperatures
        threshold: Temperature threshold (e.g., 90 for "T >= 90")
        delta_classes: Delta class values (default: DELTA_CLASSES)

    Returns:
        Brier score for the bracket event
    """
    if delta_classes is None:
        delta_classes = np.array(DELTA_CLASSES)

    n_samples = len(t_base)
    p_bracket = np.zeros(n_samples)

    # For each sample, compute P(T >= threshold) from delta probs
    for i in range(n_samples):
        t_base_i = t_base[i]
        # T = t_base + delta, so T >= threshold when delta >= threshold - t_base
        min_delta = threshold - t_base_i

        # Sum probabilities for all deltas that would give T >= threshold
        for j, d in enumerate(delta_classes):
            if t_base_i + d >= threshold:
                p_bracket[i] += proba[i, j]

    # Observed outcome: did T_settle >= threshold?
    y_bracket = (t_settle >= threshold).astype(float)

    # Brier score = mean squared error between predicted prob and outcome
    brier = np.mean((p_bracket - y_bracket) ** 2)

    return float(brier)


def compute_calibration_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Generate calibration curve data points.

    Args:
        y_true: Binary outcomes
        y_prob: Predicted probabilities
        n_bins: Number of calibration bins

    Returns:
        DataFrame with prob_pred, prob_true columns
    """
    try:
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        return pd.DataFrame({
            "prob_pred": prob_pred,
            "prob_true": prob_true,
        })
    except Exception as e:
        logger.warning(f"Could not compute calibration curve: {e}")
        return pd.DataFrame()


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    t_base: np.ndarray,
    t_settle: np.ndarray,
    delta_classes: Optional[np.ndarray] = None,
    thresholds: list[int] = [80, 85, 90, 95],
) -> dict:
    """Compute all evaluation metrics.

    Args:
        y_true: True delta values
        y_pred: Predicted delta values
        proba: Predicted probabilities
        t_base: Baseline temperatures
        t_settle: Actual settlement temperatures
        delta_classes: Delta class values
        thresholds: Temperature thresholds for bracket metrics

    Returns:
        Dict with all metrics
    """
    if delta_classes is None:
        delta_classes = np.array(DELTA_CLASSES)

    # Delta metrics
    metrics = compute_delta_metrics(y_true, y_pred)

    # Settlement metrics
    t_pred = t_base + y_pred
    settlement_metrics = compute_settlement_metrics(t_settle, t_pred)
    metrics.update(settlement_metrics)

    # Probabilistic metrics
    prob_metrics = compute_probabilistic_metrics(y_true, proba, delta_classes)
    metrics.update(prob_metrics)

    # Bracket Brier scores
    for threshold in thresholds:
        brier = compute_bracket_brier_score(
            proba, t_base, t_settle, threshold, delta_classes
        )
        metrics[f"brier_{threshold}"] = brier

    return metrics


def compute_metrics_by_snapshot_hour(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    proba: np.ndarray,
) -> pd.DataFrame:
    """Compute metrics stratified by snapshot hour.

    This reveals how model performance varies throughout the day -
    typically better (more confident) in later hours.

    Args:
        df: Test DataFrame with snapshot_hour, delta, settle_f, t_base
        y_pred: Predicted delta values
        proba: Predicted probabilities

    Returns:
        DataFrame with metrics per snapshot hour
    """
    df = df.copy()
    df["delta_pred"] = y_pred
    df["t_pred"] = df["t_base"] + df["delta_pred"]

    results = []

    for hour in sorted(df["snapshot_hour"].unique()):
        mask = df["snapshot_hour"] == hour
        df_hour = df[mask]
        proba_hour = proba[mask]

        if len(df_hour) < 10:
            continue

        y_true = df_hour["delta"].values
        y_pred_hour = df_hour["delta_pred"].values
        t_base = df_hour["t_base"].values
        t_settle = df_hour["settle_f"].values

        metrics = compute_all_metrics(
            y_true, y_pred_hour, proba_hour, t_base, t_settle
        )
        metrics["snapshot_hour"] = hour
        metrics["n_samples"] = len(df_hour)

        results.append(metrics)

    return pd.DataFrame(results)
