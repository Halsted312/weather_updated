#!/usr/bin/env python3
"""
Evaluation metrics for probability calibration.

Includes:
- ECE (Expected Calibration Error): measures calibration quality
- Calibration summary: bin-level analysis of predicted vs observed probabilities
- Standard metrics: log_loss, brier_score

References:
- ECE: https://arxiv.org/abs/1706.04599 (On Calibration of Modern Neural Networks)
- sklearn calibration: https://scikit-learn.org/stable/modules/calibration.html
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import log_loss, brier_score_loss


def compute_ece(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform"
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted probabilities and observed frequencies,
    averaged across probability bins. Lower ECE indicates better calibration.

    Args:
        y_true: True binary labels (N,)
        p_pred: Predicted probabilities (N,)
        n_bins: Number of bins for calibration (default: 10)
        strategy: Binning strategy ("uniform" or "quantile")
            - "uniform": equal-width bins in [0, 1]
            - "quantile": equal-count bins (better for imbalanced predictions)

    Returns:
        ECE: Expected Calibration Error in [0, 1]

    Formula:
        ECE = Σ (n_k / N) * |accuracy_k - confidence_k|

    where:
        - k indexes bins
        - n_k = number of samples in bin k
        - N = total samples
        - accuracy_k = fraction of positives in bin k
        - confidence_k = mean predicted probability in bin k
    """
    if len(y_true) == 0:
        return np.nan

    # Create bins
    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        # Use quantiles for equal-count bins
        bins = np.percentile(p_pred, np.linspace(0, 100, n_bins + 1))
        bins[0] = 0.0  # Ensure first bin starts at 0
        bins[-1] = 1.0  # Ensure last bin ends at 1
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Digitize predictions into bins
    bin_indices = np.digitize(p_pred, bins, right=True)
    bin_indices = np.clip(bin_indices, 1, n_bins)  # Ensure valid bin indices

    ece = 0.0
    for i in range(1, n_bins + 1):
        mask = (bin_indices == i)
        n_k = np.sum(mask)

        if n_k > 0:
            # Mean predicted probability in this bin
            confidence_k = np.mean(p_pred[mask])

            # Observed frequency of positives in this bin
            accuracy_k = np.mean(y_true[mask])

            # Weighted contribution to ECE
            ece += (n_k / len(y_true)) * np.abs(accuracy_k - confidence_k)

    return float(ece)


def calibration_summary(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform"
) -> Dict[str, any]:
    """
    Generate calibration summary with bin-level statistics.

    Args:
        y_true: True binary labels (N,)
        p_pred: Predicted probabilities (N,)
        n_bins: Number of bins (default: 10)
        strategy: Binning strategy ("uniform" or "quantile")

    Returns:
        Dict with:
            - ece: Expected Calibration Error
            - bins: List of dicts with bin-level stats:
                - bin_range: (lower, upper) edges
                - count: number of samples
                - mean_pred: mean predicted probability
                - observed_freq: observed frequency of positives
                - calibration_error: |observed_freq - mean_pred|
    """
    if len(y_true) == 0:
        return {
            "ece": np.nan,
            "bins": [],
        }

    # Create bins
    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.percentile(p_pred, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Digitize predictions
    bin_indices = np.digitize(p_pred, bin_edges, right=True)
    bin_indices = np.clip(bin_indices, 1, n_bins)

    # Compute bin-level stats
    bins = []
    ece = 0.0
    for i in range(1, n_bins + 1):
        mask = (bin_indices == i)
        n_k = np.sum(mask)

        if n_k > 0:
            mean_pred = float(np.mean(p_pred[mask]))
            observed_freq = float(np.mean(y_true[mask]))
            calib_error = abs(observed_freq - mean_pred)

            bins.append({
                "bin_range": (float(bin_edges[i - 1]), float(bin_edges[i])),
                "count": int(n_k),
                "mean_pred": mean_pred,
                "observed_freq": observed_freq,
                "calibration_error": calib_error,
            })

            ece += (n_k / len(y_true)) * calib_error

    return {
        "ece": float(ece),
        "bins": bins,
    }


def evaluate_predictions(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Comprehensive evaluation of probability predictions.

    Args:
        y_true: True binary labels (N,)
        p_pred: Predicted probabilities (N,)
        n_bins: Number of bins for calibration (default: 10)

    Returns:
        Dict with:
            - log_loss: Log loss (lower is better)
            - brier: Brier score (lower is better)
            - ece: Expected Calibration Error (lower is better)
            - accuracy: Classification accuracy (at threshold 0.5)
    """
    if len(y_true) == 0:
        return {
            "log_loss": np.nan,
            "brier": np.nan,
            "ece": np.nan,
            "accuracy": np.nan,
        }

    # Standard metrics
    ll = float(log_loss(y_true, p_pred, eps=1e-6))
    brier = float(brier_score_loss(y_true, p_pred))

    # Calibration metric
    ece = compute_ece(y_true, p_pred, n_bins=n_bins, strategy="uniform")

    # Classification accuracy (for reference)
    y_pred_class = (p_pred >= 0.5).astype(int)
    acc = float(np.mean(y_pred_class == y_true))

    return {
        "log_loss": ll,
        "brier": brier,
        "ece": ece,
        "accuracy": acc,
    }


def main():
    """Demo: Compute ECE on synthetic data."""
    print("\n" + "="*60)
    print("ECE and Calibration Demo")
    print("="*60 + "\n")

    # Generate synthetic data with varying calibration quality
    np.random.seed(42)
    N = 1000

    # Case 1: Well-calibrated (predicted = observed)
    y_true_good = np.random.binomial(1, 0.3, N)
    p_pred_good = np.clip(y_true_good + np.random.normal(0, 0.1, N), 0, 1)

    # Case 2: Overconfident (predicted > observed)
    y_true_over = np.random.binomial(1, 0.3, N)
    p_pred_over = np.clip(y_true_over * 0.9 + 0.4, 0, 1)

    # Case 3: Underconfident (predicted < observed)
    y_true_under = np.random.binomial(1, 0.7, N)
    p_pred_under = np.clip(y_true_under * 0.6, 0, 1)

    # Evaluate
    for name, y_true, p_pred in [
        ("Well-calibrated", y_true_good, p_pred_good),
        ("Overconfident", y_true_over, p_pred_over),
        ("Underconfident", y_true_under, p_pred_under),
    ]:
        metrics = evaluate_predictions(y_true, p_pred, n_bins=10)
        cal_summary = calibration_summary(y_true, p_pred, n_bins=10)

        print(f"\n{name}:")
        print(f"  Log loss: {metrics['log_loss']:.4f}")
        print(f"  Brier:    {metrics['brier']:.4f}")
        print(f"  ECE:      {metrics['ece']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")

        print(f"\n  Calibration bins:")
        for bin_info in cal_summary["bins"][:5]:  # Show first 5 bins
            print(f"    [{bin_info['bin_range'][0]:.1f}, {bin_info['bin_range'][1]:.1f}]: "
                  f"n={bin_info['count']}, pred={bin_info['mean_pred']:.2f}, "
                  f"obs={bin_info['observed_freq']:.2f}, error={bin_info['calibration_error']:.3f}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
