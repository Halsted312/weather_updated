"""
Calibration diagnostic plots and metrics.

Provides tools for assessing probability calibration quality:
- Reliability diagrams (calibration curves)
- Brier score and log loss
- Expected calibration error (ECE)
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


def plot_reliability_diagram(
    y_true,
    y_proba,
    n_bins: int = 10,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """Reliability diagram (calibration curve) for binary probabilities.

    Shows how well predicted probabilities match empirical frequencies.
    A well-calibrated model will have points near the diagonal.

    Args:
        y_true: 1D array-like of true labels {0,1}
        y_proba: 1D array-like of predicted probabilities
        n_bins: Number of bins for calibration curve
        title: Optional plot title
        save_path: If provided, save PNG here

    Returns:
        Tuple of (fig, ax) matplotlib objects

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_proba = np.array([0.1, 0.3, 0.7, 0.8, 0.9])
        >>> plot_reliability_diagram(y_true, y_proba, save_path='calib.png')
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Compute calibration curve
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    # Create plot
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot calibration curve
    ax.plot(mean_pred, frac_pos, marker="o", linestyle="-", linewidth=2,
            label="Model", color='#2E86AB', markersize=8)

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
            label="Perfect calibration", linewidth=2)

    # Styling
    ax.set_xlabel("Predicted probability", fontsize=12)
    ax.set_ylabel("Empirical frequency", fontsize=12)
    ax.set_title(title or "Reliability Diagram", fontsize=14, fontweight='bold')
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add diagonal reference
    ax.axline((0, 0), slope=1, color='gray', alpha=0.3, linestyle='--', linewidth=1)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"Saved reliability diagram to {save_path}")

    return fig, ax


def summarize_calibration(y_true, y_proba) -> dict:
    """Compute basic calibration metrics (Brier score, log loss).

    Args:
        y_true: 1D array-like of true labels {0,1}
        y_proba: 1D array-like of predicted probabilities

    Returns:
        Dict with calibration metrics:
        - brier: Brier score (lower is better, range [0, 1])
        - log_loss: Log loss / cross-entropy (lower is better)
        - ece: Expected calibration error (lower is better)

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_proba = np.array([0.1, 0.3, 0.7, 0.8, 0.9])
        >>> summarize_calibration(y_true, y_proba)
        {'brier': 0.08, 'log_loss': 0.31, 'ece': 0.03}
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Clip probabilities to avoid log(0)
    eps = 1e-15
    y_proba_clipped = np.clip(y_proba, eps, 1 - eps)

    # Brier score: mean squared error of probabilities
    brier = float(brier_score_loss(y_true, y_proba_clipped))

    # Log loss: cross-entropy
    logloss = float(log_loss(y_true, y_proba_clipped))

    # Expected Calibration Error (ECE)
    # Bins predictions and computes avg |predicted prob - empirical freq|
    n_bins = 10
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    ece = float(np.mean(np.abs(frac_pos - mean_pred)))

    return {
        "brier": brier,
        "log_loss": logloss,
        "ece": ece,
    }


def plot_calibration_histogram(
    y_proba,
    n_bins: int = 20,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """Histogram of predicted probabilities.

    Shows distribution of model's confidence. Well-calibrated models
    should not have too many extreme predictions (near 0 or 1).

    Args:
        y_proba: 1D array-like of predicted probabilities
        n_bins: Number of histogram bins
        title: Optional plot title
        save_path: If provided, save PNG here

    Returns:
        Tuple of (fig, ax) matplotlib objects
    """
    y_proba = np.asarray(y_proba)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(y_proba, bins=n_bins, color='#A23B72', alpha=0.7,
            edgecolor='black', linewidth=1.2)

    ax.set_xlabel("Predicted probability", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title or "Distribution of Predicted Probabilities",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=1)

    # Add vertical lines at 0.5 (decision boundary)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2,
               alpha=0.5, label='Decision boundary (0.5)')
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"Saved probability histogram to {save_path}")

    return fig, ax
