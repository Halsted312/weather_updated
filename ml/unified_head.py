"""
Unified Head Module for Coupling Bracket Probabilities

This module provides functions to couple six per-bracket binary probabilities
into a coherent multinomial distribution that sums to 1.

Main approaches:
1. Softmax renormalization over logits (with temperature)
2. Dirichlet temperature coupling
3. Future: Pairwise coupling implementation

The unified head ensures consistent probability distributions across the
six daily temperature brackets (less, 4×between, greater) for improved
Kelly sizing and risk management.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)


def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute logit (log-odds) with numerical stability.

    Args:
        p: Probability array
        eps: Small epsilon to avoid log(0)

    Returns:
        Log-odds array
    """
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)


def softmax_renorm(
    probs: np.ndarray,
    tau: float = 1.0,
    use_logit: bool = True
) -> np.ndarray:
    """
    Couple probabilities via softmax renormalization.

    This is the preferred method as it:
    1. Preserves relative ordering when tau=1
    2. Allows temperature tuning for calibration
    3. Works well with already-calibrated binary probabilities

    Args:
        probs: Array of shape (6,) with per-bracket probabilities
        tau: Temperature parameter (>0). Lower values increase confidence,
             higher values increase entropy. Default 1.0 preserves relative scales.
        use_logit: If True, apply softmax over log-odds (recommended).
                   If False, apply softmax over raw probabilities.

    Returns:
        Array of shape (6,) with coupled probabilities summing to 1
    """
    p = np.asarray(probs, dtype=float)

    if len(p) != 6:
        raise ValueError(f"Expected exactly 6 bracket probabilities, got {len(p)}")

    if use_logit:
        # Convert to log-odds, apply temperature, then softmax
        s = _safe_logit(p) / max(tau, 1e-6)
        s = s - np.max(s)  # Stabilize for numerical precision
        w = np.exp(s)
    else:
        # Apply softmax directly on probabilities (less theoretically sound)
        s = p / max(tau, 1e-6)
        w = np.maximum(s, 1e-12)

    # Normalize to sum to 1
    q = w / np.sum(w)

    return q


def dirichlet_temperature(
    probs: np.ndarray,
    alpha: float = 0.1
) -> np.ndarray:
    """
    Couple probabilities via Dirichlet-regularized mean.

    This adds a uniform prior with concentration alpha to smooth
    the distribution. Useful when some brackets have very low/high
    probabilities that need regularization.

    Args:
        probs: Array of shape (6,) with per-bracket probabilities
        alpha: Dirichlet concentration parameter (>0). Higher values
               add more uniform smoothing.

    Returns:
        Array of shape (6,) with coupled probabilities summing to 1
    """
    p = np.asarray(probs, dtype=float)

    if len(p) != 6:
        raise ValueError(f"Expected exactly 6 bracket probabilities, got {len(p)}")

    # Add Dirichlet prior (uniform with concentration alpha)
    a = np.full_like(p, fill_value=alpha, dtype=float)
    q = (p + a) / np.sum(p + a)

    return q


def couple_timestamp_rowset(
    df_rows: pd.DataFrame,
    p_col: str = "p_model",
    method: Literal["softmax", "dirichlet", "pairwise"] = "softmax",
    tau: float = 1.0,
    alpha: float = 0.1
) -> np.ndarray:
    """
    Couple a set of bracket predictions for a single timestamp.

    This is the main entry point for coupling. It takes a DataFrame
    with exactly 6 rows (one per bracket) and returns coupled probabilities
    in the same order.

    Args:
        df_rows: DataFrame with exactly 6 rows for one (event_date, timestamp)
        p_col: Column name containing probabilities to couple
        method: Coupling method to use
        tau: Temperature for softmax method
        alpha: Concentration for Dirichlet method

    Returns:
        Array of length 6 with coupled probabilities in row order

    Raises:
        ValueError: If not exactly 6 rows provided
    """
    if len(df_rows) != 6:
        raise ValueError(
            f"Expected exactly 6 bracket rows for coupling, got {len(df_rows)}. "
            f"Incomplete bracket sets should be skipped."
        )

    # Extract probabilities in row order
    probs = df_rows[p_col].to_numpy()

    # Apply coupling based on method
    if method == "softmax":
        q = softmax_renorm(probs, tau=tau, use_logit=True)
    elif method == "dirichlet":
        q = dirichlet_temperature(probs, alpha=alpha)
    elif method == "pairwise":
        # Future implementation: pairwise coupling following Wu et al. (2004)
        # For now, fall back to softmax
        logger.debug("Pairwise coupling not yet implemented, using softmax")
        q = softmax_renorm(probs, tau=tau, use_logit=True)
    else:
        raise ValueError(f"Unknown coupling method: {method}")

    # Validate output
    if not np.allclose(np.sum(q), 1.0, atol=1e-6):
        logger.warning(f"Coupled probabilities sum to {np.sum(q)}, normalizing")
        q = q / np.sum(q)

    return q


def validate_bracket_completeness(
    df: pd.DataFrame,
    group_cols: list = ["event_date", "timestamp"],
    bracket_col: str = "bracket_key"
) -> pd.DataFrame:
    """
    Check which timestamps have complete bracket coverage (all 6 brackets).

    Args:
        df: DataFrame with predictions
        group_cols: Columns to group by (typically event_date and timestamp)
        bracket_col: Column identifying brackets (should have 6 unique values)

    Returns:
        DataFrame with columns: group_cols + ['n_brackets', 'is_complete']
    """
    coverage = (
        df.groupby(group_cols)[bracket_col]
        .nunique()
        .reset_index()
        .rename(columns={bracket_col: "n_brackets"})
    )
    coverage["is_complete"] = coverage["n_brackets"] == 6

    incomplete = coverage[~coverage["is_complete"]]
    if len(incomplete) > 0:
        logger.warning(
            f"Found {len(incomplete)} incomplete timestamp groups "
            f"(missing brackets): {incomplete.head()}"
        )

    return coverage


def apply_unified_head(
    df: pd.DataFrame,
    group_cols: list = ["event_date", "timestamp"],
    p_col: str = "p_model",
    method: str = "softmax",
    tau: float = 1.0,
    alpha: float = 0.1,
    output_col: str = "p_unified"
) -> pd.DataFrame:
    """
    Apply unified head coupling to an entire DataFrame of predictions.

    This is a convenience function that:
    1. Groups by (event_date, timestamp)
    2. Applies coupling to each complete group
    3. Falls back to uncoupled probabilities for incomplete groups

    Args:
        df: DataFrame with predictions
        group_cols: Columns to group by
        p_col: Input probability column
        method: Coupling method
        tau: Temperature for softmax
        alpha: Concentration for Dirichlet
        output_col: Name for output column with coupled probabilities

    Returns:
        DataFrame with added output_col containing coupled probabilities
    """
    df = df.copy()

    def _couple_group(group):
        if len(group) != 6:
            # Incomplete group: keep original probabilities
            group[output_col] = group[p_col]
            group["coupling_status"] = "incomplete"
            return group

        try:
            # Apply coupling
            q = couple_timestamp_rowset(
                group,
                p_col=p_col,
                method=method,
                tau=tau,
                alpha=alpha
            )
            group[output_col] = q
            group["coupling_status"] = "coupled"
        except Exception as e:
            logger.warning(f"Coupling failed for group: {e}")
            group[output_col] = group[p_col]
            group["coupling_status"] = "failed"

        return group

    # Apply coupling by group
    df = df.groupby(group_cols, group_keys=False).apply(_couple_group)

    # Log statistics
    status_counts = df["coupling_status"].value_counts()
    logger.info(f"Coupling status: {status_counts.to_dict()}")

    return df


def compute_multiclass_metrics(
    df: pd.DataFrame,
    p_col: str = "p_unified",
    y_col: str = "y_true",
    group_cols: list = ["event_date", "timestamp"]
) -> dict:
    """
    Compute multiclass metrics for the 6-way distribution.

    For each (event_date, timestamp), we have 6 probabilities and
    exactly one true outcome (the bracket that was settled as YES).

    Args:
        df: DataFrame with coupled probabilities and true labels
        p_col: Column with coupled probabilities
        y_col: Column with true binary labels (0/1)
        group_cols: Columns defining unique predictions

    Returns:
        Dict with multiclass log-loss and ECE
    """
    # Group and compute metrics
    def _group_metrics(group):
        if len(group) != 6:
            return None

        # Get probabilities and true labels
        p = group[p_col].values
        y = group[y_col].values

        # Ensure exactly one positive (YES) label
        if np.sum(y) != 1:
            logger.warning(f"Group has {np.sum(y)} positive labels, expected 1")
            return None

        # Compute log-loss for the true class
        true_idx = np.where(y == 1)[0][0]
        log_loss = -np.log(np.clip(p[true_idx], 1e-12, 1 - 1e-12))

        return {
            "log_loss": log_loss,
            "max_p": np.max(p),
            "true_p": p[true_idx],
            "entropy": -np.sum(p * np.log(np.clip(p, 1e-12, 1)))
        }

    results = []
    for _, group in df.groupby(group_cols):
        metrics = _group_metrics(group)
        if metrics is not None:
            results.append(metrics)

    if not results:
        return {
            "multiclass_log_loss": None,
            "mean_true_p": None,
            "mean_max_p": None,
            "mean_entropy": None
        }

    metrics_df = pd.DataFrame(results)

    return {
        "multiclass_log_loss": float(metrics_df["log_loss"].mean()),
        "mean_true_p": float(metrics_df["true_p"].mean()),
        "mean_max_p": float(metrics_df["max_p"].mean()),
        "mean_entropy": float(metrics_df["entropy"].mean()),
        "n_complete_groups": len(metrics_df)
    }