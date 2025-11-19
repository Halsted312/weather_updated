#!/usr/bin/env python3
"""
Logistic regression (Ridge/Lasso/ElasticNet) with Optuna hyperparameter tuning and probability calibration.

Tuning:
- Optuna 40 trials per window (Phase 5 spec)
- Search space: penalty ∈ {l2, l1, elasticnet}, C ∈ [1e-3, 1e3] (log scale),
  l1_ratio ∈ [0, 1] (for elasticnet), class_weight ∈ {None, balanced}
- CV: GroupKFold by event_date (4 splits)
- Objective: minimize mean Brier score across folds
- Pruning: MedianPruner to stop unpromising trials

Calibration:
- Method: isotonic if N_cal ≥ 1000, else sigmoid (Platt)
- Split: 80% model train, 20% calibration (GroupShuffleSplit by day)

References:
- sklearn calibration: https://scikit-learn.org/stable/modules/calibration.html
- Optuna: https://optuna.readthedocs.io/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
import joblib
import optuna

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class WindowData:
    """Container for train/test data for one walk-forward window."""
    X_train: np.ndarray
    y_train: np.ndarray
    groups_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta_test: pd.DataFrame  # timestamps, market_ticker, event_date


@dataclass
class TrainResult:
    """Results from training one window."""
    best_params: Dict
    calib_method: str
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    model_path: str
    preds_path: str
    params_path: str


def _logit_search_space(trial: optuna.Trial, penalties: List[str] = None) -> Dict:
    """
    Optuna search space for logistic regression (Ridge/Lasso/ElasticNet).

    Args:
        trial: Optuna trial
        penalties: List of penalties to search (default: ["l2", "l1", "elasticnet"])

    Returns:
        Dict with penalty, C, l1_ratio (if elasticnet), and class_weight parameters
    """
    # Penalty type: l2 (Ridge), l1 (Lasso), or elasticnet
    if penalties is None:
        penalties = ["l2", "l1", "elasticnet"]

    penalty = trial.suggest_categorical("penalty", penalties)

    # C is inverse regularization strength; search log scale
    C = trial.suggest_float("C", 1e-3, 1e+3, log=True)

    # l1_ratio only for elasticnet (0 = pure l2, 1 = pure l1)
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    else:
        l1_ratio = None

    # Class balancing
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    params = {"penalty": penalty, "C": C, "class_weight": class_weight}
    if l1_ratio is not None:
        params["l1_ratio"] = l1_ratio

    return params


def tune_logit_with_optuna(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 4,
    n_trials: int = 40,
    penalties: List[str] = None,
    seed: int = 42
) -> Dict:
    """
    Tune logistic regression (Ridge/Lasso/ElasticNet) with Optuna using GroupKFold CV.

    Args:
        X: Feature matrix (N x F)
        y: Binary labels (N,)
        groups: Day indices for GroupKFold (N,)
        n_splits: Number of CV folds (default: 4)
        n_trials: Number of Optuna trials (default: 40)
        penalties: List of penalties to search (default: ["l2", "l1", "elasticnet"])
        seed: Random seed

    Returns:
        Dict with best parameters (penalty, C, l1_ratio, class_weight)
    """
    gkf = GroupKFold(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = _logit_search_space(trial, penalties=penalties)

        # Extract penalty-specific params for LogisticRegression
        penalty = params.pop("penalty")
        l1_ratio = params.pop("l1_ratio", None)  # Only for elasticnet

        # Build LogisticRegression kwargs
        clf_kwargs = {
            "penalty": penalty,
            "solver": "saga",  # Supports all penalties (l2, l1, elasticnet)
            "max_iter": 5000,
            "random_state": seed,
            "C": params["C"],
            "class_weight": params["class_weight"],
        }

        # Add l1_ratio for elasticnet
        if penalty == "elasticnet" and l1_ratio is not None:
            clf_kwargs["l1_ratio"] = l1_ratio

        # Pipeline with standard scaling + logistic regression
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(**clf_kwargs)),
        ])

        # Cross-validation Brier score (proper scoring rule for probabilities)
        scores: List[float] = []
        for fold_idx, (train_idx, valid_idx) in enumerate(gkf.split(X, y, groups)):
            pipe.fit(X[train_idx], y[train_idx])
            p = pipe.predict_proba(X[valid_idx])[:, 1]
            brier = brier_score_loss(y[valid_idx], p)
            scores.append(brier)

            # Report intermediate value for pruning
            trial.report(brier, fold_idx)

            # Prune if unpromising
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    # Create study with MedianPruner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        direction="minimize",
        study_name="logit_multipenalty",
        pruner=pruner
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Best trial: brier={study.best_value:.4f}, params={study.best_params}")

    return study.best_params


def fit_logit_with_calibration(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    best_params: Dict,
    seed: int = 42
) -> tuple:
    """
    Fit logistic regression model (Ridge/Lasso/ElasticNet) with calibration.

    Split train data into model-train (80%) and calibration (20%) by day.
    Use isotonic calibration if N_cal ≥ 1000, else sigmoid (Platt).

    Args:
        X: Feature matrix
        y: Binary labels
        groups: Day indices
        best_params: Best params from Optuna (penalty, C, l1_ratio, class_weight)
        seed: Random seed

    Returns:
        Tuple of (calibrated_model, calib_method)
    """
    # Split train into model-train and calibration folds by day
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    (train_idx, calib_idx) = next(gss.split(X, y, groups))

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_cal, y_cal = X[calib_idx], y[calib_idx]

    # Extract penalty-specific params
    penalty = best_params.get("penalty", "l2")
    l1_ratio = best_params.get("l1_ratio")
    C = best_params.get("C", 1.0)
    class_weight = best_params.get("class_weight")

    # Build LogisticRegression kwargs
    clf_kwargs = {
        "penalty": penalty,
        "solver": "saga",
        "max_iter": 5000,
        "random_state": seed,
        "C": C,
        "class_weight": class_weight,
    }

    # Add l1_ratio for elasticnet
    if penalty == "elasticnet" and l1_ratio is not None:
        clf_kwargs["l1_ratio"] = l1_ratio

    # Fit base model
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**clf_kwargs))
    ])
    pipe.fit(X_tr, y_tr)

    # Choose calibration method based on calibration set size
    # sklearn guidance: isotonic needs many samples to avoid overfitting
    method = "isotonic" if len(y_cal) >= 1000 else "sigmoid"
    logger.info(f"Calibration: {method} (N_cal={len(y_cal)})")

    # Calibrate using prefit estimator
    calibrated = CalibratedClassifierCV(pipe, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)

    return calibrated, method


def evaluate_probs(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """
    Evaluate binary probability predictions.

    Handles single-class test sets gracefully by providing labels parameter
    to scoring functions.

    Args:
        y_true: True binary labels (N,)
        p: Predicted probabilities (N,)

    Returns:
        Dict with log_loss and brier_score
    """
    # Check for single-class test set
    unique_labels = np.unique(y_true)

    # For single-class sets, brier score is still well-defined
    # It measures calibration error from the true class
    if len(unique_labels) == 1:
        # Single-class: define metrics carefully
        # Brier score = mean squared error from true probability
        if unique_labels[0] == 0:
            # All negatives: brier = mean(p^2)
            brier = float(np.mean(p ** 2))
        else:
            # All positives: brier = mean((1-p)^2)
            brier = float(np.mean((1 - p) ** 2))
    else:
        # Normal two-class case
        brier = float(brier_score_loss(y_true, p))

    return {
        "log_loss": float(log_loss(y_true, p, labels=[0, 1], eps=1e-6)),
        "brier": brier,
    }


def blend_with_market(
    p_model: np.ndarray,
    p_market: np.ndarray,
    model_weight: float = 0.7,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Blend model and market probabilities via log-odds (opinion pooling).

    Formula:
        logit(p_blend) = w * logit(p_model) + (1 - w) * logit(p_market)

    where logit(p) = log(p / (1 - p))

    Args:
        p_model: Model probabilities (N,)
        p_market: Market probabilities (N,)
        model_weight: Weight for model (default: 0.7), market gets (1 - w)
        eps: Small epsilon to clip probabilities away from 0/1 (default: 1e-6)

    Returns:
        Blended probabilities (N,)

    References:
        - Log-odds pooling: https://en.wikipedia.org/wiki/Aggregation_of_forecasts
        - Opinion pooling: https://arxiv.org/abs/1506.05170
    """
    # Clip probabilities to avoid log(0) or log(∞)
    p_model = np.clip(p_model, eps, 1 - eps)
    p_market = np.clip(p_market, eps, 1 - eps)

    # Convert to log-odds
    logit_model = np.log(p_model / (1 - p_model))
    logit_market = np.log(p_market / (1 - p_market))

    # Blend in log-odds space
    logit_blend = model_weight * logit_model + (1 - model_weight) * logit_market

    # Convert back to probability via sigmoid
    p_blend = 1 / (1 + np.exp(-logit_blend))

    return p_blend


def train_one_window(
    win: WindowData,
    artifacts_dir: str,
    tag: str,
    n_trials: int = 40,
    penalties: List[str] = None,
    seed: int = 42
) -> TrainResult:
    """
    Train logistic regression model (Ridge/Lasso/ElasticNet) for one walk-forward window.

    Steps:
    1. Tune hyperparameters with Optuna
    2. Fit calibrated model
    3. Predict on test set
    4. Save artifacts: model (joblib), params (JSON), predictions (CSV)

    Args:
        win: WindowData with train/test splits
        artifacts_dir: Directory to save artifacts
        tag: Identifier for this window (e.g., "chicago_between_20250801_20251031")
        n_trials: Number of Optuna trials (default: 40)
        penalties: List of penalties to search (default: ["l2", "l1", "elasticnet"])
        seed: Random seed

    Returns:
        TrainResult with paths to saved artifacts and metrics
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Tune
    penalty_str = ", ".join(penalties) if penalties else "l2, l1, elasticnet"
    logger.info(f"Tuning logistic regression ({penalty_str}) with Optuna ({n_trials} trials)...")
    best = tune_logit_with_optuna(
        win.X_train, win.y_train, win.groups_train,
        n_trials=n_trials, penalties=penalties, seed=seed
    )

    # 2. Fit with calibration
    penalty_type = best.get("penalty", "l2")
    logger.info(f"Fitting calibrated {penalty_type} model...")
    model, calib_method = fit_logit_with_calibration(
        win.X_train, win.y_train, win.groups_train, best, seed=seed
    )

    # 3. Evaluate on test set
    logger.info("Evaluating on test set...")
    p_test = model.predict_proba(win.X_test)[:, 1]
    test_metrics = evaluate_probs(win.y_test, p_test)

    logger.info(f"Test metrics: log_loss={test_metrics['log_loss']:.4f}, "
                f"brier={test_metrics['brier']:.4f}")

    # 4. Save artifacts
    model_path = os.path.join(artifacts_dir, f"model_{tag}.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Saved model: {model_path}")

    params_path = os.path.join(artifacts_dir, f"params_{tag}.json")
    with open(params_path, "w") as f:
        json.dump({
            "best_params": best,
            "calibration": calib_method,
            "n_trials": n_trials,
            "seed": seed
        }, f, indent=2)
    logger.info(f"Saved params: {params_path}")

    # Save predictions with metadata
    preds = win.meta_test.copy()
    preds["p_model"] = p_test
    preds["y_true"] = win.y_test

    # Ensure bracket identification columns are present for unified head
    # (these should already be in meta_test from build_training_dataset)
    required_cols = ["floor_strike", "cap_strike", "bracket_key"]
    missing_cols = [col for col in required_cols if col not in preds.columns]
    if missing_cols:
        logger.warning(f"Missing columns for unified head: {missing_cols}")

    preds_path = os.path.join(artifacts_dir, f"preds_{tag}.csv")
    preds.to_csv(preds_path, index=False)
    logger.info(f"Saved predictions: {preds_path} ({len(preds)} rows, {len(preds.columns)} columns)")

    return TrainResult(
        best_params=best,
        calib_method=calib_method,
        train_metrics={"cv_metric": None},  # Could add CV metrics from tuning
        test_metrics=test_metrics,
        model_path=model_path,
        preds_path=preds_path,
        params_path=params_path,
    )


def main():
    """Demo: Train Ridge on synthetic data."""
    print("\n" + "="*60)
    print("Ridge Logistic + Optuna + Calibration Demo")
    print("="*60 + "\n")

    # Generate synthetic data
    np.random.seed(42)
    N = 2000
    X = np.random.randn(N, 10)
    y = (X[:, 0] + 0.5 * X[:, 1] + np.random.randn(N) * 0.5 > 0).astype(int)
    groups = np.repeat(np.arange(N // 50), 50)  # 50 samples per day

    # Split into train/test
    N_train = 1500
    win = WindowData(
        X_train=X[:N_train],
        y_train=y[:N_train],
        groups_train=groups[:N_train],
        X_test=X[N_train:],
        y_test=y[N_train:],
        meta_test=pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=N-N_train, freq="1min"),
            "market_ticker": ["TEST-MARKET"] * (N - N_train),
            "event_date": [pd.Timestamp("2025-01-01").date()] * (N - N_train),
        })
    )

    # Train
    result = train_one_window(
        win, artifacts_dir="/tmp/ridge_demo", tag="demo", n_trials=10
    )

    print(f"\nTraining complete:")
    print(f"  Best params: {result.best_params}")
    print(f"  Calibration: {result.calib_method}")
    print(f"  Test log loss: {result.test_metrics['log_loss']:.4f}")
    print(f"  Test Brier: {result.test_metrics['brier']:.4f}")
    print(f"  Artifacts saved to: /tmp/ridge_demo/")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
