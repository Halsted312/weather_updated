#!/usr/bin/env python3
"""
CatBoost model with Optuna hyperparameter tuning for Kalshi weather prediction.

Includes monotonic constraints based on physical reasoning and calibration
for well-calibrated probability outputs.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
import optuna
from optuna_integration import CatBoostPruningCallback

logger = logging.getLogger(__name__)

# Suppress CatBoost verbose output during CV
warnings.filterwarnings("ignore", category=UserWarning, module="catboost")


def build_monotone_constraints(
    feature_cols: List[str],
    bracket_type: str,
    use_temp_to_floor: bool = True,
    use_temp_to_cap: bool = True,
    use_spread: bool = False,
    minutes_constraint: int = 0
) -> List[int]:
    """
    Build monotone constraints vector for CatBoost.

    Args:
        feature_cols: List of feature names (in order)
        bracket_type: "greater", "less", or "between"
        use_temp_to_floor: Whether to constrain temp_to_floor
        use_temp_to_cap: Whether to constrain temp_to_cap
        use_spread: Whether to constrain spread_cents
        minutes_constraint: Constraint for minutes_to_close (-1, 0, or 1)

    Returns:
        List of integers: 1 (monotone increase), -1 (decrease), 0 (no constraint)
    """
    constraints = []

    for col in feature_cols:
        if col == "temp_to_floor" and bracket_type in ["greater", "between"] and use_temp_to_floor:
            # P(T ≥ floor) increases as current temp gets closer to floor
            # temp_to_floor is negative when below floor, positive when above
            # So we want positive constraint
            constraints.append(1)
        elif col == "temp_to_cap" and bracket_type in ["less", "between"] and use_temp_to_cap:
            # P(T ≤ cap) increases as current temp is farther below cap
            # temp_to_cap is positive when below cap, negative when above
            # So we want positive constraint
            constraints.append(1)
        elif col == "spread_cents" and use_spread:
            # Wider spread → less liquidity → less confidence
            constraints.append(-1)
        elif col in ["minutes_to_close", "log_minutes_to_close"]:
            # More time → less certainty (but test empirically)
            constraints.append(minutes_constraint)
        else:
            constraints.append(0)

    return constraints


def _catboost_search_space(trial: optuna.Trial, bracket_type: str) -> Dict[str, Any]:
    """
    Define CatBoost hyperparameter search space for Optuna.

    Args:
        trial: Optuna trial object
        bracket_type: "greater", "less", or "between"

    Returns:
        Dictionary of hyperparameters
    """
    params = {
        # Tree structure
        "depth": trial.suggest_int("depth", 3, 8),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),

        # Regularization
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),

        # Learning
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "iterations": trial.suggest_int("iterations", 100, 1000),

        # Monotone constraint flags
        "use_temp_to_floor": trial.suggest_categorical("use_temp_to_floor", [True, False]),
        "use_temp_to_cap": trial.suggest_categorical("use_temp_to_cap", [True, False]),
        "use_spread": trial.suggest_categorical("use_spread", [True, False]),
        "minutes_constraint": trial.suggest_categorical("minutes_constraint", [-1, 0, 1]),

        # Fixed parameters
        "task_type": "CPU",
        "verbose": False,
        "random_state": 42,
        "eval_metric": "Logloss",
        "early_stopping_rounds": 50,
        "od_type": "Iter",  # Overfitting detector
        "use_best_model": True,
    }

    return params


def tune_catboost_with_optuna(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    bracket_type: str,
    feature_names: List[str],
    n_splits: int = 4,
    n_trials: int = 40,
    seed: int = 42
) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Tune CatBoost hyperparameters using Optuna with GroupKFold CV.

    Args:
        X: Feature matrix
        y: Binary labels
        groups: Group labels for GroupKFold (event_date)
        bracket_type: "greater", "less", or "between"
        feature_names: List of feature column names
        n_splits: Number of CV splits
        n_trials: Number of Optuna trials
        seed: Random seed

    Returns:
        Tuple of (best_params, study)
    """
    logger.info(f"Starting Optuna hyperparameter tuning with {n_trials} trials...")

    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna to minimize (Brier score)."""
        params = _catboost_search_space(trial, bracket_type)

        # Extract monotone constraint parameters
        use_temp_to_floor = params.pop("use_temp_to_floor")
        use_temp_to_cap = params.pop("use_temp_to_cap")
        use_spread = params.pop("use_spread")
        minutes_constraint = params.pop("minutes_constraint")

        # Build monotone constraints vector
        monotone_constraints = build_monotone_constraints(
            feature_names,
            bracket_type,
            use_temp_to_floor,
            use_temp_to_cap,
            use_spread,
            minutes_constraint
        )

        # Prepare for cross-validation
        gkf = GroupKFold(n_splits=n_splits)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create CatBoost pools
            train_pool = Pool(X_train, y_train)
            val_pool = Pool(X_val, y_val)

            # Train model with pruning callback
            model = CatBoostClassifier(
                **params,
                monotone_constraints=monotone_constraints,
            )

            # Add pruning callback for early stopping of bad trials
            pruning_callback = CatBoostPruningCallback(trial, metric="Logloss")

            try:
                model.fit(
                    train_pool,
                    eval_set=val_pool,
                    callbacks=[pruning_callback],
                    verbose=False,
                )
            except optuna.TrialPruned:
                raise

            # Get predictions and calculate Brier score
            y_pred = model.predict_proba(X_val)[:, 1]
            brier = brier_score_loss(y_val, y_pred)
            cv_scores.append(brier)

            # Report intermediate value for pruning
            trial.report(brier, fold)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(cv_scores)

    # Create study and optimize
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best Brier score: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study.best_params, study


def fit_catboost_with_calibration(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    best_params: Dict[str, Any],
    bracket_type: str,
    feature_names: List[str],
    calib_fraction: float = 0.2,
    seed: int = 42
) -> Tuple[Any, str, Dict[str, Any]]:
    """
    Fit CatBoost with best parameters and apply calibration.

    Args:
        X: Feature matrix
        y: Binary labels
        groups: Group labels (event_date) for train/cal split
        best_params: Best hyperparameters from Optuna
        bracket_type: "greater", "less", or "between"
        feature_names: List of feature column names
        calib_fraction: Fraction of data to use for calibration
        seed: Random seed

    Returns:
        Tuple of (calibrated_model, calib_method, metadata)
    """
    # Prepare parameters
    params = best_params.copy()

    # Extract and apply monotone constraints
    use_temp_to_floor = params.pop("use_temp_to_floor", True)
    use_temp_to_cap = params.pop("use_temp_to_cap", True)
    use_spread = params.pop("use_spread", False)
    minutes_constraint = params.pop("minutes_constraint", 0)

    monotone_constraints = build_monotone_constraints(
        feature_names,
        bracket_type,
        use_temp_to_floor,
        use_temp_to_cap,
        use_spread,
        minutes_constraint
    )

    # Add fixed parameters if not present
    if "task_type" not in params:
        params["task_type"] = "CPU"
    if "verbose" not in params:
        params["verbose"] = False
    if "random_state" not in params:
        params["random_state"] = seed
    if "eval_metric" not in params:
        params["eval_metric"] = "Logloss"

    # Split into train and calibration sets by day
    unique_groups = np.unique(groups)
    np.random.seed(seed)
    np.random.shuffle(unique_groups)

    n_cal_groups = max(1, int(len(unique_groups) * calib_fraction))
    cal_groups = unique_groups[:n_cal_groups]
    train_groups = unique_groups[n_cal_groups:]

    train_mask = np.isin(groups, train_groups)
    cal_mask = np.isin(groups, cal_groups)

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_cal = X[cal_mask]
    y_cal = y[cal_mask]

    logger.info(f"Train size: {len(X_train)}, Calibration size: {len(X_cal)}")

    # Train CatBoost model
    model = CatBoostClassifier(
        **params,
        monotone_constraints=monotone_constraints,
    )

    train_pool = Pool(X_train, y_train)

    # If we have early stopping, use validation set
    if "early_stopping_rounds" in params and params["early_stopping_rounds"] > 0 and len(X_cal) > 0:
        val_pool = Pool(X_cal, y_cal)
        model.fit(train_pool, eval_set=val_pool, verbose=False)
    else:
        model.fit(train_pool, verbose=False)

    # Apply calibration
    if len(X_cal) > 0:
        # Choose calibration method based on sample size
        if len(X_cal) >= 1000:
            calib_method = "isotonic"
            method = "isotonic"
        else:
            calib_method = "sigmoid"
            method = "sigmoid"

        logger.info(f"Applying {calib_method} calibration with {len(X_cal)} samples")

        calibrated_model = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv="prefit"  # We've already split the data
        )
        calibrated_model.fit(X_cal, y_cal)
    else:
        logger.warning("No calibration data available, using uncalibrated model")
        calibrated_model = model
        calib_method = "none"

    # Calculate feature importances
    feature_importance = model.get_feature_importance()
    importance_dict = dict(zip(feature_names, feature_importance))

    metadata = {
        "model_type": "catboost",
        "bracket_type": bracket_type,
        "n_train": len(X_train),
        "n_cal": len(X_cal),
        "calib_method": calib_method,
        "feature_importance": importance_dict,
        "monotone_constraints": dict(zip(feature_names, monotone_constraints)),
    }

    return calibrated_model, calib_method, metadata


def train_one_window_catboost(
    win_dir: Path,
    city: str,
    bracket: str,
    feature_set: str = "elasticnet_rich",
    n_trials: int = 40,
    seed: int = 42,
    max_minutes_to_close: Optional[float] = None,
    prior_peak_back_minutes: Optional[float] = None,
    prior_peak_lookup_days: int = 3,
    prior_peak_default_minutes: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Train CatBoost model for one walk-forward window.

    Args:
        win_dir: Directory for this window's artifacts
        city: City name
        bracket: Bracket type ("greater", "less", "between")
        feature_set: Feature set to use
        n_trials: Number of Optuna trials
        seed: Random seed

    Returns:
        Dictionary with training results and metrics
    """
    from ml.dataset import build_training_dataset

    # Extract window dates from directory name
    import re
    win_match = re.match(r"win_(\d{8})_(\d{8})", win_dir.name)
    if not win_match:
        raise ValueError(f"Invalid window directory name: {win_dir.name}")

    train_start = pd.to_datetime(win_match.group(1), format="%Y%m%d").date()
    test_end = pd.to_datetime(win_match.group(2), format="%Y%m%d").date()

    # Infer train_end and test_start (assuming 90-day train, 7-day test)
    from datetime import timedelta
    test_start = test_end - timedelta(days=6)
    train_end = test_start - timedelta(days=1)

    logger.info(f"Training CatBoost for {city}/{bracket} window {train_start} to {test_end}")

    # Load training data
    train_tuple = build_training_dataset(
        city=city,
        start_date=train_start,
        end_date=train_end,
        bracket_type=bracket,
        feature_set=feature_set,
        return_feature_names=True,
        max_minutes_to_close=max_minutes_to_close,
        prior_peak_back_minutes=prior_peak_back_minutes,
        prior_peak_lookup_days=prior_peak_lookup_days,
        prior_peak_default_minutes=prior_peak_default_minutes,
    )

    X_train_np, y_train, groups_train, train_meta, feature_cols = train_tuple

    if len(X_train_np) == 0:
        logger.error(f"No training data for window {win_dir.name}")
        return {"status": "failed", "reason": "no_training_data"}

    # Load test data
    test_tuple = build_training_dataset(
        city=city,
        start_date=test_start,
        end_date=test_end,
        bracket_type=bracket,
        feature_set=feature_set,
        return_feature_names=True,
        max_minutes_to_close=max_minutes_to_close,
        prior_peak_back_minutes=prior_peak_back_minutes,
        prior_peak_lookup_days=prior_peak_lookup_days,
        prior_peak_default_minutes=prior_peak_default_minutes,
    )

    X_test_np, y_test, _, test_meta, feature_cols_test = test_tuple

    if len(X_test_np) == 0:
        logger.warning(f"No test data for window {win_dir.name}")
        return {"status": "failed", "reason": "no_test_data"}

    if feature_cols != feature_cols_test:
        logger.warning("Feature columns mismatch between train and test; aligning by training columns")
    feature_cols = feature_cols

    # Handle single-class case
    unique_train = np.unique(y_train)
    if len(unique_train) == 1:
        logger.warning(f"Single class in training data: {unique_train[0]}")
        preds_df = test_meta.copy()
        preds_df["p_model"] = float(unique_train[0])
        preds_df["y_true"] = y_test

        pred_cols = ["market_ticker", "timestamp", "event_date", "bracket_type",
                     "bracket_key", "floor_strike", "cap_strike", "p_model", "y_true"]
        preds_df[pred_cols].to_csv(win_dir / f"preds_catboost_{city}_{bracket}_{win_dir.name}.csv", index=False)
        return {
            "status": "single_class",
            "train_class": int(unique_train[0]),
            "n_train": len(X_train_np),
            "n_test": len(X_test_np),
        }

    X_train = pd.DataFrame(X_train_np, columns=feature_cols)
    X_test = pd.DataFrame(X_test_np, columns=feature_cols)

    # Tune hyperparameters with Optuna
    best_params, study = tune_catboost_with_optuna(
        X_train, y_train, groups_train, bracket, feature_cols,
        n_splits=4, n_trials=n_trials, seed=seed
    )

    # Fit model with calibration
    calibrated_model, calib_method, metadata = fit_catboost_with_calibration(
        X_train, y_train, groups_train, best_params, bracket, feature_cols,
        calib_fraction=0.2, seed=seed
    )

    # Generate predictions on test set
    y_pred = calibrated_model.predict_proba(X_test)[:, 1]
    preds_df = test_meta.copy()
    preds_df["p_model"] = y_pred
    preds_df["y_true"] = y_test

    # Calculate metrics
    brier_score = brier_score_loss(y_test, y_pred)
    unique_test = np.unique(y_test)
    if len(unique_test) < 2:
        logger.warning("Single class in test labels; log_loss undefined (setting to NaN)")
        logloss = float("nan")
    else:
        logloss = log_loss(y_test, y_pred)

    # Save artifacts
    model_path = win_dir / f"model_catboost_{city}_{bracket}_{win_dir.name}.pkl"
    params_path = win_dir / f"params_catboost_{city}_{bracket}_{win_dir.name}.json"
    preds_path = win_dir / f"preds_catboost_{city}_{bracket}_{win_dir.name}.csv"

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(calibrated_model, f)

    # Save parameters and metadata
    save_params = {
        "best_params": best_params,
        "metadata": metadata,
        "metrics": {
            "brier_score": float(brier_score),
            "log_loss": float(logloss),
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        "optuna": {
            "n_trials": len(study.trials),
            "best_value": study.best_value,
        }
    }

    with open(params_path, "w") as f:
        json.dump(save_params, f, indent=2, default=str)

    # Save predictions
    pred_cols = ["market_ticker", "timestamp", "event_date", "bracket_type",
                 "bracket_key", "floor_strike", "cap_strike", "p_model", "y_true"]

    # Add missing columns if necessary
    if "bracket_type" not in preds_df.columns:
        preds_df["bracket_type"] = bracket
    if "bracket_key" not in preds_df.columns:
        # Construct bracket_key from floor/cap strikes
        if bracket == "greater":
            preds_df["bracket_key"] = "greater_" + preds_df["floor_strike"].astype(str)
        elif bracket == "less":
            preds_df["bracket_key"] = "less_" + preds_df["cap_strike"].astype(str)
        else:  # between
            preds_df["bracket_key"] = "between_" + preds_df["floor_strike"].astype(str) + "_" + preds_df["cap_strike"].astype(str)

    preds_df[pred_cols].to_csv(preds_path, index=False)

    logger.info(f"CatBoost training complete. Brier: {brier_score:.4f}, LogLoss: {logloss:.4f}")

    return {
        "status": "success",
        "metrics": save_params["metrics"],
        "best_params": best_params,
        "calib_method": calib_method,
        "model_path": str(model_path),
        "preds_path": str(preds_path),
    }


def main():
    """Demo of CatBoost model training."""
    print("CatBoost Model Demo")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Add temperature-related features
    X['temp_to_floor'] = np.random.randn(n_samples) * 5
    X['temp_to_cap'] = np.random.randn(n_samples) * 5
    X['spread_cents'] = np.random.uniform(1, 5, n_samples)
    X['minutes_to_close'] = np.random.uniform(0, 1440, n_samples)

    # Binary target (correlated with features)
    y = ((X['feature_0'] + X['feature_1'] + np.random.randn(n_samples) * 0.5) > 0).astype(int)

    # Groups for CV (simulate event dates)
    groups = np.repeat(np.arange(20), n_samples // 20)[:n_samples]

    feature_names = list(X.columns)

    print(f"Data shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print()

    # Test monotone constraint building
    print("Testing monotone constraint building...")
    constraints_greater = build_monotone_constraints(
        feature_names, "greater", use_spread=True, minutes_constraint=-1
    )
    print(f"Greater bracket constraints: {dict(zip(feature_names[-4:], constraints_greater[-4:]))}")
    print()

    # Tune with Optuna (reduced trials for demo)
    print("Tuning hyperparameters with Optuna...")
    best_params, study = tune_catboost_with_optuna(
        X, y, groups, "greater", feature_names, n_splits=3, n_trials=5, seed=42
    )
    print(f"Best params: {best_params}")
    print()

    # Fit with calibration
    print("Fitting CatBoost with calibration...")
    calibrated_model, calib_method, metadata = fit_catboost_with_calibration(
        X, y, groups, best_params, "greater", feature_names, calib_fraction=0.2, seed=42
    )
    print(f"Calibration method: {calib_method}")
    print(f"Top feature importances:")
    for feat, imp in sorted(metadata["feature_importance"].items(), key=lambda x: -x[1])[:5]:
        print(f"  {feat}: {imp:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
