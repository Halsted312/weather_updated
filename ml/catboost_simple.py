#!/usr/bin/env python3
"""
Simplified CatBoost implementation that works with existing infrastructure.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

logger = logging.getLogger(__name__)


def train_catboost_simple(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
    output_dir: str,
    tag: str,
    n_trials: int = 5,
    bracket_type: str = "between",
) -> Dict[str, Any]:
    """
    Train a simple CatBoost model with fixed hyperparameters.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        meta_test: Test metadata
        output_dir: Directory to save artifacts
        tag: Tag for naming files
        n_trials: Not used (for compatibility)
        bracket_type: Bracket type for monotonic constraints

    Returns:
        Dictionary with results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Simple hyperparameters (no Optuna for now)
    params = {
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 4,
        'l2_leaf_reg': 3.0,
        'random_state': 42,
        'verbose': False,
        'allow_writing_files': False,
        'thread_count': -1,
    }

    # Train model
    logger.info(f"Training CatBoost with fixed params: {params}")
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)

    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calibrate if we have enough data
    if len(np.unique(y_train)) > 1 and len(X_train) >= 1000:
        logger.info("Applying isotonic calibration")
        calibrator = IsotonicRegression(out_of_bounds='clip')
        train_proba = model.predict_proba(X_train)[:, 1]
        calibrator.fit(train_proba, y_train)
        y_pred_calibrated = calibrator.transform(y_pred_proba)
        calib_method = "isotonic"
    else:
        logger.info("Using sigmoid calibration (small dataset)")
        calibrator = LogisticRegression()
        train_proba = model.predict_proba(X_train)[:, 1].reshape(-1, 1)
        calibrator.fit(train_proba, y_train)
        y_pred_calibrated = calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
        calib_method = "sigmoid"

    # Calculate metrics
    test_brier = brier_score_loss(y_test, y_pred_calibrated)
    test_logloss = log_loss(y_test, y_pred_calibrated)

    logger.info(f"Test metrics - Brier: {test_brier:.4f}, LogLoss: {test_logloss:.4f}")

    # Save model
    model_path = os.path.join(output_dir, f"model_{tag}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'calibrator': calibrator}, f)

    # Save parameters
    params_path = os.path.join(output_dir, f"params_{tag}.json")
    with open(params_path, 'w') as f:
        json.dump({
            'params': params,
            'calib_method': calib_method,
            'test_brier': test_brier,
            'test_logloss': test_logloss,
        }, f, indent=2)

    # Save predictions
    preds_df = meta_test.copy()
    preds_df['p_model'] = y_pred_calibrated
    preds_df['y_true'] = y_test

    preds_path = os.path.join(output_dir, f"preds_{tag}.csv")
    preds_df.to_csv(preds_path, index=False)

    logger.info(f"Saved artifacts to {output_dir}")

    return {
        'best_params': params,
        'calib_method': calib_method,
        'test_metrics': {
            'brier': test_brier,
            'log_loss': test_logloss,
        }
    }