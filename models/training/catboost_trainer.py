"""
Model 2: CatBoost Δ-Model Trainer with Optuna Hyperparameter Tuning.

This module implements training for the CatBoost gradient boosting model
with Bayesian hyperparameter optimization via Optuna and Platt calibration.

Model characteristics:
    - CatBoostClassifier with MultiClass loss
    - Native categorical feature handling (city)
    - Optuna for hyperparameter tuning (learning_rate, depth, l2_leaf_reg, etc.)
    - Platt scaling for calibrated probabilities
    - Time-series CV to avoid lookahead

Usage:
    >>> from models.training.catboost_trainer import CatBoostDeltaTrainer
    >>> trainer = CatBoostDeltaTrainer(n_trials=10)
    >>> model = trainer.train(df_train)
    >>> proba = trainer.predict_proba(df_test)
    >>> trainer.save(Path('models/saved/catboost_v1.pkl'))

Note: Requires catboost and optuna packages:
    pip install catboost optuna
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from models.training.base_trainer import BaseTrainer
from models.features.base import DELTA_CLASSES, get_feature_columns
from models.data.splits import DayGroupedTimeSeriesSplit

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("catboost not installed. Install with: pip install catboost")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("optuna not installed. Install with: pip install optuna")


class CatBoostDeltaTrainer(BaseTrainer):
    """Trainer for CatBoost Δ-model with Optuna hyperparameter tuning.

    CatBoost handles categorical features natively and often outperforms
    other gradient boosting methods on tabular data. Optuna provides
    efficient Bayesian optimization for hyperparameters.

    Attributes:
        n_trials: Number of Optuna trials for hyperparameter search
        best_params: Best hyperparameters found (after tuning)
        study: Optuna study object (for analysis)
    """

    def __init__(
        self,
        n_trials: int = 10,
        include_forecast: bool = True,
        include_lags: bool = True,
        calibrate: bool = True,
        cv_splits: int = 3,
        verbose: bool = False,
    ):
        """Initialize CatBoost trainer.

        Args:
            n_trials: Number of Optuna optimization trials
            include_forecast: Whether to use forecast features
            include_lags: Whether to use lag features
            calibrate: Whether to apply Platt scaling
            cv_splits: Number of CV splits for tuning and calibration
            verbose: Whether to show training progress
        """
        super().__init__(
            include_forecast=include_forecast,
            include_lags=include_lags,
            calibrate=calibrate,
            cv_splits=cv_splits,
        )

        self.n_trials = n_trials
        self.verbose = verbose
        self.best_params = None
        self.study = None

        if not CATBOOST_AVAILABLE:
            raise ImportError("catboost is required. Install with: pip install catboost")

    def _create_base_model(self) -> "CatBoostClassifier":
        """Create CatBoost classifier with best params or defaults."""
        params = self.best_params or {
            "depth": 6,
            "learning_rate": 0.1,
            "iterations": 200,
            "l2_leaf_reg": 3.0,
        }

        return CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=42,
            verbose=self.verbose,
            **params,
        )

    def train(
        self,
        df_train: pd.DataFrame,
        df_val: Optional[pd.DataFrame] = None,
        tune_hyperparams: bool = True,
    ) -> Any:
        """Train CatBoost model with optional hyperparameter tuning.

        Args:
            df_train: Training DataFrame
            df_val: Validation DataFrame (optional)
            tune_hyperparams: Whether to run Optuna tuning first

        Returns:
            Trained model
        """
        logger.info(f"Training CatBoost on {len(df_train)} samples")

        X_train, y_train = self._prepare_features(df_train)

        # Run hyperparameter tuning if requested
        if tune_hyperparams and self.n_trials > 0:
            if not OPTUNA_AVAILABLE:
                logger.warning("optuna not available, using default parameters")
            else:
                self._tune_hyperparameters(X_train, y_train, df_train)

        # Train final model
        # CatBoost handles categoricals natively, so we use different approach
        return self._train_catboost(df_train, df_val)

    def _tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        df: pd.DataFrame,
    ) -> None:
        """Run Optuna hyperparameter search."""
        logger.info(f"Starting Optuna tuning with {self.n_trials} trials")

        cv = DayGroupedTimeSeriesSplit(n_splits=self.cv_splits)

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 8.0, log=True),
                "iterations": trial.suggest_int("iterations", 150, 400),
                "border_count": trial.suggest_int("border_count", 32, 128),
            }

            # Get categorical feature indices
            cat_features = [X.columns.get_loc(c) for c in self.categorical_cols if c in X.columns]

            losses = []
            for train_idx, val_idx in cv.split(df):
                X_tr = X.iloc[train_idx]
                y_tr = y.iloc[train_idx]
                X_va = X.iloc[val_idx]
                y_va = y.iloc[val_idx]

                model = CatBoostClassifier(
                    loss_function="MultiClass",
                    eval_metric="MultiClass",
                    random_seed=42,
                    verbose=False,
                    **params,
                )

                train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
                val_pool = Pool(X_va, y_va, cat_features=cat_features)

                model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

                # Compute validation loss
                proba = model.predict_proba(val_pool)
                y_true = y_va.values

                # Manual logloss calculation
                eps = 1e-15
                # Map y_true to class indices
                classes = model.classes_
                class_to_idx = {c: i for i, c in enumerate(classes)}
                y_idx = np.array([class_to_idx.get(y, 0) for y in y_true])

                p = proba[np.arange(len(y_true)), y_idx]
                p = np.clip(p, eps, 1 - eps)
                loss = -np.mean(np.log(p))
                losses.append(loss)

            return float(np.mean(losses))

        # Run optimization
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
        )

        self.best_params = self.study.best_params
        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best loss: {self.study.best_value:.4f}")

    def _train_catboost(
        self,
        df_train: pd.DataFrame,
        df_val: Optional[pd.DataFrame] = None,
    ) -> Any:
        """Train final CatBoost model with best parameters."""
        X_train, y_train = self._prepare_features(df_train)

        # Get categorical feature indices
        cat_features = [X_train.columns.get_loc(c)
                        for c in self.categorical_cols if c in X_train.columns]

        # Create model with best params
        model = self._create_base_model()

        # Create pools
        train_pool = Pool(X_train, y_train, cat_features=cat_features)

        if df_val is not None:
            X_val, y_val = self._prepare_features(df_val)
            val_pool = Pool(X_val, y_val, cat_features=cat_features)
            model.fit(train_pool, eval_set=val_pool)
        else:
            model.fit(train_pool)

        # Store for calibration
        self._cat_features = cat_features
        self._base_model = model

        if self.calibrate:
            # Wrap for Platt scaling
            self.model = CatBoostCalibratedWrapper(
                model, X_train.columns.tolist(), cat_features
            )
            # Fit calibration on training data
            cv = DayGroupedTimeSeriesSplit(n_splits=self.cv_splits)

            from sklearn.calibration import CalibratedClassifierCV
            calibrated = CalibratedClassifierCV(
                estimator=self.model,
                method="sigmoid",
                cv=list(cv.split(df_train)),
            )
            calibrated.fit(X_train, y_train)
            self.model = calibrated
        else:
            self.model = CatBoostCalibratedWrapper(
                model, X_train.columns.tolist(), cat_features
            )

        # Store metadata
        self._metadata = {
            "model_type": "catboost",
            "trained_at": pd.Timestamp.now().isoformat(),
            "n_train_samples": len(df_train),
            "n_train_days": df_train["day"].nunique(),
            "best_params": self.best_params,
            "delta_classes": DELTA_CLASSES,
            "calibrated": self.calibrate,
        }

        logger.info("CatBoost training complete")
        return self.model

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Extract CatBoost feature importances.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self._base_model is None:
            return None

        importances = self._base_model.get_feature_importance()

        # Get feature names
        X_cols = self.numeric_cols + self.categorical_cols
        feature_names = X_cols[:len(importances)]

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })

        return df.sort_values("importance", ascending=False).reset_index(drop=True)


class CatBoostCalibratedWrapper:
    """Wrapper to make CatBoost compatible with sklearn calibration.

    CatBoost uses Pool objects which aren't directly compatible with
    sklearn's CalibratedClassifierCV. This wrapper handles the conversion.
    """

    def __init__(
        self,
        model: "CatBoostClassifier",
        feature_cols: list[str],
        cat_features: list[int],
    ):
        self.model = model
        self.feature_cols = feature_cols
        self.cat_features = cat_features

    def fit(self, X, y):
        """Fit the underlying CatBoost model."""
        X_df = pd.DataFrame(X, columns=self.feature_cols)
        pool = Pool(X_df, y, cat_features=self.cat_features)
        self.model.fit(pool)
        return self

    def predict(self, X):
        """Predict class labels."""
        X_df = pd.DataFrame(X, columns=self.feature_cols)
        pool = Pool(X_df, cat_features=self.cat_features)
        return self.model.predict(pool).flatten()

    def predict_proba(self, X):
        """Predict class probabilities."""
        X_df = pd.DataFrame(X, columns=self.feature_cols)
        pool = Pool(X_df, cat_features=self.cat_features)
        return self.model.predict_proba(pool)

    @property
    def classes_(self):
        """Get class labels."""
        return self.model.classes_


def train_catboost_model(
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
    **kwargs,
) -> tuple[CatBoostDeltaTrainer, dict]:
    """Convenience function to train CatBoost model and evaluate.

    Args:
        df_train: Training DataFrame
        df_test: Optional test DataFrame for evaluation
        output_path: Optional path to save model
        **kwargs: Additional arguments for CatBoostDeltaTrainer

    Returns:
        Tuple of (trainer, results_dict)
    """
    trainer = CatBoostDeltaTrainer(**kwargs)
    trainer.train(df_train)

    results = {
        "model_type": "catboost",
        "n_train_samples": len(df_train),
        "best_params": trainer.best_params,
    }

    if df_test is not None:
        from models.evaluation.metrics import compute_delta_metrics

        y_pred = trainer.predict(df_test)
        y_true = df_test["delta"].values

        metrics = compute_delta_metrics(y_true, y_pred)
        results.update(metrics)

    if output_path is not None:
        trainer.save(output_path)

    return trainer, results
