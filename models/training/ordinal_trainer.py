"""
Ordinal Regression Δ-Model Trainer using All-Threshold Binary Classifiers.

This module implements ordinal regression for temperature delta prediction
by training K-1 binary classifiers for P(delta >= k) at each threshold.
This respects the natural ordering of delta classes.

Key advantages over multinomial classification:
    - Respects ordinal nature: -2 < -1 < 0 < ... < +10
    - Predictions naturally satisfy monotonicity (with post-processing)
    - Direct computation of P(delta >= k) for bracket probabilities
    - "Close" predictions are implicitly favored over "far" ones

Model architecture:
    - For thresholds k in {-1, 0, 1, ..., 10}: train binary P(delta >= k)
    - P(delta = k) = P(delta >= k) - P(delta >= k+1)
    - Supports any sklearn-compatible binary classifier as base

Usage:
    >>> from models.training.ordinal_trainer import OrdinalDeltaTrainer
    >>> trainer = OrdinalDeltaTrainer(base_model='catboost', n_trials=10)
    >>> model = trainer.train(df_train)
    >>> proba = trainer.predict_proba(df_test)  # Shape: (n, 13) for 13 delta classes
"""

import logging
from pathlib import Path
from typing import Any, Optional, Literal

import numpy as np
import pandas as pd
import joblib

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

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, RobustScaler
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class OrdinalDeltaTrainer(BaseTrainer):
    """Ordinal regression trainer using all-threshold binary classifiers.

    Trains K-1 binary classifiers where classifier k predicts P(delta >= k).
    The final class probabilities are computed as:
        P(delta = k) = P(delta >= k) - P(delta >= k+1)
        P(delta = min_class) = 1 - P(delta >= min_class + 1)
        P(delta = max_class) = P(delta >= max_class)

    Attributes:
        base_model: Type of base classifier ('catboost' or 'logistic')
        thresholds: List of thresholds for binary classifiers
        classifiers: Dict mapping threshold -> fitted classifier
    """

    def __init__(
        self,
        base_model: Literal['catboost', 'logistic'] = 'catboost',
        n_trials: int = 0,  # Optuna trials for CatBoost (0 = use defaults)
        include_forecast: bool = True,
        include_lags: bool = True,
        calibrate: bool = False,  # Ordinal doesn't use Platt in same way
        cv_splits: int = 3,
        catboost_params: Optional[dict] = None,
        verbose: bool = False,
    ):
        """Initialize ordinal trainer.

        Args:
            base_model: 'catboost' for CatBoostClassifier, 'logistic' for LogisticRegression
            n_trials: Optuna trials for hyperparameter tuning (CatBoost only)
            include_forecast: Whether to use forecast features
            include_lags: Whether to use lag features
            calibrate: Whether to calibrate individual threshold classifiers
            cv_splits: Number of CV splits for tuning
            catboost_params: Override CatBoost parameters (if base_model='catboost')
            verbose: Whether to show training progress
        """
        super().__init__(
            include_forecast=include_forecast,
            include_lags=include_lags,
            calibrate=calibrate,
            cv_splits=cv_splits,
        )

        self.base_model_type = base_model
        self.n_trials = n_trials
        self.verbose = verbose
        self.catboost_params = catboost_params or {}

        # Thresholds will be set dynamically during training based on actual data
        # (Some cities may have delta range [-2, +10], others [-1, +10])
        self.thresholds: list[int] = []
        self._min_delta: Optional[int] = None
        self._max_delta: Optional[int] = None
        self._delta_classes: Optional[list[int]] = None

        self.classifiers: dict[int, Any] = {}
        self._cat_features: list[int] = []
        self.best_params: dict = {}
        self.study: Optional[Any] = None

        if base_model == 'catboost' and not CATBOOST_AVAILABLE:
            raise ImportError("catboost required. Install with: pip install catboost")

    def _create_base_model(self) -> Any:
        """Create base binary classifier for one threshold."""
        if self.base_model_type == 'catboost':
            # Default params tuned for binary classification
            params = {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "depth": 6,
                "learning_rate": 0.05,
                "iterations": 300,
                "l2_leaf_reg": 3.0,
                "min_data_in_leaf": 10,
                "random_strength": 0.5,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 0.5,
                "random_seed": 42,
                "verbose": False,
            }
            # Override with Optuna best params if available
            if self.best_params:
                params.update(self.best_params)
            # Override with explicit catboost_params
            params.update(self.catboost_params)

            # Handle bootstrap-specific params (prevent conflicts)
            bootstrap_type = params.get("bootstrap_type", "Bayesian")
            if bootstrap_type != "Bayesian":
                # Remove bagging_temperature if not Bayesian
                params.pop("bagging_temperature", None)
            else:
                # Remove subsample if Bayesian
                params.pop("subsample", None)

            return CatBoostClassifier(**params)

        elif self.base_model_type == 'logistic':
            # Logistic with preprocessing pipeline
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, self.numeric_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                     self.categorical_cols),
                ],
                remainder="drop",
            )
            return Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(
                    solver="saga",
                    penalty="elasticnet",
                    l1_ratio=0.5,
                    C=1.0,
                    max_iter=2000,
                    random_state=42,
                )),
            ])

    def _tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        df: pd.DataFrame,
    ) -> None:
        """Run Optuna hyperparameter search for ordinal CatBoost.

        Optimizes params by training on a representative middle threshold (delta >= 1)
        and evaluating via cross-validation. The best params are used for all thresholds.
        """
        logger.info(f"Starting Optuna tuning with {self.n_trials} trials")

        cv = DayGroupedTimeSeriesSplit(n_splits=self.cv_splits)

        # Use middle threshold for tuning (delta >= 1 is representative)
        tuning_threshold = 1
        y_binary = (y >= tuning_threshold).astype(int)

        def objective(trial: "optuna.Trial") -> float:
            # Choose bootstrap type first (affects other params)
            bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"])

            params = {
                # Tree structure
                "depth": trial.suggest_int("depth", 4, 8),
                "iterations": trial.suggest_int("iterations", 150, 400),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "border_count": trial.suggest_int("border_count", 32, 128),
                # Regularization
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 30),
                "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
                "bootstrap_type": bootstrap_type,
                # Fixed params
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "random_seed": 42,
                "verbose": False,
            }

            # Bootstrap-specific params
            if bootstrap_type == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.0)
            else:  # Bernoulli
                params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)

            auc_scores = []
            for train_idx, val_idx in cv.split(df):
                X_tr = X.iloc[train_idx]
                y_tr = y_binary.iloc[train_idx]
                X_va = X.iloc[val_idx]
                y_va = y_binary.iloc[val_idx]

                # Skip if validation has only one class
                if len(y_va.unique()) < 2:
                    continue

                model = CatBoostClassifier(**params)
                train_pool = Pool(X_tr, y_tr, cat_features=self._cat_features)
                val_pool = Pool(X_va, y_va, cat_features=self._cat_features)

                model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30)

                # Use AUC as metric (higher is better)
                from sklearn.metrics import roc_auc_score
                proba = model.predict_proba(val_pool)[:, 1]
                auc = roc_auc_score(y_va, proba)
                auc_scores.append(auc)

            if not auc_scores:
                return 0.0
            return float(np.mean(auc_scores))

        # Run optimization (maximize AUC)
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
        )

        self.best_params = self.study.best_params
        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best AUC: {self.study.best_value:.4f}")

    def train(
        self,
        df_train: pd.DataFrame,
        df_val: Optional[pd.DataFrame] = None,
    ) -> Any:
        """Train ordinal model (K-1 binary classifiers).

        Args:
            df_train: Training DataFrame with features and 'delta' column
            df_val: Optional validation DataFrame

        Returns:
            Self (the trained model is stored in self.classifiers)
        """
        logger.info(f"Training ordinal model ({self.base_model_type}) on {len(df_train)} samples")

        X_train, y_train = self._prepare_features(df_train)

        # Dynamically compute thresholds from actual training data
        # This handles cities with different delta ranges (e.g., LA/Miami have no delta=-2)
        self._min_delta = int(y_train.min())  # -2 for Chicago/Philly, -1 for others
        self._max_delta = int(y_train.max())  # +10 for all
        self.thresholds = list(range(self._min_delta + 1, self._max_delta + 1))
        self._delta_classes = list(range(self._min_delta, self._max_delta + 1))

        logger.info(f"City delta range: [{self._min_delta}, {self._max_delta}]")
        logger.info(f"Training {len(self.thresholds)} threshold classifiers: {self.thresholds}")

        # Get categorical feature indices for CatBoost
        if self.base_model_type == 'catboost':
            self._cat_features = [
                X_train.columns.get_loc(c)
                for c in self.categorical_cols
                if c in X_train.columns
            ]

        # Store feature columns for later
        self._feature_cols = X_train.columns.tolist()

        # Run Optuna hyperparameter tuning if requested
        if self.n_trials > 0 and self.base_model_type == 'catboost':
            if not OPTUNA_AVAILABLE:
                logger.warning("optuna not available, using default parameters")
            else:
                self._tune_hyperparameters(X_train, y_train, df_train)

        # Train one classifier per threshold
        for k in self.thresholds:
            # Binary target: 1 if delta >= k, 0 otherwise
            y_binary = (y_train >= k).astype(int)

            # Check class balance
            pos_rate = y_binary.mean()

            # Handle extremely imbalanced thresholds with constant predictor
            if pos_rate > 0.995 or pos_rate < 0.005:
                logger.warning(f"Threshold {k}: extreme imbalance (pos_rate={pos_rate:.4f}), using constant predictor")
                self.classifiers[k] = {
                    "type": "constant",
                    "prob": float(pos_rate)
                }
                continue

            if pos_rate < 0.01 or pos_rate > 0.99:
                logger.warning(f"Threshold {k}: highly imbalanced ({pos_rate:.1%} positive)")

            if self.base_model_type == 'catboost':
                model = self._create_base_model()
                train_pool = Pool(X_train, y_binary, cat_features=self._cat_features)

                if df_val is not None:
                    X_val, y_val = self._prepare_features(df_val)
                    y_val_binary = (y_val >= k).astype(int)
                    val_pool = Pool(X_val, y_val_binary, cat_features=self._cat_features)
                    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
                else:
                    model.fit(train_pool)

            else:  # logistic
                model = self._create_base_model()
                model.fit(X_train, y_binary)

            self.classifiers[k] = model

            if self.verbose:
                logger.info(f"  Threshold {k:+d}: trained (pos_rate={pos_rate:.1%})")

        # Store as "model" for compatibility
        self.model = self

        # Store metadata
        self._metadata = {
            "model_type": f"ordinal_{self.base_model_type}",
            "trained_at": pd.Timestamp.now().isoformat(),
            "n_train_samples": len(df_train),
            "n_train_days": df_train["day"].nunique(),
            "delta_range": [self._min_delta, self._max_delta],  # City-specific range
            "delta_classes": self._delta_classes,  # City-specific classes
            "global_delta_classes": DELTA_CLASSES,  # For reference
            "thresholds": self.thresholds,
            "n_classifiers": len(self.classifiers),
            "n_constant_classifiers": sum(1 for c in self.classifiers.values() if isinstance(c, dict)),
            "base_model": self.base_model_type,
            "n_optuna_trials": self.n_trials,
            "best_params": self.best_params if self.best_params else None,
        }

        logger.info(f"Ordinal training complete: {len(self.classifiers)} classifiers")
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for each delta class.

        Uses the formula:
            P(delta = k) = P(delta >= k) - P(delta >= k+1)

        With monotonicity enforcement to handle classifier inconsistencies.
        Pads probabilities to global DELTA_CLASSES shape for cities with smaller delta ranges.

        Args:
            df: DataFrame with features

        Returns:
            Array of shape (n_samples, n_classes) with probabilities (padded to DELTA_CLASSES)
        """
        if not self.classifiers:
            raise ValueError("Model not trained. Call train() first.")

        X, _ = self._prepare_features(df)
        n_samples = len(X)

        # Compute probabilities for city-specific delta classes
        local_proba = self._compute_local_proba(X)

        # Pad to global DELTA_CLASSES shape
        global_proba = np.zeros((n_samples, len(DELTA_CLASSES)))

        for i, delta_val in enumerate(self._delta_classes):
            global_idx = DELTA_CLASSES.index(delta_val)
            global_proba[:, global_idx] = local_proba[:, i]

        # For missing classes (e.g., delta=-2 in LA/Miami), probability stays 0
        return global_proba

    def _compute_local_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute probabilities for city-specific delta classes.

        Args:
            X: Feature DataFrame

        Returns:
            Array of shape (n_samples, len(self._delta_classes))
        """
        n_samples = len(X)

        # Get P(delta >= k) for each threshold
        probs_ge = {}
        for k in self.thresholds:
            clf = self.classifiers[k]

            # Handle constant predictors
            if isinstance(clf, dict) and clf.get("type") == "constant":
                probs_ge[k] = np.full(n_samples, clf["prob"])
            elif self.base_model_type == 'catboost':
                pool = Pool(X, cat_features=self._cat_features)
                probs_ge[k] = clf.predict_proba(pool)[:, 1]  # P(class=1) = P(delta >= k)
            else:
                probs_ge[k] = clf.predict_proba(X)[:, 1]

        # Enforce monotonicity: P(delta >= k) >= P(delta >= k+1)
        # Process from highest threshold to lowest
        sorted_thresholds = sorted(self.thresholds, reverse=True)
        for i in range(len(sorted_thresholds) - 1):
            k_high = sorted_thresholds[i]
            k_low = sorted_thresholds[i + 1]
            # P(delta >= k_low) must be >= P(delta >= k_high)
            probs_ge[k_low] = np.maximum(probs_ge[k_low], probs_ge[k_high])

        # Convert cumulative probs to class probs
        # P(delta = k) = P(delta >= k) - P(delta >= k+1)
        local_proba = np.zeros((n_samples, len(self._delta_classes)))

        for i, delta_val in enumerate(self._delta_classes):
            if delta_val == self._min_delta:
                # P(delta = min) = 1 - P(delta >= min+1)
                local_proba[:, i] = 1.0 - probs_ge.get(delta_val + 1, 0)
            elif delta_val == self._max_delta:
                # P(delta = max) = P(delta >= max)
                local_proba[:, i] = probs_ge.get(delta_val, 1)
            else:
                # P(delta = k) = P(delta >= k) - P(delta >= k+1)
                local_proba[:, i] = probs_ge.get(delta_val, 1) - probs_ge.get(delta_val + 1, 0)

        # Clip to valid range and renormalize
        local_proba = np.clip(local_proba, 0, 1)
        row_sums = local_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1)  # Avoid division by zero
        local_proba = local_proba / row_sums

        return local_proba

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict delta classes (argmax of probabilities).

        Args:
            df: DataFrame with features

        Returns:
            Array of predicted delta values
        """
        proba = self.predict_proba(df)
        class_indices = proba.argmax(axis=1)
        return np.array([DELTA_CLASSES[i] for i in class_indices])

    def predict_cumulative(self, df: pd.DataFrame) -> dict[int, np.ndarray]:
        """Predict cumulative probabilities P(delta >= k) for each threshold.

        Useful for direct bracket probability computation.

        Args:
            df: DataFrame with features

        Returns:
            Dict mapping threshold k to array of P(delta >= k)
        """
        if not self.classifiers:
            raise ValueError("Model not trained. Call train() first.")

        X, _ = self._prepare_features(df)

        probs_ge = {}
        for k in self.thresholds:
            clf = self.classifiers[k]
            if self.base_model_type == 'catboost':
                pool = Pool(X, cat_features=self._cat_features)
                probs_ge[k] = clf.predict_proba(pool)[:, 1]
            else:
                probs_ge[k] = clf.predict_proba(X)[:, 1]

        # Add boundary: P(delta >= min_delta) = 1.0
        probs_ge[self._min_delta] = np.ones(len(X))

        return probs_ge

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get aggregated feature importance across all threshold classifiers.

        For CatBoost, averages feature importance across all K-1 classifiers.

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.classifiers or self.base_model_type != 'catboost':
            return None

        # Aggregate importance across classifiers
        all_importances = []
        for k, clf in self.classifiers.items():
            imp = clf.get_feature_importance()
            all_importances.append(imp)

        # Average across classifiers
        avg_importance = np.mean(all_importances, axis=0)

        df = pd.DataFrame({
            "feature": self._feature_cols,
            "importance": avg_importance,
        })

        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: Path) -> None:
        """Save trained ordinal model.

        Saves all threshold classifiers and metadata.

        Args:
            path: Path to save model (.pkl extension)
        """
        if not self.classifiers:
            raise ValueError("No model to save. Call train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Package everything needed for inference
        save_dict = {
            "classifiers": self.classifiers,
            "thresholds": self.thresholds,
            "base_model_type": self.base_model_type,
            "feature_cols": self._feature_cols,
            "cat_features": self._cat_features,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "best_params": self.best_params,
            "metadata": self._metadata,
        }

        joblib.dump(save_dict, path)
        logger.info(f"Saved ordinal model to {path}")

        # Save metadata JSON
        import json
        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def load(self, path: Path) -> "OrdinalDeltaTrainer":
        """Load a trained ordinal model.

        Args:
            path: Path to model file (.pkl)

        Returns:
            Self with loaded model
        """
        path = Path(path)
        save_dict = joblib.load(path)

        self.classifiers = save_dict["classifiers"]
        self.thresholds = save_dict["thresholds"]
        self.base_model_type = save_dict["base_model_type"]
        self._feature_cols = save_dict["feature_cols"]
        self._cat_features = save_dict["cat_features"]
        self.numeric_cols = save_dict["numeric_cols"]
        self.categorical_cols = save_dict["categorical_cols"]
        self.best_params = save_dict.get("best_params", {})
        self._metadata = save_dict.get("metadata", {})

        # Extract delta range from metadata (for models trained with new code)
        if "delta_range" in self._metadata:
            self._min_delta = self._metadata["delta_range"][0]
            self._max_delta = self._metadata["delta_range"][1]
            self._delta_classes = self._metadata["delta_classes"]
        else:
            # Legacy models: assume full range
            self._min_delta = min(DELTA_CLASSES)
            self._max_delta = max(DELTA_CLASSES)
            self._delta_classes = DELTA_CLASSES

        self.model = self
        logger.info(f"Loaded ordinal model from {path}")
        return self

    def get_classes(self) -> np.ndarray:
        """Get the delta classes the model predicts."""
        return np.array(DELTA_CLASSES)


def train_ordinal_model(
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
    base_model: str = 'catboost',
    **kwargs,
) -> tuple[OrdinalDeltaTrainer, dict]:
    """Convenience function to train ordinal model and evaluate.

    Args:
        df_train: Training DataFrame
        df_test: Optional test DataFrame for evaluation
        output_path: Optional path to save model
        base_model: 'catboost' or 'logistic'
        **kwargs: Additional arguments for OrdinalDeltaTrainer

    Returns:
        Tuple of (trainer, results_dict)
    """
    trainer = OrdinalDeltaTrainer(base_model=base_model, **kwargs)
    trainer.train(df_train)

    results = {
        "model_type": f"ordinal_{base_model}",
        "n_train_samples": len(df_train),
        "n_classifiers": len(trainer.thresholds),
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
