"""
Abstract base class for model trainers.

This module defines the common interface and shared functionality for
training temperature Δ-models. Both logistic and CatBoost trainers
inherit from this base class.

Key responsibilities:
    - Define common training workflow
    - Handle Platt scaling calibration
    - Model persistence (save/load)
    - Feature preprocessing

Example:
    >>> class MyTrainer(BaseTrainer):
    ...     def _create_base_model(self):
    ...         return MyClassifier()
    ...
    >>> trainer = MyTrainer()
    >>> model = trainer.train(X_train, y_train)
    >>> trainer.save(model, 'my_model.pkl')
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models.features.base import DELTA_CLASSES, get_feature_columns
from models.data.splits import DayGroupedTimeSeriesSplit

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base class for Δ-model trainers.

    Provides common functionality for training, calibration, and persistence.
    Subclasses must implement _create_base_model().

    Attributes:
        numeric_cols: List of numeric feature columns
        categorical_cols: List of categorical feature columns
        model: Trained model (after calling train())
        preprocessor: Feature preprocessor pipeline
    """

    def __init__(
        self,
        include_forecast: bool = True,
        include_lags: bool = True,
        calibrate: bool = True,
        cv_splits: int = 5,
    ):
        """Initialize trainer.

        Args:
            include_forecast: Whether to use forecast features
            include_lags: Whether to use lag features
            calibrate: Whether to apply Platt scaling calibration
            cv_splits: Number of CV splits for calibration
        """
        self.include_forecast = include_forecast
        self.include_lags = include_lags
        self.calibrate = calibrate
        self.cv_splits = cv_splits

        self.numeric_cols, self.categorical_cols = get_feature_columns(
            include_forecast=include_forecast,
            include_lags=include_lags,
        )

        self.model = None
        self.preprocessor = None
        self._metadata = {}

    @abstractmethod
    def _create_base_model(self) -> Any:
        """Create the base classifier (implemented by subclasses)."""
        pass

    def _create_preprocessor(self) -> ColumnTransformer:
        """Create the feature preprocessing pipeline.

        Numeric features: StandardScaler (optional, some models don't need it)
        Categorical features: OneHotEncoder

        Returns:
            ColumnTransformer for preprocessing
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", self.numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 self.categorical_cols),
            ],
            remainder="drop",
        )
        return preprocessor

    def _prepare_features(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target from DataFrame.

        Args:
            df: Snapshot DataFrame with features and 'delta' column

        Returns:
            Tuple of (X DataFrame, y Series)
        """
        # Ensure all feature columns exist
        all_cols = self.numeric_cols + self.categorical_cols
        missing_cols = [c for c in all_cols if c not in df.columns]

        if missing_cols:
            logger.warning(f"Missing columns (will fill with NaN): {missing_cols}")
            for col in missing_cols:
                df[col] = np.nan

        X = df[all_cols].copy()
        y = df["delta"].copy()

        # Clip delta to valid range
        y = y.clip(min(DELTA_CLASSES), max(DELTA_CLASSES))

        return X, y

    def train(
        self,
        df_train: pd.DataFrame,
        df_val: Optional[pd.DataFrame] = None,
    ) -> Any:
        """Train the model.

        Args:
            df_train: Training DataFrame with features and 'delta'
            df_val: Optional validation DataFrame (for early stopping)

        Returns:
            Trained model (Pipeline or CalibratedClassifierCV)
        """
        logger.info(f"Training on {len(df_train)} samples")

        X_train, y_train = self._prepare_features(df_train)

        # Create preprocessor and base model
        self.preprocessor = self._create_preprocessor()
        base_model = self._create_base_model()

        # Create pipeline
        pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", base_model),
        ])

        if self.calibrate:
            # Wrap in CalibratedClassifierCV for Platt scaling
            cv = DayGroupedTimeSeriesSplit(n_splits=self.cv_splits)

            # Need to pass df_train for day grouping
            calibrated = CalibratedClassifierCV(
                estimator=pipeline,
                method="sigmoid",  # Platt scaling
                cv=cv.split(df_train),
            )
            calibrated.fit(X_train, y_train)
            self.model = calibrated
        else:
            pipeline.fit(X_train, y_train)
            self.model = pipeline

        # Store metadata
        self._metadata = {
            "trained_at": datetime.now().isoformat(),
            "n_train_samples": len(df_train),
            "n_train_days": df_train["day"].nunique(),
            "delta_classes": DELTA_CLASSES,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "include_forecast": self.include_forecast,
            "include_lags": self.include_lags,
            "calibrated": self.calibrate,
        }

        logger.info("Training complete")
        return self.model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict delta classes.

        Args:
            df: DataFrame with features

        Returns:
            Array of predicted delta values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X, _ = self._prepare_features(df)
        return self.model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for each delta class.

        Args:
            df: DataFrame with features

        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X, _ = self._prepare_features(df)
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        """Save trained model with metadata.

        Saves both the model pickle and a JSON metadata file.

        Args:
            path: Path to save model (.pkl extension)
        """
        if self.model is None:
            raise ValueError("No model to save. Call train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self.model, path)
        logger.info(f"Saved model to {path}")

        # Save metadata
        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {metadata_path}")

    def load(self, path: Path) -> Any:
        """Load a trained model.

        Args:
            path: Path to model file (.pkl)

        Returns:
            Loaded model
        """
        path = Path(path)
        self.model = joblib.load(path)
        logger.info(f"Loaded model from {path}")

        # Try to load metadata
        metadata_path = path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self._metadata = json.load(f)

        return self.model

    def get_classes(self) -> np.ndarray:
        """Get the delta classes the model predicts."""
        if self.model is None:
            return np.array(DELTA_CLASSES)

        if hasattr(self.model, "classes_"):
            return self.model.classes_
        elif hasattr(self.model, "estimator") and hasattr(self.model.estimator, "classes_"):
            return self.model.estimator.classes_
        else:
            return np.array(DELTA_CLASSES)
