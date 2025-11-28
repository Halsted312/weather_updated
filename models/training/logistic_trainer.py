"""
Model 1: Multinomial Logistic Δ-Model Trainer.

This module implements training for the logistic regression model with
elastic net regularization and Platt calibration. It's the simpler of
the two models but provides interpretable coefficients.

Model characteristics:
    - Multinomial logistic regression for Δ ∈ {-2, -1, 0, +1, +2}
    - Elastic net regularization (L1 + L2) for feature selection
    - Platt scaling via CalibratedClassifierCV for calibrated probabilities
    - Time-series CV to avoid lookahead

Usage:
    >>> from models.training.logistic_trainer import LogisticDeltaTrainer
    >>> trainer = LogisticDeltaTrainer(l1_ratio=0.5, C=1.0)
    >>> model = trainer.train(df_train)
    >>> proba = trainer.predict_proba(df_test)
    >>> trainer.save(Path('models/saved/logistic_v1.pkl'))
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class LogisticDeltaTrainer(BaseTrainer):
    """Trainer for multinomial logistic Δ-model.

    Uses elastic net regularization (penalty='elasticnet') which combines
    L1 (lasso) for sparsity and L2 (ridge) for stability. The l1_ratio
    parameter controls the mix.

    Attributes:
        l1_ratio: Mix of L1/L2 in elastic net (0=pure L2, 1=pure L1)
        C: Inverse regularization strength (lower = more regularization)
        max_iter: Maximum iterations for solver convergence
    """

    def __init__(
        self,
        l1_ratio: float = 0.5,
        C: float = 1.0,
        max_iter: int = 4000,
        include_forecast: bool = True,
        include_lags: bool = True,
        calibrate: bool = True,
        cv_splits: int = 5,
    ):
        """Initialize logistic trainer.

        Args:
            l1_ratio: Elastic net mixing parameter (0-1)
            C: Inverse regularization strength
            max_iter: Maximum solver iterations
            include_forecast: Whether to use forecast features
            include_lags: Whether to use lag features
            calibrate: Whether to apply Platt scaling
            cv_splits: Number of CV splits for calibration
        """
        super().__init__(
            include_forecast=include_forecast,
            include_lags=include_lags,
            calibrate=calibrate,
            cv_splits=cv_splits,
        )

        self.l1_ratio = l1_ratio
        self.C = C
        self.max_iter = max_iter

    def _create_base_model(self) -> LogisticRegression:
        """Create the base logistic regression classifier."""
        return LogisticRegression(
            solver="saga",  # Required for elasticnet penalty
            penalty="elasticnet",
            l1_ratio=self.l1_ratio,
            C=self.C,
            max_iter=self.max_iter,
            n_jobs=-1,
            random_state=42,
        )

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Extract coefficient magnitudes as feature importance.

        For multinomial logistic regression, coefficients have shape
        (n_classes, n_features). We use the mean absolute coefficient
        across classes as the importance score.

        Returns:
            DataFrame with feature names and importance scores,
            sorted by importance. None if model not trained.
        """
        if self.model is None:
            return None

        # Get the underlying LogisticRegression from calibrated wrapper
        if hasattr(self.model, "calibrated_classifiers_"):
            # CalibratedClassifierCV - get first calibrated classifier's base
            base = self.model.calibrated_classifiers_[0].estimator
        elif hasattr(self.model, "estimator"):
            base = self.model.estimator
        else:
            base = self.model

        # Navigate through pipeline to get classifier
        if hasattr(base, "named_steps"):
            classifier = base.named_steps.get("classifier")
            preprocessor = base.named_steps.get("preprocessor")
        else:
            return None

        if classifier is None or not hasattr(classifier, "coef_"):
            return None

        # Get coefficients
        coefs = classifier.coef_  # Shape: (n_classes, n_features)

        # Get feature names from preprocessor
        if preprocessor is not None:
            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception:
                # Fallback to numbered features
                n_features = coefs.shape[1]
                feature_names = [f"feature_{i}" for i in range(n_features)]
        else:
            n_features = coefs.shape[1]
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Mean absolute coefficient across classes
        importance = np.abs(coefs).mean(axis=0)

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        })

        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def get_coefficients_by_class(self) -> Optional[pd.DataFrame]:
        """Get coefficients for each delta class.

        Returns:
            DataFrame with features as rows and delta classes as columns,
            showing the coefficient for each feature-class combination.
        """
        if self.model is None:
            return None

        # Similar extraction as get_feature_importance
        if hasattr(self.model, "calibrated_classifiers_"):
            base = self.model.calibrated_classifiers_[0].estimator
        elif hasattr(self.model, "estimator"):
            base = self.model.estimator
        else:
            base = self.model

        if hasattr(base, "named_steps"):
            classifier = base.named_steps.get("classifier")
            preprocessor = base.named_steps.get("preprocessor")
        else:
            return None

        if classifier is None or not hasattr(classifier, "coef_"):
            return None

        coefs = classifier.coef_
        classes = classifier.classes_

        # Get feature names
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(coefs.shape[1])]

        # Create DataFrame
        df = pd.DataFrame(
            coefs.T,
            index=feature_names,
            columns=[f"delta_{c}" for c in classes],
        )

        return df


def train_logistic_model(
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
    **kwargs,
) -> tuple[LogisticDeltaTrainer, dict]:
    """Convenience function to train logistic model and evaluate.

    Args:
        df_train: Training DataFrame
        df_test: Optional test DataFrame for evaluation
        output_path: Optional path to save model
        **kwargs: Additional arguments for LogisticDeltaTrainer

    Returns:
        Tuple of (trainer, results_dict)
    """
    trainer = LogisticDeltaTrainer(**kwargs)
    trainer.train(df_train)

    results = {
        "model_type": "logistic",
        "n_train_samples": len(df_train),
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
