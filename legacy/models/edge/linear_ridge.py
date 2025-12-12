"""Ridge (L2 regularization) edge classifier.

Uses all features with L2 penalty for stability.
Very fast training (~5 seconds), high interpretability, no feature selection.

Usage:
    from models.edge.linear_ridge import LinearRidgeEdgeClassifier

    model = LinearRidgeEdgeClassifier()
    model.fit(X_train, y_train, feature_names=features)
    probs = model.predict_proba(X_test)
    importance = model.get_feature_importance()
"""

from pathlib import Path
from typing import Optional, List
import joblib
import numpy as np
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.preprocessing import StandardScaler


class LinearRidgeEdgeClassifier:
    """Ridge (L2) model for edge classification.

    Attributes:
        alphas: Regularization strengths to try
        cv: Number of cross-validation folds
    """

    def __init__(
        self,
        alphas: Optional[np.ndarray] = None,
        cv: int = 5,
        use_classifier: bool = False,
    ):
        """Initialize Ridge classifier.

        Args:
            alphas: Regularization strengths (default: 20 values from 0.01 to 100)
            cv: Cross-validation folds
            use_classifier: Use RidgeClassifierCV instead of RidgeCV (faster but less flexible)
        """
        self.alphas = alphas if alphas is not None else np.logspace(-2, 2, 20)
        self.cv = cv
        self.use_classifier = use_classifier

        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.best_alpha_ = None

    def fit(self, X, y, feature_names: Optional[List[str]] = None):
        """Train model with cross-validation.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary target (0/1)
            feature_names: Names of features

        Returns:
            self
        """
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feat_{i}" for i in range(X.shape[1])]

        # Convert to numpy if DataFrame
        if hasattr(X, 'values'):
            X = X.values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train with cross-validation
        if self.use_classifier:
            self.model = RidgeClassifierCV(
                alphas=self.alphas,
                cv=self.cv,
            )
        else:
            self.model = RidgeCV(
                alphas=self.alphas,
                cv=self.cv,
            )

        self.model.fit(X_scaled, y)

        # Store best parameter
        self.best_alpha_ = self.model.alpha_

        return self

    def predict_proba(self, X):
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if hasattr(X, 'values'):
            X = X.values

        X_scaled = self.scaler.transform(X)

        if self.use_classifier:
            # RidgeClassifier doesn't have predict_proba, use decision_function
            scores = self.model.decision_function(X_scaled)
            # Convert to probabilities using sigmoid
            probs = 1 / (1 + np.exp(-scores))
        else:
            preds = self.model.predict(X_scaled)
            probs = np.clip(preds, 0, 1)

        return np.column_stack([1 - probs, probs])

    def predict(self, X, threshold: float = 0.5):
        """Predict binary classes."""
        return (self.predict_proba(X)[:, 1] > threshold).astype(int)

    def get_feature_importance(self):
        """Get feature importance (absolute coefficients)."""
        if self.model is None:
            return {}

        coefs = np.abs(self.model.coef_)
        return dict(zip(self.feature_names, coefs))

    def get_feature_coefficients(self):
        """Get raw coefficients with sign."""
        if self.model is None:
            return {}

        return dict(zip(self.feature_names, self.model.coef_))

    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'alpha': self.best_alpha_,
            'params': {
                'alphas': self.alphas.tolist() if isinstance(self.alphas, np.ndarray) else self.alphas,
                'cv': self.cv,
                'use_classifier': self.use_classifier,
            }
        }, path)

    @classmethod
    def load(cls, path: Path):
        """Load model from disk."""
        data = joblib.load(path)

        instance = cls(
            alphas=np.array(data['params']['alphas']),
            cv=data['params']['cv'],
            use_classifier=data['params']['use_classifier'],
        )

        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.best_alpha_ = data['alpha']

        return instance

    def __repr__(self):
        if self.model is None:
            return "LinearRidgeEdgeClassifier(untrained)"
        return f"LinearRidgeEdgeClassifier(alpha={self.best_alpha_:.4f})"
