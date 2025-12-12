"""Lasso (L1 regularization) edge classifier.

Performs automatic feature selection by driving weak feature coefficients to zero.
Very fast training (~5-10 seconds), highest interpretability (sparse features).

Usage:
    from models.edge.linear_lasso import LinearLassoEdgeClassifier

    model = LinearLassoEdgeClassifier()
    model.fit(X_train, y_train, feature_names=features)
    probs = model.predict_proba(X_test)
    importance = model.get_feature_importance()
    # Check which features have non-zero coefficients
"""

from pathlib import Path
from typing import Optional, List
import joblib
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


class LinearLassoEdgeClassifier:
    """Lasso (L1) model for edge classification with feature selection.

    Attributes:
        alphas: Regularization strengths to try
        cv: Number of cross-validation folds
        max_iter: Maximum iterations for optimization
    """

    def __init__(
        self,
        alphas: Optional[np.ndarray] = None,
        cv: int = 5,
        max_iter: int = 10000,
    ):
        """Initialize Lasso classifier.

        Args:
            alphas: Regularization strengths (default: 20 values from 0.0001 to 10)
            cv: Cross-validation folds
            max_iter: Max optimization iterations
        """
        self.alphas = alphas if alphas is not None else np.logspace(-4, 1, 20)
        self.cv = cv
        self.max_iter = max_iter

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
        self.model = LassoCV(
            alphas=self.alphas,
            cv=self.cv,
            max_iter=self.max_iter,
            n_jobs=-1,
            random_state=42,
            selection='random',
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
        preds = self.model.predict(X_scaled)
        preds = np.clip(preds, 0, 1)

        return np.column_stack([1 - preds, preds])

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

    def get_selected_features(self, threshold: float = 0.001):
        """Get features with non-zero coefficients (Lasso feature selection).

        Args:
            threshold: Minimum absolute coefficient to consider non-zero

        Returns:
            List of selected feature names
        """
        if self.model is None:
            return []

        coefs = np.abs(self.model.coef_)
        return [name for name, coef in zip(self.feature_names, coefs) if coef > threshold]

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
                'max_iter': self.max_iter,
            }
        }, path)

    @classmethod
    def load(cls, path: Path):
        """Load model from disk."""
        data = joblib.load(path)

        instance = cls(
            alphas=np.array(data['params']['alphas']),
            cv=data['params']['cv'],
            max_iter=data['params']['max_iter'],
        )

        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.best_alpha_ = data['alpha']

        return instance

    def __repr__(self):
        if self.model is None:
            return "LinearLassoEdgeClassifier(untrained)"
        n_selected = len(self.get_selected_features())
        return f"LinearLassoEdgeClassifier(alpha={self.best_alpha_:.4f}, {n_selected}/{len(self.feature_names)} features selected)"
