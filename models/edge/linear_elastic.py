"""ElasticNet-based edge classifier.

Combines L1 (Lasso) and L2 (Ridge) regularization for feature selection + stability.
Fast training (~10-30 seconds), high interpretability.

Usage:
    from models.edge.linear_elastic import LinearElasticEdgeClassifier

    model = LinearElasticEdgeClassifier()
    model.fit(X_train, y_train, feature_names=features)
    probs = model.predict_proba(X_test)
    importance = model.get_feature_importance()
"""

from pathlib import Path
from typing import Optional, List
import joblib
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler


class LinearElasticEdgeClassifier:
    """ElasticNet model for edge classification.

    Attributes:
        l1_ratios: L1/L2 mix ratios to try (0=Ridge, 1=Lasso)
        alphas: Regularization strengths to try
        cv: Number of cross-validation folds
        max_iter: Maximum iterations for optimization
    """

    def __init__(
        self,
        l1_ratios: Optional[List[float]] = None,
        alphas: Optional[np.ndarray] = None,
        cv: int = 5,
        max_iter: int = 10000,
    ):
        """Initialize ElasticNet classifier.

        Args:
            l1_ratios: L1 ratios to try (default: [0.1, 0.5, 0.7, 0.9, 0.95, 0.99])
            alphas: Regularization strengths (default: 20 values from 0.0001 to 10)
            cv: Cross-validation folds
            max_iter: Max optimization iterations
        """
        self.l1_ratios = l1_ratios or [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
        self.alphas = alphas if alphas is not None else np.logspace(-4, 1, 20)
        self.cv = cv
        self.max_iter = max_iter

        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.best_l1_ratio_ = None
        self.best_alpha_ = None

    def fit(self, X, y, feature_names: Optional[List[str]] = None):
        """Train model with cross-validation.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary target (0/1)
            feature_names: Names of features (for interpretability)

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

        # Scale features (important for ElasticNet)
        X_scaled = self.scaler.fit_transform(X)

        # Train with cross-validation
        self.model = ElasticNetCV(
            l1_ratio=self.l1_ratios,
            alphas=self.alphas,
            cv=self.cv,
            max_iter=self.max_iter,
            n_jobs=-1,  # Use all CPUs
            random_state=42,
            selection='random',  # Faster convergence
        )
        self.model.fit(X_scaled, y)

        # Store best parameters
        self.best_l1_ratio_ = self.model.l1_ratio_
        self.best_alpha_ = self.model.alpha_

        return self

    def predict_proba(self, X):
        """Predict probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of shape (n_samples, 2) with [prob_negative, prob_positive]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Convert to numpy if DataFrame
        if hasattr(X, 'values'):
            X = X.values

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)

        # Clip to [0, 1] range
        preds = np.clip(preds, 0, 1)

        # Return as (n_samples, 2) array for sklearn compatibility
        return np.column_stack([1 - preds, preds])

    def predict(self, X, threshold: float = 0.5):
        """Predict binary classes.

        Args:
            X: Feature matrix
            threshold: Decision threshold (default: 0.5)

        Returns:
            Binary predictions (0/1)
        """
        return (self.predict_proba(X)[:, 1] > threshold).astype(int)

    def get_feature_importance(self):
        """Get feature importance as absolute coefficients.

        Returns:
            Dict mapping feature names to importance scores
        """
        if self.model is None:
            return {}

        coefs = np.abs(self.model.coef_)
        return dict(zip(self.feature_names, coefs))

    def get_feature_coefficients(self):
        """Get raw feature coefficients (with sign).

        Returns:
            Dict mapping feature names to signed coefficients
        """
        if self.model is None:
            return {}

        return dict(zip(self.feature_names, self.model.coef_))

    def save(self, path: Path):
        """Save model to disk.

        Args:
            path: Path to save model (will create .pkl file)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'l1_ratio': self.best_l1_ratio_,
            'alpha': self.best_alpha_,
            'params': {
                'l1_ratios': self.l1_ratios,
                'alphas': self.alphas.tolist() if isinstance(self.alphas, np.ndarray) else self.alphas,
                'cv': self.cv,
                'max_iter': self.max_iter,
            }
        }, path)

    @classmethod
    def load(cls, path: Path):
        """Load model from disk.

        Args:
            path: Path to model file

        Returns:
            Loaded LinearElasticEdgeClassifier instance
        """
        data = joblib.load(path)

        instance = cls(
            l1_ratios=data['params']['l1_ratios'],
            alphas=np.array(data['params']['alphas']),
            cv=data['params']['cv'],
            max_iter=data['params']['max_iter'],
        )

        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.best_l1_ratio_ = data['l1_ratio']
        instance.best_alpha_ = data['alpha']

        return instance

    def __repr__(self):
        if self.model is None:
            return "LinearElasticEdgeClassifier(untrained)"
        return f"LinearElasticEdgeClassifier(l1_ratio={self.best_l1_ratio_:.3f}, alpha={self.best_alpha_:.4f})"
