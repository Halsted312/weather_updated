#!/usr/bin/env python3
"""
Test suite for CatBoost model implementation.

Tests:
- Monotonic constraints construction
- Model training and prediction
- Calibration pipeline
- Walk-forward training integration
- Performance comparison with ElasticNet
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from unittest.mock import Mock, patch
from sklearn.model_selection import train_test_split

from ml.catboost_model import (
    build_monotone_constraints,
    objective_catboost,
    train_one_window_catboost,
    WindowResult
)
from ml.logit_linear import WindowData


class TestMonotoneConstraints:
    """Test monotonic constraints construction."""

    def test_greater_bracket_constraints(self):
        """Test constraints for greater bracket."""
        features = ["temp_to_floor", "spread_cents", "minutes_to_close", "other_feature"]

        constraints = build_monotone_constraints(
            features,
            bracket_type="greater",
            use_temp_to_floor=True,
            use_spread=True
        )

        assert constraints == [1, -1, 0, 0]

    def test_less_bracket_constraints(self):
        """Test constraints for less bracket."""
        features = ["temp_to_cap", "spread_cents", "log_minutes_to_close", "other_feature"]

        constraints = build_monotone_constraints(
            features,
            bracket_type="less",
            use_temp_to_cap=True,
            use_spread=False
        )

        assert constraints == [1, 0, 0, 0]

    def test_between_bracket_constraints(self):
        """Test constraints for between bracket."""
        features = ["temp_to_floor", "temp_to_cap", "spread_cents", "other_feature"]

        constraints = build_monotone_constraints(
            features,
            bracket_type="between",
            use_temp_to_floor=True,
            use_temp_to_cap=True,
            use_spread=True
        )

        assert constraints == [1, 1, -1, 0]

    def test_no_constraints(self):
        """Test with all constraints disabled."""
        features = ["temp_to_floor", "temp_to_cap", "spread_cents"]

        constraints = build_monotone_constraints(
            features,
            bracket_type="greater",
            use_temp_to_floor=False,
            use_temp_to_cap=False,
            use_spread=False
        )

        assert constraints == [0, 0, 0]


class TestCatBoostTraining:
    """Test CatBoost model training."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        # Create target with some signal
        y = (X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)

        # Create groups (event dates)
        n_groups = 20
        groups = np.repeat(np.arange(n_groups), n_samples // n_groups)

        # Create metadata
        meta = pd.DataFrame({
            'market_ticker': [f'TICKER_{i}' for i in range(n_samples)],
            'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='1min'),
            'event_date': pd.date_range('2025-01-01', periods=n_samples, freq='1D').repeat(n_samples // n_groups)[:n_samples]
        })

        # Split into train/test
        train_idx, test_idx = train_test_split(
            np.arange(n_samples),
            test_size=0.2,
            random_state=42
        )

        return WindowData(
            X_train=X[train_idx],
            y_train=y[train_idx],
            groups_train=groups[train_idx],
            X_test=X[test_idx],
            y_test=y[test_idx],
            meta_test=meta.iloc[test_idx]
        )

    def test_objective_function(self, sample_data):
        """Test Optuna objective function."""
        from optuna import create_study, Trial

        # Create mock trial
        trial = Mock(spec=Trial)
        trial.suggest_int.side_effect = lambda name, low, high: (low + high) // 2
        trial.suggest_float.side_effect = lambda name, low, high, **kwargs: (low + high) / 2
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]

        # Run objective
        score = objective_catboost(
            trial,
            sample_data.X_train,
            sample_data.y_train,
            sample_data.groups_train,
            use_monotone=False,
            bracket_type="between",
            feature_cols=None
        )

        # Should return a score (lower is better for log loss)
        assert isinstance(score, float)
        assert score > 0  # Log loss is positive

    def test_train_one_window(self, sample_data, tmp_path):
        """Test training on one window."""
        artifacts_dir = str(tmp_path / "artifacts")
        tag = "test_model"

        # Train with minimal trials for speed
        result = train_one_window_catboost(
            sample_data,
            artifacts_dir,
            tag,
            n_trials=2,  # Minimal for testing
            use_monotone=False,
            bracket_type="between"
        )

        # Check result structure
        assert isinstance(result, WindowResult)
        assert result.best_params is not None
        assert result.calib_method in ["isotonic", "sigmoid"]
        assert "log_loss" in result.test_metrics
        assert "brier" in result.test_metrics

        # Check artifacts saved
        assert os.path.exists(os.path.join(artifacts_dir, f"model_{tag}.pkl"))
        assert os.path.exists(os.path.join(artifacts_dir, f"params_{tag}.json"))
        assert os.path.exists(os.path.join(artifacts_dir, f"preds_{tag}.csv"))

        # Load predictions
        preds_df = pd.read_csv(os.path.join(artifacts_dir, f"preds_{tag}.csv"))
        assert len(preds_df) == len(sample_data.X_test)
        assert "p_model" in preds_df.columns
        assert "y_true" in preds_df.columns

    def test_calibration_methods(self, sample_data, tmp_path):
        """Test different calibration methods."""
        artifacts_dir = str(tmp_path / "artifacts")

        # Test with small data (should use sigmoid)
        small_data = WindowData(
            X_train=sample_data.X_train[:500],
            y_train=sample_data.y_train[:500],
            groups_train=sample_data.groups_train[:500],
            X_test=sample_data.X_test[:50],
            y_test=sample_data.y_test[:50],
            meta_test=sample_data.meta_test.iloc[:50]
        )

        result_small = train_one_window_catboost(
            small_data,
            artifacts_dir,
            "small_model",
            n_trials=1,
            use_monotone=False,
            bracket_type="between"
        )

        assert result_small.calib_method == "sigmoid"  # Small data uses sigmoid

        # Test with large data (should use isotonic if enough data)
        if len(sample_data.X_train) >= 1000:
            result_large = train_one_window_catboost(
                sample_data,
                artifacts_dir,
                "large_model",
                n_trials=1,
                use_monotone=False,
                bracket_type="between"
            )

            assert result_large.calib_method == "isotonic"  # Large data uses isotonic

    def test_monotonic_constraints_in_training(self, sample_data, tmp_path):
        """Test training with monotonic constraints."""
        artifacts_dir = str(tmp_path / "artifacts")

        # Add feature names that trigger constraints
        feature_cols = ["temp_to_floor", "temp_to_cap", "spread_cents"] + \
                      [f"feature_{i}" for i in range(sample_data.X_train.shape[1] - 3)]

        # Adjust data shape if needed
        if sample_data.X_train.shape[1] < len(feature_cols):
            # Pad with random features
            n_extra = len(feature_cols) - sample_data.X_train.shape[1]
            X_train_padded = np.hstack([
                sample_data.X_train,
                np.random.randn(len(sample_data.X_train), n_extra)
            ])
            X_test_padded = np.hstack([
                sample_data.X_test,
                np.random.randn(len(sample_data.X_test), n_extra)
            ])

            data_with_constraints = WindowData(
                X_train=X_train_padded,
                y_train=sample_data.y_train,
                groups_train=sample_data.groups_train,
                X_test=X_test_padded,
                y_test=sample_data.y_test,
                meta_test=sample_data.meta_test
            )
        else:
            data_with_constraints = sample_data

        # Train with monotonic constraints
        result = train_one_window_catboost(
            data_with_constraints,
            artifacts_dir,
            "monotone_model",
            n_trials=1,
            use_monotone=True,
            bracket_type="between",
            feature_cols=feature_cols
        )

        # Should complete without error
        assert result is not None
        assert result.best_params is not None


class TestModelComparison:
    """Test comparison between ElasticNet and CatBoost."""

    @pytest.fixture
    def comparison_data(self):
        """Create data for model comparison."""
        np.random.seed(42)
        n_samples = 2000
        n_features = 15

        # Create features with varying importance
        X = np.random.randn(n_samples, n_features)

        # Non-linear relationship for CatBoost advantage
        y = (
            X[:, 0]**2 +
            np.sin(X[:, 1]) +
            X[:, 2] * X[:, 3] +
            np.random.randn(n_samples) * 0.3 > 1
        ).astype(int)

        groups = np.repeat(np.arange(40), n_samples // 40)

        meta = pd.DataFrame({
            'market_ticker': [f'TICKER_{i}' for i in range(n_samples)],
            'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='1min'),
            'event_date': pd.date_range('2025-01-01', periods=40).repeat(n_samples // 40)[:n_samples]
        })

        train_idx, test_idx = train_test_split(
            np.arange(n_samples),
            test_size=0.2,
            random_state=42
        )

        return WindowData(
            X_train=X[train_idx],
            y_train=y[train_idx],
            groups_train=groups[train_idx],
            X_test=X[test_idx],
            y_test=y[test_idx],
            meta_test=meta.iloc[test_idx]
        )

    def test_catboost_vs_elasticnet(self, comparison_data, tmp_path):
        """Compare CatBoost and ElasticNet performance."""
        from ml.logit_linear import train_one_window

        # Train ElasticNet
        en_dir = str(tmp_path / "elasticnet")
        en_result = train_one_window(
            comparison_data,
            en_dir,
            "elasticnet",
            n_trials=5,
            penalties=["l1", "l2", "elasticnet"]
        )

        # Train CatBoost
        cb_dir = str(tmp_path / "catboost")
        cb_result = train_one_window_catboost(
            comparison_data,
            cb_dir,
            "catboost",
            n_trials=5,
            use_monotone=False,
            bracket_type="between"
        )

        # Both should produce valid results
        assert en_result.test_metrics["log_loss"] > 0
        assert cb_result.test_metrics["log_loss"] > 0

        # Log comparison
        print(f"\nModel Comparison:")
        print(f"ElasticNet - Log Loss: {en_result.test_metrics['log_loss']:.4f}, "
              f"Brier: {en_result.test_metrics['brier']:.4f}")
        print(f"CatBoost - Log Loss: {cb_result.test_metrics['log_loss']:.4f}, "
              f"Brier: {cb_result.test_metrics['brier']:.4f}")

        # CatBoost might perform better on non-linear data
        # (but not guaranteed in this synthetic test)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_class_data(self, tmp_path):
        """Test handling of single-class data."""
        # All zeros
        X = np.random.randn(100, 5)
        y = np.zeros(100, dtype=int)
        groups = np.arange(100)

        meta = pd.DataFrame({
            'market_ticker': [f'T{i}' for i in range(20)],
            'timestamp': pd.date_range('2025-01-01', periods=20, freq='1min'),
            'event_date': pd.date_range('2025-01-01', periods=20)
        })

        data = WindowData(
            X_train=X[:80],
            y_train=y[:80],
            groups_train=groups[:80],
            X_test=X[80:],
            y_test=y[80:],
            meta_test=meta
        )

        # Should handle gracefully
        result = train_one_window_catboost(
            data,
            str(tmp_path / "single_class"),
            "single",
            n_trials=1,
            use_monotone=False,
            bracket_type="between"
        )

        # Should still return a result (possibly with warnings)
        assert result is not None

    def test_empty_feature_names(self, tmp_path):
        """Test with no feature names provided."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        groups = np.arange(100)

        meta = pd.DataFrame({
            'market_ticker': [f'T{i}' for i in range(20)],
            'timestamp': pd.date_range('2025-01-01', periods=20, freq='1min'),
            'event_date': pd.date_range('2025-01-01', periods=20)
        })

        data = WindowData(
            X_train=X[:80],
            y_train=y[:80],
            groups_train=groups[:80],
            X_test=X[80:],
            y_test=y[80:],
            meta_test=meta
        )

        # Should work without feature names
        result = train_one_window_catboost(
            data,
            str(tmp_path / "no_features"),
            "nofeats",
            n_trials=1,
            use_monotone=True,  # Will use default feature names
            bracket_type="between",
            feature_cols=None  # No feature names
        )

        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])