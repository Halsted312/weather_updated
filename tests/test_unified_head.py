"""
Unit tests for the unified head probability coupling module.
"""

import pytest
import numpy as np
import pandas as pd
from ml.unified_head import (
    _safe_logit,
    softmax_renorm,
    dirichlet_temperature,
    couple_timestamp_rowset,
    validate_bracket_completeness,
    apply_unified_head,
    compute_multiclass_metrics,
)


class TestSafeLogit:
    """Test the safe logit transformation."""

    def test_safe_logit_basic(self):
        """Test basic logit computation."""
        p = np.array([0.5, 0.8, 0.2])
        logits = _safe_logit(p)

        # logit(0.5) = 0
        assert np.allclose(logits[0], 0.0)

        # logit(0.8) > 0
        assert logits[1] > 0

        # logit(0.2) < 0
        assert logits[2] < 0

    def test_safe_logit_extreme_values(self):
        """Test that extreme values are clipped."""
        p = np.array([0.0, 1.0, 0.5])
        logits = _safe_logit(p)

        # Should not be infinite
        assert np.all(np.isfinite(logits))


class TestSoftmaxRenorm:
    """Test softmax renormalization coupling."""

    def test_softmax_renorm_sums_to_one(self):
        """Test that output probabilities sum to 1."""
        probs = np.array([0.1, 0.2, 0.3, 0.15, 0.25, 0.05])
        q = softmax_renorm(probs, tau=1.0)

        assert np.allclose(np.sum(q), 1.0)
        assert len(q) == 6

    def test_softmax_renorm_preserves_ordering_tau_1(self):
        """Test that tau=1 preserves relative ordering."""
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        q = softmax_renorm(probs, tau=1.0)

        # Check monotonicity is preserved
        assert np.all(np.diff(q) > 0), "Ordering not preserved"

    def test_softmax_renorm_temperature_effects(self):
        """Test temperature parameter effects."""
        probs = np.array([0.1, 0.2, 0.3, 0.15, 0.25, 0.05])

        # Lower temperature -> more confident (higher max)
        q_low = softmax_renorm(probs, tau=0.5)
        q_mid = softmax_renorm(probs, tau=1.0)
        q_high = softmax_renorm(probs, tau=2.0)

        assert np.max(q_low) > np.max(q_mid)
        assert np.max(q_mid) > np.max(q_high)

        # Higher temperature -> more uniform (higher entropy)
        entropy_low = -np.sum(q_low * np.log(q_low + 1e-12))
        entropy_mid = -np.sum(q_mid * np.log(q_mid + 1e-12))
        entropy_high = -np.sum(q_high * np.log(q_high + 1e-12))

        assert entropy_low < entropy_mid < entropy_high

    def test_softmax_renorm_error_on_wrong_size(self):
        """Test that error is raised for wrong number of brackets."""
        probs = np.array([0.5, 0.5])  # Only 2 probabilities
        with pytest.raises(ValueError, match="Expected exactly 6"):
            softmax_renorm(probs)


class TestDirichletTemperature:
    """Test Dirichlet temperature coupling."""

    def test_dirichlet_sums_to_one(self):
        """Test that output probabilities sum to 1."""
        probs = np.array([0.1, 0.2, 0.3, 0.15, 0.25, 0.05])
        q = dirichlet_temperature(probs, alpha=0.1)

        assert np.allclose(np.sum(q), 1.0)
        assert len(q) == 6

    def test_dirichlet_smoothing(self):
        """Test that Dirichlet adds uniform smoothing."""
        # Extreme probabilities
        probs = np.array([0.99, 0.01, 0.0, 0.0, 0.0, 0.0])

        # With smoothing, extremes should be moderated
        q = dirichlet_temperature(probs, alpha=0.5)

        assert q[0] < probs[0], "High prob should decrease"
        assert q[1] > probs[1], "Low prob should increase"
        assert np.all(q > 0), "All probs should be positive"


class TestCoupleTimestampRowset:
    """Test coupling for a set of bracket rows."""

    def test_couple_timestamp_rowset_basic(self):
        """Test basic coupling of 6 bracket predictions."""
        df_rows = pd.DataFrame({
            "bracket_key": [
                "less_68", "between_68_73", "between_73_78",
                "between_78_83", "between_83_88", "greater_88"
            ],
            "p_model": [0.1, 0.2, 0.3, 0.2, 0.15, 0.05]
        })

        q = couple_timestamp_rowset(df_rows, p_col="p_model", method="softmax")

        assert len(q) == 6
        assert np.allclose(np.sum(q), 1.0)

    def test_couple_timestamp_rowset_error_on_incomplete(self):
        """Test error when not exactly 6 rows."""
        df_rows = pd.DataFrame({
            "bracket_key": ["less_68", "between_68_73", "greater_88"],
            "p_model": [0.3, 0.4, 0.3]
        })

        with pytest.raises(ValueError, match="Expected exactly 6"):
            couple_timestamp_rowset(df_rows, p_col="p_model")

    def test_couple_timestamp_rowset_methods(self):
        """Test different coupling methods."""
        df_rows = pd.DataFrame({
            "bracket_key": [
                "less_68", "between_68_73", "between_73_78",
                "between_78_83", "between_83_88", "greater_88"
            ],
            "p_model": [0.1, 0.2, 0.3, 0.2, 0.15, 0.05]
        })

        q_softmax = couple_timestamp_rowset(df_rows, method="softmax")
        q_dirichlet = couple_timestamp_rowset(df_rows, method="dirichlet")

        # Both should sum to 1
        assert np.allclose(np.sum(q_softmax), 1.0)
        assert np.allclose(np.sum(q_dirichlet), 1.0)

        # But they should be different
        assert not np.allclose(q_softmax, q_dirichlet)


class TestValidateBracketCompleteness:
    """Test bracket completeness validation."""

    def test_validate_completeness_all_complete(self):
        """Test when all timestamps have 6 brackets."""
        df = pd.DataFrame({
            "event_date": ["2025-08-01"] * 6 + ["2025-08-02"] * 6,
            "timestamp": ["10:00"] * 6 + ["10:00"] * 6,
            "bracket_key": [
                "less_68", "between_68_73", "between_73_78",
                "between_78_83", "between_83_88", "greater_88"
            ] * 2
        })

        coverage = validate_bracket_completeness(df)

        assert len(coverage) == 2
        assert coverage["is_complete"].all()
        assert (coverage["n_brackets"] == 6).all()

    def test_validate_completeness_incomplete(self):
        """Test when some timestamps have missing brackets."""
        df = pd.DataFrame({
            "event_date": ["2025-08-01"] * 4 + ["2025-08-02"] * 6,
            "timestamp": ["10:00"] * 4 + ["10:00"] * 6,
            "bracket_key": [
                "less_68", "between_68_73", "between_73_78", "greater_88",
                "less_68", "between_68_73", "between_73_78",
                "between_78_83", "between_83_88", "greater_88"
            ]
        })

        coverage = validate_bracket_completeness(df)

        assert len(coverage) == 2
        assert coverage.iloc[0]["is_complete"] == False
        assert coverage.iloc[0]["n_brackets"] == 4
        assert coverage.iloc[1]["is_complete"] == True
        assert coverage.iloc[1]["n_brackets"] == 6


class TestApplyUnifiedHead:
    """Test applying unified head to full DataFrame."""

    def test_apply_unified_head_complete_groups(self):
        """Test applying coupling to complete groups."""
        # Create 2 complete groups
        df = pd.DataFrame({
            "event_date": ["2025-08-01"] * 6 + ["2025-08-02"] * 6,
            "timestamp": ["10:00"] * 6 + ["10:00"] * 6,
            "bracket_key": [
                "less_68", "between_68_73", "between_73_78",
                "between_78_83", "between_83_88", "greater_88"
            ] * 2,
            "p_model": [0.1, 0.2, 0.3, 0.2, 0.15, 0.05] * 2
        })

        result = apply_unified_head(df, output_col="p_unified")

        # Should have coupled probabilities
        assert "p_unified" in result.columns
        assert "coupling_status" in result.columns

        # All should be coupled
        assert (result["coupling_status"] == "coupled").all()

        # Each group should sum to 1
        for _, group in result.groupby(["event_date", "timestamp"]):
            assert np.allclose(group["p_unified"].sum(), 1.0)

    def test_apply_unified_head_incomplete_groups(self):
        """Test handling of incomplete groups."""
        # Create 1 complete and 1 incomplete group
        df = pd.DataFrame({
            "event_date": ["2025-08-01"] * 4 + ["2025-08-02"] * 6,
            "timestamp": ["10:00"] * 4 + ["10:00"] * 6,
            "bracket_key": [
                "less_68", "between_68_73", "between_73_78", "greater_88",
                "less_68", "between_68_73", "between_73_78",
                "between_78_83", "between_83_88", "greater_88"
            ],
            "p_model": [0.2, 0.3, 0.3, 0.2] + [0.1, 0.2, 0.3, 0.2, 0.15, 0.05]
        })

        result = apply_unified_head(df, output_col="p_unified")

        # Check statuses
        incomplete = result[result["event_date"] == "2025-08-01"]
        complete = result[result["event_date"] == "2025-08-02"]

        assert (incomplete["coupling_status"] == "incomplete").all()
        assert (complete["coupling_status"] == "coupled").all()

        # Incomplete group keeps original probabilities
        assert np.allclose(incomplete["p_unified"], incomplete["p_model"])

        # Complete group has coupled probabilities
        assert np.allclose(complete["p_unified"].sum(), 1.0)


class TestComputeMulticlassMetrics:
    """Test multiclass metric computation."""

    def test_multiclass_metrics_perfect_predictions(self):
        """Test metrics when predictions are perfect."""
        # 2 complete groups with perfect predictions
        df = pd.DataFrame({
            "event_date": ["2025-08-01"] * 6 + ["2025-08-02"] * 6,
            "timestamp": ["10:00"] * 6 + ["10:00"] * 6,
            "p_unified": [
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Perfect for bracket 0
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  # Perfect for bracket 3
            ],
            "y_true": [
                1, 0, 0, 0, 0, 0,  # Bracket 0 is true
                0, 0, 0, 1, 0, 0,  # Bracket 3 is true
            ]
        })

        metrics = compute_multiclass_metrics(df)

        # Perfect predictions should have ~0 log loss
        assert metrics["multiclass_log_loss"] < 0.01
        assert metrics["mean_true_p"] == 1.0
        assert metrics["mean_max_p"] == 1.0
        assert metrics["n_complete_groups"] == 2

    def test_multiclass_metrics_uniform_predictions(self):
        """Test metrics with uniform predictions."""
        # 1 complete group with uniform predictions
        df = pd.DataFrame({
            "event_date": ["2025-08-01"] * 6,
            "timestamp": ["10:00"] * 6,
            "p_unified": [1/6] * 6,
            "y_true": [1, 0, 0, 0, 0, 0]  # First bracket is true
        })

        metrics = compute_multiclass_metrics(df)

        # Uniform predictions for 6 classes
        expected_log_loss = -np.log(1/6)
        assert np.allclose(metrics["multiclass_log_loss"], expected_log_loss)
        assert np.allclose(metrics["mean_true_p"], 1/6)
        assert np.allclose(metrics["mean_max_p"], 1/6)

    def test_multiclass_metrics_incomplete_groups(self):
        """Test that incomplete groups are skipped."""
        # Only 3 brackets (incomplete)
        df = pd.DataFrame({
            "event_date": ["2025-08-01"] * 3,
            "timestamp": ["10:00"] * 3,
            "p_unified": [0.5, 0.3, 0.2],
            "y_true": [1, 0, 0]
        })

        metrics = compute_multiclass_metrics(df)

        # No complete groups
        assert metrics["multiclass_log_loss"] is None
        assert metrics["mean_true_p"] is None