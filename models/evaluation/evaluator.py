"""
Full evaluation suite for trained temperature Δ-models.

This module provides comprehensive evaluation capabilities including:
- Standard metrics (accuracy, MAE, Brier scores)
- Stratification by snapshot hour
- Calibration analysis
- Model comparison

Example:
    >>> from models.evaluation.evaluator import ModelEvaluator
    >>> evaluator = ModelEvaluator(model, df_test)
    >>> results = evaluator.full_evaluation()
    >>> evaluator.save_results('models/reports/')
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from models.evaluation.metrics import (
    compute_all_metrics,
    compute_delta_metrics,
    compute_settlement_metrics,
    compute_bracket_brier_score,
    compute_metrics_by_snapshot_hour,
    compute_calibration_curve_data,
)
from models.features.base import DELTA_CLASSES

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive evaluator for trained Δ-models.

    Computes all metrics and generates evaluation reports.

    Attributes:
        model: Trained model with predict() and predict_proba() methods
        df_test: Test DataFrame with features and labels
        results: Evaluation results (after calling full_evaluation())
    """

    def __init__(
        self,
        model: Any,
        df_test: pd.DataFrame,
        model_name: str = "model",
    ):
        """Initialize evaluator.

        Args:
            model: Trained model
            df_test: Test DataFrame with delta, settle_f, t_base columns
            model_name: Name for this model (for reports)
        """
        self.model = model
        self.df_test = df_test.copy()
        self.model_name = model_name

        # Generate predictions
        self._generate_predictions()

        self.results = {}

    def _generate_predictions(self) -> None:
        """Generate predictions and probabilities from model."""
        # Handle different model interfaces
        if hasattr(self.model, "predict"):
            self.y_pred = self.model.predict(self.df_test)
        else:
            raise ValueError("Model must have predict() method")

        if hasattr(self.model, "predict_proba"):
            self.proba = self.model.predict_proba(self.df_test)
        else:
            # Create dummy probabilities if not available
            n_samples = len(self.df_test)
            n_classes = len(DELTA_CLASSES)
            self.proba = np.zeros((n_samples, n_classes))
            for i, pred in enumerate(self.y_pred):
                class_idx = DELTA_CLASSES.index(pred) if pred in DELTA_CLASSES else 2
                self.proba[i, class_idx] = 1.0

        # Store in DataFrame
        self.df_test["delta_pred"] = self.y_pred
        self.df_test["t_pred"] = self.df_test["t_base"] + self.df_test["delta_pred"]

    def evaluate_delta(self) -> dict:
        """Compute delta-level classification metrics."""
        y_true = self.df_test["delta"].values

        metrics = compute_delta_metrics(y_true, self.y_pred)
        self.results["delta_metrics"] = metrics

        return metrics

    def evaluate_settlement(self) -> dict:
        """Compute settlement temperature metrics."""
        t_settle = self.df_test["settle_f"].values
        t_pred = self.df_test["t_pred"].values

        metrics = compute_settlement_metrics(t_settle, t_pred)
        self.results["settlement_metrics"] = metrics

        return metrics

    def evaluate_brackets(
        self,
        thresholds: list[int] = [80, 85, 90, 95],
    ) -> dict:
        """Compute Brier scores for key bracket thresholds."""
        t_base = self.df_test["t_base"].values
        t_settle = self.df_test["settle_f"].values
        delta_classes = np.array(DELTA_CLASSES)

        bracket_metrics = {}
        for threshold in thresholds:
            brier = compute_bracket_brier_score(
                self.proba, t_base, t_settle, threshold, delta_classes
            )
            bracket_metrics[f"brier_{threshold}"] = brier

        self.results["bracket_metrics"] = bracket_metrics
        return bracket_metrics

    def evaluate_by_snapshot_hour(self) -> pd.DataFrame:
        """Compute metrics stratified by snapshot hour."""
        df_hourly = compute_metrics_by_snapshot_hour(
            self.df_test, self.y_pred, self.proba
        )
        self.results["hourly_metrics"] = df_hourly

        return df_hourly

    def evaluate_calibration(
        self,
        thresholds: list[int] = [80, 85, 90, 95],
        n_bins: int = 10,
    ) -> dict[str, pd.DataFrame]:
        """Generate calibration curves for bracket events."""
        t_base = self.df_test["t_base"].values
        t_settle = self.df_test["settle_f"].values
        delta_classes = np.array(DELTA_CLASSES)

        calibration_data = {}

        for threshold in thresholds:
            # Compute P(T >= threshold) for each sample
            n_samples = len(t_base)
            p_bracket = np.zeros(n_samples)

            for i in range(n_samples):
                for j, d in enumerate(delta_classes):
                    if t_base[i] + d >= threshold:
                        p_bracket[i] += self.proba[i, j]

            y_bracket = (t_settle >= threshold).astype(int)

            calib_df = compute_calibration_curve_data(y_bracket, p_bracket, n_bins)
            if not calib_df.empty:
                calibration_data[f"T_ge_{threshold}"] = calib_df

        self.results["calibration"] = calibration_data
        return calibration_data

    def full_evaluation(
        self,
        thresholds: list[int] = [80, 85, 90, 95],
    ) -> dict:
        """Run complete evaluation suite.

        Returns:
            Dict with all evaluation results
        """
        logger.info(f"Running full evaluation for {self.model_name}")

        # Run all evaluations
        self.evaluate_delta()
        self.evaluate_settlement()
        self.evaluate_brackets(thresholds)
        self.evaluate_by_snapshot_hour()
        self.evaluate_calibration(thresholds)

        # Combine into summary
        summary = {
            "model_name": self.model_name,
            "evaluated_at": datetime.now().isoformat(),
            "n_test_samples": len(self.df_test),
            "n_test_days": self.df_test["day"].nunique() if "day" in self.df_test else None,
            **self.results.get("delta_metrics", {}),
            **self.results.get("settlement_metrics", {}),
            **self.results.get("bracket_metrics", {}),
        }

        self.results["summary"] = summary
        return self.results

    def save_results(self, output_dir: Path) -> None:
        """Save evaluation results to files.

        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary as CSV
        if "summary" in self.results:
            summary_df = pd.DataFrame([self.results["summary"]])
            summary_df.to_csv(
                output_dir / f"{self.model_name}_summary.csv", index=False
            )

        # Save hourly metrics
        if "hourly_metrics" in self.results:
            self.results["hourly_metrics"].to_csv(
                output_dir / f"{self.model_name}_hourly.csv", index=False
            )

        # Save calibration curves
        if "calibration" in self.results:
            for name, df in self.results["calibration"].items():
                df.to_csv(
                    output_dir / f"{self.model_name}_calibration_{name}.csv",
                    index=False,
                )

        # Save predictions
        pred_df = self.df_test[["city", "day", "snapshot_hour", "delta", "delta_pred",
                                "settle_f", "t_base", "t_pred"]].copy()
        pred_df.to_csv(
            output_dir / f"{self.model_name}_predictions.csv", index=False
        )

        logger.info(f"Saved evaluation results to {output_dir}")


def compare_models(
    evaluators: list[ModelEvaluator],
) -> pd.DataFrame:
    """Compare multiple models side-by-side.

    Args:
        evaluators: List of ModelEvaluator objects (already evaluated)

    Returns:
        DataFrame comparing key metrics across models
    """
    comparisons = []

    for evaluator in evaluators:
        if "summary" not in evaluator.results:
            evaluator.full_evaluation()

        summary = evaluator.results["summary"]
        comparisons.append(summary)

    return pd.DataFrame(comparisons)
