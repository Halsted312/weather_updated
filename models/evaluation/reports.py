"""
Report generation for model evaluation and comparison.

This module generates human-readable reports and comparison files
for trained temperature Δ-models.

Outputs:
    - model_comparison.csv: Side-by-side metrics for all models
    - model_comparison.md: Markdown summary with interpretation
    - Individual model reports with calibration curves

Example:
    >>> from models.evaluation.reports import generate_comparison_report
    >>> generate_comparison_report(
    ...     evaluators=[logistic_eval, catboost_eval],
    ...     output_dir=Path('models/reports/'),
    ... )
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from models.evaluation.evaluator import ModelEvaluator, compare_models

logger = logging.getLogger(__name__)


def generate_comparison_report(
    evaluators: list[ModelEvaluator],
    output_dir: Path,
    title: str = "Model Comparison Report",
) -> None:
    """Generate comprehensive comparison report for multiple models.

    Creates:
        - model_comparison.csv: Raw metrics
        - model_comparison.md: Formatted markdown report

    Args:
        evaluators: List of evaluated ModelEvaluator objects
        output_dir: Directory to save reports
        title: Report title
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure all models are evaluated
    for evaluator in evaluators:
        if "summary" not in evaluator.results:
            evaluator.full_evaluation()

    # Generate comparison DataFrame
    comparison_df = compare_models(evaluators)

    # Save CSV
    csv_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison CSV to {csv_path}")

    # Generate markdown report
    md_content = _generate_markdown_report(evaluators, comparison_df, title)
    md_path = output_dir / "model_comparison.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info(f"Saved comparison markdown to {md_path}")

    # Save individual model results
    for evaluator in evaluators:
        evaluator.save_results(output_dir)


def _generate_markdown_report(
    evaluators: list[ModelEvaluator],
    comparison_df: pd.DataFrame,
    title: str,
) -> str:
    """Generate markdown formatted report."""
    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
    ]

    # Number of models and samples
    n_models = len(evaluators)
    n_samples = evaluators[0].results.get("summary", {}).get("n_test_samples", "?")
    n_days = evaluators[0].results.get("summary", {}).get("n_test_days", "?")

    lines.extend([
        f"- **Models compared**: {n_models}",
        f"- **Test samples**: {n_samples}",
        f"- **Test days**: {n_days}",
        "",
        "## Overall Metrics",
        "",
    ])

    # Key metrics table
    key_metrics = [
        ("delta_accuracy", "Delta Accuracy", "{:.2%}"),
        ("delta_mae", "Delta MAE", "{:.3f}"),
        ("settlement_accuracy", "Settlement Accuracy", "{:.2%}"),
        ("settlement_mae", "Settlement MAE (°F)", "{:.3f}"),
        ("log_loss", "Log Loss", "{:.4f}"),
        ("brier_90", "Brier Score (T≥90)", "{:.4f}"),
    ]

    # Build table header
    header = "| Metric |"
    separator = "|--------|"
    for evaluator in evaluators:
        header += f" {evaluator.model_name} |"
        separator += "--------|"

    lines.extend([header, separator])

    # Add rows
    for col, name, fmt in key_metrics:
        row = f"| {name} |"
        for evaluator in evaluators:
            value = evaluator.results.get("summary", {}).get(col)
            if value is not None:
                row += f" {fmt.format(value)} |"
            else:
                row += " - |"
        lines.append(row)

    lines.extend(["", "## Bracket Brier Scores", ""])

    # Brier scores table
    brier_thresholds = [80, 85, 90, 95]
    header = "| Threshold |"
    separator = "|-----------|"
    for evaluator in evaluators:
        header += f" {evaluator.model_name} |"
        separator += "--------|"

    lines.extend([header, separator])

    for threshold in brier_thresholds:
        row = f"| T ≥ {threshold}°F |"
        for evaluator in evaluators:
            value = evaluator.results.get("bracket_metrics", {}).get(f"brier_{threshold}")
            if value is not None:
                row += f" {value:.4f} |"
            else:
                row += " - |"
        lines.append(row)

    # Performance by snapshot hour
    lines.extend(["", "## Performance by Snapshot Hour", ""])

    for evaluator in evaluators:
        lines.append(f"### {evaluator.model_name}")
        lines.append("")

        hourly_df = evaluator.results.get("hourly_metrics")
        if hourly_df is not None and not hourly_df.empty:
            lines.append("| Hour | Samples | Delta Acc | Settle Acc | Brier 90 |")
            lines.append("|------|---------|-----------|------------|----------|")

            for _, row in hourly_df.iterrows():
                lines.append(
                    f"| {int(row['snapshot_hour']):02d}:00 | "
                    f"{int(row['n_samples'])} | "
                    f"{row['delta_accuracy']:.2%} | "
                    f"{row['settlement_accuracy']:.2%} | "
                    f"{row.get('brier_90', 0):.4f} |"
                )
        else:
            lines.append("*No hourly data available*")

        lines.append("")

    # Interpretation
    lines.extend([
        "## Interpretation",
        "",
        "### Metrics Explained",
        "",
        "- **Delta Accuracy**: Exact match rate on predicted Δ class",
        "- **Settlement Accuracy**: Exact match rate on final temperature",
        "- **MAE**: Mean absolute error in °F",
        "- **Log Loss**: Probabilistic measure (lower is better)",
        "- **Brier Score**: Calibration quality for bracket events (lower is better)",
        "",
        "### Key Observations",
        "",
    ])

    # Add automatic observations
    observations = _generate_observations(evaluators)
    for obs in observations:
        lines.append(f"- {obs}")

    lines.extend([
        "",
        "---",
        "",
        "*Report generated by models/evaluation/reports.py*",
    ])

    return "\n".join(lines)


def _generate_observations(evaluators: list[ModelEvaluator]) -> list[str]:
    """Generate automatic observations about model performance."""
    observations = []

    if len(evaluators) < 2:
        return observations

    # Compare settlement accuracy
    accuracies = [
        (e.model_name, e.results.get("summary", {}).get("settlement_accuracy", 0))
        for e in evaluators
    ]
    best = max(accuracies, key=lambda x: x[1])
    observations.append(f"**{best[0]}** has the highest settlement accuracy ({best[1]:.2%})")

    # Compare Brier scores
    brier_90s = [
        (e.model_name, e.results.get("bracket_metrics", {}).get("brier_90", 1))
        for e in evaluators
    ]
    best_brier = min(brier_90s, key=lambda x: x[1])
    observations.append(f"**{best_brier[0]}** has the best calibration at T≥90 (Brier: {best_brier[1]:.4f})")

    # Check if models improve with later snapshot hours
    for evaluator in evaluators:
        hourly_df = evaluator.results.get("hourly_metrics")
        if hourly_df is not None and len(hourly_df) > 2:
            early = hourly_df[hourly_df["snapshot_hour"] <= 14]["settlement_accuracy"].mean()
            late = hourly_df[hourly_df["snapshot_hour"] >= 20]["settlement_accuracy"].mean()
            if late > early:
                improvement = late - early
                observations.append(
                    f"**{evaluator.model_name}**: Late-day snapshots are {improvement:.1%} more accurate than early-day"
                )

    return observations


def generate_single_model_report(
    evaluator: ModelEvaluator,
    output_dir: Path,
) -> None:
    """Generate detailed report for a single model.

    Args:
        evaluator: Evaluated ModelEvaluator
        output_dir: Directory to save report
    """
    if "summary" not in evaluator.results:
        evaluator.full_evaluation()

    evaluator.save_results(output_dir)

    # Generate individual markdown
    lines = [
        f"# {evaluator.model_name} Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Statistics",
        "",
    ]

    summary = evaluator.results.get("summary", {})
    for key, value in summary.items():
        if isinstance(value, float):
            if "accuracy" in key or "rate" in key:
                lines.append(f"- **{key}**: {value:.2%}")
            else:
                lines.append(f"- **{key}**: {value:.4f}")
        else:
            lines.append(f"- **{key}**: {value}")

    md_path = output_dir / f"{evaluator.model_name}_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Saved model report to {md_path}")
