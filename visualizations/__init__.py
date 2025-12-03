"""
Visualization tools for model calibration and diagnostics.

This package provides plotting and reporting tools for:
1. Calibration assessment (reliability diagrams, Brier score)
2. Edge classifier performance (PnL curves, Sharpe analysis)
3. Model diagnostics and quality checks
"""

from visualizations.calibration_plots import (
    plot_reliability_diagram,
    summarize_calibration,
)
from visualizations.edge_reports import (
    load_edge_model_and_data,
    plot_edge_calibration_for_city,
    plot_pnl_sharpe_vs_threshold,
    edge_report_for_city,
)

__all__ = [
    # Calibration plots
    "plot_reliability_diagram",
    "summarize_calibration",
    # Edge reports
    "load_edge_model_and_data",
    "plot_edge_calibration_for_city",
    "plot_pnl_sharpe_vs_threshold",
    "edge_report_for_city",
]
