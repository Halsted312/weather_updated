"""
Edge classifier performance reports and diagnostics.

Provides tools for analyzing edge classifier quality:
- Calibration curves
- PnL vs threshold analysis
- Sharpe vs threshold analysis
- Comprehensive reports per city
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.edge.classifier import EdgeClassifier
from visualizations.calibration_plots import (
    plot_reliability_diagram,
    summarize_calibration,
    plot_calibration_histogram,
)


def load_edge_model_and_data(city: str) -> Tuple[EdgeClassifier, pd.DataFrame]:
    """Helper: load edge classifier + edge_training_data for a city.

    Args:
        city: City name (e.g., 'austin')

    Returns:
        Tuple of (EdgeClassifier, edge_training_data_df)

    Raises:
        FileNotFoundError: If model or data not found
    """
    base = Path(f"models/saved/{city}")
    model_path = base / "edge_classifier"
    data_path = base / "edge_training_data.parquet"

    if not model_path.with_suffix(".pkl").exists():
        raise FileNotFoundError(f"Edge classifier not found: {model_path}.pkl")

    if not data_path.exists():
        raise FileNotFoundError(f"Edge training data not found: {data_path}")

    clf = EdgeClassifier()
    clf.load(str(model_path))

    df_edge = pd.read_parquet(data_path)

    # Filter to signals only (exclude NO_TRADE), like training script
    if "signal" in df_edge.columns:
        df_signals = df_edge[df_edge["signal"] != "no_trade"].copy()
    else:
        df_signals = df_edge.copy()

    print(f"Loaded {city}: {len(df_signals):,} edge signals")

    return clf, df_signals


def plot_edge_calibration_for_city(
    city: str,
    save_dir: Optional[Path] = None,
    test_fraction: float = 0.2,
):
    """Calibration plots for edge classifier on a hold-out slice.

    Args:
        city: City name
        save_dir: Directory to save plots (default: visualizations/edge/{city})
        test_fraction: Fraction of days for test set (last days)

    Returns:
        Dict with metrics and paths
    """
    clf, df = load_edge_model_and_data(city)

    # Time-ordered split (last test_fraction of days)
    df = df.sort_values("day").reset_index(drop=True)

    unique_days = df["day"].unique()
    n_days = len(unique_days)
    n_test_days = max(1, int(n_days * test_fraction))
    test_days = unique_days[-n_test_days:]

    df_test = df[df["day"].isin(test_days)].copy()

    # True labels and predicted probabilities
    y_true = (df_test["pnl"] > 0).astype(int).values
    y_proba = clf.predict(df_test)

    # Calibration metrics
    metrics = summarize_calibration(y_true, y_proba)

    # Set up save directory
    if save_dir is None:
        save_dir = Path(f"visualizations/edge/{city}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot reliability diagram
    title = f"{city.title()} Edge Classifier – Calibration (Brier={metrics['brier']:.4f})"
    fig_path_calib = save_dir / "edge_calibration.png"
    plot_reliability_diagram(y_true, y_proba, n_bins=15, title=title, save_path=fig_path_calib)

    # Plot probability histogram
    fig_path_hist = save_dir / "edge_prob_histogram.png"
    plot_calibration_histogram(
        y_proba,
        title=f"{city.title()} – Predicted Probability Distribution",
        save_path=fig_path_hist
    )

    return {
        "metrics": metrics,
        "calibration_plot": fig_path_calib,
        "histogram_plot": fig_path_hist,
    }


def plot_pnl_sharpe_vs_threshold(
    df_signals: pd.DataFrame,
    proba: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
    min_trades: int = 10,
    save_path: Optional[Path] = None,
    city: Optional[str] = None,
):
    """Plot mean PnL and Sharpe as functions of decision_threshold.

    Args:
        df_signals: DataFrame with 'pnl' column
        proba: Predicted probabilities for each signal
        thresholds: Threshold values to evaluate (default: linspace(0.5, 0.99, 25))
        min_trades: Minimum trades required for valid metric
        save_path: Optional path to save plot
        city: City name for plot title

    Returns:
        Dict with threshold scan results
    """
    y_pnl = df_signals["pnl"].astype(float).values
    proba = np.asarray(proba)

    if thresholds is None:
        thresholds = np.linspace(0.5, 0.99, 25)

    mean_pnls = []
    sharpes = []
    trade_counts = []
    win_rates = []

    for thr in thresholds:
        mask = proba >= thr
        trades = int(mask.sum())
        trade_counts.append(trades)

        if trades < min_trades:
            mean_pnls.append(np.nan)
            sharpes.append(np.nan)
            win_rates.append(np.nan)
            continue

        pnl_trades = y_pnl[mask]
        mean_pnl = float(pnl_trades.mean())
        mean_pnls.append(mean_pnl)

        # Win rate
        y_true_trades = (pnl_trades > 0).astype(int)
        win_rate = float(y_true_trades.mean())
        win_rates.append(win_rate)

        # Sharpe
        std = float(pnl_trades.std())
        if std == 0:
            sharpes.append(np.nan)
        else:
            sharpes.append(mean_pnl / std)

    thresholds = np.asarray(thresholds)
    mean_pnls = np.asarray(mean_pnls)
    sharpes = np.asarray(sharpes)
    trade_counts = np.asarray(trade_counts)
    win_rates = np.asarray(win_rates)

    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    city_title = f"{city.title()} – " if city else ""

    # Plot 1: Mean PnL vs threshold
    ax1.plot(thresholds, mean_pnls, marker="o", linewidth=2, color='#2E86AB')
    ax1.set_xlabel("Decision threshold", fontsize=11)
    ax1.set_ylabel("Mean PnL per trade", fontsize=11)
    ax1.set_title(f"{city_title}Mean PnL vs Threshold", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Plot 2: Sharpe vs threshold
    ax2.plot(thresholds, sharpes, marker="x", linewidth=2, color='#A23B72', markersize=8)
    ax2.set_xlabel("Decision threshold", fontsize=11)
    ax2.set_ylabel("Sharpe ratio (per trade)", fontsize=11)
    ax2.set_title(f"{city_title}Sharpe Ratio vs Threshold", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Plot 3: Trade count and win rate vs threshold
    ax3_twin = ax3.twinx()

    ax3.plot(thresholds, trade_counts, marker="s", linewidth=2,
             color='#F18F01', label='Trade count', markersize=6)
    ax3.set_xlabel("Decision threshold", fontsize=11)
    ax3.set_ylabel("Trade count", fontsize=11, color='#F18F01')
    ax3.tick_params(axis='y', labelcolor='#F18F01')

    ax3_twin.plot(thresholds, win_rates, marker="^", linewidth=2, linestyle='--',
                  color='#06A77D', label='Win rate', markersize=6)
    ax3_twin.set_ylabel("Win rate", fontsize=11, color='#06A77D')
    ax3_twin.tick_params(axis='y', labelcolor='#06A77D')
    ax3_twin.set_ylim([0, 1])
    ax3_twin.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax3.set_title(f"{city_title}Trade Count & Win Rate vs Threshold",
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=1)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"Saved PnL/Sharpe analysis to {save_path}")

    return {
        "thresholds": thresholds,
        "mean_pnls": mean_pnls,
        "sharpes": sharpes,
        "trade_counts": trade_counts,
        "win_rates": win_rates,
    }


def edge_report_for_city(
    city: str,
    save_dir: Optional[Path] = None,
    test_fraction: float = 0.2,
):
    """Generate comprehensive report for edge classifier.

    Creates:
    1. Calibration curve
    2. Probability histogram
    3. PnL/Sharpe vs threshold analysis

    Args:
        city: City name
        save_dir: Directory to save plots (default: visualizations/edge/{city})
        test_fraction: Fraction of days for test set

    Returns:
        Dict with all generated plots and metrics
    """
    print(f"\n{'='*60}")
    print(f"Generating Edge Classifier Report for {city.title()}")
    print(f"{'='*60}\n")

    clf, df_signals = load_edge_model_and_data(city)

    # Time-ordered split
    df_signals = df_signals.sort_values("day").reset_index(drop=True)
    unique_days = df_signals["day"].unique()
    n_days = len(unique_days)
    n_test_days = max(1, int(n_days * test_fraction))
    test_days = unique_days[-n_test_days:]

    df_test = df_signals[df_signals["day"].isin(test_days)].copy()

    print(f"Test set: {len(df_test):,} signals from {len(test_days)} days")
    print(f"Date range: {test_days[0]} to {test_days[-1]}")

    # Predict probabilities
    proba_test = clf.predict(df_test)

    # Set up save directory
    if save_dir is None:
        save_dir = Path(f"visualizations/edge/{city}")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving reports to: {save_dir}")

    # 1. Calibration analysis
    print("\n1. Generating calibration plots...")
    cal = plot_edge_calibration_for_city(city, save_dir, test_fraction)
    print(f"   Brier score: {cal['metrics']['brier']:.4f}")
    print(f"   Log loss: {cal['metrics']['log_loss']:.4f}")
    print(f"   ECE: {cal['metrics']['ece']:.4f}")

    # 2. PnL/Sharpe analysis
    print("\n2. Generating PnL/Sharpe threshold analysis...")
    pnl_sharpe = plot_pnl_sharpe_vs_threshold(
        df_signals=df_test,
        proba=proba_test,
        save_path=save_dir / "edge_pnl_sharpe_vs_threshold.png",
        city=city,
    )

    # Find optimal threshold by Sharpe
    valid_idx = ~np.isnan(pnl_sharpe["sharpes"])
    if valid_idx.sum() > 0:
        best_idx = np.nanargmax(pnl_sharpe["sharpes"])
        best_threshold = pnl_sharpe["thresholds"][best_idx]
        best_sharpe = pnl_sharpe["sharpes"][best_idx]
        best_mean_pnl = pnl_sharpe["mean_pnls"][best_idx]
        best_trades = pnl_sharpe["trade_counts"][best_idx]
        best_win_rate = pnl_sharpe["win_rates"][best_idx]

        print(f"\n   Optimal threshold: {best_threshold:.2f}")
        print(f"   Best Sharpe: {best_sharpe:.3f}")
        print(f"   Mean PnL: {best_mean_pnl:.3f}")
        print(f"   Win rate: {best_win_rate:.1%}")
        print(f"   Trade count: {best_trades}")

    print(f"\n{'='*60}")
    print(f"Report complete! Check {save_dir} for plots.")
    print(f"{'='*60}\n")

    return {
        "calibration": cal,
        "pnl_sharpe": pnl_sharpe,
        "save_dir": save_dir,
        "city": city,
    }
