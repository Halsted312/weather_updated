#!/usr/bin/env python3
"""Visualize edge classifier performance across thresholds and time periods.

Creates comprehensive plots to understand:
1. Threshold vs Sharpe/P&L/Win Rate
2. Temporal patterns (rolling windows)
3. Training window size comparison (3mo vs 6mo vs 12mo)
4. Feature importance visualization

Usage:
    python visualizations/plot_edge_analysis.py --city miami
    python visualizations/plot_edge_analysis.py --city miami --save-plots
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_edge_data(city: str):
    """Load edge data for analysis."""
    cache_path = Path(f"models/saved/{city}/edge_training_data_realistic.parquet")

    if not cache_path.exists():
        print(f"❌ Edge data not found: {cache_path}")
        print(f"   Run: python scripts/train_edge_classifier.py --city {city} --regenerate-only")
        sys.exit(1)

    df = pd.read_parquet(cache_path)
    df['date'] = pd.to_datetime(df['day'])

    # Filter to signals with P&L only
    df = df[df['signal'] != 'no_trade'].copy()
    df = df[df['pnl'].notna()].copy()

    return df


def compute_metrics_by_threshold(df, thresholds):
    """Compute performance metrics for each threshold."""
    results = []

    for thresh in thresholds:
        df_thresh = df[df['edge'].abs() >= thresh].copy()

        if len(df_thresh) == 0:
            continue

        n_trades = len(df_thresh)
        mean_pnl = df_thresh['pnl'].mean()
        std_pnl = df_thresh['pnl'].std()
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
        win_rate = (df_thresh['pnl'] > 0).mean()
        total_pnl = df_thresh['pnl'].sum()

        results.append({
            'threshold': thresh,
            'n_trades': n_trades,
            'mean_pnl': mean_pnl,
            'std_pnl': std_pnl,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
        })

    return pd.DataFrame(results)


def plot_threshold_performance(df_metrics, city, save_path=None):
    """Plot threshold vs performance metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Threshold vs Sharpe Ratio
    ax1 = axes[0, 0]
    ax1.plot(df_metrics['threshold'], df_metrics['sharpe'], 'o-', linewidth=2, markersize=8)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Edge Threshold (°F)', fontsize=12)
    ax1.set_ylabel('Sharpe Ratio', fontsize=12)
    ax1.set_title(f'{city.upper()}: Threshold vs Sharpe Ratio', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Highlight best Sharpe
    best_sharpe_idx = df_metrics['sharpe'].idxmax()
    best_thresh = df_metrics.loc[best_sharpe_idx, 'threshold']
    best_sharpe = df_metrics.loc[best_sharpe_idx, 'sharpe']
    ax1.scatter([best_thresh], [best_sharpe], color='red', s=200, zorder=5, label=f'Best: {best_thresh}°F')
    ax1.legend()

    # Plot 2: Threshold vs Win Rate
    ax2 = axes[0, 1]
    ax2.plot(df_metrics['threshold'], df_metrics['win_rate'] * 100, 'o-', linewidth=2, markersize=8, color='green')
    ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% (breakeven)')
    ax2.set_xlabel('Edge Threshold (°F)', fontsize=12)
    ax2.set_ylabel('Win Rate (%)', fontsize=12)
    ax2.set_title(f'{city.upper()}: Threshold vs Win Rate', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Threshold vs Total P&L
    ax3 = axes[1, 0]
    colors = ['red' if x < 0 else 'green' for x in df_metrics['total_pnl']]
    ax3.bar(df_metrics['threshold'], df_metrics['total_pnl'], color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Edge Threshold (°F)', fontsize=12)
    ax3.set_ylabel('Total P&L ($)', fontsize=12)
    ax3.set_title(f'{city.upper()}: Threshold vs Total P&L', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Threshold vs N Trades (log scale)
    ax4 = axes[1, 1]
    ax4.semilogy(df_metrics['threshold'], df_metrics['n_trades'], 'o-', linewidth=2, markersize=8, color='purple')
    ax4.set_xlabel('Edge Threshold (°F)', fontsize=12)
    ax4.set_ylabel('Number of Trades (log scale)', fontsize=12)
    ax4.set_title(f'{city.upper()}: Threshold vs Trade Volume', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    return fig


def plot_training_window_comparison(df, city, save_path=None):
    """Compare performance across different training window sizes."""
    windows = [3, 6, 9, 12]  # months
    thresholds = np.arange(1.0, 12.5, 0.5)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics_to_plot = [
        ('sharpe', 'Sharpe Ratio', axes[0, 0]),
        ('win_rate', 'Win Rate (%)', axes[0, 1]),
        ('mean_pnl', 'Mean P&L ($)', axes[1, 0]),
        ('n_trades', 'N Trades', axes[1, 1]),
    ]

    for window in windows:
        # Filter to recent N months
        cutoff = df['date'].max() - pd.DateOffset(months=window)
        df_window = df[df['date'] >= cutoff].copy()

        # Compute metrics for each threshold
        metrics = compute_metrics_by_threshold(df_window, thresholds)

        # Plot on each subplot
        for metric_col, ylabel, ax in metrics_to_plot:
            y_values = metrics[metric_col]
            if metric_col == 'win_rate':
                y_values = y_values * 100

            ax.plot(metrics['threshold'], y_values, 'o-', label=f'{window}mo', alpha=0.7, linewidth=2)
            ax.set_xlabel('Threshold (°F)', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'{ylabel} by Training Window', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.suptitle(f'{city.upper()}: Training Window Comparison', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    return fig


def plot_temporal_performance(df, city, save_path=None):
    """Plot performance over time (rolling monthly analysis)."""
    df = df.sort_values('date')

    # Group by month
    df['year_month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('year_month').agg({
        'pnl': ['count', 'mean', 'sum'],
        'edge': 'mean'
    })

    monthly.columns = ['n_trades', 'mean_pnl', 'total_pnl', 'mean_edge']
    monthly['win_rate'] = df.groupby('year_month').apply(lambda x: (x['pnl'] > 0).mean())

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Plot 1: Monthly P&L
    ax1 = axes[0]
    colors = ['red' if x < 0 else 'green' for x in monthly['total_pnl']]
    ax1.bar(range(len(monthly)), monthly['total_pnl'], color=colors, alpha=0.7)
    ax1.set_xticks(range(len(monthly)))
    ax1.set_xticklabels([str(x) for x in monthly.index], rotation=45)
    ax1.set_ylabel('Monthly P&L ($)', fontsize=12)
    ax1.set_title(f'{city.upper()}: Monthly P&L Over Time', fontsize=14, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Monthly Win Rate
    ax2 = axes[1]
    ax2.plot(range(len(monthly)), monthly['win_rate'] * 100, 'o-', linewidth=2, markersize=6, color='blue')
    ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Breakeven')
    ax2.set_xticks(range(len(monthly)))
    ax2.set_xticklabels([str(x) for x in monthly.index], rotation=45)
    ax2.set_ylabel('Win Rate (%)', fontsize=12)
    ax2.set_title(f'{city.upper()}: Win Rate Trend', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Trade Volume
    ax3 = axes[2]
    ax3.bar(range(len(monthly)), monthly['n_trades'], alpha=0.7, color='purple')
    ax3.set_xticks(range(len(monthly)))
    ax3.set_xticklabels([str(x) for x in monthly.index], rotation=45)
    ax3.set_ylabel('Number of Trades', fontsize=12)
    ax3.set_title(f'{city.upper()}: Monthly Trade Volume', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize edge analysis")
    parser.add_argument('--city', required=True)
    parser.add_argument('--save-plots', action='store_true', help="Save plots to files")
    parser.add_argument('--threshold-min', type=float, default=1.0)
    parser.add_argument('--threshold-max', type=float, default=12.0)
    parser.add_argument('--threshold-step', type=float, default=0.25)
    args = parser.parse_args()

    print(f"Loading edge data for {args.city}...")
    df = load_edge_data(args.city)

    print(f"Loaded {len(df):,} tradeable edges")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Create output directory
    output_dir = Path(f"visualizations/{args.city}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate threshold range
    thresholds = np.arange(args.threshold_min, args.threshold_max + args.threshold_step, args.threshold_step)
    print(f"\nAnalyzing {len(thresholds)} thresholds from {args.threshold_min}°F to {args.threshold_max}°F...")

    # Compute metrics for full dataset
    df_metrics = compute_metrics_by_threshold(df, thresholds)

    # Print summary table
    print("\n" + "="*80)
    print("THRESHOLD PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Threshold':<10} {'N Trades':<10} {'Mean P&L':<12} {'Sharpe':<10} {'Win Rate':<10} {'Total P&L':<12}")
    print("-"*80)
    for _, row in df_metrics.iterrows():
        if row['n_trades'] >= 50:  # Only show thresholds with enough trades
            print(f"{row['threshold']:<10.2f} {row['n_trades']:<10,} ${row['mean_pnl']:<11.4f} "
                  f"{row['sharpe']:<10.3f} {row['win_rate']:<9.1%} ${row['total_pnl']:<11.2f}")

    # Plot 1: Threshold Performance
    save_path1 = output_dir / "threshold_performance.png" if args.save_plots else None
    plot_threshold_performance(df_metrics, args.city, save_path1)

    # Plot 2: Training Window Comparison
    save_path2 = output_dir / "training_window_comparison.png" if args.save_plots else None
    plot_training_window_comparison(df, args.city, save_path2)

    # Plot 3: Temporal Patterns
    save_path3 = output_dir / "temporal_performance.png" if args.save_plots else None
    plot_temporal_performance(df, args.city, save_path3)

    # Find optimal threshold
    # Min trades filter
    df_valid = df_metrics[df_metrics['n_trades'] >= 100].copy()
    if len(df_valid) > 0:
        best_sharpe_idx = df_valid['sharpe'].idxmax()
        best = df_valid.loc[best_sharpe_idx]

        print("\n" + "="*80)
        print("OPTIMAL THRESHOLD (min 100 trades, max Sharpe)")
        print("="*80)
        print(f"Threshold: {best['threshold']:.2f}°F")
        print(f"N Trades: {best['n_trades']:.0f}")
        print(f"Mean P&L: ${best['mean_pnl']:.4f}")
        print(f"Sharpe: {best['sharpe']:.3f}")
        print(f"Win Rate: {best['win_rate']:.1%}")
        print(f"Total P&L: ${best['total_pnl']:.2f}")

    if args.save_plots:
        print(f"\n✅ All plots saved to: {output_dir}/")
    else:
        print("\n📊 Displaying plots...")
        plt.show()


if __name__ == "__main__":
    main()
