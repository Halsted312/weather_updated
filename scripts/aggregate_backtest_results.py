#!/usr/bin/env python3
"""
Aggregate backtest results across multiple walk-forward windows.

This script:
1. Loads all individual window backtest results
2. Calculates aggregate metrics (mean/std Sharpe, total P&L, etc.)
3. Compares ElasticNet vs CatBoost performance
4. Generates a comprehensive report
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def load_window_results(results_dir: Path, model_type: str) -> List[Dict[str, Any]]:
    """Load all window results for a given model type."""
    pattern = f"{model_type}_win*.json"
    result_files = sorted(results_dir.glob(pattern))

    results = []
    for filepath in result_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                data['window_file'] = filepath.name
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    return results


def calculate_aggregate_metrics(results: List[Dict[str, Any]], model_type: str) -> Dict[str, Any]:
    """Calculate aggregate metrics across all windows."""
    if not results:
        return {
            "model_type": model_type,
            "n_windows": 0,
            "error": "No results found"
        }

    # Extract metrics
    sharpes = []
    pnls = []
    fees = []
    n_trades = []
    briers = []
    drawdowns = []
    returns = []

    for r in results:
        sharpes.append(r.get("sharpe", 0))
        pnls.append(r.get("total_pnl_cents", 0) / 100)
        fees.append(r.get("total_fees_cents", 0) / 100)
        n_trades.append(r.get("n_trades", 0))

        # Handle different formats for Brier score
        brier = r.get("ece_trades") or r.get("ece_all_minutes")
        if brier is not None:
            briers.append(brier)

        dd = r.get("max_drawdown", 0)
        if isinstance(dd, (int, float)):
            drawdowns.append(abs(dd))

        # Calculate return percentage
        initial_capital = 10000  # Default $10,000
        ret = (r.get("total_pnl_cents", 0) / 100) / initial_capital * 100
        returns.append(ret)

    # Calculate statistics
    aggregate = {
        "model_type": model_type,
        "n_windows": len(results),
        "n_test_days": len(results) * 7,  # 7 days per window

        # Sharpe ratio
        "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_std": float(np.std(sharpes)),
        "sharpe_min": float(np.min(sharpes)),
        "sharpe_max": float(np.max(sharpes)),

        # P&L
        "pnl_total": float(np.sum(pnls)),
        "pnl_mean": float(np.mean(pnls)),
        "pnl_std": float(np.std(pnls)),
        "pnl_positive_windows": int(np.sum([p > 0 for p in pnls])),
        "pnl_win_rate": float(np.sum([p > 0 for p in pnls]) / len(pnls) * 100),

        # Returns
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_total": float(np.sum(returns)),

        # Fees
        "fees_total": float(np.sum(fees)),
        "fees_mean": float(np.mean(fees)),
        "fee_ratio": float(np.sum(fees) / np.sum([abs(p) for p in pnls]) * 100) if np.sum(pnls) != 0 else 0,

        # Trading activity
        "trades_total": int(np.sum(n_trades)),
        "trades_mean": float(np.mean(n_trades)),
        "trades_per_day": float(np.sum(n_trades) / (len(results) * 7)),

        # Calibration
        "brier_mean": float(np.mean(briers)) if briers else None,
        "brier_std": float(np.std(briers)) if briers else None,

        # Risk
        "max_drawdown_mean": float(np.mean(drawdowns)) if drawdowns else None,
        "max_drawdown_worst": float(np.max(drawdowns)) if drawdowns else None,
    }

    return aggregate


def compare_models(elasticnet_agg: Dict, catboost_agg: Dict) -> pd.DataFrame:
    """Create a comparison table between models."""
    comparison_data = {
        'Metric': [],
        'ElasticNet': [],
        'CatBoost': [],
        'Winner': [],
        'Improvement': []
    }

    # Define metrics to compare
    metrics = [
        ('Windows Tested', 'n_windows', 'equal'),
        ('Mean Sharpe', 'sharpe_mean', 'higher'),
        ('Sharpe Std Dev', 'sharpe_std', 'lower'),
        ('Total P&L ($)', 'pnl_total', 'higher'),
        ('Mean P&L ($)', 'pnl_mean', 'higher'),
        ('Win Rate (%)', 'pnl_win_rate', 'higher'),
        ('Total Return (%)', 'return_total', 'higher'),
        ('Total Fees ($)', 'fees_total', 'lower'),
        ('Fee Ratio (%)', 'fee_ratio', 'lower'),
        ('Total Trades', 'trades_total', 'neutral'),
        ('Mean Brier', 'brier_mean', 'lower'),
        ('Mean Max DD (%)', 'max_drawdown_mean', 'lower'),
    ]

    for display_name, key, better in metrics:
        en_val = elasticnet_agg.get(key)
        cb_val = catboost_agg.get(key)

        comparison_data['Metric'].append(display_name)

        # Format values
        if en_val is None:
            comparison_data['ElasticNet'].append('N/A')
        elif key in ['n_windows', 'trades_total']:
            comparison_data['ElasticNet'].append(f"{en_val:.0f}")
        elif 'pnl' in key or 'fees' in key:
            comparison_data['ElasticNet'].append(f"{en_val:.2f}")
        else:
            comparison_data['ElasticNet'].append(f"{en_val:.3f}")

        if cb_val is None:
            comparison_data['CatBoost'].append('N/A')
        elif key in ['n_windows', 'trades_total']:
            comparison_data['CatBoost'].append(f"{cb_val:.0f}")
        elif 'pnl' in key or 'fees' in key:
            comparison_data['CatBoost'].append(f"{cb_val:.2f}")
        else:
            comparison_data['CatBoost'].append(f"{cb_val:.3f}")

        # Determine winner
        if en_val is None or cb_val is None or better == 'neutral':
            comparison_data['Winner'].append('-')
            comparison_data['Improvement'].append('-')
        else:
            if better == 'higher':
                if cb_val > en_val:
                    comparison_data['Winner'].append('CatBoost ↑')
                    imp = ((cb_val - en_val) / abs(en_val) * 100) if en_val != 0 else 0
                    comparison_data['Improvement'].append(f"+{imp:.1f}%")
                elif en_val > cb_val:
                    comparison_data['Winner'].append('ElasticNet ↑')
                    imp = ((en_val - cb_val) / abs(cb_val) * 100) if cb_val != 0 else 0
                    comparison_data['Improvement'].append(f"+{imp:.1f}%")
                else:
                    comparison_data['Winner'].append('Tie')
                    comparison_data['Improvement'].append('0.0%')
            elif better == 'lower':
                if cb_val < en_val:
                    comparison_data['Winner'].append('CatBoost ↓')
                    imp = ((en_val - cb_val) / abs(en_val) * 100) if en_val != 0 else 0
                    comparison_data['Improvement'].append(f"-{imp:.1f}%")
                elif en_val < cb_val:
                    comparison_data['Winner'].append('ElasticNet ↓')
                    imp = ((cb_val - en_val) / abs(cb_val) * 100) if cb_val != 0 else 0
                    comparison_data['Improvement'].append(f"-{imp:.1f}%")
                else:
                    comparison_data['Winner'].append('Tie')
                    comparison_data['Improvement'].append('0.0%')
            else:
                comparison_data['Winner'].append('Tie')
                comparison_data['Improvement'].append('-')

    return pd.DataFrame(comparison_data)


def generate_report(
    elasticnet_results: List[Dict],
    catboost_results: List[Dict],
    elasticnet_agg: Dict,
    catboost_agg: Dict,
    output_dir: Path
):
    """Generate comprehensive report."""
    print("\n" + "="*80)
    print("MULTI-WINDOW BACKTEST RESULTS")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {output_dir}")

    # Overall summary
    print("\n" + "-"*80)
    print("OVERALL SUMMARY")
    print("-"*80)

    total_windows = max(elasticnet_agg['n_windows'], catboost_agg['n_windows'])
    print(f"Windows tested: {total_windows}")
    print(f"Test days per window: 7")
    print(f"Total test coverage: {total_windows * 7} days")

    # ElasticNet summary
    print("\n" + "-"*80)
    print("ELASTICNET AGGREGATE METRICS")
    print("-"*80)

    if elasticnet_agg['n_windows'] > 0:
        print(f"Windows: {elasticnet_agg['n_windows']}")
        print(f"Sharpe: {elasticnet_agg['sharpe_mean']:.2f} ± {elasticnet_agg['sharpe_std']:.2f} (range: {elasticnet_agg['sharpe_min']:.2f} to {elasticnet_agg['sharpe_max']:.2f})")
        print(f"Total P&L: ${elasticnet_agg['pnl_total']:.2f} | Mean: ${elasticnet_agg['pnl_mean']:.2f}/window")
        print(f"Win rate: {elasticnet_agg['pnl_win_rate']:.1f}% ({elasticnet_agg['pnl_positive_windows']}/{elasticnet_agg['n_windows']} profitable)")
        print(f"Total return: {elasticnet_agg['return_total']:.1f}% | Mean: {elasticnet_agg['return_mean']:.1f}%/window")
        print(f"Total trades: {elasticnet_agg['trades_total']} | Mean: {elasticnet_agg['trades_mean']:.1f}/window")
        if elasticnet_agg['brier_mean'] is not None:
            print(f"Brier score: {elasticnet_agg['brier_mean']:.4f} ± {elasticnet_agg['brier_std']:.4f}")
        print(f"Fee ratio: {elasticnet_agg['fee_ratio']:.1f}%")
    else:
        print("No ElasticNet results found")

    # CatBoost summary
    print("\n" + "-"*80)
    print("CATBOOST AGGREGATE METRICS")
    print("-"*80)

    if catboost_agg['n_windows'] > 0:
        print(f"Windows: {catboost_agg['n_windows']}")
        print(f"Sharpe: {catboost_agg['sharpe_mean']:.2f} ± {catboost_agg['sharpe_std']:.2f} (range: {catboost_agg['sharpe_min']:.2f} to {catboost_agg['sharpe_max']:.2f})")
        print(f"Total P&L: ${catboost_agg['pnl_total']:.2f} | Mean: ${catboost_agg['pnl_mean']:.2f}/window")
        print(f"Win rate: {catboost_agg['pnl_win_rate']:.1f}% ({catboost_agg['pnl_positive_windows']}/{catboost_agg['n_windows']} profitable)")
        print(f"Total return: {catboost_agg['return_total']:.1f}% | Mean: {catboost_agg['return_mean']:.1f}%/window")
        print(f"Total trades: {catboost_agg['trades_total']} | Mean: {catboost_agg['trades_mean']:.1f}/window")
        if catboost_agg['brier_mean'] is not None:
            print(f"Brier score: {catboost_agg['brier_mean']:.4f} ± {catboost_agg['brier_std']:.4f}")
        print(f"Fee ratio: {catboost_agg['fee_ratio']:.1f}%")
    else:
        print("No CatBoost results found")

    # Model comparison
    if elasticnet_agg['n_windows'] > 0 and catboost_agg['n_windows'] > 0:
        print("\n" + "-"*80)
        print("MODEL COMPARISON")
        print("-"*80)

        comparison_df = compare_models(elasticnet_agg, catboost_agg)
        print("\n" + comparison_df.to_string(index=False))

    # Per-window details
    print("\n" + "-"*80)
    print("PER-WINDOW DETAILS")
    print("-"*80)

    # Create window comparison
    window_data = []
    for i in range(max(len(elasticnet_results), len(catboost_results))):
        row = {"Window": i + 1}

        if i < len(elasticnet_results):
            en = elasticnet_results[i]
            row["EN_Sharpe"] = f"{en.get('sharpe', 0):.2f}"
            row["EN_P&L"] = f"${en.get('total_pnl_cents', 0) / 100:.0f}"
            row["EN_Trades"] = en.get('n_trades', 0)
        else:
            row["EN_Sharpe"] = "-"
            row["EN_P&L"] = "-"
            row["EN_Trades"] = "-"

        if i < len(catboost_results):
            cb = catboost_results[i]
            row["CB_Sharpe"] = f"{cb.get('sharpe', 0):.2f}"
            row["CB_P&L"] = f"${cb.get('total_pnl_cents', 0) / 100:.0f}"
            row["CB_Trades"] = cb.get('n_trades', 0)
        else:
            row["CB_Sharpe"] = "-"
            row["CB_P&L"] = "-"
            row["CB_Trades"] = "-"

        window_data.append(row)

    window_df = pd.DataFrame(window_data)
    print("\n" + window_df.to_string(index=False))

    # Trading recommendations
    print("\n" + "-"*80)
    print("RECOMMENDATIONS")
    print("-"*80)

    # Check if models pass production criteria
    en_pass = (elasticnet_agg['sharpe_mean'] >= 2.0 and
               elasticnet_agg['pnl_win_rate'] >= 60 and
               (elasticnet_agg['brier_mean'] <= 0.09 if elasticnet_agg['brier_mean'] else True))

    cb_pass = (catboost_agg['sharpe_mean'] >= 2.0 and
               catboost_agg['pnl_win_rate'] >= 60 and
               (catboost_agg['brier_mean'] <= 0.09 if catboost_agg['brier_mean'] else True))

    if en_pass and cb_pass:
        print("✓ BOTH models pass production criteria (Sharpe ≥ 2.0, Win rate ≥ 60%, Brier ≤ 0.09)")
        if catboost_agg['sharpe_mean'] > elasticnet_agg['sharpe_mean']:
            print("→ Recommendation: Deploy CatBoost (higher Sharpe)")
        else:
            print("→ Recommendation: Deploy ElasticNet (simpler model, comparable performance)")
    elif en_pass:
        print("✓ ElasticNet passes production criteria")
        print("✗ CatBoost does not meet all criteria")
        print("→ Recommendation: Deploy ElasticNet")
    elif cb_pass:
        print("✗ ElasticNet does not meet all criteria")
        print("✓ CatBoost passes production criteria")
        print("→ Recommendation: Deploy CatBoost")
    else:
        print("⚠ Neither model meets all production criteria")
        print("→ Recommendation: Continue development or reduce position sizing")

    print("\nNext steps:")
    print("1. Review per-window performance for consistency")
    print("2. Start shadow testing with recommended model")
    print("3. Begin live trading with 10% of intended bankroll")
    print("4. Monitor daily and increase position size gradually")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-window backtest results"
    )
    parser.add_argument(
        "--results-dir",
        default="results/multiwindow",
        help="Directory containing window result JSONs"
    )
    parser.add_argument(
        "--output",
        default="aggregated_summary.json",
        help="Output filename for aggregated summary"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        print("Please run the multi-window backtest first")
        sys.exit(1)

    # Load all results
    elasticnet_results = load_window_results(results_dir, "elasticnet")
    catboost_results = load_window_results(results_dir, "catboost")

    if not elasticnet_results and not catboost_results:
        print("Error: No results found in", results_dir)
        sys.exit(1)

    # Calculate aggregate metrics
    elasticnet_agg = calculate_aggregate_metrics(elasticnet_results, "elasticnet")
    catboost_agg = calculate_aggregate_metrics(catboost_results, "catboost")

    # Save aggregated summary
    summary = {
        "generated": datetime.now().isoformat(),
        "elasticnet": elasticnet_agg,
        "catboost": catboost_agg,
    }

    output_path = results_dir / args.output
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved aggregated summary to: {output_path}")

    # Generate report
    generate_report(
        elasticnet_results,
        catboost_results,
        elasticnet_agg,
        catboost_agg,
        results_dir
    )


if __name__ == "__main__":
    main()