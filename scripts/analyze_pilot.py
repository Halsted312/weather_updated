#!/usr/bin/env python3
"""
Analyze pilot training results and generate summary metrics + calibration curves.

Usage:
    python scripts/analyze_pilot.py --pilot-dir models/pilots/chicago/between --output-dir models/pilots/chicago/between
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import joblib

from ml.eval import compute_ece, calibration_summary


def count_nonzero_coefficients(model_path: str) -> int:
    """
    Extract number of non-zero coefficients from a trained model.

    Args:
        model_path: Path to joblib model file

    Returns:
        Number of non-zero coefficients
    """
    model = joblib.load(model_path)

    # CalibratedClassifierCV wraps the base estimator
    # Navigate: CalibratedClassifierCV -> calibrated_classifiers_ -> estimator -> Pipeline -> LogisticRegression
    if hasattr(model, 'calibrated_classifiers_'):
        base_estimator = model.calibrated_classifiers_[0].estimator
    else:
        base_estimator = model

    # Extract LogisticRegression from Pipeline
    if hasattr(base_estimator, 'named_steps'):
        clf = base_estimator.named_steps['clf']
    else:
        clf = base_estimator

    # Get coefficients
    if hasattr(clf, 'coef_'):
        coef = clf.coef_[0]  # Binary classification, shape (1, n_features)
        return int(np.sum(np.abs(coef) > 1e-8))  # Count non-zero with threshold

    return 0


def analyze_pilot_windows(pilot_dir: Path) -> Dict:
    """
    Aggregate metrics across all walk-forward windows for one pilot.

    Args:
        pilot_dir: Root directory containing window subdirectories

    Returns:
        Dict with aggregated metrics
    """
    window_dirs = sorted([d for d in pilot_dir.iterdir() if d.is_dir() and d.name.startswith('win_')])

    metrics_list = []

    for win_dir in window_dirs:
        # Find preds CSV and model
        preds_files = list(win_dir.glob('preds_*.csv'))
        model_files = list(win_dir.glob('model_*.pkl'))
        params_files = list(win_dir.glob('params_*.json'))

        if not preds_files or not model_files:
            print(f"Warning: Skipping {win_dir.name} - missing files")
            continue

        # Load predictions
        preds_df = pd.read_csv(preds_files[0])
        y_true = preds_df['y_true'].values
        p_model = preds_df['p_model'].values

        # Compute ECE
        ece = compute_ece(y_true, p_model, n_bins=10, strategy='uniform')

        # Load params to get test metrics
        with open(params_files[0]) as f:
            params_data = json.load(f)

        # Count non-zero coefficients
        n_nonzero = count_nonzero_coefficients(str(model_files[0]))

        # Extract penalty type
        penalty = params_data.get('best_params', {}).get('penalty', 'unknown')

        # Compute basic metrics (log_loss, brier already in preds via test_metrics)
        from sklearn.metrics import log_loss, brier_score_loss
        ll = log_loss(y_true, p_model, eps=1e-6)
        brier = brier_score_loss(y_true, p_model)

        metrics_list.append({
            'window': win_dir.name,
            'n_test': len(y_true),
            'log_loss': ll,
            'brier': brier,
            'ece': ece,
            'n_nonzero_coef': n_nonzero,
            'penalty': penalty,
        })

    # Aggregate
    if not metrics_list:
        return {}

    df = pd.DataFrame(metrics_list)

    summary = {
        'n_windows': len(metrics_list),
        'total_test_rows': int(df['n_test'].sum()),
        'log_loss_mean': float(df['log_loss'].mean()),
        'log_loss_std': float(df['log_loss'].std()),
        'brier_mean': float(df['brier'].mean()),
        'brier_std': float(df['brier'].std()),
        'ece_mean': float(df['ece'].mean()),
        'ece_std': float(df['ece'].std()),
        'penalty': metrics_list[0]['penalty'],
        'per_window_metrics': metrics_list,
    }

    # Add coefficient stats if available (for EN)
    if df['n_nonzero_coef'].sum() > 0:
        summary['n_nonzero_coef_mean'] = float(df['n_nonzero_coef'].mean())
        summary['n_nonzero_coef_std'] = float(df['n_nonzero_coef'].std())

    return summary


def generate_calibration_curve(pilot_dir: Path, output_path: Path):
    """
    Generate calibration curve for the LAST test window.

    Args:
        pilot_dir: Root directory containing window subdirectories
        output_path: Path to save calibration.json
    """
    window_dirs = sorted([d for d in pilot_dir.iterdir() if d.is_dir() and d.name.startswith('win_')])

    if not window_dirs:
        print("No windows found")
        return

    # Use last window
    last_win = window_dirs[-1]
    preds_files = list(last_win.glob('preds_*.csv'))

    if not preds_files:
        print(f"No predictions found in {last_win.name}")
        return

    # Load predictions
    preds_df = pd.read_csv(preds_files[0])
    y_true = preds_df['y_true'].values
    p_model = preds_df['p_model'].values

    # Compute calibration summary
    cal_summary = calibration_summary(y_true, p_model, n_bins=10, strategy='uniform')

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(cal_summary, f, indent=2)

    print(f"Saved calibration curve: {output_path}")


def generate_sample_predictions(pilot_dir: Path, output_path: Path, n_samples: int = 1000):
    """
    Generate sample predictions CSV from the last test window.

    Args:
        pilot_dir: Root directory containing window subdirectories
        output_path: Path to save sample_preds.csv
        n_samples: Number of samples to include
    """
    window_dirs = sorted([d for d in pilot_dir.iterdir() if d.is_dir() and d.name.startswith('win_')])

    if not window_dirs:
        return

    # Use last window
    last_win = window_dirs[-1]
    preds_files = list(last_win.glob('preds_*.csv'))

    if not preds_files:
        return

    # Load predictions
    preds_df = pd.read_csv(preds_files[0])

    # Sample
    if len(preds_df) > n_samples:
        preds_df = preds_df.sample(n=n_samples, random_state=42)

    # Save subset of columns
    cols_to_save = ['timestamp', 'market_ticker', 'p_model', 'y_true']

    # Add event_date and market_ticker if available
    if 'event_date' in preds_df.columns:
        cols_to_save.insert(1, 'event_date')

    preds_df[cols_to_save].to_csv(output_path, index=False)
    print(f"Saved sample predictions: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze pilot training results")
    parser.add_argument('--pilot-dir', type=str, required=True, help='Directory containing pilot windows')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for summary files')
    parser.add_argument('--feature-set', type=str, required=True, help='Feature set name (ridge_conservative or elasticnet_rich)')

    args = parser.parse_args()

    pilot_dir = Path(args.pilot_dir)
    output_dir = Path(args.output_dir) / args.feature_set
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze windows
    print(f"\n{'='*60}")
    print(f"Analyzing pilot: {args.feature_set}")
    print(f"{'='*60}\n")

    summary = analyze_pilot_windows(pilot_dir)

    if not summary:
        print("No metrics found!")
        return

    # Save summary
    summary_path = output_dir / 'metrics_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved metrics summary: {summary_path}")
    print(f"\nSummary:")
    print(f"  Windows: {summary['n_windows']}")
    print(f"  Log loss: {summary['log_loss_mean']:.4f} ± {summary['log_loss_std']:.4f}")
    print(f"  Brier:    {summary['brier_mean']:.4f} ± {summary['brier_std']:.4f}")
    print(f"  ECE:      {summary['ece_mean']:.4f} ± {summary['ece_std']:.4f}")

    if 'n_nonzero_coef_mean' in summary:
        print(f"  Non-zero coef: {summary['n_nonzero_coef_mean']:.1f} ± {summary['n_nonzero_coef_std']:.1f}")

    # Generate calibration curve
    cal_path = output_dir / 'calibration.json'
    generate_calibration_curve(pilot_dir, cal_path)

    # Generate sample predictions
    sample_path = output_dir / 'sample_predictions.csv'
    generate_sample_predictions(pilot_dir, sample_path, n_samples=1000)

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
