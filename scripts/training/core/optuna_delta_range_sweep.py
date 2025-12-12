#!/usr/bin/env python3
"""
Optuna Delta Range Sweep for Market-Clock Model

This script optimizes the delta range (asymmetric) along with trading parameters.
Instead of rebuilding the dataset, it re-clips delta_raw at training time.

Key insight: The dataset already has delta_raw (unclipped). We can clip to
different ranges at training time without rebuilding.

Asymmetric range candidates (based on Chicago 2025 distribution):
- [-8, +8] to [-12, +10] with left-skew bias

Usage:
    .venv/bin/python scripts/optuna_delta_range_sweep.py \
        --input data/market_clock_chicago_2025.parquet \
        --trials 50 \
        --test-days 66
"""

import argparse
import logging
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

# Add project root to path (scripts/training/core/ -> project root is 3 levels up)
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.training.core.train_market_clock_tod_v1 import MarketClockOrdinalTrainer
from models.evaluation.metrics import compute_delta_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Candidate asymmetric delta ranges (low, high)
# Based on Chicago 2025 analysis: left-skewed distribution
DELTA_RANGE_OPTIONS = [
    (-8, 8),    # 17 classes, 79.7% coverage (baseline symmetric)
    (-9, 8),    # 18 classes, 80.6% coverage
    (-10, 8),   # 19 classes, 82.3% coverage
    (-10, 9),   # 20 classes, 83.5% coverage
    (-11, 8),   # 20 classes, 84.3% coverage
    (-11, 9),   # 21 classes, 85.5% coverage
    (-11, 10),  # 22 classes, 86.6% coverage
    (-12, 9),   # 22 classes, 85.8% coverage
    (-12, 10),  # 23 classes, 86.9% coverage
]


def clip_delta_to_range(df: pd.DataFrame, delta_low: int, delta_high: int) -> pd.DataFrame:
    """Clip delta_raw to specified range and update delta column."""
    df = df.copy()
    if 'delta_raw' in df.columns:
        df['delta'] = df['delta_raw'].clip(lower=delta_low, upper=delta_high)
    else:
        df['delta'] = df['delta'].clip(lower=delta_low, upper=delta_high)
    return df


def train_and_evaluate(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    delta_low: int,
    delta_high: int,
    n_optuna_trials: int = 0,
    quiet: bool = True,
) -> dict:
    """
    Train model with specified delta range and evaluate.

    Returns dict with accuracy, mae, within_2, model object
    """
    # Clip both train and test to the same range
    df_train_clipped = clip_delta_to_range(df_train, delta_low, delta_high)
    df_test_clipped = clip_delta_to_range(df_test, delta_low, delta_high)

    # Train model
    trainer = MarketClockOrdinalTrainer(
        n_trials=n_optuna_trials,
        verbose=not quiet,
    )
    trainer.train(df_train_clipped, df_test_clipped)

    # Evaluate
    y_true = df_test_clipped['delta'].values
    y_pred = trainer.predict(df_test_clipped)

    metrics = compute_delta_metrics(y_true, y_pred)

    return {
        'accuracy': metrics['delta_accuracy'],
        'mae': metrics['delta_mae'],
        'within_1': metrics['within_1_rate'],
        'within_2': metrics['within_2_rate'],
        'n_classes': delta_high - delta_low + 1,
        'delta_range': (delta_low, delta_high),
        'trainer': trainer,
    }


def run_optuna_sweep(
    df: pd.DataFrame,
    test_days: int,
    n_trials: int,
    output_dir: Path,
) -> dict:
    """
    Run Optuna optimization for delta range selection.

    Optimizes: delta_range (categorical from options)
    Objective: Maximize within_2 accuracy (most relevant for bracket trading)
    """
    # Time-based train/test split
    df['event_date'] = pd.to_datetime(df['event_date'])
    test_cutoff = df['event_date'].max() - pd.Timedelta(days=test_days)
    df_train = df[df['event_date'] <= test_cutoff].copy()
    df_test = df[df['event_date'] > test_cutoff].copy()

    logger.info(f"Train: {len(df_train):,} rows ({df_train['event_date'].nunique()} days)")
    logger.info(f"Test: {len(df_test):,} rows ({df_test['event_date'].nunique()} days)")
    logger.info(f"Test cutoff: {test_cutoff.date()}")

    # Track best model
    best_result = {'within_2': 0}

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_result

        # Select delta range
        range_idx = trial.suggest_categorical('delta_range_idx', list(range(len(DELTA_RANGE_OPTIONS))))
        delta_low, delta_high = DELTA_RANGE_OPTIONS[range_idx]

        # Train and evaluate
        result = train_and_evaluate(
            df_train=df_train,
            df_test=df_test,
            delta_low=delta_low,
            delta_high=delta_high,
            n_optuna_trials=0,  # No nested Optuna for speed
            quiet=True,
        )

        # Log progress
        logger.info(
            f"Trial {trial.number}: [{delta_low:+d}, {delta_high:+d}] "
            f"({result['n_classes']} classes) → "
            f"Acc={result['accuracy']:.1%}, W2={result['within_2']:.1%}, MAE={result['mae']:.2f}"
        )

        # Track best
        if result['within_2'] > best_result.get('within_2', 0):
            best_result = result
            # Save best model
            model_path = output_dir / 'best_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'trainer': result['trainer'],
                    'delta_range': result['delta_range'],
                    'metrics': {k: v for k, v in result.items() if k != 'trainer'},
                }, f)

        # Optimize for within_2 (most relevant for bracket trading)
        return result['within_2']

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
    )

    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(f"\nStarting Optuna sweep with {n_trials} trials...")
    logger.info(f"Delta range options: {DELTA_RANGE_OPTIONS}")
    logger.info("="*70)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Get best result
    best_idx = study.best_params['delta_range_idx']
    best_range = DELTA_RANGE_OPTIONS[best_idx]

    logger.info("\n" + "="*70)
    logger.info("OPTUNA SWEEP COMPLETE")
    logger.info("="*70)
    logger.info(f"Best delta range: [{best_range[0]:+d}, {best_range[1]:+d}]")
    logger.info(f"Number of classes: {best_range[1] - best_range[0] + 1}")
    logger.info(f"Best within_2: {study.best_value:.1%}")

    # Summary of all ranges tried
    logger.info("\n" + "="*70)
    logger.info("RESULTS BY DELTA RANGE")
    logger.info("="*70)

    range_results = {}
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            idx = trial.params['delta_range_idx']
            r = DELTA_RANGE_OPTIONS[idx]
            key = f"[{r[0]:+d}, {r[1]:+d}]"
            if key not in range_results or trial.value > range_results[key]:
                range_results[key] = trial.value

    for key, val in sorted(range_results.items(), key=lambda x: -x[1]):
        logger.info(f"  {key}: {val:.1%}")

    return {
        'best_range': best_range,
        'best_within_2': study.best_value,
        'best_result': best_result,
        'study': study,
        'df_train': df_train,
        'df_test': df_test,
    }


def main():
    parser = argparse.ArgumentParser(description='Optuna delta range sweep')
    parser.add_argument('--input', type=str, required=True, help='Input parquet file')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--test-days', type=int, default=66, help='Test days (20% of 331)')
    parser.add_argument('--output-dir', type=str, default='models/saved/delta_range_sweep/',
                       help='Output directory for best model')
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("="*70)
    logger.info("OPTUNA DELTA RANGE SWEEP")
    logger.info("="*70)
    logger.info(f"Input: {args.input}")
    logger.info(f"Trials: {args.trials}")
    logger.info(f"Test days: {args.test_days}")
    logger.info(f"Output: {output_dir}")

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} rows, {df['event_date'].nunique()} days")

    # Check for delta_raw
    if 'delta_raw' not in df.columns:
        logger.warning("delta_raw not found, using delta column (may already be clipped)")

    # Run sweep
    results = run_optuna_sweep(
        df=df,
        test_days=args.test_days,
        n_trials=args.trials,
        output_dir=output_dir,
    )

    # Save final summary
    summary = {
        'best_delta_range': results['best_range'],
        'best_within_2': results['best_within_2'],
        'n_trials': args.trials,
        'test_days': args.test_days,
        'input_file': args.input,
    }

    import json
    with open(output_dir / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nSummary saved to {output_dir / 'sweep_summary.json'}")
    logger.info(f"Best model saved to {output_dir / 'best_model.pkl'}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
