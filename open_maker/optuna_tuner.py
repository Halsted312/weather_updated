#!/usr/bin/env python3
"""
Optuna parameter tuning for the open-maker strategy.

Optimizes parameters (entry_price, temp_bias, basis_offset) to maximize strategy performance.
Uses chronological train/test split to prevent overfitting.

Usage:
    python -m open_maker.optuna_tuner --all-cities --days 365 --trials 50
    python -m open_maker.optuna_tuner --city chicago --days 180 --metric win_rate
    python -m open_maker.optuna_tuner --all-cities --days 365 --storage sqlite:///optuna_open_maker.db
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from typing import List, Optional

import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from open_maker.core import (
    OpenMakerParams,
    run_backtest,
    print_results,
)
from src.config import CITIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Reduce optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_objective(
    cities: List[str],
    start_date: date,
    end_date: date,
    metric: str = "total_pnl",
    bet_amount_usd: float = 200.0,
):
    """
    Create an Optuna objective function for the open-maker strategy.

    Args:
        cities: List of cities to backtest
        start_date: Start date for backtest (train period)
        end_date: End date for backtest (train period)
        metric: Metric to optimize ('total_pnl', 'win_rate', 'avg_pnl', 'sharpe', 'roi')
        bet_amount_usd: Fixed bet amount for all trials

    Returns:
        Objective function for Optuna
    """

    def objective(trial: optuna.Trial) -> float:
        # Sample parameters
        params = OpenMakerParams(
            # Entry price: 30-60 cents in 5-cent increments
            entry_price_cents=trial.suggest_categorical(
                "entry_price_cents", [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
            ),
            # Temperature bias: -3 to +3 degrees
            temp_bias_deg=trial.suggest_float("temp_bias_deg", -3.0, 3.0),
            # Basis offset: 0 (same day) or 1 (previous day)
            basis_offset_days=trial.suggest_int("basis_offset_days", 0, 1),
            # Fixed bet amount
            bet_amount_usd=bet_amount_usd,
        )

        try:
            result = run_backtest(
                cities=cities,
                start_date=start_date,
                end_date=end_date,
                params=params,
                strategy_name=f"optuna_trial_{trial.number}",
            )

            if not result.trades:
                return float("-inf")

            if metric == "total_pnl":
                return result.total_pnl
            elif metric == "win_rate":
                return result.win_rate
            elif metric == "avg_pnl":
                return result.total_pnl / result.num_trades
            elif metric == "roi":
                return result.total_pnl / result.total_wagered if result.total_wagered > 0 else 0
            elif metric == "sharpe":
                # Rough Sharpe approximation
                pnls = [t.pnl_net for t in result.trades]
                if len(pnls) < 2:
                    return float("-inf")
                import statistics
                mean_pnl = statistics.mean(pnls)
                std_pnl = statistics.stdev(pnls)
                if std_pnl == 0:
                    return float("inf") if mean_pnl > 0 else float("-inf")
                return mean_pnl / std_pnl
            else:
                return result.total_pnl

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float("-inf")

    return objective


def run_optimization(
    cities: List[str],
    start_date: date,
    end_date: date,
    n_trials: int = 50,
    metric: str = "total_pnl",
    storage: Optional[str] = None,
    study_name: str = "open_maker",
    bet_amount_usd: float = 200.0,
) -> optuna.Study:
    """
    Run Optuna optimization.

    Args:
        cities: List of cities to optimize over
        start_date: Backtest start date (train period)
        end_date: Backtest end date (train period)
        n_trials: Number of optimization trials
        metric: Metric to optimize
        storage: Optional Optuna storage URL (e.g., sqlite:///optuna.db)
        study_name: Name for the Optuna study
        bet_amount_usd: Fixed bet amount

    Returns:
        Completed Optuna study
    """
    sampler = TPESampler(seed=42)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    objective = create_objective(
        cities=cities,
        start_date=start_date,
        end_date=end_date,
        metric=metric,
        bet_amount_usd=bet_amount_usd,
    )

    logger.info(f"Starting optimization: {n_trials} trials, metric={metric}")
    logger.info(f"Cities: {cities}")
    logger.info(f"Date range: {start_date} to {end_date}")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study


def print_optimization_results(study: optuna.Study) -> None:
    """Print optimization results."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.2f}")

    print("\nBest parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Parameter importance (if enough trials)
    if len(study.trials) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            print("\nParameter importance:")
            for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
                print(f"  {param}: {imp:.3f}")
        except Exception:
            pass

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Optuna parameter tuning for open-maker strategy"
    )
    parser.add_argument(
        "--city", action="append",
        help="City to optimize (can specify multiple)"
    )
    parser.add_argument(
        "--all-cities", action="store_true",
        help="Optimize across all cities"
    )
    parser.add_argument(
        "--start-date", type=str,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--days", type=int, default=180,
        help="Number of days to backtest (if start/end not specified)"
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of optimization trials"
    )
    parser.add_argument(
        "--metric", type=str, default="total_pnl",
        choices=["total_pnl", "win_rate", "avg_pnl", "sharpe", "roi"],
        help="Metric to optimize"
    )
    parser.add_argument(
        "--storage", type=str,
        help="Optuna storage URL (e.g., sqlite:///optuna.db)"
    )
    parser.add_argument(
        "--study-name", type=str, default="open_maker",
        help="Name for the Optuna study"
    )
    parser.add_argument(
        "--run-best", action="store_true",
        help="Run backtest with best parameters from existing study"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7,
        help="Fraction of data to use for training (default 0.7 = 70%% train, 30%% test)"
    )
    parser.add_argument(
        "--bet-amount", type=float, default=200.0,
        help="Bet amount in USD (default: 200)"
    )

    args = parser.parse_args()

    # Determine cities
    if args.all_cities:
        cities = list(CITIES.keys())
    elif args.city:
        cities = args.city
    else:
        cities = ["chicago"]

    # Determine date range
    today = date.today()
    if args.start_date and args.end_date:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date)
    else:
        end_date = today - timedelta(days=1)
        start_date = end_date - timedelta(days=args.days)

    # Split into train/test periods
    total_days = (end_date - start_date).days
    train_days = int(total_days * args.train_ratio)
    train_end = start_date + timedelta(days=train_days)
    test_start = train_end + timedelta(days=1)
    test_end = end_date

    logger.info(f"Train period: {start_date} to {train_end} ({train_days} days)")
    logger.info(f"Test period: {test_start} to {test_end} ({total_days - train_days - 1} days)")

    if args.run_best:
        # Load existing study and run best params on TEST period
        if not args.storage:
            print("Error: --storage required with --run-best")
            return

        study = optuna.load_study(
            study_name=args.study_name,
            storage=args.storage,
        )

        print(f"\nRunning backtest with best parameters on TEST period ({test_start} to {test_end})...")
        best_params = OpenMakerParams(
            entry_price_cents=study.best_params["entry_price_cents"],
            temp_bias_deg=study.best_params["temp_bias_deg"],
            basis_offset_days=study.best_params["basis_offset_days"],
            bet_amount_usd=args.bet_amount,
        )
        result = run_backtest(
            cities=cities,
            start_date=test_start,
            end_date=test_end,
            params=best_params,
        )
        print_results(result)
        return

    # Run optimization on TRAIN period
    study = run_optimization(
        cities=cities,
        start_date=start_date,
        end_date=train_end,  # Use train period
        n_trials=args.trials,
        metric=args.metric,
        storage=args.storage,
        study_name=args.study_name,
        bet_amount_usd=args.bet_amount,
    )

    print_optimization_results(study)

    # Run final backtest with best params on TEST period (out-of-sample)
    print(f"\n{'='*60}")
    print(f"OUT-OF-SAMPLE TEST ({test_start} to {test_end})")
    print(f"{'='*60}")

    best_params = OpenMakerParams(
        entry_price_cents=study.best_params["entry_price_cents"],
        temp_bias_deg=study.best_params["temp_bias_deg"],
        basis_offset_days=study.best_params["basis_offset_days"],
        bet_amount_usd=args.bet_amount,
    )
    test_result = run_backtest(
        cities=cities,
        start_date=test_start,
        end_date=test_end,
        params=best_params,
        strategy_name="open_maker_best_test",
    )
    print_results(test_result)

    # Also show train performance for comparison
    print(f"\n{'='*60}")
    print(f"TRAIN PERIOD COMPARISON ({start_date} to {train_end})")
    print(f"{'='*60}")
    train_result = run_backtest(
        cities=cities,
        start_date=start_date,
        end_date=train_end,
        params=best_params,
        strategy_name="open_maker_best_train",
    )
    print_results(train_result)


if __name__ == "__main__":
    main()
