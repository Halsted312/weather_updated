#!/usr/bin/env python3
"""
Optuna tuner for the peak-temp probability-vs-price strategy.

Splits train/val by time; optimizes on validation window only; then can be reused
to run the fixed best params on a held-out test period.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import Any, Dict

import optuna

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.peak_temp_strategy import run_backtest  # noqa: E402

LOGGER = logging.getLogger("tune_peak_temp")


def configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


def parse_args() -> argparse.Namespace:
    configure_logging()
    parser = argparse.ArgumentParser(description="Optuna tuner for peak_temp_strategy")
    parser.add_argument("--city", required=True)
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--val-start", required=True)
    parser.add_argument("--val-end", required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    return parser.parse_args()


def build_namespace(args: argparse.Namespace, trial: optuna.Trial, window: Dict[str, str]) -> argparse.Namespace:
    horizon = trial.suggest_categorical("hold_minutes", [20, 30, 40, 60])
    epsilon = trial.suggest_float("epsilon", 0.02, 0.08)
    hazard_min = trial.suggest_float("hazard_min", 0.0, 0.2)
    hazard_mode = trial.suggest_categorical("hazard_mode", ["any", "high", "low"])
    hazard_high_min = trial.suggest_float("hazard_high_min", 0.2, 0.6)
    hazard_low_max = trial.suggest_float("hazard_low_max", 0.05, 0.3)
    tod_start = trial.suggest_int("tod_start", 8, 12)
    tod_end = trial.suggest_int("tod_end", 15, 21)
    exit_edge_epsilon = trial.suggest_float("exit_edge_epsilon", 0.005, 0.03)
    mkt_prob_col = trial.suggest_categorical("mkt_prob_col", ["mid_prob", "close_c", "p_mkt"])

    ns = argparse.Namespace(
        city=args.city,
        start_date=window["start"],
        end_date=window["end"],
        epsilon=epsilon,
        hazard_min=hazard_min,
        hazard_mode=hazard_mode,
        hazard_high_min=hazard_high_min,
        hazard_low_max=hazard_low_max,
        tod_start=tod_start,
        tod_end=tod_end,
        hold_minutes=horizon,
        order_size=1,
        maker_slippage_cents=0,
        taker_slippage_cents=1,
        allow_taker=False,
        taker_threshold_cents=5,
        exit_edge_epsilon=exit_edge_epsilon,
        mkt_prob_col=mkt_prob_col,
    )
    return ns


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    ns = build_namespace(
        args,
        trial,
        {"start": args.val_start, "end": args.val_end},
    )
    _, summary = run_backtest(ns)
    pnl = summary["pnl_dollars"]
    dd = summary["max_drawdown_dollars"]
    # Risk-adjusted objective: minimize -pnl + 0.5*dd
    obj = -pnl + 0.5 * dd
    trial.set_user_attr("summary", summary)
    return obj


def main() -> None:
    args = parse_args()
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials)
    best = study.best_trial
    LOGGER.info("Best trial objective=%.4f params=%s summary=%s", best.value, best.params, best.user_attrs.get("summary"))


if __name__ == "__main__":
    main()
