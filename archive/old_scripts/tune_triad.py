#!/usr/bin/env python3
"""
Optuna scaffold for triad backtest tuning.

This wraps the Python backtester (no subprocess) and returns a simple
risk-adjusted objective so we can later run sweeps for horizon/weights/thresholds.
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

from scripts.backtest_triad import run_backtest
from scripts.triad_momentum import TriadConfig  # re-exported usage in objective

LOGGER = logging.getLogger("tune_triad")


def configure_logging(level: int = logging.INFO) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


def parse_args() -> argparse.Namespace:
    configure_logging()
    parser = argparse.ArgumentParser(description="Optuna triad tuner (scaffold)")
    parser.add_argument("--city", default="chicago")
    parser.add_argument("--start-date", default="2024-11-01")
    parser.add_argument("--end-date", default="2024-11-30")
    parser.add_argument("--n-trials", type=int, default=60, help="Number of trials to run")
    return parser.parse_args()


def build_namespace(args: argparse.Namespace, trial: optuna.Trial) -> argparse.Namespace:
    horizon_min = trial.suggest_categorical("horizon_min", [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60])
    alpha_ras = trial.suggest_float("alpha_ras", 0.5, 3.0)
    alpha_accel = trial.suggest_float("alpha_accel", 0.1, 2.0)
    alpha_volume = trial.suggest_float("alpha_volume", 0.0, 1.0)
    alpha_hazard = trial.suggest_float("alpha_hazard", 0.0, 2.0)
    alpha_misprice = trial.suggest_float("alpha_misprice", 0.0, 2.0)
    min_score = trial.suggest_float("min_score", 0.2, 1.2)
    min_volume = trial.suggest_float("min_volume", 1, 12)
    triad_mass_min = trial.suggest_float("triad_mass_min", 0.02, 0.2)
    hazard_min = trial.suggest_float("hazard_min", 0.0, 0.15)
    edge_wx_min = trial.suggest_float("edge_wx_min", 0.0, 0.05)
    tod_start = trial.suggest_int("tod_start", 8, 12)
    tod_end = trial.suggest_int("tod_end", 15, 21)

    # Note: TriadConfig is constructed inside run_backtest; we pass overrides via argparse namespace.
    ns = argparse.Namespace(
        city=args.city,
        start_date=args.start_date,
        end_date=args.end_date,
        min_volume=min_volume,
        min_score=min_score,
        max_spread=5.0,
        hold_minutes=horizon_min,
        order_size=1,
        hedge_multiplier=0.5,
        maker_slippage_cents=0,
        taker_slippage_cents=1,
        allow_taker=False,
        taker_threshold_cents=0,
        implied_prob_col=None,
        calibrated_prob_col="p_up_calibrated_5m",
    )
    # store triad weights on the namespace so run_backtest can pick them up via TriadConfig defaults
    ns.alpha_ras = alpha_ras
    ns.alpha_accel = alpha_accel
    ns.alpha_volume = alpha_volume
    ns.alpha_hazard = alpha_hazard
    ns.alpha_misprice = alpha_misprice
    ns.triad_mass_min = triad_mass_min
    ns.hazard_min = hazard_min
    ns.edge_wx_min = edge_wx_min
    ns.tod_start = tod_start
    ns.tod_end = tod_end
    return ns


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    ns = build_namespace(args, trial)

    # Monkey-patch TriadConfig defaults for this trial.
    cfg = TriadConfig(
        min_volume=ns.min_volume,
        min_score=ns.min_score,
        max_spread_cents=ns.max_spread,
        alpha_ras=ns.alpha_ras,
        alpha_accel=ns.alpha_accel,
        alpha_volume=ns.alpha_volume,
        alpha_hazard=ns.alpha_hazard,
        alpha_misprice=ns.alpha_misprice,
        hazard_min=ns.hazard_min,
        triad_mass_min=ns.triad_mass_min,
        edge_wx_min=ns.edge_wx_min,
        tod_start=ns.tod_start,
        tod_end=ns.tod_end,
    )

    # run_backtest builds its own TriadConfig; we pass overrides via namespace fields.
    trades, summary = run_backtest(ns, triad_cfg=cfg)
    pnl = summary["pnl_dollars"]
    dd = summary["max_drawdown_dollars"]
    obj = -pnl + 0.5 * dd
    trial.set_user_attr("summary", summary)
    return obj


def main() -> None:
    args = parse_args()

    def _objective(trial: optuna.Trial) -> float:
        return objective(trial, args)

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=args.n_trials)
    best = study.best_trial
    LOGGER.info("Best trial objective=%.4f params=%s summary=%s", best.value, best.params, best.user_attrs.get("summary"))


if __name__ == "__main__":
    main()
