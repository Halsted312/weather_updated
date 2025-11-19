#!/usr/bin/env python3
"""
Walk-forward trainer for minute-level EV CatBoost models.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.ev_dataset import build_ev_dataset
from ml.ev_catboost_model import tune_ev_catboost, fit_ev_catboost, save_ev_artifacts

logger = logging.getLogger(__name__)


def windows(
    start: date,
    end: date,
    train_days: int,
    test_days: int,
    step_days: int,
) -> Iterable[Tuple[date, date, date, date]]:
    cur = start
    while True:
        train_start = cur
        train_end = cur + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days - 1)
        if test_end > end:
            break
        yield train_start, train_end, test_start, test_end
        cur = cur + timedelta(days=step_days)


def train_city_ev_walkforward(
    city: str,
    bracket: str,
    start: date,
    end: date,
    horizon_minutes: int,
    feature_set: str,
    n_trials: int,
    train_days: int,
    test_days: int,
    step_days: int,
    outdir: Path,
    max_minutes_to_close: float | None,
    prior_peak_back_minutes: float | None,
    prior_peak_lookup_days: int,
    prior_peak_default_minutes: float | None,
) -> None:
    outdir = outdir / city / f"{bracket}_ev_catboost"
    outdir.mkdir(parents=True, exist_ok=True)

    for idx, (tr_s, tr_e, te_s, te_e) in enumerate(
        windows(start, end, train_days, test_days, step_days), start=1
    ):
        logger.info(
            "\n%s\nEV Window %d: Train %s→%s  Test %s→%s\n%s",
            "─" * 70,
            idx,
            tr_s,
            tr_e,
            te_s,
            te_e,
            "─" * 70,
        )

        train_data = build_ev_dataset(
            city=city,
            start_date=tr_s,
            end_date=tr_e,
            bracket_type=bracket,
            horizon_minutes=horizon_minutes,
            feature_set=feature_set,
            max_minutes_to_close=max_minutes_to_close,
            prior_peak_back_minutes=prior_peak_back_minutes,
            prior_peak_lookup_days=prior_peak_lookup_days,
            prior_peak_default_minutes=prior_peak_default_minutes,
        )
        X_tr, y_tr, g_tr, meta_tr, feature_cols = train_data
        if len(X_tr) == 0:
            logger.warning("No training data for window %s", idx)
            continue

        test_data = build_ev_dataset(
            city=city,
            start_date=te_s,
            end_date=te_e,
            bracket_type=bracket,
            horizon_minutes=horizon_minutes,
            feature_set=feature_set,
            max_minutes_to_close=max_minutes_to_close,
            prior_peak_back_minutes=prior_peak_back_minutes,
            prior_peak_lookup_days=prior_peak_lookup_days,
            prior_peak_default_minutes=prior_peak_default_minutes,
        )
        X_te, y_te, g_te, meta_te, _ = test_data
        if len(X_te) == 0:
            logger.warning("No test data for window %s", idx)
            continue

        X_train_df = pd.DataFrame(X_tr, columns=feature_cols)
        X_test_df = pd.DataFrame(X_te, columns=feature_cols)

        best_params, study = tune_ev_catboost(
            X_train_df,
            y_tr,
            g_tr,
            bracket,
            feature_cols,
            n_trials=n_trials,
        )
        model = fit_ev_catboost(
            X_train_df,
            y_tr,
            feature_cols,
            bracket,
            best_params,
        )

        y_pred = model.predict(X_test_df)
        rmse = float(np.sqrt(np.mean((y_pred - y_te) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_te)))

        logger.info("Window %d metrics: RMSE=%.4f MAE=%.4f", idx, rmse, mae)

        win_dir = outdir / f"win_{tr_s:%Y%m%d}_{te_e:%Y%m%d}"
        save_ev_artifacts(
            model=model,
            win_dir=win_dir,
            model_name=f"ev_catboost_{city}_{bracket}_{tr_s:%Y%m%d}_{te_e:%Y%m%d}",
            params=best_params,
            metrics={"rmse": rmse, "mae": mae},
            study=study,
        )

        preds = meta_te.copy()
        preds["pred_delta_cents"] = y_pred
        preds["actual_delta_cents"] = y_te
        preds["pred_future_mid_cents"] = preds["current_mid_cents"] + preds["pred_delta_cents"]
        preds["pred_future_mid_cents"] = preds["pred_future_mid_cents"].clip(lower=0, upper=100)
        preds["horizon_minutes"] = horizon_minutes
        preds["p_model"] = preds["pred_future_mid_cents"] / 100.0
        preds_path = win_dir / f"preds_ev_{city}_{bracket}_{tr_s:%Y%m%d}_{te_e:%Y%m%d}.csv"
        preds.to_csv(preds_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EV CatBoost models (walk-forward)")
    parser.add_argument("--city", required=True, help="City name")
    parser.add_argument("--bracket", required=True, choices=["between", "greater", "less"])
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--feature-set", default="nextgen")
    parser.add_argument("--horizon", type=int, default=60, help="Prediction horizon minutes")
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials per window")
    parser.add_argument("--outdir", default="models/trained", help="Output directory root")
    parser.add_argument("--max-minutes-to-close", type=float)
    parser.add_argument("--prior-peak-back-minutes", type=float)
    parser.add_argument("--prior-peak-lookup-days", type=int, default=3)
    parser.add_argument("--prior-peak-default-hour", type=float)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    prior_peak_default_minutes = (
        args.prior_peak_default_hour * 60 if args.prior_peak_default_hour is not None else None
    )

    train_city_ev_walkforward(
        city=args.city,
        bracket=args.bracket,
        start=start_date,
        end=end_date,
        horizon_minutes=args.horizon,
        feature_set=args.feature_set,
        n_trials=args.trials,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        outdir=Path(args.outdir),
        max_minutes_to_close=args.max_minutes_to_close,
        prior_peak_back_minutes=args.prior_peak_back_minutes,
        prior_peak_lookup_days=args.prior_peak_lookup_days,
        prior_peak_default_minutes=prior_peak_default_minutes,
    )


if __name__ == "__main__":
    main()
