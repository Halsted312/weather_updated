#!/usr/bin/env python3
"""
Walk-forward trainer for ML models (ElasticNet and CatBoost).

Walk-forward protocol (Phase 5 spec):
- Window: 90-day train → 7-day test
- Step: 7 days (non-overlapping test windows)
- Per window: tune with Optuna (40-60 trials), fit calibrated model, predict on test, save artifacts

Artifacts per window saved to: models/trained/{city}/{bracket}/win_{dates}/ (ElasticNet)
                               models/trained/{city}/{bracket}_catboost/win_{dates}/ (CatBoost)
- model_{tag}.pkl (pickled model)
- params_{tag}.json (best params + calibration method)
- preds_{tag}.csv (predictions with timestamps, tickers, p_model, y_true)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import date, timedelta
from typing import Iterable, Literal, Tuple, List, Optional
from pathlib import Path

from ml.logit_linear import WindowData, train_one_window
from ml.catboost_model import train_one_window_catboost
from ml.dataset import build_training_dataset

logger = logging.getLogger(__name__)

Bracket = Literal["between", "greater", "less"]


def windows(
    start: date,
    end: date,
    train_days: int = 90,
    test_days: int = 7,
    step_days: int = 7
) -> Iterable[Tuple[date, date, date, date]]:
    """
    Generate walk-forward windows.

    Args:
        start: Overall start date
        end: Overall end date
        train_days: Training window size (default: 90, Phase 5 spec)
        test_days: Test window size (default: 7)
        step_days: Step size (default: 7)

    Yields:
        Tuples of (train_start, train_end, test_start, test_end)
    """
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


def train_city_bracket_walkforward(
    city: str,
    bracket: Bracket,
    start: date,
    end: date,
    outdir: str = "models/trained",
    feature_set: str = "baseline",
    blend_weight: float = 0.7,
    n_trials: int = 40,
    penalties: List[str] = None,
    train_days: int = 90,
    model_type: str = "elasticnet",
    persist_features: bool = False,
    max_minutes_to_close: Optional[float] = None,
    prior_peak_back_minutes: Optional[float] = None,
    prior_peak_lookup_days: int = 3,
    prior_peak_default_hour: Optional[float] = None,
):
    """
    Run walk-forward training for a city + bracket combination.

    Args:
        city: City name (must be in CITY_CONFIG)
        bracket: Bracket type ("between", "greater", "less")
        start: Overall start date
        end: Overall end date
        outdir: Root directory for saving artifacts (default: models/trained)
        feature_set: Feature set to use (default: "baseline"; options: "ridge_conservative", "elasticnet_rich")
        blend_weight: Model weight for opinion pooling (default: 0.7; market gets 1-w)
        n_trials: Number of Optuna trials per window (default: 40 for ElasticNet, 60 for CatBoost)
        penalties: List of penalties to search (default: ["l2", "l1", "elasticnet"]) - ElasticNet only
        train_days: Training window size in days (default: 90)
        model_type: Model type to train ("elasticnet", "catboost", or "both")

    Side effects:
        Saves artifacts for each window to:
        {outdir}/{city}/{bracket}/win_{train_start}_{test_end}/ (ElasticNet)
        {outdir}/{city}/{bracket}_catboost/win_{train_start}_{test_end}/ (CatBoost)
    """
    if model_type not in ["elasticnet", "catboost", "both"]:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'elasticnet', 'catboost', or 'both'")

    need_elasticnet = model_type in ["elasticnet", "both"]
    need_catboost = model_type in ["catboost", "both"]
    prior_peak_default_minutes = (
        prior_peak_default_hour * 60 if prior_peak_default_hour is not None else None
    )

    # Adjust n_trials for CatBoost if not explicitly set
    catboost_trials = 60 if model_type in ["catboost", "both"] and n_trials == 40 else n_trials

    penalty_str = ", ".join(penalties) if penalties else "all (l2, l1, elasticnet)"
    logger.info(f"\n{'='*70}")
    logger.info(f"Walk-Forward Training: {city} / {bracket}")
    logger.info(f"Model type: {model_type.upper()}")
    logger.info(f"Date range: {start} to {end}")
    logger.info(f"Window: {train_days}→7 days, step 7 days")
    logger.info(f"Feature set: {feature_set}")
    if model_type in ["elasticnet", "both"]:
        logger.info(f"ElasticNet - Penalties: {penalty_str}, Trials: {n_trials}")
    if model_type in ["catboost", "both"]:
        logger.info(f"CatBoost - Trials: {catboost_trials}")
    logger.info(f"Blend weight: {blend_weight} (model), {1-blend_weight:.1f} (market)")
    logger.info(f"{'='*70}\n")

    os.makedirs(outdir, exist_ok=True)

    window_count = 0
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows(start, end, train_days=train_days)):
        window_count += 1
        logger.info(f"\n{'─'*70}")
        logger.info(f"Window {i+1}: Train {tr_s} to {tr_e} → Test {te_s} to {te_e}")
        logger.info(f"{'─'*70}")

        if need_elasticnet:
            # Build train dataset
            logger.info(f"Loading train data: {tr_s} to {tr_e}...")
            X_tr, y_tr, g_tr, meta_tr = build_training_dataset(
                city=city,
                start_date=tr_s,
                end_date=tr_e,
                bracket_type=bracket,
                feature_set=feature_set,
                persist_feature_snapshots=persist_features,
                max_minutes_to_close=max_minutes_to_close,
                prior_peak_back_minutes=prior_peak_back_minutes,
                prior_peak_lookup_days=prior_peak_lookup_days,
                prior_peak_default_minutes=prior_peak_default_minutes,
            )

            if len(X_tr) == 0:
                logger.warning(f"No train data for window {i+1}, skipping...")
                continue

            logger.info(f"Train: {len(X_tr)} rows ({X_tr.shape[1]} features), {len(set(g_tr))} days, "
                        f"YES={sum(y_tr)}/{len(y_tr)} ({100*sum(y_tr)/len(y_tr):.1f}%)")

            # Build test dataset
            logger.info(f"Loading test data: {te_s} to {te_e}...")
            X_te, y_te, g_te, meta_te = build_training_dataset(
                city=city,
                start_date=te_s,
                end_date=te_e,
                bracket_type=bracket,
                feature_set=feature_set,
                persist_feature_snapshots=persist_features,
                max_minutes_to_close=max_minutes_to_close,
                prior_peak_back_minutes=prior_peak_back_minutes,
                prior_peak_lookup_days=prior_peak_lookup_days,
                prior_peak_default_minutes=prior_peak_default_minutes,
            )

            if len(X_te) == 0:
                logger.warning(f"No test data for window {i+1}, skipping...")
                continue

            logger.info(f"Test: {len(X_te)} rows, {len(set(g_te))} days, "
                        f"YES={sum(y_te)}/{len(y_te)} ({100*sum(y_te)/len(y_te):.1f}%)")

            # Create WindowData
            win = WindowData(
                X_train=X_tr,
                y_train=y_tr,
                groups_train=g_tr,
                X_test=X_te,
                y_test=y_te,
                meta_test=meta_te,
            )

        # Train ElasticNet
        if model_type in ["elasticnet", "both"]:
            tag = f"{city}_{bracket}_{tr_s:%Y%m%d}_{te_e:%Y%m%d}"
            art_dir = os.path.join(outdir, city, bracket, f"win_{tr_s:%Y%m%d}_{te_e:%Y%m%d}")

            logger.info(f"Training ElasticNet model...")
            result = train_one_window(win, art_dir, tag, n_trials=n_trials, penalties=penalties)

            logger.info(f"✓ ElasticNet Window {i+1} complete:")
            logger.info(f"  Best params: {result.best_params}")
            logger.info(f"  Calibration: {result.calib_method}")
            logger.info(f"  Test log loss: {result.test_metrics['log_loss']:.4f}")
            logger.info(f"  Test Brier: {result.test_metrics['brier']:.4f}")
            logger.info(f"  Artifacts: {art_dir}")

        # Train CatBoost
        if model_type in ["catboost", "both"]:
            art_dir = Path(outdir) / city / f"{bracket}_catboost" / f"win_{tr_s:%Y%m%d}_{te_e:%Y%m%d}"
            art_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Training CatBoost model (Optuna, {catboost_trials} trials)...")
            result_cb = train_one_window_catboost(
                win_dir=art_dir,
                city=city,
                bracket=bracket,
                feature_set=feature_set,
                n_trials=catboost_trials,
                max_minutes_to_close=max_minutes_to_close,
                prior_peak_back_minutes=prior_peak_back_minutes,
                prior_peak_lookup_days=prior_peak_lookup_days,
                prior_peak_default_minutes=prior_peak_default_minutes,
            )

            status = result_cb.get("status")
            if status != "success":
                logger.warning(f"CatBoost window {i+1} failed: {result_cb}")
            else:
                metrics = result_cb.get("metrics", {})
                logger.info(f"✓ CatBoost Window {i+1} complete:")
                logger.info(f"  Brier: {metrics.get('brier_score', float('nan')):.4f}")
                logger.info(f"  Log loss: {metrics.get('log_loss', float('nan')):.4f}")
                logger.info(f"  Artifacts: {art_dir}")

    logger.info(f"\n{'='*70}")
    logger.info(f"Walk-Forward Training Complete: {window_count} windows trained")
    logger.info(f"{'='*70}\n")


def main():
    """Demo: Run walk-forward for a short date range."""
    import argparse

    parser = argparse.ArgumentParser(description="Walk-forward training (ElasticNet and/or CatBoost)")
    parser.add_argument("--city", default="chicago", help="City name")
    parser.add_argument("--bracket", choices=["between", "greater", "less"],
                        default="between", help="Bracket type")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--model-type", choices=["elasticnet", "catboost", "both"],
                        default="elasticnet", help="Model type to train (default: elasticnet)")
    parser.add_argument("--feature-set", default="baseline",
                        choices=["baseline", "ridge_conservative", "elasticnet_rich", "nextgen"],
                        help="Feature set to use (default: baseline)")
    parser.add_argument("--blend-weight", type=float, default=0.7,
                        help="Model weight for opinion pooling (default: 0.7)")
    parser.add_argument("--penalties", nargs="+", choices=["l2", "l1", "elasticnet"],
                        help="Penalties to search for ElasticNet (default: all)")
    parser.add_argument("--trials", type=int, default=40,
                        help="Optuna trials per window (default: 40 for ElasticNet, 60 for CatBoost)")
    parser.add_argument("--train-days", type=int, default=90,
                        help="Training window size in days (default: 90)")
    parser.add_argument("--outdir", default="models/trained", help="Output directory")
    parser.add_argument("--persist-features", action="store_true",
                        help="Persist engineered features to Postgres for auditing/debugging")
    parser.add_argument("--max-minutes-to-close", type=float,
                        help="Only keep samples within this many minutes of market close")
    parser.add_argument("--prior-peak-back-minutes", type=float,
                        help="Start window at (yesterday's peak time minus this many minutes)")
    parser.add_argument("--prior-peak-lookup-days", type=int, default=3,
                        help="Days to look back when searching for prior peak time (default: 3)")
    parser.add_argument("--prior-peak-default-hour", type=float,
                        help="Fallback local hour (e.g., 11 for 11:00) if no prior peak found")

    args = parser.parse_args()

    # Parse dates
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    # Run walk-forward
    train_city_bracket_walkforward(
        city=args.city,
        bracket=args.bracket,
        start=start_date,
        end=end_date,
        outdir=args.outdir,
        feature_set=args.feature_set,
        blend_weight=args.blend_weight,
        n_trials=args.trials,
        penalties=args.penalties,
        train_days=args.train_days,
        model_type=args.model_type,
        persist_features=args.persist_features,
        max_minutes_to_close=args.max_minutes_to_close,
        prior_peak_back_minutes=args.prior_peak_back_minutes,
        prior_peak_lookup_days=args.prior_peak_lookup_days,
        prior_peak_default_hour=args.prior_peak_default_hour,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
