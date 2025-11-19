#!/usr/bin/env python3
"""
Walk-forward trainer for ordinal unified models.

Produces a coherent probability distribution across brackets for each timestamp
and stores predictions/artifacts similar to the binary pipelines.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.ordinal_dataset import OrdinalDataset, build_ordinal_dataset
from ml.ordinal_head import fit_cdf_models, predict_brackets, map_to_bracket_probs
from ml.train_walkforward import windows

logger = logging.getLogger(__name__)


@dataclass
class OrdinalWindowResult:
    models_path: str
    params_path: str
    preds_path: str
    metrics: dict


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _train_window(
    city: str,
    train_ds: OrdinalDataset,
    test_ds: OrdinalDataset,
    art_dir: Path,
    tag: str,
    l1_ratio: float = 0.5,
    C: float = 1.0,
) -> OrdinalWindowResult:
    """Train ordinal models for a single window and write artifacts."""
    _ensure_dir(art_dir)

    if len(train_ds.features) == 0 or len(test_ds.features) == 0:
        raise ValueError("Training or test dataset empty for ordinal window")

    taus = train_ds.taus
    if not taus:
        raise ValueError("Not enough ordinal bins to train models")

    models, calibrators = fit_cdf_models(
        df=pd.DataFrame(train_ds.features, columns=train_ds.feature_cols).assign(
            ordinal_target=train_ds.target
        ),
        feature_cols=train_ds.feature_cols,
        label_temp_col="ordinal_target",
        taus=taus,
        l1_ratio=l1_ratio,
        C=C,
    )

    joblib.dump(
        {
            "models": models,
            "calibrators": calibrators,
            "taus": taus,
            "feature_cols": train_ds.feature_cols,
        },
        art_dir / f"model_{tag}.pkl",
    )

    preds = predict_brackets(
        test_ds.features[train_ds.feature_cols].values,
        models,
        calibrators,
        taus,
    )

    preds_df = map_to_bracket_probs(
        test_ds.metadata,
        preds,
        bracket_mapping=None,
        bin_col="ordinal_bin",
    )

    preds_df["p_ordinal"] = preds_df["p_ordinal"].clip(1e-6, 1 - 1e-6)
    y_true = (preds_df["settlement_value"] == 100.0).astype(int)
    test_logloss = float(log_loss(y_true, preds_df["p_ordinal"]))
    test_brier = float(brier_score_loss(y_true, preds_df["p_ordinal"]))

    preds_path = art_dir / f"preds_{tag}.csv"
    preds_df.to_csv(preds_path, index=False)

    params = {
        "taus": taus,
        "feature_cols": train_ds.feature_cols,
        "l1_ratio": l1_ratio,
        "C": C,
        "metrics": {
            "log_loss": test_logloss,
            "brier": test_brier,
        },
    }
    with open(art_dir / f"params_{tag}.json", "w") as f:
        json.dump(params, f, indent=2)

    logger.info("Ordinal metrics - logloss: %.4f  brier: %.4f", test_logloss, test_brier)

    return OrdinalWindowResult(
        models_path=str(art_dir / f"model_{tag}.pkl"),
        params_path=str(art_dir / f"params_{tag}.json"),
        preds_path=str(preds_path),
        metrics=params["metrics"],
    )


def train_city_ordinal_walkforward(
    city: str,
    start: date,
    end: date,
    outdir: str,
    feature_set: str = "nextgen",
    l1_ratio: float = 0.5,
    C: float = 1.0,
    minutes_to_close_cutoff: int = 180,
    train_days: int = 90,
) -> List[OrdinalWindowResult]:
    """Run walk-forward ordinal training for a city."""
    results: List[OrdinalWindowResult] = []
    out_path = Path(outdir) / city / "ordinal"
    _ensure_dir(out_path)

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows(start, end, train_days=train_days)):
        logger.info(
            "Ordinal window %s: train %s→%s  test %s→%s",
            i + 1,
            tr_s,
            tr_e,
            te_s,
            te_e,
        )
        train_ds = build_ordinal_dataset(
            city=city,
            start_date=tr_s,
            end_date=tr_e,
            feature_set=feature_set,
            minutes_to_close_cutoff=minutes_to_close_cutoff,
        )
        test_ds = build_ordinal_dataset(
            city=city,
            start_date=te_s,
            end_date=te_e,
            feature_set=feature_set,
            minutes_to_close_cutoff=minutes_to_close_cutoff,
        )

        if len(train_ds.features) == 0 or len(test_ds.features) == 0:
            logger.warning("Skipping ordinal window %s due to insufficient data", i + 1)
            continue

        tag = f"{city}_ordinal_{tr_s:%Y%m%d}_{te_e:%Y%m%d}"
        art_dir = out_path / f"win_{tr_s:%Y%m%d}_{te_e:%Y%m%d}"
        result = _train_window(
            city=city,
            train_ds=train_ds,
            test_ds=test_ds,
            art_dir=art_dir,
            tag=tag,
            l1_ratio=l1_ratio,
            C=C,
        )
        results.append(result)

    logger.info("Ordinal training finished: %s windows", len(results))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward ordinal trainer")
    parser.add_argument("--city", required=True, help="City name")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--outdir", default="models/trained", help="Output directory")
    parser.add_argument("--feature-set", default="nextgen", help="Feature set (default: nextgen)")
    parser.add_argument("--l1-ratio", type=float, default=0.5, help="ElasticNet l1_ratio for ordinal CDF models")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument("--minutes-to-close", type=int, default=180,
                        help="Only use samples within this many minutes of close")
    parser.add_argument("--train-days", type=int, default=90, help="Training window length")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    train_city_ordinal_walkforward(
        city=args.city,
        start=start_date,
        end=end_date,
        outdir=args.outdir,
        feature_set=args.feature_set,
        l1_ratio=args.l1_ratio,
        C=args.C,
        minutes_to_close_cutoff=args.minutes_to_close,
        train_days=args.train_days,
    )


if __name__ == "__main__":
    main()
