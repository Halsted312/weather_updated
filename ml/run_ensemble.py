#!/usr/bin/env python3
"""
Command-line helper to couple binary predictions, merge ordinal outputs,
and compute ensemble probabilities.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ml.ensembling import blend_predictions, couple_binary_predictions

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend binary and ordinal predictions")
    parser.add_argument("--binary-preds", required=True, help="CSV from binary walk-forward (preds_*.csv)")
    parser.add_argument("--ordinal-preds", required=True, help="CSV from ordinal walk-forward")
    parser.add_argument("--output", required=True, help="Destination CSV for ensemble probabilities")
    parser.add_argument("--weight", type=float, default=0.5, help="Ensemble weight on binary model")
    parser.add_argument("--method", choices=["softmax", "dirichlet"], default="softmax",
                        help="Coupling method for binary predictions")
    parser.add_argument("--tau", type=float, default=1.0, help="Temperature for softmax coupling")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    binary_df = pd.read_csv(args.binary_preds)
    ordinal_df = pd.read_csv(args.ordinal_preds)

    binary_df = couple_binary_predictions(binary_df, method=args.method, tau=args.tau)
    ensemble_df = blend_predictions(binary_df, ordinal_df, weight=args.weight)

    if "settlement_value" in ordinal_df.columns:
        merged_truth = pd.merge(
            ensemble_df,
            ordinal_df[["market_ticker", "timestamp", "bracket_key", "settlement_value"]],
            on=["market_ticker", "timestamp", "bracket_key"],
            how="left",
        )
        y_true = (merged_truth["settlement_value"] == 100.0).astype(int)
        logloss = log_loss(y_true, ensemble_df["p_ensemble"], eps=1e-6)
        brier = brier_score_loss(y_true, ensemble_df["p_ensemble"])
        logger.info("Ensemble metrics - logloss: %.4f  brier: %.4f", logloss, brier)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_df.to_csv(out_path, index=False)
    logger.info("Ensemble predictions written to %s", out_path)


if __name__ == "__main__":
    main()
