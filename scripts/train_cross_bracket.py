#!/usr/bin/env python3
"""
Cross-bracket modeling scaffold.

Steps:
  1. Load features from feat.minute_panel_full
  2. Generate labels for 1-minute / 5-minute horizons
  3. Split by day (train/val/test)
  4. Train a baseline classifier and output metrics
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sqlalchemy import text

from db.connection import engine

LOGGER = logging.getLogger("train_cross_bracket")


def configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


def parse_args() -> argparse.Namespace:
    configure_logging()
    parser = argparse.ArgumentParser(description="Cross-bracket training scaffold")
    parser.add_argument("--city", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--horizon-min", type=int, choices=[1, 5], default=1)
    parser.add_argument("--epsilon", type=float, default=0.005, help="Ignore moves smaller than epsilon")
    parser.add_argument(
        "--model",
        choices=["logreg", "gbdt"],
        default="logreg",
        help="Model type (logistic regression or gradient boosting)",
    )
    parser.add_argument("--export-val", type=str, default=None, help="Optional CSV path to export validation probs")
    return parser.parse_args()


@dataclass
class Dataset:
    splits: Dict[str, pd.DataFrame]


def load_features(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    query = text(
        """
        SELECT *
        FROM feat.minute_panel_full
        WHERE city = :city
          AND local_date BETWEEN :start AND :end
        ORDER BY ts_utc
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"city": city, "start": start_date, "end": end_date})
    if df.empty:
        raise ValueError("No rows for given range")
    return df


def build_labels(df: pd.DataFrame, horizon_min: int, epsilon: float) -> pd.DataFrame:
    delta = pd.to_timedelta(horizon_min, unit="min")
    df = df.sort_values(["market_ticker", "ts_utc"]).copy()
    df["mid_prob_shift"] = df.groupby("market_ticker")["mid_prob"].shift(-horizon_min)
    df["delta_mid"] = df["mid_prob_shift"] - df["mid_prob"]
    df = df.dropna(subset=["mid_prob_shift"])
    df = df[np.abs(df["delta_mid"]) >= epsilon]
    df["label_dir"] = np.sign(df["delta_mid"])
    return df


def features_and_target(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    feature_cols = [
        "mid_prob",
        "mid_velocity",
        "mid_acceleration",
        "clv",
        "volume",
        "volume_delta",
        "mid_prob_left_diff",
        "mid_prob_right_diff",
        "hazard_next_5m",
        "hazard_next_60m",
        "p_wx",
        "p_mkt",
        "p_fused_norm",
    ]
    X = df[feature_cols].fillna(0.0).to_numpy()
    y = (df["label_dir"] > 0).astype(int).to_numpy()
    return X, y


def split_by_day(df: pd.DataFrame) -> Dataset:
    unique_days = sorted(df["local_date"].unique())
    n = len(unique_days)
    train_days = unique_days[: int(0.6 * n)]
    val_days = unique_days[int(0.6 * n) : int(0.8 * n)]
    test_days = unique_days[int(0.8 * n) :]

    def subset(days: List[dt.date]) -> pd.DataFrame:
        return df[df["local_date"].isin(days)]

    return Dataset(
        splits={
            "train": subset(train_days),
            "val": subset(val_days),
            "test": subset(test_days),
        }
    )


def build_model(model_name: str):
    if model_name == "gbdt":
        return GradientBoostingClassifier()
    return LogisticRegression(max_iter=1000)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return brier_score_loss(y_true, y_prob)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += np.abs(bin_conf - bin_acc) * (mask.sum() / len(y_true))
    return float(ece)


def train_model(dataset: Dataset, model_name: str, export_val: str | None = None) -> None:
    model = build_model(model_name)
    X_train, y_train = features_and_target(dataset.splits["train"])
    model.fit(X_train, y_train)

    for split_name in ["train", "val", "test"]:
        split_df = dataset.splits[split_name]
        X, y = features_and_target(split_df)
        if len(y) == 0:
            LOGGER.warning("No samples for %s split", split_name)
            continue
        y_prob = model.predict_proba(X)[:, 1]
        acc = accuracy_score(y, (y_prob > 0.5).astype(int))
        auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else float("nan")
        brier = brier_score(y, y_prob)
        ece = expected_calibration_error(y, y_prob)
        LOGGER.info("%s: acc=%.3f AUC=%.3f Brier=%.4f ECE=%.4f", split_name, acc, auc, brier, ece)

        if split_name == "val" and export_val:
            export_df = split_df[["ts_utc", "market_ticker", "local_date"]].copy()
            export_df["y_true"] = y
            export_df["y_prob"] = y_prob
            export_df.to_csv(export_val, index=False)
            LOGGER.info("Exported validation predictions to %s", export_val)


def main() -> None:
    args = parse_args()
    df = load_features(args.city, args.start_date, args.end_date)
    df = build_labels(df, args.horizon_min, args.epsilon)
    dataset = split_by_day(df)
    train_model(dataset, args.model, args.export_val)


if __name__ == "__main__":
    main()
