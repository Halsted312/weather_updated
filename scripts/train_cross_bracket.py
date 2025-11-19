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
from sklearn.metrics import accuracy_score, roc_auc_score
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
    return parser.parse_args()


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


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


def build_labels(df: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    delta = pd.to_timedelta(horizon_min, unit="min")
    df = df.sort_values(["market_ticker", "ts_utc"]).copy()
    df["mid_prob_shift"] = df.groupby("market_ticker")["mid_prob"].shift(-horizon_min)
    df["label_dir"] = np.sign(df["mid_prob_shift"] - df["mid_prob"]).fillna(0)
    df = df.dropna(subset=["mid_prob_shift"])
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

    X_train, y_train = features_and_target(subset(train_days))
    X_val, y_val = features_and_target(subset(val_days))
    X_test, y_test = features_and_target(subset(test_days))

    return Dataset(X_train, y_train, X_val, y_val, X_test, y_test)


def train_baseline(dataset: Dataset) -> None:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(dataset.X_train, dataset.y_train)

    for split, X, y in [
        ("train", dataset.X_train, dataset.y_train),
        ("val", dataset.X_val, dataset.y_val),
        ("test", dataset.X_test, dataset.y_test),
    ]:
        if len(y) == 0:
            LOGGER.warning("No samples for %s split", split)
            continue
        y_prob = clf.predict_proba(X)[:, 1]
        acc = accuracy_score(y, (y_prob > 0.5).astype(int))
        auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else float("nan")
        LOGGER.info("%s: acc=%.3f AUC=%.3f", split, acc, auc)


def main() -> None:
    args = parse_args()
    df = load_features(args.city, args.start_date, args.end_date)
    df = build_labels(df, args.horizon_min)
    dataset = split_by_day(df)
    train_baseline(dataset)


if __name__ == "__main__":
    main()
