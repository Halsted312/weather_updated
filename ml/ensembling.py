#!/usr/bin/env python3
"""
Utilities for coupling binary predictions and blending them with ordinal outputs.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from ml.unified_head import couple_timestamp_rowset

logger = logging.getLogger(__name__)


def couple_binary_predictions(
    preds_df: pd.DataFrame,
    p_col: str = "p_model",
    method: Literal["softmax", "dirichlet"] = "softmax",
    tau: float = 1.0,
) -> pd.DataFrame:
    """
    Apply unified-head coupling to per-bracket binary predictions so that
    each timestamp has a coherent distribution.
    """
    required_cols = {"event_date", "timestamp", "bracket_key", p_col}
    missing = required_cols - set(preds_df.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")

    df = preds_df.copy()
    df["p_coupled"] = np.nan

    for (_, _), group in df.groupby(["event_date", "timestamp"]):
        if len(group) != 6:
            logger.debug("Skipping coupling for %s rows (expected 6)", len(group))
            continue
        coupled = couple_timestamp_rowset(
            group,
            p_col=p_col,
            method=method,
            tau=tau,
        )
        df.loc[group.index, "p_coupled"] = coupled

    df["p_coupled"] = df["p_coupled"].fillna(df[p_col])
    return df


def blend_predictions(
    binary_df: pd.DataFrame,
    ordinal_df: pd.DataFrame,
    weight: float = 0.5,
    binary_col: str = "p_coupled",
    ordinal_col: str = "p_ordinal",
) -> pd.DataFrame:
    """
    Merge binary-coupled and ordinal predictions into a single ensemble.
    """
    merge_cols = ["market_ticker", "timestamp", "bracket_key"]
    for col in merge_cols:
        if col not in binary_df.columns or col not in ordinal_df.columns:
            raise ValueError(f"Missing merge column {col} in predictions")

    merged = pd.merge(
        binary_df[merge_cols + [binary_col]],
        ordinal_df[merge_cols + [ordinal_col]],
        on=merge_cols,
        how="inner",
        suffixes=("_bin", "_ord"),
    )

    merged["p_ensemble"] = np.clip(
        weight * merged[binary_col] + (1 - weight) * merged[ordinal_col],
        0.0,
        1.0,
    )
    return merged
