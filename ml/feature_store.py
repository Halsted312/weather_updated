#!/usr/bin/env python3
"""
Utility helpers for persisting engineered features into Postgres.

Snapshots make it easier to debug models, audit features, and reuse
expensive calculations across experiments.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from db.connection import get_session
from db.models import FeatureSnapshot

logger = logging.getLogger(__name__)


def _to_native(value):
    """Convert numpy/pandas scalars to JSON-serializable Python types."""
    if isinstance(value, (np.floating, np.float32, np.float64)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _prepare_records(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    city: str,
    feature_set: str,
) -> List[dict]:
    """Transform a feature DataFrame into FeatureSnapshot rows."""
    records: List[dict] = []

    required_cols = {"market_ticker", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Feature snapshot DataFrame missing columns: {missing}")

    for _, row in df.iterrows():
        ts = pd.to_datetime(row["timestamp"], utc=True)

        feature_payload = {}
        for col in feature_cols:
            if col not in df.columns:
                continue  # Column was optional/missing for this feature set
            feature_payload[col] = _to_native(row[col])

        records.append({
            "city": city,
            "market_ticker": row["market_ticker"],
            "timestamp": ts.to_pydatetime(),
            "feature_set": feature_set,
            "features": feature_payload,
            "created_at": datetime.utcnow(),
        })

    return records


def persist_feature_snapshots(
    city: str,
    feature_set: str,
    features_df: pd.DataFrame,
    feature_cols: Sequence[str],
    chunk_size: int = 1000,
) -> int:
    """
    Persist engineered features to Postgres for later reuse/debugging.

    Args:
        city: City name
        feature_set: Feature set identifier
        features_df: DataFrame containing at least market_ticker/timestamp + features
        feature_cols: Columns to store inside the JSON payload
        chunk_size: Batch size for inserts

    Returns:
        Number of rows persisted (inserted or updated)
    """
    if features_df.empty:
        logger.info("No features to persist for %s/%s", city, feature_set)
        return 0

    features_df = (
        features_df
        .drop_duplicates(subset=["market_ticker", "timestamp"])
        .reset_index(drop=True)
    )

    total_written = 0
    with get_session() as session:
        for start in range(0, len(features_df), chunk_size):
            batch = features_df.iloc[start:start + chunk_size]
            records = _prepare_records(batch, feature_cols, city, feature_set)
            if not records:
                continue

            stmt = insert(FeatureSnapshot).values(records)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_feature_snapshot",
                set_={
                    "features": stmt.excluded.features,
                    "feature_set": stmt.excluded.feature_set,
                    "city": stmt.excluded.city,
                    "created_at": stmt.excluded.created_at,
                },
            )
            session.execute(stmt)
            total_written += len(records)

        session.commit()

    logger.info(
        "Persisted %s feature snapshots for %s (%s set)",
        total_written,
        city,
        feature_set,
    )
    return total_written
