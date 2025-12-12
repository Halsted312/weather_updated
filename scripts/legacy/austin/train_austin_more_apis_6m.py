"""
Train a quick CatBoost ordinal model for Austin using the last ~6 months of data,
with the new weather_more_apis (NBM + HRRR) features turned on.

- City: austin
- Date window: (today - 6 months) .. (yesterday)
- Split: time-based 80/20 (reuse your existing splitter)
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.db.connection import get_db_session
from models.data.dataset import DatasetConfig, build_dataset  # adjust import if needed
from models.features.base import NUMERIC_FEATURE_COLS  # includes weather_more_apis features

# TODO: import your existing 80/20 time-based split helper.
# For example, if it's in models/data/splits.py:
# from models.data.splits import make_time_based_train_valid_split


# ---- configuration constants -------------------------------------------------

CITY_ID = "austin"
LOOKBACK_DAYS = 180        # ~6 months
TARGET_COL = "delta_class"  # TODO: set to your actual ordinal target column name


def compute_date_window() -> tuple[date, date]:
    """Return (start_date, end_date) for the last ~6 months."""
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=LOOKBACK_DAYS)
    return start, end


def build_austin_dataset(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Build the event-day (or market-clock) dataset for Austin only,
    with weather_more_apis features enabled.
    """
    # Adjust DatasetConfig fields to match your pipeline.
    # Key points:
    #   - cities=["austin"]
    #   - date range = [start_date, end_date]
    #   - include_more_apis / use_more_apis flag = True
    cfg = DatasetConfig(
        cities=[CITY_ID],
        start_date=start_date,
        end_date=end_date,
        # These fields depend on how you wired dataset.py:
        # time_window="event_day",  # or "market_clock"
        # snapshot_interval_min=15,
        # use_more_apis=True,
        # any other knobs you already use in 01_build_dataset / training
    )

    with get_db_session() as session:
        df = build_dataset(session=session, config=cfg)
    return df


def time_based_80_20_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wrapper around your existing 80/20 split logic.

    Replace this body with a call to the function you already use in
    scripts/train_city_ordinal_optuna.py (or similar).

    Example (if you have a helper):

        from models.data.splits import make_time_based_train_valid_split
        train_df, valid_df = make_time_based_train_valid_split(df, train_frac=0.8)

    For now, here's a simple timestamp-based split stub.
    """
    # TODO: DELETE this stub and call your real splitter instead.
    # This is just a fallback if you want it:
    if "event_date" in df.columns:
        df_sorted = df.sort_values("event_date")
    elif "snapshot_time" in df.columns:
        df_sorted = df.sort_values("snapshot_time")
    else:
        df_sorted = df.sort_index()

    n = len(df_sorted)
    split_idx = int(n * 0.8)
    train_df = df_sorted.iloc[:split_idx].copy()
    valid_df = df_sorted.iloc[split_idx:].copy()
    return train_df, valid_df


def train_catboost_model(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> CatBoostClassifier:
    """
    Train a CatBoost ordinal model on Austin data using the numeric feature set.
    """
    # Ensure target is present
    assert TARGET_COL in train_df.columns, f"{TARGET_COL} not found in dataset"

    # Features: use your global NUMERIC_FEATURE_COLS, which should already
    # include the weather_more_apis features.
    feature_cols = NUMERIC_FEATURE_COLS

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[TARGET_COL]

    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)

    # Basic CatBoost config — you can tune this later or mirror your Optuna settings.
    model = CatBoostClassifier(
        loss_function="MultiClass",      # or your ordinal loss if you use a specific one
        eval_metric="MultiClass",        # or your preferred eval metric
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        od_type="Iter",
        od_wait=100,
        verbose=100,
    )

    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
    )

    return model


def main() -> None:
    start_date, end_date = compute_date_window()
    print(f"Building Austin dataset from {start_date} to {end_date}...")

    df = build_austin_dataset(start_date, end_date)
    print(f"Dataset built: {df.shape[0]} rows, {df.shape[1]} columns")

    # Plug in your existing 80/20 time-based split
    train_df, valid_df = time_based_80_20_split(df)
    print(f"Train size: {len(train_df)} | Valid size: {len(valid_df)}")

    model = train_catboost_model(train_df, valid_df)

    # Optional: save the model to disk
    model_path = f"models/saved/austin_more_apis_6m.cbm"
    model.save_model(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
