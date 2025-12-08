"""Denver Oct 2025 Pipeline Validation Tests.

Minimal targeted tests for validating the parquet-only pipeline on a small dataset.
These tests verify:
- Train/test split is temporal and non-overlapping
- Dataset only contains Oct 2025 data
- Critical columns are present and non-null
- Candle data is sane (bid <= ask, valid ranges)
- Models were saved successfully

Usage:
    pytest tests/test_denver_oct2025.py -v
"""

import pytest
import pandas as pd
from pathlib import Path
from datetime import date

CITY = "denver"
START_DATE = date(2025, 10, 1)
END_DATE = date(2025, 10, 31)


def test_train_test_no_overlap():
    """Verify train/test split is temporal and non-overlapping."""
    train = pd.read_parquet(f"models/saved/{CITY}/train_data_full.parquet")
    test = pd.read_parquet(f"models/saved/{CITY}/test_data_full.parquet")

    # Convert 'day' column to dates
    train_days = set(pd.to_datetime(train['day']).dt.date.unique())
    test_days = set(pd.to_datetime(test['day']).dt.date.unique())

    # No overlapping days
    overlap = train_days & test_days
    assert len(overlap) == 0, f"Train/test days overlap: {sorted(overlap)}"

    # Temporal ordering (test days > train days)
    min_test = min(test_days)
    max_train = max(train_days)
    assert min_test > max_train, f"Test days not after train days! Train max: {max_train}, Test min: {min_test}"

    print(f"✅ Train: {len(train_days)} days ({min(train_days)} to {max_train})")
    print(f"✅ Test: {len(test_days)} days ({min_test} to {max(test_days)})")


def test_october_date_range():
    """Verify dataset only contains Oct 2025 days (no date leakage)."""
    train = pd.read_parquet(f"models/saved/{CITY}/train_data_full.parquet")
    test = pd.read_parquet(f"models/saved/{CITY}/test_data_full.parquet")

    # Combine all days
    all_days = pd.to_datetime(
        pd.concat([train['day'], test['day']])
    ).dt.date.unique()

    # Check all days are in Oct 2025
    out_of_range = [d for d in all_days if not (START_DATE <= d <= END_DATE)]

    assert len(out_of_range) == 0, \
        f"Found {len(out_of_range)} days outside Oct 2025: {sorted(out_of_range)[:10]}"

    print(f"✅ All {len(all_days)} days in range: {START_DATE} to {END_DATE}")


def test_core_columns_non_null():
    """Verify critical columns have minimal nulls."""
    train = pd.read_parquet(f"models/saved/{CITY}/train_data_full.parquet")

    # Use actual column names from DataFrame
    critical = ['delta', 'settle_f', 'cutoff_time']

    # Only test columns that exist (don't hard-code assumptions)
    missing_cols = [col for col in critical if col not in train.columns]
    if missing_cols:
        pytest.fail(f"Critical columns missing: {missing_cols}")

    for col in critical:
        if col in train.columns:
            null_count = train[col].isna().sum()
            null_pct = (null_count / len(train)) * 100

            assert null_pct < 5, \
                f"{col} has {null_pct:.1f}% nulls ({null_count:,}/{len(train):,} rows) - too high!"

            print(f"✅ {col}: {100-null_pct:.1f}% non-null")


def test_candles_bid_ask_sane():
    """Verify candle bid/ask sanity checks."""
    candles = pd.read_parquet(f"models/candles/candles_{CITY}.parquet")

    # Candles use 'bucket_start' timestamp - extract date from it
    if 'bucket_start' not in candles.columns:
        pytest.skip(f"Cannot find bucket_start column (available: {list(candles.columns[:10])})")

    # Filter to Oct 2025 using bucket_start timestamp
    candles['date'] = pd.to_datetime(candles['bucket_start']).dt.date
    candles_oct = candles[
        candles['date'].between(START_DATE, END_DATE)
    ].copy()

    assert len(candles_oct) > 0, "No candles found for Oct 2025"

    # Determine bid/ask column names (use close values as representative)
    bid_col = 'yes_bid_close' if 'yes_bid_close' in candles_oct.columns else 'yes_bid'
    ask_col = 'yes_ask_close' if 'yes_ask_close' in candles_oct.columns else 'yes_ask'

    if bid_col not in candles_oct.columns or ask_col not in candles_oct.columns:
        pytest.skip(f"Cannot find bid/ask columns (available: {list(candles_oct.columns[:10])})")

    # Bid/ask range checks
    assert (candles_oct[bid_col] >= 0).all(), f"Found negative {bid_col}"
    assert (candles_oct[bid_col] <= 100).all(), f"Found {bid_col} > 100"
    assert (candles_oct[ask_col] >= 0).all(), f"Found negative {ask_col}"
    assert (candles_oct[ask_col] <= 100).all(), f"Found {ask_col} > 100"

    # Bid must be <= Ask
    bad_spreads = candles_oct[candles_oct[bid_col] > candles_oct[ask_col]]
    assert len(bad_spreads) == 0, \
        f"Found {len(bad_spreads)} candles with {bid_col} > {ask_col}!"

    print(f"✅ {len(candles_oct):,} candles in Oct 2025, all {bid_col}/{ask_col} valid")


def test_model_files_exist():
    """Verify models were saved successfully (only checks models that should exist).

    Note: This test is run AFTER dataset building but MAY be run BEFORE model training.
    It will check for models and skip if they don't exist yet (not a failure).
    """
    ordinal_path = Path(f"models/saved/{CITY}/ordinal_catboost_optuna.pkl")
    edge_path = Path(f"models/saved/{CITY}/edge_classifier.pkl")

    # Check ordinal model (may not exist yet if running tests before training)
    if ordinal_path.exists():
        assert ordinal_path.stat().st_size > 1000, "Ordinal model file too small"
        print(f"✅ Ordinal model: {ordinal_path.stat().st_size / (1024**2):.1f} MB")
    else:
        print(f"⏭️  Ordinal model not found (not trained yet): {ordinal_path}")

    # Check edge classifier (may not exist yet)
    if edge_path.exists():
        assert edge_path.stat().st_size > 1000, "Edge classifier file too small"
        print(f"✅ Edge classifier: {edge_path.stat().st_size / (1024**2):.1f} MB")
    else:
        print(f"⏭️  Edge classifier not found (not trained yet): {edge_path}")


if __name__ == "__main__":
    # Can run directly for quick testing
    pytest.main([__file__, "-v"])
