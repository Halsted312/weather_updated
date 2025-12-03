#!/usr/bin/env python3
"""Test _process_single_day to see why it's returning empty results."""

import sys
from pathlib import Path
from datetime import date

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.training.ordinal_trainer import OrdinalDeltaTrainer
from scripts.train_edge_classifier import (
    load_combined_data,
    load_all_settlements,
    load_all_candles_batch,
    _process_single_day,
)

CITY = "austin"
TEST_DATE = date(2024, 6, 1)

print("Loading model...")
model = OrdinalDeltaTrainer()
model.load(Path(f"models/saved/{CITY}/ordinal_catboost_optuna.pkl"))

print("Loading data...")
df = load_combined_data(CITY)
day_df = df[df["day"] == TEST_DATE].copy()
print(f"Day {TEST_DATE}: {len(day_df)} rows")

print("Loading settlements...")
unique_days = sorted(df["day"].unique())
settlements = load_all_settlements(CITY, unique_days)
settlement = settlements.get(TEST_DATE)
print(f"Settlement for {TEST_DATE}: {settlement}°F")

print("Loading candle cache (just for this day)...")
candle_cache = load_all_candles_batch(CITY, [TEST_DATE])
print(f"Cache entries: {len(candle_cache)}")

# Show what's in the cache for this day
day_entries = [(d, label) for d, label in candle_cache.keys() if d == TEST_DATE]
print(f"Entries for {TEST_DATE}: {len(day_entries)}")
for d, label in day_entries[:5]:
    print(f"  {label}: {len(candle_cache[(d, label)])} candles")

print(f"\nCalling _process_single_day...")
results = _process_single_day(
    day=TEST_DATE,
    day_df=day_df,
    model=model,
    candle_cache=candle_cache,
    settlement=settlement,
    edge_threshold=1.5,
    sample_rate=10,  # Every 10th snapshot
)

print(f"\nResults: {len(results)} edge signals")
if results:
    print("Sample result:")
    print(results[0])
else:
    print("NO RESULTS - all snapshots returned continue!")
