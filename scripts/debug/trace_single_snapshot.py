#!/usr/bin/env python3
"""Trace through ONE snapshot to see where it fails."""

import sys
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np

# scripts/debug/ -> 2 levels up
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from models.training.ordinal_trainer import OrdinalDeltaTrainer
from models.edge.implied_temp import compute_forecast_implied_temp, compute_market_implied_temp
from models.edge.detector import detect_edge, EdgeSignal
from scripts.training.core.train_edge_classifier import load_combined_data, load_all_candles_batch, get_candles_from_cache

CITY = "austin"
TEST_DATE = date(2024, 6, 1)

print("="*60)
print("SINGLE SNAPSHOT TRACE")
print("="*60)

# Load everything
print("\n1. Loading model...")
model = OrdinalDeltaTrainer()
model.load(Path(f"models/saved/{CITY}/ordinal_catboost_optuna.pkl"))

print("2. Loading data...")
df = load_combined_data(CITY)
day_df = df[df["day"] == TEST_DATE].copy()
print(f"   Day rows: {len(day_df)}")

print("3. Loading candle cache...")
candle_cache = load_all_candles_batch(CITY, [TEST_DATE])
print(f"   Cache entries: {len(candle_cache)}")

# Get FIRST snapshot
unique_times = day_df["cutoff_time"].unique()
snapshot_time = unique_times[0]  # First one

print(f"\n4. Processing snapshot at {snapshot_time}...")
snapshot_df = day_df[day_df["cutoff_time"] == snapshot_time]
print(f"   Snapshot rows: {len(snapshot_df)}")

# Check 1: Base temp
base_temp = snapshot_df["t_forecast_base"].iloc[0]
print(f"   Base temp: {base_temp}°F")
if pd.isna(base_temp):
    print("   ERROR: Base temp is NaN!")
    sys.exit(1)

# Check 2: Model prediction
print("\n5. Running ordinal model...")
try:
    delta_probs = model.predict_proba(snapshot_df)
    print(f"   Delta probs shape: {delta_probs.shape}")
    print(f"   Sample probs: {delta_probs[0][:5]}")

    forecast_result = compute_forecast_implied_temp(
        delta_probs=delta_probs[0],
        base_temp=base_temp,
    )
    print(f"   ✓ Forecast temp: {forecast_result.implied_temp:.2f}°F")
except Exception as e:
    print(f"   ERROR in model prediction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 3: Candle cache lookup
print("\n6. Looking up candles in cache...")
print(f"   Looking for day={TEST_DATE}, snapshot_time={snapshot_time}")
print(f"   Cache has {len(candle_cache)} total entries")

# Show what days are in the cache
cache_days = set([d for d, _ in candle_cache.keys()])
print(f"   Unique days in cache: {len(cache_days)}")
print(f"   Test date in cache? {TEST_DATE in cache_days}")

bracket_candles = get_candles_from_cache(candle_cache, TEST_DATE, snapshot_time)
print(f"   Brackets returned: {len(bracket_candles)}")

if not bracket_candles:
    print("   ERROR: No bracket candles returned from cache!")
    print(f"   Cache keys sample: {list(candle_cache.keys())[:3]}")
    print(f"   Snapshot time type: {type(snapshot_time)} = {snapshot_time}")
    sys.exit(1)

print(f"   ✓ Got {len(bracket_candles)} brackets")
for label in list(bracket_candles.keys())[:3]:
    print(f"      {label}: {len(bracket_candles[label])} candles")

# Check 4: Market implied temp
print("\n7. Computing market implied temp...")
market_result = compute_market_implied_temp(bracket_candles, snapshot_time=snapshot_time)
print(f"   Valid: {market_result.valid}")
print(f"   Implied temp: {market_result.implied_temp:.2f}°F")

if not market_result.valid:
    print("   ERROR: Market result not valid!")
    sys.exit(1)

# Check 5: Edge detection
print("\n8. Detecting edge...")
edge_result = detect_edge(
    forecast_implied=forecast_result.implied_temp,
    market_implied=market_result.implied_temp,
    forecast_uncertainty=forecast_result.uncertainty,
    market_uncertainty=market_result.uncertainty,
    threshold=1.5,
)
print(f"   Signal: {edge_result.signal}")
print(f"   Edge: {edge_result.edge:.2f}°F")
print(f"   Confidence: {edge_result.confidence:.2f}")

print("\n" + "="*60)
print("SUCCESS - All steps completed!")
print("="*60)
