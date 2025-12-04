#!/usr/bin/env python3
"""Comprehensive diagnostic for edge generation failures.

Counts exactly where snapshots are being dropped and logs sample failures.
"""

import sys
from pathlib import Path
from datetime import date
from collections import Counter

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.training.ordinal_trainer import OrdinalDeltaTrainer
from models.edge.implied_temp import compute_forecast_implied_temp, compute_market_implied_temp
from models.edge.detector import detect_edge, EdgeSignal
from scripts.train_edge_classifier import (
    load_combined_data,
    load_all_settlements,
    load_all_candles_batch,
    get_candles_from_cache,
    CITY_CONFIG,
)

# Configuration
CITY = "chicago"  # Change to test different cities
TEST_DAYS = 10  # Number of days to test

print("="*70)
print(f"COMPREHENSIVE EDGE GENERATION DIAGNOSTIC - {CITY.upper()}")
print("="*70)

# Load everything
print("\n1. Loading ordinal model...")
model = OrdinalDeltaTrainer()
model.load(Path(f"models/saved/{CITY}/ordinal_catboost_optuna.pkl"))

print("2. Loading training data...")
df = load_combined_data(CITY)
unique_days = sorted(df["day"].unique())
test_days_list = unique_days[:TEST_DAYS]

print(f"3. Loading settlements for {TEST_DAYS} days...")
settlements = load_all_settlements(CITY, test_days_list)
print(f"   Loaded {len(settlements)} settlements")

print(f"4. Loading candle cache for {TEST_DAYS} days...")
candle_cache = load_all_candles_batch(CITY, test_days_list)
print(f"   Cache entries: {len(candle_cache)}")

# Counters for failure reasons
counters = Counter()
sample_failures = {
    "no_snapshot": [],
    "no_base_temp": [],
    "model_error": [],
    "no_candles": [],
    "invalid_market": [],
    "no_trade": [],
    "success": [],
}

print(f"\n5. Processing {TEST_DAYS} days...")
print("-" * 70)

for day in test_days_list:
    day_df = df[df["day"] == day].copy()
    settlement = settlements.get(day)

    if day_df.empty or settlement is None:
        counters["no_data"] += 1
        continue

    unique_times = day_df["cutoff_time"].unique()
    sample_rate = 10
    sampled_times = unique_times[::sample_rate]

    for snapshot_time in sampled_times:
        snapshot_df = day_df[day_df["cutoff_time"] == snapshot_time]

        # Check 1: Empty snapshot
        if snapshot_df.empty:
            counters["no_snapshot"] += 1
            if len(sample_failures["no_snapshot"]) < 3:
                sample_failures["no_snapshot"].append((day, snapshot_time))
            continue

        # Check 2: Base temp
        base_temp = snapshot_df["t_forecast_base"].iloc[0]
        if pd.isna(base_temp):
            base_temp = snapshot_df["fcst_prev_max_f"].iloc[0]
        if pd.isna(base_temp):
            counters["no_base_temp"] += 1
            if len(sample_failures["no_base_temp"]) < 3:
                sample_failures["no_base_temp"].append((day, snapshot_time))
            continue

        # Check 3: Ordinal model prediction
        try:
            delta_probs = model.predict_proba(snapshot_df)
            forecast_result = compute_forecast_implied_temp(
                delta_probs=delta_probs[0],
                base_temp=base_temp,
            )
        except Exception as e:
            counters["model_error"] += 1
            if len(sample_failures["model_error"]) < 3:
                sample_failures["model_error"].append((day, snapshot_time, str(e)))
            continue

        # Check 4: Candle cache lookup
        bracket_candles = get_candles_from_cache(candle_cache, day, snapshot_time, CITY)
        if not bracket_candles:
            counters["no_candles"] += 1
            if len(sample_failures["no_candles"]) < 3:
                sample_failures["no_candles"].append((day, snapshot_time))
            continue

        # Check 5: Market implied temp
        market_result = compute_market_implied_temp(
            bracket_candles=bracket_candles,
            snapshot_time=snapshot_time,
        )
        if not market_result.valid:
            counters["invalid_market"] += 1
            if len(sample_failures["invalid_market"]) < 3:
                sample_failures["invalid_market"].append((day, snapshot_time))
            continue

        # Check 6: Edge detection
        try:
            edge_result = detect_edge(
                forecast_temp=forecast_result.implied_temp,
                market_temp=market_result.implied_temp,
                forecast_uncertainty=forecast_result.uncertainty,
                market_uncertainty=market_result.uncertainty,
                threshold_f=1.5,
            )
        except Exception as e:
            counters["detect_edge_error"] += 1
            if len(sample_failures["detect_edge_error"]) < 3:
                sample_failures["detect_edge_error"].append((day, snapshot_time, str(e)))
            continue

        if edge_result.signal == EdgeSignal.NO_TRADE:
            counters["no_trade"] += 1
        else:
            counters["success"] += 1
            if len(sample_failures["success"]) < 3:
                sample_failures["success"].append((day, snapshot_time, edge_result.signal.value))

# Print results
print("\n" + "="*70)
print("DIAGNOSTIC RESULTS")
print("="*70)

total = sum(counters.values())
print(f"\nTotal snapshots processed: {total}")
print(f"\nFailure breakdown:")
for reason, count in counters.most_common():
    pct = 100 * count / total if total > 0 else 0
    print(f"  {reason:<20} {count:>6} ({pct:>5.1f}%)")

print(f"\nSample failures (first 3 of each type):")
for reason, samples in sample_failures.items():
    if samples and reason != "success":
        print(f"\n  {reason}:")
        for sample in samples:
            if len(sample) == 2:
                print(f"    {sample[0]} at {sample[1]}")
            else:
                print(f"    {sample[0]} at {sample[1]}: {sample[2]}")

if counters["success"] > 0:
    print(f"\n✓ SUCCESS! Generated {counters['success']} edges + {counters['no_trade']} no-trades")
else:
    print(f"\n✗ FAILURE! Zero edges generated")
    print(f"\nTop failure reason: {counters.most_common(1)[0][0]}")

print("="*70)
