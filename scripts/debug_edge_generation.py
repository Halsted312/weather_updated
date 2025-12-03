#!/usr/bin/env python3
"""Debug edge generation for a single day."""

import sys
from pathlib import Path
from datetime import date

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.training.ordinal_trainer import OrdinalDeltaTrainer
from models.edge.implied_temp import compute_forecast_implied_temp, compute_market_implied_temp
from models.edge.detector import detect_edge, EdgeSignal
from src.db import get_db_session
from sqlalchemy import text

# Test parameters
CITY = "austin"
TEST_DATE = date(2024, 6, 1)  # Known to have candles
SAMPLE_SNAPSHOT_HOUR = 14  # 2pm

print("="*60)
print("EDGE GENERATION DEBUG")
print("="*60)

# Step 1: Load ordinal model
print("\n1. Loading ordinal model...")
model_path = f"models/saved/{CITY}/ordinal_catboost_optuna.pkl"
model = OrdinalDeltaTrainer()
model.load(Path(model_path))
print(f"   ✓ Model loaded")

# Step 2: Load training data for test date
print("\n2. Loading training data...")
df_train = pd.read_parquet(f"models/saved/{CITY}/train_data_full.parquet")
df_test = pd.read_parquet(f"models/saved/{CITY}/test_data_full.parquet")
df = pd.concat([df_train, df_test])
print(f"   Total rows: {len(df):,}")

day_df = df[df["day"] == TEST_DATE]
print(f"   Rows for {TEST_DATE}: {len(day_df)}")

if day_df.empty:
    print(f"   ERROR: No data for {TEST_DATE}")
    sys.exit(1)

# Step 3: Get snapshot at specific hour
snapshot_df = day_df[day_df["snapshot_hour"] == SAMPLE_SNAPSHOT_HOUR]
if snapshot_df.empty:
    print(f"   ERROR: No snapshot at hour {SAMPLE_SNAPSHOT_HOUR}")
    sys.exit(1)

print(f"   Snapshot at {SAMPLE_SNAPSHOT_HOUR}:00: {len(snapshot_df)} rows")

# Step 4: Get base temp and run ordinal model
print("\n3. Running ordinal model...")
base_temp = snapshot_df["t_forecast_base"].iloc[0]
print(f"   Base temp (t_forecast_base): {base_temp}°F")

delta_probs_array = model.predict_proba(snapshot_df)
delta_probs = delta_probs_array[0]  # First (only) row

# Convert array to dict
delta_classes = list(range(-10, 11))
delta_probs_dict = {k: float(v) for k, v in zip(delta_classes, delta_probs)}

print(f"   Delta probs (top 5):")
sorted_probs = sorted(delta_probs_dict.items(), key=lambda x: -x[1])[:5]
for delta, prob in sorted_probs:
    print(f"     delta={delta:+3d}: {prob:.3f}")

# Step 5: Compute forecast-implied temp
print("\n4. Computing forecast-implied temp...")
forecast_result = compute_forecast_implied_temp(
    delta_probs=delta_probs_dict,
    base_temp=base_temp,
)
print(f"   Forecast temp: {forecast_result.implied_temp:.2f}°F")
print(f"   Uncertainty: {forecast_result.uncertainty:.2f}°F")
print(f"   Predicted delta: {forecast_result.predicted_delta}")

# Step 6: Load Kalshi candles
print("\n5. Loading Kalshi candles...")
event_suffix = TEST_DATE.strftime("%y%b%d").upper()
print(f"   Event suffix: {event_suffix}")

with get_db_session() as session:
    query = text("""
        SELECT ticker, COUNT(*) as cnt,
               MAX(yes_bid_close) as max_bid,
               MAX(yes_ask_close) as max_ask
        FROM kalshi.candles_1m_dense
        WHERE (ticker LIKE :new_pat OR ticker LIKE :old_pat)
        GROUP BY ticker
        ORDER BY ticker
    """)

    result = session.execute(query, {
        'new_pat': f'KXHIGHAUS-{event_suffix}%',
        'old_pat': f'HIGHAUS-{event_suffix}%'
    })

    candle_rows = result.fetchall()

print(f"   Found {len(candle_rows)} brackets:")
for ticker, cnt, max_bid, max_ask in candle_rows[:5]:
    print(f"     {ticker}: {cnt} candles, bid={max_bid}, ask={max_ask}")

if not candle_rows:
    print("   ERROR: No candles found!")
    sys.exit(1)

# Step 7: Compute market-implied temp
print("\n6. Computing market-implied temp...")
print("   NOTE: Need to load actual candle DataFrames, not just counts")
print("   Simulating with bracket prices...")

# For now, show what would happen if we had candle data
print(f"\n   If candles loaded successfully:")
print(f"   - Would call compute_market_implied_temp()")
print(f"   - Would get market_result with .implied_temp, .uncertainty, .valid")
print(f"   - Would call detect_edge() to compare forecast vs market")

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)
print("\nNext steps:")
print("1. Verify bracket_candles dict is being built correctly")
print("2. Check compute_market_implied_temp() is getting valid data")
print("3. Add logging to _process_single_day to see where it's failing")
