#!/usr/bin/env python3
"""Debug single event backtest - step by step diagnosis"""

import sys
from pathlib import Path
# scripts/debug/ -> 2 levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import date, datetime
from zoneinfo import ZoneInfo
import pandas as pd

from src.db import get_session_factory
from config import live_trader_config as config
from scripts.backtesting.backtest_utils import (
    query_candle_at_time,
    load_settlement,
    load_brackets_for_event,
    delta_probs_to_bracket_probs,
)

# Test event
city = 'chicago'
event_date = date(2025, 11, 1)  # Nov 1, 2025

print(f"=== Debugging {city} {event_date} ===\n")

# 1. Check if brackets exist
SessionLocal = get_session_factory()
session = SessionLocal()

brackets = load_brackets_for_event(session, city, event_date)
print(f"1. Brackets loaded: {len(brackets)}")
if not brackets.empty:
    print(f"   Sample: {brackets.iloc[0]['ticker']}")
else:
    print("   ❌ NO BRACKETS - Event not in DB or wrong city/date")
    sys.exit(1)

# 2. Check if candles exist at 10:00
city_tz = ZoneInfo(config.CITY_TIMEZONES[city])
timestamp_10am = datetime(event_date.year, event_date.month, event_date.day, 10, 0, tzinfo=city_tz)

print(f"\n2. Looking for candles at: {timestamp_10am}")
print(f"   Timezone: {timestamp_10am.tzinfo}")
print(f"   UTC equivalent: {timestamp_10am.astimezone(ZoneInfo('UTC'))}")

sample_ticker = brackets.iloc[0]['ticker']
candle = query_candle_at_time(session, sample_ticker, timestamp_10am)

if candle:
    print(f"   ✅ Candle found!")
    print(f"   Bid: {candle['yes_bid']}, Ask: {candle['yes_ask']}")
else:
    print(f"   ❌ NO CANDLE at 10:00")
    # Check if ANY candles exist for this ticker
    from sqlalchemy import text
    result = session.execute(text("""
        SELECT COUNT(*), MIN(bucket_start), MAX(bucket_start)
        FROM kalshi.candles_1m
        WHERE ticker = :ticker
    """), {'ticker': sample_ticker})
    row = result.fetchone()
    print(f"   Candles for this ticker: {row[0]} total")
    if row[0] > 0:
        print(f"   Time range: {row[1]} to {row[2]}")

# 3. Check test data
tod_test = pd.read_parquet(f'models/saved/chicago_tod_v1/test_data.parquet')
event_snapshots = tod_test[tod_test['event_date'] == event_date]
print(f"\n3. TOD v1 test data for this event: {len(event_snapshots)} snapshots")
if len(event_snapshots) > 0:
    snapshot_10 = event_snapshots[event_snapshots['snapshot_hour'] == 10]
    print(f"   Snapshots at hour 10: {len(snapshot_10)}")
    if len(snapshot_10) > 0:
        print(f"   t_base: {snapshot_10.iloc[0]['t_base']}")

session.close()
print("\n=== Debug Complete ===")
