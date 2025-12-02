---
plan_id: debug-backtest-zero-trades
created: 2025-11-30
status: urgent
priority: critical
---

# Debug: Backtest Producing 0-1 Trades (Should be ~20-40)

## Problem

Backtest ran on Chicago, 60 events (Sep 29 - Nov 27):
- **TOD v1-only:** 1 trade (extremely low)
- **Hybrid:** 0 trades (impossible - should match or exceed TOD v1)
- **Expected:** ~20-40 trades (33% of events with sufficient edge)

## Root Cause Hypotheses

### Hypothesis 1: Candles Not Found
**Symptom:** query_candle_at_time() returning None for most events

**Check:**
```python
# Test if candles exist at 10:00 for test events
SELECT COUNT(DISTINCT DATE(bucket_start))
FROM kalshi.candles_1m
WHERE ticker LIKE 'KXHIGHCHI%'
  AND EXTRACT(HOUR FROM bucket_start) = 10
  AND bucket_start >= '2025-09-29'
  AND bucket_start <= '2025-11-27'
```

**Possible causes:**
- Candles exist but not at 10:00 (market opens later?)
- Timezone mismatch (Chicago = America/Chicago, candles in UTC)
- Window too narrow (±5 minutes might miss candles)

### Hypothesis 2: find_best_bracket() Failing
**Symptom:** EV threshold not met or Kelly sizing returning 0

**Check:**
- MIN_EV_PER_CONTRACT_CENTS = 3.0 (event day)
- D_MINUS_1_MIN_EV_PER_CONTRACT_CENTS = 5.0 (D-1)
- Model edge might be < 3¢ for most brackets

**Debug:** Add logging in find_best_bracket() to see:
- How many brackets evaluated
- Max EV found
- Why filtered out

### Hypothesis 3: Bracket Probabilities Wrong
**Symptom:** delta_probs_to_bracket_probs() not mapping correctly

**Check:**
- t_base from snapshot
- Bracket strikes from markets table
- Delta probability distribution

### Hypothesis 4: Hybrid D Trade Path Different
**Symptom:** run_hybrid_scenario() calls _simulate_d_trade() but gets different result than run_tod_only_scenario()

**This is impossible unless:**
- Shared state between scenarios (cache not cleared?)
- Exception being caught silently

## Investigation Steps

1. **Add debug logging:**
```python
# In _simulate_d_trade():
logger.info(f"[DEBUG] {city} {event_date}: Starting D trade simulation")
logger.info(f"[DEBUG] Snapshot found: {not snapshot.empty}")
logger.info(f"[DEBUG] Brackets: {len(brackets)}")
logger.info(f"[DEBUG] Candles found: {len(candles)}")
logger.info(f"[DEBUG] Best trade: {best_trade}")
```

2. **Test with single event:**
```bash
# Run on just 1 event with debug
python scripts/backtest_hybrid_vs_tod_v1.py --days 1 --cities chicago --debug
```

3. **Check candle availability at 10:00:**
```sql
-- For each test event, check if candles exist at 10:00
SELECT m.event_date,
       COUNT(c.ticker) as candles_at_10am
FROM kalshi.markets m
LEFT JOIN kalshi.candles_1m c
  ON m.ticker = c.ticker
  AND EXTRACT(HOUR FROM c.bucket_start AT TIME ZONE 'America/Chicago') = 10
WHERE m.city = 'chicago'
  AND m.event_date BETWEEN '2025-09-29' AND '2025-11-27'
GROUP BY m.event_date
ORDER BY m.event_date
LIMIT 10
```

4. **Check test data has correct event_dates:**
```python
# Verify test data covers Sep 29 - Nov 27
mc_test = pd.read_parquet('models/saved/market_clock_tod_v1/test_data.parquet')
chicago_test = mc_test[mc_test['city'] == 'chicago']
print(f"Chicago test events: {chicago_test['event_date'].min()} to {chicago_test['event_date'].max()}")
print(f"Count: {chicago_test['event_date'].nunique()}")
```

## Likely Root Cause (Prediction)

**Timezone issue in candle query:**
- Candles stored in UTC (bucket_start is TIMESTAMPTZ)
- Test passes Chicago local time (America/Chicago)
- Query might be off by 5-6 hours (CDT offset)
- So looking for 10:00 Chicago = 15:00 UTC, but candles might be at different UTC times

**Fix:**
```python
# In _simulate_d_trade(), ensure timestamp is timezone-aware
city_tz = ZoneInfo(config.CITY_TIMEZONES[city])
timestamp = datetime(event_date.year, event_date.month, event_date.day, D_HOUR, 0, tzinfo=city_tz)
# This will be converted to UTC for DB query
```

## Next Actions

1. Exit plan mode
2. Add extensive debug logging to backtest script
3. Run single-event test with --debug
4. Fix timezone/candle query issue
5. Re-run full backtest
6. Verify 20-40 trades (reasonable hit rate)
