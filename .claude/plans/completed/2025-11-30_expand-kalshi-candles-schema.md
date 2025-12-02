---
plan_id: expand-kalshi-candles-schema
created: 2025-11-30
status: in_progress
priority: critical
agent: kalshi-weather-quant
---

# Expand Kalshi Candles Schema for Future-Proof Storage

## Objective

Expand `kalshi.candles_1m` schema to capture full bid/ask OHLC from Kalshi API before backfilling historical data. Ensures we never need to re-backfill due to missing fields.

## Context

**Current State:**
- Partial schema: only yes_bid_close, yes_ask_close (CLOSE only)
- OHLC columns (open_c/high_c/low_c/close_c) mix trade and ask prices
- Missing: yes_bid/ask open/high/low
- Have 1.3M candles with partial data

**Why Expand:**
- Professor confirmed: Need full bid/ask OHLC for future intraday strategies
- Current schema limits advanced backtesting
- API provides all fields - just need to capture them

**Professor's Decision:**
- ✅ Store full YES bid/ask OHLC
- ✅ Store clean trade OHLC with clear naming
- ✅ Store optional trade_mean/previous/min/max
- ❌ Don't store NO prices (derive: no_bid = 100 - yes_ask)
- ✅ Add period_minutes for future hourly/daily aggregations

---

## Tasks

### Phase 1: Update Pydantic Schema
- [ ] Add price_mean, price_previous, price_min, price_max to Candle model

### Phase 2: Database Migration
- [ ] Create migration: expand_candles_full_ohlc
- [ ] Truncate existing candles (1.3M rows)
- [ ] Rename columns for clarity (yes_bid_c → yes_bid_close, etc.)
- [ ] Add yes_bid_open/high/low columns
- [ ] Add yes_ask_open/high/low columns
- [ ] Add trade_mean/previous/min/max columns
- [ ] Add period_minutes column
- [ ] Run migration

### Phase 3: Update Backfill Script
- [ ] Modify candle_to_db_dict() to capture all fields
- [ ] Verify upsert logic handles new columns

### Phase 4: Test with Chicago (Jan 1, 2025 - Nov 30, 2025)
- [ ] Backfill Chicago only (~335 events, ~30 sec)
- [ ] Verify all fields populated (no unexpected NULLs)
- [ ] Spot-check sample candles for data quality
- [ ] Check bid/ask OHLC makes sense (bid_high <= ask_low, etc.)

### Phase 5: Backfill All Cities
- [ ] Backfill remaining 5 cities (all history)
- [ ] Final verification across all cities
- [ ] Document final candle counts per city

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/kalshi/schemas.py` | Add price_mean/previous/min/max to Candle |
| `migrations/versions/0XX_expand_candles_full_ohlc.py` | New migration |
| `scripts/backfill_kalshi_candles.py` | Update candle_to_db_dict() |

---

## New Schema Structure

```sql
kalshi.candles_1m (
    -- Keys
    ticker              TEXT NOT NULL,
    bucket_start        TIMESTAMPTZ NOT NULL,
    source              TEXT NOT NULL DEFAULT 'api_event',
    period_minutes      SMALLINT NOT NULL DEFAULT 1,

    -- YES Bid OHLC (4 fields)
    yes_bid_open        SMALLINT,
    yes_bid_high        SMALLINT,
    yes_bid_low         SMALLINT,
    yes_bid_close       SMALLINT,

    -- YES Ask OHLC (4 fields)
    yes_ask_open        SMALLINT,
    yes_ask_high        SMALLINT,
    yes_ask_low         SMALLINT,
    yes_ask_close       SMALLINT,

    -- Last Trade OHLC (4 fields)
    trade_open          SMALLINT,
    trade_high          SMALLINT,
    trade_low           SMALLINT,
    trade_close         SMALLINT,

    -- Trade Stats (4 fields - optional but future-proof)
    trade_mean          SMALLINT,  -- Average trade price in bar
    trade_previous      SMALLINT,  -- Last trade before bar
    trade_min           SMALLINT,  -- Extreme low
    trade_max           SMALLINT,  -- Extreme high

    -- Volume/Interest
    volume              INTEGER,
    open_interest       INTEGER,

    PRIMARY KEY (ticker, bucket_start, source)
)
```

**Total:** 20 price fields (vs current 6) + 2 volume + 4 meta = 26 columns

---

## Implementation Steps

### 1. Update Candle Schema (src/kalshi/schemas.py)

```python
class Candle(BaseModel):
    # ... existing fields ...

    # Add missing trade price stats
    price_mean: Optional[int] = None
    price_previous: Optional[int] = None
    price_min: Optional[int] = None
    price_max: Optional[int] = None
```

### 2. Create Migration

```sql
-- Rename for clarity
ALTER TABLE kalshi.candles_1m RENAME COLUMN open_c TO trade_open;
ALTER TABLE kalshi.candles_1m RENAME COLUMN high_c TO trade_high;
ALTER TABLE kalshi.candles_1m RENAME COLUMN low_c TO trade_low;
ALTER TABLE kalshi.candles_1m RENAME COLUMN close_c TO trade_close;
ALTER TABLE kalshi.candles_1m RENAME COLUMN yes_bid_c TO yes_bid_close;
ALTER TABLE kalshi.candles_1m RENAME COLUMN yes_ask_c TO yes_ask_close;

-- Add YES bid open/high/low
ALTER TABLE kalshi.candles_1m ADD COLUMN yes_bid_open SMALLINT;
ALTER TABLE kalshi.candles_1m ADD COLUMN yes_bid_high SMALLINT;
ALTER TABLE kalshi.candles_1m ADD COLUMN yes_bid_low SMALLINT;

-- Add YES ask open/high/low
ALTER TABLE kalshi.candles_1m ADD COLUMN yes_ask_open SMALLINT;
ALTER TABLE kalshi.candles_1m ADD COLUMN yes_ask_high SMALLINT;
ALTER TABLE kalshi.candles_1m ADD COLUMN yes_ask_low SMALLINT;

-- Add trade stats
ALTER TABLE kalshi.candles_1m ADD COLUMN trade_mean SMALLINT;
ALTER TABLE kalshi.candles_1m ADD COLUMN trade_previous SMALLINT;
ALTER TABLE kalshi.candles_1m ADD COLUMN trade_min SMALLINT;
ALTER TABLE kalshi.candles_1m ADD COLUMN trade_max SMALLINT;

-- Add period tracking
ALTER TABLE kalshi.candles_1m ADD COLUMN period_minutes SMALLINT NOT NULL DEFAULT 1;
```

### 3. Update Backfill Logic

```python
def candle_to_db_dict(ticker: str, candle: Candle, source: str) -> dict:
    return {
        "ticker": ticker,
        "bucket_start": datetime.fromtimestamp(candle.end_period_ts - 60, tz=timezone.utc),
        "source": source,
        "period_minutes": candle.period_minutes or 1,

        # Trade OHLC + stats
        "trade_open": candle.price_open,
        "trade_high": candle.price_high,
        "trade_low": candle.price_low,
        "trade_close": candle.price_close,
        "trade_mean": candle.price_mean,
        "trade_previous": candle.price_previous,
        "trade_min": candle.price_min,
        "trade_max": candle.price_max,

        # YES Bid OHLC (FULL)
        "yes_bid_open": candle.yes_bid_open,
        "yes_bid_high": candle.yes_bid_high,
        "yes_bid_low": candle.yes_bid_low,
        "yes_bid_close": candle.yes_bid_close,

        # YES Ask OHLC (FULL)
        "yes_ask_open": candle.yes_ask_open,
        "yes_ask_high": candle.yes_ask_high,
        "yes_ask_low": candle.yes_ask_low,
        "yes_ask_close": candle.yes_ask_close,

        # Volume/OI
        "volume": candle.volume,
        "open_interest": candle.open_interest,
    }
```

---

## Testing Strategy

### Chicago Test (Jan 1 - Nov 30, 2025)

```bash
# Get Chicago event count
# Estimated: ~335 events, ~30 seconds at 29 req/sec

python scripts/backfill_kalshi_candles.py \
    --city chicago \
    --start-date 2025-01-01 \
    --end-date 2025-11-30 \
    --source api_event
```

**Validation Checks:**
```sql
-- Check field population
SELECT
    COUNT(*) as total,
    COUNT(yes_bid_open) as bid_open_count,
    COUNT(yes_bid_high) as bid_high_count,
    COUNT(yes_ask_open) as ask_open_count,
    COUNT(trade_mean) as trade_mean_count,
    100.0 * COUNT(yes_bid_open) / COUNT(*) as bid_open_pct
FROM kalshi.candles_1m
WHERE ticker LIKE 'KXHIGHCHI%';

-- Sanity checks
SELECT ticker, bucket_start,
       yes_bid_close, yes_ask_close,
       yes_bid_high, yes_ask_low
FROM kalshi.candles_1m
WHERE ticker LIKE 'KXHIGHCHI%'
  AND (yes_bid_close > yes_ask_close  -- Crossed spread (bad)
    OR yes_bid_high > yes_ask_low)     -- Bid above ask (bad)
LIMIT 10;

-- Check for NULLs
SELECT
    SUM(CASE WHEN yes_bid_open IS NULL THEN 1 ELSE 0 END) as bid_open_nulls,
    SUM(CASE WHEN yes_ask_open IS NULL THEN 1 ELSE 0 END) as ask_open_nulls
FROM kalshi.candles_1m
WHERE ticker LIKE 'KXHIGHCHI%';
```

---

## Completion Criteria

- [ ] Schema expanded with all fields
- [ ] Chicago backfilled (Jan 1 - Nov 30, 2025)
- [ ] All price fields populated (verify %non-NULL)
- [ ] No data quality issues (crossed spreads, invalid ranges)
- [ ] All cities backfilled
- [ ] Ready for hybrid backtest

---

## Sign-off

Professor approved future-proof approach. Expanding schema before full historical backfill to avoid re-work.
