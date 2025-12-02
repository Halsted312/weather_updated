# Kalshi Candlestick Schema Review & Questions

**Date:** 2025-11-30
**Purpose:** Schema design review for future-proof Kalshi candle storage
**For:** Professor review before full historical backfill

---

## Executive Summary

**Current Issue:** Database schema captures only partial candlestick data from Kalshi API

**Decision Needed:** Expand schema to capture full bid/ask OHLC before backfilling 3,980 events (~3 years of history)?

**Impact:** If we expand schema, need to:
1. Create new migration
2. Drop existing 1.3M candles
3. Re-backfill (~2-3 minutes for all history)

---

## 1. What Kalshi API Provides

**Source:** [src/kalshi/schemas.py:106-134](../src/kalshi/schemas.py#L106-L134), [tests/test_kalshi_client.py:80-106](../tests/test_kalshi_client.py#L80-L106)

### Candle Object Fields (from API)

```python
class Candle:
    # Timestamp
    end_period_ts: int              # Unix timestamp (bar end)
    period_minutes: int = 1          # 1, 60, or 1440

    # YES Bid OHLC (8 fields)
    yes_bid_open: int               # Opening bid
    yes_bid_high: int               # Highest bid
    yes_bid_low: int                # Lowest bid
    yes_bid_close: int              # Closing bid

    # YES Ask OHLC (8 fields)
    yes_ask_open: int               # Opening ask
    yes_ask_high: int               # Highest ask
    yes_ask_low: int                # Lowest ask
    yes_ask_close: int              # Closing ask

    # Last Trade Price OHLC (8 fields)
    price_open: int                 # Opening trade price
    price_high: int                 # Highest trade
    price_low: int                  # Lowest trade
    price_close: int                # Closing trade

    # Volume/Interest (2 fields)
    volume: int                     # Contracts traded
    open_interest: int              # Open contracts
```

**Total:** 18 price fields + 2 volume fields per candle

### NO Prices (Derived, Not in API)

**Key Insight from [src/trading/fees.py:120-130](../src/trading/fees.py#L120-L130):**

```python
# NO contracts are complement of YES:
no_bid = 100 - yes_ask    # What you'd receive selling NO
no_ask = 100 - yes_bid    # What you'd pay buying NO
```

**Candle API does NOT provide no_bid/no_ask OHLC** - we derive from YES prices if needed.

---

## 2. Current Database Schema

**Table:** `kalshi.candles_1m`
**Migration:** [migrations/versions/001_initial_schema.py:140-156](../migrations/versions/001_initial_schema.py#L140-L156)

### Current Columns (11 total)

```sql
ticker              TEXT NOT NULL
bucket_start        TIMESTAMPTZ NOT NULL
source              TEXT NOT NULL DEFAULT 'api_event'  -- Added in migration 061c683440ed

-- OHLC (4 fields) - MIXED SEMANTICS
open_c              SMALLINT              -- Trade price OR yes_ask (fallback)
high_c              SMALLINT              -- Trade price OR yes_ask
low_c               SMALLINT              -- Trade price OR yes_ask
close_c             SMALLINT              -- Trade price OR yes_ask

-- Bid/Ask Snapshot (2 fields) - CLOSE ONLY
yes_bid_c           SMALLINT              -- YES bid at close
yes_ask_c           SMALLINT              -- YES ask at close

-- Volume (2 fields)
volume              INTEGER
open_interest       INTEGER

PRIMARY KEY (ticker, bucket_start, source)
```

### Current Storage Logic

**From [scripts/backfill_kalshi_candles.py:79-97](../scripts/backfill_kalshi_candles.py#L79-L97):**

```python
# OHLC: Prefer trade prices, fall back to yes_ask
open_c = candle.price_open if candle.price_open else candle.yes_ask_open
high_c = candle.price_high if candle.price_high else candle.yes_ask_high
low_c = candle.price_low if candle.price_low else candle.yes_ask_low
close_c = candle.price_close if candle.price_close else candle.yes_ask_close

# Bid/Ask: Only CLOSE values
yes_bid_c = candle.yes_bid_close
yes_ask_c = candle.yes_ask_close
```

---

## 3. Data Loss Analysis

### What We're Capturing ✅

1. **yes_bid_close** - Good for maker order simulation at snapshot time
2. **yes_ask_close** - Good for taker order simulation
3. **Trade price OHLC** (if available) or **yes_ask OHLC** (fallback)
4. **Volume, open_interest** - Good for liquidity analysis

### What We're LOSING ❌

1. **yes_bid OPEN/HIGH/LOW** - Can't simulate bid movements within the minute
2. **yes_ask OPEN/HIGH/LOW** - Can't simulate ask movements within the minute
3. **Semantic clarity** - OHLC columns mix trade prices and ask prices

### Impact on Current Backtest

**For 10:00-only backtest:** ✅ **Current schema is SUFFICIENT**
- We only need bid/ask at snapshot time (10:00)
- `yes_bid_c` and `yes_ask_c` provide exactly this

**For future intraday strategies:** ⚠️ **Limited**
- Can't analyze bid/ask spread dynamics within a minute
- Can't detect rapid price moves (high/low different from close)
- Can't optimize entry timing within a minute bar

---

## 4. Proposed Future-Proof Schema

### Option A: Full OHLC for Bid/Ask (RECOMMENDED)

**Add columns:**
```sql
-- YES Bid OHLC (keep existing yes_bid_c as yes_bid_close)
yes_bid_open        SMALLINT
yes_bid_high        SMALLINT
yes_bid_low         SMALLINT
yes_bid_close       SMALLINT  -- Rename yes_bid_c

-- YES Ask OHLC (keep existing yes_ask_c as yes_ask_close)
yes_ask_open        SMALLINT
yes_ask_high        SMALLINT
yes_ask_low         SMALLINT
yes_ask_close       SMALLINT  -- Rename yes_ask_c

-- Last Trade OHLC (keep current open_c/high_c/low_c/close_c)
trade_open          SMALLINT  -- Rename open_c for clarity
trade_high          SMALLINT  -- Rename high_c
trade_low           SMALLINT  -- Rename low_c
trade_close         SMALLINT  -- Rename close_c
```

**Total:** 12 price fields (vs current 6)

**Benefits:**
- Can analyze bid/ask spread dynamics
- Can detect price volatility within bars
- Clear semantics (bid vs ask vs trade)
- Supports advanced order types (stop-loss, TWAP)

**Costs:**
- 2× storage for prices (12 fields vs 6)
- Need migration + re-backfill

### Option B: Keep Current Schema (PRAGMATIC)

**No changes needed**

**Benefits:**
- Already backfilled 1.3M candles
- Sufficient for current 10:00 snapshot backtest
- Simpler

**Costs:**
- Limited for future intraday strategies
- OHLC semantics unclear (mixed trade/ask prices)

---

## 5. NO Contract Prices

**Confirmed:** Kalshi Candle API **does NOT** provide `no_bid`/`no_ask` OHLC.

**Derivation** (from [src/trading/fees.py:120-130](../src/trading/fees.py#L120-L130)):
```python
# Binary contract math:
no_bid = 100 - yes_ask    # What you receive selling NO
no_ask = 100 - yes_bid    # What you pay buying NO

# For OHLC:
no_bid_open = 100 - yes_ask_open
no_bid_high = 100 - yes_ask_low   # Inverse!
no_bid_low = 100 - yes_ask_high   # Inverse!
no_bid_close = 100 - yes_ask_close
```

**Recommendation:** Store YES prices only, derive NO on-the-fly when needed. Don't duplicate data.

---

## 6. Relevant Code Files for Professor Review

### Core API & Schema
| File | Purpose | Lines of Interest |
|------|---------|-------------------|
| [src/kalshi/schemas.py](../src/kalshi/schemas.py) | Pydantic models for API responses | 106-134 (Candle), 45-87 (Market) |
| [src/kalshi/client.py](../src/kalshi/client.py) | Kalshi API client with auth | 465-544 (get_all_event_candlesticks) |
| [tests/test_kalshi_client.py](../tests/test_kalshi_client.py) | Integration tests showing API usage | 80-106 (candle schema test) |

### Ingestion & Storage
| File | Purpose | Lines of Interest |
|------|---------|-------------------|
| [scripts/backfill_kalshi_candles.py](../scripts/backfill_kalshi_candles.py) | Backfill script | 60-101 (candle_to_db_dict), 212-289 (backfill logic) |
| [scripts/backfill_kalshi_markets.py](../scripts/backfill_kalshi_markets.py) | Market metadata backfill | Full file |
| [migrations/versions/001_initial_schema.py](../migrations/versions/001_initial_schema.py) | Original table schema | 140-166 (candles_1m definition) |
| [migrations/versions/061c683440ed_add_source_column_to_candles.py](../migrations/versions/061c683440ed_add_source_column_to_candles.py) | Recent migration adding source tracking | 21-48 (upgrade logic) |

### Trading Logic (Shows NO price derivation)
| File | Purpose | Lines of Interest |
|------|---------|-------------------|
| [src/trading/fees.py](../src/trading/fees.py) | Fee calculation & NO price math | 83-134 (classify_liquidity_role), 120-130 (NO bid/ask formulas) |
| [src/trading/risk.py](../src/trading/risk.py) | Kelly sizing (uses prices from candles) | 53-145 (PositionSizer.calculate) |

### Backtest Usage
| File | Purpose | Lines of Interest |
|------|---------|-------------------|
| [scripts/backtest_utils.py](../scripts/backtest_utils.py) | Helper to query candles | 31-77 (query_candle_at_time) |
| [scripts/backtest_hybrid_vs_tod_v1.py](../scripts/backtest_hybrid_vs_tod_v1.py) | Main backtest (uses yes_bid_c, yes_ask_c) | 410-430 (candle queries) |

---

## 7. Questions for Professor

### Q1: Schema Expansion Priority

**For future-proof design, should we expand schema now to capture:**
- ✅ Full YES bid OHLC? (yes_bid_open/high/low/close)
- ✅ Full YES ask OHLC? (yes_ask_open/high/low/close)
- ✅ Separate trade price OHLC with clear naming? (trade_open/high/low/close)
- ❌ NO prices? (Can derive from YES, don't store)

**Or:**
- ⏸️ Keep current schema (sufficient for 10:00 snapshot backtest)
- ⏸️ Expand later if needed for intraday strategies

**Tradeoff:**
- Expand now: One-time 2-3 min re-backfill, future-proof
- Keep current: Run backtest today, risk re-work later

### Q2: Intraday Strategy Plans

**Do you plan to implement:**
- 📊 Multiple decision times per day? (not just 10:00)
- 📊 Intraday exits based on price moves?
- 📊 TWAP or other time-weighted strategies?
- 📊 Volatility-based position sizing?

**If YES to any** → Expand schema to full OHLC
**If NO** → Current schema is fine

### Q3: Open/High/Low vs Close Trade-offs

**For maker order backtesting:**
- Close prices: What you'd see at snapshot time (good for our use case)
- Open prices: What was available at bar start
- High/low: Price extremes (useful for stop-loss, volatility analysis)

**Which matters more for your strategies?**

### Q4: Storage vs Computation

**Alternative approach:** Store only close prices, **compute** OHLC aggregates from:
- Individual trades (if we backfill trades separately)
- OR accept we lose intra-minute resolution

**Preferred:** Store full OHLC from API (cheap storage, fast queries) vs recompute?

---

## 8. Current Database Status

**As of 2025-11-30 16:40:**
```
Total candles:        1,296,151  (partial backfill, stopped)
Test period coverage: 972,052 candles (Sep 29 - Nov 27)
Coverage:             100% of test period markets (2,094/2,094)
Data quality:         0 NULLs in bid/ask
Schema:               Current (11 columns, partial API data)
```

**Historical data available:**
```
Chicago:       1,060 days (2023-01-01 to 2025-11-26) - 3 years!
Austin/Miami:  ~925 days (2023-05-12 to 2025-11-25) - 2.5 years
Denver/Phil:   ~371 days (2024-11-20 to 2025-11-25) - 1 year
LA:            ~325 days (2025-01-05 to 2025-11-25)

Total events:  3,980
Backfill time: ~2-3 minutes at 29 req/sec
```

---

## 9. Recommendation

**For immediate hybrid backtest:**
✅ Current schema is **sufficient** - proceed with backtest now

**For long-term robustness:**
⚠️ **Expand schema before full historical backfill**

**Suggested approach:**
1. Run hybrid backtest with current data (proves methodology)
2. Review results with professor
3. If backtest valuable → expand schema + backfill full history
4. If backtest not valuable → schema doesn't matter

**This avoids premature optimization while keeping options open.**

---

## 10. Proposed Migration (If Expanding)

```sql
-- Migration: Expand candlestick schema to full OHLC

-- Rename existing columns for clarity
ALTER TABLE kalshi.candles_1m RENAME COLUMN open_c TO trade_open;
ALTER TABLE kalshi.candles_1m RENAME COLUMN high_c TO trade_high;
ALTER TABLE kalshi.candles_1m RENAME COLUMN low_c TO trade_low;
ALTER TABLE kalshi.candles_1m RENAME COLUMN close_c TO trade_close;
ALTER TABLE kalshi.candles_1m RENAME COLUMN yes_bid_c TO yes_bid_close;
ALTER TABLE kalshi.candles_1m RENAME COLUMN yes_ask_c TO yes_ask_close;

-- Add YES bid OHLC (open/high/low)
ALTER TABLE kalshi.candles_1m ADD COLUMN yes_bid_open SMALLINT;
ALTER TABLE kalshi.candles_1m ADD COLUMN yes_bid_high SMALLINT;
ALTER TABLE kalshi.candles_1m ADD COLUMN yes_bid_low SMALLINT;

-- Add YES ask OHLC (open/high/low)
ALTER TABLE kalshi.candles_1m ADD COLUMN yes_ask_open SMALLINT;
ALTER TABLE kalshi.candles_1m ADD COLUMN yes_ask_high SMALLINT;
ALTER TABLE kalshi.candles_1m ADD COLUMN yes_ask_low SMALLINT;
```

**Then update [scripts/backfill_kalshi_candles.py:60-101](../scripts/backfill_kalshi_candles.py#L60-L101):**
```python
def candle_to_db_dict(ticker: str, candle: Candle, source: str) -> dict:
    return {
        "ticker": ticker,
        "bucket_start": datetime.fromtimestamp(candle.end_period_ts - 60, tz=timezone.utc),
        "source": source,

        # Last trade OHLC
        "trade_open": candle.price_open,
        "trade_high": candle.price_high,
        "trade_low": candle.price_low,
        "trade_close": candle.price_close,

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

## 11. Files Professor Should Review

**Critical for schema decision:**
1. [src/kalshi/schemas.py](../src/kalshi/schemas.py) - API data model (lines 106-134)
2. [src/trading/fees.py](../src/trading/fees.py) - NO price derivation (lines 120-130)
3. [scripts/backfill_kalshi_candles.py](../scripts/backfill_kalshi_candles.py) - Current storage logic (lines 60-101)
4. [tests/test_kalshi_client.py](../tests/test_kalshi_client.py) - API examples (lines 80-106)

**Supporting files:**
5. [migrations/versions/001_initial_schema.py](../migrations/versions/001_initial_schema.py) - Original schema (lines 140-166)
6. [src/kalshi/client.py](../src/kalshi/client.py) - API client methods (lines 465-544)
7. [scripts/backtest_utils.py](../scripts/backtest_utils.py) - How backtest uses candles (lines 31-77)

---

## 12. Next Steps (Awaiting Professor Decision)

**Option 1: Proceed with Current Schema**
```bash
# Already have test period data
python scripts/backtest_hybrid_vs_tod_v1.py --days 60
# ~10-20 min to run full backtest
```

**Option 2: Expand Schema First**
```bash
# 1. Create migration (add bid/ask OHLC columns)
alembic revision -m "expand_candles_full_ohlc"

# 2. Clear existing candles
TRUNCATE kalshi.candles_1m;

# 3. Run migration
alembic upgrade head

# 4. Update backfill_kalshi_candles.py to capture all fields

# 5. Re-backfill all history
python scripts/backfill_kalshi_candles.py --all-history
# ~2-3 min

# 6. Then run backtest
python scripts/backtest_hybrid_vs_tod_v1.py --days 60
```

---

## 13. My Questions for Professor

1. **Do you need intra-minute bid/ask resolution?** (open/high/low vs just close)

2. **Are you planning intraday strategies beyond 10:00 snapshots?**

3. **Storage preference:** Full OHLC (12 price fields) vs minimal (2 close prices)?

4. **Timing:** Expand schema now (one-time cost) vs later (if needed)?

5. **Visual Crossing integration:** Should we join 1-min candles with 1-min weather obs for features? (Separate question from schema design)

---

## 14. Recommendation

**My suggestion:**

1. ✅ **Keep current schema for now** - Run hybrid backtest with existing 972k test candles
2. ✅ Review backtest results with professor
3. ✅ **If backtest shows value** → Expand schema + backfill full history (~2 min)
4. ✅ **If backtest doesn't work** → Schema expansion doesn't matter

**Rationale:** Validate the trading strategy before optimizing data infrastructure. Classic "measure twice, cut once" approach.

---

**Status:** Awaiting professor's decision on schema expansion before proceeding with full backfill.

**Test period data:** Ready for immediate backtest (100% coverage, 0 NULLs)
