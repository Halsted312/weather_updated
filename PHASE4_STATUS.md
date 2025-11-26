# Phase 4 Status - Kalshi Market Data Ingestion

**Date:** 2025-11-15
**Status:** IN PROGRESS - Discovery running in background

---

## Current State

### Background Process Running
**Process ID:** ecbe7b
**Command:**
```bash
python scripts/discover_all_cities.py \
    --start-date 2024-01-01 \
    --end-date 2025-11-14 \
    --cities chicago miami austin la denver philadelphia \
    --output data/kalshi_full_2024_2025
```

**Progress:** Processing Chicago market 285/4,104 (~7% complete for Chicago alone)
**Expected Duration:** ~3-4 hours total for all 7 cities
**Output Directory:** `data/kalshi_full_2024_2025/`

### What's Being Fetched
- **6 cities:** Chicago, Miami, Austin, LA, Denver, Philadelphia (NYC retired)
- **Date range:** 2024-01-01 to 2025-11-14 (683 days)
- **Estimated markets:** ~28,000 total (~4,000 per city, 6 bins per day)
- **Data types:**
  - Series metadata (1 per city)
  - Markets (ticker, strikes, status, settlement)
  - Trades (all historical trades per market)
  - 1-minute candles (aggregated from trades)
  - 5-minute candles (aggregated from trades)

---

## Files Created for Phase 4

### 1. Data Loader: `ingest/load_kalshi_data.py`
**Purpose:** Load parquet files from discovery into PostgreSQL

**What it does:**
- Reads parquet files from city subdirectories
- Loads series metadata using `upsert_series()`
- Loads markets using `upsert_market()`
- Bulk loads trades using `bulk_upsert_trades()` (chunked, 5k per batch)
- Bulk loads 1-min candles using `bulk_upsert_candles()` (chunked, 5k per batch)
- Bulk loads 5-min candles using `bulk_upsert_candles()` (chunked, 5k per batch)
- Logs ingestion stats using `log_ingestion()`
- Optional: Refreshes `wx.minute_obs_1m` materialized view

**Usage:**
```bash
# Load all cities
python ingest/load_kalshi_data.py \
    --input data/kalshi_full_2024_2025 \
    --refresh-grid

# Load specific cities
python ingest/load_kalshi_data.py \
    --input data/kalshi_full_2024_2025 \
    --cities chicago miami \
    --refresh-grid
```

**Database tables updated:**
- `series` - Series metadata
- `markets` - Market records with settlement data
- `trades` - All trade records
- `candles` - 1-min and 5-min OHLCV bars
- `ingestion_log` - Tracking metadata
- `wx.minute_obs_1m` - Materialized view (if `--refresh-grid`)

---

### 2. Coverage Checker: `scripts/check_phase4_coverage.py`
**Purpose:** Validate that all expected data is present after loading

**What it checks:**
1. **Markets:** Each city/date has 6 markets (bins), all settled
2. **Candles:** 1-minute candles cover trading hours
3. **Settlement:** NWS settlement data exists in `wx.settlement`
4. **VC Minutes:** 6 good cities have VC minute data with <2% forward-fill

**Usage:**
```bash
python scripts/check_phase4_coverage.py \
    --start-date 2024-01-01 \
    --end-date 2025-11-14
```

**Output:**
- Markets summary (per city): days with 6 markets, fully settled
- Candles summary: total candles, avg per day
- Settlement summary: sources (CLI, CF6, IEM_CF6, GHCND)
- VC summary: coverage %, avg forward-fill %

---

## Expected Output Structure

After discovery completes, the directory structure will be:

```
data/kalshi_full_2024_2025/
├── chicago/
│   ├── series.parquet          # 1 row (series metadata)
│   ├── markets.parquet         # ~4,104 rows (684 days × 6 bins)
│   ├── trades.parquet          # ~500k-1M rows
│   ├── candles_1m.parquet      # ~200k-400k rows
│   ├── candles_5m.parquet      # ~40k-80k rows
│   └── summary_report.txt      # Human-readable summary
├── miami/
│   └── ...
├── austin/
│   └── ...
├── la/
│   └── ...
├── denver/
│   └── ...
└── philadelphia/
    └── ...
```

---

## Next Steps (After Discovery Completes)

### Step 1: Verify Discovery Output
```bash
# Check that all cities have data
ls -lh data/kalshi_full_2024_2025/*/

# Check row counts
wc -l data/kalshi_full_2024_2025/*/summary_report.txt
```

### Step 2: Load Data into PostgreSQL
```bash
python ingest/load_kalshi_data.py \
    --input data/kalshi_full_2024_2025 \
    --refresh-grid
```

**Expected results:**
- Series: 7 rows
- Markets: ~28,000 rows
- Trades: ~3-7 million rows
- Candles (1-min): ~1-3 million rows
- Candles (5-min): ~200-600k rows

### Step 3: Run Coverage Checks
```bash
python scripts/check_phase4_coverage.py \
    --start-date 2024-01-01 \
    --end-date 2025-11-14
```

**Expected coverage:**
- Markets: 100% of days have 6 markets
- Settlement: 100% of days have NWS settlement
- VC: 6 cities with >99% complete days, <2% forward-fill
- NYC: Excluded from VC features (82% forward-fill)

### Step 4: Verify Candle Generation
Query the database to confirm both 1-min and 5-min candles exist:

```sql
-- Check candle counts per period
SELECT
    period_minutes,
    COUNT(*) as candle_count,
    COUNT(DISTINCT ticker) as unique_markets
FROM candles
GROUP BY period_minutes;

-- Expected:
-- period_minutes | candle_count | unique_markets
-- 1              | ~1-3M        | ~28,000
-- 5              | ~200-600k    | ~28,000
```

### Step 5: Quick Sanity Checks

**Markets vs Settlement:**
```sql
-- Each market date should have settlement
SELECT
    m.series_ticker,
    DATE(m.close_time AT TIME ZONE 'America/Chicago') as market_date,
    COUNT(m.ticker) as markets,
    COUNT(s.tmax_final_f) as settlements
FROM markets m
LEFT JOIN wx.settlement s
    ON s.date_local = DATE(m.close_time AT TIME ZONE 'America/Chicago')
WHERE m.status = 'settled'
GROUP BY m.series_ticker, market_date
HAVING COUNT(s.tmax_final_f) = 0;

-- Should return 0 rows (all markets have settlement)
```

**Candles vs Markets:**
```sql
-- Each settled market should have candles
SELECT
    ticker,
    status,
    (SELECT COUNT(*) FROM candles c WHERE c.ticker = m.ticker AND c.period_minutes = 1) as candles_1m
FROM markets m
WHERE status = 'settled'
  AND ticker IN (SELECT ticker FROM markets LIMIT 10);

-- All should have > 0 candles
```

**VC vs Settlement:**
```sql
-- 6 good cities should have VC data
SELECT
    s.loc_id,
    s.date_local,
    COUNT(DISTINCT vo.ts_utc) as vc_minutes,
    s.tmax_final_f
FROM wx.settlement s
LEFT JOIN wx.minute_obs vo
    ON vo.loc_id = s.loc_id
    AND DATE(vo.ts_utc AT TIME ZONE 'America/Chicago') = s.date_local
WHERE s.loc_id IN ('KAUS', 'KMDW', 'KLAX', 'KMIA', 'KDEN', 'KPHL')
  AND s.date_local >= '2024-01-01'
  AND s.date_local <= '2025-11-14'
GROUP BY s.loc_id, s.date_local, s.tmax_final_f
HAVING COUNT(DISTINCT vo.ts_utc) < 200;

-- Should return very few rows (complete days have 288 rows)
```

---

## Phase 5 Preview (After Phase 4 Complete)

Once Phase 4 is validated, proceed to Phase 5:

**Goal:** Build ML feature dataset using VC weather + market microstructure

**Key tasks:**
1. Extend `ml/dataset.py` to join:
   - Market candles (prices, spreads, volume, time-to-close)
   - VC minute features (temp, dew, humidity, windspeed)
   - NYC retired: `EXCLUDED_VC_CITIES` now empty, and no new NYC data is ingested
2. Configure walk-forward: train_days=90, test_days=7, step_days=7
3. Train Ridge per bracket type with calibration (isotonic/sigmoid)
4. Save per-window models and predictions

**Expected outcome:**
- Longer training history (90 days vs previous)
- VC weather features for 6 cities
- Improved calibration and Brier scores
- Ready for Phase 6 backtesting

---

## Important Notes

### VC Data Quality (From Phase 3)
- **6 Good Cities:** Austin, Chicago, LA, Miami, Denver, Philadelphia
  - Coverage: >99% complete days
  - Forward-fill: <2% average
  - Temp agreement with CF6: ~0.5°F average delta
- **NYC (Excluded from VC features):**
  - Coverage: 99.2% complete days
  - Forward-fill: **82.2%** (KNYC is climate station, not ASOS)
  - Keep NYC market features and labels, exclude VC minute features

### Database Schema Updates (Phase 3)
- `wx.minute_obs.ffilled` - Boolean flag for forward-filled vs real observations
- `wx.minute_obs.stations` - Station ID used by VC for diagnostics (VARCHAR(50))

### Existing Loader Functions (db/loaders.py)
All necessary functions already exist:
- `upsert_series()` - Series metadata
- `upsert_market()` - Individual markets
- `bulk_upsert_candles()` - Batch candle upserts
- `bulk_upsert_trades()` - Batch trade upserts
- `log_ingestion()` - Track ingestion metadata
- `refresh_1m_grid()` - Refresh materialized view

---

## Troubleshooting

### If Discovery Fails Partway
The discovery script is idempotent and can be restarted. It will skip cities that already have complete data.

```bash
# Check which cities completed
ls -lh data/kalshi_full_2024_2025/

# Restart for missing cities only
python scripts/discover_all_cities.py \
    --start-date 2024-01-01 \
    --end-date 2025-11-14 \
    --cities <missing_cities> \
    --output data/kalshi_full_2024_2025
```

### If Loading Fails
The loader uses idempotent upserts, so it can be rerun safely:

```bash
# Reload specific city
python ingest/load_kalshi_data.py \
    --input data/kalshi_full_2024_2025 \
    --cities chicago \
    --refresh-grid
```

### If Coverage is Low
Check ingestion logs and Kalshi API status:
- `ingestion_log` table for per-market stats
- Kalshi API docs for service status
- Market status field (should be 'settled' for historical data)

---

## Summary

**Phase 4 Status:** Discovery running (~7% complete)
**ETA:** ~3-4 hours
**Next Action:** Wait for discovery completion, then run loader and coverage checks
**Files Ready:** Loader and coverage checker scripts created and tested
**Database:** All necessary functions and schema updates in place
**Phase 5 Preview:** ML feature dataset with VC + market microstructure
