# Edge Classifier Complete File Reference & Data Flow

**Created:** 2025-12-03
**Purpose:** Complete mapping of all files, data flows, and connections for edge classifier training

---

## Complete File List (In Order of Execution)

### 1. Configuration Files
| File | Purpose | Key Content |
|------|---------|-------------|
| `src/config/cities.py` | City definitions | ICAO codes, timezones, Kalshi tickers |
| `src/config/vc_elements.py` | Visual Crossing elements | Weather data fields to fetch |
| `config/live_trader_config.py` | Trading parameters | Thresholds, position limits |

### 2. Database Schema & Models
| File | Purpose | Tables |
|------|---------|--------|
| `src/db/models.py` | SQLAlchemy models | All table definitions |
| `src/db/connection.py` | Database connection | Connection pooling |
| `migrations/versions/*.py` | Alembic migrations | Schema changes |

**Key Tables:**
- `wx.settlement` - Daily TMAX settlements (ground truth)
- `wx.vc_minute_weather` - 5-minute observations
- `wx.vc_forecast_daily` - Daily forecasts (T-1 through T-6)
- `kalshi.candles_1m_dense` - 1-minute market candles
- `kalshi.markets` - Market metadata

### 3. Data Ingestion Scripts
| File | Purpose | Output |
|------|---------|--------|
| `scripts/ingest_vc_obs_backfill.py` | Visual Crossing observations | → wx.vc_minute_weather |
| `scripts/ingest_vc_historical_forecast_parallel.py` | Historical forecasts | → wx.vc_forecast_daily |
| `scripts/ingest_settlement_multi.py` | NWS settlements | → wx.settlement |
| `scripts/backfill_kalshi_candles.py` | Kalshi candles | → kalshi.candles_1m_dense |
| `scripts/build_dense_candles.py` | Fill candle gaps | → kalshi.candles_1m_dense (dense) |

### 4. Feature Engineering
| File | Purpose | Features |
|------|---------|----------|
| `models/features/base.py` | Core types, DELTA_CLASSES | Feature registry |
| `models/features/partial_day.py` | Partial-day stats | vc_max_f_sofar, t_base, quantiles |
| `models/features/shape.py` | Temperature patterns | Plateau, spike, slope features |
| `models/features/forecast.py` | T-1 forecast features | Forecast errors, gaps |
| `models/features/calendar.py` | Time-of-day | Hour, day, seasonal encoding |
| `models/features/rules.py` | Heuristics | Rule-based predictions |
| `models/features/quality.py` | Data quality | Missing data flags |
| `models/features/meteo.py` | Meteorological | Humidity, wind, clouds |
| `models/features/market.py` | Market features | Spread, volume, volatility |
| `models/features/station_city.py` | Station vs city gap | Interpolation artifacts |

### 5. Dataset Building
| File | Purpose | Output |
|------|---------|--------|
| `models/data/loader.py` | Load raw data from DB | DataFrames |
| `models/data/snapshot_builder.py` | Build training snapshots | Snapshot datasets |
| `models/data/splits.py` | **DayGroupedTimeSeriesSplit** | Time-based CV |
| `models/pipeline/01_build_dataset.py` | Full pipeline | → train_data_full.parquet, test_data_full.parquet |
| `scripts/train_city_ordinal_optuna.py` | Dataset builder (called by pipeline) | Feature computation |

**Output Files Per City:**
- `models/saved/{city}/train_data_full.parquet` - Training snapshots
- `models/saved/{city}/test_data_full.parquet` - Test snapshots

### 6. Ordinal Model Training
| File | Purpose | Output |
|------|---------|--------|
| `models/training/base_trainer.py` | Base trainer class | Abstract methods |
| `models/training/ordinal_trainer.py` | **OrdinalDeltaTrainer** | K-1 binary classifiers |
| `models/pipeline/03_train_ordinal.py` | Training pipeline | → ordinal_catboost_optuna.pkl |

**Output Files Per City:**
- `models/saved/{city}/ordinal_catboost_optuna.pkl` - Trained model
- `models/saved/{city}/ordinal_catboost_optuna.json` - Metadata

### 7. Edge Module Implementation (NEW - Created in this session)
| File | Purpose | Key Functions |
|------|---------|---------------|
| `models/edge/__init__.py` | Module exports | Package initialization |
| `models/edge/implied_temp.py` | **Temperature inference** | compute_forecast_implied_temp(), compute_market_implied_temp() |
| `models/edge/detector.py` | **Edge detection** | detect_edge(), EdgeSignal, EdgeResult |
| `models/edge/classifier.py` | **EdgeClassifier ML model** | train(), predict(), save(), load() |

**Key Classes/Functions:**
- `ForecastImpliedResult` - Dataclass for forecast temps
- `MarketImpliedResult` - Dataclass for market temps
- `EdgeSignal` - Enum: BUY_HIGH, BUY_LOW, NO_TRADE
- `EdgeClassifier` - CatBoost + calibration + Optuna

### 8. Edge Classifier Training Script
| File | Purpose | Critical Functions |
|------|---------|-------------------|
| `scripts/train_edge_classifier.py` | **Main training script** | See detailed breakdown below |
| `models/pipeline/04_train_edge_classifier.py` | Pipeline wrapper | Calls train_edge_classifier.py |

**Critical Functions in train_edge_classifier.py:**
- **Line 54**: `CITY_CONFIG` - City ticker prefixes and timezones
- **Line 64**: `load_combined_data()` - Load train+test parquets
- **Line 109**: `load_bracket_candles_for_event()` - Load candles for one day (OLD CODE - not used)
- **Line 206**: `load_all_settlements()` - Batch load all settlements
- **Line 238**: `load_all_candles_batch()` - **CRITICAL: Batch load candles with both ticker formats**
- **Line 365**: `get_candles_from_cache()` - **CRITICAL: Filter candles by snapshot time (TIMEZONE BUG HERE!)**
- **Line 375**: `_process_single_day()` - Process one day's edges (threaded)
- **Line 512**: `generate_edge_data()` - Main edge generation loop

**Output Files Per City:**
- `models/saved/{city}/edge_training_data.parquet` - Generated edge dataset
- `models/saved/{city}/edge_classifier.pkl` - Trained EdgeClassifier
- `models/saved/{city}/edge_classifier.json` - Model metadata

### 9. Visualization Scripts (NEW - Created in this session)
| File | Purpose | Output |
|------|---------|--------|
| `visualizations/calibration_plots.py` | Reliability diagrams | Calibration curves |
| `visualizations/edge_reports.py` | Edge analysis | PnL/Sharpe plots |
| `scripts/visualize_edge_model.py` | Standalone visualization | → visualizations/edge/{city}/*.png |

### 10. Debug Scripts (Created in this session)
| File | Purpose |
|------|---------|
| `scripts/debug_edge_generation.py` | Debug ordinal model prediction |
| `scripts/test_process_day.py` | Test single day processing |
| `scripts/trace_single_snapshot.py` | Trace single snapshot end-to-end |

---

## Data Flow Diagram

```
DATABASE TABLES
├── wx.vc_minute_weather (observations)
├── wx.vc_forecast_daily (T-1 forecasts)
├── wx.settlement (TMAX ground truth)
└── kalshi.candles_1m_dense (market prices)
         ↓
DATASET BUILDING (models/pipeline/01_build_dataset.py)
├── Load observations, forecasts, settlements from DB
├── Compute features (models/features/*.py)
├── Create 5-min snapshots during market hours
└── Split into train/test by date
         ↓
OUTPUT: train_data_full.parquet, test_data_full.parquet
         ↓
ORDINAL MODEL TRAINING (models/pipeline/03_train_ordinal.py)
├── Load train/test parquets
├── Train K-1 binary classifiers (delta >= k)
├── Optuna hyperparameter tuning
└── Save model
         ↓
OUTPUT: ordinal_catboost_optuna.pkl
         ↓
EDGE DATA GENERATION (scripts/train_edge_classifier.py)
├── Load train_data_full + test_data_full
├── Load ordinal model
├── Batch load settlements from wx.settlement
├── Batch load candles from kalshi.candles_1m_dense
│   ├── Query BOTH formats: KXHIGH{CITY}- and HIGH{CITY}-
│   ├── Batch into 200-day chunks (avoid query size limit)
│   └── Build cache: (day, bracket_label) → DataFrame
├── For each day (threaded):
│   ├── For each snapshot:
│   │   ├── Run ordinal model → delta probabilities
│   │   ├── compute_forecast_implied_temp() → forecast temp
│   │   ├── get_candles_from_cache() → bracket candles **[TIMEZONE BUG HERE]**
│   │   ├── compute_market_implied_temp() → market temp
│   │   ├── detect_edge() → edge signal
│   │   └── Compute PnL using settlement
│   └── Append edge signals
└── Save edge dataset
         ↓
OUTPUT: edge_training_data.parquet
         ↓
EDGE CLASSIFIER TRAINING (models/edge/classifier.py)
├── Load edge_training_data.parquet
├── DayGroupedTimeSeriesSplit (time-based, no leakage)
├── Optuna: CatBoost + calibration + threshold
├── Evaluate on test set
└── Save model
         ↓
OUTPUT: edge_classifier.pkl, edge_classifier.json
```

---

## CRITICAL BUGS & FIXES APPLIED

### Bug #1: Ticker Format Compatibility
**Location:** `scripts/train_edge_classifier.py:280-282`
**Issue:** Only queried `KXHIGH{CITY}-` format, missed old `HIGH{CITY}-` format
**Fix:** Query BOTH formats with OR clause
**Status:** ✅ FIXED

### Bug #2: PostgreSQL Query Size Limit
**Location:** `scripts/train_edge_classifier.py:270-303`
**Issue:** 1062 days × 2 formats = 42KB query (too large)
**Fix:** Batch into 200-day chunks
**Status:** ✅ FIXED

### Bug #3: Ordinal Model Delta Range Mismatch
**Location:** `models/training/ordinal_trainer.py:416`
**Issue:** Austin model has deltas [-10, +10] but DELTA_CLASSES only [-2, +10]
**Fix:** Skip unmapped deltas
**Status:** ✅ FIXED

### Bug #4: Array vs Dict in compute_forecast_implied_temp
**Location:** `models/edge/implied_temp.py:82-95`
**Issue:** Function expected dict but got numpy array
**Fix:** Accept both dict and ndarray
**Status:** ✅ FIXED

### Bug #5: TIMEZONE CONVERSION BUG ⚠️ **STILL BROKEN FOR CHICAGO**
**Location:** `scripts/train_edge_classifier.py:377-386`
**Issue:** Hardcoded "America/Chicago" timezone for ALL cities
**Current Code:**
```python
if snapshot_ts.tz is None:
    snapshot_ts = snapshot_ts.tz_localize("America/Chicago", ...)  # WRONG FOR NON-CHICAGO CITIES!
```

**Problem:** Denver uses "America/Denver", LA uses "America/Los_Angeles", etc.

**Why Chicago fails even though it IS in America/Chicago:**
- The timezone localization might still have edge cases
- DST transitions still causing issues
- Candle timestamps might be in different timezone than expected

---

## IMMEDIATE FIX NEEDED: Dynamic Timezone Lookup

**Current Bug:** `get_candles_from_cache()` hardcodes "America/Chicago"

**Fix Required:**
```python
def get_candles_from_cache(candle_cache: dict, day: date, snapshot_time, city: str):
    # ... existing code ...

    # Get city-specific timezone from CITY_CONFIG
    city_tz = CITY_CONFIG[city]["tz"]

    if snapshot_ts.tz is None:
        try:
            snapshot_ts = snapshot_ts.tz_localize(city_tz, ambiguous='infer', nonexistent='shift_forward')
            snapshot_ts = snapshot_ts.tz_convert("UTC")
        except Exception:
            return {}
```

**Changes needed:**
1. `get_candles_from_cache()` must accept `city` parameter
2. `_process_single_day()` must pass `city` to `get_candles_from_cache()`
3. Use dynamic timezone from CITY_CONFIG

---

## DEBUG COMMAND FOR CHICAGO

Run this to see EXACTLY where Chicago is failing:

```bash
# Edit scripts/trace_single_snapshot.py:
# Change CITY = "austin" to CITY = "chicago"
# Change TEST_DATE to a known Chicago date with candles

PYTHONPATH=. python -c "
from datetime import date
import pandas as pd

# Find a Chicago date with candles
df = pd.read_parquet('models/saved/chicago/train_data_full.parquet')
chicago_days = sorted(df['day'].unique())
test_date = chicago_days[100]  # Pick day 100

print(f'Test date: {test_date}')
print(f'Expected timezone: America/Chicago')

# Check if candles exist for this date
from src.db import get_db_session
from sqlalchemy import text

with get_db_session() as session:
    suffix = test_date.strftime('%y%b%d').upper()
    query = text('''
        SELECT COUNT(*) FROM kalshi.candles_1m_dense
        WHERE (ticker LIKE :new OR ticker LIKE :old)
    ''')
    result = session.execute(query, {
        'new': f'KXHIGHCHI-{suffix}%',
        'old': f'HIGHCHI-{suffix}%'
    })
    count = result.fetchone()[0]
    print(f'Candles for {test_date}: {count} rows')
"

# Then trace through that specific date
```

---

## ALL FILES INVOLVED (For Senior Developer Review)

### Core Pipeline Files (11 files):
1. `scripts/train_edge_classifier.py` - **MAIN SCRIPT** (800+ lines)
2. `models/edge/classifier.py` - EdgeClassifier class (670 lines)
3. `models/edge/impl ied_temp.py` - Temperature inference (320 lines)
4. `models/edge/detector.py` - Edge detection (220 lines)
5. `models/training/ordinal_trainer.py` - Ordinal model (500+ lines)
6. `models/data/splits.py` - DayGroupedTimeSeriesSplit (220 lines)
7. `models/features/base.py` - DELTA_CLASSES, feature registry
8. `src/db/connection.py` - Database connection
9. `src/db/models.py` - Table schemas
10. `src/config/cities.py` - City configurations
11. `config/live_trader_config.py` - Trading parameters

### Supporting Files (8 files):
12. `models/features/*.py` - 10 feature modules
13. `visualizations/*.py` - 3 visualization modules
14. `scripts/visualize_edge_model.py` - Standalone viz
15. Debug scripts (3 files)

### Data Files Per City:
- `models/saved/{city}/train_data_full.parquet` - Input
- `models/saved/{city}/test_data_full.parquet` - Input
- `models/saved/{city}/ordinal_catboost_optuna.pkl` - Input (ordinal model)
- `models/saved/{city}/edge_training_data.parquet` - Generated
- `models/saved/{city}/edge_classifier.pkl` - Output
- `models/saved/{city}/edge_classifier.json` - Output

### Database Tables Used:
- `wx.settlement` - Daily TMAX (city, date_local, tmax_final)
- `kalshi.candles_1m_dense` - 1-min candles (ticker, bucket_start, yes_bid_close, yes_ask_close)

---

## SUSPECTED ROOT CAUSE (Chicago Failure)

**Hypothesis:** `get_candles_from_cache()` at line 377 hardcodes "America/Chicago" but the actual issue is:

1. **Candle timestamps are in UTC** (confirmed: `bucket_start: timestamp with time zone`)
2. **Snapshot times are NAIVE LOCAL time** (confirmed: no timezone info)
3. **Timezone conversion might be failing silently** or creating wrong comparisons

**Test to verify:**
```bash
PYTHONPATH=. python -c "
import pandas as pd
from datetime import datetime

# Check actual snapshot time format
df = pd.read_parquet('models/saved/chicago/train_data_full.parquet')
snapshot = df['cutoff_time'].iloc[100]

print(f'Snapshot: {snapshot}')
print(f'Type: {type(snapshot)}')
print(f'Has tzinfo: {hasattr(snapshot, \"tzinfo\")}')
print(f'Tzinfo value: {snapshot.tzinfo}')

# Convert to UTC
ts = pd.Timestamp(snapshot)
print(f'\\nAs Timestamp: {ts}')
print(f'TZ: {ts.tz}')

# Try localization
try:
    ts_local = ts.tz_localize('America/Chicago', ambiguous='infer')
    ts_utc = ts_local.tz_convert('UTC')
    print(f'Localized: {ts_local}')
    print(f'UTC: {ts_utc}')
except Exception as e:
    print(f'ERROR: {e}')
"
```

---

## RECOMMENDED NEXT STEPS

1. **Fix timezone handling** to use city-specific timezone dynamically
2. **Add extensive logging** to `_process_single_day()` to count failure reasons
3. **Test with ONE Chicago day** using debug scripts
4. **Once working, train all cities**

**Estimated time to fix:** 30-60 minutes
**Estimated time to train all 6 cities:** 8-10 hours
