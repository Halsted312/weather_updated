# Kalshi Weather Trading System - Developer Handoff Guide

## Executive Summary

This is a **quantitative trading system** for Kalshi weather prediction markets. It uses ML models to predict daily high temperatures and detect mispricings between model predictions and market prices.

**Core Pipeline:**
```
Weather Data → Features → Ordinal Model → Edge Detection → Edge Classifier → Live Trading
```

---

## 1. PROJECT STRUCTURE OVERVIEW

### Active Directories (USED)

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `models/features/` | Feature engineering (16 active modules) | `pipeline.py` (orchestrator) |
| `models/data/` | Data loading and dataset building | `loader.py`, `dataset.py` |
| `models/training/` | Model trainers | `ordinal_trainer.py` |
| `models/edge/` | Edge detection and classification | `classifier.py`, `detector.py` |
| `models/inference/` | Live prediction | `predictor.py`, `probability.py` |
| `models/saved/` | Trained model artifacts | `{city}/*.pkl`, `*.json` |
| `models/pipeline/` | Numbered pipeline scripts | `01_build_dataset.py` → `05_backtest_edge.py` |
| `scripts/` | Training and utility scripts | `train_edge_classifier.py` |
| `src/` | Core infrastructure | DB, Kalshi client, weather APIs |
| `config/` | Runtime configuration | `live_trader_config.py` |

### Deprecated/Unused

| Path | Status |
|------|--------|
| `models/data/deprecated/` | Old builders, moved |
| `models/features/interpolation.py` | Dead code (not imported) |
| `legacy/` | Old VC ingestion (reference only) |

---

## 2. FEATURE ENGINEERING (`models/features/`)

### Pipeline Flow

All features funnel through **`pipeline.py`** which calls 16 feature modules in sequence:

```
pipeline.py: compute_snapshot_features(SnapshotContext)
    │
    ├── Step 4:  partial_day.py    → t_base, vc_max_f_sofar, temp_range
    ├── Step 5:  shape.py          → plateau, spike, curvature
    ├── Step 6:  rules.py          → threshold crossings
    ├── Step 7:  calendar.py       → day_of_week, month, hour
    ├── Step 8:  quality.py        → num_samples, data completeness
    ├── Step 9:  forecast.py       → tempmax_f, errors, drift (5 sub-steps)
    ├── Step 10: momentum.py       → temp rates, EMA, time-since-max
    ├── Step 12: interactions.py   → regime (heating/plateau/cooling)
    ├── Step 14: market.py         → Kalshi bid/ask spread, volume
    ├── Step 15: station_city.py   → station vs city gap
    ├── Step 16: meteo.py          → humidity, wind, cloudcover
    ├── Step 16b: meteo_advanced.py → wet bulb, wind chill
    ├── Step 16c: engineered.py    → log/sqrt transforms, polynomials
    └── Final:  imputation.py     → fill None values
```

### File Details

| File | Lines | Purpose | Key Functions |
|------|-------|---------|---------------|
| **base.py** | 363 | Registry, FeatureSet class | `compose_features()`, `register_feature_group()` |
| **pipeline.py** | 689 | Central orchestrator | `SnapshotContext`, `compute_snapshot_features()` |
| **partial_day.py** | 200 | Core temp stats from observations | `compute_partial_day_features()` |
| **shape.py** | 206 | Plateau vs spike detection | `compute_shape_features()` |
| **rules.py** | 200 | Rule-based meta-features | `compute_rule_features()` |
| **calendar.py** | 283 | Time encoding, lags | `compute_calendar_features()` |
| **quality.py** | 173 | Data quality indicators | `compute_quality_features()` |
| **forecast.py** | 626 | T-1 forecast signals (largest) | 5 functions for different forecast aspects |
| **momentum.py** | 285 | Temperature trajectory | `compute_momentum_features()` |
| **interactions.py** | 366 | Regime, derived features | `compute_interaction_features()` |
| **market.py** | 257 | Kalshi candle signals | `compute_market_features()` |
| **station_city.py** | 174 | Station vs city comparison | `compute_station_city_features()` |
| **meteo.py** | 296 | Humidity, wind, cloudcover | `compute_meteo_features()` |
| **meteo_advanced.py** | 417 | Wet bulb, wind chill | `compute_meteo_advanced_features()` |
| **engineered.py** | 154 | Transform features | `compute_engineered_features()` |
| **imputation.py** | 295 | Null-filling strategies | `fill_*_nulls()` functions |
| ~~interpolation.py~~ | 184 | **DEAD CODE** - not imported anywhere | - |

---

## 3. DATA LOADING (`models/data/`)

### Active Files

| File | Purpose | Key Functions | Called By |
|------|---------|---------------|-----------|
| **loader.py** | DB interface (unified) | `load_vc_observations()`, `load_settlements()`, `load_historical_forecast_*()` | All dataset builders |
| **dataset.py** | **MODERN** flexible builder | `DatasetConfig`, `build_dataset()` | `train_city_ordinal_optuna.py` |
| **snapshot_builder.py** | Legacy hourly snapshots | `build_snapshot_dataset()` | `train_all_cities_hourly.py` |
| **tod_dataset_builder.py** | TOD v1 (15-min intervals) | `build_tod_snapshot_dataset()` | `train_tod_v1_all_cities.py` |
| **snapshot.py** | Thin wrapper to features | `build_snapshot()` → `compute_snapshot_features()` | dataset.py |
| **splits.py** | Temporal train/test splits | `DayGroupedTimeSeriesSplit` | All training |
| **vc_minute_queries.py** | VC minute data queries | SQL query builders | loader.py |

### Builder Comparison

| Builder | Snapshot Interval | Use Case | Status |
|---------|-------------------|----------|--------|
| `snapshot_builder.py` | Fixed hourly (8/day) | Legacy training | Active but old |
| `dataset.py` | Configurable (5-15 min) | **PREFERRED** | Modern |
| `tod_dataset_builder.py` | 15-min (56/day) | TOD v1 models | Specialized |

---

## 4. MODEL TRAINING (`models/training/`)

### Files

| File | Purpose | Key Class |
|------|---------|-----------|
| **ordinal_trainer.py** | Ordinal regression for delta | `OrdinalDeltaTrainer` |
| **catboost_trainer.py** | Base CatBoost wrapper | `CatBoostTrainer` |
| **logistic_trainer.py** | Logistic regression | `LogisticTrainer` |
| **base_trainer.py** | Abstract base class | `BaseTrainer` |

### Ordinal Model Architecture

```python
For each threshold k in [-1, 0, 1, ..., 10]:
    Train: P(delta >= k | features) using CatBoost

Final output: P(delta = k) = P(delta >= k) - P(delta >= k+1)
```

**Key Parameters:**
- `n_trials`: 80 (Optuna hyperparameter search)
- `base_model`: 'catboost'
- Outputs: `ordinal_catboost_optuna.pkl` + `.json`

---

## 5. EDGE DETECTION (`models/edge/`)

### Files

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| **implied_temp.py** | Convert predictions to temps | `ForecastImpliedResult`, `MarketImpliedResult`, `compute_*_implied_temp()` |
| **detector.py** | Compare forecast vs market | `EdgeSignal`, `EdgeResult`, `detect_edge()` |
| **classifier.py** | Filter profitable edges | `EdgeClassifier` (CatBoost + Optuna) |

### Edge Detection Flow

```
Ordinal Model → delta_probs → ForecastImpliedResult (E[settlement])
                                      ↓
Candle Prices → bracket probs → MarketImpliedResult (market pricing)
                                      ↓
                              detect_edge()
                                      ↓
                    EdgeSignal: BUY_HIGH / BUY_LOW / NO_TRADE
                                      ↓
                              EdgeClassifier.predict()
                                      ↓
                    P(profitable) >= threshold → TRADE / NO_TRADE
```

### Edge Classifier Training

**Input:** DataFrame with columns:
- `edge`, `confidence`, `forecast_temp`, `market_temp`
- `snapshot_hour`, `hours_to_event_close`
- `pnl` (target: >0 = profitable)

**Optuna tunes:**
- CatBoost hyperparameters
- `decision_threshold` (0.55-0.85)
- `calibration_method` (none/sigmoid/isotonic)

**Metrics optimized:** `sharpe`, `mean_pnl`, `filtered_precision`, `f1`, `auc`

---

## 6. INFERENCE (`models/inference/`)

| File | Purpose | Key Class |
|------|---------|-----------|
| **predictor.py** | Load model + predict live | `DeltaPredictor` |
| **probability.py** | Delta → bracket probs | `compute_bracket_probabilities()` |
| **live_engine.py** | Full live inference | `LiveInferenceEngine` |

---

## 7. TRAINING SCRIPTS (`scripts/`)

### Primary Scripts (USED)

| Script | Purpose | Output |
|--------|---------|--------|
| **train_edge_classifier.py** | Train edge classifier | `edge_classifier.pkl` |
| **train_city_ordinal_optuna.py** | Train ordinal model | `ordinal_catboost_optuna.pkl` |
| **train_tod_v1_all_cities.py** | Train TOD v1 models | `ordinal_catboost_tod_v1.pkl` |
| **export_kalshi_candles.py** | Export candles to parquet | `candles_{city}.parquet` |

### Supporting Scripts

| Script | Purpose |
|--------|---------|
| `backtest_edge_classifier.py` | Backtest edge classifier |
| `check_data_freshness.py` | Verify data recency |
| `live_ws_trader.py` | Live WebSocket trading |

---

## 8. INFRASTRUCTURE (`src/`)

### Database (`src/db/`)

| File | Purpose |
|------|---------|
| **connection.py** | SQLAlchemy engine, `get_engine()`, `get_session()` |
| **models.py** | ORM models for all tables |
| **checkpoint.py** | Ingestion checkpointing |

### Kalshi (`src/kalshi/`)

| File | Purpose |
|------|---------|
| **client.py** | REST + WebSocket client |
| **schemas.py** | Pydantic models for API |

### Weather (`src/weather/`)

| File | Purpose |
|------|---------|
| **visual_crossing.py** | VC Timeline API client |
| **nws_cf6.py** | NWS CF6 climate reports |
| **iem_cli.py** | Iowa Environmental Mesonet |
| **noaa_ncei.py** | NOAA historical data |

### Config (`src/config/`)

| File | Purpose |
|------|---------|
| **cities.py** | City definitions (KMDW, KAUS, etc.) |
| **settings.py** | Environment settings |
| **vc_elements.py** | VC API elements builder |

### Trading (`src/trading/`)

| File | Purpose |
|------|---------|
| **fees.py** | Kalshi fee calculations |
| **risk.py** | Position sizing, Kelly criterion |

---

## 9. SAVED MODELS (`models/saved/`)

### Per-City Structure

```
models/saved/{city}/
├── ordinal_catboost_optuna.pkl     # Ordinal model (trained)
├── ordinal_catboost_optuna.json    # Ordinal metadata
├── edge_classifier.pkl              # Edge classifier (trained)
├── edge_classifier.json             # Edge metrics
├── edge_training_data_realistic.parquet  # Cached edge samples
├── train_data_full.parquet          # Training snapshots
└── test_data_full.parquet           # Test snapshots
```

### Candle Exports

```
models/candles/
├── candles_chicago.parquet      (63 MB, 14.8M rows)
├── candles_austin.parquet       (45 MB, 11.2M rows)
├── candles_miami.parquet        (44 MB, 10.5M rows)
├── candles_denver.parquet       (23 MB, 5.0M rows)
├── candles_los_angeles.parquet  (20 MB, 4.5M rows)
└── candles_philadelphia.parquet (20 MB, 4.7M rows)
```

---

## 10. PIPELINE EXECUTION ORDER

### Full Training Pipeline

```bash
# 1. Build dataset (features + settlements)
python models/pipeline/01_build_dataset.py --city chicago

# 2. (Optional) Delta sweep analysis
python models/pipeline/02_delta_sweep.py --city chicago

# 3. Train ordinal model
python models/pipeline/03_train_ordinal.py --city chicago --trials 80

# 4. Train edge classifier
python scripts/train_edge_classifier.py --city chicago --trials 80 --optuna-metric sharpe

# 5. Backtest edge classifier
python models/pipeline/05_backtest_edge.py --city chicago
```

### Quick Commands

```bash
# Export candles (for remote training)
python scripts/export_kalshi_candles.py --all

# Train edge classifier with parquet (faster)
python scripts/train_edge_classifier.py --city austin --trials 80 --regenerate

# Live trading (dry-run)
python scripts/live_ws_trader.py --dry-run
```

---

## 11. KEY DESIGN DECISIONS

### Data Leakage Prevention

1. **Day-grouped splits**: `DayGroupedTimeSeriesSplit` keeps all snapshots from same day together
2. **Temporal ordering**: `shuffle=False` enforced in edge classifier
3. **T-1 forecasts only**: Never use same-day forecasts
4. **Cutoff filtering**: Features only use obs with `datetime_local < cutoff_time`

### Edge Threshold

- Default: **1.5°F** (forecast vs market difference)
- LA issue: Edge signals are mostly `buy_high` which loses money
- Fix needed: Add `min_edge` to Optuna search space

### 1-Minute Candle Usage (Current Limitation)

Currently only uses **last candle** per 15-min window:
```python
latest = candle_df.iloc[-1]
yes_bid = latest.get("yes_bid_close")
```

**Opportunity:** Aggregate 15 candles for volatility, volume, momentum features.

---

## 12. CONFIGURATION (`config/live_trader_config.py`)

### Key Settings

```python
# Trading thresholds
MIN_EV_PER_CONTRACT_CENTS = 3.0
MIN_BRACKET_PROB = 0.35
MAKER_FILL_PROBABILITY = 0.4

# Position limits
MAX_BET_SIZE_USD = 50.0
MAX_DAILY_LOSS_USD = 500.0
MAX_POSITIONS = 20

# Model variant
ORDINAL_MODEL_VARIANT = "tod_v1"  # baseline | hourly | tod_v1
```

---

## 13. DATABASE SCHEMA (Key Tables)

### Weather Schema (`wx.*`)

| Table | Purpose |
|-------|---------|
| `wx.vc_location` | Location dimension |
| `wx.vc_minute_weather` | 5-min observations + forecasts |
| `wx.settlement` | Daily TMAX ground truth |
| `wx.vc_forecast_daily` | T-1 to T-6 daily forecasts |
| `wx.vc_forecast_hourly` | T-1 hourly forecasts |

### Kalshi Schema (`kalshi.*`)

| Table | Purpose |
|-------|---------|
| `kalshi.candles_1m_dense` | 1-min market candles (bid/ask) |
| `kalshi.events` | Event metadata |
| `kalshi.markets` | Market/bracket metadata |

---

## 14. CITIES

| City | Station | Ticker Prefix | Timezone |
|------|---------|---------------|----------|
| Chicago | KMDW | KXHIGHCHI | America/Chicago |
| Austin | KAUS | KXHIGHAUS | America/Chicago |
| Denver | KDEN | KXHIGHDEN | America/Denver |
| Los Angeles | KLAX | KXHIGHLAX | America/Los_Angeles |
| Miami | KMIA | KXHIGHMIA | America/New_York |
| Philadelphia | KPHL | KXHIGHPHIL | America/New_York |

---

## 15. CURRENT STATUS (as of 2025-12-05)

### Edge Classifier Training Results

| City | Sharpe | Win Rate | Trades | Status |
|------|--------|----------|--------|--------|
| Chicago | 1.35 | 90.2% | 6,810 | Complete |
| Austin | 0.96 | 87.6% | 2,227 | Complete |
| Denver | 0.61 | 91.7% | 1,296 | Complete |
| Los Angeles | - | 0% | 0 | Broken (buy_high bias) |
| Miami | - | - | - | Pending |
| Philadelphia | - | - | - | Pending |

### Known Issues

1. **LA 0 trades**: Ordinal model has `buy_high` bias (97% of signals lose money)
2. **Candle underutilization**: Only using last candle of 15-min window
3. **Edge threshold fixed**: Should be tunable per-city via Optuna

---

## 16. GLOSSARY

| Term | Definition |
|------|------------|
| **Delta (Δ)** | Settlement temp - forecast base temp (in °F) |
| **Edge** | Forecast implied temp - market implied temp |
| **Ordinal model** | Predicts P(Δ = k) for k in [-2, -1, 0, 1, 2, ...] |
| **Edge classifier** | Binary classifier: P(trade profitable \| edge features) |
| **Snapshot** | Point-in-time feature vector (hourly or 15-min) |
| **Settlement** | Official daily high temp from NWS (ground truth) |
| **T-1 forecast** | Forecast made day before event (no lookahead) |
| **Bracket** | Kalshi market for temp range (e.g., 85-86°F) |
| **Maker** | Limit order (0% fee), lower fill probability |
| **Taker** | Market order (7% fee), immediate fill |

---

*Last updated: 2025-12-05*
