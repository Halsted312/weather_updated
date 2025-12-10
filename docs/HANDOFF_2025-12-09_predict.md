# Handoff: Live Inference Pipeline

**Date**: 2025-12-09
**Previous Agent**: kalshi-weather-quant
**Status**: Inference pipeline aligned, ready for live prediction
**Prerequisite**: Data is current and live ingestion running (see HANDOFF_2025-12-09.md)

---

## Overview

The inference pipeline is now aligned with training. Both use:
- `compute_snapshot_features()` from `models/features/pipeline.py`
- 220 features (219 numeric + 1 categorical "city")
- 5-minute market-clock snapshots
- Baseline per-city CatBoost models at `models/saved/{city}/ordinal_catboost_optuna.pkl`

---

## Quick Start: Run Inference

### 1. Single City Prediction

```python
from datetime import date, datetime
from zoneinfo import ZoneInfo

from models.inference.live_engine import LiveInferenceEngine
from src.db.connection import get_db_session

# Initialize engine (loads all 6 city models)
engine = LiveInferenceEngine()

with get_db_session() as session:
    # Get prediction for Chicago today
    result = engine.predict(
        city="chicago",
        event_date=date.today(),
        session=session,
    )

    if result:
        print(f"Expected settlement: {result.expected_settle:.1f}°F")
        print(f"Uncertainty (std): {result.settlement_std:.2f}°F")
        print(f"90% CI: [{result.ci_90_low}, {result.ci_90_high}]°F")
        print(f"Current max observed: {result.t_base}°F")
        print(f"\nBracket probabilities:")
        for ticker, prob in sorted(result.bracket_probs.items(), key=lambda x: -x[1])[:5]:
            print(f"  {ticker}: {prob:.1%}")
```

### 2. All Cities Scan

```python
from datetime import date
from models.inference.live_engine import LiveInferenceEngine
from src.db.connection import get_db_session

engine = LiveInferenceEngine()
cities = ["chicago", "austin", "denver", "los_angeles", "miami", "philadelphia"]

with get_db_session() as session:
    for city in cities:
        result = engine.predict(city, date.today(), session)
        if result:
            print(f"{city}: E[settle]={result.expected_settle:.1f}°F, "
                  f"std={result.settlement_std:.2f}°F, "
                  f"90%CI=[{result.ci_90_low},{result.ci_90_high}]")
        else:
            print(f"{city}: FAILED - check data")
```

---

## Prediction Result Structure

```python
@dataclass
class PredictionResult:
    city: str                      # 'chicago', 'austin', etc.
    event_date: date               # Settlement date
    bracket_probs: Dict[str, float]  # ticker → P(win)
    t_base: int                    # Current max observed temp (rounded)
    expected_settle: float         # E[settlement temp]
    settlement_std: float          # Std of settlement prediction
    ci_90_low: int                 # 90% CI lower bound
    ci_90_high: int                # 90% CI upper bound
    timestamp: datetime            # When prediction was made
    snapshot_hour: int             # Hour of snapshot used
```

### Key Metrics

| Metric | Description | Use |
|--------|-------------|-----|
| `expected_settle` | Expected final high temperature | Point estimate |
| `settlement_std` | Standard deviation of prediction | Model confidence |
| `ci_90_low/high` | 90% confidence interval | Risk bounds |
| `t_base` | Current observed max | Delta anchor |
| `bracket_probs` | P(win) per bracket | Trading signals |

---

## How Inference Works

### Data Flow

```
1. Load Data (loader.py)
   └── load_full_inference_data()
       ├── Observations (D-1 10:00 → cutoff)
       ├── T-1 daily forecast
       ├── T-1 hourly forecast
       ├── Multi-horizon forecasts (T-1 to T-6)
       ├── Market candles
       ├── City observations
       ├── NOAA guidance (NBM, HRRR)
       ├── 30-day stats
       └── Lag data (past 7 days settlements)

2. Build SnapshotContext (pipeline.py)
   └── Struct with all data for feature computation

3. Compute Features (pipeline.py)
   └── compute_snapshot_features()
       ├── Partial-day features (temp stats, slopes)
       ├── Forecast features (T-1 errors, multi-horizon)
       ├── Shape features (plateau, spike detection)
       ├── Calendar features (day-of-week, month)
       ├── Market features (candle momentum)
       ├── Station-city features (urban heat)
       └── NOAA features (NBM, HRRR guidance)

4. Add Lag Features (calendar.py)
   └── compute_lag_features()
       ├── settle_f_lag1/2/7
       ├── vc_max_f_lag1/7
       └── delta_vcmax_lag1

5. Model Prediction (live_engine.py)
   └── OrdinalDeltaTrainer.predict_proba()
       ├── 13 binary classifiers (ordinal regression)
       └── Returns P(delta) for delta ∈ [-2, +10]

6. Convert to Brackets (live_engine.py)
   └── _delta_to_bracket_probs()
       └── Map delta distribution → bracket P(win)
```

### Snapshot Timing

The engine uses **5-minute floored intervals** matching training:

```python
# Current time: 14:37:22
# Cutoff time:  14:35:00 (floored to 5-min)
```

This ensures observations up to 14:35 are included but not 14:36-14:37.

---

## Model Details

### Ordinal Delta Model

The model predicts **delta = settlement - t_base** where:
- `t_base` = current observed max temperature (rounded)
- `settlement` = final NWS settlement temperature
- `delta` ∈ [-2, +10] (13 classes)

```
Delta Classes: [-2, -1, 0, +1, +2, +3, +4, +5, +6, +7, +8, +9, +10]
              |____________|_______|_________________________________|
                cooling      match            warming
```

**Ordinal regression**: Uses 12 binary classifiers predicting P(delta ≤ k) for k in [-2, +9].
Bracket probabilities are derived from these cumulative probabilities.

### Per-City Models

Each city has its own model trained on city-specific data:

| City | Model Path | Training Period |
|------|------------|-----------------|
| chicago | `models/saved/chicago/ordinal_catboost_optuna.pkl` | Historical |
| austin | `models/saved/austin/ordinal_catboost_optuna.pkl` | Historical |
| denver | `models/saved/denver/ordinal_catboost_optuna.pkl` | Historical |
| los_angeles | `models/saved/los_angeles/ordinal_catboost_optuna.pkl` | Historical |
| miami | `models/saved/miami/ordinal_catboost_optuna.pkl` | Historical |
| philadelphia | `models/saved/philadelphia/ordinal_catboost_optuna.pkl` | Historical |

---

## Trading Integration

### Using Predictions for Trading

```python
from datetime import date
from models.inference.live_engine import LiveInferenceEngine
from src.db.connection import get_db_session

engine = LiveInferenceEngine()

with get_db_session() as session:
    result = engine.predict("chicago", date.today(), session)

    if result is None:
        print("Cannot trade - prediction failed")
        return

    # Check model confidence
    if result.settlement_std > 3.0:
        print(f"Model uncertain (std={result.settlement_std:.2f}°F) - reduce size or skip")

    # Find best bracket
    best_ticker = max(result.bracket_probs, key=result.bracket_probs.get)
    best_prob = result.bracket_probs[best_ticker]

    print(f"Best bracket: {best_ticker} with {best_prob:.1%} probability")

    # Edge calculation (simplified)
    # If market prices YES at 40¢ and model says 60%, edge = 20%
    market_price = 0.40  # Get from Kalshi API
    edge = best_prob - market_price

    if edge > 0.10:  # 10% edge threshold
        print(f"TRADE: Buy {best_ticker} at {market_price:.0%}, edge={edge:.1%}")
```

### Confidence Filters

```python
from config import live_trader_config as config

# These are checked automatically by predict():
# - MIN_OBSERVATIONS: Minimum obs required (default: 10)
# - REQUIRE_MIN_OBSERVATIONS: Whether to fail on insufficient obs
# - REQUIRE_MODEL_CONFIDENCE: Whether to check std/CI
# - MAX_MODEL_STD_DEGF: Max allowed std (default: ~4°F)
# - MAX_MODEL_CI_SPAN_DEGF: Max allowed 90% CI span (default: ~12°F)
```

---

## Debugging Inference

### Test Feature Parity

```bash
# Should show 220/220 features, 0 missing
python test_inference_parity.py chicago 2025-12-09
```

### Check Data Availability

```python
from datetime import date, datetime, time
from models.data.loader import load_full_inference_data
from src.db.connection import get_db_session

with get_db_session() as session:
    data = load_full_inference_data(
        "chicago",
        date.today(),
        datetime.combine(date.today(), time(14, 30)),
        session
    )

    print(f"Observations: {len(data['temps_sofar'])}")
    print(f"Window: {data['window_start']} → {data['cutoff_time']}")
    print(f"Forecast daily: {'Yes' if data['fcst_daily'] else 'No'}")
    print(f"Forecast hourly: {'Yes' if data['fcst_hourly_df'] is not None else 'No'}")
    print(f"Multi-horizon: {sum(1 for v in data['fcst_multi'].values() if v)}/6")
    print(f"Candles: {'Yes' if data['candles_df'] is not None else 'No'}")
    print(f"City obs: {'Yes' if data['city_obs_df'] is not None else 'No'}")
    print(f"NOAA guidance: {'Yes' if data['more_apis'] else 'No'}")
    print(f"Lag data: {len(data['lag_data'])} days")
```

### Inspect Feature Values

```python
from models.features.pipeline import SnapshotContext, compute_snapshot_features
from models.features.calendar import compute_lag_features

# After loading data...
ctx = SnapshotContext(
    city="chicago",
    event_date=date.today(),
    # ... other fields from data dict
)

features = compute_snapshot_features(ctx, include_labels=False)

# Add lag features
lag_df = data["lag_data"]
if lag_df is not None and not lag_df.empty:
    lag_fs = compute_lag_features(lag_df, "chicago", date.today())
    features.update(lag_fs.to_dict())

# Inspect key features
print(f"t_base: {features.get('t_base')}")
print(f"vc_max_f_sofar: {features.get('vc_max_f_sofar')}")
print(f"fcst_tmax_t1: {features.get('fcst_tmax_t1')}")
print(f"temp_slope_last_2h: {features.get('temp_slope_last_2h')}")
print(f"settle_f_lag1: {features.get('settle_f_lag1')}")
```

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `predict()` returns None | Missing observations | Check data ingestion |
| ValueError: Missing features | Pipeline mismatch | Run test_inference_parity.py |
| High settlement_std | Low confidence | More obs needed, or unusual day |
| Empty bracket_probs | No markets in DB | Check Kalshi market ingestion |
| Lag features NULL | No recent settlements | Backfill settlement data |

---

## Caching Behavior

The engine caches predictions to avoid re-running on every tick:

```python
engine = LiveInferenceEngine(inference_cooldown_sec=30.0)  # Default: 30 seconds

# First call: computes fresh prediction
result1 = engine.predict("chicago", date.today(), session)

# Within 30 sec: returns cached result
result2 = engine.predict("chicago", date.today(), session)

# Force refresh ignores cache
result3 = engine.predict("chicago", date.today(), session, force_refresh=True)
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `models/inference/live_engine.py` | Main inference orchestrator |
| `models/data/loader.py` | Data loading (load_full_inference_data) |
| `models/features/pipeline.py` | Feature computation (compute_snapshot_features) |
| `models/features/calendar.py` | Lag features (compute_lag_features) |
| `models/training/ordinal_trainer.py` | Model class (OrdinalDeltaTrainer) |
| `models/inference/probability.py` | Delta → settlement conversion |
| `config/live_trader_config.py` | Trading configuration |
| `test_inference_parity.py` | Feature parity validation |

---

## Next Steps for Live Trading

1. **Verify Inference Works**
   ```bash
   python test_inference_parity.py chicago
   ```

2. **Run All-City Scan**
   ```python
   # See "All Cities Scan" example above
   ```

3. **Integrate with Trading Scripts**
   - `scripts/live_ws_trader.py` - WebSocket-based live trading
   - `scripts/live_midnight_trader.py` - Midnight heuristic trading
   - `open_maker/manual_trade.py` - Manual CLI for discretionary trades

4. **Monitor Model Performance**
   - Track actual settlements vs predictions
   - Monitor std and CI span over time
   - Log bracket probability accuracy

---

## Summary

| Item | Status |
|------|--------|
| Inference aligned with training | ✅ DONE |
| 220/220 features computed | ✅ DONE |
| Lag features in live_engine.py | ✅ DONE |
| Per-city models loaded | ✅ READY |
| Bracket probability conversion | ✅ READY |
| Confidence filtering | ✅ READY |
| Caching for performance | ✅ READY |

**The inference pipeline is ready for live prediction. Focus on trading strategy integration.**
