# Time-of-Day Model Implementation Handoff

**Date:** 2025-11-28
**Status:** Feature engineering complete, models trained, needs integration
**For:** Implementer agent to complete LiveInferenceEngine integration

---

## ✅ What's Been Completed

### 1. Feature Engineering (DONE)

**Files Modified:**
- `models/features/calendar.py` - Now accepts `cutoff_time` (datetime) + 9 new tod features
- `models/features/quality.py` - Now accepts `cutoff_time` for exact time calculations
- `models/features/base.py` - Registered all tod feature columns
- `models/data/snapshot_builder.py` - `build_snapshot_for_inference()` accepts cutoff_time
- `models/data/tod_dataset_builder.py` - NEW: Builds 15-min training datasets

**Backward compatibility:** All functions accept BOTH `snapshot_hour` (int) and `cutoff_time` (datetime)

### 2. Training Infrastructure (DONE)

**Script:** `scripts/train_tod_v1_all_cities.py`
- Trains all 6 cities with configurable interval (15-min default)
- Configurable Optuna trials
- Saves to `models/saved/{city}_tod_v1/`

### 3. Models Trained (DONE)

**All 6 cities with 40 Optuna trials, 15-minute intervals:**

| City | Model Path | Accuracy | MAE |
|------|------------|----------|-----|
| Chicago | `models/saved/chicago_tod_v1/ordinal_catboost_tod_v1.pkl` | 58.9% | 0.60 |
| Austin | `models/saved/austin_tod_v1/ordinal_catboost_tod_v1.pkl` | 58.9% | 0.78 |
| Denver | `models/saved/denver_tod_v1/ordinal_catboost_tod_v1.pkl` | 54.0% | 1.01 |
| Los Angeles | `models/saved/los_angeles_tod_v1/ordinal_catboost_tod_v1.pkl` | 63.8% | 0.52 |
| Miami | `models/saved/miami_tod_v1/ordinal_catboost_tod_v1.pkl` | 71.7% | 0.39 |
| Philadelphia | `models/saved/philadelphia_tod_v1/ordinal_catboost_tod_v1.pkl` | 47.4% | 0.75 |

**Each folder contains:**
```
models/saved/{city}_tod_v1/
├── ordinal_catboost_tod_v1.pkl       # Trained model
├── best_params.json                   # Optuna hyperparameters
├── training_metadata.json             # Full training details
├── train_data.parquet                 # Training dataset (35k rows)
└── test_data.parquet                  # Test dataset (3.3k rows)
```

### 4. Config Infrastructure (DONE)

**File:** `config/live_trader_config.py`

**Added:**
```python
ORDINAL_MODEL_VARIANT = "hourly"  # Change to "tod_v1" to use tod models

MODEL_VARIANTS = {
    "baseline": {...},  # Sparse 8-hour models
    "hourly": {...},    # 14-hour models (CURRENT DEFAULT)
    "tod_v1": {...},    # 15-min tod models (AVAILABLE)
}

TOD_SNAPSHOT_INTERVAL_MIN = 15  # 15-minute snapshots
```

---

## ⏳ What Needs to Be Done (FOR YOU)

### Task 1: Update LiveInferenceEngine (HIGH PRIORITY)

**File:** `models/inference/live_engine.py`

**What to change:**

1. **Update model loading** (lines 65-97):
```python
def _load_all_models(self):
    variant = config.ORDINAL_MODEL_VARIANT
    variant_config = config.MODEL_VARIANTS[variant]

    for city in config.CITIES:
        folder = f"{city}{variant_config['folder_suffix']}"
        model_path = config.MODEL_DIR / folder / variant_config['filename']
        # ... load model

    self.variant_config = variant_config  # Store for inference
```

2. **Replace `_snap_to_training_hour()`** with `_get_snapshot_params()` (lines 242-244):
```python
def _get_snapshot_params(self, current_time: datetime) -> tuple[datetime, Optional[int]]:
    """Get snapshot parameters based on model variant."""

    if not self.variant_config['requires_snapping']:
        # TOD model: use exact timestamp (floored to interval)
        interval_min = config.TOD_SNAPSHOT_INTERVAL_MIN
        total_minutes = current_time.hour * 60 + current_time.minute
        floored_minutes = (total_minutes // interval_min) * interval_min

        cutoff_time = current_time.replace(
            hour=floored_minutes // 60,
            minute=floored_minutes % 60,
            second=0,
            microsecond=0
        )
        return cutoff_time, None

    else:
        # Baseline/hourly: snap to nearest training hour
        snapshot_hour = min(
            self.variant_config['snapshot_hours'],
            key=lambda x: abs(x - current_time.hour)
        )
        cutoff_time = current_time.replace(hour=snapshot_hour, minute=0, second=0, microsecond=0)
        return cutoff_time, snapshot_hour
```

3. **Update `predict()` method** (line 136-180):
```python
def predict(...) -> Optional[PredictionResult]:
    current_time = current_time or datetime.now(city_tz)

    # Get snapshot parameters based on variant
    cutoff_time, snapshot_hour = self._get_snapshot_params(current_time)

    # Build features
    features = build_snapshot_for_inference(
        city=city,
        day=event_date,
        temps_sofar=temps_sofar,
        timestamps_sofar=timestamps_sofar,
        cutoff_time=cutoff_time,  # Use this instead of snapshot_hour
        snapshot_hour=snapshot_hour,  # For backward compat
        fcst_daily=fcst_daily,
        fcst_hourly_df=fcst_hourly_df,
    )
```

**Critical:** Don't change `PredictionResult` interface - live trading depends on it!

---

### Task 2: Update Adhoc Tools (MEDIUM PRIORITY)

**Files:**
- `tools/adhoc/predict_now.py`
- `tools/adhoc/predict_all_cities.py`

**What to change:**

1. **Detect model variant from metadata:**
```python
model_metadata = trainer._metadata
is_tod_model = model_metadata.get('model_variant') == 'tod_v1'
```

2. **Conditional time handling:**
```python
if is_tod_model:
    # Use exact time (floored to interval)
    interval_min = model_metadata.get('snapshot_interval_min', 15)
    total_minutes = hour * 60 + minute
    floored_minutes = (total_minutes // interval_min) * interval_min
    cutoff_hour = floored_minutes // 60
    cutoff_minute = floored_minutes % 60
    cutoff_time = datetime.combine(target_date, ...).replace(hour=cutoff_hour, minute=cutoff_minute)
    snapshot_hour = None
else:
    # Baseline/hourly: snap to nearest hour
    snapshot_hour = snap_to_nearest_snapshot_hour(hour)
    cutoff_time = datetime.combine(target_date, ...).replace(hour=snapshot_hour, minute=0)
```

3. **Update snapshot building:**
```python
snapshot = build_snapshot_for_inference(
    city=city,
    day=target_date,
    temps_sofar=temps_sofar,
    timestamps_sofar=timestamps_sofar,
    cutoff_time=cutoff_time,  # Primary
    snapshot_hour=snapshot_hour,  # For backward compat
    fcst_daily=fcst_daily,
    fcst_hourly_df=fcst_hourly_df,
)
```

---

### Task 3: Test Integration (REQUIRED BEFORE LIVE)

**Step 1: Test Model Switching**

Edit `config/live_trader_config.py`:
```python
ORDINAL_MODEL_VARIANT = "tod_v1"  # Switch to tod
```

Run adhoc tool:
```bash
.venv/bin/python tools/adhoc/predict_now.py
```

**Expected output:**
```
City: Chicago (KMDW)
Snapshot Time: 10:17 local → Using 10:15 model (15-min interval)
Current Observed Max: 32°F

MODEL PREDICTION:
Delta = Settlement - Current Max
Most Likely Delta: +2°F
Expected Settlement: 34°F ± 1.5°F

DELTA PROBABILITY DISTRIBUTION:
  delta=+2: 42.3% ⭐
  delta=+3: 31.2%
  ...
```

**Step 2: Compare All 3 Variants**

Run predictions with each variant:
```bash
# Edit config, set ORDINAL_MODEL_VARIANT = "baseline"
.venv/bin/python tools/adhoc/predict_now.py > baseline_output.txt

# Edit config, set ORDINAL_MODEL_VARIANT = "hourly"
.venv/bin/python tools/adhoc/predict_now.py > hourly_output.txt

# Edit config, set ORDINAL_MODEL_VARIANT = "tod_v1"
.venv/bin/python tools/adhoc/predict_now.py > tod_output.txt
```

Compare predictions for same time - should be similar but tod uses more data points.

**Step 3: Test Live Inference Engine**

If you have `live_ws_trader.py` or similar:
```python
# Set config to tod_v1
engine = LiveInferenceEngine()

# Should load tod models from {city}_tod_v1/ folders
# Should use exact timestamps (not snap to hour)
```

---

## 📊 Model Performance Summary

### Comparison: Hourly (80 trials) vs TOD (40 trials)

| City | Hourly Acc | TOD Acc | Hourly MAE | TOD MAE | Winner |
|------|------------|---------|------------|---------|--------|
| **Miami** | 65.5% | **71.7%** | 0.46 | **0.39** | 🏆 TOD |
| **LA** | 62.6% | **63.8%** | 0.56 | **0.52** | 🏆 TOD |
| **Chicago** | 56.7% | **58.9%** | 0.64 | **0.60** | 🏆 TOD |
| **Austin** | **66.2%** | 58.9% | **0.49** | 0.78 | ⚠️ Hourly |
| **Denver** | **60.4%** | 54.0% | **0.73** | 1.01 | ⚠️ Hourly |
| **Philadelphia** | **50.2%** | 47.4% | **0.70** | 0.75 | ⚠️ Hourly |

**Note:** TOD used 40 trials vs hourly's 80 trials. TOD might improve with more Optuna tuning.

**TOD wins:** 3/6 cities (Miami, LA, Chicago)
**Hourly wins:** 3/6 cities (Austin, Denver, Philadelphia)

**Recommendation:** Test both in dry-run mode and choose based on actual P&L performance.

---

## 🔧 Configuration Reference

### Model Variant Selection

**To use TOD models in live trading:**

Edit `config/live_trader_config.py`:
```python
ORDINAL_MODEL_VARIANT = "tod_v1"
```

Restart trader - it will automatically:
- Load tod models from `{city}_tod_v1/` folders
- Use 15-minute time flooring (10:17 → 10:15, 14:32 → 14:30)
- Generate predictions with full time-of-day features

**To rollback to hourly:**
```python
ORDINAL_MODEL_VARIANT = "hourly"
```

Restart - instant rollback, no code changes needed.

---

## 📁 File Locations Summary

### Models

**Baseline (sparse):**
```
models/saved/chicago/ordinal_catboost_optuna.pkl
models/saved/austin/ordinal_catboost_optuna.pkl
... (6 cities)
```

**Hourly (14 hours, 80 trials):**
```
models/saved/chicago_hourly80/ordinal_catboost_hourly_80trials.pkl
models/saved/austin_hourly80/ordinal_catboost_hourly_80trials.pkl
... (6 cities)
```

**TOD (56 snapshots/day, 40 trials):**
```
models/saved/chicago_tod_v1/ordinal_catboost_tod_v1.pkl
models/saved/austin_tod_v1/ordinal_catboost_tod_v1.pkl
... (6 cities)
```

### Code

**Feature Engineering:**
- `models/features/calendar.py` - Time-of-day features
- `models/features/quality.py` - Cutoff-time aware
- `models/features/base.py` - Feature registry
- `models/data/snapshot_builder.py` - Updated signature

**Training:**
- `models/data/tod_dataset_builder.py` - TOD dataset generator
- `scripts/train_tod_v1_all_cities.py` - TOD training script

**Inference (NEEDS YOUR WORK):**
- `models/inference/live_engine.py` - ❌ NOT UPDATED YET
- `tools/adhoc/predict_now.py` - ❌ NOT UPDATED YET
- `tools/adhoc/predict_all_cities.py` - ❌ NOT UPDATED YET

**Config:**
- `config/live_trader_config.py` - ✅ Variant config added

**Documentation:**
- `.claude/plans/replicated-bubbling-gehret.md` - Full implementation plan
- `docs/INFERENCE_SYSTEM_HANDOFF.md` - Current system architecture
- `models/reports/HOURLY_VS_TOD_COMPARISON.md` - Performance comparison

---

## 🎯 Your Tasks

### Priority 1: Update LiveInferenceEngine

**Goal:** Make it variant-aware (loads correct models, handles tod timestamps)

**Changes needed:**
1. Variant-aware model loading
2. Replace `_snap_to_training_hour()` with `_get_snapshot_params()`
3. Pass `cutoff_time` to `build_snapshot_for_inference()`

**Test:** Run with `ORDINAL_MODEL_VARIANT = "tod_v1"` and verify no errors

### Priority 2: Update Adhoc Tools

**Goal:** Support tod models (no snapping when using tod_v1)

**Changes needed:**
1. Detect model variant from metadata
2. Conditional time handling (snap vs exact)
3. Pass `cutoff_time` parameter

**Test:** Run predict_now.py at 10:17am and verify it uses 10:15 model (not 10:00)

### Priority 3: Testing & Validation

**Create:** `backtest/compare_tod_vs_hourly.py`

**Goal:** Compare actual P&L in backtest mode

**Test both variants:**
- Run same days/times with hourly vs tod_v1
- Compare: accuracy, MAE, calibration, pseudo-P&L
- Decide which is better for production

---

## 💡 Quick Start for Testing

### Test TOD Models Work

```bash
# 1. Load tod model directly
.venv/bin/python3 <<'EOF'
from models.training.ordinal_trainer import OrdinalDeltaTrainer

trainer = OrdinalDeltaTrainer()
trainer.load("models/saved/chicago_tod_v1/ordinal_catboost_tod_v1.pkl")

print(f"Loaded: {trainer._metadata['model_variant']}")
print(f"Interval: {trainer._metadata['snapshot_interval_min']} min")
print(f"Snapshots/day: {trainer._metadata['n_snapshots_per_day']}")
print(f"Delta range: {trainer._metadata['delta_range']}")
print(f"Accuracy: {trainer._metadata['accuracy']*100:.1f}%")
EOF
```

**Expected output:**
```
Loaded: tod_v1
Interval: 15 min
Snapshots/day: 56
Delta range: [-2, 10]
Accuracy: 58.9%
```

### Test Time-of-Day Features

```bash
.venv/bin/python3 <<'EOF'
import pandas as pd

# Load tod training data
df = pd.read_parquet("models/saved/chicago_tod_v1/train_data.parquet")

print(f"Columns: {len(df.columns)}")
print(f"\nTime-of-day features:")
for col in ['hour', 'minute', 'minutes_since_midnight', 'hour_sin', 'minute_sin', 'time_of_day_sin']:
    if col in df.columns:
        print(f"  ✅ {col}")

print(f"\nForecast features:")
for col in ['fcst_prev_max_f', 'fcst_prev_mean_f']:
    if col in df.columns:
        null_pct = df[col].isna().sum() / len(df) * 100
        print(f"  ✅ {col}: {null_pct:.1f}% null")
EOF
```

**Expected:**
- 87 columns
- All tod features present
- Forecast features present with low nulls (~0.3%)

---

## 🚨 Known Issues & Workarounds

### Issue 1: Missing Error Features

**Warning during training:**
```
Missing columns (will fill with NaN): err_mean_sofar, err_std_sofar, ...
```

**Impact:** 7 forecast error features are NaN
**Workaround:** Model still works, fills with NaN (acceptable)
**Fix:** Add intraday forecast error computation to tod_dataset_builder (future enhancement)

### Issue 2: TOD vs Hourly Performance

**Some cities:** TOD worse than hourly (Austin, Denver)
**Some cities:** TOD better than hourly (Miami, LA, Chicago)

**Hypothesis:** 40 trials vs 80 trials difference, or city-specific patterns
**Solution:** Run more Optuna trials for tod (80-100) OR use backtest to determine best variant per city

### Issue 3: Interval Parameter

**Current:** `--interval 15` means 15-minute snapshots
**User wants:** Predictions "2 seconds after each other"

**Clarification needed:**
- Interval is for **training snapshots**, not prediction frequency
- With 15-min tod models, you can predict at 10:00, 10:15, 10:30, etc.
- For "continuous" predictions, you'd floor current time to nearest 15-min mark
- To predict every minute, you'd need `--interval 1` (impractical - 840 snapshots/day)

---

## 📖 Usage Examples

### Example 1: Use TOD in Adhoc Tool

```python
# tools/adhoc/config.py
CITY = "miami"
TIME = "1417"  # 2:17pm
DATE = "today"

# After you update predict_now.py to support tod:
# .venv/bin/python tools/adhoc/predict_now.py

# With tod_v1: Uses 14:15 snapshot (floored to 15-min)
# With hourly: Uses 14:00 snapshot (snapped to hour)
```

### Example 2: Switch Variants in Live Trader

```python
# config/live_trader_config.py
ORDINAL_MODEL_VARIANT = "tod_v1"

# Restart live trader
# It will now:
# - Load tod models
# - Use 15-min time flooring
# - Predict at 10:15, 10:30, 10:45, etc.
```

---

## 🎯 Success Criteria

Before using tod_v1 in production:

1. ✅ **LiveInferenceEngine updated** and tested
2. ✅ **Adhoc tools work** with tod models
3. ✅ **Dry-run testing** completed (3-5 days)
4. ✅ **Backtest comparison** shows tod ≥ hourly
5. ✅ **No interface changes** broke live trading

Then:
```python
# config/live_trader_config.py
ORDINAL_MODEL_VARIANT = "tod_v1"
```

Restart and monitor closely!

---

## 🔗 Critical Files Reference

**Read these before starting:**
- `.claude/plans/replicated-bubbling-gehret.md` - Full implementation plan
- `models/inference/live_engine.py` - What you need to modify
- `config/live_trader_config.py` - Model variant config

**Reference for patterns:**
- `models/data/snapshot_builder.py:93-203` - How hourly loads forecasts
- `models/data/tod_dataset_builder.py:40-180` - How tod builds datasets
- `models/features/calendar.py:35-132` - Backward-compatible parameter pattern

---

**Status:** Ready for LiveInferenceEngine integration and testing!
**Estimated effort:** 2-3 hours to complete all tasks + testing
**Risk:** Low (backward compatibility preserved, easy rollback)
