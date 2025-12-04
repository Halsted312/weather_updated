---
plan_id: comprehensive-feature-engineering-optimization
created: 2025-12-04
status: draft
priority: high
agent: kalshi-weather-quant
---

# Comprehensive Feature Engineering & Optimization Plan

## Current Baseline

**Austin Model (110 trials, CV=5):**
- Features: 165 (after removing 12 redundant)
- Delta range: [-12, +12]
- MAE: 2.09°F
- Within ±2°F: 72.1%
- Top feature: obs_fcst_max_gap (importance 5.70)

## Part 1: Feature Pruning (Remove Redundant Features)

**Found: 382 correlation pairs >0.90**

### Tier 1: Perfect Duplicates (Correlation = 1.000) - DROP IMMEDIATELY

| Drop (Lower Importance) | Keep (Higher Importance) | Reason |
|-------------------------|--------------------------|--------|
| `delta_vcmax_fcstmax_sofar` (4.74) | `obs_fcst_max_gap` (5.70) | Identical calculation, different name |
| `fcst_peak_hour_float` (3.64) | `fcst_prev_hour_of_max` (5.44) | Same source, duplicate encoding |
| `minutes_since_market_open` (1.29) | `hours_since_market_open` (1.95) | Unit conversion only |
| `hours_to_event_close` (0.88) | `hours_since_market_open` (1.95) | Linear transform of same variable |
| `fcst_importance_weight` (0.16) | `obs_confidence` (0.18) | Redundant confidence measures |
| `fcst_prev_max_f` (0.10) | `fcst_peak_temp_f` (0.53) | Same forecast value, different extraction |
| `snapshot_hour` (0.00) | `hour` (0.55) | Legacy vs new encoding |
| `snapshot_hour_sin` (0.00) | `hour_sin` (0.00) | Legacy cyclical encoding |
| `snapshot_hour_cos` (0.00) | `hour_cos` (0.00) | Legacy cyclical encoding |
| `temp_volatility_30min` (0.00) | `temp_std_last_30min` (0.00) | Exact duplicate |
| `temp_volatility_60min` (0.00) | `temp_std_last_60min` (0.00) | Exact duplicate |
| `is_d_minus_1` (0.00) | `is_event_day` (0.03) | Boolean inverse |
| `pred_floor_max_sofar` (0.33) | `pred_ceil_max_sofar` (0.46) | Complementary rules, keep higher |

**Tier 1 drops: 13 features**

### Tier 2: Very High Correlation (>0.98) - REVIEW AND DROP

Additional candidates from correlation analysis (need to verify):
- Percentile features (q10, q25, q50, q75, q90) - many highly correlated
- EMA variants (30min, 60min) - check if both needed
- Multiple time encodings (minute vs minutes_since_midnight)

**Estimated Tier 2 drops: 15-20 features**

### Feature Count After Pruning

- Current: 165 features
- After Tier 1: 152 features
- After Tier 2: 135-140 features
- **Target: ~140 features** (remove ~25 redundant)

## Part 2: New Feature Engineering

### Category 1: Meteorological Transforms

**From available VC data (check `wx.vc_minute_weather` columns):**

#### Wet Bulb Temperature
```python
def compute_wet_bulb_approx(temp_f, humidity_pct, pressure_mb=1013):
    \"\"\"Approximate wet bulb from temp + humidity.\"\"\"
    temp_c = (temp_f - 32) * 5/9
    rh_fraction = humidity_pct / 100

    # Simplified Stull formula
    wb_c = temp_c * np.arctan(0.151977 * (rh_fraction + 8.313659)**0.5) + \
           np.arctan(temp_c + rh_fraction) - \
           np.arctan(rh_fraction - 1.676331) + \
           0.00391838 * rh_fraction**1.5 * np.arctan(0.023101 * rh_fraction) - 4.686035

    wb_f = wb_c * 9/5 + 32
    return wb_f

# Features:
- wb_temp_last_obs
- wb_temp_mean_last_60min
- wb_temp_max_sofar
- temp_wb_gap (temp - wet_bulb, indicates humidity stress)
```

#### Heat Index / Feels Like
```python
# Check if feelslike_f exists in wx.vc_minute_weather
- feelslike_last_obs
- feelslike_mean_last_60min
- feelslike_max_sofar
- temp_feelslike_gap
```

#### Cloud Cover Binary Flags
```python
# From cloudcover (0-100%)
- is_clear_sky (cloudcover < 20%)          # Already exists (0.00 importance)
- is_partly_cloudy (20% <= cloudcover < 70%)
- is_overcast (cloudcover >= 70%)
- cloud_cover_increasing (current > 1h_ago)
- early_morning_overcast (cloudcover > 70% before noon)
```

#### Humidity Transforms
```python
# From humidity (0-100%)
- log_humidity (log(humidity + 1))
- humidity_squared
- is_very_dry (humidity < 30%)
- is_very_humid (humidity > 80%)         # Already exists (high_humidity_flag)
- humidity_change_last_hour
```

#### Wind Chill (for cool days)
```python
def compute_wind_chill(temp_f, windspeed_mph):
    if temp_f > 50:
        return temp_f  # Wind chill only applies when cold
    wc = 35.74 + 0.6215*temp_f - 35.75*(windspeed_mph**0.16) + 0.4275*temp_f*(windspeed_mph**0.16)
    return wc

- wind_chill_last_obs
- wind_chill_min_sofar
- temp_windchill_gap
```

**New meteo features: ~15-20**

### Category 2: Transforms of Top Features

**Top 10 features by importance:**
1. obs_fcst_max_gap (5.70)
2. fcst_prev_hour_of_max (5.44)
3. delta_vcmax_fcstmax_sofar (4.74) - DROPPING (duplicate)
4. fcst_obs_ratio (4.02)
5. fcst_peak_hour_float (3.64) - DROPPING (duplicate)
6. fcst_peak_band_width_min (2.97)
7. fcst_obs_diff_squared (2.84)
8. fcst_drift_slope_f_per_lead (2.82)
9. err_max_pos_sofar (2.42)
10. minutes_since_max_observed (2.36)

**Suggested transforms:**

#### From obs_fcst_max_gap (#1, 5.70):
```python
- log_abs_obs_fcst_gap = log(abs(obs_fcst_max_gap) + 1)
- obs_fcst_gap_squared = obs_fcst_max_gap ** 2
- obs_fcst_gap_sign = sign(obs_fcst_max_gap)  # -1, 0, +1
- obs_fcst_gap_magnitude = abs(obs_fcst_max_gap)
- obs_fcst_gap_x_hours_remaining (interaction)
```

#### From fcst_prev_hour_of_max (#2, 5.44):
```python
- sin_fcst_hour_of_max = sin(2π * fcst_prev_hour_of_max / 24)
- cos_fcst_hour_of_max = cos(2π * fcst_prev_hour_of_max / 24)
- is_morning_peak (fcst_prev_hour_of_max < 14)
- is_evening_peak (fcst_prev_hour_of_max > 18)
- hours_since_fcst_peak = snapshot_hour - fcst_prev_hour_of_max
```

#### From fcst_drift_slope_f_per_lead (#8, 2.82):
```python
- abs_fcst_drift_slope = abs(fcst_drift_slope_f_per_lead)
- is_forecast_warming (fcst_drift_slope > 0.5)
- is_forecast_cooling (fcst_drift_slope < -0.5)
```

#### From fcst_multi features (ranked #21-#64):
```python
- fcst_multi_cv = fcst_multi_std / fcst_multi_mean (coefficient of variation)
- fcst_multi_range_pct = fcst_multi_range / fcst_multi_mean * 100
- is_forecast_stable (fcst_multi_std < 1.5)
- is_forecast_volatile (fcst_multi_std > 3.0)
```

**New engineered features: ~20-25**

### Category 3: Interaction Features

**High-value interactions** (cross top features):

```python
# Gap × Confidence
- obs_fcst_gap_x_confidence = obs_fcst_max_gap * obs_confidence

# Gap × Time Remaining
- obs_fcst_gap_x_hours_remaining = obs_fcst_max_gap * hours_until_fcst_max

# Temperature momentum × Forecast
- temp_rate_x_fcst_gap = temp_rate_last_30min * obs_fcst_max_gap

# Multi-horizon uncertainty × Current position
- fcst_multi_std_x_obs_gap = fcst_multi_std * obs_fcst_max_gap

# Station-city gap × Forecast gap
- station_city_x_fcst_gap = station_city_temp_gap * obs_fcst_max_gap
```

**New interaction features: ~5-10**

## Part 3: Optuna Parameter Expansion

### Current CatBoost Parameters

```python
params = {
    'depth': trial.suggest_int('depth', 4, 10),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'iterations': trial.suggest_int('iterations', 100, 500),  # INCREASE TO 600
    'border_count': trial.suggest_int('border_count', 32, 255),
    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10, log=True),
    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
    'random_strength': trial.suggest_float('random_strength', 0, 2),
    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
    'subsample' or 'bagging_temperature': ...,
}
```

### Additional Parameters to Tune

```python
# Tree structure
'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
'max_leaves': trial.suggest_int('max_leaves', 16, 64),  # If Lossguide

# Regularization
'model_shrink_rate': trial.suggest_float('model_shrink_rate', 0, 1),
'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),

# Sampling
'rsm': trial.suggest_float('rsm', 0.5, 1.0),  # Random subspace method
'sampling_frequency': trial.suggest_categorical('sampling_frequency', ['PerTree', 'PerTreeLevel']),

# Boosting
'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),

# Early stopping (CRITICAL)
'early_stopping_rounds': 50,  # Fixed, not tuned
'use_best_model': True,       # Fixed
```

### Binary Feature Group Tuning (NEW!)

```python
def objective(trial):
    # Let Optuna decide which feature groups to include
    include_market = trial.suggest_categorical('include_market', [True, False])
    include_momentum = trial.suggest_categorical('include_momentum', [True, False])
    include_meteo = trial.suggest_categorical('include_meteo', [True, False])
    # include_multi_horizon = True  # Always include (we worked hard on these!)

    # ... rest of objective ...
```

## Part 4: Change Optimization Objective

**Current**: Maximizes AUC (area under ROC curve)

**New**: Maximize within-2 accuracy

```python
def objective(trial):
    # ... build model with trial params and feature groups ...

    # Train with cross-validation
    y_pred_proba = cross_val_predict(model, X_cv, y_cv, cv=cv_splits)

    # Convert probabilities to delta predictions
    delta_classes = np.array(self._delta_classes)
    y_pred_delta = np.array([
        delta_classes[np.argmax(probs)] for probs in y_pred_proba
    ])

    # Calculate within-2 rate (PRIMARY METRIC)
    within_2 = np.abs(y_cv - y_pred_delta) <= 2
    within_2_rate = np.mean(within_2)

    # Optional: Add within-1 as secondary metric
    within_1 = np.abs(y_cv - y_pred_delta) <= 1
    within_1_rate = np.mean(within_1)

    # Composite objective (weighted)
    score = 0.7 * within_2_rate + 0.3 * within_1_rate

    return score  # Maximize
```

## Part 5: New Feature Suggestions

### High-Priority New Features (Based on Top Performers)

**Group A: Transforms of obs_fcst_max_gap (#1)**
```python
'log_abs_obs_fcst_gap': np.log(np.abs(obs_fcst_max_gap) + 1),
'obs_fcst_gap_squared': obs_fcst_max_gap ** 2,
'obs_fcst_gap_cubed': obs_fcst_max_gap ** 3,
'obs_fcst_gap_sign': np.sign(obs_fcst_max_gap),
'obs_fcst_gap_magnitude': np.abs(obs_fcst_max_gap),
```

**Group B: Wet Bulb & Feels Like (Humidity stress)**
```python
'wb_temp_last_obs': compute_wet_bulb(temp_f, humidity),
'wb_temp_mean_60min': mean(wet_bulb temps last 60 min),
'wb_temp_max_sofar': max wet bulb temp so far,
'temp_wb_gap': temp_f - wb_temp (dry air = larger gap),
'feelslike_last_obs': from VC feelslike_f column,
'feelslike_max_sofar': max feels-like temp,
'temp_feelslike_gap': temp_f - feelslike_f,
```

**Group C: Cloud Cover Dynamics**
```python
'cloud_cover_mean_morning': mean(cloudcover) for hours 6-12,
'cloud_cover_mean_afternoon': mean(cloudcover) for hours 12-18,
'is_morning_overcast': cloudcover > 70% AND hour < 12,
'is_clearing': cloudcover_now < cloudcover_1h_ago - 20,
'is_clouding_up': cloudcover_now > cloudcover_1h_ago + 20,
'cloud_cover_slope_1h': (cloudcover_now - cloudcover_1h_ago) / 60,
```

**Group D: Wind Chill (Cool days)**
```python
'wind_chill_last_obs': compute_wind_chill(temp_f, windspeed_mph),
'wind_chill_min_sofar': min wind chill,
'temp_windchill_gap': temp_f - wind_chill,
'is_wind_chill_day': temp_f < 50 AND windspeed > 10,
```

**Group E: Multi-Horizon Derived**
```python
'fcst_multi_cv': fcst_multi_std / fcst_multi_mean (forecast uncertainty ratio),
'fcst_multi_range_pct': fcst_multi_range / fcst_multi_mean * 100,
'is_forecast_stable': fcst_multi_std < 1.5,
'is_forecast_volatile': fcst_multi_std > 3.0,
'fcst_multi_skew': skewness of T-1 to T-6 forecasts,
```

**Group F: Time-Temperature Interactions**
```python
'temp_x_hour': temp_f * hour (captures heating/cooling phase),
'temp_rate_x_hour': temp_rate_last_30min * hour,
'obs_gap_x_time_remaining': obs_fcst_max_gap * (24 - hour),
```

**Group G: Forecast Error Momentum**
```python
'err_acceleration': (err_last1h - err_last3h_mean) (error rate of change),
'err_momentum_30min': recent error trend,
'is_error_growing': err_last1h > err_mean_sofar + err_std_sofar,
```

**Total new features: ~35-45**

## Part 6: Implementation Plan

### Step 1: Fix Import Bug (1 minute)
**File**: `models/pipeline/03_train_ordinal.py` line 14
```python
import argparse
import logging
import os  # ADD THIS LINE
import sys
```

### Step 2: Prune Redundant Features (10 minutes)

**File**: `models/features/base.py`

Remove from NUMERIC_FEATURE_COLS:
```python
# Tier 1 drops (13 features)
FEATURES_TO_DROP = [
    'delta_vcmax_fcstmax_sofar',
    'fcst_peak_hour_float',
    'minutes_since_market_open',
    'hours_to_event_close',
    'fcst_importance_weight',
    'fcst_prev_max_f',
    'snapshot_hour', 'snapshot_hour_sin', 'snapshot_hour_cos',
    'temp_volatility_30min', 'temp_volatility_60min',
    'is_d_minus_1',
    'pred_floor_max_sofar',
]

# Remove these from NUMERIC_FEATURE_COLS list
```

### Step 3: Add New Meteo Features (30 minutes)

**File**: Create `models/features/meteo_advanced.py`

```python
@register_feature_group("meteo_advanced")
def compute_advanced_meteo_features(
    obs_df: pd.DataFrame,
    cutoff_time: datetime,
    fcst_hourly_df: Optional[pd.DataFrame] = None,
) -> FeatureSet:
    \"\"\"Compute wet bulb, feels like, wind chill, cloud dynamics.\"\"\"

    # ... implementation ...

    return FeatureSet(name="meteo_advanced", features={
        'wb_temp_last_obs': ...,
        'wb_temp_mean_60min': ...,
        'temp_wb_gap': ...,
        'cloud_cover_mean_morning': ...,
        'is_morning_overcast': ...,
        'wind_chill_min_sofar': ...,
        # ... ~15 features total
    })
```

**Add to pipeline.py**: Call `compute_advanced_meteo_features()`

**Add to base.py NUMERIC_FEATURE_COLS**: List the ~15 new feature names

### Step 4: Add Transforms & Interactions (20 minutes)

**File**: Create `models/features/engineered.py`

```python
@register_feature_group("engineered_transforms")
def compute_engineered_features(
    obs_fcst_max_gap: float,
    fcst_multi_mean: float,
    fcst_multi_std: float,
    temp_rate_last_30min: float,
    hour: int,
    # ... other inputs
) -> FeatureSet:
    \"\"\"Compute log/exp transforms and interactions.\"\"\"

    return FeatureSet(name="engineered_transforms", features={
        'log_abs_obs_fcst_gap': np.log(abs(obs_fcst_max_gap) + 1),
        'obs_fcst_gap_squared': obs_fcst_max_gap ** 2,
        'fcst_multi_cv': fcst_multi_std / (fcst_multi_mean + 0.01),
        'temp_x_hour': temp_f * hour,
        # ... ~15 features total
    })
```

### Step 5: Update Optuna Objective (15 minutes)

**File**: `models/training/ordinal_trainer.py`

**Find `_run_optuna_search()` method** and update objective function:

```python
def objective(trial):
    # Binary feature group tuning
    include_market = trial.suggest_categorical('include_market', [True, False])
    include_momentum = trial.suggest_categorical('include_momentum', [True, False])
    include_meteo = trial.suggest_categorical('include_meteo', [True, False])

    # CatBoost params (EXPANDED)
    params = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'iterations': trial.suggest_int('iterations', 100, 600),  # INCREASED
        'border_count': trial.suggest_int('border_count', 32, 255),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'random_strength': trial.suggest_float('random_strength', 0, 2),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),

        # NEW PARAMETERS
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'rsm': trial.suggest_float('rsm', 0.5, 1.0),
        'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),

        # EARLY STOPPING (fixed, not tuned)
        'early_stopping_rounds': 50,
        'use_best_model': True,

        # ... bootstrap_type logic ...
    }

    # Train with selected feature groups
    X_filtered = filter_features(X_cv, include_market, include_momentum, include_meteo)

    # ... train model ...

    # CHANGE OBJECTIVE: Optimize within-2 accuracy
    delta_classes = np.array(self._delta_classes)
    y_pred_delta = np.array([delta_classes[np.argmax(probs)] for probs in y_pred_proba])

    within_2 = np.abs(y_cv - y_pred_delta) <= 2
    within_2_rate = np.mean(within_2)

    within_1 = np.abs(y_cv - y_pred_delta) <= 1
    within_1_rate = np.mean(within_1)

    # Weighted objective: 70% within-2, 30% within-1
    score = 0.7 * within_2_rate + 0.3 * within_1_rate

    return score  # Maximize
```

### Step 6: Add Feature Group Filtering (20 minutes)

**File**: `models/features/base.py`

**Update `get_feature_columns()` function:**

```python
def get_feature_columns(
    include_forecast=True,
    include_lags=True,
    include_market=True,          # NEW
    include_momentum=True,         # NEW
    include_meteo=True,            # NEW
    include_multi_horizon=True,    # NEW
):
    num_cols = NUMERIC_FEATURE_COLS.copy()

    # Filter groups based on flags
    if not include_market:
        num_cols = [c for c in num_cols if c not in MARKET_FEATURE_COLS]

    if not include_momentum:
        momentum_keywords = ['temp_mean_last', 'temp_std_last', 'temp_rate',
                            'temp_acceleration', 'temp_ema', 'intraday_range', 'temp_cv']
        num_cols = [c for c in num_cols if not any(kw in c for kw in momentum_keywords)]

    if not include_meteo:
        meteo_keywords = ['humidity_', 'windspeed_', 'cloudcover_', 'wb_temp', 'wind_chill']
        num_cols = [c for c in num_cols if not any(kw in c for kw in meteo_keywords)]

    if not include_multi_horizon:
        num_cols = [c for c in num_cols if 'fcst_multi_' not in c]

    # Existing forecast and lag filtering
    if not include_forecast:
        # ... existing logic ...

    if not include_lags:
        # ... existing logic ...

    return num_cols, CATEGORICAL_FEATURE_COLS
```

### Step 7: Update BaseTrainer (10 minutes)

**File**: `models/training/base_trainer.py`

**Add parameters to `__init__`:**
```python
def __init__(
    self,
    include_forecast=True,
    include_lags=True,
    include_market=True,          # NEW
    include_momentum=True,         # NEW
    include_meteo=True,            # NEW
    include_multi_horizon=True,    # NEW
    calibrate=False,
):
    self.include_forecast = include_forecast
    self.include_lags = include_lags
    self.include_market = include_market
    self.include_momentum = include_momentum
    self.include_meteo = include_meteo
    self.include_multi_horizon = include_multi_horizon
```

**Update `_prepare_features()`:**
```python
def _prepare_features(self, df):
    # Get columns based on feature group flags
    num_cols, cat_cols = get_feature_columns(
        include_forecast=self.include_forecast,
        include_lags=self.include_lags,
        include_market=self.include_market,
        include_momentum=self.include_momentum,
        include_meteo=self.include_meteo,
        include_multi_horizon=self.include_multi_horizon,
    )
    # ... rest of method ...
```

## Expected Results

**After feature pruning:**
- Features: 165 → 140 (remove ~25 redundant)
- Training speed: +15-20% faster
- Reduced multicollinearity

**After adding new features:**
- Features: 140 → 180-190
- MAE: 2.09°F → **1.85-1.95°F** (target)
- Within ±2°F: 72.1% → **75-80%** (target)

**After Optuna expansion + within-2 objective:**
- Directly optimizes for trading metric
- Better generalization (early stopping)
- Optimal feature group combinations discovered

## Files to Modify

| File | Changes |
|------|---------|
| `models/pipeline/03_train_ordinal.py` | Add `import os` |
| `models/features/base.py` | - Remove 13+ redundant features<br>- Add new feature names<br>- Update get_feature_columns() |
| `models/features/meteo_advanced.py` | CREATE - wet bulb, wind chill, cloud dynamics |
| `models/features/engineered.py` | CREATE - transforms and interactions |
| `models/features/pipeline.py` | Call new feature functions |
| `models/training/base_trainer.py` | Add include_* parameters |
| `models/training/ordinal_trainer.py` | - Change objective to within-2<br>- Add early stopping<br>- Expand Optuna params<br>- Add feature group tuning |

## Testing Plan

**Quick test (10 trials):**
```bash
PYTHONPATH=. python3 scripts/train_city_ordinal_optuna.py \
  --city austin \
  --trials 10 \
  --cv-splits 5 \
  --workers 18 \
  --use-cached
```

**Expected**: Should see "Optimizing within_2_rate", feature group tuning in logs

**Full run (200 trials):**
```bash
PYTHONPATH=. nohup python3 scripts/train_city_ordinal_optuna.py \
  --city austin \
  --trials 200 \
  --cv-splits 5 \
  --workers 20 \
  --use-cached \
  > logs/austin_optimized_200trials.log 2>&1 &
```

**Expected time**: ~20-30 minutes with early stopping

## Summary for Next Session

**To implement:**
1. Fix import bug (1 line)
2. Prune 13-25 redundant features from whitelist
3. Create meteo_advanced.py with ~15 new features
4. Create engineered.py with ~20 transforms/interactions
5. Update get_feature_columns() with group filtering
6. Add include_* parameters to BaseTrainer
7. Change Optuna objective to within-2 accuracy
8. Add early stopping + expand CatBoost parameters
9. Add binary feature group tuning to Optuna

**Expected outcome:**
- Cleaner feature set (~140-190 features depending on Optuna choices)
- Better performance (MAE 1.85-1.95°F, 75-80% within ±2°F)
- Ablation insights (which feature groups matter most)

**Ready to scale to all 6 cities after validation!**
