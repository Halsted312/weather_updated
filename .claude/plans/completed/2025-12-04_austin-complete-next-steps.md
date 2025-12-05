---
plan_id: austin-model-optimization-and-next-steps
created: 2025-12-04
status: draft
priority: medium
agent: kalshi-weather-quant
---

# Austin Model Complete - Optimization & Next Steps

## Current Status ✅

**Austin Ordinal Model (Step 3 Complete):**
- 177 features (multi-horizon, station-city, temp momentum, meteo, market)
- Delta range: [-12, +12] (25 classes, 90.4% coverage)
- MAE: 2.09°F
- Within ±2°F: 72.1%
- 110 trials, CV=5, 20 workers
- **Production ready!**

## Issue 1: Missing 25 Features

**Parquet has 207 columns, model uses 177.**

**Missing 25 features (most correctly excluded):**

### Deliberately Excluded (Should NOT add):
1. **City one-hot encodings (6)**: `city_austin`, `city_chicago`, etc.
   - Already have `city` as categorical feature
   - One-hot redundant with CatBoost's native categorical handling

2. **Error-on-rule features (7)**: `err_c_first_sofar`, `err_ceil_max_sofar`, etc.
   - These compare settlement to rule predictions
   - **Potential target leakage** (rules computed from settlement)
   - Correctly excluded

3. **Redundant rule predictions (4)**: `pred_c_first_sofar`, `pred_max_round_sofar`, etc.
   - Duplicates of rules already included
   - Low differentiation from t_base

4. **Interaction terms (3)**: `temp_x_day_fraction`, `temp_x_hours_remaining`, `temp_zscore_vs_forecast`
   - Can be reconstructed from base features
   - Not critical

### Should Consider Adding (5):
1. **city_warmer_flag** - Binary indicator if city warmer than station
2. **max_gap_minutes** - Max gap in observation window
3. **disagree_flag_sofar** - Rules disagree on prediction
4. **minutes_ge_base_p1**, **max_run_ge_base_p1** - Plateau detection variants

**Recommendation:** Add the 5 "consider" features → **182 total features**

## Issue 2: Import Bug

**File**: `models/pipeline/03_train_ordinal.py` line 41
**Fix**: Add `import os` after line 14

```python
import argparse
import logging
import os  # ADD THIS
import sys
from pathlib import Path
```

## Issue 3: Optuna Parameter Expansion

**Current CatBoost parameters tuned:**
- `depth`: [4, 10]
- `learning_rate`: [0.01, 0.3]
- `iterations`: [100, 500]
- `border_count`: [32, 255]
- `l2_leaf_reg`: [0.1, 10]
- `min_data_in_leaf`: [1, 50]
- `random_strength`: [0, 2]
- `colsample_bylevel`: [0.3, 1.0]
- `subsample` or `bagging_temperature`: [0.5, 1.0]

**Suggested additions for better performance:**

###  Additional CatBoost Parameters:
1. **grow_policy**: ['SymmetricTree', 'Depthwise', 'Lossguide']
   - Different tree building strategies
   - Lossguide can improve accuracy

2. **max_leaves**: [16, 64] (if using Lossguide)
   - Controls tree complexity

3. **min_child_samples**: [1, 100]
   - More granular than min_data_in_leaf

4. **rsm** (feature sampling): [0.5, 1.0]
   - Random subspace method

5. **sampling_frequency**: ['PerTree', 'PerTreeLevel']
   - When to sample features

6. **boosting_type**: ['Ordered', 'Plain']
   - Ordered can reduce overfitting

7. **leaf_estimation_iterations**: [1, 10]
   - Gradient steps per leaf

**Expected impact:** 2-5% improvement in MAE (from 2.09°F to ~2.0°F)

## Issue 4: Feature Category Flags

**Desired flexibility:** Toggle feature groups for ablation studies

**Current state:**
- `include_forecast`: Exists in BaseTrainer ✓
- `include_lags`: Exists in BaseTrainer ✓
- `include_market`: **Missing** - need to add

**Proposed new flags:**
```python
class OrdinalDeltaTrainer:
    def __init__(
        self,
        base_model='catboost',
        n_trials=0,
        include_forecast=True,       # Existing
        include_lags=True,            # Existing
        include_market=True,          # ADD
        include_momentum=True,        # ADD - temp_*, intraday_*
        include_meteo=True,           # ADD - humidity_*, wind*, cloud*
        include_multi_horizon=True,   # ADD - fcst_multi_*
        include_station_city=True,    # ADD - station_city_*
        cv_splits=3,
        catboost_params=None,
        verbose=False,
    ):
```

**Implementation:**
- Update `get_feature_columns()` in base.py
- Filter NUMERIC_FEATURE_COLS based on flags
- Add CLI arguments: `--no-market`, `--no-momentum`, etc.

**Use cases:**
```bash
# Baseline (no multi-horizon, no momentum)
python scripts/train_city_ordinal_optuna.py --city austin --trials 100 \
  --no-multi-horizon --no-momentum

# Market-only ablation
python scripts/train_city_ordinal_optuna.py --city austin --trials 100 \
  --no-market

# Full feature set (default)
python scripts/train_city_ordinal_optuna.py --city austin --trials 100
```

## Issue 5: Additional Model Improvements

### 1. Early Stopping
**Current:** Trains to fixed iteration count from Optuna
**Proposed:** Add early stopping with patience
```python
catboost_params = {
    'early_stopping_rounds': 50,
    'use_best_model': True,
}
```
**Impact:** Faster training, less overfitting

### 2. Feature Interaction Depth
**Current:** Auto-determined by CatBoost
**Proposed:** Tune `max_ctr_complexity` for categorical×numeric interactions
```python
trial.suggest_int('max_ctr_complexity', 1, 4)
```

### 3. Regularization
**Current:** Only `l2_leaf_reg`
**Proposed:** Add `model_shrink_rate` and `model_shrink_mode`

### 4. Eval Metric Tuning
**Current:** Uses AUC (area under ROC curve)
**Consideration:** Switch to `custom_metric` that directly optimizes for MAE on delta predictions
**Pros:** More aligned with trading objective (minimize forecast error)
**Cons:** More complex, needs custom objective function

## Recommended Action Plan

### Priority 1: Quick Wins (Today)
1. ✅ Fix import bug in 03_train_ordinal.py
2. ✅ Add 5 missing useful features (city_warmer_flag, etc.)
3. Run Austin edge classifier training (Step 4)
4. Run Austin edge backtest (Step 5)

### Priority 2: Feature Flags (This Week)
1. Add include_market, include_momentum, include_meteo, include_multi_horizon flags
2. Test ablation: Train Austin without market features
3. Compare MAE: with vs without each feature group

### Priority 3: Optuna Expansion (Next Week)
1. Add grow_policy, max_leaves, rsm parameters
2. Retrain Austin with expanded search space (150+ trials)
3. Compare to current best (MAE 2.09°F)

### Priority 4: Scale to All Cities (Next 2 Weeks)
1. Train ordinal models for remaining 5 cities
2. Train edge classifiers for all 6 cities
3. Run backtests and generate reports

## Expected Performance Gains

| Improvement | Estimated MAE Reduction | Effort |
|-------------|------------------------|--------|
| Add 5 missing features | -0.02°F | 5 min |
| Expand Optuna parameters | -0.05°F to -0.10°F | 2 hours (150 trials) |
| Early stopping + regularization | -0.03°F | 30 min |
| Custom MAE-focused objective | -0.10°F to -0.15°F | 4 hours (complex) |
| **Total potential** | **-0.20°F to -0.30°F** | - |

**Target**: MAE from 2.09°F → **1.80-1.90°F**

## User Decision Needed

1. **Add 5 missing features now?** (city_warmer_flag, etc.)
2. **Run edge classifier training?** (Step 4 of pipeline)
3. **Expand Optuna parameters?** (grow_policy, max_leaves, etc.)
4. **Implement feature category flags?** (for ablation studies)

**My recommendation:**
- ✅ Fix import bug (1 min)
- ✅ Run edge training (Step 4) with current model
- ⏸️ Hold on additional optimizations until edge classifier works
- ✅ Then decide on further tuning based on end-to-end performance
