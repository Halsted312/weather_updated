---
plan_id: optimize-catboost-within2-feature-groups
created: 2025-12-04
status: draft
priority: high
agent: kalshi-weather-quant
---

# Optimize CatBoost for Within-2 Accuracy + Feature Group Tuning

## Objective

1. Change Optuna objective from AUC → **within-2 accuracy** (aligned with trading goals)
2. Add early stopping (prevent overfitting)
3. Increase max iterations to 600
4. **Let Optuna tune which feature groups to include** (binary hyperparameters)

## Changes Needed

### 1. Custom Optuna Objective (Optimize Within-2 Rate)

**File**: `models/training/ordinal_trainer.py`

**Current** (line ~340): Uses AUC (area under ROC curve)
```python
def objective(trial):
    # ... train model ...
    auc = roc_auc_score(y_cv, y_pred_proba)
    return auc  # Maximize AUC
```

**New**: Optimize within-2 accuracy
```python
def objective(trial):
    # ... train model ...

    # Predict delta from probabilities
    delta_classes = np.array(self._delta_classes)
    y_pred_delta = np.array([
        delta_classes[np.argmax(probs)] for probs in y_pred_proba
    ])

    # Calculate within-2 rate
    within_2 = np.abs(y_cv - y_pred_delta) <= 2
    within_2_rate = np.mean(within_2)

    return within_2_rate  # Maximize within-2 accuracy
```

**Alternative**: Optimize within-1 rate (stricter)
**Recommendation**: Optimize within-2 (more stable, aligns with bracket width)

### 2. Add Early Stopping

**File**: `models/training/ordinal_trainer.py`

**Add to CatBoost params**:
```python
catboost_params = {
    'iterations': trial.suggest_int('iterations', 100, 600),  # Increased from 500
    'early_stopping_rounds': 50,  # Stop if no improvement for 50 rounds
    'use_best_model': True,       # Use best iteration, not last
    # ... other params ...
}
```

### 3. Add Feature Group Binary Tuning

**Files**:
- `models/training/base_trainer.py` (add parameters)
- `models/training/ordinal_trainer.py` (tune with Optuna)
- `models/features/base.py` (filter features)

**Step 3a: Add parameters to BaseTrainer**
```python
class BaseTrainer:
    def __init__(
        self,
        include_forecast=True,      # Existing
        include_lags=True,           # Existing
        include_market=True,         # NEW
        include_momentum=True,       # NEW - temp_*, intraday_*
        include_meteo=True,          # NEW - humidity_*, wind*, cloud*
        include_multi_horizon=True,  # NEW - fcst_multi_*
    ):
        self.include_market = include_market
        self.include_momentum = include_momentum
        self.include_meteo = include_meteo
        self.include_multi_horizon = include_multi_horizon
```

**Step 3b: Let Optuna tune feature groups**
```python
def objective(trial):
    # Let Optuna decide which feature groups to use
    include_market = trial.suggest_categorical('include_market', [True, False])
    include_momentum = trial.suggest_categorical('include_momentum', [True, False])
    include_meteo = trial.suggest_categorical('include_meteo', [True, False])

    # Create trainer with tuned feature groups
    trainer = OrdinalDeltaTrainer(
        base_model='catboost',
        include_market=include_market,
        include_momentum=include_momentum,
        include_meteo=include_meteo,
        include_multi_horizon=True,  # Always include (we built these!)
        cv_splits=self.cv_splits,
        verbose=False,
    )

    # ... train and evaluate ...
    return within_2_rate
```

**Step 3c: Update get_feature_columns()**
```python
def get_feature_columns(
    include_forecast=True,
    include_lags=True,
    include_market=True,
    include_momentum=True,
    include_meteo=True,
    include_multi_horizon=True,
):
    num_cols = NUMERIC_FEATURE_COLS.copy()

    # Filter market features
    if not include_market:
        num_cols = [c for c in num_cols if c not in MARKET_FEATURE_COLS]

    # Filter momentum features
    if not include_momentum:
        momentum_cols = [c for c in num_cols if 'temp_' in c and any(x in c for x in ['_last_', '_ema_', '_rate_', '_acceleration', '_volatility'])]
        num_cols = [c for c in num_cols if c not in momentum_cols]

    # Filter meteo features
    if not include_meteo:
        meteo_cols = [c for c in num_cols if any(x in c for x in ['humidity_', 'windspeed_', 'cloudcover_'])]
        num_cols = [c for c in num_cols if c not in meteo_cols]

    # Filter multi-horizon features
    if not include_multi_horizon:
        multi_cols = [c for c in num_cols if 'fcst_multi_' in c]
        num_cols = [c for c in num_cols if c not in multi_cols]

    # ... existing forecast and lag filtering ...

    return num_cols, cat_cols
```

### 4. Increased Iteration Range

**Current**: `suggest_int('iterations', 100, 500)`
**New**: `suggest_int('iterations', 100, 600)`

**With early stopping**: Most trials will stop before 600, preventing overfit

## Implementation Files

| File | Changes |
|------|---------|
| `models/pipeline/03_train_ordinal.py` | Line 14: Add `import os` |
| `models/training/base_trainer.py` | Add include_market, include_momentum, include_meteo, include_multi_horizon parameters |
| `models/training/ordinal_trainer.py` | - Change objective to within-2 rate<br>- Add early stopping<br>- Increase iterations to 600<br>- Add binary feature group tuning |
| `models/features/base.py` | Update get_feature_columns() to filter by flags |

## Expected Results

**With within-2 optimization:**
- Current: 72.1% within ±2°F
- Target: **75-80%** within ±2°F
- May sacrifice exact accuracy (25% → 22-23%) but improve practical performance

**With feature group tuning:**
- Optuna discovers which combinations work best
- Might find that momentum features hurt (overfitting) or help (signal)
- Market features might be redundant with forecast features

**With early stopping + 600 iterations:**
- Better generalization (less overfitting)
- Longer search space for complex patterns
- Most trials stop at 200-400 iterations (optimal point)

## Testing Strategy

**Quick test (10 trials):**
```bash
PYTHONPATH=. python3 scripts/train_city_ordinal_optuna.py \
  --city austin \
  --trials 10 \
  --cv-splits 5 \
  --workers 18 \
  --use-cached
```

**Expected**: Should see "Optimizing within_2_rate" in logs, early stopping messages

**Full run (150 trials with feature group tuning):**
```bash
PYTHONPATH=. nohup python3 scripts/train_city_ordinal_optuna.py \
  --city austin \
  --trials 150 \
  --cv-splits 5 \
  --workers 20 \
  --use-cached \
  > logs/austin_optimized_within2.log 2>&1 &
```

**Expected time**: ~15-20 minutes (110 trials took 8 min, 150 with early stopping ~15 min)

## User Approval

**Approved changes:**
1. ✅ Optimize for within-2 accuracy (not AUC)
2. ✅ Add early stopping (rounds=50)
3. ✅ Increase iterations to 600 max
4. ✅ CV=5
5. ✅ Let Optuna tune 2-3 feature groups (market, momentum, meteo)

**Proceed with implementation?**
