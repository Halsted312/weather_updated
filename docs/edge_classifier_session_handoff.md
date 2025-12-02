# Edge Classifier Training Session Handoff

**Date:** 2025-12-02
**Status:** In Progress - Plan Mode (awaiting implementation approval)
**Last Action:** Created refactoring plan, awaiting decision on threshold tuning approach

---

## Executive Summary

We were training edge classifier models for Kalshi weather trading. Training failed with "no edge generated" error. Root cause analysis revealed two bugs that were fixed. During investigation, we also identified a design issue with how `decision_threshold` is tuned (grid search vs Optuna). A refactoring plan was created but implementation is pending user decision.

---

## System Overview

### What This System Does

This is a **quantitative trading system for Kalshi weather markets**. It predicts daily high temperatures in 6 US cities and trades temperature bracket contracts on Kalshi.

**Cities:** Chicago (KMDW), Austin (KAUS), Denver (KDEN), Los Angeles (KLAX), Miami (KMIA), Philadelphia (KPHL)

### The Two-Model Architecture

1. **Ordinal Delta Classifier** (`models/training/ordinal_trainer.py`)
   - Predicts temperature delta (observed - forecast) as ordinal classes
   - Uses CatBoost with Optuna hyperparameter tuning
   - Trained per-city with ~100 Optuna trials
   - Output: Probability distribution over delta values (-10 to +10)

2. **Edge Classifier** (`models/edge/classifier.py`)
   - Binary classifier: "Is this trading edge real?" (will it win?)
   - Takes features like implied temperature, edge magnitude, market prices
   - Uses CatBoost with Optuna hyperparameter tuning
   - Output: Probability that a detected edge is profitable

### Data Flow

```
Visual Crossing API → wx.vc_* tables (observations, forecasts)
         ↓
Kalshi API/WebSocket → kalshi.* tables (candles, order book)
         ↓
Feature Pipeline → 18 feature modules compute ~50 features
         ↓
Ordinal Model → Temperature delta prediction
         ↓
Edge Detection → Identifies mispriced brackets
         ↓
Edge Classifier → Filters edges by win probability
         ↓
Live Trading → Executes filtered trades
```

---

## What We Were Doing

### Original Task
Train edge classifier for Austin with 100 Optuna trials:
```bash
python models/pipeline/04_train_edge_classifier.py --city austin --trials 100 --workers 12 --sample-rate 1 --optuna-metric filtered_precision
```

### Error Encountered
```
scripts.train_edge_classifier: no edge generated
```

---

## Root Cause Analysis

### Bug #1: Decimal Strike Parsing (FIXED)

**Location:** `scripts/train_edge_classifier.py` lines 162-192 and 297-335

**Problem:** Kalshi changed bracket ticker format in Oct 2024:
- Old format: `HIGHAUS-24MAY24-B86` (integer strikes)
- New format: `KXHIGHAUS-24OCT25-B86.5` (decimal strikes)

The code used `int(strike)` which failed on `"86.5"`:
```python
# BROKEN
strike_val = int(strike)  # ValueError on "86.5"
label = f"{strike_val}-{strike_val + 1}"  # Never reached
```

**Fix Applied:**
```python
# FIXED
strike_val = float(strike)
label = f"{strike_val:g}-{strike_val + 1:g}"  # "86.5-87.5"
```

This fix was applied to both `load_bracket_candles()` and `load_all_candles_batch()` functions.

### Bug #2: CatBoost Parameter Leak (FIXED)

**Location:** `models/edge/classifier.py` line 350-357

**Problem:** When `optimize_metric="filtered_precision"`, Optuna tunes `decision_threshold` along with CatBoost hyperparameters. But `decision_threshold` is not a CatBoost constructor parameter - it's used for prediction thresholding.

The code passed ALL `best_params` to CatBoost:
```python
# BROKEN
final_params = {
    "loss_function": "Logloss",
    **self.best_params,  # Includes decision_threshold!
}
self.model = CatBoostClassifier(**final_params)  # TypeError!
```

**Fix Applied:**
```python
# FIXED
catboost_params = {k: v for k, v in self.best_params.items() if k != "decision_threshold"}
final_params = {
    "loss_function": "Logloss",
    **catboost_params,
}
```

---

## Design Issue Discovered

### The Threshold Tuning Problem

There are currently **two different paths** for tuning `decision_threshold`:

| Mode | How Threshold is Tuned | Code Location |
|------|------------------------|---------------|
| `--optuna-metric auc` | Grid search (17 fixed points) AFTER Optuna | `_tune_decision_threshold()` lines 205-257 |
| `--optuna-metric filtered_precision` | Jointly with Optuna (TPE sampler) | `_create_optuna_objective()` line 171 |

**Why This Matters:**
- Grid search is less efficient than Optuna's TPE sampler
- Inconsistent behavior depending on which metric is chosen
- For `filtered_precision`, the optimal model hyperparameters depend on the threshold value, so they should be tuned together

### Decision Needed

**Option A: Always Use Optuna for Threshold (Recommended)**
- Remove grid search entirely
- Include `decision_threshold` in every Optuna trial
- Simpler code, one path, more efficient

**Option B: Keep Separate Paths**
- `auc` mode: model first, then grid search threshold
- `filtered_precision` mode: joint Optuna
- More complex but threshold only affects objective when relevant

---

## Current File State

### Files Modified (Fixes Applied)

| File | Change | Status |
|------|--------|--------|
| `scripts/train_edge_classifier.py` | `int(strike)` → `float(strike)` for decimal strikes | DONE |
| `models/edge/classifier.py` | Filter `decision_threshold` from CatBoost params | DONE |

### Files to Modify (Pending Approval)

| File | Planned Change |
|------|----------------|
| `models/edge/classifier.py` | Remove grid search, always use Optuna, enhance save/load |
| `scripts/train_edge_classifier.py` | Pass city/training_info to save(), update CLI |
| `models/pipeline/04_train_edge_classifier.py` | Update CLI to match |

---

## Key File Locations

### Edge Classifier Implementation
```
models/edge/
├── classifier.py          # EdgeClassifier class (main file)
├── detector.py            # Edge detection logic
├── implied_temp.py        # Temperature inference from prices
└── __init__.py            # Module exports
```

### Training Scripts
```
scripts/
├── train_edge_classifier.py           # Main training script (standalone)
└── ...

models/pipeline/
├── 04_train_edge_classifier.py        # Pipeline wrapper for training
└── ...
```

### Saved Models
```
models/saved/{city}/
├── edge_classifier.pkl                # CatBoost model + feature_cols
├── edge_classifier.json               # Metadata (currently minimal)
├── edge_training_data.parquet         # Cached training data
├── final_metrics_{city}.json          # Ordinal model metrics
└── ordinal_classifier.pkl             # Ordinal model
```

### Live Trading Integration
```
live_trading/
├── inference.py           # InferenceWrapper (loads edge classifier)
├── edge_trader.py         # Live trading execution
└── db/models.py           # TradingDecision schema
```

---

## Relevant Code Sections

### Where Threshold is Tuned (Current)

**Grid Search Path (for `auc` metric):**
```python
# models/edge/classifier.py lines 205-257
def _tune_decision_threshold(self, y_true, y_proba, min_trades=10):
    """Grid-search decision threshold on validation set."""
    grid = list(np.linspace(0.1, 0.9, 17))  # 17 fixed points
    for thr in grid:
        # ... find best precision
```

**Optuna Path (for `filtered_precision` metric):**
```python
# models/edge/classifier.py lines 169-171
if self.optimize_metric != "auc":
    trial_threshold = trial.suggest_float("decision_threshold", 0.1, 0.9)
```

### Where Models are Saved
```python
# models/edge/classifier.py lines 468-502
def save(self, path: str, train_metrics: dict = None):
    # Saves .pkl (model) and .json (metadata)
```

### Where Models are Loaded
```python
# models/edge/classifier.py lines 504-517
def load(self, path: str):
    # Loads from .pkl only, JSON not currently used
```

---

## Proposed JSON Metadata Structure

The refactoring plan includes enhanced metadata:

```json
{
  "city": "austin",
  "trained_at": "2025-12-02T11:30:00",
  "model_version": "1.0",

  "training_info": {
    "n_trials": 100,
    "n_train_samples": 5000,
    "n_val_samples": 1000,
    "n_test_samples": 1000,
    "date_range": {
      "min_date": "2024-01-01",
      "max_date": "2025-11-27"
    },
    "optimize_metric": "filtered_precision",
    "edge_threshold_f": 1.5
  },

  "best_params": {
    "depth": 6,
    "iterations": 189,
    "learning_rate": 0.058,
    "decision_threshold": 0.85,
    "..."
  },

  "test_metrics": {
    "auc": 0.973,
    "filtered_win_rate": 0.82,
    "n_trades_recommended": 450,
    "..."
  },

  "feature_cols": ["implied_temp", "edge_magnitude", "..."],
  "feature_importance": {"implied_temp": 0.25, "..."}
}
```

---

## Background Processes Running

Several forecast backfill processes were started in background:
- AUS, DEN, LAX, MIA, PHL historical forecast backfill
- These populate `wx.vc_forecast_daily` with lead_days 4,5,6 (previously only 0-3)
- Check status: `tail /tmp/forecast_backfill_all.log`

---

## Next Steps (When Resuming)

1. **Get Decision:** Option A (always Optuna) vs Option B (keep separate paths)?

2. **Implement Refactoring:**
   - Remove `_tune_decision_threshold()` grid search method
   - Modify `_create_optuna_objective()` to always include threshold
   - Update `save()` and `load()` methods for enhanced metadata
   - Update training scripts

3. **Test:**
   - Run 5-trial quick test for Austin
   - Verify JSON metadata saves correctly
   - Run full 100-trial training

4. **Train All Cities:**
   - After testing, train edge classifiers for all 6 cities

---

## Related Documentation

- Plan file: `~/.claude/plans/snug-launching-bumblebee.md`
- Project instructions: `CLAUDE.md` (project root)
- Datetime reference: `docs/permanent/DATETIME_AND_API_REFERENCE.md`
- File dictionary: `docs/permanent/FILE_DICTIONARY_GUIDE.md`

---

## Commands to Resume

**Check if bugs are fixed (quick test):**
```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/train_edge_classifier.py --city austin --trials 5 --workers 4 --sample-rate 50 --optuna-metric filtered_precision
```

**Full training after refactor:**
```bash
python models/pipeline/04_train_edge_classifier.py --city austin --trials 100 --workers 12 --sample-rate 1 --optuna-metric filtered_precision
```

**Check background backfill status:**
```bash
tail -50 /tmp/forecast_backfill_all.log
```

---

## Summary Table

| Item | Status |
|------|--------|
| Decimal strike bug | FIXED |
| CatBoost param leak bug | FIXED |
| Grid search removal | PLANNED (pending approval) |
| Enhanced metadata JSON | PLANNED (pending approval) |
| Austin edge classifier training | BLOCKED (awaiting refactor decision) |
| Forecast backfill (T-4,5,6) | RUNNING in background |
