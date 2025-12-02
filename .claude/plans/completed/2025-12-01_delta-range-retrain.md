# Plan: Retrain with [-10,+10] Delta Range + Optuna Backtest

---
**STATUS**: PAUSED - Moving to new computer
**Last Updated**: 2025-12-01 ~02:00 CST
---

## Objective
1. Rebuild Chicago dataset with new delta range [-10, +10] (21 classes)
2. Retrain Market-Clock model with Optuna
3. Run Optuna-optimized backtest sweeping trading parameters

## ✅ COMPLETED CODE CHANGES

| File | Change | Status |
|------|--------|--------|
| `models/features/base.py` | `DELTA_CLASSES = list(range(-10, 11))` | ✅ Done |
| `models/features/partial_day.py` | `clip_range=(-10, 10)` default | ✅ Done |
| `models/inference/live_engine.py` | Delta range defaults updated | ✅ Done |
| `open_maker/prob_to_orders.py` | Docstrings + HorizonRiskConfig | ✅ Done |

## ⏳ PENDING TASKS (Run on new machine)

### Step 1: Build Small Dataset (~5 min)
```bash
.venv/bin/python scripts/build_market_clock_dataset.py \
    --mode custom --cities chicago \
    --start-date 2025-01-01 \
    --output data/market_clock_chicago_small.parquet
```

### Step 2: Train Model (~10 min)
```bash
.venv/bin/python scripts/train_market_clock_tod_v1.py \
    --input data/market_clock_chicago_small.parquet \
    --test-days 50 \
    --use-optuna --trials 20 \
    --output-dir models/saved/market_clock_small/
```

### Step 3: Run Backtest
```bash
.venv/bin/python scripts/backtest_ml_hybrid.py --city chicago --days 30
```

---

## SMALL TEST CASE (Fast Iteration)

Use smaller dataset for quick validation before full run.

| Parameter | Small Test | Full Run |
|-----------|------------|----------|
| Date range | 2025-01-01+ (~330 days) | 2023-12-31+ (~700 days) |
| Test days | 50 (~15%) | 30 (~4%) |
| Optuna trials | 30 | 150 |
| Est. dataset time | ~5 min | ~12 min |
| Est. training time | ~10-15 min | ~60 min |
| **Total** | **~20 min** | **~1.5 hours** |

---

## Current Session Tasks

### Step 1: Rebuild Chicago Dataset (Small)
```bash
.venv/bin/python scripts/build_market_clock_dataset.py \
    --mode custom --cities chicago \
    --start-date 2025-01-01 \
    --output data/market_clock_chicago_small.parquet
```
- New delta range: [-10, +10] (was [-2, +10])
- ~330 days of data (vs 700)

### Step 2: Retrain Market-Clock (30 Optuna trials)
```bash
.venv/bin/python scripts/train_market_clock_tod_v1.py \
    --input data/market_clock_chicago_small.parquet \
    --test-days 50 \
    --use-optuna --trials 30 \
    --output-dir models/saved/market_clock_small/
```

### Step 3: Optuna Backtest with Trading Parameter Sweep

**Parameters to optimize:**

| Parameter | Range | Steps | Purpose |
|-----------|-------|-------|---------|
| `min_edge_pct` | 2-16% | 1% | Minimum edge to trade |
| `min_prob` | 5-25% | 2% | Minimum model confidence |
| `edge_mult_scale` | 0.5-2.0 | 0.25 | Scale horizon edge multipliers |
| `size_mult_scale` | 0.5-2.0 | 0.25 | Scale horizon size multipliers |
| `max_positions` | 1-3 | 1 | Max simultaneous bracket bets |

**Objective function:** Maximize Sharpe ratio (or total P&L)

### Step 4: Compare Baseline vs Optimized
- Run backtest with default params (baseline)
- Run backtest with Optuna-optimized params
- Compare by time bucket (0-6h, 6-12h, 12h+)

---

## Files to Modify

| File | Changes |
|------|---------|
| `scripts/backtest_ml_hybrid.py` | Add Optuna integration for trading params |
| `open_maker/prob_to_orders.py` | Already has HorizonRiskConfig - add scale params |

---

## Code Changes Already Made

- `DELTA_CLASSES` → [-10, +10] in `models/features/base.py`
- `compute_delta_target()` → clip_range=(-10, 10) in `models/features/partial_day.py`
- Delta range defaults updated in `models/inference/live_engine.py`
- Docstrings updated in `open_maker/prob_to_orders.py`

---

## Key Insight

**Point accuracy (30.4%) is misleading for bracket trading!**
- Random baseline for 21 delta classes = 4.8% (was 7.7% for 13 classes)
- **Within-2 accuracy** is the key metric for 5°F bracket trading
- Trading strategy should size positions ∝ 1/hours_to_close
