---
plan_id: market-clock-tod-v1
created: 2025-11-30
status: draft
priority: high
agent: kalshi-weather-quant
---

# Market-Clock TOD v1 – All-Cities Global Model

## Objective

Implement a new ordinal CatBoost model that predicts daily high temperature using **market-clock time** (minutes since market open at D-1 10:00) across all 6 cities as a single global model.

## Context

**Why this model:**
- TOD v1 only sees same-day data (D 10:00-23:45) and misses the ~14 hours from market open
- Markets open at D-1 10:00 local, so traders need predictions starting then
- A global model pools data across cities for more robust learning

**Key differences from TOD v1:**
| Aspect | TOD v1 | Market-Clock TOD v1 |
|--------|--------|---------------------|
| Time window | D 10:00 - 23:45 | D-1 10:00 - D 23:55 |
| Training scope | Per-city | Global (all 6 cities) |
| City encoding | Categorical | One-hot numeric |
| Snapshots/day | 56 (15-min) | ~456 (5-min) |
| Time features | Calendar-based | Market-clock + calendar |

**Source document:** `docs/permanent/how-tos/updated_modeling_TOD.md`

---

## Tasks

### Phase 1: Dataset Builder
- [ ] Create `models/data/market_clock_dataset_builder.py`
- [ ] Implement `build_market_clock_snapshot_dataset()` function
- [ ] Implement `build_market_clock_snapshot_for_training()` helper
- [ ] Add `_generate_snapshot_times()` helper for D-1 10:00 to D 23:55

### Phase 2: Feature Engineering Updates
- [ ] Add `compute_quality_features_market_clock()` to `models/features/quality.py`
- [ ] Add `get_feature_columns_for_market_clock()` to `models/features/base.py`
- [ ] Implement `city_one_hot()` helper function

### Phase 3: Training Infrastructure
- [ ] Create `scripts/train_market_clock_tod_v1.py`
- [ ] Create output directory `models/saved/market_clock_tod_v1/`
- [ ] Create data directory `data/market_clock_tod_v1/`

### Phase 4: Configuration
- [ ] Add `market_clock_tod_v1` variant to `config/live_trader_config.py`

### Phase 5: Inference Integration
- [ ] Add `load_inference_data_market_clock()` to `models/data/loader.py`
- [ ] Add `get_market_open_time()` helper to `models/inference/live_engine.py`
- [ ] Add `build_market_clock_snapshot_for_inference()` helper
- [ ] Update `LiveInferenceEngine.predict()` for market-clock variant

### Phase 6: Testing & Validation
- [ ] Create `scripts/test_market_clock_tod_v1_inference.py`
- [ ] Run training and capture metrics
- [ ] Compare vs TOD v1 baseline
- [ ] Write results to `docs/MARKET_CLOCK_TOD_V1_RESULTS.md`

---

## Files to Create/Modify

| Action | Path | Notes |
|--------|------|-------|
| CREATE | `models/data/market_clock_dataset_builder.py` | Main dataset builder |
| CREATE | `scripts/train_market_clock_tod_v1.py` | Training script |
| CREATE | `scripts/test_market_clock_tod_v1_inference.py` | Integration tests |
| CREATE | `data/market_clock_tod_v1/` | Training data directory |
| CREATE | `models/saved/market_clock_tod_v1/` | Model artifacts |
| MODIFY | `models/features/quality.py` | Add `compute_quality_features_market_clock()` |
| MODIFY | `models/features/base.py` | Add `get_feature_columns_for_market_clock()` |
| MODIFY | `models/data/loader.py` | Add `load_inference_data_market_clock()` |
| MODIFY | `models/inference/live_engine.py` | Add market-clock prediction path |
| MODIFY | `config/live_trader_config.py` | Add `market_clock_tod_v1` variant |

---

## Technical Details

### Market-Clock Features (New)
```python
minutes_since_market_open = max(0, int((cutoff_time - market_open).total_seconds() // 60))
hours_since_market_open = minutes_since_market_open / 60.0
is_d_minus_1 = int(cutoff_time.date() == (event_date - timedelta(days=1)))
is_event_day = 1 - is_d_minus_1
```

### City One-Hot Encoding
```python
city_chicago, city_austin, city_denver, city_los_angeles, city_miami, city_philadelphia
# Exactly one is 1, others are 0
```

### CatBoost Configuration
```python
params = {
    "task_type": "CPU",
    "thread_count": 26,
    "loss_function": "MultiClass",
    "depth": 8,
    "learning_rate": 0.05,
    "iterations": 1000,
    "l2_leaf_reg": 5.0,
    "border_count": 128,
    "early_stopping_rounds": 100,
}
```

### Data Volume Estimate
- ~456 snapshots per event (38 hours × 12/hour at 5-min intervals)
- ~700 days of history
- 6 cities
- Total: ~1.9M training rows

### Model Config Entry
```python
"market_clock_tod_v1": {
    "folder_suffix": "_market_clock_tod_v1",
    "filename": "ordinal_catboost_market_clock_tod_v1.pkl",
    "snapshot_hours": None,
    "requires_snapping": False,
    "snapshot_interval_min": 5,
}
```

---

## Completion Criteria

- [ ] Dataset builder generates valid parquet with all features
- [ ] Training completes without errors on CPU (26 threads)
- [ ] Model achieves comparable or better metrics vs TOD v1
- [ ] Inference works at arbitrary timestamps (not just 5-min intervals)
- [ ] All existing TOD v1 / hourly / baseline paths remain unchanged
- [ ] Tests pass for multiple cities and date ranges

---

## Sign-off Log

### 2025-11-30
**Status**: Draft - awaiting user confirmation
**Notes**: Plan created from professor's specification in `docs/permanent/how-tos/updated_modeling_TOD.md`
