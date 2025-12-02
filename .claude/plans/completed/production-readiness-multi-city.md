---
plan_id: production-readiness-multi-city
created: 2025-11-30
status: in_progress
priority: high
agent: kalshi-weather-quant
---

# Production Readiness: Multi-City Hybrid Trading

## Objective

Extend the validated Chicago hybrid trading system to all 6 cities with proper models, dense candles, and production-ready configuration.

## Context

**Completed (from prior plans):**
- Market-clock TOD v1 global model trained and validated
- Hybrid mode implemented (D-1 market-clock, D TOD v1)
- Chicago backtest: 38 trades, $802 P&L, 34.2% win rate
- Chicago dense candles built: 1.2M rows
- Config enabled: `USE_HYBRID_ORDINAL_MODEL = True`

**Current State:**

| City | Sparse Candles | Dense Candles | TOD v1 Model | Status |
|------|----------------|---------------|--------------|--------|
| chicago | 1.98M | 1.23M | CatBoost | Ready |
| austin | 1.84M | 0 | constant* | Needs work |
| denver | 1.75M | 0 | constant* | Needs work |
| los_angeles | 1.82M | 0 | constant* | Needs work |
| miami | 1.57M | 0 | constant* | Needs work |
| philadelphia | 1.50M | 0 | constant* | Needs work |

*constant = degenerate classifier, not a real model

**Key Issue:**
Non-Chicago TOD v1 models are `sklearn.dummy.DummyClassifier` with strategy='constant', meaning they predict the same class for all inputs. Need to either:
1. Retrain proper TOD v1 models for each city, OR
2. Use market-clock global model for all predictions (simpler but less accurate on event day)

---

## Tasks

### Phase 1: Dense Candles for All Cities
- [x] Sparse candles backfilled for all 6 cities (12M+ rows total)
- [x] Dense candles built for chicago (1.2M rows, Aug-Nov 2025)
- [x] Dense candles partially built for denver (862K rows, Nov 2024 - Jan 2025)
- [ ] Complete dense candles for all cities (overnight script ready)

**Overnight Script Created:** `run_dense_overnight.sh`
- Processes 1 week at a time (memory-safe)
- ~4 hours estimated runtime
- Run with: `nohup bash run_dense_overnight.sh > /tmp/dense_all.log 2>&1 &`

### Phase 2: TOD v1 Model Decision
- [ ] Investigate why non-Chicago TOD v1 models are degenerate
- [ ] Check if training data exists for other cities
- [ ] Decision point:
  - Option A: Retrain TOD v1 for all cities (requires data)
  - Option B: Use market-clock only (global model for all predictions)
  - Option C: Hybrid with fallback (use market-clock when TOD v1 unavailable)

### Phase 3: Multi-City Backtest
- [ ] Run hybrid backtest for all 6 cities
- [ ] Compare performance across cities
- [ ] Identify any city-specific issues
- [ ] Document results

**Command:**
```bash
python scripts/backtest_hybrid_vs_tod_v1.py --cities chicago austin denver los_angeles miami philadelphia
```

### Phase 4: Production Configuration
- [ ] Set appropriate EV thresholds per city (if needed)
- [ ] Configure Kelly fractions
- [ ] Enable dry-run mode
- [ ] Run live trader for 1-2 days in observation mode
- [ ] Enable live trading with conservative limits

---

## Files to Modify

| File | Changes |
|------|---------|
| `config/live_trader_config.py` | Adjust city-specific parameters if needed |
| `models/inference/live_engine.py` | Add fallback logic if TOD v1 unavailable |

---

## Technical Details

### Dense Candle Build Time Estimate
- ~2-3 minutes per city for 3-month window
- Total: ~15 minutes for all 5 cities

### Model Fallback Logic (Option C)
```python
def _predict_hybrid(self, city, event_date, session, current_time):
    current_date = current_time.date()
    d_minus_1 = event_date - timedelta(days=1)

    if current_date == d_minus_1:
        # D-1: Use market-clock global model
        return self._predict_global(city, event_date, session, current_time)
    elif current_date == event_date:
        # Event day: Try TOD v1 first, fallback to market-clock
        if city in self.tod_v1_models and not self._is_degenerate(city):
            return self._predict_tod_v1(city, event_date, session, current_time)
        else:
            return self._predict_global(city, event_date, session, current_time)
```

---

## Completion Criteria

- [ ] Dense candles built for all 6 cities (test period)
- [ ] Model decision made and implemented
- [ ] Multi-city backtest run with documented results
- [ ] Dry-run validated for 1-2 days
- [ ] Production enabled with conservative limits

---

## Sign-off Log

### 2025-11-30 21:15 (Plan Created)
**Status**: Draft - ready to begin

**Summary of Prior Work:**
- debug-backtest-zero-trades: COMPLETE - fixed datetime comparison bug
- expand-kalshi-candles-schema: COMPLETE - schema has all OHLC columns
- kalshi-data-ingestion-verification: COMPLETE - 11.5M sparse candles exist
- market-clock-tod-v1: COMPLETE - hybrid system validated for Chicago

**Immediate Next Steps:**
1. Build dense candles for remaining 5 cities
2. Investigate TOD v1 model issue for non-Chicago cities
3. Run multi-city backtest

**Blockers**: None
