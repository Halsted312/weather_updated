# Kalshi Weather Agent – Roadmap

## Completed Phases

### Phase 0 – Environment & Data Contracts
- Docker stack + .env scaffolding.
- Postgres schemas for markets, weather, and ingestion already in place via earlier migrations.

### Phase 1 – Minute Panel Base (DONE)
- `feat.minute_panel_base` materialized view with per-market 1m candles, mid probability, CLV, velocity/acceleration, volume deltas, local timestamps.

### Phase 2 – Neighbor & Weather Join (DONE)
- `feat.minute_panel_neighbors` adds cross-bracket deltas and triad mass using ordered strikes per event.
- `feat.minute_panel_with_weather` joins 1m Visual Crossing temps/humidity/dew/wind + running max.

### Phase 3 – Weather Hazard Monte Carlo (DONE)
- `wx.mc_params` stores AR(1) residual parameters per city.
- `pmf.minute` stores `p_wx`, hazard scalars, and metadata via `scripts/hazard_mc.py`.
- Diagnostics (`scripts/hazard_mc_diagnostics.py`) check PMF normalization, morning calibration, and hazard traces.

### Phase 3.5 – PMF Fusion (DONE)
- `pmf.minute` extended with `p_mkt`, `p_fused`, `p_fused_norm`.
- `feat.minute_panel_full` view exposes candles + weather + hazards + fused PMFs.
- `scripts/pmf_fusion.py` logit-pools p_mkt and p_wx with hazard/volume-aware weights and writes coherent PMFs.

### Phase 4 – Modeling Scaffold (IN PROGRESS)
- `scripts/train_cross_bracket.py` loads from `feat.minute_panel_full`, generates Δ=1m/5m direction labels (with epsilon filter), splits by day, and trains baseline (`logreg` or `gbdt`) while reporting Accuracy/AUC/Brier/ECE. Supports exporting validation probs for calibration.

## Remaining Phases / Workstreams

1. **Phase 4.T – Fusion & Model Tuning**
   - Parameterize fusion weights (sigmoid of hazard/volume features) for grid search.
   - Evaluate fusion via PMF metrics (Brier/log-loss on bracket PMF) per city/horizon.
   - Benchmark models (logreg vs GBDT, potentially others) for Δ=1m/5m, logging class balance.

2. **Phase 4.C – Probability Calibration**
   - Fit Platt and Isotonic calibrators using exported validation predictions per city/horizon.
   - Evaluate calibrated models on test set (Brier, ECE, reliability plots) and store calibrator params.

3. **Phase 5 – Signal & Execution Layer**
   - Build composite signal score (RAS/velocity/flow/hazard/p_fused) with liquidity gates.
   - Map signals to maker-first execution with fee-aware taker switch; enforce risk caps.

4. **Phase 6 – Backtest & Shadow**
   - Integrate calibrated model + fusion into backtester; simulate fills with maker/taker models.
   - Scenario slicing (city, hazard regime, time of day) and diagnose gaps.
   - Run shadow trading mode logging decisions/fills before flipping live.

5. **Phase 7 – Monitoring & Ops**
   - Metrics dashboards (Brier/ECE drift, hazard health, runtime), alerting, and periodic retraining cadence.

## Immediate Next Actions
1. Sweep fusion weight parameters and compare PMF metrics on validation days.
2. Train/evaluate models for Chicago (Δ=1m, Δ=5m) with both logreg and GBDT across metrics.
3. Implement calibration script to fit Platt/Isotonic on validation predictions and recompute test metrics.
