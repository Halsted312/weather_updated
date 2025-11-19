

`codex_session_2025-11-19_2359.md`

It weaves together:

* The roadmap and phase breakdown 
* Your previous codex session summary 
* The next-steps modeling notes and enhancements we discussed 
* The agent README / stack layout 

It’s long, detailed, and explicitly tells a future agent what’s already done, what’s in progress, what’s risky, and what to do next.

---

````markdown
# Codex Session – 2025-11-19_2359  
**Project:** Kalshi Weather Bracket Agent (Chicago/Miami + multi-city)  
**Owner:** Stephen  
**Context Snapshot:** End of Phase 4 scaffolding – hazard MC, PMF fusion, and modeling baselines all wired; ready for fusion tuning, richer models, calibration, signal/exec, and backtest + shadow.

---

## 0. Big Picture

We’re building a **production-grade agent** to trade Kalshi **daily-high temperature brackets** (e.g., “Chicago high temp 81–82°F today”) using:

- **Visual Crossing** station-exact, sub-hourly data (5-minute timeline for Midway/Miami/etc.).
- **Kalshi** 1-minute candlesticks and trades (and later WebSocket L2 for microstructure).
- A **hazard-aware Monte Carlo** for today’s Tmax (p_wx + hazard scalars).
- A **PMF fusion layer** combining market-implied p_mkt and weather-implied p_wx.
- **Cross-bracket models** predicting short-horizon moves, with calibration.
- A **signal + execution layer** that is maker-first, fee-aware, and risk-bounded.
- A full **backtest + shadow + live** loop, all running on Docker + Postgres.

We’ve already built several phases of this stack in the repo. This document is the **current canonical state** plus **next actions**.

---

## 1. Current Stack Overview

### 1.1 Infrastructure

- **Docker-compose stack** (from README + AGENT_DESIGN): :contentReference[oaicite:4]{index=4}  
  - `postgres`: main DB (markets, weather, features, pmf, pnl).
  - `vc_ingest`: Visual Crossing Timeline poller (5-minute station data).
  - `candles_ingest`: Kalshi 1-minute candlesticks & optional trades.
  - `features`: builds `feat.minute_panel_*` views (kinematics, neighbors, weather, pmf).
  - `hazard_mc`: Monte Carlo hazard engine writing `pmf.minute`.
  - `pmf_fusion`: fusion engine writing `p_mkt`, `p_fused`, `p_fused_norm`.
  - `train_cross_bracket`: modeling & metrics.
  - `backtest` (to be finished): fill simulation, accounting & slicing.
  - `exec` (later): live/paper execution engine.
- **.env** defines:
  - `CITY`, `STATION_ID`, `VC_API_KEY`, `KALSHI_API_KEY`, `TRADE_MODE` (paper|live).
  - Database URL, MC path count, fusion thresholds, fee coefficients, etc.

### 1.2 DB Schemas (key objects)

**Markets & candles** (already in your DB):

- `markets`, `events`, `series`: Kalshi metadata (tickers, strikes, strike_type).
- `md.candles_1m`: 1-minute OHLCV per `market_ticker`.
- `md.trades`: tick-level trades per `market_ticker` (optional, but useful for fill sim).

**Weather:**

- `wx.minute_obs`: minute-level obs (or normalized VC sub-hourly) with `loc_id`, `temp_f`, `humidity`, `dew`, `wind`, etc.
- `wx.minute_obs_1m` / `wx.minute_obs_5m`: 1m and 5m grids, as views or tables.
- `wx.location`: maps `city` to `loc_id` and station metadata.
- `wx.mc_params`: MC residual parameters per city: `rho`, `sigma_buckets`, `baseline` meta (grid minutes, bucket labels). (Fitted by `fit-params`.)

**Features:**

- `feat.minute_panel_base`  
  - Per `market_ticker` 1-minute row: mid (close/100), CLV, velocity, acceleration, volume deltas, `ts_utc`, `ts_local`, `local_date`, `city`, etc. :contentReference[oaicite:5]{index=5}  
- `feat.minute_panel_neighbors`  
  - Adds neighbor deltas based on ordered strikes within each event (left/right bracket, triad mass, etc.).
- `feat.minute_panel_with_weather`  
  - Joins weather info: 1m temps/humidity/dew/wind, running max, plus bracket/city metadata.
- `feat.minute_panel_full`  
  - Joins `minute_panel_with_weather` with `pmf.minute` → includes `p_wx`, hazard scalars, `p_mkt`, `p_fused`, `p_fused_norm`, plus future labeled metrics for modeling. :contentReference[oaicite:6]{index=6}  

**Probabilities (PMF):**

- `pmf.minute` (long form)  
  Key: `(market_ticker, ts_utc)` plus:
  - `city`, `series_ticker`, `event_ticker`.
  - `ts_local`, `local_date`.
  - `floor_strike`, `cap_strike`, `strike_type`.
  - `m_run_temp_f` (running Tmax at ts).
  - `p_wx` (weather PMF component).
  - `p_mkt` (normalized market-implied probability from prices).
  - `p_fused` (raw logit-pool fusion).
  - `p_fused_norm` (renormalized fused PMF across brackets at each (ts_utc, event)).
  - `hazard_next_5m`, `hazard_next_60m`.
  - `mc_version` (e.g., `v1_grid5_paths4000`).

---

## 2. Completed Phases (0–4)

### Phase 0 – Environment & Data Contracts (DONE)

- Docker stack defined with `.env` + compose.  
- Core schemas for markets, MD candles/trades, weather, features, pmf.  
- README and AGENT_DESIGN established for stack usage.   

### Phase 1 – Minute Panel Base (DONE)

- `feat.minute_panel_base` built: per-market 1-minute panel:
  - `mid_prob = close_c / 100.0`.
  - `clv = (close - low)/(high - low)`.
  - `velocity`, `acceleration` via window functions.
  - Volume deltas, local timestamps, city/event metadata.
- Indexed on `(market_ticker, ts_utc)` and `(city, ts_utc)` for fast refresh. :contentReference[oaicite:8]{index=8}  

### Phase 2 – Neighbor & Weather Join (DONE)

- `feat.minute_panel_neighbors`:
  - Partitions by `event_ticker` and sorts by `(floor_strike, cap_strike)` to ensure physical bracket ordering.
  - Adds left/right bracket features and triad mass; edge bins have `NULL` neighbors.  
- `feat.minute_panel_with_weather`:
  - Joins Visual Crossing 1m/5m data via city → loc_id.
  - Adds `wx_temp_1m`, `wx_temp_5m`, humidity, dew, wind, and a **running max** column.
- Time alignment:
  - `ts_local` uses `dim_city.tz` (IANA); local day definition matches Kalshi’s daily event definition.   

### Phase 3 – Weather Hazard Monte Carlo (DONE)

**Core implementation:**

- `scripts/hazard_mc.py`:
  - CLI: `fit-params`, `run-day`, `backfill`.
  - **Baseline** (currently template-based):
    - Build a 5-minute local grid from midnight → end-of-day.
    - Use observed VC wx temps up to now (wx_temp_1m / wx_temp_5m), forward-fill, then hold last temp for future steps.
    - Timeline-based baseline is stubbed and preferred later; template acts as stable fallback.
  - **Residual params**:
    - `fit_residual_params` loops over a date range per city:
      - Resamples temps to 5-minute grid.
      - Computes diffs → AR(1) ρ.
      - Buckets lead-time variance by minute-of-day into labels like `"0-60"`, `"60-180"`, etc.
    - Saves `rho` + `sigma_buckets` in `wx.mc_params`.
  - **Monte Carlo**:
    - For each `(ts_utc, city)`:
      - Use baseline for times > ts_utc.
      - Use `wx_running_max` where present; fallback to actual temps if missing (always °F now, no more `close_c` fallback).
      - Map time indices to variance buckets via `_sigma_sequence_for_index`.
      - Simulate `N` paths:
        - `eps_k = rho * eps_{k-1} + sigma_k * z_k`.
        - `T_k = F_k + eps_k`.
      - Compute future max per path → combine with `m_run_temp_f` → full daily Tmax distribution.
      - Derive:
        - `p_wx` per bracket (bins: between/greater/less).
        - `hazard_next_5m`, `hazard_next_60m` = P(new high in those windows).
    - Writes MC results to `pmf.minute` with `mc_version`.

**Diagnostics:** `scripts/hazard_mc_diagnostics.py`

- `check-pmf-sum`: sum `p_wx` across brackets at each ts and report deviations from 1.0.
- `evaluate-morning`: average morning `p_wx` in [window_start, window_end] (local) vs actual daily Tmax from `wx.minute_obs` → writes CSV.
- `hazard-trace`: exports `hazard_5m`, `hazard_60m`, `m_run_temp_f`, and actual Tmax time to CSV for plotting hazard curves around the real daily high.   

### Phase 3.5 – PMF Fusion (DONE)

**Schema & views:**

- Alembic migrations added:
  - `p_mkt` and `p_fused` fields to `pmf.minute`, then `p_fused_norm` for normalized PMF.
- `feat.minute_panel_full` now joins:
  - base panel + neighbor features + weather + hazard + MC outputs + fusion outputs.

**Fusion engine: `scripts/pmf_fusion.py`**

- Reads from `feat.minute_panel_full`.
- For each `(ts_utc, event_ticker)`:
  - Renormalizes raw `mid_prob` across brackets → `p_mkt`.
  - Clamps `p_wx` into sane range; falls back to `p_mkt` if missing.
  - Computes weights `w_mkt`, `w_wx` (currently hazard/volume-aware heuristics).
  - Performs **logit-pool fusion**:
    - `logit_fused = w_mkt * log(p_mkt) + w_wx * log(p_wx)`.
    - `p_fused = softmax(logit_fused)`.
  - Stores both `p_fused` and `p_fused_norm` (ensuring ∑_j p_fused_norm_j ≈ 1 per (ts, event)).
- Upserts results into `pmf.minute` (idempotent).

### Phase 4 – Modeling Scaffold (DONE / REFINED)

**Baseline modeling pipeline: `scripts/train_cross_bracket.py`**

- Loads from `feat.minute_panel_full` which now contains:
  - Kinematics (`mid_prob`, `v`, `a`, `clv`, volume).
  - Neighbor deltas (RAS, triad mass).
  - Weather & hazard features.
  - `p_wx`, `p_mkt`, `p_fused`, `p_fused_norm`.
- Labeling:
  - `Δ = 1m` or `Δ = 5m`:
    - `mid_prob_shift = mid_prob(t+Δ)` per bracket.
    - `label_dir_raw = sign(mid_prob_shift − mid_prob)`.
    - `--epsilon` filter: drops rows where `|mid_prob_shift - mid_prob| < epsilon` to remove noisy flat examples.
    - Final label: `y = 1 if label_dir_raw > 0 else 0` (binary “up vs not up”).
- Splits:
  - By `local_date` (daily splits) into train/val/test (e.g., 60/20/20) to avoid lookahead and leakage.
- Models:
  - `--model logreg` → logistic regression baseline.
  - `--model gbdt` → gradient boosted trees baseline.
- Metrics:
  - Accuracy.
  - ROC AUC.
  - **Brier score** (probability quality).
  - **Expected Calibration Error (ECE)** (calibration).
- Additional functionality:
  - `--export-val` to dump validation set predictions (y_true, y_prob, features) as CSV/Parquet for calibration and reliability plots.

**Status:** Phase 4 is now **fully scaffolded and refined**; we can start tuning fusion weights, model hyperparameters, and calibration.

---

## 3. Roadmap – Remaining Phases (4.T onward)

This section merges the high-level roadmap from `ROADMAP.md`, the earlier codex session, and the “enhanced plan” notes.   

### Phase 4.T – Fusion Weight & Model Tuning

**Goal:** Make `p_fused_norm` and the model’s probabilities as **informative and calibrated** as possible.

**Tasks:**

1. **Parameterize fusion weights**:

   - Current weights are heuristic; we want a parametric form, e.g.:

     ```text
     s(t) = α0 + α1 * log(volume + 1) + α2 * hazard_next_60m + α3 * volatility
     w_mkt(t) = σ(s(t))
     w_wx(t)  = 1 - w_mkt(t)
     ```

   - Hyperparameters: `α0, α1, α2, α3` per city (or shared with city-specific offsets).

2. **Grid search / tuning**:

   - For each city and horizon (1m, 5m):

     - Sweep `α` values on a subset of days (validation set).
     - Evaluate using **PMF metrics**:
       - Brier score over bracket PMF per ts.
       - Log-loss on the realized bracket at event resolution (winner bracket).
       - ECE for bracket-level probabilities.

   - Update `pmf_fusion` to accept a config for `α` and write results to `p_fused_norm`.

3. **Model hyperparameter sweeps**:

   - For each city/horizon:

     - `logreg` with L2 penalty and different C (regularization strengths).
     - `gbdt` with:
       - depths 3–5,
       - estimators ~50–200,
       - learning rate ~0.05–0.2.

   - Compare by:
     - ROC AUC (ranking).
     - Brier/ECE (probability quality).
     - Class imbalance (positive vs negative counts).

4. **Result logging**:

   - Save each experiment’s metrics into a `model_runs` table or CSV with:
     - city, horizon, model, alpha config, metrics.

### Phase 4.C – Probability Calibration

**Goal:** Wrap classifier outputs in a **calibration layer** (per city/horizon).

**Tasks:**

1. **Calibration tooling:**

   - Use exported validation predictions from `train_cross_bracket.py` (via `--export-val`):

     - Fit **Platt scaling** (1D logistic regression) on `score → y`.  
     - Fit **Isotonic Regression** as a non-parametric monotone mapping.

   - Evaluate on **test set**:

     - Brier score.
     - ECE (10–20 bins).
     - Reliability curve.

2. **Store calibrators:**

   - Define `calibration_params` table:

     ```text
     city, horizon, model_type, method, params_blob, fitted_at
     ```

     - `method ∈ {platt, isotonic}`.
     - `params_blob` holds intercept/slope for Platt, or interpolation breakpoints for isotonic.

3. **Integration:**

   - Update `train_cross_bracket.py` or a separate `apply_calibration.py` to:

     - Load raw model probabilities for train/val/test.
     - Apply calibrator.
     - Recompute metrics.

4. **Decision on default calibrator:**

   - For each city/horizon, choose the method that provides:

     - Lower Brier & ECE.
     - No obvious overfitting in reliability curves.

   - Document chosen calibrator in this codex and in DB.

### Phase 5 – Signal & Execution Layer

**Goal:** Turn model + fused PMF into actual **trade signals** and executable orders.

**Design:**

1. **Signal score S_j(t):**

   Example form:

   ```text
   S_j(t) = α * normalized_RAS_j(t)
          + β * normalized_velocity_j(t)
          + γ * flow_percentile_j(t)
          + δ * hazard_score(t)
          + ε * (p_calibrated_j(t) - neighbor_avg(t))
````

* Where:

  * `normalized_RAS`: bracket j’s RAS scaled by rolling std.
  * `normalized_velocity`: scaled first derivative.
  * `flow_percentile`: percentile of recent volume or CLV.
  * `hazard_score`: monotone transform of hazard (e.g., logit of hazard_60m).
  * `p_calibrated`: calibrated probability from model + fusion (1m/5m horizon).

2. **Entry rules:**

   * Trade only when:

     * `|S_j(t)| > S_threshold`.
     * `p_fused_norm_j(t)` between ~5% and 95% (avoid tails dominated by fees).
     * Volume/spread gates pass:

       * Rolling 20m volume above `vol_min`.
       * Estimated spread < 3–4¢.
     * Risk budget available.

3. **Execution policy:**

   * **Maker by default:**

     * If predicted edge after maker fee ≥ `MAKER_THRESH_CENTS`, place limit at or just inside best bid/ask.

   * **Taker only when necessary:**

     * If predicted edge after taker fee + slippage ≥ `TAKER_THRESH_CENTS`, cross the spread.

4. **Risk management:**

   * Per-bracket `max_position`.
   * Per-city and global `MAX_DAILY_LOSS`.
   * Max gross exposure per day.
   * Time-based stops (e.g., auto-exit after 5 minutes or near market lock).

5. **Implementation:**

   * New module `signals/accel_signal.py` or similar:

     * Accepts features + probabilities.
     * Returns a list of trade intents with signal strength and desired execution style.

   * Execution engine `execution/engine.py`:

     * Converts intents to actual Kalshi orders (REST or later WebSocket trading).
     * Respects risk caps and TRADE_MODE (paper|live).

### Phase 6 – Backtest & Shadow

**Goal:** Evaluate the full strategy historically and in “shadow mode” before live.

**Backtest tasks:**

1. **Integrated backtest:**

   * Use `feat.minute_panel_full` + model outputs + calibrator + PMF fusion to generate signals for each minute.

2. **Fill models (multiple):**

   * Maker fill Model A:

     * Limit within [low_c, close_c] and assume fill fraction based on minute volume and prints.

   * Maker fill Model B:

     * More conservative; require trades at/through limit and CLV supportive.

   * Taker fill:

     * Fill at close_c + half-spread estimate (or at mid + half-spread; calibrate from occasional orderbook snapshots).

3. **P&L & metrics:**

   * Net P&L after fees.
   * Sharpe ratio.
   * Max drawdown.
   * AUC/Brier/ECE of probabilities.
   * Maker vs taker distribution (counts, fees).
   * Scenario slicing:

     * City, hazard regime, time-of-day, season.
     * Stress days with extreme weather or low liquidity.

**Shadow mode tasks:**

* Implement TRADE_MODE=`paper`:

  * Use the **live** feed (candles or L2) and **exact same** signal & execution logic.
  * Log intended orders and hypothetical fills using the same fill models.
  * No real orders sent to Kalshi.

* Decide go-live criteria:

  * e.g., 2–4 weeks of shadow P&L shows positive edge, stable calibration, and acceptable drawdown.

### Phase 7 – Monitoring & Ops

**Goal:** Make the system observable and maintainable.

**Tasks:**

1. **Metrics & dashboards:**

   * Daily P&L, drawdown, fee breakdown (maker vs taker).
   * Brier/ECE drift for probabilities.
   * Hazard MC health:

     * ∑p_wx stability.
     * Morning vs settlement calibration.

2. **Logging:**

   * Always log for each trade decision:

     * features snapshot.
     * p_mkt, p_wx, p_fused_norm.
     * hazard scalars.
     * model probabilities.
     * chosen signal S_j(t), execution type, and realized outcome.

3. **Retraining cadence:**

   * Weekly/monthly hazard refit (`fit-params`).
   * Monthly calibrator refit if Brier/ECE degrade.
   * Periodic model retrains with new data, versioned.

### Phase 8 – Live / WebSocket L2 Upgrade

**Goal:** Enhance microstructure and fills once the main strategy is validated.

**Tasks:**

1. **WS recorder:**

   * `kalshi_ws.py` to subscribe to `orderbook_delta` for all relevant brackets:

     * Receive `orderbook_snapshot` then deltas.
     * Maintain in-memory L2; persist snapshots/deltas to DB or log.

2. **Microstructure features:**

   * Order flow imbalance (OFI) at best levels.
   * Queue imbalance, cancellations, etc.
   * Hawkes intensities for aggressive trades (optional).

3. **Queue-aware fill model:**

   * Use L2 to simulate more realistic maker fills:

     * Your order’s position in the queue.
     * Trade-throughs at your level.

4. **Incremental rollout:**

   * Use L2 only to refine fills and triggers first; leave main structure alone.
   * Gradually incorporate microstructure features into the model once stable.

### Phase 9 – Multi-City / Multi-Series Scaling

**Goal:** Expand beyond one city/series with correct config separation.

**Tasks:**

* Parameterize everything by `(city, station_id, series_prefix)` in config.
* Per-city overrides:

  * hazard weights, volume gates, thresholds.
  * calibrators & models.
* Run multi-city backtests and evaluate whether cross-city models generalize or need per-city training.

---

## 4. Open Questions / Risks (as of now)

Pulled from the earlier codex + roadmap, updated for Phase 4 state.

1. **Fusion weights are still heuristic.**

   * Need grid search / learning of `α` in hazard/volume weighting.

2. **Calibration not yet deployed live.**

   * Baseline metrics show Brier/ECE, but no Platt/Isotonic wrappers integrated yet.
   * Without calibration, `gbdt` in particular may be miscalibrated.

3. **Hazard baseline uses template fallback.**

   * Timeline-forecast baseline is not yet implemented; on days with weird VC behavior, MC may mis-estimate tails.

4. **Class imbalance & rare events.**

   * Need to inspect “up” vs “not-up” ratios per city/horizon.
   * Might need class weights or alternative loss.

5. **Runtime scaling.**

   * Hazard MC currently runs per minute; with more cities and longer periods, daily runtime may grow.
   * Option: evaluate MC only on 5-minute grid or coarser and forward-fill.

---

## 5. Immediate Next Actions (for the next session)

Take these as **the next 3–6 hours of work** when you come back:

1. **Fusion tuning (lightweight first pass):**

   * For Chicago, Δ=1m and Δ=5m:

     * Define 2–3 candidate weight formulas (`w_mkt` as sigmoid of hazard + log volume).
     * Re-run `pmf_fusion.py` on a small validation window (e.g., last 30–60 days).
     * Compare Brier/log-loss across those configurations.

2. **Model experiments (Chicago first):**

   * Run `train_cross_bracket.py` for:

     * `--city chicago --horizon 1m --model logreg`
     * `--city chicago --horizon 1m --model gbdt`
     * `--city chicago --horizon 5m --model logreg`
     * `--city chicago --horizon 5m --model gbdt`
   * Capture metrics (Acc/AUC/Brier/ECE) and class balance.

3. **Calibration scaffolding:**

   * Implement a `scripts/calibrate_probs.py` that:

     * Takes validation exports (`--export-val` CSV/Parquet).
     * Fits Platt and Isotonic calibrators.
     * Writes out calibration parameters and test-set metrics.
   * Add a small table or YAML for storing calibrator params.

4. **Document results in this codex:**

   * After running the above, update this file (or a new codex_session_YYYY-MM-DD_HHMM) with:

     * Best model+calibrator per city/horizon.
     * Fusion config that looked best.
     * Any surprising behaviors (e.g., hazard helping early in day, hurting near close).

---

## 6. How to Use This File in the Next Conversation

When you start a new ChatGPT / coding-agent session:

1. Upload this file (and relevant code files if needed).
2. Tell the agent:

   * “This is my latest codex session context for the Kalshi weather bracket agent; please read it and then ask for clarifications if needed.”
3. Then issue a concrete task, e.g.:

   * “Implement the calibration script described in Phase 4.C.”
   * “Run the fusion tuning experiment for Chicago and summarize metrics.”
   * “Wire a new signal_*.py module that implements S_j(t) given RAS, hazard, and p_fused_norm.”

This codex should give the next agent everything it needs to **pick up without any ambiguity** and avoid re-deriving the context you’ve already painstakingly established.

---

```

::contentReference[oaicite:13]{index=13}
```
