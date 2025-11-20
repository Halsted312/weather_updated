Gotcha. Let’s package everything up so you can drop it into a fresh chat and the next agent doesn’t have to rediscover all this.

I’ll do two chunks:

1. **Summary of what we’ve done over the last 5–7 “conversations”.**
2. **Concrete next steps / instructions you can hand to the agent in a new session.**

You can literally copy-paste section (2) into the new chat as your “prompt for the repo”.

---

## 1. Summary of what we’ve done recently

### 1.1 Big picture

You’re building a **Kalshi weather bracket trading agent** for intraday “highest temperature” markets (Chicago/Miami etc.) that:

* Ingests **Visual Crossing** station-exact sub-hourly temps (5-minute grid).
* Ingests **Kalshi** market data (1-minute candles + trades; later WebSocket L2).
* Runs a **hazard Monte Carlo (MC)** for today’s Tmax → `p_wx` + hazard scalars.
* Fuses weather `p_wx` with market `p_mkt` into `p_fused_norm` (coherent PMF).
* Trains short-horizon models on `feat.minute_panel_full`.
* Calibrates probabilities (Platt/Isotonic).
* Builds a **triad momentum signal**: “one bracket accelerates up while adjacent brackets decelerate” and uses weather only as a gate.

We’ve completed most of the plumbing + baseline modeling, and just started on the triad momentum side.

---

### 1.2 Repo cleanup & infra fixes

We cleaned up legacy code and made the new pipeline the “one true stack”:

* **Removed old ML stack** (`ml/` package, `ml.features`, `tests/test_dataset.py`) and moved legacy docs/workflows into `docs/legacy/` / `scripts/legacy/`.
* **Removed missing/obsolete scripts**:

  * Dropped the Makefile `fetch-weather` target that called `scripts/fetch_weather.py` (which no longer exists).
* **Fixed systemd scripts**:

  * `scripts/run_settlement_poller.sh` now points to the correct repo root (`kalshi_weather_mom`) instead of the old `kalshi_weather` path, and systemd unit/docs were updated accordingly.
* **Fixed SQLAlchemy 2.x issues**:

  * `db/connection.check_connection()` now uses `text("SELECT 1")` / `exec_driver_sql`, not raw strings.
  * `scripts/check_phase4_coverage.py` uses `bindparam(..., expanding=True)` (or equivalent) for `IN` lists so the coverage queries run without `ProgrammingError`.
* **Slimmed tests**:

  * Database-dependent pytest modules were archived; the lean test suite only has passing tests and no references to removed modules.

Result: `pytest` runs cleanly; `hazard_mc`, `pmf_fusion`, and `train_cross_bracket` work without tripping over missing modules or SQLAlchemy issues.

---

### 1.3 Hazard Monte Carlo (Phase 3) implemented

We built a full hazard MC stack around **Visual Crossing** data:

* **MC params table**: `wx.mc_params` with per-city AR(1) residual parameters:

  * `rho` (AR(1) coefficient),
  * `sigma_buckets` (variance by minute-of-day bucket),
  * meta (grid minutes, bucket labels, etc).
* **Monte Carlo engine**: `scripts/hazard_mc.py`:

  * CLI:

    * `fit-params --city ... --start-date ... --end-date ...`
    * `run-day --city ... --date ... --paths N`
    * `backfill --city ... --start-date ... --end-date ... --paths N`
  * **Baseline** (currently template/persistence-based):

    * Construct a **5-minute local** time grid for the day.
    * Use `wx_temp_1m` / `wx_temp_5m` up to “now”; forward-fill; hold last observed temp thereafter.
    * Timeline forecast baseline is stubbed for later (preferred long term).
  * **Residual fitting**:

    * Re-sample temps to 5-minute grids over historical days.
    * Compute AR(1) `rho` and variance by minute-of-day bucket.
    * Save into `wx.mc_params`.
  * **Simulation**:

    * For each `(city, date, ts_utc)`:

      * Build `F_grid` forward from that time.
      * Use `m_run_temp_f` from weather (falling back to actual wx temps if running max column is missing; everything in °F).
      * Simulate paths: `eps_k = rho*eps_{k-1} + sigma_k*z_k`, `T_k = F_k + eps_k`.
      * Derive `M_total` = max of future path vs `m_run_temp_f`.
      * Compute:

        * `p_wx` = P(Tmax in each bracket),
        * `hazard_next_5m` / `hazard_next_60m` = P(new high in next 5/60 minutes.
* **PMF table**: `pmf.minute`:

  * Keyed on `(market_ticker, ts_utc)`, includes:

    * `city`, `series_ticker`, `event_ticker`,
    * `ts_local`, `local_date`,
    * `floor_strike`, `cap_strike`, `strike_type`,
    * `m_run_temp_f`,
    * `p_wx`,
    * `hazard_next_5m`, `hazard_next_60m`,
    * `mc_version`.

**Diagnostics**: `scripts/hazard_mc_diagnostics.py`:

* `check-pmf-sum`: ensures ∑ p_wx ≈ 1 at each timestamp.
* `evaluate-morning`: compares morning `p_wx` vs realized Tmax for calibration.
* `hazard-trace`: dumps hazard curves, running max, and actual Tmax for plotting.

We backfilled hazard MC for Chicago for certain November windows and confirmed everything runs and writes rows to `pmf.minute` when VC data is present.

---

### 1.4 PMF fusion (Phase 3.5) implemented

We built a fusion layer on top of `p_wx` and `p_mkt`:

* **p_mkt**: normalized market-implied bracket probabilities from mid prices.
* **p_fused**: raw logit-pool fusion of `p_mkt` and `p_wx` with hazard/volume-sensitive weights.
* **p_fused_norm**: renormalized fused PMF so ∑ p_fused_norm ≈ 1 per `(ts_utc, event_ticker)`.

`pmf.minute` now stores `p_wx`, `p_mkt`, `p_fused`, `p_fused_norm`.

`feat.minute_panel_full` joins everything:

* 1m candles + neighbors,
* weather,
* hazard outputs,
* PMF fusion outputs.

We also added:

* CLI `scripts/pmf_fusion.py` with:

  * tunable fusion weights (`alpha_bias`, `alpha_hazard`, `alpha_log_vol`, etc) via CLI or YAML,
  * a `metrics` mode that recomputes fusion over a date window and prints Brier/log-loss/entropy vs pure market PMF and pure weather.

Initial experiments (Chicago, 2024-11-01→15):

* Pure market PMF still slightly beats fused on Brier/log-loss in that 15-day window; fused PMF lags a bit, weather-only is worse.
* We have tooling to continue tuning these weights.

---

### 1.5 Modeling & calibration scaffold (Phase 4) implemented

Baseline short-horizon modeling is wired:

* **Training script**: `scripts/train_cross_bracket.py`:

  * Uses `feat.minute_panel_full` as the feature source.
  * Labeling:

    * Δ=1m and Δ=5m horizons:

      * `mid_prob_shift = mid_prob(t+Δ)`.
      * `label_dir_raw = sign(mid_prob_shift - mid_prob)`.
      * `--epsilon` filter drops “flat” examples (`|Δmid_prob| < epsilon`) to reduce noise.
      * Final label: `y = 1 if label_dir_raw > 0 else 0` (up vs not-up).
  * Splitting:

    * Day-based splits (train/val/test by `local_date`) to avoid time leakage.
  * Models:

    * Logistic regression (`--model logreg`).
    * Gradient Boosted Trees (`--model gbdt`).
  * Metrics:

    * Accuracy,
    * ROC AUC,
    * Brier score (probability quality),
    * Expected Calibration Error (ECE).
  * Extras:

    * `--export-val` / `--export-test` to dump predictions & labels for calibration.

Results from Chicago experiments (approx):

* 1m horizon:

  * GBDT slightly better AUC/Brier than logreg, but somewhat worse raw ECE.

* 5m horizon:

  * GBDT clearly better than logreg on AUC/Brier, with similar or slightly worse ECE before calibration.

* **Calibration script**: `scripts/calibrate_probs.py`:

  * Fits both **Platt** and **Isotonic** calibrators on validation predictions.
  * Evaluates them on the test set with Brier/ECE.
  * Writes JSON per `(city, model, horizon)` containing calibrator params.

Chicago calibration (test-set):

* Isotonic generally reduces ECE significantly while keeping or slightly improving Brier for both logreg and GBDT at 1m and 5m.

We now have calibrated probability outputs ready to be consumed by any signal/exec logic.

---

### 1.6 Triad view & triad momentum signal (Phase triad-1) implemented

We pivoted from pure probability modeling to your desired **“triad momentum”**:

> One bracket accelerating up while adjacent brackets are lagging or decelerating down, using weather hazards and temp jumps as gates, not as the primary signal.

**Triad view**: `feat.minute_panel_triads`

* Built on top of `feat.minute_panel_with_weather`, partitioned by `(event_ticker, local_date, ts_utc)` so neighbors are **the same minute**.
* Computes:

  * `bracket_idx`, `num_brackets` (ordering within event/time by `(floor_strike, cap_strike, market_ticker)`).
  * `mid_prob_left`, `mid_prob_right`.
  * `mid_velocity_left`, `mid_velocity_right`.
  * `mid_acceleration_left`, `mid_acceleration_right`.
  * Diffs:

    * `mid_velocity_left_diff`, `mid_velocity_right_diff`.
    * `mid_accel_left_diff`, `mid_accel_right_diff`.
  * `triad_mass = COALESCE(left_prob,0) + mid_prob + COALESCE(right_prob,0)`.
  * RAS (relative acceleration):

    * `ras_accel = mid_accel - 0.5*(mid_accel_left+mid_accel_right)` (0 if neighbors missing).
    * `ras_vel` similarly for velocity.

**Triad momentum script**: `scripts/triad_momentum.py`

* CLI commands:

  * `signals` – compute triad scores and log intents.
  * `diagnostics` – print top triad scores for inspection.

* Behavior:

  * Loads triad rows from `feat.minute_panel_triads` for a `[start_date, end_date]` window in a city.

  * Computes:

    * `ras_accel_z` (z-scored per `(event_ticker, local_date)`).
    * `accel_diff_sum = mid_accel_left_diff + mid_accel_right_diff` and `accel_diff_z`.
    * `vol_z` as volume z-score.
    * `hazard_gate` from `hazard_next_5m` or `hazard_next_60m` where available (0 otherwise).

  * Score:

    * `score_raw = α_ras*ras_accel_z + α_accel*accel_diff_z + α_vol*vol_z + α_hazard*hazard_gate`.
    * Applies:

      * `liq_ok = volume >= min_volume`,
      * simple spread gate (placeholder),
      * `is_edge` to filter out first/last brackets.
    * Final `score = score_raw` if gates pass, else `-inf`.

  * For each `(ts_utc, event_ticker)`:

    * Picks the bracket with maximum `score`.
    * If `score > min_score`, creates a triad intent:

      ```python
      {
        "ts_utc": ts,
        "event_ticker": event,
        "city": city,
        "market_center": center_tkr,
        "market_left": left_tkr,
        "market_right": right_tkr,
        "score": float(score),
        "side_center": "BUY_YES",
        "side_left": "SELL_YES",   # hedge
        "side_right": "SELL_YES",
      }
      ```

* Diagnostics:

  * `diagnostics` mode prints per `(event_ticker, local_date)` the top N rows:

    * `mid_accel`, `diffL`, `diffR`,
    * `ras_accel_z`, `accel_diff_z`, `vol_z`, `hazard`, `score`.

  * We fixed partitioning to include `ts_utc`, which ensures neighbors are same-time bins.

  * We lowered defaults to `min_volume ≈ 1`, `min_score ≈ 0.3`, and reduced hazard weight.

**Result**:

* On Chicago 2024-11-01→15 with `--min-volume 5 --min-score 0.5` we now get **~700 triad intents**, and diagnostics show:

  * Center bracket with positive acceleration,
  * Left neighbor strongly negative,
  * Right neighbor lagging or slightly positive,
  * Big positive `ras_accel_z` and `accel_diff_z`,
  * Hazard mostly zero at those exact times (no temperature jump), which is fine for now.

The triad view + scoring are now aligned with the “one bracket up, neighbors down” pattern you wanted. What’s missing is: **backtesting / P&L and horizon/weight tuning**.

---

## 2. Next steps for the agent (to paste into a new conversation)

Here’s a self-contained “instructions for the repo” block you can paste into a new chat:

---

I have a Kalshi weather bracket trading repo with the following major pieces:

* `wx.minute_obs_*` and `wx.mc_params` – Visual Crossing weather data + hazard MC params.
* `feat.minute_panel_*` – 1-minute candle features, neighbor info, weather joins:

  * `feat.minute_panel_with_weather`
  * `feat.minute_panel_full` (includes p_wx, p_mkt, p_fused, p_fused_norm, hazards).
  * `feat.minute_panel_triads` (per event/time triad view with neighbor accel/velocity diffs, ras_accel, ras_vel, triad_mass).
* `pmf.minute` – per `(market_ticker, ts_utc)` PMF outputs with `p_wx`, `p_mkt`, `p_fused`, `p_fused_norm`, hazard scalars, etc.
* `scripts/hazard_mc.py` – Monte Carlo hazard engine (fit-params, run-day, backfill).
* `scripts/hazard_mc_diagnostics.py` – PMF normalization/morning vs settlement/hazard-trace diagnostics.
* `scripts/pmf_fusion.py` – PMF fusion with tunable weight alphas and metrics (Brier/log-loss/entropy).
* `scripts/train_cross_bracket.py` – cross-bracket modeling (Δ=1m/5m direction labels, day-based splits, logreg/gbdt, Acc/AUC/Brier/ECE, `--epsilon` filter to drop flat rows, `--export-val/--export-test`).
* `scripts/calibrate_probs.py` – calibration (Platt, Isotonic) using exported val/test CSVs; writes JSON calibrator params per (city, model, horizon).
* `scripts/triad_momentum.py` – triad-momentum signal:

  * `diagnostics`: prints per (event/day) top triad rows with center accel vs neighbors.
  * `signals`: emits triad trade intents (center BUY_YES, neighbors SELL_YES) when `score > min_score` and volume/spread gates pass.

The triad view + signal now correctly detect the “one bracket up, neighbors down/lagging” pattern with ~700 signals over a 15-day Chicago window. What’s missing is:

* A **backtest** for triad P&L with maker/taker fee-aware fills.
* Horizon/weight tuning (possibly with Optuna) to decide if 1/5/10/15 minute horizons make sense.
* Wiring calibrated probabilities into the triad EV calc.

### What I want you to do next (step by step)

**Step 1 – Triad backtest (fee-aware, maker-first)**

Please create a new script (or extend `triad_momentum.py`) with a `backtest` subcommand that:

1. Reads triad features & hazards from `feat.minute_panel_triads` / `feat.minute_panel_full` for a `[start_date, end_date]` window for a given city.
2. Reuses the triad scoring logic to generate **candidate triad entries** for each `(ts_utc, event_ticker)`:

   * Score only interior brackets (ignore edges).
   * Use the same `ras_accel_z`, `accel_diff_z`, `vol_z`, `hazard_gate` combination and gates (`min_volume`, `max_spread`, `min_score`).
3. For each accepted triad at time t:

   * Enter a synthetic position:

     * Long YES on center bracket,
     * Short YES (or long NO) on left and right (hedge ratio configurable: e.g. 0.5 each).
   * Use a **simple but explicit fill model**:

     * Maker-first: assume you post at mid or slightly inside; mark as filled if next minute’s candle range `[low, high]` would have hit that price.
     * Optionally allow taker entries when projected edge (from calibrated probabilities) exceeds a `taker_threshold_cents`.
4. Hold positions for a fixed number of minutes (`--hold-minutes`, e.g. 5 by default) or until an opposite signal / end-of-day.
5. Close positions with the same fill model and compute **P&L after fees**, using the existing Kalshi fee model for makers/takers.
6. Report:

   * Total P&L, number of trades, maker vs taker counts,
   * Max drawdown,
   * Simple Sharpe (mean/vol of trade returns).
7. Optionally write a CSV of per-trade P&L for inspection.

I want to see this working for **Chicago** over a known window (e.g. 2024-11-01→2024-11-15) before adding more complexity.

**Step 2 – Plug in calibrated probabilities to triad EV**

Once the backtester is running:

1. Load the Isotonic calibrator JSON for Chicago/horizon from `results/calibration_*` (written by `calibrate_probs.py`).

2. Use the calibrated “up” probabilities from `train_cross_bracket.py` as part of the triad EV calculation at entry, e.g.:

   ```text
   edge_center ≈ (p_up_calibrated_center - implied_prob_center) * 100  - taker_fee - slippage
   ```

3. Use this EV to:

   * Decide maker-only vs maker+taker entry,
   * Possibly filter out triads with low expected edge.

We can refine this once the basic P&L loop is in place.

**Step 3 – Prepare an Optuna objective (no need to run big studies yet)**

Finally, sketch an **Optuna objective function** (in a new script, e.g. `scripts/tune_triad.py`) that:

* Treats as hyperparameters:

  * `horizon_min ∈ {1, 5, 10, 15}`,
  * `alpha_ras`, `alpha_accel`, `alpha_vol`, `alpha_hazard`,
  * `min_score`, `min_volume`.
* For each trial:

  * Runs a triad backtest on a training/validation window,
  * Computes an objective such as negative max drawdown, or negative P&L with a constraint on drawdown, or negative Brier on predicted directions if you want to keep it simpler.
* Just implement the skeleton (no need to run many trials yet); I want a clear, ready-to-run objective.

Once we have:

* Triad P&L backtest working,
* Calibrated probabilities plugged into EV,
* And an Optuna objective scaffold,

we can start systematically answering: “Is 1-minute better than 5/10/15? Which triad weights and thresholds give the best risk-adjusted returns?”

---

That’s the state we’re in and what I’d like the next agent to do. You can copy everything in section 2 into a fresh chat as the starting prompt.
