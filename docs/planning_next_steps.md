### This is a generated planning document for the project described 

`docs/planning_next_steps.md`

---

# Planning – Next Steps & Implementation Checklist

**Purpose**

This document is a **checkpoint + QA guide** for the whole system:

* Verify that everything from the earlier planning docs has actually been implemented correctly.
* Ensure the data pipelines, models, strategies, and live-trading scripts are wired together coherently.
* Create a stable base so we can confidently add and validate new strategies (curve-based, ML-based, etc.).

This checklist assumes the agent has already:

* Loaded the full codebase into context.
* Read the permanent docs under `docs/permanent/` (especially the date/time & API docs).
* Read the earlier planning files:

  * `PLANNING_2025-11-26_144243.md`
  * `PLANNING_2025-11-26_170117.md` 
  * `PLANNING_2025-11-27_020512.md` 

---

## 0. How to Use This Document

1. Work **top–down**: start with infrastructure/data, then forecasts, then strategies, then live trading.
2. For each section, tick off the checkboxes only after:

   * You’ve located the relevant code;
   * You’ve verified it matches the spec here (and the older planning docs);
   * You’ve run at least one small sanity test (where applicable).
3. If something is inconsistent, **stop and fix** before moving to “next steps” strategies.

Use this as: **“Are we really ready to build / tune more strategies?”**

---

## 1. Core Data & Infrastructure

### 1.1 Database & Schemas

* [ ] Confirm the **Postgres / Timescale** instance used by all scripts is the same:

  * `DATABASE_URL` in `.env` (or config) matches what `src.db.get_db_session()` actually connects to.
  * There is no confusion between ports/containers (e.g., 5432 vs 5433 vs 5434).

* [ ] Verify core schemas and tables exist and match the expected models:

  **Weather schema (`wx`):**

  * [ ] `wx.location`
  * [ ] `wx.minute_obs`
  * [ ] `wx.settlement`
  * [ ] `wx.forecast_snapshot`
  * [ ] `wx.forecast_snapshot_hourly`

  **Kalshi / trading schema(s)** (names may vary by project version):

  * [ ] `kalshi.markets` (or `public.markets` / `kalshi_markets`) – market metadata
  * [ ] `kalshi.candles` or `kalshi.candles_1m` / `market_candles` – price history
  * [ ] Simulation tables: e.g., `sim.run`, `sim.trade`, `sim.live_orders` (or equivalent)

* [ ] Cross-check that the **SQLAlchemy models** in `src/db/models.py` match the actual table DDL:

  * Column names, types, PKs/unique constraints.
  * Especially for:

    * `WxSettlement`
    * `WxMinuteObs`
    * `WxForecastSnapshot`
    * `WxForecastSnapshotHourly`
    * Kalshi market/candle models.

### 1.2 Kalshi Market & Candle Data

From the earlier planning docs, we know there were a few critical issues and migrations around float strikes, Philly series, and candle formats. 

* [ ] **Float strike migration**:

  * All temperature strikes are stored as `FLOAT` (e.g., 43.5, 45.5), not integers.
  * Strike types:

    * `less` contracts have `floor_strike = NULL`, `cap_strike != NULL`.
    * `greater` contracts have `floor_strike != NULL`, `cap_strike = NULL`.
    * `between` contracts have both `floor_strike` and `cap_strike` non-null.

* [ ] **Philly series fix**:

  * Confirm the Philadelphia high-temp series ticker used in code matches reality (e.g., `KXHIGHPHIL`).
  * Backfill of Philly markets is complete for the historical window used in backtests.

* [ ] **Candles**:

  * Candle ingestion uses the correct nested price fields from Kalshi’s API (e.g. `yes_ask` / `yes_bid` dictionaries, not the old `price_*` nulls).
  * `bucket_start` / timestamp is correctly parsed and stored in UTC.
  * 1-minute candles (`candles_1m`) exist for all 6 cities and cover the backtest period (check row counts against expectations from the planning docs). 

---

## 2. Weather Data, Dates & Time Zones

This part must line up with your permanent *date/time & API* documentation.

### 2.1 Observed Data (Actuals)

* [ ] `wx.minute_obs`:

  * Minute-level temperature observations loaded for all 6 cities across the backtest period.
  * Time is stored as **UTC** (or documented clearly if using local with tz info), with a known mapping to local time via IANA time zones.

* [ ] `wx.settlement`:

  * Daily TMAX values (integer °F) for the relevant markets/dates.
  * For each city & event_date used in backtests, there is a settlement row.

* [ ] DST handling:

  * Minute observations at DST transitions do **not** cause broken queries (e.g. duplicate local hours).
  * Any DST-specific logic in ingestion scripts is reflected and tested (earlier bug around duplicate 1am local was fixed).

### 2.2 Forecast Data: Historical vs Current

Key design requirement: **Historical forecasts for backtesting** must be clearly separated from **current forecasts for live trading**.

* [ ] In `src/weather/visual_crossing.py`, confirm there are **three distinct methods** (or equivalent behavior):

  1. **Historical daily forecasts**

     * Uses Timeline API with `forecastBasisDate` (past basis date) and station IDs (`stn:ICAO`) or well-defined coordinates.
     * Only used by `scripts/ingest_vc_forecast_history.py`.
     * Writes to `wx.forecast_snapshot` with:

       * `city`
       * `target_date`
       * `basis_date`
       * `lead_days = target_date - basis_date`
       * `tempmax_fcst_f`, etc.

  2. **Historical hourly forecasts**

     * Same as above but with `include=hours,days`.
     * Used to populate `wx.forecast_snapshot_hourly` with hourly curves for backtests.

  3. **Current forecast**

     * Timeline API **without** `forecastBasisDate`.
     * Uses city names (`"Chicago,IL"` etc.) or lat/long.
     * Returns today + future days; used only for **live trading** and the **midnight snapshot daemon**.

* [ ] `scripts/ingest_vc_forecast_history.py`:

  * Only uses the historical forecast methods (1) + (2).
  * Backfills 2022–present for all 6 cities.
  * For each `(city, target_date)` you use in backtests, there is a forecast for at least one `basis_date` with `tempmax_fcst_f > 0`.

* [ ] `scripts/poll_vc_forecast_daemon.py`:

  * Uses the **current forecast** method only (no `forecastBasisDate`).
  * Once per night per city (around local midnight) it writes snapshot rows with:

    * `basis_date = today`
    * `target_date` for the next N days
    * `lead_days = target_date - basis_date`
  * Logs show non-zero `tempmax_fcst_f` for future dates (no `0.0F` bug).

---

## 3. Strategy Framework & Backtesting

The planning docs define a progression from a midnight-based strategy to open-maker strategies and variants. 

### 3.1 Strategy Registry & Core Runner

* [ ] `open_maker/strategies/`:

  * Contains at least:

    * `base.py` → `OpenMakerBase` (or similar)
    * `next_over.py` → exit / neighbor-bracket strategy
    * `curve_gap.py` → temperature-curve-driven bin shift strategy
  * There is a registry in `open_maker/strategies/__init__.py` mapping:

    * `"open_maker_base"` → (BaseStrategy, BaseParams)
    * `"open_maker_next_over"` → (NextOverStrategy, NextOverParams)
    * `"open_maker_curve_gap"` → (CurveGapStrategy, CurveGapParams)

* [ ] `open_maker/core.py`:

  * Implements a **single** `run_strategy(...)` (or equivalent) that:

    * Loads forecast, settlement, markets, and candles.
    * Constructs a `TradeContext` per trade (city, event_date, forecast, bracket, bet size, etc.).
    * Calls `strategy.decide(context, candles_df | obs_data)` to compute actions.
    * Computes P&L using a consistent fee model and settlement logic.
  * Old separate `run_backtest()` and `run_backtest_next_over()` functions:

    * Either removed or turned into thin wrappers that simply call `run_strategy` with the appropriate `strategy_id`.

* [ ] CLI:

  * `python -m open_maker.core --city chicago --days 30` runs the **base strategy** with defaults.
  * `--strategy` flag allows running multiple strategies and prints a comparison table:

    * e.g., `--strategy open_maker_base --strategy open_maker_curve_gap`.

### 3.2 Base Strategy – `open_maker_base`

**Concept:** Pure “maker-at-open” strategy. When next-day market opens (10:00 a.m. ET), pick the bracket matching the forecast (with bias and entry price), place a limit buy as a maker, then hold to settlement.

* [ ] Parameters:

  * `entry_price_cents` (e.g. 30c tuned)
  * `temp_bias_deg` (e.g. +1.1°F tuned)
  * `basis_offset_days` (typically 1 → use yesterday’s forecast of today)
  * `bet_amount_usd` (e.g. $100 in backtests; configurable for live)

* [ ] Behavior:

  * Uses `get_forecast_at_open` from historical `wx.forecast_snapshot`.
  * Adjusts forecast with bias: `T_adj = T_fcst + temp_bias_deg`.
  * Uses `find_bracket_for_temp(..., round_for_trading=True)` to determine the bracket.
  * Uses realistic **ask price** (e.g., `yes_ask_c`) for P&L simulation (not stale `close_c`).
  * Settlement uses `determine_winning_bracket` on TMAX from `wx.settlement`.

* [ ] Backtest:

  * Produces per-trade and per-day P&L, win rate, ROI, and Sharpe metrics:

    * `sharpe_per_trade`
    * `sharpe_daily`

### 3.3 Next-Over Strategy – `open_maker_next_over`

**Concept:** Start in the forecast bin at open; at a pre-defined intraday time (e.g., T-3h before predicted high), **optionally exit** if the next-higher bracket’s price moves strongly vs our bin.

* [ ] Parameters (check names in `NextOverParams`):

  * `entry_price_cents`
  * `temp_bias_deg`
  * `basis_offset_days`
  * `bet_amount_usd`
  * `decision_offset_min` (e.g. -180, -120 minutes before predicted high)
  * `neighbor_price_min_c` (min price for the next-over bracket to trigger exit)
  * `our_price_max_c` (max price for our current bracket to still be low enough)

* [ ] Behavior:

  * Uses **hourly forecast curves** to compute predicted high time.
  * Decision time = predicted high time + `decision_offset_min`.
  * Loads candle data for both our current bracket and the “next-over” bracket in a small window around decision time.
  * If conditions (neighbor price vs ours) are met, simulate exit at the appropriate price; otherwise hold.

* [ ] Optuna:

  * `open_maker/optuna_tuner.py` supports `--strategy open_maker_next_over` with a search space over the above parameters, optimizing `sharpe_daily` (not just total P&L).

### 3.4 Curve-Gap Strategy – `open_maker_curve_gap`

**Concept:** Use **observed vs forecast temperature gaps** plus slope near an estimated high time to decide whether to shift one or more bins.

* [ ] Inputs:

  * Hourly forecast curve (`wx.forecast_snapshot_hourly`).
  * Minute observations (`wx.minute_obs`) to compute:

    * `T_obs` at decision time (e.g., 15-min average).
    * `slope_1h` = trend over the last hour.

* [ ] Parameters (check `CurveGapParams`):

  * `entry_price_cents`
  * `temp_bias_deg`
  * `decision_offset_min` (e.g. -120)
  * `delta_obs_fcst_min_deg` (threshold for T_obs − T_fcst)
  * `slope_min_deg_per_hour` (minimum warming rate)
  * `max_shift_bins` (e.g. 1)

* [ ] Behavior:

  * At decision time:

    * Compute `delta = T_obs − T_fcst`.
    * Compute `slope_1h`.
  * If `delta` and `slope_1h` exceed thresholds:

    * Shift the assumed bin upwards by up to `max_shift_bins`.
    * For now, the implementation may be a **hindsight adjustment** (“what if we’d entered shifted bin”), or explicit simulated intraday exit/entry. Check which is implemented and document it.

* [ ] Optuna:

  * There is an objective for `open_maker_curve_gap` tuning:

    * Search over `delta_obs_fcst_min_deg`, `slope_min_deg_per_hour`, `decision_offset_min`, etc.
    * Optimize `sharpe_daily`, with train/test split.

---

## 4. Optimization & Parameter Storage

### 4.1 Optuna Tuning

* [ ] `open_maker/optuna_tuner.py`:

  * Supports:

    * `--strategy open_maker_base`
    * `--strategy open_maker_next_over`
    * `--strategy open_maker_curve_gap`
  * Uses time-based train/test split (e.g., 70/30): **no leakage across time**.
  * Default objective:

    * `total_pnl` for base strategy.
    * `sharpe_daily` for exit/curve strategies.

* [ ] Train/Test reporting:

  * Prints both train and test performance (win rate, ROI, Sharpe) to catch overfitting.
  * You’ve already seen realistic numbers (~74% win, ~Sharpe 2.0 daily) for base strategy – ensure those are reproducible.

### 4.2 JSON Export of Best Params

* [ ] Parameter saving:

  * `optuna_tuner.py` writes best params to:

    * `config/open_maker_base_best_params.json`
    * `config/open_maker_next_over_best_params.json`
    * `config/open_maker_curve_gap_best_params.json`

* [ ] Loading in code:

  * `core.py`, `live_trader.py`, `manual_trade.py` support `--use-tuned` flag that loads the correct JSON file for the chosen strategy.
  * JSON schema matches the relevant Params dataclass (no missing/extra keys).

---

## 5. Live Trading & Manual Tests

### 5.1 Manual Trade Script

* [ ] `open_maker/manual_trade.py`:

  * Can fetch markets directly from the Kalshi API for a given city + event_date.
  * Uses **the same bracket selection logic** as backtests (`find_bracket_for_temp`).
  * Forecast choices:

    * Use `--manual-fcst` when DB forecasts are missing.
    * Otherwise, use a flexible DB lookup for `wx.forecast_snapshot` with cheap, safe fallback.
  * Supports:

    * `--dry-run` mode (no orders submitted).
    * Live mode (submits orders via the Kalshi client).
  * Logs every order to a simulation or logging table (e.g., `sim.live_orders`) with a unique client order id.

* [ ] Confirm you can reproduce the earlier successful $5 Philly trade with:

  ```bash
  python -m open_maker.manual_trade \
      --city philadelphia \
      --event-date YYYY-MM-DD \
      --bet-amount 5 \
      --use-tuned
  ```

### 5.2 Live Trader Script

* [ ] `open_maker/live_trader.py`:

  * Subscribes to Kalshi WebSocket `market_lifecycle` events (or v2) to detect **market open** for the 6 weather series.
  * On market open for a given (city, event_date):

    * Uses `_get_live_forecast_for_event(...)` (or equivalent) to retrieve **current** forecast:

      * Prefer `wx.forecast_snapshot` snapshot from the midnight daemon.
      * Fallback to a direct **current forecast** call to Visual Crossing if DB is missing.
    * Applies `open_maker_base` strategy (only) for now with tuned params.
    * Places maker limit orders at the specified `entry_price_cents`.
  * Has:

    * `--dry-run` mode.
    * Logging to `sim.live_orders` (or similar) including event date, city, forecast, bracket, and price.

* [ ] Fill realism:

  * There is a “fill achievable” check, ensuring that the simulated entry price was actually available (or close) in the first couple of hours of trading; backtest uses this.

---

## 6. Ready for Next Strategies?

Once all sections above are **checked and sane**, we are ready to:

1. Add more complex **exit heuristics** on top of `next_over` and `curve_gap`.
2. Design and integrate **linear / elastic-net models** that take forecast curves, obs trends, and market microstructure features as input and output:

   * “Adjust bin up/down by k bins” or
   * “Scale bet size up/down” or
   * “Skip trade today.”

Before doing that, the agent should:

* Reread:

  * `docs/permanent/date_time_apis.md` (or equivalent),
  * `docs/permanent/strategy_design.md` (if present),
  * This `planning_next_steps.md` file **end-to-end**.
* Re-run a small end-to-end **backtest** + **manual trade dry-run** to confirm nothing regressed.

At that point, we can safely layer in:

* `open_maker_linear_model` (strategy 4),
* Additional ML models (CatBoost or similar) as a separate, well-tested module,
* And more sophisticated live position management.

---

### Further info on planning next steps below:
3.1 Phase 0 – Read and Map

Before any changes, have the refactor-planner agent:

Read:

FILE_DICTIONARY_GUIDE.md

docs/file_inventory.md 

file_inventory

docs/planning_next_steps.md

docs/permanent/* (dates & APIs, strategy overview)

Build an internal map of:

Largest Python files (already listed; e.g. open_maker/core.py, open_maker/utils.py, open_maker/optuna_tuner.py, live_trader.py, poll_vc_forecast_daemon.py). 

file_inventory

Which folder each belongs to and its logical role per FILE_DICTIONARY_GUIDE.md.

Do nothing else in Phase 0 except propose a refactor plan in a new doc (e.g., docs/refactor_plan_v1.md).

3.2 Phase 1 – Split open_maker/core.py

Goal: make core.py small, readable, and clearly separated between:

Backtest orchestration,

Data loading,

Result reporting.

Suggested layout:

open_maker/core_runner.py

run_strategy(...),

shared code to loop over cities/dates,

P&L aggregation.

open_maker/data_loading.py

Helpers for:

loading forecasts (historical only),

loading settlement,

loading markets/candles.

open_maker/reporting.py

Printing of run summaries / comparison tables,

Debug output.

Then:

Shrink open_maker/core.py to primarily:

CLI parsing,

delegating to core_runner.run_strategy,

selecting strategies based on --strategy.

Constraints for the agent:

Do not change strategy behavior while moving functions.

Update imports in all callers (optuna_tuner.py, tests, etc.).

Run a small backtest (e.g. --city chicago --days 30) after refactor to confirm results match pre-refactor.

3.3 Phase 2 – Split open_maker/utils.py

open_maker/utils.py (851 lines) is currently doing too many things. From your file dictionary + inventory, likely responsibilities include:

Bracket mapping,

Forecast selection,

Fee calculations,

Fill realism checks,

Misc helpers.

Proposed split:

open_maker/utils/brackets.py

find_bracket_for_temp

determine_winning_bracket

any bracket/ticker parsing helpers.

open_maker/utils/forecast.py

get_forecast_at_open

get_forecast_flexible

any forecast DB lookup logic (for backtests & manual trade).

open_maker/utils/fees.py

Fee calculations, maker vs taker,

Possibly constants like max fee rates.

open_maker/utils/fills.py

check_fill_achievable

Candle-based entry checks.

open_maker/utils/obs_curve.py

load_minute_obs

compute_obs_stats

get_forecast_temp_at_time

Then:

open_maker/utils/__init__.py can re-export the main functions for backward compatibility in the short term.

Again:

Move functions without changing behavior.

Update imports across strategies, core, manual_trade, live_trader, tests.

Run tests & small backtests to confirm.

3.4 Phase 3 – Slim scripts into thin CLIs

file_inventory.md shows that scripts/ has 13 Python files with 5265 lines total, some quite large (e.g., poll_vc_forecast_daemon.py, backfill_kalshi_candles.py). 

file_inventory

Refactor strategy:

Move “core logic” into src/ or open_maker/ modules:

e.g. src/ingest/forecast_history.py, src/ingest/settlement.py, src/ingest/kalshi_markets.py.

Leave scripts/*.py as thin wrappers:

parse CLI args,

call src.ingest.* functions.

This reduces duplication and makes it easier to invoke ingestion/backtests programmatically or from tests.

3.5 Phase 4 – Align tests with new structure

After splitting modules:

Update tests/ imports to use the new module layout.

Add tests to cover key helpers (especially bracket mapping, forecast selection, fee/fill logic).

Ensure pytest passes before and after each major phase.