# Permanent doc: Strategy Overview & Next Steps
`STRATEGY_OVERVIEW_AND_NEXT_STEPS.md`

---

> **Note (Nov 2025):** Script names in this document reference the legacy VC ingestion pipeline.
> The current Phase 1 pipeline uses:
> - `scripts/ingest_vc_obs_backfill.py` (replaces `ingest_vc_minutes.py`)
> - `scripts/ingest_vc_forecast_snapshot.py` (replaces `poll_vc_forecast_daemon.py`)
> - `scripts/ingest_vc_historical_forecast.py` (replaces `ingest_vc_forecast_history.py` and `ingest_vc_forecast_hourly.py`)
>
> Legacy scripts are archived in `legacy/` folder for reference.

---

# Kalshi Weather Trading – Strategy & Data Flow Overview

This document is the high‑level reference for **how data flows** through the system and **how the main strategies work**. It’s meant for any coding agent (or human) who needs to understand:

* Where weather and market data come from
* How dates, times, and time zones are handled
* How backtests vs live trading differ
* What each strategy (`open_maker_*`) is actually doing
* What to implement or check next

It builds on the separate **Dates / Time / API** reference you already created; this doc is more about **strategy and data flow**.

---

## 1. Core Concepts & Entities

### 1.1 External Data Sources

**Weather / Observations**

* **NOAA / NWS settlement data** – daily high temperature (TMAX) for each city’s reference station, published as part of daily climate summaries. TMAX is the maximum observed temperature for the local calendar day and is effectively an integer Fahrenheit value when reported in the climate products we use.
* **Visual Crossing Timeline API** – One endpoint, multiple modes:

  * **Historical actuals** – past observations.
  * **Forecasts (current)** – 15‑day weather forecast from “today” into the future.
  * **Historical forecasts** – what the forecast looked like on a prior basis date using `forecastBasisDate`.

**Kalshi**

* **Markets & contracts** – weather series like `KXHIGHCHIC` (Chicago daily high) with per‑day “between / less / greater” brackets around integer temperatures.
* **Trading API** – REST endpoints for markets, candles, orders, positions, etc.
* **WebSockets** – streams for market data and lifecycle events, e.g. `market_lifecycle` and `market` type messages over `wss://api.elections.kalshi.com`.

---

### 1.2 Internal Database Schemas (conceptual)

In the Timescale/Postgres DB, we conceptually have:

**Weather schema (`wx`)**

* `wx.settlement`

  * One row per `(city, event_date)`
  * Columns:

    * `city` – enum / text key (`chicago`, `austin`, …)
    * `event_date` – **DATE, local calendar day of the station**
    * `tmax_final_f` – final daily high (F), integer
    * metadata (station id, source, insert_ts, etc.)

* `wx.minute_obs`

  * High‑frequency observed temps (and other fields) from Visual Crossing (or other sources).
  * Columns (conceptual):

    * `city`
    * `obs_time_utc` – `TIMESTAMPTZ` in UTC
    * `obs_time_local` – `TIMESTAMP` in city’s IANA timezone (optional)
    * `temp_f` – instantaneous temp in F
    * plus humidity, windspeed, precip, conditions, etc.

* `wx.forecast_snapshot` (daily forecasts)

  * One row per `(city, target_date, basis_date)`
  * Columns:

    * `city`
    * `target_date` – date being forecast
    * `basis_date` – date the forecast was issued
    * `lead_days = target_date - basis_date`
    * `tempmax_fcst_f`, `tempmin_fcst_f`, `precip_fcst_in`, etc.
    * `provider = 'visualcrossing'`
    * `created_at`

* `wx.forecast_snapshot_hourly` (hourly forecast curves)

  * One row per `(city, target_hour_local, basis_date)`
  * Columns:

    * `city`
    * `target_hour_local` – `TIMESTAMP` in **local time** of the city
    * `basis_date`
    * `lead_hours` / `target_hour_epoch`
    * `tz_name` – IANA timezone string (e.g. `America/Chicago`)
    * `temp_fcst_f`, `feelslike_fcst_f`, `humidity_fcst`, etc.

**Kalshi schema (public/sim)**

* `kalshi_markets` / `markets`

  * One row per contract (bracket).
  * Key fields: `series_ticker`, `event_ticker`, `ticker`, `strike_type` (`less`, `between`, `greater`), `floor_strike`, `cap_strike`, `event_date`, `open_ts`, `close_ts`, `strike_ts` (timestamps from Kalshi).

* `candles` / `market_candles`

  * 1‑minute OHLCV candles from Kalshi Event Candlesticks API:

    * nested `yes_bid.price`, `yes_ask.price`, and sizes; we’ve already fixed ingestion to read the **nested keys**, not flat `price_*`.

* `sim.live_orders`

  * Records our submitted live orders (city, market ticker, side, price, size, timestamp, order_id).

* `sim.run`, `sim.trade`

  * Used by backtest modules (`midnight_heuristic`, `open_maker`) to log simulated runs and trade‑level P&L.

---

## 2. Time & Time‑Zone Conventions (Summary)

You have a separate timestamps / API doc; this is the “strategy‑level” summary.

### 2.1 General Rules

* **Store absolute times in UTC** as `TIMESTAMPTZ` (`*_utc` columns, websocket timestamps, open_ts/close_ts).
* **Store event days as local dates**:

  * `event_date` always means “the calendar day in the weather station’s local timezone,” matching how NOAA / NWS define daily TMAX.
* **Store local hour timestamps** when reasoning about daily cycles:

  * `target_hour_local` in `wx.forecast_snapshot_hourly` is the local civil clock time; pair with `tz_name`.

### 2.2 Visual Crossing Times

* Visual Crossing **Timeline API** returns `days[i].datetime` and `hours[i].datetime` in **local time for the location by default**.
* For historical forecasts with `forecastBasisDate`, you still get `days[].datetime` as local **target_date**, but the content is “as if forecast on basis_date”.

### 2.3 Kalshi Times

* Kalshi API fields like `open_ts`, `close_ts`, `strike_ts` are ISO timestamps (documented as standard timestamps; treat them as UTC in our code and convert to local if needed).
* WebSocket `market_lifecycle` messages include the same timestamps / state transitions; we always convert to UTC internally and derive local `event_date` by converting to the station’s timezone when needed.

---

## 3. Weather Data Ingestion – What Each Script Should Do

### 3.1 Actual Observations (for labels & intraday features)

**Script:** `scripts/ingest_vc_minutes.py` (and/or NWS ingestion scripts)

**Goal:** populate `wx.minute_obs` and `wx.settlement`.

* Use Visual Crossing Timeline API (no `forecastBasisDate`) with a past date range and `include=minutes` (or `include=obs`) to pull historical minute‑level temps and store them as:

  * `obs_time_utc`
  * `temp_f`
  * other features (humidity, windspeed, precip, conditions).
* Daily settlement (`wx.settlement`) is derived from:

  * NWS / NOAA daily products (CF6 / Daily Climate data) or IEM, which give integer daily TMAX per station / city.

We never use “forecast” fields here—this is ground truth.

---

### 3.2 Historical Forecasts (for backtesting)

**Scripts:**

* `scripts/ingest_vc_forecast_history.py`
* `scripts/ingest_vc_forecast_hourly.py`

**Goal:** build a rich history of **what Visual Crossing predicted**, per basis date, for all target dates.

#### 3.2.1 Daily historical forecasts

Use a Visual Crossing helper like:

```python
fetch_historical_daily_forecast(location, basis_date, horizon_days)
```

* `location` – station code (`stn:KMDW`, `stn:KDEN`, …) or fixed lat/long.
* `basis_date` – the date you want the forecast to be “as of”.
* `horizon_days` – e.g. 7–15 days.

Under the hood this calls Timeline with:

* `startDate=basis_date`
* `endDate=basis_date + horizon_days - 1`
* `forecastBasisDate=basis_date`
* `include=days`

Then we:

* For each `day` in `payload['days']`:

  * `target_date = day['datetime']`
  * `lead_days = target_date - basis_date`
  * Upsert row into `wx.forecast_snapshot` with all forecast fields.

This gives us a matrix: for any `(city, target_date)` we can look at what VC thought at various `basis_date`s leading up to it.

#### 3.2.2 Hourly historical forecasts

Use:

```python
fetch_historical_hourly_forecast(location, basis_date, horizon_hours, horizon_days)
```

Same idea, but:

* `include=days,hours`
* We fill `wx.forecast_snapshot_hourly`:

  * for each hour, `target_hour_local`, `basis_date`, `lead_hours`, hourly temps, etc.

This supports:

* Predicted high time, not just high value.
* Curve‑based features for strategies like `curve_gap`.

---

### 3.3 Current / Live Forecasts (for trading today & next few days)

**Script:** `scripts/poll_vc_forecast_daemon.py`

**Goal:** once per night per city, capture the **current live forecast** for the next 7 days and store it so live trading can read it without hitting VC again.

Use a helper like:

```python
fetch_current_forecast(location="Chicago,IL", horizon_days=7)
```

* No `forecastBasisDate`.
* `location` is a city name or lat/long (station codes can behave differently for future days).

Then write to `wx.forecast_snapshot`:

* `basis_date = today`
* For each returned `day`:

  * `target_date = day['datetime']`
  * `lead_days = target_date - basis_date`
  * `tempmax_fcst_f` etc.

This is separate from historical forecast ingestion. Historical scripts are “offline;” the daemon is “online”.

---

## 4. Kalshi Data Ingestion – Markets, Candles, WebSockets

### 4.1 Market Metadata & Brackets

**Script:** `scripts/backfill_kalshi_markets.py`

* Use Kalshi REST `GET /markets` with `series_ticker` (e.g., `KXHIGHCHIC`) and `status` filters to:

  * Backfill all historical weather contracts into `kalshi_markets`.
* For each contract we store:

  * `event_date` (local day the contract settles),
  * `ticker` (full market ticker),
  * `strike_type` (`less`, `between`, `greater`),
  * `floor_strike`, `cap_strike` (Fahrenheit strikes),
  * `open_ts`, `close_ts`, `strike_ts`, etc.

The code already knows how to parse these fields; just keep them consistent.

### 4.2 Candles (intraday prices)

**Script:** `scripts/backfill_kalshi_candles.py`

* Uses Kalshi **Event Candlesticks** endpoint to fetch 1‑minute OHLCV for each market.
* Correctly reads nested fields:

  * `yes_ask.price`, `yes_bid.price`, etc. (Bug already fixed.)
* Stores into `candles` or `market_candles` with:

  * `ts_utc` (minute start),
  * `yes_bid_c`, `yes_ask_c`, etc. (in cents),
  * `volume`, etc.

These are used for:

* Entry price realism (was there actually a 30c ask near open?).
* Intraday strategies (e.g., `next_over`, eventual exit heuristics).

### 4.3 WebSockets (detecting market opens & live signals)

**Scripts:**

* `scripts/kalshi_ws_recorder.py`
* `open_maker/market_open_listener.py`
* `open_maker/live_trader.py`

We subscribe to Kalshi WebSockets at `wss://api.elections.kalshi.com/v2/ws` (per docs) and listen to:

* `market_lifecycle` stream:

  * Messages of type `"market_lifecycle"` giving lifecycle events: market created, opened, struck, closed.
  * This is how we know a new daily weather market is live (we *don’t* hard‑code “10am ET” – we react to lifecycle).

* `market` / `order` / `account` streams (optional) for monitoring quotes and positions in real‑time.

Live trader uses these events to:

* Detect when an event’s `status` transitions to `"open"` and `open_ts <= now < close_ts`.
* Immediately run the strategy logic and submit orders via REST.

---

## 5. Bracket Mapping & Settlement Logic

These utilities live in `open_maker/utils.py`.

### 5.1 find_bracket_for_temp (trading mode)

Inputs:

* `markets_df` for one `(series, event_date)`
* `event_date`
* `temp` (float degrees F, forecast + bias)

Behavior:

1. **Round temp to nearest integer** (because NOAA settlement is integer).
2. For `strike_type='between'`:

   * treat `floor_strike` ≤ T ≤ `cap_strike` as in‑range.
3. For `less`/`greater`:

   * `less` contracts handle lower tail (“≤ X° or below”)
   * `greater` contracts handle upper tail (“≥ Y° or above”).
4. Return `(ticker, floor_strike, cap_strike)` of the matching bracket.

This is used by all strategies for both:

* choosing brackets at entry (with forecast temps),
* simulating which bracket would win for a hypothetical temp (e.g., `curve_gap`).

### 5.2 determine_winning_bracket (settlement mode)

Inputs:

* `markets_df`,
* `event_date`,
* `tmax_final` (integer TMAX from `wx.settlement`)

Behavior:

* Calls `find_bracket_for_temp(...)` with `tmax_final` (effectively rounding no‑op).
* Returns the winning ticker; used for backtest P&L labeling.

No “nearest bracket” hacks; the logic matches NOAA integer semantics and Kalshi subtitles.

---

## 6. Strategy Overview – Names & Intuition

We currently think in terms of **four main strategy families**. Each strategy sits on top of the same data flow described above.

### 6.1 Strategy 1 – `open_maker_base`

**File(s):** `open_maker/core.py`, `open_maker/strategies/base.py`

**Idea:** “VC says today’s high is X°F. Take a maker position in the matching bracket right when the market opens and hold to settlement.”

**Key steps per `(city, event_date)`**

1. **Forecast lookup**

   * At market open, get `tempmax_fcst_f` from `wx.forecast_snapshot` using:

     * `target_date = event_date`
     * `basis_date = event_date - basis_offset_days` (usually 1).
   * Apply tuned bias: `temp_adj = temp_fcst + temp_bias_deg`.

2. **Bracket selection**

   * Use `find_bracket_for_temp` with `temp_adj` to find bracket.

3. **Entry**

   * Determine `entry_price_cents` (tuned by Optuna, often ~30c–45c).
   * Compute max size given `bet_amount_usd` (e.g., $100 or $20 per trade).
   * Place **maker YES** limit order at `entry_price_cents`.

4. **Exit**

   * No intraday exit; hold until settlement.
   * P&L = (contract pays $1 if bracket wins, else $0) minus fees, but note weather contracts may have zero maker fee on Kalshi; we still maintain the fee infrastructure.

**Tunable parameters (via Optuna):**

* `entry_price_cents` (30c–50c),
* `temp_bias_deg` (e.g. −2°F to +2°F),
* `basis_offset_days` (0 or 1).

This is the clean benchmark; everything else is relative to this.

---

### 6.2 Strategy 2 – `open_maker_next_over`

**File(s):** `open_maker/strategies/next_over.py`, `open_maker/core.py`

**Idea:** “Start in the forecast bracket. If the market’s order book later strongly prefers the next hotter bracket, switch to that one just before the high.”

**Key concepts:**

* Use forecast hourly curve to estimate **predicted high hour**.
* Choose a **decision time** (e.g., 2–5 hours before predicted high).
* Compare prices between:

  * our current bracket, and
  * the “next over” bracket (one bin above ours).

**Parameters (example):**

* `entry_price_cents` – same as base.
* `temp_bias_deg` – same concept as base.
* `basis_offset_days` – usually 1.
* `decision_offset_min` – e.g. −180 means 3h before predicted high.
* `neighbor_price_min_c` – minimum YES price on the higher bracket to consider switching (e.g. 50c).
* `our_price_max_c` – if our current bracket is trading below some threshold (e.g. 30c), it suggests market thinks we’re low; we switch.

**Flow:**

1. Enter like base at open.

2. At decision time `t_decision` (computed from `wx.forecast_snapshot_hourly` predicted high time and offset), load 1‑min candles around `t_decision` for both brackets.

3. If:

   * `neighbor_yes_price >= neighbor_price_min_c` and
   * `our_yes_price <= our_price_max_c`,

   then we **exit current bracket** and **enter the next over bracket** at that time.

4. Hold the new position to settlement and compute P&L.

This strategy tries to exploit information in the intraday order book that suggests the “true” high will be hotter than VC’s baseline.

---

### 6.3 Strategy 3 – `open_maker_curve_gap`

**File(s):** `open_maker/strategies/curve_gap.py`, `open_maker/utils.py`, `open_maker/core.py`

**Idea:** “If the observed temperature curve is running hotter than the forecast curve before the high, shift our bracket up by one (or more) bins.”

**Inputs at decision time:**

* `T_fcst` – forecast temperature at decision time from `wx.forecast_snapshot_hourly` (`get_forecast_temp_at_time`).
* `T_obs` – observed temperature from `wx.minute_obs`, aggregated (e.g. 15‑min average) via `load_minute_obs`.
* `slope_1h` – trend of observed temp over the last hour (`compute_obs_stats`).

**Key parameters:**

* `decision_offset_min` – minutes relative to predicted high (e.g. −120).
* `delta_obs_fcst_min_deg` – threshold for `T_obs − T_fcst` (e.g. +2.25°F).
* `slope_min_deg_per_hour` – threshold for temp slope (e.g. 0.5°F/h).
* `max_shift_bins` – how many brackets we can move up (usually 1).

**V1 behavior (what you have):**

* Start in base bracket at open.
* At decision time:

  * If `T_obs − T_fcst >= delta_threshold` **and**

    * `slope_1h >= slope_threshold`
  * then “shift” to the next one (or up to `max_shift_bins`).
* In current implementation this is handled as a **conceptual shift** affecting P&L (we simulate as if we had chosen the hotter bracket). The logging you pasted shows real shifts being triggered in LA, Miami, etc.

Later you can implement true intraday exits/entries that replicate this behavior with real orders.

---

### 6.4 Strategy 4 – `open_maker_linear_model` (planned)

**File(s):** planned: `open_maker/strategies/linear_model.py`, training script under `ml/` or `backtest/`.

**Idea:** “Use a simple statistical model trained on historical data to estimate either:

* the error `forecast_error = TMAX − T_fcst`, or
* the probability of each bracket,

and use that to adjust or filter base entries.”

**Candidate features:**

* **Forecast features**

  * `tempmax_fcst_f` for event_date at various basis_dates (yesterday, 2 days ago, 3 days ago),
  * trend across basis dates (is forecast drifting up or down?),
  * day‑of‑week, month, seasonality.

* **Curve features** (if using near‑real‑time obs)

  * `delta_obs_fcst` at decision time,
  * `slope_1h`,
  * humidity, windspeed, conditions category (e.g. clear/overcast/rain/snow).

* **Outcome / label**

  * integer `tmax_final_f`
  * or bracket index (0 .. N‑1),
  * or classification: { “too cold”, “ok”, “too hot” compared to VC bracket }.

**Model types (lightweight to heavy):**

* Start with **linear / elastic net** regression or logistic regression:

  * stable, interpretable, cheap to train.
* Later consider **gradient boosting (e.g. CatBoost / XGBoost)** for non‑linearities if needed; requires careful validation and more compute.

**Strategy behavior:**

* At open, compute model’s suggested **degree adjustment** (ΔT) or bracket index:

  * e.g. `temp_adj_model = temp_fcst + ΔT_model`.
* Apply the same bracket selection flow as base, but using `temp_adj_model`.
* Alternatively, use model as a **filter**:

  * only trade when predicted edge (P(win) – implied prob) exceeds threshold.

This is your “Strategy 4”; it sits on top of the same ingestion pipelines.

---

## 7. Backtesting, Tuning, and Data Leakage Precautions

**Backtesting code:** `open_maker/core.py`, `open_maker/optuna_tuner.py`, legacy `backtest/midnight_heuristic.py`

Key principles:

1. **Time‑based train/test split**

   * Always split on `event_date` (e.g. older 70% for train, recent 30% for test).
   * Your current Optuna tuner already does this.

2. **Use historical forecasts only**

   * Backtests must read from `wx.forecast_snapshot*` populated via `ingest_vc_forecast_history.py` (historical basis dates), not from the live daemon or current VC calls.

3. **No look‑ahead in features**

   * When computing features for event_date `D` using basis_date `B`, ensure all inputs come from `≤ B` (no inspection of later forecasts or observations).

4. **Hyperparameter tuning with Optuna**

   * `optuna_tuner.py` defines objectives for:

     * `open_maker_base`,
     * `open_maker_next_over`,
     * `open_maker_curve_gap`.
   * Metrics:

     * `total_pnl` for base,
     * `sharpe_daily` for exit/adjustment strategies (`next_over`, `curve_gap`).

5. **Saving best params**

   * Best parameter sets are exported to JSON:

     * `config/open_maker_base_best_params.json`, etc.
   * The live code (`live_trader.py`, `manual_trade.py`) can load them via `--use-tuned`.

---

## 8. Live Trading – Today’s Flow

For live trading (small stakes, e.g. $5–20 per city):

1. **Before 10am ET / market open window**

   * Ensure `poll_vc_forecast_daemon.py` has run overnight and populated `wx.forecast_snapshot` with `basis_date = today-1` (or `today`) for the upcoming event_date.
   * Alternatively, run a small “current forecast ingest” script that calls `fetch_current_forecast("City,State")` and writes `basis_date = today` into `wx.forecast_snapshot` for `lead_days` 0–3.

2. **At market open (detected by WebSocket `market_lifecycle`)**

   * `live_trader.py`:

     * Finds event market for each city / event_date from API or DB.
     * Calls `_get_live_forecast_for_event`:

       * First tries DB for `(city, target_date=event_date, basis_date in {today-1, today})` with `tempmax_fcst_f > 0`,
       * If missing, calls VC `fetch_current_forecast("City,State")` directly for that date.
     * Applies tuned base parameters (e.g., `entry=30c`, `bias=+1.1°F`).
     * Uses `find_bracket_for_temp` to pick bracket.
     * Submits maker order via REST.
     * Logs to `sim.live_orders`.

3. **Manual / test mode**

   * `open_maker/manual_trade.py` is your “single‑shot” script:

     * `--dry-run` to print what it *would* trade,
     * `--manual-fcst` to override when DB is incomplete,
     * `--use-tuned` to pull the best params.

This is the safest place to test small $5 trades while verifying everything lines up (markets, brackets, temps).

---

## 9. Next Steps for the Coding Agent

When the agent reads through all the code, this is the concrete checklist:

1. **Visual Crossing client**

   * Implement / verify three explicit methods:

     * `fetch_historical_daily_forecast(...)`
     * `fetch_historical_hourly_forecast(...)`
     * `fetch_current_forecast(...)`
   * Ensure historical vs current are never mixed.

2. **Historical ingestion**

   * Confirm `ingest_vc_forecast_history.py` + `ingest_vc_forecast_hourly.py` only use the **historical** methods and write to `wx.forecast_snapshot*` with correct `(city, target_date, basis_date)`.

3. **Live forecast daemon**

   * Update `poll_vc_forecast_daemon.py` to use `fetch_current_forecast("City,State")` nightly and store current forecasts with `basis_date = today`.

4. **Backtest data loading**

   * `open_maker.core` must only read from `wx.forecast_snapshot*` and Kalshi tables; no live VC calls.

5. **Bracket mapping**

   * Keep the current rounding + inclusive logic in `find_bracket_for_temp` and settlement in `determine_winning_bracket` as the single source of truth.

6. **Strategies**

   * Ensure:

     * `open_maker_base` uses the unified runner (`run_strategy`) and tuned parameters.
     * `open_maker_next_over` uses hourly forecast + candles at decision times.
     * `open_maker_curve_gap` uses `wx.minute_obs` + forecast curve gap logic as currently implemented.
     * The planned `open_maker_linear_model` strategy gets its own param dataclass and training pipeline (later).

7. **Live trader**

   * Keep it **simple**: for now, only run base strategy (`open_maker_base`) with tuned params and small bet sizes ($5–$20).
   * Use WebSocket `market_lifecycle` to detect opens rather than guessing clock times.
   * Use `_get_live_forecast_for_event` with DB‑first and VC‑fallback logic.

8. **Observability**

   * Ensure all live actions log to:

     * `sim.live_orders` (submitted orders),
     * logs with basis_date/target_date/forecast used,
     * optional `rt_signals` table for which strategy and parameters drove the trade.

Once this is stable, you can step up:

* add `next_over` and `curve_gap` as **paper trading overlays**,
* wire in the linear model as a filter,
* and gradually increase size as live performance confirms the backtest story.

---

