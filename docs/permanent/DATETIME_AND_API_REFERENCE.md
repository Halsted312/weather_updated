# Permanent doc: Datetime, Timezones & API Reference
`docs/permanent/DATETIME_AND_API_REFERENCE.md`.

---

# Datetime, Timezones & API Reference

*for Kalshi Weather + Visual Crossing + Settlement Data*

This document is meant to be a permanent reference for any agent or human working on the project. It defines:

* How we use **dates, times, and time zones**
* How to call and interpret **Visual Crossing** APIs
* How to call and interpret **Kalshi** REST + WebSocket APIs
* How we handle **settlement data** (NOAA / IEM)
* How to wire all of this into **backtests** vs **live trading**

---

## 1. Core Concepts & Conventions

### 1.1 Canonical concepts

We use the following vocabulary everywhere:

* **event_date**
  The local calendar date that the Kalshi weather market is about (e.g. “highest temperature in Chicago on 2025‑11‑28”).

  * Stored as `DATE` (no timezone) in the DB.
  * Always interpreted in the city’s local time zone (e.g. `America/Chicago`).

* **basis_date**
  The date on which a forecast was **issued**.

  * Example: for a forecast made late on 2025‑11‑27 about 2025‑11‑28, `basis_date = 2025‑11‑27`.
  * For backtests, this is **historical** (e.g. “what did the model think on that day?”).
  * For live trading, this is typically **today’s date in the city’s local time**.

* **lead_days / lead_hours**

  * `lead_days = (target_date - basis_date)`.
  * `lead_hours = floor((target_hour_local - basis_midnight_local) / 1 hour)`.

* **target_date / target_hour**

  * `target_date`: the date you’re forecasting (same as `event_date` for day‑level high temperature).
  * `target_hour_local`: timestamp (local) of an hourly forecast or observation.

* **local time vs UTC**

  * **Local time**: the time in the city’s own time zone (e.g. Chicago = `America/Chicago`).
  * **UTC**: used for:

    * Kalshi timestamps (`open_ts`, `close_ts`, `strike_ts`) – Unix epoch seconds in UTC.
    * Database `TIMESTAMPTZ` columns.
    * WebSocket timestamps.

### 1.2 Python & SQL types

* In Python:

  * Use `datetime.date` for `event_date`, `basis_date`, `target_date`.
  * Use `datetime.datetime` **with tzinfo** (`zoneinfo.ZoneInfo(city_tz)`) for real timestamps.
  * Convert to UTC with `.astimezone(timezone.utc)` whenever you store `TIMESTAMPTZ`.

* In Postgres:

  * Use `DATE` for `event_date`, `basis_date`, `target_date`.
  * Use `TIMESTAMP` (no tz) for `*_local` fields **plus** a `tz_name` column.
  * Use `TIMESTAMPTZ` for `*_utc` fields and `created_at`.

### 1.3 City configuration

Each city should have, in `src/config/cities.py` (or equivalent):

* `icao`: station ID (e.g. `"KMDW"`)
* `location_query`: city string for Visual Crossing (e.g. `"Chicago,IL"`)
* `tz_name`: IANA timezone (e.g. `"America/Chicago"`)
* `series_ticker`: Kalshi weather series (e.g. `"KXHIGHCHI"`)

Agents must always go through this config instead of hard‑coding strings.

---

## 2. Visual Crossing Weather API

Visual Crossing’s **Timeline API** is the core for both **historical** and **current** forecasts, plus observations.

### 2.1 Key endpoints & patterns

Visual Crossing supports a generic “timeline” endpoint:

```text
GET https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start}/{end}
  ?key=API_KEY
  &unitGroup=us
  &include=days,hours
  &contentType=json
```

* `location` can be:

  * `"City,State"` (e.g. `"Chicago,IL"`) – **preferred for current forecasts**.
  * `lat,lon` – an alternative.
  * Station IDs like `stn:KMDW` – works better for **historical** data, but in your experiments returned `0.0` for future `tempmax`.

* `include=days,hours`:

  * `days[]`: daily summaries (e.g. `tempmax`, `tempmin`, `humidity`, `conditions`).
  * `hours[]`: hourly forecasts/observations.

#### 2.1.1 Historical forecast (for backtesting)

You can request **historical forecasts** (what VC predicted on a given basis date) via the `forecastBasisDate` parameter.

Example pattern:

```text
GET .../timeline/{location}/{target_date}/{target_date}
  ?key=API_KEY
  &unitGroup=us
  &include=days,hours
  &forecastBasisDate={basis_date}
```

* `basis_date` is in the past.
* `target_date` can be equal to or after `basis_date`.
* VC returns what their model predicted **on `basis_date`** for that `target_date`.

**Use case** in your code:

* `scripts/ingest_vc_forecast_history.py`
  loops over **historical `basis_date` values** and populates:

  ```sql
  wx.forecast_snapshot (
    city,
    target_date,
    basis_date,
    lead_days,
    tempmax_fcst_f,
    ...
  )
  ```

  plus `wx.forecast_snapshot_hourly` for hourly curves.

#### 2.1.2 Current forecast (for live trading)

For **live** trading, we want **today’s best forecast for the next ~7–15 days**, not a historical snapshot.

Pattern:

```text
GET .../timeline/{location}  # no explicit start/end
  ?key=API_KEY
  &unitGroup=us
  &include=days,hours
  &contentType=json
```

* Using a city string like `"Chicago,IL"` returns ~15 days of forecast with non‑zero `tempmax`.
* For each `day`:

  * `day["datetime"]` → `"YYYY-MM-DD"` (local date)
  * `day["tempmax"]` → forecast daily high in °F
* The **current forecast basis** is “now” (when you call it). VC does not expose a separate timestamp but for this project we treat the basis_date as **today’s local date**.

**Important**: It’s a mistake to use the **historical forecast** endpoint (`forecastBasisDate`) to get current forecasts – that returns `0.0` for `tempmax` for future days in your tests.

#### 2.1.3 Minute / hourly observations

Visual Crossing’s timeline can also return historical observations:

* For hourly/5‑min observations you used a dedicated client in `visual_crossing.py` + `scripts/ingest_vc_minutes.py` (5‑min intervals into `wx.minute_obs`).
* Those data are in **local time** in the API, but you convert to:

  * `obs_time_local` (`TIMESTAMP` + `tz_name`)
  * `obs_time_utc` (`TIMESTAMPTZ`)

### 2.2 Date & time semantics in Visual Crossing

From the docs and examples:

* `days[i].datetime`

  * String `"YYYY-MM-DD"`.
  * Represents the **local calendar day**.
* `hours[i].datetime`

  * Local time (often with explicit offset, e.g. `"2020-06-25T21:00:00-04:00"`).
  * Interpreted in the **location’s timezone**.
* VC also returns timezone info for the location (e.g. `timezone`, `tzoffset`); your code already uses city config to interpret local times.

Your DB conventions:

* `wx.forecast_snapshot`:

  * `target_date` = `days[i].datetime` (local date)
  * `basis_date` = local date of the forecast issuance (for historical: the `forecastBasisDate`; for live: `today()` in that city)
  * `lead_days` = `(target_date - basis_date)`
* `wx.forecast_snapshot_hourly`:

  * `target_hour_local` = parsed from `hours[i].datetime` in local time
  * `target_hour_epoch` = `int(target_hour_local.astimezone(UTC).timestamp())`
  * `tz_name` = the city’s timezone string

### 2.3 Summary of how agents should think

* **Backtesting:**

  * Use `forecastBasisDate` to reconstruct what VC believed on a past date.
  * Always store that as `basis_date` and keep `lead_days` consistent.
* **Live trading:**

  * Use **plain timeline forecast** with city name, no `forecastBasisDate`.
  * Map `target_date` to `event_date` on Kalshi.
  * Treat `basis_date` as “today in city local time”.

---

## 3. Kalshi Trading & WebSocket APIs

Kalshi has a REST Trading API and a WebSocket API that share common timestamps and event structures. The official docs specify date/time fields as Unix **epoch seconds in UTC**.

### 3.1 REST Trading API (high‑level)

Base URL (prod) is documented as something like:

```text
https://trading-api.kalshi.com/trade-api/v2/
```

(You already use this in your `KalshiClient`.)

Important endpoints for this project:

* **Markets listing** – used by your ingestion and manual trader:

  * Takes `series_ticker`, `status`, etc.
  * Returns a list of markets with:

    * `ticker`
    * `event_ticker`
    * `series_ticker`
    * `status`
    * `open_ts`, `close_ts`, `strike_ts` (Unix seconds, UTC)
    * `strike_type` (e.g. `"between"`, `"less"`, `"greater"`)
    * `floor_strike`, `cap_strike` (°F for weather markets)
* **Candlesticks** – used to get price history for backtests:

  * Returns OHLC candles with timestamps in **epoch seconds** or ISO UTC, depending on endpoint (your `schemas.py` already handles this).
* **Trades / order placement**:

  * Orders include:

    * `ticker`
    * `action` (BUY/SELL)
    * `type` (limit/market)
    * `side` (YES/NO)
    * `price` (cents, 0–100)
    * `client_order_id`
  * Responses include:

    * `order_id`
    * `status`
    * `created_time` (ISO UTC), etc.

### 3.2 Event & market dates

Weather markets encode dates in their tickers. Example: `KXHIGHCHI-25NOV28-B33.5`.

* `25NOV28` → `DDMMMYY`:

  * Day: `25`
  * Month: `NOV`
  * Year: `2028` (assume `20YY`).
* This is the **event_date in local calendar** for that city.

Your ingestion logic:

* Parse `event_ticker` into a Python `date` using `datetime.strptime("25NOV28", "%d%b%y")`.
* Store as `event_date` (`DATE`) in DB.
* Weather markets always settle based on **local NOAA station daily TMAX** for that `event_date`.

### 3.3 WebSocket API: market lifecycle

The WebSocket endpoint is documented as:

```text
wss://trading-api.kalshi.com/trade-api/ws/v2
```

Channels include `market_lifecycle` and `event_lifecycle`. Messages carry fields like:

* `event_ticker`
* `series_ticker`
* `market_ticker`
* `status` (e.g. `open`, `closed`, `resolved`)
* `open_ts`, `close_ts`, `strike_ts` – **Unix epoch seconds (UTC)**.

To detect **when a next‑day weather market opens**:

1. Subscribe to `market_lifecycle` for the relevant series tickers:

   * `KXHIGHCHI`, `KXHIGHAUST`, `KXHIGHDENV`, `KXHIGHLAX`, `KXHIGHMIA`, `KXHIGHPHIL` (exact values are in your `CITIES` config).
2. Watch for messages where:

   * `status` transitions to `open`.
   * `event_ticker` includes the desired date code (e.g. `25NOV28`).
3. Interpret `open_ts` as:

   ```python
   open_time_utc = datetime.fromtimestamp(open_ts, tz=timezone.utc)
   open_time_local = open_time_utc.astimezone(city_tz)
   ```
4. Empirically, weather markets **tend to list mid‑morning the day before** the event, but this is *not guaranteed* and not documented. Never hard‑code 10:00; always rely on the WebSocket lifecycle events.

### 3.4 Brackets & temperatures

Weather markets use discrete brackets:

* Types:

  * `less`: “X° or below” – temp ≤ X
  * `greater`: “Y° or above” – temp ≥ Y+1
  * `between`: `[floor_strike, cap_strike)` in °F, but subtitles specify which whole degrees win.
* Example from your Philadelphia inspection:

  | Ticker      | Type      | floor | cap  | Subtitle       |
  | ----------- | --------- | ----- | ---- | -------------- |
  | `...-T43`   | `less`    | None  | 43.0 | `42° or below` |
  | `...-B43.5` | `between` | 43.0  | 44.0 | `43° to 44°`   |
  | `...-B45.5` | `between` | 45.0  | 46.0 | `45° to 46°`   |
  | `...-B47.5` | `between` | 47.0  | 48.0 | `47° to 48°`   |
  | `...-B49.5` | `between` | 49.0  | 50.0 | `49° to 50°`   |
  | `...-T50`   | `greater` | 50.0  | None | `51° or above` |

**Important rules:**

* Settlement uses **whole degree values** from NOAA (TMAX), so:

  * We should **round the temperature to nearest integer** before selecting a bracket (e.g. 46.1°F → 46).
* Your `find_bracket_for_temp()` logic (trading mode) should:

  1. Round `temp_adj` to nearest integer.
  2. Map to `less / between / greater` using the integer:

     * `less`: wins when rounded T ≤ `cap_strike`.
     * `between`: wins when `floor_strike ≤ rounded T ≤ cap_strike`.
     * `greater`: wins when rounded T ≥ `floor_strike + 1`.

---

## 4. Settlement Data: NOAA / IEM

You use **multiple sources** to reconstruct daily high temperatures (TMAX) for each city.

### 4.1 Iowa Environmental Mesonet (IEM)

* Provides archived daily climate summaries, including TMAX, for many stations.
* Accessed via simple HTTP APIs (CSV/JSON) keyed by station ID and date.
* Data are already processed into daily values for the station’s **local calendar day** (00–24 local standard time).

In your code:

* `src/weather/iem_cli.py` fetches daily TMAX.
* You write into:

  ```sql
  wx.settlement (
    city,
    event_date,
    tmax_iem_f,
    ...
  )
  ```

### 4.2 NOAA NCEI Daily Summaries

* NOAA NCEI’s daily summary datasets (e.g. GHCN‑Daily / ISD‑derived) expose daily **TMAX** as the maximum temperature over the **local day** at the station.
* They are used primarily for validation and redundancy; you do not need to query them in real time.

In your code:

* `src/weather/noaa_ncei.py` is an auxiliary client.
* You bring in `tmax_ncei_f` for cross‑checking against IEM.

### 4.3 NWS CLI/CF6

* CLI/CF6 products (daily CLI and monthly CF6 bulletins) provide recent daily climatology for an airport, but:

  * They are not a robust historical API.
  * They generally only contain **current and recent months**.
* You initially tried to use them for history, then switched to IEM + NCEI for full historical coverage.

---

## 5. Database & Schema Conventions (Datetime‑specific)

These are the **canonical expectations** for existing tables and any new ones.

### 5.1 Forecast tables

**Daily forecasts:**

```sql
wx.forecast_snapshot (
  city            TEXT NOT NULL,
  target_date     DATE NOT NULL,    -- local date being forecast
  basis_date      DATE NOT NULL,    -- local date forecast was issued
  lead_days       INTEGER NOT NULL, -- target_date - basis_date
  provider        TEXT DEFAULT 'visualcrossing',
  tempmax_fcst_f  FLOAT,
  tempmin_fcst_f  FLOAT,
  precip_fcst_in  FLOAT,
  precip_prob_fcst FLOAT,
  humidity_fcst   FLOAT,
  windspeed_fcst_mph FLOAT,
  conditions_fcst TEXT,
  raw_json        JSONB,
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (city, target_date, basis_date)
);
```

**Hourly forecast curves:**

```sql
wx.forecast_snapshot_hourly (
  city              TEXT NOT NULL,
  target_hour_local TIMESTAMP NOT NULL,  -- local time
  basis_date        DATE NOT NULL,
  target_hour_epoch INTEGER NOT NULL,    -- epoch seconds UTC
  lead_hours        INTEGER NOT NULL,
  provider          TEXT DEFAULT 'visualcrossing',
  tz_name           TEXT NOT NULL,
  temp_fcst_f       FLOAT,
  feelslike_fcst_f  FLOAT,
  humidity_fcst     FLOAT,
  precip_fcst_in    FLOAT,
  precip_prob_fcst  FLOAT,
  windspeed_fcst_mph FLOAT,
  conditions_fcst   TEXT,
  raw_json          JSONB,
  created_at        TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (city, target_hour_local, basis_date)
);
```

### 5.2 Observations

```sql
wx.minute_obs (
  city           TEXT,
  obs_time_local TIMESTAMP NOT NULL,     -- local
  obs_time_utc   TIMESTAMPTZ NOT NULL,   -- UTC
  tz_name        TEXT NOT NULL,
  temp_f         FLOAT,                  -- observed temp
  ...
);
```

Rules for agents:

* When inserting:

  * Start from a timezone‑aware `obs_time_local` and `city.tz_name`.
  * Compute `obs_time_utc = obs_time_local.astimezone(UTC)`.

### 5.3 Kalshi markets, candles, trades

You already have tables like:

* `kalshi.markets`
* `kalshi.candles_1m`
* `sim.live_orders`
* `sim.trade` / `sim.run` etc.

Datetime rules:

* Store `open_ts`, `close_ts`, `strike_ts` as **epoch seconds** OR `TIMESTAMPTZ`.
* If you store epoch seconds in an integer column, name it `*_epoch`.
* Whenever you need local time:

  ```python
  open_local = datetime.fromtimestamp(open_ts, tz=timezone.utc).astimezone(city_tz)
  ```

---

## 6. Putting It Together: Typical Flows

### 6.1 Backtesting flow (historical)

**Goal:** Evaluate a strategy using only information that would have been known at the time.

For each `(city, event_date)`:

1. **Determine basis_date**

   * For “open maker” style strategy, use `basis_date = event_date - 1 day` (city local).
   * This mimics “we trade when the market opens the day before using that day’s forecast”.

2. **Fetch forecast from DB**

   * From `wx.forecast_snapshot`:

     ```sql
     SELECT tempmax_fcst_f
     FROM wx.forecast_snapshot
     WHERE city = :city
       AND target_date = :event_date
       AND basis_date = :basis_date;
     ```
   * This should come from **historical forecast ingestion** using `forecastBasisDate = basis_date`.

3. **Pick bracket**

   * Adjust forecast by tuned bias (e.g. +1.1°F).
   * Round to nearest integer.
   * Use `find_bracket_for_temp()` with `round_for_trading=True` to pick the bracket ticker.

4. **Simulate price at market open**

   * From `kalshi.candles_1m`:

     * Use the first candle after `market_open_utc` (from Kalshi lifecycle or from DB `open_ts`).
     * Use the `yes_ask_c` (ask) price to simulate a realistic fill.
   * Apply taker/maker fees correctly.

5. **Settlement**

   * Get `tmax_final` from `wx.settlement` (merged IEM + NCEI).
   * Round to nearest integer.
   * Determine winning bracket using same bracket logic.

6. **P&L and metrics**

   * Compute trade P&L, daily P&L, Sharpe, etc.
   * Ensure all train/test splits are **time‑based** (e.g. first 70% days train, last 30% test).

### 6.2 Live trading flow (current)

**Goal:** At 10:00 local (or whenever Kalshi actually opens the next‑day market), decide whether and how to trade.

For each city:

1. **Detect market open (WebSocket)**

   * Subscribe to `market_lifecycle` for that city’s weather series.
   * When a `market` event comes in with:

     * `status` = `open`
     * `event_ticker` for **tomorrow’s date** (parsed from `DDMMMYY`),
   * Convert `open_ts` (epoch UTC) into local time.

2. **Fetch current forecast**

   * Option A (preferred): read from `wx.forecast_snapshot` row where:

     * `basis_date = today_local`
     * `target_date = event_date`
   * Option B (fallback): call Visual Crossing `timeline/{city.location_query}` and pull `day["tempmax"]` where `day["datetime"] == event_date`.

3. **Pick bracket & size**

   * Apply tuned bias and bracket logic as in backtest.
   * Use current best params from `config/open_maker_base_best_params.json`.
   * Bet amount for live is small (e.g. `$20`) to avoid moving markets.

4. **Place order**

   * Use the same logic as `manual_trade.py` / `live_trader.py`:

     * `BUY YES` @ `entry_price_cents` (maker/order that sets price).
     * `client_order_id` with clear prefix (e.g. `lt-{city}-{event_date}-{uuid4}`).
   * Log to `sim.live_orders` with:

     * `strategy_id`
     * `city`, `event_date`
     * `basis_date`
     * `forecast_used`
     * `trade_time_utc`

5. **Intraday adjustments (curve_gap / next_over)**

   * At decision time τ (e.g. 2h before predicted high):

     * Use `wx.minute_obs` + `wx.forecast_snapshot_hourly` to compute:

       * T_obs (near‑term average)
       * T_fcst (hourly forecast)
       * slope over last 1h
     * Strategy decides whether to:

       * Hold
       * Shift bin (sell one bracket, buy higher one)
       * Exit entirely
   * All these decisions must reference **local timestamps** for the city, but executed in UTC.

---

## 7. Practical Guidelines for Agents

When writing new code:

1. **Always specify timezone when constructing datetimes.**

   * Wrong: `datetime(2025, 11, 28, 10, 0)`
   * Right: `datetime(2025, 11, 28, 10, 0, tzinfo=ZoneInfo(city.tz_name))`

2. **Never mix historical vs current VC endpoints.**

   * Historical backfill: use `forecastBasisDate` (past) → `wx.forecast_snapshot`.
   * Live: use plain `timeline/{city}` (no basis) for current forecast.

3. **Differentiate object types clearly:**

   * Historical forecast function (e.g. `fetch_historical_hourly_forecast(...)`) **must not** be used for live trading.
   * Live forecast function can be something like `fetch_current_forecast(city.location_query, horizon_days=7)`.

4. **Brackets: always round before mapping.**

   * Round the forecast (and the settled TMAX) to nearest integer, then map to bracket.

5. **WebSocket lifecycle is the source of truth for market open times.**

   * Do not hard‑code “markets open at 10:00”.
   * Save `open_ts` and `strike_ts` for each `(city, event_date)` into a small `sim.kalshi_market_open` table if helpful.

6. **Train/test splits must be time‑based.**

   * For Optuna and backtests, always split by **date** (e.g. first 70% of days train, last 30% test) to avoid look‑ahead bias.

---
