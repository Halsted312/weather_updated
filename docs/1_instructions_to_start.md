I’ll break it into two big chunks:

1. **Database + logging design** – what DB, what tables, what to record (fine‑grained).
2. **Option‑1 backtest spec** – exactly what data joins + computations the agent should implement.

I’ll write bits you can basically paste straight into Claude.

---

## 1. DB + logging design (Timescale + schemas)

### 1.1. Overall stack

What to tell Claude:

> **Stack & infra**
>
> * Use **PostgreSQL + TimescaleDB** (Timescale extension) as the primary database.
> * Use Docker Compose with two services:
>
>   * `db`: Postgres 16 + TimescaleDB extension enabled.
>   * `app`: Python 3.11 container where all ingestion/backtest code runs.
> * Use **SQLAlchemy + Alembic** for migrations.
> * Create three logical schemas:
>
>   * `wx` – weather / labels (NWS, Visual Crossing).
>   * `kalshi` – market data (metadata, candles, WebSocket logs, my fills).
>   * `sim` – backtest runs & simulated trades.

---

### 1.2. Weather schema (`wx`)

You’ve already done most of the thinking here in your docs; we’re just lifting it into v2.

#### a) Official settlement temps – `wx.settlement`

Use the reconciled NWS pipeline from your “how‑to weather history” doc: IEM ADS + NWS CLI/CF6 → `tmax_final` with `source_final`.

Tell Claude:

> **Table: `wx.settlement`**
>
> * Purpose: one row per (city, date_local) with the **official NWS daily max temp** that Kalshi will settle on.
> * Columns:
>
>   * `city` TEXT – canonical id, e.g. `'chicago'`, `'denver'`, `'austin'`.
>   * `date_local` DATE – climate day in station local time.
>   * `tmax_cli_f` INT NULL – NWS CLI daily max.
>   * `tmax_cf6_f` INT NULL – NWS CF6 daily max.
>   * `tmax_ads_f` INT NULL – IEM ASOS ADS max (fallback).
>   * `tmax_final` INT NOT NULL – chosen settlement temp.
>   * `source_final` TEXT NOT NULL – `'cli' | 'cf6' | 'ads'`.
>   * `created_at` TIMESTAMPTZ, `updated_at` TIMESTAMPTZ.
> * PK: `(city, date_local)`.
> * Make this a *regular* table (daily grain; no need for hypertable).
> * Ingestion code:
>
>   * Implement the ADS+CLI+CF6 reconciliation exactly as in the previous repo:
>
>     * Pull IEM daily ASOS (`daily.py`), NWS CLI via AFOS, and CF6 JSON.
>     * Apply precedence **CLI > CF6 > ADS**.
>   * Backfill from `2024‑01‑01` to “today” for your 6 cities (no NYC).
>   * Write a small validation script that compares `tmax_final` to raw ADS & CF6 to spot discrepancies.

(That’s essentially the pipeline in `hot-to_weather_history.md`, just re‑implemented in the new repo.)

#### b) Visual Crossing minute obs – `wx.minute_obs`

Reuse your VC design almost verbatim: station‑pinned, 5‑min grid, forward‑fill, `ffilled` flag.

> **Table: `wx.minute_obs` (Timescale hypertable)**
>
> * Purpose: 5‑minute **observed** temps from Visual Crossing for each station, used as features (never settlement).
> * Columns:
>
>   * `loc_id` VARCHAR(10) NOT NULL – station id: `KMDW`, `KDEN`, `KAUS`, `KLAX`, `KMIA`, `KPHL`.
>   * `ts_utc` TIMESTAMPTZ NOT NULL – 5‑minute timestamp in UTC.
>   * `temp_f` DOUBLE PRECISION – temperature.
>   * `humidity` DOUBLE PRECISION.
>   * `dew_f` DOUBLE PRECISION.
>   * `windspeed_mph` DOUBLE PRECISION.
>   * `source` VARCHAR(20) DEFAULT `'visualcrossing'`.
>   * `stations` VARCHAR(50) – station list returned by VC (`"KMDW"` etc.) for diagnostics.
>   * `ffilled` BOOLEAN NOT NULL DEFAULT FALSE – TRUE if this row was forward‑filled.
>   * `raw_json` JSONB – optional raw VC segment for diagnostics.
> * PK: `(loc_id, ts_utc)`.
> * Hypertable:
>
>   * Time column: `ts_utc`.
>   * Space partition: `loc_id`.
>   * Chunk interval: `INTERVAL '7 days'` (fine).
> * VC client behavior (match your docs):
>
>   * Use Timeline API `/timeline/stn:{STATION}/{start}/{end}` with
>     `include=minutes`, `options=useobs,minuteinterval_5,nonulls`, `unitGroup=us`, `maxStations=1`, `maxDistance=0`, `elements=datetimeEpoch,temp,dew,humidity,windspeed,stations`.
>   * Only ingest the **6 airport stations**; NYC is excluded because of the high forward‑fill rate.
>   * For each day/city:
>
>     * Build a complete 5‑minute UTC grid (288 rows).
>     * Join VC minutes onto that grid; forward‑fill gaps; mark `ffilled`.
>   * Record coverage metrics per day (this matches your VC summary docs).

#### c) Visual Crossing daily forecast snapshots – `wx.forecast_snapshot`

This is new for Option‑1.

> **Table: `wx.forecast_snapshot`**
>
> * Purpose: store **historical forecast of the daily high** for each (city, target_date) as of a given basis date (lead time).
> * Columns:
>
>   * `city` TEXT – same ids as `wx.settlement`.
>   * `target_date` DATE – date whose high we’re forecasting.
>   * `basis_date` DATE – date the forecast was issued (midnight run date).
>   * `lead_days` INT – `target_date - basis_date`.
>   * `provider` TEXT DEFAULT `'visualcrossing'`.
>   * `tempmax_fcst_f` DOUBLE PRECISION – VC forecast for daily high `tempmax` (°F) as of `basis_date`.
>   * `raw_json` JSONB – optional raw VC response slice.
>   * `created_at` TIMESTAMPTZ.
> * PK: `(city, target_date, basis_date)`.
> * Ingestion:
>
>   * Use the **historical forecast** feature of VC’s Timeline API: `forecastBasisDate` or `forecastBasisDay` as documented.
>   * For each city and `target_date` from 2024‑01‑01 onward, and for each lead in some set like `{0,1,2,3,5,7}`:
>
>     * Compute `basis_date = target_date - lead_days`.
>     * Call VC Timeline: `/timeline/stn:{STATION}/{target_date}/{target_date}?forecastBasisDate={basis_date}&include=days&unitGroup=us`.
>     * Extract `days[0].tempmax` and store as `tempmax_fcst_f`.
>   * Only do this for the 6 non‑NYC cities.

---

### 1.3. Kalshi schema (`kalshi`)

#### a) Market metadata – `kalshi.markets`

> **Table: `kalshi.markets`**
>
> * Purpose: static info per Kalshi contract (bin).
> * Columns:
>
>   * `ticker` TEXT PRIMARY KEY – Kalshi ticker.
>   * `city` TEXT – matches our city ids; can store `NULL` for non‑weather.
>   * `event_date` DATE – the weather date (settlement day).
>   * `exchange_market_id` TEXT – Kalshi’s ID if available.
>   * `strike_type` TEXT – `'between' | 'less' | 'greater'`.
>   * `floor_strike` INT NULL – low bound in °F.
>   * `cap_strike` INT NULL – high bound in °F.
>   * `listed_at` TIMESTAMPTZ.
>   * `close_time` TIMESTAMPTZ.
>   * `status` TEXT – `'open' | 'closed' | 'settled'`, etc.
>   * `raw_json` JSONB – the exchange definition payload.
> * This table lets you map from `ticker` to (city, date, bin).

#### b) One‑minute candles – `kalshi.candles_1m`

You already have 1‑minute historical data; let’s store it as a hypertable.

> **Table: `kalshi.candles_1m` (Timescale hypertable)**
>
> * Columns:
>
>   * `ticker` TEXT NOT NULL REFERENCES `kalshi.markets`(ticker).
>   * `bucket_start` TIMESTAMPTZ NOT NULL – start of 1‑minute bar.
>   * `open` SMALLINT – YES price at start (0–100).
>   * `high` SMALLINT.
>   * `low` SMALLINT.
>   * `close` SMALLINT.
>   * `volume` INT – contracts traded in that minute (if available).
> * PK: `(ticker, bucket_start)`.
> * Hypertable:
>
>   * Time: `bucket_start`.
>   * Space: `ticker`.
> * Backfill:
>
>   * Write a job that loads your existing minute data here.
>   * For new days, a REST poller can fetch completed bars periodically (even if WS is also running).

#### c) WebSocket raw log – `kalshi.ws_raw`

This is your “record absolutely everything” table.

> **Table: `kalshi.ws_raw` (Timescale hypertable)**
>
> * Purpose: log *every* Kalshi WebSocket message as‑is so nothing is lost.
> * Columns:
>
>   * `id` BIGSERIAL PRIMARY KEY.
>   * `ts_utc` TIMESTAMPTZ NOT NULL – time of receipt (monotonic based on local clock).
>   * `source` TEXT NOT NULL DEFAULT `'kalshi'`.
>   * `stream` TEXT – e.g. `'market-data'`, `'trades'`, `'orderbook'`, `'fills'`.
>   * `topic` TEXT – usually ticker or channel key.
>   * `payload` JSONB – raw decoded JSON frame.
> * Hypertable on `ts_utc`.
> * Recorder process:
>
>   * Python script that:
>
>     * Opens Kalshi WS connections to:
>
>       * market data / quotes for your 6 weather markets,
>       * order book / trades,
>       * your account fills.
>     * For every frame received, insert a row into `kalshi.ws_raw`.
>   * Keep insertion simple (batch insert per second).

Later you can build derived tables (`orderbook_snapshots`, `trade_ticks`, etc.) by replaying `ws_raw`, but v1 just needs to **store** the stream.

#### d) My orders/fills – `kalshi.orders`, `kalshi.fills`

> **Tables:**
>
> * `kalshi.orders` – my sent orders (id, ticker, side, qty, price, status, created_at, etc.).
> * `kalshi.fills` – my fills (fill_id, order_id, ticker, side, qty, price, ts_utc, raw_json).
> * Both keyed by `ts_utc` + `order_id`/`fill_id` (make them hypertables if you want).

---

### 1.4. Simulation / backtest schema (`sim`)

For later, but easy to outline:

> **Core tables**
>
> * `sim.run`:
>
>   * `run_id` UUID PRIMARY KEY.
>   * `strategy_name`, `params_json`, `train_start`, `train_end`, `test_start`, `test_end`, `created_at`.
> * `sim.trade` (hypertable on `trade_ts_utc`):
>
>   * `run_id` UUID, `trade_ts_utc`, `ticker`, `city`, `event_date`, `side`, `qty`, `price`, `fees`, `pnl`, `position_after`, etc.

Option‑1 backtest results will write into these.

---

## 2. Phase‑1 ingestion: start recording “everything” tonight

Here’s the concrete “to‑do” for Claude, focused on just getting data in:

> **Ingestion focus for v1**
>
> 1. Implement `wx` ingestion:
>
>    * `scripts/ingest_settlement_nws.py`:
>
>      * Ingest NWS/IEM ADS + CLI + CF6 into `wx.settlement` for 6 cities from 2024‑01‑01 → today. (Replicate the logic from `hot-to_weather_history.md` to compute `tmax_final`.)
>    * `scripts/ingest_vc_minutes.py`:
>
>      * For the same range, call VC Timeline API for `stn:{station}` and load 5‑min `wx.minute_obs` rows using the station‑pinned approach in `how-to_visual_crossing.md`. Exclude NYC entirely as in `how-to_weather_non_NYC.md`.
>    * `scripts/ingest_vc_forecast_snapshots.py`:
>
>      * Build `wx.forecast_snapshot` using VC historical forecasts (`forecastBasisDate`) for each (city, target_date, basis_date) combination.
> 2. Implement `kalshi` ingestion:
>
>    * `scripts/backfill_kalshi_markets.py`:
>
>      * Hit Kalshi REST to download all weather markets for the 6 cities (past ~1–2 years).
>      * Populate `kalshi.markets` with city, event_date, bracket, floor/cap strikes, etc.
>    * `scripts/backfill_kalshi_candles.py`:
>
>      * For each weather ticker, fetch 1‑minute candlesticks for the full lifetime and load into `kalshi.candles_1m`.
>    * `scripts/run_kalshi_ws_recorder.py`:
>
>      * Open WebSocket streams and log every message into `kalshi.ws_raw`.
>      * Run as a long‑lived process during trading hours.
> 3. Scheduling:
>
>    * Use a simple `systemd` service or Python scheduler to:
>
>      * Run the VC minute ingestion **once per day** for backfill, and
>      * For live, poll VC every 5–15 minutes during trading hours to keep `wx.minute_obs` up to date.
>      * Keep the Kalshi WS recorder running continuously when the exchange is open.

That’s your “record absolutely everything” phase.

---

## 3. Option‑1 backtest (daily forecast edge) – spec for Claude

Now the report you asked for: how to implement Option‑1 in this repo.

Recall Option‑1: for each day and city, use Visual Crossing’s historical forecast for that day’s high at some lead time(s), compare to Kalshi’s bracket prices at a chosen decision time, and simulate “bet on the bracket implied by the forecast.”

### 3.1. Build the Option‑1 dataset

Tell Claude to create a module `backtest/option1_dataset.py` that:

> **Inputs**
>
> * `wx.settlement` – gives `tmax_final` per (city, date_local) (official NWS).
> * `wx.forecast_snapshot` – VC `tempmax_fcst_f` for each (city, target_date, basis_date, lead_days).
> * `kalshi.markets` – bin definitions for each ticker (city, event_date, strike_type, floor/cap).
> * `kalshi.candles_1m` – minute prices per ticker.

> **Processing logic**
>
> 1. For each `(city, target_date)` where we have both settlement and VC forecasts:
>
>    * Pull `tmax_final` from `wx.settlement`.
>    * For each `lead_days` in some set (e.g. 0,1,2,3,5,7):
>
>      * Find `basis_date = target_date - lead_days`.
>      * Load the corresponding `tempmax_fcst_f` from `wx.forecast_snapshot`.
>      * Decide a **decision time** `t_decision` for trading at that basis:
>
>        * Simple choice: `t_decision = basis_date 10:00 local` (Kalshi weather markets typically launch morning of the previous day or so – we just need a consistent rule).
>      * For each Kalshi weather contract (bin) on that `(city, target_date)`:
>
>        * From `kalshi.candles_1m`, get the 1‑minute candle covering `t_decision`:
>
>          * Use closing price `close` as the executable price; if no candle exists yet (market not open), skip this (basis_date, lead_days) combo.
>        * Compute:
>
>          * `is_true_bin = bin_resolves_yes(tmax_final, strike_type, floor_strike, cap_strike)` – deterministic YES/NO label from NWS temp. (Recreate your `resolve_bin` logic.)
>          * `is_forecast_bin` – whether this bin contains `round(tempmax_fcst_f)`.
>          * For now, treat forecast as “degenerate distribution at `round(tempmax_fcst_f)`.”
>      * Emit one dataset row per bin with:
>
>        * `city`, `target_date`, `basis_date`, `lead_days`, `ticker`,
>        * `tempmax_fcst_f`, `tmax_final`,
>        * `price_at_decision` (0–100),
>        * `is_true_bin` (0/1),
>        * `is_forecast_bin` (0/1).
> 2. Store this as a table `sim.option1_daily` OR as a parquet/CSV artifact for analysis.

### 3.2. Implement basic Option‑1 strategies

Now implement a simple backtest module `backtest/option1_run.py` that uses that dataset:

> **Strategies to implement**
>
> 1. **Naive “always bet forecast bin”**
>
>    * For each `(city, target_date, basis_date, lead_days)`:
>
>      * Find the bin where `is_forecast_bin = 1`.
>      * If that bin exists and has a price at decision:
>
>        * Assume we **buy 1 YES** contract at `price_at_decision` (in cents).
>        * Payout at settlement:
>
>          * `pnl = 100 - price` if `is_true_bin = 1`, else `-price`.
>        * Optionally subtract a flat fee per contract.
> 2. **Filtered version** (to tune later with Optuna):
>
>    * Only take the trade if the forecast is “strong” – e.g., distance between forecast temp and nearest bin boundary > 1°F, or forecast error distribution suggests high confidence.
>    * Only trade for certain `lead_days` (e.g., 0–2 days ahead).
>
> **Outputs**
>
> * Per `(city, lead_days)`:
>
>   * Number of trades.
>   * Win rate (fraction of YES contracts finishing in‑the‑money).
>   * Average PnL per trade.
>   * Cumulative PnL over time.
> * Optionally store results into `sim.run` + `sim.trade` tables.

This gives you a quick “is VC forecast systematically better than prices?” read before any fancy ML.

### 3.3. Bonus: fit a simple error model for later

Add a small module `models/forecast_error_model.py`:

> * For each `(city, lead_days)`:
>
>   * Compute `err = tmax_final - tempmax_fcst_f` across the whole history.
>   * Fit:
>
>     * Mean & stddev per lead,
>     * Maybe a Gaussian or empirical distribution.
> * Use that later to map `tempmax_fcst_f` into **bin probabilities** instead of a single “forecast bin” – but that’s v2.

---

## 4. TL;DR you can paste to Claude

If you want something short to drop into Claude, use this:

> I’m starting a fresh repo for a Kalshi weather trading system. Please:
>
> 1. Set up a **Postgres + TimescaleDB** database with schemas `wx`, `kalshi`, `sim`. Use SQLAlchemy + Alembic for migrations.
> 2. In `wx`, implement:
>
>    * `wx.settlement` – one row per (city, date_local) with NWS official daily high (`tmax_final`) and sources (`tmax_cli_f`, `tmax_cf6_f`, `tmax_ads_f`, `source_final`). Use the reconciliation logic from my old “hot-to_weather_history.md” file: pull IEM daily ASOS, NWS CLI, and CF6, then prefer CLI>CF6>ADS.
>    * `wx.minute_obs` – Timescale hypertable with 5‑min Visual Crossing observations per station, columns `(loc_id, ts_utc, temp_f, humidity, dew_f, windspeed_mph, ffilled, stations, raw_json)`, following the station‑pinned, 5‑minute, forward‑fill design in `how-to_visual_crossing.md` and excluding NYC as described in `how-to_weather_non_NYC.md`.
>    * `wx.forecast_snapshot` – VC **historical forecast** table: `(city, target_date, basis_date, lead_days, tempmax_fcst_f, raw_json)`. Use Visual Crossing’s `forecastBasisDate` parameter to get each target day’s `tempmax` as of earlier basis dates.
> 3. In `kalshi`, implement:
>
>    * `kalshi.markets` – static contract metadata for all weather markets (ticker, city, event_date, strike_type, floor_strike, cap_strike, listed_at, close_time, raw_json).
>    * `kalshi.candles_1m` – Timescale hypertable of 1‑minute candles per ticker (`ticker, bucket_start, open, high, low, close, volume`).
>    * `kalshi.ws_raw` – Timescale hypertable logging **every WebSocket frame**: `(id, ts_utc, source, stream, topic, payload_json)`.
>    * `kalshi.orders` and `kalshi.fills` – my orders/fills.
> 4. In `sim`, add `sim.run` and `sim.trade` to store backtest runs and trades.
> 5. Implement ingestion scripts:
>
>    * `ingest_settlement_nws.py` – fill `wx.settlement` from 2024‑01‑01→today for my six cities using IEM ADS + NWS CLI/CF6 reconciliation.
>    * `ingest_vc_minutes.py` – fill `wx.minute_obs` using VC Timeline with station‑pinned minute mode as in `how-to_visual_crossing.md` (6 cities only).
>    * `ingest_vc_forecast_snapshots.py` – populate `wx.forecast_snapshot` with VC daily `tempmax` forecasts for each (city, target_date, basis_date, lead_days).
>    * `backfill_kalshi_markets.py` and `backfill_kalshi_candles.py` – populate `kalshi.markets` and `kalshi.candles_1m` from Kalshi REST for all historical weather markets.
>    * `run_kalshi_ws_recorder.py` – long‑running process that logs every Kalshi WebSocket frame into `kalshi.ws_raw`.
> 6. Implement **Option‑1 “daily forecast edge” backtest**:
>
>    * Build an `option1` dataset table or CSV by joining `wx.settlement`, `wx.forecast_snapshot`, `kalshi.markets`, and `kalshi.candles_1m`:
>
>      * For each (city, target_date, basis_date, lead_days), take VC `tempmax_fcst_f`, NWS `tmax_final`, and Kalshi prices at a chosen decision time, and compute bin labels using my existing `resolve_bin`/`bin_resolves_yes` logic.
>    * Implement a simple strategy: “for each (city, target_date, basis_date), buy 1 YES contract in the bin that contains `round(tempmax_fcst_f)` at the decision time; compute PnL against `tmax_final`.” Summarize PnL by city and lead_days.
>    * Store results in `sim.run` and `sim.trade` and produce a small report (per‑city win rate, average PnL, cumulative curve).
>
> The goal is:
>
> * Tonight: get the DB + schemas + ingestion scripts in place and start the Kalshi WebSocket recorder and VC ingestion so we don’t miss data going forward.
> * Next: run the Option‑1 backtest on 2024‑01‑01→today to see whether simple VC forecast vs Kalshi prices had exploitable edge.

---
