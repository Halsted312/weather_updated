### README.md

# Weather Edge Trading on Kalshi

## Overview

This repository contains a full **research and trading stack** for **weather markets on Kalshi**, a US event-contract exchange regulated as a Designated Contract Market (DCM) by the CFTC. :contentReference[oaicite:0]{index=0}

Kalshi lists contracts that pay \$1 if an event occurs and \$0 otherwise (“event contracts”). In weather markets, these events are usually **daily high-temperature brackets** in specific cities (e.g., “Highest temperature in Chicago on April 15 between 71°F and 72°F?”). The weather markets are structured across **six brackets**: four 2°F middle brackets, plus one “below” tail and one “above” tail that capture everything outside the middle range. :contentReference[oaicite:1]{index=1}

The core idea of this project:

> Use high-quality weather data and forecasts to estimate the true probability that each temperature bracket will settle YES, then trade mispriced contracts on Kalshi.

The stack includes:

- **Historical observations** from NOAA / NCEI / IEM (daily TMAX and sub-hourly obs). :contentReference[oaicite:2]{index=2}  
- **Historical, current, and sub-hourly forecasts** from Visual Crossing’s Timeline Weather API. :contentReference[oaicite:3]{index=3}  
- **Kalshi market & candle data** (series, markets, minute-level prices, settlements). :contentReference[oaicite:4]{index=4}  
- A set of **trading strategies**, **backtesting tools**, and **Optuna tuning**.  
- **Live and manual trading scripts** that connect to Kalshi via REST + WebSocket.

Long-term goals:

- **Statistically robust** edges (out-of-sample P&L with reasonable Sharpe).
- **Operationally safe** tactics (small size vs market depth, realistic fill assumptions).
- A **modular, extensible** engine for adding new strategies, features, and markets.

---

## Key Concepts

### Kalshi Weather Markets

- Each city/day has a **series of markets** that cover a partition of possible daily high temperatures into 6 brackets. :contentReference[oaicite:5]{index=5}  
- Contracts trade in cents between 1 and 99, representing the market-implied probability of a YES outcome.
- Settlement is based on the **maximum temperature recorded by the National Weather Service** for the relevant station (e.g., KMDW for Chicago), rounded to whole degrees Fahrenheit. :contentReference[oaicite:6]{index=6}  

### Weather Data

We combine several sources:

1. **NOAA / NCEI / IEM** – “ground truth”
   - Daily **TMAX** (max temperature) for each city/station.  
   - Sub-hourly observations from airport stations (e.g., KMDW, KAUS, KDEN, KLAX, KMIA, KPHL). :contentReference[oaicite:7]{index=7}  
   - Drives the **settlement temperature** for each city/day.

2. **Visual Crossing Weather API**
   - Timeline API provides **historical, current, and forecast data in a single API**, including historical forecasts (what the model thought on past dates). :contentReference[oaicite:8]{index=8}  
   - Sub-hourly support (e.g., 5-minute obs, 15-minute forecast minutes) for intraday patterns. :contentReference[oaicite:9]{index=9}  
   - We use:
     - **Historical forecasts** via `forecastBasisDate` → “What did the model predict on basis date B for target date T?”  
     - **Current forecasts** without `forecastBasisDate` → live view of the next ~15 days.  
     - Hourly/minute forecast curves as features (e.g., predicted high time, temperature slopes, wind, cloud cover, CAPE/CIN).

---

## High-Level Architecture

The exact names evolve, but conceptually the system is split into:

### Database Models (`src/db/`)

SQLAlchemy models for:

- **Weather (`wx` schema)**  
  - Settlement tables for daily TMAX and Kalshi bucket mapping (e.g., `wx.settlement`).  
  - Sub-hourly observations from Visual Crossing and/or IEM (e.g., `wx.minute_obs` or `wx.vc_minute_weather`).  
  - Forecast snapshot tables:
    - Daily snapshots (target_date, basis_date, lead_days, temps, precip, etc.).  
    - Hourly curves (target_hour, basis_date, lead_hours, temps, wind, etc.).  

- **Kalshi (`kalshi` schema)**  
  - Series, events, and markets (metadata, strike structure, source docs). :contentReference[oaicite:10]{index=10}  
  - 1-minute candles (OHLC, yes_bid/yes_ask, volume, open interest).  
  - Any live orderbook logs and WS-derived data.

- **Simulation (`sim` schema)**  
  - Backtest runs, per-trade records, and live trade logs (e.g., `sim.run`, `sim.trade`, `sim.live_orders`).

**Authoritative structure**: always see `src/db/models.py` and `docs/permanent/FILE_DICTIONARY_GUIDE.md`.

### Weather Clients (`src/weather/`)

- `visual_crossing.py` – Visual Crossing client:
  - Timeline API calls for historical obs, current forecasts, and historical forecasts.  
  - Station-locked endpoints (`stn:KMDW`, etc.) for “pure station” feeds.  
  - City queries (`Chicago,IL`, etc.) for aggregate feeds.
  - Parses JSON into DB-ready structures (daily, hourly, minute).

- Additional clients for NOAA / IEM / NCEI as needed for daily TMAX and station metadata.

### Kalshi Client (`src/kalshi/`)

- REST API client for:  
  - Series, events, markets, and orderbooks. :contentReference[oaicite:11]{index=11}  
  - Account balances, orders, fills (for live trading).

- WebSocket client for:  
  - Market lifecycle events (open/close, halts).  
  - Real-time price updates and order book snapshots.

### Ingestion & Maintenance Scripts (`scripts/`)

Typical scripts include (names may vary slightly by branch):

- **Visual Crossing** (Phase 1 VC schema - `wx.vc_*` tables)
  - `ingest_vc_obs_backfill.py` – backfill 5-minute observations for each station/city.
  - `ingest_vc_forecast_snapshot.py` – nightly current+forecast snapshots.
  - `ingest_vc_historical_forecast.py` – backfill historical daily/hourly forecasts via `forecastBasisDate`.
  - *Legacy scripts are in `legacy/` folder (old schema, kept for reference).*

- **NOAA / Settlement**
  - `ingest_settlement_multi.py` – combine NWS sources (CLI, CF6, IEM, NCEI) into a single `wx.settlement` record per date/city.

- **Kalshi**
  - `backfill_kalshi_markets.py` – historical market/series backfill.  
  - `backfill_kalshi_candles.py` – historical 1-minute candle backfill.  
  - `kalshi_ws_recorder.py` – record live WebSocket streams for replay and analysis.

- **Ops / Utilities**
  - Health checks, Alembic migration helpers, file inventory tools.

### Strategy & Trading Engine (`open_maker/`)

Core components:

- `core.py`  
  - Orchestrates backtests:
    - Loads forecasts, obs, settlement, and market data.  
    - Runs strategies over cities / dates.  
    - Computes P&L, Sharpe, and other metrics.

- `utils.py` (possibly split into a package later)  
  - Bracket selection and settlement mapping (`find_bracket_for_temp`, `determine_winning_bracket`).  
  - Fee and position sizing utilities.  
  - Forecast helpers (predicted high hour, decision time).  
  - Intraday fill realism (checking whether prices actually traded).

- `strategies/`  
  - `base.py` – `open_maker_base`.  
  - `next_over.py` – `open_maker_next_over`.  
  - `curve_gap.py` – `open_maker_curve_gap`.  
  - Hooks for additional ML / linear strategies.

- `optuna_tuner.py`  
  - Parameter search for strategies (bias, entry price, decision time offsets, etc.) with train/test splits.

- `live_trader.py`  
  - Listens to Kalshi WS lifecycle events.  
  - On market open, fetches the appropriate forecast snapshot and places a maker order with tuned parameters.  
  - Logs orders to `sim.live_orders`.

- `manual_trade.py`  
  - CLI to inspect forecasts/markets and optionally place a single discretionary trade.  
  - Useful for dry-run experiments and live testing with tiny size.

---

## Strategies (High-Level)

### 1. `open_maker_base`

**Goal**: Simple “forecast → bracket” mapping, held to settlement.

- When a market opens (e.g., ~10:00 local time, as per Kalshi’s weather market design). :contentReference[oaicite:12]{index=12}  
- Fetch the Visual Crossing forecast high for that city and date, using a controlled basis offset (e.g., yesterday’s forecast).  
- Optionally apply a bias (e.g., +1°F if forecasts are systematically cold-biased).  
- Map the adjusted temperature to the closest bracket using the central utilities.  
- Place a **maker** limit order (e.g., at 30¢) on that bracket and hold to settlement.

Parameters tuned with Optuna include:

- `basis_offset_days` (how many days before the event we trust the forecast).  
- `temp_bias_deg` (forecast bias).  
- `entry_price_cents` (maker quote).  
- Optional filters like minimum edge or liquidity.

### 2. `open_maker_next_over`

**Goal**: Give the system one chance to “climb” to the next bracket if the curve rallies.

- Enter like `open_maker_base`.  
- Later, near the **predicted high time**:
  - Use hourly forecasts to estimate when the high is expected.  
  - Define a **decision window** around that time, load 1-minute candles for both:
    - Our current bracket.  
    - The next higher bracket.
- If the higher bracket is very strong and our bracket is cheap (e.g., next > X¢ while ours < Y¢), choose to:
  - Exit our bracket at the bid; or  
  - Shift exposure upward (logic depends on implementation and parameters).

This strategy uses both **forecast curves** and **intraday prices** to opportunistically move one bin higher while controlling risk.

### 3. `open_maker_curve_gap`

**Goal**: Compare actual vs forecasted curve to understand “curve surprises.”

- Enter like `open_maker_base`.  
- At a decision time (e.g., 2–3 hours before predicted high):
  - Look at **observed minute-level temps** vs **forecast temps** at that same time.  
  - Compute short-term slopes (warming/cooling).  
- If actual temps are significantly above/below the forecast and continuing in that direction, treat this as a signal that the eventual TMAX may land in a different bracket.

Initial implementation may be “hindsight only”:

- Keep the original entry in the backtest, but compute a **counterfactual** where we had selected a higher or lower bin based purely on the curve gap signal.  
- This informs whether a real intraday exit/shift strategy is worth implementing.

### 4. (Planned) ML / Linear Strategy

**Goal**: Turn the feature space into a probabilistic view of TMAX.

Potential features:

- Forecast path over time (sequence of daily tempmax forecasts for a given event date).  
- Hourly forecast curve shape vs observed curve.  
- Humidity, cloud cover, wind, CAPE/CIN, precip flags.  
- Deviations between station-locked and city-aggregate feeds.  
- Kalshi price/time series (implied probabilities and volatility).

Candidate models:

- Simple linear/regularized models (Lasso / Elastic Net).  
- Gradient boosting / tree models (CatBoost or similar).  
- Later, structured deep models if warranted.

Output:

- Predicted distribution of TMAX or bracket probabilities.  
- Trade decision and sizing based on price vs model probabilities.

---

## Project Goals

1. **Research & Edge Discovery**
   - Use historical temps, forecasts, and prices to identify persistent mispricings.  
   - Start with simple rules, then evolve toward more complex models.

2. **Robust Backtesting**
   - Use **time-based train/test splits** to avoid look-ahead bias.  
   - Model fees and fills realistically using Kalshi’s bid/ask and volumes. :contentReference[oaicite:13]{index=13}  
   - Explicitly separate **historical forecast data** from **current forecasts**.

3. **Live Trading**
   - Begin with very small positions (e.g., \$5–\$20 per event) and scale only once behaviour is well-understood.  
   - Use WebSockets to respond quickly at market open and around key decision times.  
   - Log all decisions, trades, and P&L into the DB for monitoring.

4. **Modularity & Extensibility**
   - Make it easy to:
     - Add new strategies or swap out forecasting models.  
     - Add new cities or entirely new market types (e.g., rain, snow, ENSO, non-weather event contracts).  
     - Plug in additional data sources while keeping a coherent schema.

---

## Getting Started (High-Level)

Exact commands may vary with environment, but the rough workflow is:

### 1. Prerequisites

- Python 3.11+  
- PostgreSQL  
- Visual Crossing API key :contentReference[oaicite:14]{index=14}  
- Kalshi API credentials (for authenticated endpoints) :contentReference[oaicite:15]{index=15}  

### 2. Environment & Dependencies

- Create a virtualenv and install dependencies:

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
````

* Configure DB connection and API keys via `.env` or config files (see `src/config/` and any example env files).

### 3. Initialize the Database

* Run Alembic migrations:

  ```bash
  alembic upgrade head
  ```

### 4. Ingest Data

Rough sequence (adapt as scripts evolve):

* Backfill Kalshi markets and candles.
* Ingest NOAA / settlement data.
* Backfill Visual Crossing historical obs and forecasts.

### 5. Run a Backtest

Example:

```bash
python -m open_maker.core \
  --strategy open_maker_base \
  --all-cities \
  --days 180
```

Or run Optuna tuning:

```bash
python -m open_maker.optuna_tuner \
  --strategy open_maker_base \
  --trials 100
```

Then inspect:

* Total P&L, daily P&L, Sharpe.
* Per-city breakdowns.
* Parameter configurations that perform best.

### 6. Live / Manual Testing

* Start with `manual_trade.py` in **dry-run** mode to see what the system *would* trade.
* When comfortable:

  * Enable small **real** orders (e.g., $5–$20) via `live_trader.py` or `manual_trade.py`.
  * Monitor `sim.live_orders` and Kalshi account P&L closely.

---

## Documentation & Guidance for Agents

Permanent docs live under `docs/permanent/` (paths may vary slightly):

* **Datetime & API Reference** – explains:

  * How to handle UTC vs local time.
  * The meaning of `event_date`, `basis_date`, `lead_days`, etc.
  * How Visual Crossing and Kalshi timestamps map to each other.

* **File Dictionary Guide** – maps code modules, scripts, and tables to their responsibilities.

* **Strategy Overview / Design Notes** – deeper descriptions of each strategy and future ideas.

Agents and tools (Claude, other LLMs) working in this repo should always:

1. Skim `README.md` (this file).
2. Read `docs/permanent/FILE_DICTIONARY_GUIDE.md`.
3. Read `docs/permanent/DATETIME_AND_API_REFERENCE.md`.
4. Check `AGENT_INSTRUCTIONS.md` or `CLAUDE.md` for project-specific behavioural rules.

This README is meant to stay **high-level and stable**, while the permanent docs and code provide the precise details.
