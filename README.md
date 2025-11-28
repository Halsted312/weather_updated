# Weather Edge Trading on Kalshi

## Overview

This repository contains a full research and trading stack for **weather markets on Kalshi**, a regulated event‑based prediction market in the US. Kalshi lists contracts on whether specific events occur (e.g., “Will the high temperature in Chicago on Nov 30 be between 35°F and 36°F?”).   

The core idea of this project:

> Use high‑quality weather data and forecasts to estimate the true probability that each weather bracket will settle YES, then trade mispriced contracts on Kalshi.

The stack includes:

- **Historical observations** (actual temperatures) from NOAA / related sources.   
- **Historical and current forecasts** from Visual Crossing’s weather API.   
- **Kalshi market & candle data** (markets, minute‑level prices, and settlements).
- A set of **trading strategies** and **backtesting tools** (with Optuna tuning).
- **Live trading scripts** that connect to Kalshi via REST + WebSocket.

The long‑term goal is to build a portfolio of **weather‑driven edges** that are:

- Statistically robust (out‑of‑sample positive P&L and decent Sharpe).
- Operationally safe (small size vs market depth, realistic fill assumptions).
- Modular and extensible (easy to add new strategies or markets).

---

## Data Sources

### Kalshi

- **What it is**: A CFTC‑regulated event‑based exchange where users trade contracts that pay \$1 if an event occurs, \$0 otherwise.   
- **What we pull**:
  - Market metadata (series, tickers, strike structure, open/close times).
  - 1‑minute candlesticks (best bid/ask, mid, volume).
  - Settlement outcomes (which bracket actually won).
  - WebSocket events for market lifecycle and live prices.

### Weather Data

We use multiple weather sources for robustness:

1. **NOAA / NCEI / IEM (ground truth)**  
   - Daily **TMAX** (daily high temperature) and sub‑hourly observations.   
   - This defines the *true settlement temperature* for each city/day.

2. **Visual Crossing Weather API (forecasts + history)**   
   - **Historical forecasts** using the Timeline API with `forecastBasisDate` → “What did the model predict on date B for date T?”  
   - **Current forecasts** (today’s best guess for the next ~15 days) using Timeline without `forecastBasisDate`.
   - Hourly forecast curves (for intraday “temperature vs forecast” features).

---

## High‑Level Architecture

Rough structure (names may vary slightly depending on branch):

- `src/db/`
  - SQLAlchemy models for:
    - `wx.settlement` – daily TMAX + Kalshi bucket mapping.
    - `wx.minute_obs` – minute‑level actual temps.
    - `wx.forecast_snapshot` – daily forecast snapshots (target_date, basis_date, lead_days).
    - `wx.forecast_snapshot_hourly` – hourly forecast curves.
    - `kalshi_markets`, `candles`, `market_candles`, etc.

- `src/weather/visual_crossing.py`
  - Client for Visual Crossing Timeline API (historical & current).

- `scripts/`
  - `ingest_vc_forecast_history.py` – backfill historical forecasts.
  - `poll_vc_forecast_daemon.py` – nightly snapshot of current forecasts.
  - `ingest_settlement_multi.py` – multi‑source settlement ingestion.
  - `backfill_kalshi_markets.py`, `backfill_kalshi_candles.py` – Kalshi market + candle backfill.
  - Health checks, migrations, etc.

- `open_maker/`
  - Core strategies + backtests + live trading:
    - `core.py` – common backtest runner + metrics.
    - `utils.py` – bracket selection, fee calc, forecast helpers, etc.
    - `strategies/` – individual strategy classes:
      - `base.py`
      - `next_over.py`
      - `curve_gap.py`
      - (room for ML / linear model strategies)
    - `optuna_tuner.py` – parameter search / train‑test split.
    - `live_trader.py` – automated live trading at market open.
    - `manual_trade.py` – one‑off manual trades using strategy logic.
    - `live_midnight_trader.py` (or similar) – for strategies keyed to forecast updates.

---

## Current Strategies (High Level)

These are the main strategies implemented so far:

1. **`open_maker_base`**
   - When a new weather market opens (e.g., 10:00 local time), look at the Visual Crossing forecast high for that event date.
   - Apply a small bias (e.g., +1°F), map that to the closest temperature bracket, and place a **maker order** (e.g., 30¢) in that bracket.
   - Hold to settlement; no intraday exits. Parameters (entry price, bias, basis_offset_days) are tuned with Optuna.

2. **`open_maker_next_over`**
   - Start with the same base entry.
   - Later (e.g., a few hours before the predicted high), look at prices in the next higher bracket.
   - If that higher bracket is very strong and our bracket price is weak (e.g., our price < 30¢, next bracket > 50¢), exit or shift exposure.

3. **`open_maker_curve_gap`**
   - Start with base entry.
   - At a decision time (e.g., 2–3 hours before the predicted high), compare:
     - Observed temperature (from minute obs) vs forecast at that time.
     - Short‑term observed slope (warming/cooling).
   - If the observed is much hotter/colder than forecast and trending further, “pretend” we had entered in a higher (or lower) bin and adjust the backtest P&L accordingly (hindsight adjustment). This is a stepping stone toward a real intraday shift strategy.

4. **(Planned) ML / Linear Model Strategy**
   - Use a linear / regularized model (Lasso / Elastic Net) or later CatBoost to map features:
     - Forecast path, hourly deviations, humidity, wind, rain/fog flags, etc.
   - To a predicted TMAX distribution and/or implied edge vs Kalshi prices.
   - Use that to decide whether to bet, size, and possibly exit earlier.

All strategies share the same **data plumbing** and **bracket mapping** logic and are tuned via Optuna with **time‑based train/test splits** to avoid look‑ahead bias.

---

## Project Goals

1. **Research & Edge Discovery**
   - Use historical data (temps, forecasts, prices) to quantify when Kalshi weather markets systematically misprice outcomes.
   - Explore simple heuristics first (base, next_over, curve_gap), then more complex models.

2. **Robust Backtesting**
   - Time‑based train/test splits.
   - Realistic fee and fill modelling (using yes_ask / yes_bid from Kalshi’s event candlesticks).
   - Separate historical forecast usage (for backtests) from current forecast usage (for live trades).

3. **Live Trading**
   - Start with **tiny sizes** (e.g., \$5–\$20 per trade) to avoid moving markets.
   - Use WebSockets to detect market open events and place maker orders immediately.
   - Log all live trades to a `sim.live_orders` table for monitoring.

4. **Modular, Extensible System**
   - Easy to add:
     - New strategies (e.g., intraday only, or options on non‑weather markets).
     - New data features (e.g., weather anomalies, ENSO indices, etc.).
     - New exchanges or geographic regions.

---

## How to Use This Repo (Quick Sketch)

This will vary as the code evolves, but the typical workflow is:

1. **Backfill & Ingest Data**
   - Run Kalshi market & candle backfill scripts.
   - Ingest NOAA / IEM / settlement data.
   - Run Visual Crossing historical forecast ingestion.

2. **Run Backtests**
   - `python -m open_maker.core --strategy open_maker_base --all-cities --days 180`
   - `python -m open_maker.optuna_tuner --strategy open_maker_base --trials 100`

3. **Check Metrics**
   - Win rate, P&L, Sharpe (per trade and per day).
   - Per‑city breakdowns.

4. **Live Test (Tiny Size)**
   - Use `manual_trade.py` with `--dry-run` first.
   - Then place small live orders (\$5–\$20) using tuned parameters.

5. **Iterate on Strategies**
   - Adjust heuristics.
   - Add new features or models.
   - Rerun backtests and Optuna with time‑split validation.

---

## Documentation for Agents

Additional permanent docs live under `docs/permanent/` (or similar):

- **Dates & Timezones Guide** – how to handle UTC vs local, basis_date vs target_date, Kalshi timestamps vs weather timestamps.
- **API Reference** – how to call Visual Crossing, Kalshi REST, and WebSockets safely and consistently.
- **Strategy Design Notes** – detailed descriptions for each strategy and future ideas.

Agents working in this repo should always read those docs plus `AGENT_INSTRUCTIONS.md` before making changes.
