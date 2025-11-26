# CLAUDE.md

This file guides Claude Code (claude.ai/code) when working in this repository.

## Project Mission

Build a **reproducible data + backtest pipeline** for Kalshi’s “Highest temperature — Chicago (KXHIGHCHI)” markets that:

1. pulls historical **minute bars** (or trade prints as fallback) and market metadata,
2. joins to **official observed Tmax** for Chicago Midway (NWS/NCEI),
3. stores everything in **PostgreSQL**,
4. runs a baseline **ML model** (Ridge/Lasso) with **probability calibration**,
5. executes a fee-aware backtest optimizing **Sharpe**, and
6. scales to other cities later.

> Weather series, markets, and orderbook snapshots are publicly accessible via Kalshi’s unauthenticated REST endpoints; minute **candlesticks** (1m/60m/1d) are available per market (and per event). Use these for historical minute-level prices; use **trades** as a fallback. ([Kalshi API Documentation][1])
> Chicago weather markets resolve to **Chicago Midway** using NWS climatological products; observed Tmax can be pulled from NCEI’s **Access Data Service** (dataset `daily-summaries`). ([Kalshi][2])

---

## How Claude should work (first reply & workflow)

When I ask for help, **start your first message with**:

> **Plan: We’ll go step by step.**
>
> 1. Verify we can fetch the data (Kalshi + NOAA).
> 2. Stand up Postgres and write idempotent ingestion.
> 3. Validate joins and produce a tiny backtest.
> 4. Expand date range + add tests.
> 5. Train a simple model (Ridge/Lasso), calibrate, trade with fees.
> 6. Report Sharpe/ROI; iterate and extend.

Then proceed to implement Phase 0 → Phase 6 below, committing code in small, reviewable PR-sized steps.

---

## Phased roadmap (with acceptance criteria)

### Phase 0 — Repo scaffolding & environment

* Create `pyproject.toml` (Python ≥3.11) with deps: `requests`, `pandas`, `sqlalchemy`, `psycopg2-binary`, `pydantic`, `scikit-learn`, `pyyaml`, `pytest`.
* Add `docker-compose.yml` for Postgres 16 and `.env.example` with `DB_URL=postgresql://kalshi:kalshi@localhost:5432/kalshi`.
* Create top-level **Makefile** with: `init`, `ingest_demo`, `backtest_demo`, `test`.

**Done when:** `make init` builds containers and tests can connect.

---

### Phase 1 — Kalshi discovery & minute data ingestion

**Endpoints to use (exact):**

* **Get Series:** `GET /series/{series_ticker}` (e.g., `KXHIGHCHI`) — confirms settlement source and metadata. ([Kalshi API Documentation][3])
* **Get Markets:** `GET /markets?series_ticker=KXHIGHCHI&status=closed,settled&min_close_ts=...&max_close_ts=...` — page last ~60 days. Supports cursor pagination and time filters. ([Kalshi API Documentation][4])
* **Get Market:** `GET /markets/{ticker}` — read settlement fields/rules per market. ([Kalshi API Documentation][5])
* **Market Candlesticks (preferred):**
  `GET /series/{series_ticker}/markets/{ticker}/candlesticks?start_ts=..&end_ts=..&period_interval=1` — returns **1-minute** OHLC for `yes_bid`, `yes_ask`, **price** plus `volume`/`open_interest`. ([Kalshi API Documentation][6])
* **Event Candlesticks (optional bulk):**
  `GET /series/{series_ticker}/events/{event_ticker}/candlesticks?start_ts=..&end_ts=..&period_interval=1` — aggregates all markets within an event; use `adjusted_end_ts` to paginate. ([Kalshi API Documentation][7])
* **Trades (fallback for minute bars):**
  `GET /markets/trades?ticker={market_ticker}&limit=1000&min_ts=..&max_ts=..` — page using `cursor`. ([Kalshi API Documentation][8])
* **Orderbook snapshot (FYI only):** `GET /markets/{ticker}/orderbook` — snapshot **bids** only; asks are implied via 100¢ symmetry (YES ask = 100 − best NO bid). We won’t rely on snapshots historically. ([Kalshi API Documentation][9])

**Implement:**

* `kalshi/client.py` with thin `requests` wrappers + pagination helpers.
* `ingest/markets.py` to fetch series, markets, and **minute candlesticks** for last 60 days.
* Fallback path: if candlesticks unavailable, aggregate **trades → 1-minute OHLC/VWAP**.

**Done when:** parquet/CSV previews exist and row counts match expected minutes per market.

---

### Phase 2 — Database schema & loaders

**Tables (PostgreSQL):**

* `series(series_ticker PK, title, category, frequency, settlement_source_json, raw_json)`
* `markets(ticker PK, series_ticker FK, title, event_ticker, open_time, close_time, expiration_time, status, settlement_value, result, rules_text, raw_json)`
* `candles(market_ticker FK, end_period_ts, period_minutes, yes_bid_open, yes_bid_high, yes_bid_low, yes_bid_close, yes_ask_open, yes_ask_high, yes_ask_low, yes_ask_close, price_open, price_high, price_low, price_close, volume, open_interest, PRIMARY KEY(market_ticker,end_period_ts,period_minutes))`
* `trades(trade_id PK, market_ticker, price, count, taker_side, created_time)`
* `weather_observed(station_id, date, tmax_f, tmax_c, source, raw_json, PRIMARY KEY(station_id,date))`

**Implement:**

* `db/models.py` (SQLAlchemy), `db/loaders.py` with **idempotent upserts**.
* `ingest/load_to_db.py` that populates `series`, `markets`, `candles` (or `trades`).

**Done when:** `make ingest_demo` populates DB for a 7-day window.

---

### Phase 3 — NOAA observed Tmax (ground truth)

**Primary API:** NCEI **Access Data Service** (no token), dataset `daily-summaries`.
Example (JSON, °F):
`/access/services/data/v1?dataset=daily-summaries&stations=GHCND:USW00014819&startDate=2025-09-01&endDate=2025-11-10&dataTypes=TMAX&units=standard&format=json` ([NCEI][10])

* Station: **Chicago Midway = `GHCND:USW00014819`**. ([NCEI][11])
* (Alt.) CDO v2 also provides `TMAX` with a token (`5 req/s`, `10k/day`) if needed. ([NCEI][12])

**Implement:**

* `weather/noaa_ads.py` pulls `[date, TMAX]` for the ingestion window and writes `weather_observed`.

**Done when:** each market date maps to a single `tmax_f` for Midway.

---

### Phase 4 — Fee-aware backtest harness

Use **minute candlesticks** where available; else trade prints aggregated to 1-minute.

**Fees (per contract):**

* **Taker:** `ceil_to_cent(0.07 * C * P * (1 - P))`
* **Maker:** `ceil_to_cent(0.0175 * C * P * (1 - P))`
* **No settlement fee.** ([Kalshi][13])

**Assumptions:**

* Default fills **cross the spread** (buy at `yes_ask_close`, sell at `yes_bid_close` of that minute) with optional `slippage_cents`.
* Liquidity guardrails: skip minutes with volume below a threshold.

**Implement:**

* `fees.py`, `backtest/portfolio.py` (single-market simulator), `backtest/run_backtest.py` with CLI flags.

**Done when:** we can run a 60-day Chicago backtest and output P&L, Sharpe, max DD.

---

### Phase 5 — Modeling & calibration

Start **simple**, then iterate.

**Baseline model:** Ridge or Lasso (scikit-learn) per bracket to estimate `Pr(YES)` using features:

* price signals: mid, last, momentum (1/5/15m), time-to-close, volume, OI, cross-bin probability consistency;
* weather signals: prior-day Tmax, **observed forecast** proxy (optional later), day-of-week.

**Calibration:** Platt scaling or isotonic on a rolling validation; compute Brier/LogLoss; enforce bracket monotonicity.

**Execution rule:** trade only when **expected edge net of fees** > threshold; size via **fractional Kelly** cap (e.g., 0.25×). Optimize **Sharpe** on walk-forward.

**Done when:** calibrated model improves Brier and Sharpe vs price-only baseline.

---

### Phase 6 — Scale out & docs

* Add cities (NYC: `KXHIGHNY`, Miami, Austin), reuse pipeline.
* Produce `reports/` with daily metrics and a README explaining how to extend.

---

## Code layout (target)

```
/kalshi
  client.py
  schemas.py
/ingest
  markets.py
  load_to_db.py
/weather
  noaa_ads.py
/db
  models.py
  loaders.py
/backtest
  fees.py
  portfolio.py
  run_backtest.py
/models
  features.py
  train_baseline.py
  calibrate.py
/tests
  test_endpoints.py
  test_loaders.py
  test_backtest.py
Makefile
docker-compose.yml
.env.example
```

---

## Concrete API snippets (to implement)

### Kalshi (curl examples)

* Series:

  ```bash
  curl -s "https://api.elections.kalshi.com/trade-api/v2/series/KXHIGHCHI"
  ```

  ([Kalshi API Documentation][3])

* Markets (last 60 days; fill `min_close_ts`/`max_close_ts` as UNIX seconds):

  ```bash
  curl -s "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXHIGHCHI&status=closed,settled&min_close_ts=...&max_close_ts=..."
  ```

  ([Kalshi API Documentation][4])

* Market candlesticks (1-minute):

  ```bash
  curl -s "https://api.elections.kalshi.com/trade-api/v2/series/KXHIGHCHI/markets/{TICKER}/candlesticks?start_ts=...&end_ts=...&period_interval=1"
  ```

  ([Kalshi API Documentation][6])

* Trades (fallback):

  ```bash
  curl -s "https://api.elections.kalshi.com/trade-api/v2/markets/trades?ticker={TICKER}&limit=1000&min_ts=...&max_ts=..."
  ```

  ([Kalshi API Documentation][8])

* Orderbook snapshot (for debugging only):

  ```bash
  curl -s "https://api.elections.kalshi.com/trade-api/v2/markets/{TICKER}/orderbook"
  ```

  *(Bids only; asks are implied as `100 - best_other_side`.)* ([Kalshi API Documentation][9])

### NOAA (observed Tmax @ Midway)

```bash
curl -s "https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=GHCND:USW00014819&startDate=2025-09-01&endDate=2025-11-10&dataTypes=TMAX&units=standard&format=json"
```

([NCEI][10])

---

## Database notes

* Use **UTC timestamps**; store Kalshi minute bars in **cents** (ints), keep `*_dollars` only for derived views.
* Enforce **idempotency**: upsert on `(market_ticker, end_period_ts, period_minutes)`.
* Index `candles(market_ticker, end_period_ts)` and `trades(market_ticker, created_time)`.

---

## Backtest requirements

* Inputs: per-minute `yes_bid_*`, `yes_ask_*`, `price_*`, `volume`, `open_interest`; settlement value from `markets`.
* Fees: implement maker/taker per schedule; round **up** to next cent. ([Kalshi][13])
* Outputs: CSV with per-trade P&L, cumulative equity; summary with ROI, **Sharpe**, max DD, Brier/LogLoss.

---

## Testing checklist

* **Endpoint smoke tests**: series → markets (pagination) → candlesticks/trades for 1–2 markets. ([Kalshi API Documentation][14])
* **Schema tests**: PK/UK constraints, upsert semantics.
* **NOAA join**: every market date has exactly one `tmax_f`.
* **Backtest invariants**: zero positions → zero P&L; taker/maker fee unit tests.

---

## Guardrails & assumptions

* Prefer **candlesticks** over trades for minute bars; fall back to trades aggregation only if required. ([Kalshi API Documentation][6])
* Treat orderbook snapshots as informational; no attempt to reconstruct historical spreads without WebSocket archives. ([Kalshi API Documentation][9])
* Chicago markets settle to **Midway**; ensure date alignment with NWS/NCEI daily period. ([Kalshi][2])

---

## References

* Kalshi Market Data Quick Start (public endpoints, series/markets/orderbook). ([Kalshi API Documentation][1])
* **Get Market Candlesticks** (1/60/1440 min; OHLC for yes_bid/yes_ask/price + volume/OI). ([Kalshi API Documentation][6])
* **Get Event Candlesticks** (bulk per event). ([Kalshi API Documentation][7])
* **Get Trades** (ticker, min/max ts, cursor pagination). ([Kalshi API Documentation][8])
* **Orderbook Responses** (bids only; asks implied). ([Kalshi API Documentation][9])
* **Get Market** (settlement fields/rules). ([Kalshi API Documentation][5])
* NCEI **Access Data Service** (dataset=`daily-summaries`, params: `stations`, `startDate`, `endDate`, `dataTypes=TMAX`, `units`, `format`). ([NCEI][10])
* Midway station id **GHCND:USW00014819**. ([NCEI][11])
* Kalshi **Fee Schedule** (taker/maker formulas; no settlement fee). ([Kalshi][13])

---

### What to build first (summary)

1. Minimal Kalshi/NOAA fetchers (+ tiny 3-day sample).
2. Postgres schema + idempotent loaders.
3. Tiny backtest with fees over the sample.
4. Expand to 60 days; add Ridge/Lasso + calibration; report Sharpe.

[1]: https://docs.kalshi.com/getting_started/quick_start_market_data "Quick Start: Market Data - API Documentation"
[2]: https://kalshi.com/markets/kxhighchi/highest-temperature-in-chicago?utm_source=chatgpt.com "Highest temperature in Chicago today?"
[3]: https://docs.kalshi.com/api-reference/market/get-series?utm_source=chatgpt.com "Get Series - API Documentation"
[4]: https://docs.kalshi.com/python-sdk/api/MarketsApi "Markets - API Documentation"
[5]: https://docs.kalshi.com/api-reference/market/get-market?utm_source=chatgpt.com "Get Market - API Documentation"
[6]: https://docs.kalshi.com/api-reference/market/get-market-candlesticks "Get Market Candlesticks - API Documentation"
[7]: https://docs.kalshi.com/api-reference/events/get-event-candlesticks "Get Event Candlesticks - API Documentation"
[8]: https://docs.kalshi.com/api-reference/market/get-trades "Get Trades - API Documentation"
[9]: https://docs.kalshi.com/getting_started/orderbook_responses "Orderbook Responses - API Documentation"
[10]: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation "NCEI Data Service API User Documentation | National Centers for Environmental Information (NCEI)"
[11]: https://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND%3AUSW00014819/detail?utm_source=chatgpt.com "Daily Summaries Station Details: CHICAGO MIDWAY ..."
[12]: https://www.ncdc.noaa.gov/cdo-web/webservices/getstarted?utm_source=chatgpt.com "Climate Data Online: Web Services Documentation"
[13]: https://kalshi.com/docs/kalshi-fee-schedule.pdf?utm_source=chatgpt.com "Fee Schedule for Oct 2025"
[14]: https://docs.kalshi.com/getting_started/pagination?utm_source=chatgpt.com "Understanding Pagination - API Documentation"
