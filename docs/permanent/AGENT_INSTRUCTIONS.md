# Agent Instructions – Weather Trading Project

This document describes **how an AI coding agent should work in this repository**.  
The goal is to keep the codebase correct, consistent, and extensible while supporting systematic trading on Kalshi weather markets.

---

## 0. Responsibilities and Mindset

As an agent in this repo, you should:

- Treat the existing code and docs as **the source of truth**.
- Make changes **incrementally**, with clear reasoning and minimal breakage.
- Preserve and respect:
  - Time‑based data logic (no leakage).
  - Correct fee/fill modeling.
  - Separation between **historical** vs **current** forecasts.
  - Correct temperature/bracket mapping.

Whenever you’re unsure:

- Prefer **reading existing code + docs** over guessing.
- Prefer **narrow, local changes** over refactors that touch everything.
- Leave TODO comments if you must skip a non‑critical detail, explaining why.

---

## 1. Before You Change Anything

### 1.1 Read the Permanent Docs

Before editing code, always:

1. Read `README.md` (this project’s goals + architecture).
2. Read all files in `docs/permanent/`:
   - Dates & Timezones guide.
   - API usage guide (Visual Crossing, Kalshi, NOAA).
   - Strategy design notes.

These docs contain key invariants, especially around:

- How `basis_date` and `target_date` relate.
- How UTC/local time is handled for each data source.
- Which API endpoints must be used for **historical** vs **current** data.

### 1.2 Understand the Data Model

Skim the ORM models in `src/db/models.py`:

- `wx.settlement`, `wx.minute_obs`, `wx.forecast_snapshot`, `wx.forecast_snapshot_hourly`.
- Kalshi tables: `kalshi_markets`, `candles`/`market_candles`, `trades`, `series`.
- Simulation / logging tables (e.g., `sim.*` as defined).

Confirm table names and columns match the actual schema before writing queries.

---

## 2. General Coding Best Practices for This Repo

### 2.1 Modularity

- Put strategy logic in **strategy classes** in `open_maker/strategies/`.
- Keep **I/O and side effects** (DB, HTTP, WebSocket, CLI) in separate modules:
  - e.g., `core.py`, `live_trader.py`, `manual_trade.py`, `scripts/ingest_*.py`.
- Keep reusable helpers in:
  - `open_maker/utils.py` for trading / forecast / bracket logic.
  - `src/weather/visual_crossing.py` for weather API calls.
  - `src/kalshi/*` for Kalshi clients / schemas.

Do **not** duplicate logic; use a single canonical implementation whenever possible.

### 2.2 Imports and Structure

- Use **absolute imports** within the package (e.g., `from open_maker.utils import ...`).
- Avoid circular imports. If a helper is needed in multiple places, move it into `utils.py` or a small dedicated module.
- Keep each file focused:
  - Strategies → `open_maker/strategies/`.
  - Backtest orchestration → `open_maker/core.py`.
  - Optuna → `open_maker/optuna_tuner.py`.
  - Live trading → `open_maker/live_trader.py`, `open_maker/manual_trade.py`.

### 2.3 Syntax & Type Safety

- Ensure all Python files:
  - Run without syntax errors.
  - Have correct imports (resolve symbols, no unused imports where avoidable).
- Prefer:
  - Type hints for function signatures and dataclasses.
  - Dataclasses for structured results (trades, P&L, params, etc.).
- When adding new functions, *reuse* existing types (e.g., `OpenMakerParams`, `OpenMakerTrade`, `OpenMakerResult`) where reasonable.

---

## 3. Time, Dates, and Timezones

This project is extremely sensitive to dates and times. You **must**:

### 3.1 Terminology

- **`event_date`**: The local date the Kalshi weather market settles on (e.g. “High temp in Chicago on 2025‑11‑28”).
- **`basis_date`**: The date the forecast was **issued** (for historical forecast snapshots).
- **`target_date`**: The date the forecast refers to (usually equal to `event_date` in this project).
- **Local time vs UTC**:
  - Kalshi’s timestamps are mostly in UTC.
  - Weather markets refer to *local calendar days* in each city.
  - Visual Crossing API uses local times in its timeline payload (per the location).

### 3.2 Core Rules

- **Backtests**:
  - Only use **historical forecast snapshots** (`wx.forecast_snapshot` / `_hourly`) with `basis_date <= event_date`.
  - Avoid any use of “current” forecast API in backtests.
  - Use **time‑based train/test splits** (e.g., first 70% of days for training, last 30% for testing).

- **Live trading**:
  - Use **current forecasts** (basis = “today”) from Visual Crossing, via the dedicated current‑forecast method.
  - For a market `event_date = D`, a common pattern is:
    - Use forecast from `basis_date = D-1` or `basis_date = D` (depending on design).
  - Always convert event open time correctly:
    - Market opens at 10:00 local time (per current assumption). Confirm via WebSocket `market_lifecycle` events.

### 3.3 When to Use UTC vs Local

- Use **local time** when reasoning about:
  - Calendar day the weather market refers to.
  - “Midnight” in forecast logic (23:55–00:10 local window).
  - Decision times like “2 hours before predicted high”.

- Use **UTC** when:
  - Working with Kalshi WebSocket timestamps.
  - Logging internal event times.
  - Doing cross‑city synchronization.

Always be explicit: store both `decision_time_local` and `decision_time_utc` where appropriate.

---

## 4. API Usage

### 4.1 Visual Crossing

Use the Timeline API according to the **three separate modes**:

1. **Historical observations**  
   - For settlement & actual temperatures.
   - Called from ingestion scripts that populate `wx.minute_obs` and/or `wx.settlement`.

2. **Historical forecasts**  
   - Use Timeline with `forecastBasisDate` to get “forecast as of basis_date for target_date”.
   - Ingest into `wx.forecast_snapshot` and `_hourly` via `ingest_vc_forecast_history.py`.
   - Only used in backtests.

3. **Current forecast**  
   - Use Timeline **without** `forecastBasisDate`, with city names (e.g., `Chicago,IL`).
   - Used by:
     - `poll_vc_forecast_daemon.py` (nightly snapshots).
     - Live trading helper functions (as a fallback if DB is missing a snapshot).

Never mix these modes in a single helper. Keep methods clearly named and documented.

### 4.2 Kalshi

Use the Kalshi API consistently:

- **REST**:
  - For listing markets, placing orders, and fetching historical trades/candles.
  - Use the **event candlesticks** endpoint for accurate prices:
    - Prices are nested under `yes_ask`, `yes_bid` fields, not flat `price_*`.
- **WebSocket**:
  - Subscribe to market lifecycle channels (e.g., `market_lifecycle` or v2) to detect:
    - Market opening (first tradable state).
    - Market halts / closes (later).
  - Subscribe to price channels only when needed (e.g., intraday strategies).

For trading scripts (`live_trader`, `manual_trade`, etc.):

- Use **rate limiting** as configured.
- Log all orders (even dry‑run) to a `sim.live_orders` or equivalent table.

---

## 5. Strategy Layer

Strategies live under `open_maker/strategies/` and should follow a common pattern:

- A `Params` dataclass (hyperparameters).
- A `Strategy` class with methods like:
  - `decide(context, candles_df=None, obs_data=None) -> TradeDecision`.
- No direct REST or DB calls inside strategy classes; they should work on inputs passed from `core.py`.

Current strategies include:

1. `open_maker_base`
2. `open_maker_next_over`
3. `open_maker_curve_gap`
4. (Reserved) `open_maker_linear_model` / ML strategies

When implementing new strategies:

- Use existing helpers (bracket finder, forecast loaders, fee calculators).
- Keep the entry logic and P&L calculation in `core.py` and/or utility functions.
- Keep strategies **pure**: given context + data → decision object.

---

## 6. Backtesting & Optuna

- Use `open_maker.optuna_tuner.py` as the canonical place for:
  - Defining parameter search spaces.
  - Defining objective functions (e.g., maximize `sharpe_daily`, or total P&L).
  - Implementing train/test split.
- Always:
  - Use **chronological splits** (not random).
  - Report both train and test metrics.
  - Save best parameters to JSON (e.g., `config/open_maker_base_best_params.json`).

When adding new tunable parameters:

- Add them to the strategy’s `Params` dataclass.
- Expose them via CLI flags where relevant.
- Respect reasonable bounds and step sizes.

---

## 7. Live Trading Safety

For any live trading logic:

- Start from **small bet amounts** (e.g., \$5–\$20).
- Assume we should **not move the market**; be conservative about:
  - Order size.
  - Entry prices (don’t assume we always get filled at the best theoretical edge).
- Implement:
  - `--dry-run` for any new live script, defaulting to dry unless explicitly overridden.
  - Logging:
    - Orders submitted.
    - Fills (if available).
    - Daily P&L, win rate, etc.

Never silently place live orders without:

- Clear CLI flags (`--live`, `--bet-amount`).
- Logging to both stdout and DB.

---

## 8. When You Change Something

Whenever you modify or add code:

1. **Run** relevant unit/integration scripts if available.
2. **Add or adjust docstrings** for new functions/methods.
3. **Update docs** in `docs/permanent/` if you change:
   - How dates or forecasts are interpreted.
   - How a strategy decides to enter or exit.
4. Leave a short summary in comments or in a planning doc if the change is non‑trivial.

If you’re unsure whether a change is compatible with the rest of the system, favor:

- Adding a new strategy variant (e.g., `*_v2`) rather than altering the behavior of an existing, validated one.

---

## 9. Non‑Goals (for Now)

The agent should **not**:

- Add new exchanges, asset classes, or non‑weather markets without explicit instructions.
- Replace existing data sources (NOAA, Visual Crossing, Kalshi) with different providers.
- Drastically refactor database schema without:
  - A migration.
  - Updated models.
  - Updated docs.

The focus is **deepening** and **refining** the current weather trading pipeline.

---

If you (the agent) follow these instructions, you will help maintain a clean, reliable, and extensible research and trading system on Kalshi’s weather markets.
