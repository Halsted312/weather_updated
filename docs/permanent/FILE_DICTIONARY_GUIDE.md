# File Dictionary Guide

A comprehensive guide to all Python files in the Kalshi Weather Trading project.

**Last Updated:** 2025-11-27

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [open_maker/ - Maker Strategy Module](#open_maker---maker-strategy-module)
4. [backtest/ - Backtesting Framework](#backtest---backtesting-framework)
5. [src/ - Core Infrastructure](#src---core-infrastructure)
6. [scripts/ - Data Ingestion & Daemons](#scripts---data-ingestion--daemons)
7. [Summary Table](#summary-table)
8. [Refactoring Recommendations](#refactoring-recommendations)

---

## Project Overview

This is a Python weather prediction trading system for Kalshi markets. It combines weather forecasting APIs with Kalshi market data to execute algorithmic trading strategies.

**Main Components:**
- **open_maker/** - Maker limit order execution strategies at market open
- **backtest/** - Backtesting framework for midnight heuristic strategy
- **src/** - Core infrastructure (DB models, API clients, weather data, config)
- **scripts/** - Data ingestion and live trading daemons

---

## Directory Structure

```
weather_updated/
├── open_maker/           # Maker strategy module
│   ├── strategies/       # Strategy implementations
│   ├── core.py          # Main backtest engine (1278 lines)
│   ├── utils.py         # Utility functions
│   ├── live_trader.py   # Live WebSocket trader
│   ├── manual_trade.py  # Manual order placement
│   ├── market_open_listener.py  # Event listener
│   └── optuna_tuner.py  # Parameter optimization
├── backtest/            # Midnight heuristic backtester
│   ├── midnight_heuristic.py
│   └── optuna_tuning.py
├── src/                 # Core infrastructure
│   ├── config/          # Settings and city configs
│   ├── db/              # Database models and connection
│   ├── kalshi/          # Kalshi API client
│   ├── weather/         # Weather API clients
│   └── utils/           # Rate limiting, retries
├── scripts/             # Ingestion and daemons
│   ├── ingest_*.py      # Data ingestion scripts
│   ├── backfill_*.py    # Historical data backfill
│   ├── poll_*.py        # Polling daemons
│   └── kalshi_ws_*.py   # WebSocket recorders
├── config/              # Tuned parameters (JSON)
└── docs/                # Documentation
```

---

## open_maker/ - Maker Strategy Module

### `core.py` (1278 lines) - REFACTORING CANDIDATE

**Purpose:** Main backtesting engine for open-maker strategies. Unified runner handling entry setup, strategy decisions, and P&L calculation for three strategies.

**Classes:**
| Class | Description |
|-------|-------------|
| `OpenMakerParams` | Strategy parameters (entry_price_cents, temp_bias_deg, basis_offset_days, bet_amount_usd) |
| `OpenMakerTrade` | Single trade record with entry, exit, and P&L info |
| `OpenMakerResult` | Backtest results container with metrics |

**Functions:**
| Function | Description |
|----------|-------------|
| `load_tuned_params(strategy_id, bet_amount_usd)` | Load optimized params from JSON files |
| `load_forecast_data(session, city, start_date, end_date)` | Query WxForecastSnapshot |
| `load_settlement_data(session, city, start_date, end_date)` | Query WxSettlement for actual temps |
| `load_market_data(session, city, start_date, end_date)` | Query KalshiMarket for brackets |
| `load_candle_data(session, tickers, start_time, end_time)` | Query KalshiCandle1m |
| `get_forecast_at_open(forecast_df, event_date, basis_offset_days)` | Retrieve forecast with fallback |
| `run_strategy(strategy_id, cities, start_date, end_date, params, fill_check)` | Main unified backtest runner (400+ lines) |
| `run_backtest(...)` | Legacy wrapper for base strategy |
| `run_backtest_next_over(...)` | Legacy wrapper for next_over strategy |
| `save_results_to_db(result)` | Persist to sim.sim_run and sim.sim_trade |
| `print_results(result)` | Pretty-print backtest summary |
| `print_debug_trades(result, n)` | Print sample trades for debugging |
| `print_comparison_table(results)` | Compare multiple strategies |
| `main()` | CLI entry point |

---

### `utils.py` (852 lines)

**Purpose:** Self-contained utilities for open-maker. Fee calculations, bracket selection, position sizing, observation/forecast loading.

**Functions:**
| Function | Description |
|----------|-------------|
| `get_city_timezone(city)` | Map city ID to ZoneInfo timezone |
| `kalshi_taker_fee(price_cents, num_contracts)` | Calculate taker fee: 0.07 * price * (100-price) / 100 |
| `kalshi_maker_fee(price_cents, num_contracts)` | Returns 0.0 (weather markets have no maker fees) |
| `find_bracket_for_temp(markets_df, event_date, temp, round_for_trading)` | Find winning bracket for temperature |
| `determine_winning_bracket(markets_df, event_date, tmax_final)` | Wrapper for settlement determination |
| `calculate_position_size(entry_price_cents, bet_amount_usd)` | Return (num_contracts, actual_cost_usd) |
| `calculate_pnl(entry_price_cents, num_contracts, bin_won, fee_usd)` | Net P&L for held trade |
| `calculate_exit_pnl(entry_price, exit_price, num_contracts, exit_fee)` | P&L for early exit |
| `get_predicted_high_hour(session, city, event_date)` | Query hourly forecast for max temp hour |
| `compute_decision_time_utc(city, event_date, predicted_high_hour, offset)` | Compute decision time in UTC |
| `get_candle_price_for_exit(candles_df, ticker, decision_time, window)` | Exit price with fallback: bid > mid > close |
| `get_candle_price_for_signal(candles_df, ticker, decision_time, window)` | Signal price: bid > close |
| `get_bracket_index(markets_df, event_date, ticker)` | Get bracket position in sorted order |
| `get_neighbor_ticker(sorted_brackets, current_index, direction)` | Get adjacent bracket ticker |
| `load_minute_obs(session, city, start_time, end_time)` | Load WxMinuteObs for curve_gap |
| `get_forecast_temp_at_time(session, city, event_date, target_time, basis_date)` | Interpolate hourly forecast |
| `compute_obs_stats(obs_df, decision_time, avg_window, slope_window)` | Compute T_obs and slope_1h |
| `check_fill_achievable(session, ticker, listed_at, entry_price, window)` | Verify entry price was achievable |

---

### `market_open_listener.py` (370 lines)

**Purpose:** Read-only WebSocket listener for market lifecycle events. Logs "open" state transitions to sim.market_open_log.

**Classes:**
| Class | Description |
|-------|-------------|
| `MarketOpenListener` | WebSocket listener with subscription management and deduplication |

**Functions:**
| Function | Description |
|----------|-------------|
| `get_ws_url(base_url)` | Convert REST URL to WebSocket URL |
| `sign_pss_text(private_key, message)` | Sign with RSA-PSS SHA256 |
| `create_ws_auth_headers(api_key, private_key)` | Create WebSocket auth headers |
| `parse_event_ticker(event_ticker)` | Parse date from ticker (KXHIGHCHI-25NOV28 -> 2025-11-28) |
| `signal_handler(signum, frame)` | Handle graceful shutdown |
| `_build_subscription()` | Build market_lifecycle subscription |
| `_log_market_open(data)` | Log event with deduplication |
| `_handle_message(message)` | Process WebSocket messages |
| `connect_and_listen()` | Main loop with exponential backoff (1s to 60s) |

---

### `optuna_tuner.py` (672 lines)

**Purpose:** Optuna parameter optimization for three strategies with train/test split.

**Functions:**
| Function | Description |
|----------|-------------|
| `save_best_params(strategy_id, params, metric, best_value)` | Save to config/{strategy_id}_best_params.json |
| `load_best_params(strategy_id)` | Load saved parameters |
| `create_objective_base(...)` | Objective for base strategy |
| `create_objective_next_over(...)` | Objective for next_over strategy |
| `create_objective_curve_gap(...)` | Objective for curve_gap strategy |
| `_extract_metric(result, metric)` | Extract metric from backtest result |
| `run_optimization(...)` | Main Optuna optimization with TPE sampler |
| `print_optimization_results(study, strategy_id)` | Print best trial and importance |

---

### `live_trader.py` (604 lines)

**Purpose:** Live WebSocket trader executing open-maker base strategy in real-time.

**Classes:**
| Class | Description |
|-------|-------------|
| `LiveOrder` | Dataclass tracking order placement |
| `LiveOpenMakerTrader` | Live trader with WebSocket and REST |

**Functions:**
| Function | Description |
|----------|-------------|
| `_get_forecast_at_open(session, city, event_date)` | Forecast with flexible fallback |
| `_load_market_data(session, city, event_date)` | Query KalshiMarket |
| `_handle_market_open(data)` | Process open event with deduplication |
| `_run_strategy(city, event_date, market_ticker)` | Execute strategy logic |
| `_place_order(...)` | Place maker limit order |
| `_log_order(order)` | Log to sim.live_orders |
| `connect_and_trade()` | Main connection loop |

---

### `manual_trade.py` (~654 lines)

**Purpose:** Manual trade placement for testing without WebSocket events.

**Functions:**
| Function | Description |
|----------|-------------|
| `get_forecast_at_open(session, city, event_date, basis_offset)` | Get forecast for trading |
| `load_market_data(session, city, event_date)` | Load market brackets |
| `fetch_markets_from_api(client, city, event_date)` | API fallback for markets |
| `place_order(client, ticker, num_contracts, price_cents, city, event_date)` | Place maker limit order |
| `log_order_to_db(session, trade_result)` | Log to sim.live_orders |
| `get_forecast_flexible(session, city, event_date, preferred_basis_offset)` | Flexible forecast lookup |
| `run_manual_trade(city, event_date, params, dry_run, client, manual_fcst)` | Execute manual trade logic |

---

### `strategies/__init__.py` (56 lines)

**Purpose:** Strategy registry for dynamic loading.

**Functions:**
| Function | Description |
|----------|-------------|
| `STRATEGY_REGISTRY` | Dict mapping strategy_id -> (class, params_class) |
| `register_strategy(strategy_id, strategy_class, params_class)` | Register strategy |
| `get_strategy(strategy_id)` | Retrieve strategy by ID |
| `list_strategies()` | List available strategies |

---

### `strategies/base.py` (132 lines)

**Purpose:** Base strategy class and simple buy-and-hold implementation.

**Classes:**
| Class | Description |
|-------|-------------|
| `StrategyParamsBase` | Abstract base parameters |
| `OpenMakerParams` | Base strategy parameters |
| `TradeContext` | Context passed to decide() |
| `TradeDecision` | Decision output (action, exit_price, reason) |
| `StrategyBase` | Abstract base with decide() method |
| `BaseStrategy` | Always hold to settlement |

---

### `strategies/next_over.py` (134 lines)

**Purpose:** Exit early if neighbor bracket is rich while ours is cheap.

**Classes:**
| Class | Description |
|-------|-------------|
| `NextOverParams` | Parameters (decision_offset_min, neighbor_price_min_c, our_price_max_c) |
| `NextOverStrategy` | Exit logic implementation |

---

### `strategies/curve_gap.py` (166 lines)

**Purpose:** Shift bracket selection based on observation vs forecast curve.

**Classes:**
| Class | Description |
|-------|-------------|
| `CurveGapParams` | Parameters (delta_obs_fcst_min_deg, slope_min_deg_per_hour, max_shift_bins) |
| `CurveGapDecision` | Decision with override_bin_index |
| `CurveGapStrategy` | Shift logic implementation |

---

## backtest/ - Backtesting Framework

### `midnight_heuristic.py` (781 lines)

**Purpose:** Midnight-based heuristic strategy. Makes entry decisions at midnight using 3-day trend, adjusts at T-2h before predicted high.

**Classes:**
| Class | Description |
|-------|-------------|
| `HeuristicParams` | Parameters (alpha, beta, gamma, delta, edge_threshold, split_threshold) |
| `Trade` | Single trade record |
| `BacktestResult` | Results container |

**Functions:**
| Function | Description |
|----------|-------------|
| `kalshi_taker_fee(price_cents, num_contracts)` | Calculate taker fee |
| `kalshi_maker_fee(price_cents, num_contracts)` | Returns 0.0 |
| `run_backtest(cities, start_date, end_date, params, strategy_name)` | Main backtest runner |
| `save_results_to_db(result)` | Persist to sim schema |
| `print_results(result)` | Pretty-print summary |

---

### `optuna_tuning.py` (324 lines)

**Purpose:** Optuna optimization for midnight heuristic strategy.

**Functions:**
| Function | Description |
|----------|-------------|
| `create_objective(cities, start_date, end_date, metric)` | Create Optuna objective |
| `run_optimization(...)` | Run Optuna study |

---

## src/ - Core Infrastructure

### `config/settings.py`

**Purpose:** Load settings from environment (Kalshi API, Visual Crossing API, database URL).

### `config/cities.py`

**Purpose:** City configuration with timezone and series ticker mapping.

| City | Series Ticker | Timezone |
|------|---------------|----------|
| chicago | KXHIGHCHI | America/Chicago |
| austin | KXHIGHAUS | America/Chicago |
| denver | KXHIGHDEN | America/Denver |
| los_angeles | KXHIGHLA | America/Los_Angeles |
| miami | KXHIGHMIA | America/New_York |
| philadelphia | KXHIGHPHIL | America/New_York |

---

### `db/models.py`

**Purpose:** SQLAlchemy ORM models for three schemas.

**wx Schema (Weather):**
| Model | Description |
|-------|-------------|
| `WxSettlement` | Actual daily max temps from NWS CLI, CF6, IEM, NCEI |
| `WxMinuteObs` | 5-minute weather observations |
| `WxForecastSnapshot` | Daily max temp forecasts (basis_date, lead_days) |
| `WxForecastSnapshotHourly` | Hourly forecasts for intraday strategies |

**kalshi Schema (Markets):**
| Model | Description |
|-------|-------------|
| `KalshiMarket` | Market metadata (ticker, strike_type, floor/cap_strike) |
| `KalshiCandle1m` | 1-minute candlesticks (close_c, yes_bid_c, yes_ask_c) |

**sim Schema (Simulation):**
| Model | Description |
|-------|-------------|
| `SimRun` | Backtest run summary |
| `SimTrade` | Individual trades |

### `db/connection.py`

**Purpose:** Database session management with SQLAlchemy.

### `db/checkpoint.py`

**Purpose:** Checkpoint management for resumable ingestion.

---

### `kalshi/client.py`

**Purpose:** Kalshi API client with RSA-PSS auth and adaptive rate limiting.

**Classes:**
| Class | Description |
|-------|-------------|
| `KalshiClient` | REST API client with auth and rate limiting |

**Methods:**
| Method | Description |
|--------|-------------|
| `get_markets(series_ticker, status, limit)` | Fetch markets for series |
| `create_order(ticker, side, action, count, order_type, yes_price)` | Place order |

### `kalshi/schemas.py`

**Purpose:** Pydantic schemas for API responses.

---

### `weather/visual_crossing.py`

**Purpose:** Visual Crossing Timeline API client.

**Classes:**
| Class | Description |
|-------|-------------|
| `VisualCrossingClient` | Client for weather data |

**Methods:**
| Method | Description |
|--------|-------------|
| `fetch_minutes(location, start_date, end_date)` | Fetch minute-level data |
| `fetch_historical_daily_forecast(location, basis_date, horizon_days)` | Fetch historical forecast |
| `fetch_historical_hourly_forecast(location, basis_date, horizon_hours)` | Fetch hourly forecast |

### `weather/nws_cli.py`, `nws_cf6.py`, `iem_cli.py`, `noaa_ncei.py`

**Purpose:** Multi-source weather API clients for settlement verification.

---

### `utils/rate_limiter.py`

**Purpose:** Adaptive rate limiting with exponential backoff on 429 responses.

### `utils/retry.py`

**Purpose:** Retry decorators with exponential backoff.

---

## scripts/ - Data Ingestion & Daemons

### Data Ingestion

| Script | Lines | Description |
|--------|-------|-------------|
| `ingest_vc_minutes.py` | 310 | Ingest Visual Crossing 5-min observations to wx.minute_obs |
| `ingest_nws_settlement.py` | 355 | Ingest NWS settlement data to wx.settlement |
| `ingest_settlement_multi.py` | 327 | Multi-source settlement with fallback logic |
| `ingest_vc_forecast_history.py` | 376 | Ingest historical forecast snapshots |
| `ingest_vc_forecast_hourly.py` | 420 | Ingest hourly forecast data |

### Kalshi Data

| Script | Lines | Description |
|--------|-------|-------------|
| `backfill_kalshi_markets.py` | 394 | Backfill market metadata (tickers, strikes) |
| `backfill_kalshi_candles.py` | 645 | Backfill 1-min candles with checkpointing |

### Daemons

| Script | Lines | Description |
|--------|-------|-------------|
| `poll_vc_forecast_daemon.py` | 735 | Continuous forecast polling (midnight snapshot window) |
| `kalshi_ws_recorder.py` | 553 | WebSocket recorder for candles, trades, lifecycle |
| `live_midnight_trader.py` | 435 | Live trader for midnight heuristic strategy |

### Monitoring

| Script | Lines | Description |
|--------|-------|-------------|
| `check_data_state.py` | 202 | Check ingestion state across cities |
| `check_data_freshness.py` | 379 | Monitor data freshness and gaps |

---

## Summary Table

| Directory | File | Lines | Primary Responsibility |
|-----------|------|-------|------------------------|
| open_maker/ | `core.py` | 1278 | Backtest engine (3 strategies) |
| | `utils.py` | 852 | Bracket selection, fees, position sizing |
| | `live_trader.py` | 604 | Live WebSocket trading |
| | `optuna_tuner.py` | 672 | Parameter optimization |
| | `market_open_listener.py` | 370 | Event monitoring |
| | `manual_trade.py` | 654 | Manual order testing |
| open_maker/strategies/ | `base.py` | 132 | Base strategy (hold to settlement) |
| | `next_over.py` | 134 | Exit on neighbor price signal |
| | `curve_gap.py` | 166 | Shift on obs vs forecast gap |
| backtest/ | `midnight_heuristic.py` | 781 | Midnight strategy backtester |
| | `optuna_tuning.py` | 324 | Midnight param optimization |
| scripts/ | Various | ~4500 | Data ingestion, daemons, monitoring |
| src/ | Various | ~1500 | Core infrastructure |

**Total estimated lines:** ~11,000+ lines of Python

---

## Refactoring Recommendations

The following suggestions aim to improve maintainability, reduce code duplication, and make the codebase more modular.

### 1. Split `core.py` (1278 lines) - HIGH PRIORITY

**Problem:** The `run_strategy()` function is 400+ lines with nested conditionals for three different strategies.

**Recommendation:** Extract into separate modules:

```
open_maker/
├── core.py              # Keep: OpenMakerParams, OpenMakerTrade, OpenMakerResult, data loaders
├── runners/
│   ├── __init__.py
│   ├── base_runner.py       # run_strategy_base() - ~100 lines
│   ├── next_over_runner.py  # run_strategy_next_over() - ~150 lines
│   └── curve_gap_runner.py  # run_strategy_curve_gap() - ~150 lines
└── reporting.py         # print_results(), print_debug_trades(), print_comparison_table()
```

**Benefits:**
- Each strategy runner is self-contained and testable
- Easier to add new strategies
- `core.py` becomes a thin coordinator

---

### 2. Create Shared Utilities Module - MEDIUM PRIORITY

**Problem:** Code duplication between `backtest/midnight_heuristic.py` and `open_maker/core.py`:
- Both implement bracket selection
- Both implement fee calculations
- Both implement P&L calculations

**Recommendation:** Extract common utilities to `src/trading/`:

```
src/
└── trading/
    ├── __init__.py
    ├── fees.py          # kalshi_taker_fee(), kalshi_maker_fee()
    ├── brackets.py      # find_bracket_for_temp(), determine_winning_bracket()
    ├── position.py      # calculate_position_size(), calculate_pnl()
    └── timezones.py     # get_city_timezone(), compute_decision_time_utc()
```

**Benefits:**
- Single source of truth for fee calculations
- Consistent bracket logic across strategies
- Easier to test and maintain

---

### 3. Consolidate Optuna Tuning - LOW PRIORITY

**Problem:** `backtest/optuna_tuning.py` and `open_maker/optuna_tuner.py` share similar patterns.

**Recommendation:** Create base optimization framework:

```
src/
└── optimization/
    ├── __init__.py
    ├── base_optimizer.py    # Common optimization logic, param saving/loading
    └── metrics.py           # _extract_metric(), Sharpe calculations
```

**Benefits:**
- Consistent optimization patterns
- Easier to add new strategies to tuning

---

### 4. Unify Weather API Clients - LOW PRIORITY

**Problem:** Multiple weather source clients (`nws_cli.py`, `nws_cf6.py`, `iem_cli.py`, `noaa_ncei.py`) with similar patterns.

**Recommendation:** Create abstract base class:

```
src/weather/
├── __init__.py
├── base.py              # WeatherSourceBase abstract class
├── visual_crossing.py   # Extends WeatherSourceBase
├── nws/
│   ├── __init__.py
│   ├── cli.py          # NWS CLI source
│   └── cf6.py          # NWS CF6 source
└── noaa/
    ├── __init__.py
    ├── iem.py          # IEM source
    └── ncei.py         # NCEI source
```

**Benefits:**
- Consistent interface for all weather sources
- Easier to add new sources
- Better testing via mock implementations

---

### 5. Consolidate Ingestion Scripts - LOW PRIORITY

**Problem:** Five forecast-related ingestion scripts with similar patterns.

**Recommendation:** Create configurable pipeline:

```
scripts/
└── ingest/
    ├── __init__.py
    ├── base_ingestor.py     # Common checkpoint, retry, logging
    ├── forecast_ingestor.py # Configurable forecast ingestion
    └── settlement_ingestor.py # Multi-source settlement
```

Run with flags:
```bash
python -m scripts.ingest.forecast_ingestor --source visualcrossing --type daily
python -m scripts.ingest.forecast_ingestor --source visualcrossing --type hourly
```

---

### 6. Proposed Final Structure

```
weather_updated/
├── open_maker/
│   ├── strategies/          # Strategy implementations (keep as-is)
│   ├── runners/             # NEW: Strategy-specific runners
│   ├── core.py              # Reduced to ~400 lines
│   ├── utils.py             # Keep, but move some to src/trading/
│   ├── live_trader.py
│   ├── manual_trade.py
│   └── optuna_tuner.py
├── backtest/
│   ├── midnight_heuristic.py  # Reduce by using src/trading/
│   └── optuna_tuning.py
├── src/
│   ├── config/
│   ├── db/
│   ├── kalshi/
│   ├── weather/              # Reorganized with base class
│   ├── trading/              # NEW: Shared trading utilities
│   ├── optimization/         # NEW: Shared optimization
│   └── utils/
├── scripts/
│   ├── ingest/               # NEW: Consolidated ingestion
│   ├── daemons/              # poll_*, kalshi_ws_*, live_*
│   └── monitoring/           # check_*
└── config/                   # Tuned parameters
```

---

### Priority Summary

| Priority | Recommendation | Effort | Impact |
|----------|----------------|--------|--------|
| HIGH | Split core.py into runners/ | Medium | High - improves maintainability |
| MEDIUM | Create src/trading/ shared utils | Medium | High - reduces duplication |
| LOW | Consolidate Optuna tuning | Low | Medium - cleaner patterns |
| LOW | Unify weather API clients | Medium | Low - mostly cosmetic |
| LOW | Consolidate ingestion scripts | Medium | Low - rarely modified |

---

*This guide should be updated when significant structural changes are made to the codebase.*
