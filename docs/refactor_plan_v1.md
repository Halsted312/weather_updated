# Refactor Plan v1: Split open_maker/core.py and utils.py

**Created:** 2025-11-27
**Status:** In Progress
**Phase:** 1 of 4

---

## Summary

This document details the refactoring of the two largest Python files in the `open_maker/` module:

| File | Current Lines | Target Lines | Reduction |
|------|---------------|--------------|-----------|
| `core.py` | 1,277 | ~250 | -80% |
| `utils.py` | 851 | ~50 (package `__init__`) | -94% |

The goal is to improve maintainability, testability, and separation of concerns without changing any strategy behavior.

---

## Phase 1: Split open_maker/core.py

### Target Module Structure

```
open_maker/
├── core.py              # Thin CLI + re-exports (~250 lines)
├── core_runner.py       # Strategy execution (~400 lines)
├── data_loading.py      # DB queries + JSON loading (~180 lines)
├── reporting.py         # Output + persistence (~130 lines)
├── ... (other existing files unchanged)
```

### Functions/Classes to Move

#### 1. `data_loading.py` (NEW)

| Item | Type | Original Lines | Purpose |
|------|------|----------------|---------|
| `load_tuned_params()` | function | 98-157 | Load optimized params from JSON files |
| `load_forecast_data()` | function | 328-358 | Query WxForecastSnapshot table |
| `load_settlement_data()` | function | 361-378 | Query WxSettlement for actual temps |
| `load_market_data()` | function | 381-405 | Query KalshiMarket for brackets |
| `load_candle_data()` | function | 408-444 | Query KalshiCandle1m for prices |
| `get_forecast_at_open()` | function | 447-481 | Extract forecast with fallback logic |

**Dependencies:**
- `pathlib.Path` for JSON loading
- `pandas` for DataFrame operations
- SQLAlchemy models: `WxForecastSnapshot`, `WxSettlement`, `KalshiMarket`, `KalshiCandle1m`
- Strategy param classes from `strategies/__init__.py`

#### 2. `core_runner.py` (NEW)

| Item | Type | Original Lines | Purpose |
|------|------|----------------|---------|
| `OpenMakerTrade` | dataclass | 163-188 | Single trade record |
| `OpenMakerResult` | dataclass | 191-321 | Backtest results container with metrics |
| `run_strategy()` | function | 488-864 | Main unified backtest runner |

**Dependencies:**
- All data loading functions from `data_loading.py`
- All utility functions from `utils.py`
- Strategy classes from `strategies/`

#### 3. `reporting.py` (NEW)

| Item | Type | Original Lines | Purpose |
|------|------|----------------|---------|
| `save_results_to_db()` | function | 940-988 | Persist SimRun + SimTrade records |
| `print_results()` | function | 995-1035 | Pretty-print backtest summary |
| `print_debug_trades()` | function | 1038-1068 | Print sample trades for debugging |
| `print_comparison_table()` | function | 1075-1086 | Compare multiple strategies |

**Dependencies:**
- `OpenMakerResult` class (from `core_runner.py`)
- `OpenMakerTrade` class (from `core_runner.py`)
- SQLAlchemy models: `SimRun`, `SimTrade`

#### 4. `core.py` (REDUCED)

| Item | Type | Original Lines | Purpose |
|------|------|----------------|---------|
| `OpenMakerParams` | dataclass | 81-96 | Strategy parameters (keep here) |
| `run_backtest()` | function | 872-901 | Legacy wrapper |
| `run_backtest_next_over()` | function | 904-933 | Legacy wrapper |
| `main()` | function | 1089-1274 | CLI entry point |

**Re-exports for backward compatibility:**
```python
from .data_loading import load_tuned_params, get_forecast_at_open
from .core_runner import run_strategy, OpenMakerTrade, OpenMakerResult
from .reporting import print_results, print_debug_trades, print_comparison_table, save_results_to_db
```

### Import Dependencies to Update

No external files need import changes - all will continue to work via re-exports from `core.py`:

| File | Current Import | After Refactor |
|------|----------------|----------------|
| `optuna_tuner.py` | `from .core import OpenMakerParams, run_strategy, print_results` | No change (re-exported) |
| `live_trader.py` | `from .core import OpenMakerParams, load_tuned_params` | No change (re-exported) |
| `manual_trade.py` | `from .core import OpenMakerParams, load_tuned_params` | No change (re-exported) |
| `__init__.py` | `from .core import run_backtest, ...` | No change (re-exported) |

---

## Phase 2: Split open_maker/utils.py

### Target Module Structure

```
open_maker/utils/
├── __init__.py          # Re-exports all functions
├── common.py            # Timezone helpers (~25 lines)
├── brackets.py          # Bracket selection (~100 lines)
├── fees.py              # Fee calculations (~80 lines)
├── forecast.py          # Forecast lookup (~150 lines)
├── fills.py             # Candle/fill checks (~120 lines)
└── obs_curve.py         # Observation data (~130 lines)
```

### Functions to Move by Module

#### `common.py`
- `CITY_TIMEZONES` (constant)
- `get_city_timezone()`

#### `brackets.py`
- `find_bracket_for_temp()`
- `determine_winning_bracket()`
- `get_bracket_index()`
- `get_neighbor_ticker()`

#### `fees.py`
- `kalshi_taker_fee()`
- `kalshi_maker_fee()`
- `calculate_position_size()`
- `calculate_pnl()`
- `calculate_exit_pnl()`

#### `forecast.py`
- `get_predicted_high_hour()`
- `compute_decision_time_utc()`
- `get_forecast_temp_at_time()`

#### `fills.py`
- `get_candle_price_for_exit()`
- `get_candle_price_for_signal()`
- `check_fill_achievable()`

#### `obs_curve.py`
- `CITY_STATION_MAP` (constant)
- `load_minute_obs()`
- `compute_obs_stats()`

#### `__init__.py`
```python
# Re-export everything for backward compatibility
from .common import *
from .brackets import *
from .fees import *
from .forecast import *
from .fills import *
from .obs_curve import *
```

---

## Phase 3: Slim scripts/ (Future)

Move core logic from large scripts into `src/ingest/` modules, keeping scripts as thin CLI wrappers.

---

## Phase 4: Align tests/ (Future)

Update test imports and add coverage for new modules.

---

## Testing Strategy

### Before any changes:
```bash
python -m open_maker.core --city chicago --days 30 --strategy open_maker_base > baseline_output.txt
```

### After each module extraction:
```bash
python -m open_maker.core --city chicago --days 30 --strategy open_maker_base > after_output.txt
diff baseline_output.txt after_output.txt
# Should show no differences
```

### Unit tests to add:
- `tests/test_data_loading.py` - Mock DB, test query functions
- `tests/test_core_runner.py` - Test run_strategy with fixtures
- `tests/test_reporting.py` - Test output formatting

---

## Constraints

1. **Zero behavior change** - Functions are moved, not modified
2. **Full backward compatibility** - All existing imports continue to work
3. **Incremental migration** - Each step is independently verifiable
4. **No circular imports** - Careful ordering of new modules
