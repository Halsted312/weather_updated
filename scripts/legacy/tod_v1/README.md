# TOD v1 Scripts - Archived 2025-12-11

## Purpose
Time-of-Day version 1 training and health check scripts.

## Why Archived
These scripts are NOT used by the main pipeline (`models/pipeline/01-05`).
The pipeline uses:
- `train_city_ordinal_optuna.py` for ordinal training
- `train_market_clock_tod_v1.py` (via `optuna_delta_range_sweep.py`) for delta sweep

## Files

| File | Description | Was Imported By |
|------|-------------|-----------------|
| `train_tod_v1_all_cities.py` | Train TOD v1 for all cities | Nothing |
| `health_check_tod_v1.py` | TOD v1 health checks | Nothing |
| `backtest_hybrid_vs_tod_v1.py` | Hybrid vs TOD v1 backtest | `backtest_utils.py` |

## Revival Notes
If reviving `backtest_hybrid_vs_tod_v1.py`:
- Update import: `from scripts.backtesting.backtest_utils import ...`
