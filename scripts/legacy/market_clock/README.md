# Market Clock Scripts - Archived 2025-12-11

## Purpose
Market clock approach experiments - a training window from D-1 10:00 to D 23:55.

## Why Archived
These comparison/testing scripts are no longer needed.
NOTE: `train_market_clock_tod_v1.py` is NOT archived - it's still used by the
pipeline via `optuna_delta_range_sweep.py`.

## Files

| File | Description | Was Imported By |
|------|-------------|-----------------|
| `build_market_clock_dataset.py` | Build market-clock datasets | Nothing |
| `compare_market_clock_vs_tod_v1.py` | Compare approaches | Nothing |
| `test_market_clock_inference_offline.py` | Offline inference test | Nothing |
| `health_check_market_clock.py` | Health checks | Nothing |

## Revival Notes
The market clock approach is still active in the main pipeline.
These scripts were just experimental/comparison tools.
