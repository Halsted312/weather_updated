# Chicago Scripts - Archived 2025-12-11

## Purpose
Chicago-specific training experiments from early development phase.

## Why Archived
Superseded by all-cities scripts:
- `train_city_ordinal_optuna.py` handles all 6 cities
- `models/pipeline/03_train_ordinal.py` is the official training entry point

## Files

| File | Description | Was Imported By |
|------|-------------|-----------------|
| `train_chicago_simple.py` | Simple Chicago model training | Nothing |
| `train_chicago_optuna.py` | Chicago Optuna tuning | Nothing |
| `train_chicago_parallel.py` | Parallel Chicago training | Nothing |
| `train_chicago_30min.py` | 30-minute snapshot training | Nothing |

## Revival Notes
These scripts contain Chicago-specific hardcoded configs. If reviving:
1. Update to use `CityConfig` from `src/config/cities.py`
2. Use the standard feature pipeline from `models/features/`
3. Consider merging logic into `train_city_ordinal_optuna.py` instead
