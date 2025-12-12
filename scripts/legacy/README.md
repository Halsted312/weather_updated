# Scripts Legacy - Archived 2025-12-11

This folder contains archived scripts that are no longer part of the active workflow.
They are preserved for reference and potential future revival.

## Subfolders

| Folder | Purpose | File Count |
|--------|---------|------------|
| `chicago/` | Chicago-specific training experiments | 4 |
| `austin/` | Austin 15-minute feature experiments | 10 |
| `market_clock/` | Market clock approach experiments | 4 |
| `tod_v1/` | TOD v1 scripts not used by pipeline | 3 |
| `analysis/` | One-off analysis scripts | 6 |

## Why Archived

These scripts were superseded by:
- All-cities training scripts (`train_city_ordinal_optuna.py`)
- The unified pipeline in `models/pipeline/01-05`
- Feature standardization in `models/features/`

## Revival Notes

To bring any script back:
1. Check imports - they may reference old module paths
2. Verify database schema compatibility
3. Test with a small date range first
4. Update any hardcoded paths or configs
