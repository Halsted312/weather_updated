# Prompt for Claude Coding Agent — Dataset Pipeline Audit & Ordinal Training Review

You are auditing the dataset creation + feature coverage for all cities and verifying the ordinal CatBoost/Optuna training flow. Work strictly in read/analysis mode (no code changes). Use the provided logs for evidence:
- `logs/12-6-2025_afternoon_multi_city_train.md`
- `logs/terminal_outputs_afternoon_training.md`
- `logs/multi_city_pipeline_20251206_123213.log`

## Goals
- Map the end-to-end pipeline from raw sources to cached parquets (`train_data_full.parquet` / `test_data_full.parquet`) for each city.
- Confirm feature coverage (station-city, NOAA guidance, multi-horizon, market features) and identify any missing/nullable columns per city.
- Verify train/test date ranges, holdout logic, and that date filters are respected when using cached data.
- Surface errors/warnings/coverage gaps from the logs and prioritize fixes.
- Assess whether Optuna training ran on the intended data slices and produced expected metrics.

## Key Code/Artifacts to Inspect (read-only)
- Dataset build: `models/data/dataset.py`, `models.data.splits`, any feature engineering helpers referenced there.
- Training orchestration: `scripts/train_city_ordinal_optuna.py` (including cached path + date filtering), `models/pipeline/03_train_ordinal.py`.
- Cached data layout: `models/saved/{city}/train_data_full.parquet` and `test_data_full.parquet` (metadata only if needed).
- Any referenced configs in `config/` relevant to city lists, feature toggles, or holdout ratios.

## What to Extract from Logs
- For each city run, capture:
  - Date windows used (train/test start/end), total days, holdout days.
  - Row counts and column counts per split.
  - Feature presence checks (e.g., NOAA, station_city, multi-horizon) and any missing columns.
  - Errors/warnings about missing parquets, empty splits, or filter-induced empty datasets.
  - Optuna summary: trials run, objective, final metrics, and any anomalies (e.g., unusually low sample counts).
- Note any discrepancies between requested date filters and the actual ranges used when `--use-cached` was set.

## Output to Produce
- A concise report (per city) with:
  - Data ranges and split summary.
  - Feature coverage status (present/missing) with examples.
  - Issues/risks flagged from logs (with log file + line/section references).
  - Actionable recommendations (e.g., rebuild cache, add feature column, re-run with corrected dates).
- A short cross-city comparison highlighting systemic issues (e.g., consistently missing NOAA features, cache stale vs requested dates).

## Workflow Hints
- Start by skimming `logs/multi_city_pipeline_20251206_123213.log` for the pipeline flow, then cross-check details in the two training logs.
- Pay attention to any `--start-date/--end-date` usage in the logs and ensure cached paths honored them.
- If you need schema context, consult the dataset builder and splits modules to understand expected columns and split rules.

## Deliverable Format
- Markdown summary with per-city sections and a final “Next Actions” list ordered by impact.
