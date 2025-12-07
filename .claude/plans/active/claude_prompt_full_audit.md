# Prompt for Claude Coding Agent — Full-Codebase Discovery & Dataset/Training Audit

You are investigating the entire weather trading codebase to understand data flows, feature coverage, and training pipelines. Work in read/analysis mode only (no edits). Use the existing logs for evidence:
- `logs/multi_city_pipeline_20251206_123213.log`
- `logs/12-6-2025_afternoon_multi_city_train.md`
- `logs/terminal_outputs_afternoon_training.md`
- `logs/station_obs_backfill.log` (context on raw obs coverage/backfills)

## Objectives
1) Map architecture and responsibilities (ingestion → dataset build → training → strategy/trading).
2) Validate dataset creation for all cities and feature completeness (station-city, NOAA guidance, multi-horizon, market features).
3) Verify train/test splits, date filters, and cached parquet handling (honor `--start-date/--end-date`).
4) Assess training runs (Optuna trials/objective, sample counts, metrics) and flag anomalies.
5) Surface operational risks, stale caches, or missing data signals from logs.
6) Identify upstream data gaps (obs/forecasts/settlements/market candles) that could explain missing features or rows.

## Code Paths to Read
- **Dataset**: `models/data/dataset.py`, `models/data/splits.py`, related feature builders/helpers.
- **Training**: `scripts/train_city_ordinal_optuna.py` (cached path, date filtering, holdout), `models/pipeline/03_train_ordinal.py` (auto-splitting `full.parquet`), any config in `config/` that affects city lists/holdout ratios.
- **Ingestion**: key `scripts/ingest_*` for Visual Crossing obs/forecasts, settlement, Kalshi markets/candles; note expected tables/columns.
- **Trading/Strategies**: `open_maker/` (core, strategies, optuna_tuner, live/manual trading) to understand downstream consumers of the models.
- **Artifacts**: cached parquets in `models/saved/{city}`; trained model/params/metrics outputs.
- **Schema/Docs**: `src/db/models.py`, `docs/permanent/FILE_DICTIONARY_GUIDE.md`, README, any city/feature config under `config/`.

## Log Extraction Checklist
For each city/run found in the logs:
- Train/test date ranges, total days, holdout days, and whether date filters were requested.
- Row/column counts per split; any filtering warnings (empty after filters).
- Feature presence checks (NOAA, station-city, multi-horizon) and missing-column warnings.
- Errors about missing parquet files, stale cache usage, or split failures.
- Optuna configuration (trials, objective, CV splits), final metrics, and any anomalies.
- Upstream ingestion/backfill signals from `station_obs_backfill.log` (gaps, failed fetches) that could propagate to missing features or days.

## Reporting Format
- Per-city sections summarizing data ranges, feature coverage status, issues/risks (with log file + line/section references), and recommended actions (rebuild cache, fix missing feature, rerun training with corrected dates).
- Cross-city summary highlighting systemic issues (e.g., repeated missing NOAA features, date-filter mismatches, stale caches).
- Short “Next Actions” list ordered by impact/urgency.

## Workflow Tips
- Start with `logs/multi_city_pipeline_20251206_123213.log` for pipeline sequencing, then cross-check details in the training logs (`12-6-2025_afternoon_multi_city_train.md`, `terminal_outputs_afternoon_training.md`) and ingestion context (`station_obs_backfill.log`).
- Confirm cached path behavior vs requested date filters (`--use-cached`, `--start-date`, `--end-date`).
- When unsure about expected columns, refer to the dataset builder; when unsure about split rules, check `models/data/splits.py`.
- If metrics seem off, validate sample sizes vs expected coverage from the ingestion logs and cached parquet row counts (log outputs list rows/cols).
- Capture any TODOs/notes present in logs that imply known tech debt or pending fixes.
