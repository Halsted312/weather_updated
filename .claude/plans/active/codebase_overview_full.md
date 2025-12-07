# Codebase Overview — Weather Trading Stack (Deep Inventory)

High-level purpose
- Combines weather observations/forecasts with Kalshi market data to price and trade daily high-temperature brackets across six cities.
- Builds multi-source feature-rich datasets, trains ordinal CatBoost delta models with Optuna, and powers backtests plus live/manual trading.
- Stores cached datasets/models for reproducibility and faster experimentation.

Top-level directory landmarks
- README.md — narrative overview, market context, goals, and architecture summary.
- config/ — environment, city lists, feature toggles; check for knobs that affect dataset build/training.
- scripts/ — ingestion, maintenance, dataset build, training, diagnostics, and trading utilities.
- models/ — pipeline orchestrators, cached datasets, trained models, candles, and saved artifacts.
- src/ — core library code (DB models, clients, data loaders, features, training utilities, strategies).
- docs/ — architecture notes, how-tos, and planning artifacts.
- logs/ — run logs for ingestion, pipelines, training; key for auditing coverage and errors.
- open_maker/ — strategy and trading engine code (backtests, live trading, optuna tuning).
- tests/ — validation/backtest scripts and health checks.

Data layer (DB + schema references)
- src/db/models.py — SQLAlchemy models for weather (obs, forecasts, settlement), Kalshi markets/candles, simulation tables.
- src/db/connection.py — engine/session management.
- src/db/utils.py (if present) — helpers for session handling and retries.
- docs/permanent/FILE_DICTIONARY_GUIDE.md — authoritative field references by table.
- alembic.ini + migrations/ — schema migrations for wx/kalshi/sim schemas.

Configuration
- config/ (various YAML/JSON/py) — city lists, API keys/paths, feature flags, holdout ratios; ensure alignment with training scripts.
- tempest/tempest_api_key.txt, weather.pem — API credential artifacts (do not commit).
- Dockerfile, docker-compose.yml, Makefile — runtime/ops setup for services and local dev.

Ingestion and maintenance scripts (scripts/)
- scripts/ingest_vc_obs_backfill.py — backfill 5-min Visual Crossing observations per station/city.
- scripts/ingest_vc_obs_parallel.py — parallelized obs ingest across stations/cities.
- scripts/ingest_vc_forecast_snapshot.py — nightly/current VC forecast snapshots.
- scripts/ingest_vc_hist_forecast_v2.py — historical forecast backfill (basis-date driven).
- scripts/ingest_vc_historical_forecast.py — legacy historical forecasts ingestion.
- scripts/ingest_vc_historical_forecast_parallel.py — parallel variant for speed.
- scripts/backfill_vc_historical_forecasts.py — orchestrates historical forecast fills.
- scripts/backfill_vc_historical_forecast_minutes_austin.py — minute-level historical forecasts for Austin.
- scripts/ingest_weather_more_apis_guidance.py — NOAA/HRRR/NBM guidance ingestion.
- scripts/ingest_weather_more_apis_guidance_FAST.py — faster variant for guidance ingest.
- scripts/augment_austin_noaa_features.py — adds NOAA-derived features to Austin dataset.
- scripts/backfill_station_city_features.py — rebuilds station-city aggregate features.
- scripts/ingest_nws_settlement.py — ingests NWS settlement data per city/date.
- scripts/ingest_settlement_multi.py — consolidates settlement from multiple sources (CLI/CF6/IEM/NCEI).
- scripts/backfill_kalshi_markets.py — backfills Kalshi market/series metadata.
- scripts/backfill_kalshi_candles.py — backfills 1-minute Kalshi candles.
- scripts/poll_kalshi_candles.py — periodic candle polling daemon.
- scripts/export_kalshi_candles.py — export candles to parquet for analysis.
- scripts/kalshi_ws_recorder.py — records Kalshi WebSocket streams for replay.
- scripts/poll_vc_live_daemon.py — live VC ingest daemon for current forecasts.
- scripts/validate_15min_ingestion.py — QA for 15-min ingest.
- scripts/validate_austin_15min_data.sql — SQL validation for Austin 15-min data.
- scripts/validate_austin_data_sources.sql — SQL checks across Austin sources.
- scripts/validate_vc_minute_historical_nulls.sql — null checks for minute-level historical VC data.
- scripts/check_data_state.py — snapshot of ingest health and freshness.
- scripts/check_data_freshness.py — recency monitor for key tables.
- scripts/check_pipeline_health.py — high-level pipeline health checks.
- scripts/audit_data_coverage.py — coverage auditing for date ranges per city.
- scripts/check_candle_coverage.py — verifies candle completeness.
- scripts/clean_vc_smart.py — cleans VC anomalies.
- scripts/clean_vc_temp_errors.py — fixes temperature-specific ingestion errors.
- scripts/patch_cloudcover_all_cities.py — corrects cloud cover features across caches.
- scripts/patch_cloudcover_t1_aligned.py — cloud cover alignment fix.
- scripts/dense_candle_daemon.py — densifies candle data for modeling.
- scripts/build_dense_candles.py — constructs dense candle datasets (configurable chunking).
- scripts/build_market_clock_dataset.py — builds market-clock-aligned datasets.
- scripts/backfill_market_clock_* (if present) — fills market-clock data gaps.
- scripts/extract_raw_data_to_parquet.py — exports raw DB data to parquet for offline work.
- scripts/build_dataset_parallel.py — helper to parallelize dataset builds (legacy).
- scripts/build_dataset_from_parquets.py — builds train/test from cached full parquet.
- scripts/build_all_city_datasets.py — orchestrates dataset builds for all cities.
- scripts/rebuild_all_datasets.py — rebuilds cached datasets with new features.
- scripts/rebuild_all_cities_fresh.py — force rebuild all city caches from DB.
- scripts/run_multi_city_pipeline.py — orchestrates multi-city pipeline end-to-end.
- scripts/run_full_backfill.sh — shell wrapper for end-to-end backfill.
- scripts/backtest_edge.py — backtests edge model outputs.
- scripts/backtest_edge_classifier.py — classifier backtest.
- scripts/backtest_ml_hybrid.py — hybrid ML backtest experiments.
- scripts/backtest_hybrid_vs_tod_v1.py — compares hybrid vs time-of-day model.
- scripts/backtest_utils.py — helpers for backtests.
- scripts/compare_market_clock_vs_tod_v1.py — compares feature sets/time regimes.
- scripts/compare_station_vs_city_forecasts.py — evaluates station vs city forecast performance.
- scripts/analyze_skill_vs_horizon.py — forecast skill decay over lead time.
- scripts/analyze_temp_rules.py — rules-based temperature analysis.
- scripts/diagnose_edge_failure.py — inspects failed edge predictions.
- scripts/debug_edge_generation.py — debug dataset/feature issues in edge pipeline.
- scripts/debug_single_event.py — drill into one city/date event.
- scripts/debug_austin_features_snapshot.py — quick feature dump for Austin.
- scripts/debug_vc_15min_forecast.py — inspect 15-min forecast ingestion.
- scripts/debug_weather_more_apis_guidance.py — inspect NOAA guidance ingestion.
- scripts/trace_single_snapshot.py — trace dataset row construction.
- scripts/optuna_delta_range_sweep.py — sweeps delta targets with Optuna.
- scripts/train_chicago_simple.py — simplified training entry for Chicago.
- scripts/train_chicago_parallel.py — Chicago training with parallel build.
- scripts/train_chicago_30min.py — Chicago training variant with 30-min data.
- scripts/train_chicago_optuna.py — Chicago Optuna tuner (single-city).
- scripts/train_city_ordinal_optuna.py — primary multi-city ordinal training entry.
- scripts/train_all_cities_ordinal.py — batch train ordinal models for all cities.
- scripts/train_all_cities_hourly.py — trains hourly variant across cities.
- scripts/train_la_miami_ordinal.py — focused training for LA + Miami.
- scripts/train_austin_more_apis_6m.py — Austin training using more-apis features.
- scripts/train_market_clock_tod_v1.py — market-clock specific model training.
- scripts/train_tod_v1_all_cities.py — time-of-day model training.
- scripts/train_austin_chicago_auc.sh — shell to run AUC-focused training.
- scripts/train_all_cities_overnight.sh — overnight multi-city training wrapper.
- scripts/live_active_trader.py — live trading loop using model signals.
- scripts/live_midnight_trader.py — time-specific live trading variant.
- scripts/live_ws_trader.py — WS-driven live trading.
- scripts/edge_decision_cli.py — CLI to make edge decisions for a target date.
- scripts/verify_pipeline_parity.py — compare DB-built vs cached dataset parity.
- scripts/verify_austin_augmented.py — checks augmented Austin cache.
- scripts/test_feature_implementation.py — feature correctness tests.
- scripts/test_inference_all_cities.py — inference validation across cities.
- scripts/test_market_clock_inference_offline.py — offline inference tests for market-clock model.
- scripts/test_process_day.py — validates per-day processing.
- scripts/test_query_helpers.py — checks DB query helpers.
- scripts/health_check_market_clock.py — health check for market-clock data.
- scripts/health_check_tod_v1.py — health check for time-of-day model.

Dataset building (models/data, core loaders)
- models/data/dataset.py — central dataset builder; assembles features across weather, market, NOAA guidance, station-city aggregates, multi-horizon curves.
- models/data/splits.py — train/test splits by date ratio; day-grouped TimeSeriesSplit to prevent leakage.
- models/data/__init__.py — package init.
- models/data/feature_helpers.py (if present) — feature calculations.
- models/data/loader.py — DB query utilities for available dates, table accessors.
- models/data/market_features.py (if present) — market-derived features (spreads, depth).
- models/data/noaa_features.py (if present) — NOAA guidance feature extraction.
- models/data/multi_horizon.py (if present) — multi-horizon feature generation.

Training and orchestration
- scripts/train_city_ordinal_optuna.py — builds/loads datasets, honors date filters, re-splits cached data by holdout pct, Optuna tunes ordinal CatBoost, saves model/params/metrics.
- models/pipeline/03_train_ordinal.py — pipeline wrapper; ensures cached splits (or auto-splits full.parquet), sets CLI args, invokes trainer.
- models/pipeline/01_build_dataset.py — pipeline step to build cached datasets from DB.
- models/pipeline/02_delta_sweep.py — sweeps delta targets/parameters.
- models/pipeline/04_train_edge_classifier.py — trains edge classifier after ordinal delta model.
- models/pipeline/05_backtest_edge.py — backtests edge classifier outputs.
- scripts/run_multi_city_pipeline.py — orchestrates multi-city build/train/backtest sequence.
- scripts/run_full_backfill.sh — shell to run full backfill before training.
- scripts/build_dataset_from_parquets.py — rebuilds training set from existing parquets.
- scripts/build_all_city_datasets.py — batch builder for per-city caches.
- scripts/rebuild_all_cities_fresh.py — rebuilds caches from DB ignoring existing files.
- models/pipeline/README.md — pipeline usage notes and step ordering.

Model artifacts and cached data (models/saved, models/candles)
- models/saved/{city}/train_data_full.parquet — cached training snapshots.
- models/saved/{city}/test_data_full.parquet — cached holdout snapshots.
- models/saved/{city}/train_data.parquet, test_data.parquet — older or alternate splits.
- models/saved/{city}/ordinal_catboost_optuna.pkl/json — trained ordinal CatBoost model + params.
- models/saved/{city}/best_params.json — Optuna best parameters.
- models/saved/{city}/final_metrics_{city}.json — saved metrics and metadata.
- models/saved/{city}/edge_classifier.pkl/json — edge classifier artifacts.
- models/saved/{city}/edge_training_data*.parquet — classifier training data.
- models/saved/market_clock_tod_v1/* — market-clock model artifacts.
- models/saved/*_tod_v1/* — time-of-day variant artifacts.
- models/candles/candles_{city}.parquet — offline candle snapshots per city.
- models/candles_chicago_may2025.parquet — specific candle slice.

Strategies, backtests, trading (open_maker/)
- open_maker/core.py — orchestrates backtests; loads forecasts/obs/markets, runs strategies, computes P&L.
- open_maker/strategies/base.py — baseline strategy implementation.
- open_maker/strategies/next_over.py — next-over strategy to climb brackets.
- open_maker/strategies/curve_gap.py — curve gap strategy comparing forecast curves.
- open_maker/optuna_tuner.py — Optuna tuner for strategy parameters (bias, entry time/price, filters).
- open_maker/utils.py — utilities for bracket mapping, fees, liquidity realism, forecast helpers.
- open_maker/live_trader.py (if present) — live execution harness using WS/REST.
- open_maker/manual_trade.py (if present) — discretionary single-trade CLI.
- open_maker/backtests/* (if present) — reusable backtest components.

Utilities and helpers
- src/weather/visual_crossing.py — VC client for obs, current forecasts, historical forecasts.
- src/kalshi/* — REST/WS clients for Kalshi markets, events, order placement, account.
- src/db/utils/helpers.py (if present) — DB helper functions.
- tools/ — miscellaneous scripts/utilities for ops or analysis.
- legacy/ — older ingestion/training scripts kept for reference.
- analysis/, reports/, visuals/, visualizations/ — notebooks/reports and visualization outputs.

Testing and validation
- tests/ — Python tests/backtest validations.
- scripts/test_feature_implementation.py — feature parity checks.
- scripts/test_inference_all_cities.py — inference smoke tests across cached data.
- scripts/test_market_clock_inference_offline.py — offline inference for market-clock model.
- scripts/test_process_day.py — per-day dataset validation.
- scripts/test_query_helpers.py — DB query helper coverage.

Logs and run artifacts
- logs/multi_city_pipeline_20251206_123213.log — multi-city pipeline run; shows sequencing, per-city stats/errors.
- logs/12-6-2025_afternoon_multi_city_train.md — training session notes for multi-city run.
- logs/terminal_outputs_afternoon_training.md — terminal captures from training.
- logs/station_obs_backfill.log — ingestion/backfill status for station observations (upstream gaps).
- logs/backfill_austin.log, backfill_austin_full.log — city-specific backfill runs.
- logs/training_log.txt — generic training log.

Docs and planning
- docs/ (permanent, how-tos, planning) — guides for ingestion, feature plans, historical forecast ingestion.
- docs/EDGE_CLASSIFIER_COMPLETE_FILE_REFERENCE.md — enumerates pipeline files for edge classifier.
- docs/DEVELOPER_HANDOFF.md — quick handoff summary.
- docs/how-tos_archive, docs/permanent — historical plans and reference material.
- .claude/plans/active/*.md — current planning/prompts (including this file).

Key data flows (end-to-end)
- Ingest raw obs/forecasts/settlements/markets into DB via scripts/ingest_* and backfill_*.
- Validate coverage via audit/health scripts (check_data_freshness, audit_data_coverage, station_obs_backfill.log).
- Build datasets from DB using models/data/dataset.py (optionally via pipeline 01_build_dataset).
- Cache per-city parquets under models/saved/{city}; auto-split from full.parquet if present.
- Train ordinal CatBoost with Optuna via scripts/train_city_ordinal_optuna.py or pipeline 03_train_ordinal.py; metrics logged and saved.
- Optionally run delta sweeps and edge classifier training (pipeline 02 + 04).
- Backtest edges (pipeline 05_backtest_edge, backtest_edge.py) and run strategy backtests in open_maker/core.py with strategy modules.
- Deploy signals to live traders (scripts/live_*), integrating with Kalshi REST/WS.

Operational notes
- Cached path must honor date filters (`--start-date`, `--end-date`) even when using `--use-cached`.
- Holdout split uses day-based ratio (default 80/20); splits handled by models/data/splits.py and training scripts.
- Feature checks (NOAA guidance, station-city, multi-horizon) logged during training for QA.
- Schema changes require Alembic migrations; keep DB models and migrations aligned.

Artifacts to monitor
- models/saved/{city}/final_metrics_{city}.json — training metadata and metrics.
- models/saved/{city}/best_params.json — Optuna results.
- models/saved/{city}/ordinal_catboost_optuna.pkl — deployed model.
- models/saved/{city}/edge_classifier.pkl — classifier for trade/no-trade gating.
- logs/* — for pipeline/training errors, coverage gaps, and anomalies.

Suggested review order (for audits)
- Read logs/multi_city_pipeline_20251206_123213.log to understand recent pipeline run and errors.
- Cross-check training logs (12-6-2025_afternoon_multi_city_train.md, terminal_outputs_afternoon_training.md) for date ranges, row counts, and feature presence.
- Inspect station_obs_backfill.log for upstream obs gaps that explain missing days/features.
- Skim scripts/train_city_ordinal_optuna.py for cached handling, date filters, and split math.
- Skim models/data/dataset.py and models/data/splits.py for expected columns/split rules.
- Inspect models/pipeline/03_train_ordinal.py for auto-splitting and cached usage.
- Review open_maker/core.py and strategies for downstream consumption of model outputs.
- Validate cached parquet presence and sizes under models/saved/{city}.

Path index (quick jump list)
- Training entry: scripts/train_city_ordinal_optuna.py
- Pipeline wrapper: models/pipeline/03_train_ordinal.py
- Dataset builder: models/data/dataset.py
- Split helpers: models/data/splits.py
- Ingest obs: scripts/ingest_vc_obs_backfill.py, scripts/ingest_vc_obs_parallel.py
- Ingest forecasts: scripts/ingest_vc_forecast_snapshot.py, scripts/ingest_vc_hist_forecast_v2.py
- Ingest settlements: scripts/ingest_settlement_multi.py, scripts/ingest_nws_settlement.py
- Ingest market data: scripts/backfill_kalshi_markets.py, scripts/backfill_kalshi_candles.py, scripts/poll_kalshi_candles.py
- Guidance features: scripts/ingest_weather_more_apis_guidance.py, scripts/augment_austin_noaa_features.py
- Dataset rebuild: scripts/rebuild_all_datasets.py, scripts/rebuild_all_cities_fresh.py, scripts/build_all_city_datasets.py
- Parquet tooling: scripts/build_dataset_from_parquets.py
- Backtests: scripts/backtest_edge.py, open_maker/core.py + strategies/
- Strategy tuner: open_maker/optuna_tuner.py
- Live trading: scripts/live_active_trader.py, scripts/live_ws_trader.py, scripts/live_midnight_trader.py
- Health checks: scripts/check_data_freshness.py, scripts/audit_data_coverage.py, scripts/check_pipeline_health.py
- Coverage logs: logs/station_obs_backfill.log, logs/multi_city_pipeline_20251206_123213.log
