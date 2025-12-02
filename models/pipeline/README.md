# Weather ML Pipeline (All Cities)

Numbered wrappers for the end-to-end per-city pipeline. Each step calls the existing scripts with sane defaults (market-clock window, 5-min snapshots, multi-horizon forecasts, market features ON, station-city features ON). Run from repo root:

1) Build dataset (train/test parquet)  
`python models/pipeline/01_build_dataset.py --city chicago --workers 24 --holdout-pct 0.20`

2) Delta range sweep (optional but recommended)  
`python models/pipeline/02_delta_sweep.py --city chicago --trials 50 --test-days 66`

3) Train ordinal CatBoost (Optuna)  
`python models/pipeline/03_train_ordinal.py --city chicago --trials 80 --cv-splits 3 --workers 24`

4) Train edge classifier (Optuna)  
`python models/pipeline/04_train_edge_classifier.py --city chicago --trials 80 --workers 12 --decision-threshold 0.5`

5) Edge backtest (sanity check)  
`python models/pipeline/05_backtest_edge.py --city chicago --days 60 --threshold 1.5 --interval 60`

Notes
- Outputs live under `models/saved/{city}/`.
- Dataset builder always enables: `include_multi_horizon=True`, `include_market=True`, `include_station_city=True`, `include_meteo=True`.
- Workers: use whatever your machine can handle (e.g., 24–32 on a 64-thread box).
- Delta sweep uses `scripts/optuna_delta_range_sweep.py`; it saves results to `models/saved/{city}/delta_range_sweep/`.
- Ordinal training uses cached parquets if present; delete them to force rebuild.
- Edge classifier and backtest require Kalshi candles + settlements in the DB for that city/date range.
