# Next Steps — 2025-11-19

## Context
- **Cities in scope:** Chicago, Miami, Austin, Los Angeles, Denver, Philadelphia (NYC retired permanently).
- **Live ingest:** `scripts/poll_settlements.py` now loops every 30 minutes via `scripts/run_settlement_poller.sh` + systemd (`init/systemd/kalshi-settlement-poller.service`). CLI + CF6 rows refresh continuously with a 3-day backfill window.
- **Latest Tmax artifacts:** `results/tmax_preds_chicago_latest.csv`, `results/tmax_model_chicago_latest.json`, `models/trained/tmax_seq_chicago_latest.pt` (Optuna-tuned GBDT/spline/GRU ensemble with linear calibration + TorchScript export).
- **Reference backtest:** `results/backtest_chicago_tmax_latest.json` (2024-08-01→2025-11-16, gating 0.60/0.75). Current run shows 313 trades / Sharpe −9.61 because upstream markets contain `strike_type=NULL` rows.

## Feature Overview (Tmax Ensemble)
1. **GBDT forecaster** – minute-level engineered features:
   - `temp_f`, `running_max/min`, `delta_from_max`, rolling means (`rolling_mean_30`, `rolling_mean_60`), rolling std, slopes (30/60 minute), `running_max_slope`, `minute_of_day` normalized, `prior_tmax`.
   - Optuna tunes: depth 4–10, learning rate 0.03–0.30, iterations 200–600, `l2_regularization`, `min_samples_leaf`.
2. **Spline extrapolator** – `scipy.interpolate.UnivariateSpline` on the intraday curve. Optuna tunes smoothing multiplier (0.1–3.0), polynomial degree (2–5), and forecast horizon (30–180 minutes). Predictions are clipped to `[running_min-10, running_max+10]`.
3. **Sequence model (GRU)** – inputs per minute:
   - `[temp_f, running_max, minute_of_day_normalized, prior_tmax]` across the last ~180 minutes (stride tunable). Trains on GPU (3090) with Adam (lr search), dropout, hidden size, epochs, batch size all tuned if `--seq-optuna-trials` > 0.
   - Export via TorchScript for fast inference (`--export-seq-model`).
4. **Ensemble & calibration:**
   - Optuna learns convex weights across `[GBDT, spline, GRU]`.
   - Linear regression calibrates the ensemble output to minimize bias on train+validation.
   - HistGradientBoosting residual model yields `sigma_est` for each snapshot (`minute_of_day`, `running_std_60`, `slope_30/60`, `delta_from_max`).
5. **Artifacts:**
   - CSV columns: `timestamp`, `date`, `minute_of_day`, `pred_gbdt`, `pred_spline`, `pred_seq`, `pred_raw`, `pred` (calibrated), `sigma_est`, plus actual residuals for diagnostics.
   - Metadata JSON: hyperparameters, best Optuna trials, weights, calibration slope/intercept, and pointer to the TorchScript file.

## Completed 2025-11-18 → 2025-11-19
- [x] Dropped NYC from every active config (discover scripts, ingest loaders, coverage checks, docs), while preserving historical data for archive-only use.
- [x] Hardened `ingest/load_kalshi_data.py` to auto-derive missing columns, summarize ingestion logs, and handle new parquet schemas.
- [x] Added 30-minute settlement poller loop + systemd unit + doc.
- [x] Rebuilt the Tmax trainer with Optuna + calibration + TorchScript and generated fresh Chicago artifacts.
- [x] Ran a full ModelKelly backtest using the new predictions (exposed gaps in market strike metadata).
- [x] Rehydrated Chicago discovery parquet with canonical strike metadata, reloaded Postgres, and reran the gated Tmax ModelKelly reference (3 trades, $51.64 gross, Sharpe 0.00) using the cleaned markets.【152d7d†L1-L4】【F:results/backtest_chicago_tmax_latest.json†L1-L9】

## Next Actions (Detailed)
### 1. Data Hygiene & Backfill
- **Fix `strike_type=NULL` markets**
  - Query `SELECT ticker FROM markets WHERE strike_type IS NULL AND series_ticker='KXHIGHCHI';`
  - Determine whether the discovery script missed `strike_type`; patch `scripts/discover_chicago.py` to persist it; reload affected markets with `ingest/load_kalshi_data.py --cities chicago --refresh-grid`.
  - ✅ `scripts/discover_*` now call a shared parser, `ingest/load_kalshi_data.py` runs the same fallback, and `scripts/fix_strike_metadata.py` can backfill existing rows.
  - 2025-11-19 update: Chicago now reports zero `strike_type` gaps after rewriting the parquet via the new saver and reloading metadata.【d7cbaf†L1-L11】
- **Finalize removal of NYC directories**  
  - Archive `data/kalshi_full_2024_2025/nyc/` and `data/discovery/.../nyc/` under `/home/halsted/archive/` or delete once backups are done.
  - Drop any NYC tables/rows if they clutter Postgres (optional).

### 2. Settlement Poller Ops
- **Enable systemd job**  
  - `sudo cp init/systemd/kalshi-settlement-poller.service /etc/systemd/system/`  
  - `sudo systemctl daemon-reload && sudo systemctl enable --now kalshi-settlement-poller.service`
- **Monitoring**
  - `journalctl -u kalshi-settlement-poller.service -f` to confirm 30-minute cadence and check for IEM/CF6 errors.
  - Add a daily log roll-up (cron) summarizing count of CLI/CF6 rows inserted per city.
  - ✅ Covered in `docs/ops/settlement_poller.md` with `scripts/settlement_rollup.py` for the daily summary.

### 3. Modeling & Backtesting
- **Re-run Tmax training for remaining cities**  
  - For each city in `CITY_CONFIG`, run `scripts/train_tmax_regressor.py` with at least `--optuna-trials 20 --seq-optuna-trials 5`, export CSV/metadata/TorchScript to `results/` and `models/trained/`.
- **Batch backtest**  
  - Extend `scripts/run_tmax_batch.py` to accept metadata/TorchScript paths per city. Ensure ModelKelly adapters read the metadata calibration.
  - Recompute `results/backtest_<city>_tmax_latest.json` for all six cities with the new predictions; document Sharpe, drawdown, trades.
- **Investigate negative Sharpe**  
  - After the strike fix, rerun Chicago to confirm gating reduces trade count and improves P&L.
  - Consider additional gates (time-of-day, humidity/variance) now that predictions are calibrated.

### 4. Documentation & Hand-off
- **Update `status_2025-11-18.md`** with the new tasks’ outcomes and metrics (after reruns).
- **Add an Ops checklist** for starting/stopping the settlement poller, nightly batch, and verifying artifacts.
- **Share metadata contract** (fields in `results/tmax_model_<city>.json`) with downstream consumers so they can parse weights/calibration reliably.
