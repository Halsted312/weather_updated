# Tmax Forecast Ensemble (Updated 2025-11-18)

This document captures the current state of the “predict-the-daily-high” pipeline, how to reproduce it, and how it feeds the trading stack.

## Overview

Goal: At every 5‑minute bar, estimate the eventual CLI high temperature (`Tmax`) using only the thermometer path so far (Visual Crossing minute data + prior day info). Convert those estimates into bracket probabilities and feed ModelKelly. Every component is now Optuna-tuned and linearly calibrated so the ensemble automatically rebalances weights per city/season.

The ensemble currently contains:

1. **GBDT Forecaster** (`HistGradientBoostingRegressor`)  
   Features: `temp_now`, running max/min, gap-to-max, rolling 30/60 minute means/std, slope over 30/60 min, running-max slope, prior-day CLI, minute-of-day. Hyperparameters (max depth, learning rate, L2, iterations, min leaf) are tuned via Optuna.

2. **Spline Extrapolator** (`scipy.interpolate.UnivariateSpline`)  
   Fits the partial intraday curve, extrapolates the remaining hours, clips to `[running_min-10, running_max+10]` to avoid explosive predictions. Optuna sweeps the smoothing multiplier, polynomial degree, and forecast horizon.

3. **Sequence Model (GRU)**
   Uses the last ≥180 minutes of `temp_f`, `running_max`, minute-of-day, and `prior_tmax` to learn curvature/mean reversion patterns. Runs on the 24 GB RTX 3090 (CUDA if available) and can run its own Optuna sweep (`--seq-optuna-trials`) to pick hidden size, dropout, lookback window, stride, epochs, and batch size. Pass `--export-seq-model` to emit a TorchScript artifact for fast inference.

4. **CatBoost residual forecaster (optional)**
   Enable via `--enable-catboost`.  Optuna tunes depth/learning rate/L2/iterations with monotone constraints so the regressor respects basic physics (later minutes and higher running-max should not decrease the eventual Tmax).  The CatBoost output is blended with the other components via the same convex-weight Optuna sweep, so downstream consumers only need to read `component_weights`.

Weights: learned via Optuna on the validation window (subject to a convex sum).  Residual sigma is no longer a single global number; we fit a small HGBR on features such as minute-of-day, running std, and slopes to estimate the expected absolute error per snapshot. Finally, a linear calibration step removes any remaining bias and its coefficients are stored in the metadata JSON.

## Reproduce Chicago Run

```bash
# Generate features + train ensemble. Saves per-minute predictions to CSV.
python scripts/train_tmax_regressor.py \
  --city chicago \
  --start 2024-10-25 \
  --end   2025-11-16 \
  --cutoffs 12:00 14:00 16:00 18:00 \
  --optuna-trials 40 \
  --seq-optuna-trials 10 \
  --enable-catboost --catboost-trials 30 \
  --export-csv results/tmax_preds_chicago.csv \
  --export-metadata results/tmax_model_chicago.json \
  --export-seq-model models/trained/tmax_seq_chicago.pt
```

Key metrics (test fold, 78 days):

| Cutoff (local) | MAE (°F) | Bias (°F) |
|----------------|----------|-----------|
| All minutes    | 3.01     | 1.28      |
| 12:00          | 2.33     | 0.77      |
| 14:00          | 2.23     | 1.56      |
| 16:00          | 1.57     | 1.17      |
| 18:00          | 1.32     | 0.76      |

Residual std dev (global): **4.34 °F**.  The learned sigma regressor tightens this considerably intraday (median ≈3.1 °F on Chicago).

## Feeding ModelKelly

```
python backtest/run_backtest.py \
  --strategy model_kelly \
  --city chicago \
  --bracket between \
  --start-date 2024-10-25 \
  --end-date   2025-11-16 \
  --model-type tmax_reg \
  --tmax-preds-csv results/tmax_preds_chicago.csv \
  --tmax-min-prob 0.60 \
  --tmax-sigma-multiplier 0.75 \
  --exec-time-window between@09:00-21:00 \
  --exec-sigma-gate 4.5 \
  --market-odds-weight 0.25 \
  --initial-cash 10000 \
  --output-json results/backtest_chicago_tmax_gated.json
```

`model_type='tmax_reg'` causes the adapter to:
1. Load `results/tmax_preds_chicago.csv`.
2. Lookup the latest snapshot for each minute (`pred`, `sigma_est`).
3. Convert to bracket probabilities using a Gaussian CDF.
4. Run ModelKelly with the same fee/spread limits as the other models.

**Confidence Gating Results (Chicago, 2024‑10‑25 → 2025‑11‑16):**

| `tmax_min_prob` | `tmax_sigma_mult` | `hybrid_model_type` | Trades | Sharpe | P&L ($) | Max DD |
|-----------------|-------------------|----------------------|--------|--------|---------|--------|
| 0.55            | 0.50              | –                    | 23     | −3.88  | −3,571  | −36.8% |
| 0.60            | 0.50              | –                    | 23     | −3.88  | −3,570  | −36.7% |
| **0.60**        | **0.75**          | –                    | **3**  | **−1.40** | **−587** | **−6.1%** |
| 0.55            | 0.50              | elasticnet (0.50)    | 0      | n/a    | 0       | 0%     |

(Hybrid row uses `--hybrid-model-type elasticnet --hybrid-min-prob 0.50`; it refused to trade because the settlement model never agreed with the Tmax view over this window.)

Without gating the strategy fired 639 times with Sharpe −17.6 / −$9.7k. The probability + sigma thresholds cut activity to a handful of high-conviction events and remove the catastrophic drawdowns. More tuning is available (time-of-day grids, city-specific parameters), but even these coarse gates slash loss severity, which is the first requirement before layering settlement-model agreement.

`--market-odds-weight` enables log-odds opinion pooling so the adapter can lean into live prices without surrendering control to the book. A value of `0.25` says “keep 75 % of the model logit, but nudge 25 % toward the mid-price view,” which smooths sharp transitions without hand-written heuristics.

## Sequence Model + Dynamic Sigma

Running `scripts/train_tmax_regressor.py` now trains three components: HistGBR (Optuna-tuned), spline (Optuna-tuned smoothing & degree), and a GRU that ingests the last 180 minutes of temps. The GRU outputs are written to the CSV alongside the other components; the TorchScript export can be loaded directly inside cron/backtest jobs for sub-millisecond inference. After generating residuals we fit a small HistGBR (`minute_of_day`, `running_std_60`, `slope_30/60`, `delta_from_max`) to predict the expected absolute error, producing `sigma_est` per row rather than a single global standard deviation. The backtester consumes this sigma to determine the Gaussian bracket conversion and the sigma-based gating threshold.

Every training run now emits:

1. `results/tmax_preds_<city>.csv` – the calibrated, per-minute predictions with one column per enabled component, `sigma_est`, and the derived quantiles `pred_p10`/`pred_p90`.
2. `results/tmax_model_<city>.json` – a metadata bundle describing the Optuna winners (GBDT params, spline params, GRU params, CatBoost params when enabled, ensemble weights, calibration coefficients). Backtesting/production jobs read this file so inference stays in sync with training (see `docs/tmax_metadata_contract.md`).
3. (Optional) `models/trained/tmax_seq_<city>.pt` – TorchScript GRU for lightning-fast inference.

## Daily Baseline

Before relying on minute-by-minute execution we can sanity-check the raw temperature forecasts by trading once per day at a fixed cutoff:

```
python scripts/backtest_tmax_daily.py \
  --city chicago \
  --bracket between \
  --tmax-preds-csv results/tmax_preds_chicago.csv \
  --start-date 2024-10-25 \
  --end-date 2025-11-16 \
  --cutoff 16:00 \
  --min-edge 0.05
```

The script pulls the 16:00 local snapshot, converts it to bracket probabilities, crosses the book once per day, and writes both a summary JSON and a CSV in `results/tmax_daily_trades_*`. Use this to benchmark alternative gating schemes and to confirm that the forecast is directionally useful before re-enabling minute-level execution.

## Multi-City Batch & Nightly Automation

Use `scripts/run_tmax_batch.py` to regenerate forecasts and backtests for every city in `CITY_CONFIG`:

```
python scripts/run_tmax_batch.py \
  --cities all \
  --start 2024-10-25 \
  --end   2025-11-16 \
  --cutoffs 12:00 14:00 16:00 18:00 \
  --tag latest \
  --results-dir results \
  --models-dir models/trained \
  --enable-catboost --catboost-trials 25 \
  --tmax-min-prob 0.60 \
  --tmax-sigma-multiplier 0.75 \
  --hybrid-model-type elasticnet \
  --hybrid-min-prob 0.50 \
  --run-daily-baseline
```

The script loops through each city, calls `train_tmax_regressor.py`, runs the ModelKelly backtest with your preferred gates, and (optionally) emits the daily baseline JSON/CSV. The new `--tag` flag keeps artifact names stable (`*_latest`), while `--results-dir`/`--models-dir` let cron jobs write into versioned directories before atomically swapping the symlink.

## Next Steps

1. **Hybrid Probability Gating:** Require agreement between the legacy settlement models (ElasticNet / CatBoost) and the Tmax forecasts before entering a trade. This should further reduce churn while we continue to improve the standalone temperature signal.
2. **Monitoring & Documentation:** Surface the 30-minute settlement poller status (see `docs/SYSTEMD_JOBS.md`) in a daily report and keep the project status doc up-to-date as new gating sweeps land.
3. **City Rollout:** Once Chicago is consistently bounded, run `scripts/train_tmax_regressor.py` for Philadelphia and Los Angeles, export the `tmax_preds_<city>.csv` files, and backtest with the tuned gating settings.
4. **Sequence Model Nightly Automation:** Add the GRU training/inference to the nightly cron/systemd routine so that each city’s CSV refreshes automatically with the latest weights and per-minute sigma regression.
