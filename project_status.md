# Current State and Next Steps (Austin EV Modeling)

## Overview
- **Data scope:** Austin, between-bracket, 1-minute candles and Visual Crossing weather. Database now holds post-January 2025 markets/candles and late-session feature snapshots (NextGen set).
- **Model baselines:** ElasticNet + CatBoost (Optuna-tuned) for settlement probabilities; new CatBoost regression for minute-level EV (future mid-price delta).
- **Execution harness:** Maker/Taker fee-aware ModelKelly strategy with per-trade 2% bankroll cap, 5¢ edge gate, and EV predictions plumbed into the decision loop.

## Experiments Completed
1. **CatBoost (Optuna) on settlement labels**
   - 30→7 rolling windows (June–July 2025), NextGen features, monotone constraints.
   - Best out-of-sample window: Brier ≈ 0.18, log-loss ≈ 0.50; other windows ~0.21–0.35 Brier → indicates inconsistent calibration across weeks.
   - Maker-first backtests with these probabilities (and the 2% cap) still produce negative Sharpe: e.g., `results/backtest_austin_between_catboost_optuna.json` → Sharpe −7.6, max DD −9.8%, net P&L −$956.

2. **Late-session filter**
   - Added `--max-minutes-to-close` and “start at (prior peak − X minutes)” gating to `build_training_dataset`.
   - CatBoost (10 trials) with prior-peak window (start 3 hours before yesterday’s observed max) still produced weak windows (Brier 0.23–0.34), and backtests remained negative (Sharpe −14.8, P&L −$3.4k).

3. **Minute-level EV modeling**
   - New dataset: label is `future_mid_{+60m} − current_mid`. Filter rows between prior-day peak and close; trained CatBoost regression (Optuna 10 trials per window) with RMSE ≈ 30¢.
   - Predictions saved under `models/trained/austin/between_ev_catboost/win_*` with `pred_delta_cents`, `pred_future_mid_cents`, and `p_model = pred_future_mid/100`.
   - Backtest: `python backtest/run_backtest.py --model-type ev_catboost ...` → Sharpe −34, max DD −56%, net P&L −$5.6k over 63 trades. The EV model often shorted aggressively with poor calibration, so losses accelerated.

## Gaps / Why P&L Is Negative
- **Calibration drift:** Settlement-based CatBoost remains overconfident/out of sync for some weeks (Brier >0.25). Without consistent calibration, even the 5¢ edge gate can’t prevent systematic losses.
- **EV model quality:** RMSE around ~30¢ is too high vs. the trading edge we need (fees alone ≈ 2–3¢). The EV model frequently mispredicts the short-term direction, causing serial losses despite the new risk cap.
- **Late-session coverage:** The prior-peak window cuts the dataset by ~40–50%. When the actual peak shifts (fronts, rain), yesterday’s timing is a poor proxy, leading to sparse/noisy training samples.
- **Execution gating:** Even with the 2% bankroll cap and 5¢ edge threshold, the strategy is still entering trades that realize negative EV because the underlying probabilities are misaligned with actual market drift.

## Proposed Next Steps
1. **Improve EV model signal**
   - Try shorter horizons (e.g., 15–30 minutes) where the price path is more predictable; compute multi-horizon labels so the policy can choose based on time-to-close.
   - Incorporate NOAA/VC forecast features (future TMAX/TMIN, hourly predictions) so the model can anticipate tomorrow’s peak rather than extrapolating from yesterday.
   - Use quantile regression or uncertainty estimation to rank trades by confidence rather than raw point delta.

2. **Hybrid calibration**
   - Blend the EV output with the settlement probabilities (e.g., rescale EV deltas using daily Brier/log-loss diagnostics) to stabilize the edge. Ensemble predictions might reduce the overconfidence seen in single models.

3. **Execution filters**
   - Introduce a dual gate: (a) EV delta ≥ fees + cushion; (b) settlement probability edge still positive (so we don’t trade against the daily view). Only allow trades when both align.
   - Consider maker-only fill assumption for EV-based trades (resting orders closer to market close) to reduce fee drag.

4. **Extended backtest coverage**
   - Run the new EV pipeline over a longer window (e.g., May–August 2025) to identify which weeks degrade Sharpe; use diagnostics to auto-reject windows with high RMSE/Brier before the policy consumes them.

5. **Optional future work**
   - Implement a rolling retrain (online update) so the EV model adapts minute-by-minute rather than relying on a static walk-forward.
   - Evaluate RL-style policies that directly optimize reward instead of a two-step (predict → threshold) workflow.

## Summary
We now have the tooling to generate late-session features, minute-level EV labels, and Optuna-tuned CatBoost models, plus a backtester that can consume either settlement probabilities or EV predictions. However, the current EV model is too noisy—RMSE and calibration issues still drive Sharpe deeply negative despite stricter execution caps. The next focus should be improving the signal (shorter horizons, forecast features), combining probability and EV cues, and tightening trade gating to ensure positive net edge before scaling to other brackets/cities.

---

## 2025-11-18 Update — Intraday Tmax Ensemble

### What’s New
- `scripts/train_tmax_regressor.py` trains an ensemble of GBDT + spline extrapolator + GRU using only minute-level temps observed so far.
- Backtester supports `--model-type tmax_reg` with `--tmax-preds-csv <file>` to convert predicted highs into bracket probabilities, now with optional probability/sigma/hybrid gates.
- `scripts/run_tmax_batch.py` automates the entire pipeline for every city (training, ModelKelly backtest, optional daily baseline), making nightly cron refreshes trivial.
- Chicago run (2024-10-25→2025-11-16) achieves sub-2°F MAE by 4 PM and exports per-minute predictions to `results/tmax_preds_chicago.csv`.

### Current Backtest (no gating)
- City: Chicago
- Dates: 2024-10-25 to 2025-11-16
- Trades: 639
- P&L: −$9.7k, Sharpe −17.6
- Reason: probabilities fire on nearly every minute; need confidence gates (|pred − boundary| > kσ and `P(YES)` above fee-adjusted threshold).

- `backtest/run_backtest.py` now accepts `--tmax-min-prob` / `--tmax-sigma-multiplier`. Sweeps on Chicago show the combo (0.60, 0.75) trims trades from 639 → 3 with Sharpe improving from −17.6 to −1.4 and max drawdown shrinking from −57% → −6% (`results/backtest_chicago_tmax_gated_060_075.json`).
- Added optional hybrid gating via `--hybrid-model-type`/`--hybrid-min-prob`. Requiring agreement with the ElasticNet settlement probabilities filtered out all remaining Chicago trades in this window (`results/backtest_chicago_tmax_hybrid.json`), confirming the two models currently disagree whenever the Tmax ensemble wants to trade.
- `scripts/train_tmax_regressor.py` trains the GRU branch, blends it at 30%, and writes per-row sigma estimates learned from a HistGBR. The CSV exposes `pred_seq` and dynamic `sigma_est` for gating.
- Added `scripts/backtest_tmax_daily.py` to run the “trade once at 16:00” sanity check and archive the per-day trades.
- Added `scripts/run_tmax_batch.py` so nightly cron/systemd jobs can refresh `results/tmax_preds_<city>.csv`/`backtest_<city>_tmax.json` for every market without hand-holding.

### Immediate Next Steps
1. Blend the hybrid gating signal with settlement models (ElasticNet / CatBoost) to decide which disagreements should still trade vs. be ignored. Investigate why Chicago has zero overlapping trades and whether parameter tweaks or refreshed settlement models yield overlap.
2. Refresh the settlement-model walk-forward runs so hybrid gating has meaningful overlap, then document/monitor Sharpe per city.
3. Expand monitoring: integrate cron/log outputs from `scripts/run_tmax_batch.py` + settlement poller into a single daily health report.

See `docs/TMAX_FORECAST_PIPELINE.md` and `docs/NEXT_STEPS_2025-11-18.md` for detailed commands and roadmap.
