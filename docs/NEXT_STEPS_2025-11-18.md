# Next Steps — 2025-11-18

This roadmap captures the highest-priority follow-ups after today’s temperature-ensemble work.  All dates are local (Chicago time).

## 1. Confidence Gating for `tmax_reg` *(Completed)*
- `ModelKellyBacktestStrategy` now accepts `--tmax-min-prob` and `--tmax-sigma-multiplier`. Chicago sweeps (`0.55–0.60`, `0.50–0.75`) reduced trades from 639 → 3–23 while shrinking drawdown from −57% to −6% (see `docs/TMAX_FORECAST_PIPELINE.md`). Continue to tune per city, but the scaffolding is done.

## 2. Neural Sequence Model (GPU) *(Completed)*
- `scripts/train_tmax_regressor.py` trains a GRU on the last 180 minutes of temps, blends it at 30% weight, and predicts a dynamic `sigma_est` via a lightweight HistGBR. CSVs now contain `pred_seq` and per-row sigmas.

## 3. Event-Level Baseline *(Completed)*
- `scripts/backtest_tmax_daily.py` trades once at a configurable cutoff (`--cutoff 16:00`), writing JSON + CSV summaries so we can benchmark the signal without minute-level execution noise.

## 4. Hybrid Probability Gating *(In Progress)*
1. `ModelKellyBacktestStrategy` now accepts `--hybrid-model-type`/`--hybrid-min-prob` to require agreement between `tmax_reg` and the settlement predictions. Chicago (ElasticNet, min prob 0.50) produced zero trades because the two signals disagree everywhere; this is a useful safety valve, but we need to refresh the settlement calibration before expecting overlap.
2. Next: rerun the walk-forward settlement models (ElasticNet + CatBoost) so their predictions cover the 2024-10 → 2025-11 window cleanly, then re-test hybrid gating to quantify how often the signals align and what Sharpe/P&L looks like when they do.

## 5. Documentation & Monitoring
1. Wire the hourly systemd poller logs (`journalctl -u kalshi-settlement-poller.service`) into a daily report: number of CLI rows inserted, lag vs. live clock, etc.
2. Update `project_status.md` once the gating experiments complete (include Sharpe/P&L per city for the winning configuration).

## 6. City Rollout *(Completed / automated)*
- `scripts/run_tmax_batch.py` loops through every entry in `CITY_CONFIG`, runs `train_tmax_regressor.py`, executes the ModelKelly backtest with the gating flags, and (optionally) produces the daily baseline JSON/CSV. Use `--cities all` + your preferred cutoffs to keep Philadelphia, Los Angeles, Miami, Austin, LA, Denver, etc. in lockstep with Chicago.

## 7. Sequence Model Nightly Job *(Completed)*
- The same batch script is cron-friendly; add it to systemd/cron alongside the settlement poller so `results/tmax_preds_<city>.csv` and the GRU weights refresh automatically. All outputs land under `results/` per city and can be archived or promoted as needed.

These tasks should keep the research pipeline moving without needing this chat session.  When ready to resume, start with Task 4 (hybrid gating), rerun the tmax backtests with the combined filters, and update the README/docs with the improved metrics.  Then progress down the list.  Ping future-you with questions via comments in the relevant scripts/documents.
