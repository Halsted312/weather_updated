# Weather Bracket Agent – README

This doc is the short version of `docs/AGENT_DESIGN.md`. Use it when you just need the key command cheatsheet for the hazard → PMF fusion → modeling stack.

## Core Stack

- **Visual Crossing ingest** (`ingest/backfill_visualcrossing.py`, `ingest/poll_visualcrossing.py`)
- **Kalshi ingest** (`scripts/discover_*.py`, `ingest/load_kalshi_data.py`)
- **Hazard Monte Carlo** (`scripts/hazard_mc.py`)
- **PMF fusion** (`scripts/pmf_fusion.py`)
- **Cross-bracket modeling** (`scripts/train_cross_bracket.py`)
- **Settlement poller** (`scripts/poll_settlements.py` + `scripts/run_settlement_poller.sh`)

## Quick Workflow

```bash
# 0) env + db
pip install -e .
cp .env.example .env  # fill CITY, VC/Kalshi keys, DB_URL
make db-up
make db-migrate

# 1) ingest Kalshi parquet + load into Postgres
make ingest-chicago-100d
make load-to-db

# 2) Visual Crossing minutes
make backfill-wx                # or backfill-wx-demo for smoke test

# 3) Hazard Monte Carlo
python scripts/hazard_mc.py fit-params --city chicago --start-date 2024-09-01 --end-date 2024-12-01
python scripts/hazard_mc.py backfill --city chicago --start-date 2024-12-01 --end-date 2024-12-15

# 4) PMF fusion
python scripts/pmf_fusion.py backfill --city chicago --start-date 2024-12-01 --end-date 2024-12-15

# 5) Cross-bracket modeling
python scripts/train_cross_bracket.py \
  --city chicago \
  --start-date 2024-11-15 \
  --end-date 2024-12-15 \
  --horizon-min 1 \
  --model logreg \
  --epsilon 0.005 \
  --export-val results/chicago_val_preds.csv

# 6) Hazard diagnostics / PMF sanity checks (optional)
python scripts/hazard_mc_diagnostics.py check-pmf-sum --city chicago --date 2024-12-10
python scripts/hazard_mc_diagnostics.py evaluate-morning --city chicago --start-date 2024-11-20 --end-date 2024-12-15
```

## Files to Know

| File | Purpose |
|------|---------|
| `docs/AGENT_DESIGN.md` | full architecture spec (docker services, execution layer plan, etc.) |
| `notes/codex_session_2025-11-19.md` | current state + next actions for Phase 4 and beyond |
| `scripts/hazard_mc.py` | fit/backfill hazard Monte Carlo + hazards |
| `scripts/hazard_mc_diagnostics.py` | PMF mass + calibration diagnostics |
| `scripts/pmf_fusion.py` | logit-pool of market/weather PMFs |
| `scripts/train_cross_bracket.py` | short-horizon modeling scaffold |
| `scripts/check_phase4_coverage.py` | coverage audit for markets/weather/settlements |
| `scripts/run_settlement_poller.sh` | systemd-friendly CLI/CF6 poller loop |

## Notes

- Older Ridge/Lasso datasets, NOAA-only workflows, or `ml.*` packages now live in `docs/legacy/` and are no longer part of the live agent.
- Hazard MC currently uses a template baseline; swapping to VC timeline forecasts is Phase 4.T work.
- Fusion weights are still heuristic – see `notes/codex_session_2025-11-19.md` for the tuning plan.
