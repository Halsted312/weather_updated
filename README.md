# Kalshi Weather Trading Agent

A production-grade stack for trading Kalshi "highest temperature" brackets using Visual Crossing minutes, hazard-aware Monte Carlo, PMF fusion, and cross-bracket modeling.

## Overview

The current architecture (Phase 4) is built around these components:

1. **Data ingestion** – Kalshi series/markets/candles/trades plus Visual Crossing 5-minute station data.
2. **PostgreSQL storage** – Normalized schemas for markets (`markets`, `md.candles_1m`, `md.trades`), weather (`wx.*`), features (`feat.minute_panel_*`), and probability mass (`pmf.minute`).
3. **Hazard Monte Carlo** – `scripts/hazard_mc.py` fits AR(1) residuals and simulates future Tmax paths to populate `p_wx` and hazard scalars.
4. **PMF fusion** – `scripts/pmf_fusion.py` logit-pools market (`p_mkt`) and weather (`p_wx`) probabilities into `p_fused_norm` for each bracket.
5. **Cross-bracket modeling** – `scripts/train_cross_bracket.py` trains short-horizon classifiers directly from `feat.minute_panel_full` (which already joins kinematics + weather + fused PMF).
6. **Backtest / execution scaffolding** – Fee-aware fill modeling, maker-first execution, and systemd pollers keep settlements up to date.

For prior iterations (Ridge/Lasso datasets, NOAA-only fetchers, etc.) see `docs/legacy/`.

## Project Structure

```
.
├── backtest/          # Fee + fill modeling helpers
├── db/                # SQLAlchemy models and loaders
├── docs/              # Current design documents
│   └── legacy/        # Archived instructions from the pre-hazard stack
├── ingest/            # Kalshi + Visual Crossing ingestion/backfill scripts
├── kalshi/            # API client, schemas, strike parsing
├── scripts/           # Hazard MC, PMF fusion, coverage utilities, etc.
├── tests/             # Active smoke/unit tests
├── weather/           # Visual Crossing + settlement helpers (CLI/CF6/IEM)
└── notes/             # Session notes (e.g., `notes/codex_session_2025-11-19.md`)
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Kalshi API key (for authenticated endpoints)
- Visual Crossing API key

### 2. Setup

```bash
# clone and install
pip install -e .
cp .env.example .env  # fill in Kalshi/VC/API + DB settings
make db-up
make db-migrate
```

### 3. Ingest Kalshi market data

```bash
make ingest-chicago-demo      # 7-day sample
make ingest-chicago-100d      # 100-day Chicago backfill
make ingest-all-cities        # multi-city backfill
make load-to-db               # push parquet → Postgres
```

### 4. Backfill Visual Crossing minutes

```bash
make backfill-wx-demo         # 3 days for Chicago
make backfill-wx              # 100 days for all cities (includes VC station lock)
```

### 5. Fit and run Hazard MC

```bash
# fit residual parameters for Chicago (adjust range/city as needed)
python scripts/hazard_mc.py fit-params --city chicago --start-date 2024-09-01 --end-date 2024-12-01

# populate p_wx + hazard scalars for each minute
python scripts/hazard_mc.py backfill --city chicago --start-date 2024-12-01 --end-date 2024-12-15
```

### 6. Fuse PMFs

```bash
python scripts/pmf_fusion.py backfill --city chicago --start-date 2024-12-01 --end-date 2024-12-15
```

### 7. Train cross-bracket models

```bash
python scripts/train_cross_bracket.py \
  --city chicago \
  --start-date 2024-11-15 \
  --end-date 2024-12-15 \
  --horizon-min 1 \
  --model logreg \
  --epsilon 0.005
```

(Use `--export-val` to dump probabilities for calibration, run again with `--horizon-min 5`, etc.)

### 8. Backtesting (optional)

Backtest tooling is under `backtest/` and will plug into the fused PMF outputs and classifier probabilities. See `docs/AGENT_DESIGN.md` and `notes/codex_session_2025-11-19.md` for the Phase 5/6 plan.

### Background Pollers

The CLI/CF6 settlement poller can run continuously via systemd. See `docs/SYSTEMD_JOBS.md` and `init/systemd/kalshi-settlement-poller.service`. The service wraps `scripts/run_settlement_poller.sh`, which loads `.env`, polls every 1,800 seconds, and keeps `wx.settlement` current.

## Development Workflow

- `make test` – run pytest
- `make lint` – run ruff + mypy
- `make format` – run black
- `make clean` – drop caches/egg-info
- `make backfill-wx[-demo]` – Visual Crossing backfill helpers
- `make ingest-*` – Kalshi data discovery

Common database helpers:

```bash
make db-up
make db-down
make db-reset   # wipes volume!
```

## Architecture Details

### Data & Weather Sources

- **Kalshi API** – series/markets/candles/trades land in `series`, `markets`, `md.candles_1m`, and `md.trades` through `ingest/load_kalshi_data.py`.
- **Visual Crossing** – `ingest/backfill_visualcrossing.py` locks to `stn:<ICAO>` and writes `wx.minute_obs` (+ materialized 1-minute grid). Real-time polling lives in `ingest/poll_visualcrossing.py`.
- **Settlements** – `weather/iem_cli.py` + `scripts/poll_settlements.py` maintain `wx.settlement` for official Tmax.

### Hazard Monte Carlo (`scripts/hazard_mc.py`)

1. Pull `feat.minute_panel_with_weather` for a city/day.
2. Build a 5-minute baseline temperature path (template fallback today, VC timeline forecast later).
3. Fit/resample AR(1) residuals per minute-of-day bucket.
4. Simulate residual paths, compute `p_wx` per bracket plus `hazard_next_5m/60m`, and upsert into `pmf.minute`.

Diagnostics live in `scripts/hazard_mc_diagnostics.py` (PMF sum, morning calibration, hazard traces).

### PMF Fusion (`scripts/pmf_fusion.py`)

- Normalizes market-implied probabilities (`p_mkt`) per event/minute.
- Clamps/renormalizes `p_wx` and applies logit-pooling with hazard/volume-driven weights.
- Stores both raw (`p_fused`) and normalized (`p_fused_norm`) results alongside `p_mkt` in `pmf.minute`.

### Cross-Bracket Modeling (`scripts/train_cross_bracket.py`)

- Reads `feat.minute_panel_full`, which joins kinematics, neighbor deltas, weather metrics, hazard scalars, and fused PMF fields.
- Labels short-horizon moves (`mid_prob` vs `mid_prob_shift`, epsilon filter) per bracket.
- Splits by `local_date` (train/val/test), fits `logreg` or `gbdt`, reports Accuracy/AUC/Brier/ECE, and can export validation predictions for calibration.

Next phases (4.T/5/6) focus on fusion-weight tuning, calibration, signal/execution, and backtesting/shadow mode as captured in `notes/codex_session_2025-11-19.md` and `docs/AGENT_DESIGN.md`.

## Legacy Materials

Older Ridge/Lasso ML recipes, NOAA-only workflows, and miscellaneous docs were moved to [`docs/legacy/`](docs/legacy) for reference. They no longer drive the active hazard → PMF → modeling pipeline.

## License

MIT

## Contributing

Personal project; see `CLAUDE.md` for guidelines.
