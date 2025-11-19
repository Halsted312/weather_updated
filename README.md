# Kalshi Weather Trading Bot

A reproducible data + backtest pipeline for Kalshi's weather prediction markets with ML-driven strategies.

## Overview

This project builds a complete trading system for Kalshi weather markets:
1. **Data Ingestion**: Fetch historical market data (candlesticks, trades) and NOAA weather observations
2. **Database**: Store everything in PostgreSQL for efficient querying
3. **Backtesting**: Fee-aware simulation with realistic execution assumptions
4. **ML Models**: Calibrated probability estimates using Ridge/Lasso + Platt scaling
5. **Multi-City**: Chicago, Miami, Austin, LA, Denver, Philadelphia  
   > NYC (KXHIGHNY) is permanently excluded due to unreliable sub-hourly weather data.

## Project Structure

```
kalshi_weather/
├── kalshi/          # Kalshi API client & schemas
├── ingest/          # Data ingestion & loading
├── weather/         # NOAA weather data fetchers
├── db/              # Database models & loaders
├── backtest/        # Backtesting engine & strategies
├── models/          # ML features, training, calibration
├── scripts/         # Executable scripts
├── tests/           # Test suite
├── config/          # Configuration files
└── data/            # Local data cache (gitignored)
    ├── raw/         # Fetched parquet files
    └── results/     # Backtest results
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Kalshi API key (optional for public data, required for authenticated endpoints)

### 2. Setup

**Clone and navigate:**
```bash
cd /home/halsted/Documents/python/kalshi_weather/
```

**Move Docker Compose to docker directory:**
```bash
sudo mkdir -p /home/halsted/docker/kalshi_weather
sudo mv docker-compose.yml /home/halsted/docker/kalshi_weather/
sudo chown -R $USER:$USER /home/halsted/docker/kalshi_weather
```

**Install dependencies:**
```bash
make install
```

**Configure environment:**
```bash
cp .env.example .env
# Edit .env and add your KALSHI_API_KEY
```

**Start database:**
```bash
make db-up
```

**Run migrations:**
```bash
make db-migrate
```

### 3. Fetch Data

**Demo (7 days Chicago):**
```bash
make ingest-chicago-demo
```

**Full Chicago (100 days):**
```bash
make ingest-chicago-100d
```

**All cities (100 days):**
```bash
make ingest-all-cities
```

### 4. Load to Database

```bash
make load-to-db
```

### 5. Fetch Weather Data

```bash
make fetch-weather
```

### 6. Run Backtest

**Demo backtest:**
```bash
make backtest-demo
```

**Full backtest:**
```bash
make backtest-chicago
```

### Background Pollers

See `docs/SYSTEMD_JOBS.md` for instructions on enabling the 30-minute settlement poller via systemd. The provided service wraps `scripts/run_settlement_poller.sh`, which loops forever, reloads CLI/CF6 data every 1,800 seconds, and keeps the `wx.settlement` source of truth current.

## Development

### Running Tests

```bash
make test
```

### Linting

```bash
make lint
```

### Formatting

```bash
make format
```

### Database Management

**Reset database (WARNING: destroys all data):**
```bash
make db-reset
```

**Stop database:**
```bash
make db-down
```

**Access pgAdmin (optional):**
```bash
cd /home/halsted/docker/kalshi_weather
docker-compose --profile tools up -d pgadmin
# Visit http://localhost:5050 (admin@kalshi.local / admin)
```

## Makefile Commands

Run `make help` to see all available commands:

- `make init` - Initialize project (install + start Docker)
- `make ingest-chicago-100d` - Fetch 100 days Chicago data
- `make load-to-db` - Load data into database
- `make fetch-weather` - Get NOAA observations
- `make backtest-chicago` - Run backtest
- `make test` - Run tests
- `make clean` - Clean generated files

## Architecture

### Data Pipeline

1. **Kalshi API** → Fetch series, markets, candlesticks (1-minute OHLC)
2. **NOAA API** → Fetch daily Tmax observations
3. **PostgreSQL** → Store normalized data
4. **Feature Engineering** → Extract price/volume/weather signals
5. **ML Models** → Train calibrated probability estimators
6. **Backtest** → Simulate trading with fees & slippage
7. **Optimization** → Maximize Sharpe ratio

### Temperature Forecast Ensemble (New)

We now generate per-minute forecasts of the daily CLI high temperature using only information available up to that minute (no external forecasts).  This powers a new `tmax_reg` model type that converts the predicted high into bracket probabilities inside `ModelKellyStrategy`.

Components:

1. **GBDT Forecaster** – `HistGradientBoostingRegressor` on features such as running max/min, rolling means, slopes, and prior-day highs.
2. **Spline Extrapolator** – Smooths the intraday path and extrapolates the remaining hours; values are clipped to realistic bands.
3. **Sequence Model** – A lightweight GRU trained on the last three hours of temperature/running-max history (on GPU when available) that captures intraday curvature.

Usage:

```bash
# Train ensemble + export per-minute predictions (example: Chicago)
python scripts/train_tmax_regressor.py \
  --city chicago \
  --start 2024-10-25 \
  --end   2025-11-16 \
  --cutoffs 12:00 14:00 16:00 18:00 \
  --export-csv results/tmax_preds_chicago.csv

# Backtest ModelKelly with the Tmax ensemble probabilities
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
  --hybrid-model-type elasticnet \
  --hybrid-min-prob 0.50 \
  --initial-cash 10000 \
  --output-json results/backtest_chicago_tmax.json
```

The `results/tmax_preds_<city>.csv` file contains `timestamp`, ensemble predictions (`pred`), component predictions (including the GRU output), residuals, and a data-driven `sigma_est` column learned from minute-level residuals.  `ModelKellyBacktestStrategy` now accepts `--model-type tmax_reg` and `--tmax-preds-csv` to convert these forecasts into bracket probabilities via a Gaussian CDF.

Use `--tmax-min-prob` to require a minimum confidence (e.g., `0.60`) and `--tmax-sigma-multiplier` to insist the predicted high be at least `k·sigma` away from the nearest bracket boundary. Add `--hybrid-model-type elasticnet|catboost|ev_catboost` plus `--hybrid-min-prob` to require agreement with the settlement models before trading. These gates dramatically reduce over-trading and can be tuned per city/date range.

For a daily baseline that trades once per day at a fixed cutoff, run:

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

### Multi-City / Nightly Batch

Drive the entire Tmax pipeline (training, ModelKelly backtest, optional daily baseline) for every configured city via `scripts/run_tmax_batch.py`. This script is cron-friendly and will emit the same CSV/JSON artifacts you would create manually:

```
python scripts/run_tmax_batch.py \
  --cities all \
  --start 2024-10-25 \
  --end   2025-11-16 \
  --cutoffs 12:00 14:00 16:00 18:00 \
  --tmax-min-prob 0.60 \
  --tmax-sigma-multiplier 0.75 \
  --hybrid-model-type elasticnet \
  --hybrid-min-prob 0.50 \
  --run-daily-baseline
```

Each city receives `results/tmax_preds_<city>.csv`, `results/backtest_<city>_tmax.json`, and (optionally) a daily baseline JSON/CSV in `results/`. Add a nightly cron like the following to keep the ensemble fresh:

```
0 3 * * * cd /home/halsted/Documents/python/kalshi_weather && \
  /usr/bin/python scripts/run_tmax_batch.py --cities all --start $(date -d '25 days ago' +%Y-%m-%d) --end $(date -d 'yesterday' +%Y-%m-%d) >> logs/tmax_batch.log 2>&1
```

### Database Schema

- `series` - Market series metadata
- `markets` - Individual market contracts
- `candles` - 1-minute OHLC data (yes_bid, yes_ask, price, volume, OI)
- `trades` - Trade prints (fallback)
- `weather_observed` - Daily Tmax from NOAA

### Backtest Assumptions

- **Execution**: Cross spread (buy at `yes_ask_close`, sell at `yes_bid_close`)
- **Fees**:
  - Taker: `ceil(0.07 * C * P * (1-P))`
  - Maker: `ceil(0.0175 * C * P * (1-P))`
  - No settlement fee
- **Slippage**: Configurable (default 1 cent)
- **Liquidity**: Skip minutes below volume threshold

## Configuration

### Cities & Stations

Configured in `config/cities.yaml`:

| City | Series Ticker | NOAA Station |
|------|---------------|--------------|
| Chicago | KXHIGHCHI | GHCND:USW00014819 (Midway) |
| Miami | KXHIGHMIA | GHCND:USW00012839 (Miami Airport) |
| Austin | KXHIGHAUS | GHCND:USW00013958 (Austin Airport) |
| LA | KXHIGHLAX | GHCND:USW00023174 (LAX) |
| Denver | KXHIGHDEN | GHCND:USW00003017 (Denver Airport) |
| Philadelphia | KXHIGHPHIL | GHCND:USW00013739 (Philly Airport) |

## API References

- [Kalshi API Documentation](https://docs.kalshi.com/)
- [NOAA NCEI Data Service](https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation)

## Roadmap

- [x] Phase 0: Scaffolding
- [ ] Phase 1: Kalshi data ingestion (Chicago + all cities)
- [ ] Phase 2: Database schema & loaders
- [ ] Phase 3: NOAA weather data
- [ ] Phase 4: Backtest engine
- [ ] Phase 5: ML models & calibration
- [ ] Phase 6: Production deployment & monitoring

## License

MIT

## Contributing

This is a personal project. See [CLAUDE.md](CLAUDE.md) for development guidelines.
