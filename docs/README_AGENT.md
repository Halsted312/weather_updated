
# Weather Bracket Agent – README

This repository contains a **docker‑compose** stack and code stubs to trade Kalshi **daily‑high temperature** brackets using:
- Visual Crossing **5‑minute station** data
- Kalshi **1‑minute candlesticks** (backtest now), WebSocket L2 later
- Cross‑bracket **acceleration** + weather **hazard** signals
- **Maker‑first**, fee‑aware execution
- **Dual mode**: `TRADE_MODE=paper|live`

## Quick Start

1) Create `.env`:
```ini
CITY=chicago
STATION_ID=KMDW
VC_API_KEY=...
KALSHI_API_KEY=...
TRADE_MODE=paper
DB_URL=postgres://user:pass@postgres:5432/kalshi
```

2) Launch:
```bash
docker compose up --build
```

3) Backfill candles (if running standalone):
```bash
python -m backtest.candles_backfill --event-tickers <EVENT_TICKER> --days 60 --out data/candles.parquet
```

4) Run backtest:
```bash
python -m backtest.run --candles data/candles.parquet --calibration isotonic
```

5) Flip to live **only after** shadow performance is stable:
```bash
export TRADE_MODE=live
```

## Layout

See `docs/AGENT_DESIGN.md` for the detailed spec and code stubs. The key packages are:

```
agent/
  config.py
  data/ (vc loop, ws recorder)
  features/ (kinematics, cross‑bracket, hazard)
  models/ (pmf filter, calibration, monte‑carlo)
  signals/ (acceleration)
  execution/ (fees, engine)
  backtest/ (backfill, features, fill‑sim, runner)
docs/
  AGENT_DESIGN.md
```

## Notes

- Keep your existing Visual Crossing client; set `stn:<STATION_ID>` and `aggregateMinutes=5`.
- Start with **candles + trades**; begin recording WebSocket L2 in parallel for later microstructure upgrades.
- Always compute P&L **after fees** and respect daily loss caps.
