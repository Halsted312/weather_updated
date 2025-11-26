
# Weather Bracket Agent – Design & Implementation Guide
**Date:** 2025-11-19

**Goal**  
Build a dual‑mode (paper/live) trading agent for Kalshi **“Highest temperature today”** markets that:
1) ingests **Visual Crossing** 5‑minute station data for the settlement station;  
2) ingests **Kalshi** market data (candles now; WebSocket L2 later) for brackets of a city;  
3) computes coherent, calibrated **per‑bracket probabilities**;  
4) detects **cross‑bracket acceleration** and order‑book momentum;  
5) executes **maker‑first** with fee‑aware taker switches;  
6) supports **backtest + shadow trading** before going live.

> This doc focuses on wiring, data flows, feature/strategy definitions, fee‑aware execution, backtesting and calibration. It assumes your existing Visual Crossing client is working—keep it and wire it into the new stack.

---

## 1) Exchange & Weather Semantics (in 60 seconds)

- **Kalshi markets** are binary. For a bracket market (e.g., “83–84°F”), the **YES** price ≈ market‑implied probability that the daily high ends in that bracket.  
- The **order book** returns **YES** bids and **NO** bids; **asks are implied** (YES‑ask = 100 − NO‑bid, NO‑ask = 100 − YES‑bid).  
- **WebSocket** streaming sends an initial **snapshot** then **incremental deltas**; you can subscribe to many tickers in one stream.  
- **Candlesticks** are available at **1‑minute** granularity (per market, per event, and multi‑event). Use these for historical backtests immediately.  
- **Visual Crossing** Timeline API supports **sub‑hourly** via `aggregateMinutes=5` for the **specific station** (use `stn:<ID>`). You’ll compute running Tmax and a “new‑high hazard” from this 5‑minute grid.

**Why candles now?** You don’t need to wait weeks to collect WebSocket L2. Use 1‑minute OHLCV + trade prints to backtest the whole strategy; in parallel, start recording L2 for later queue‑aware fills and microstructure features.

---

## 2) Architecture (docker‑compose)

**Services**

- `vc_ingest` – Visual Crossing 5‑minute station obs/forecast → DB (reuse your working client).  
- `candles_ingest` – Kalshi **1‑minute** candlesticks (per event or per market) and optional trade prints → DB.  
- `features` – stream processor that builds minute features (kinematics, cross‑bracket deltas, weather hazard).  
- `pmf_filter` – logistic‑normal / logit‑pool fusion of market vs weather PMFs + calibration.  
- `signals` – cross‑bracket **acceleration** signal with hazard gating and volume/liquidity gates.  
- `exec` – fee‑aware execution: **maker first**, taker when projected edge ≥ (fees + slippage). Dual mode via `TRADE_MODE=paper|live`.  
- `backtest` – candles+trades backtester with fill model and fees.  
- `postgres` – storage (candles, trades, features, P&L, calibration curves).

**.env (example)**
```ini
CITY=chicago
STATION_ID=KMDW
VC_API_KEY=...
KALSHI_API_KEY=...
TRADE_MODE=paper               # paper|live
DB_URL=postgres://user:pass@postgres:5432/kalshi
LOG_LEVEL=INFO

# thresholds / knobs
MAKER_THRESH_CENTS=1.5
TAKER_THRESH_CENTS=3.0
MAX_DAILY_LOSS=300.0

# Monte Carlo
MC_PATHS=4000
```

**docker‑compose.yml (skeleton)**
```yaml
version: "3.9"
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: kalshi
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]

  vc_ingest:
    build: .
    command: python -m agent.data.visualcrossing_loop
    environment: [DB_URL=${DB_URL}, VC_API_KEY=${VC_API_KEY}, CITY=${CITY}, STATION_ID=${STATION_ID}]
    depends_on: [postgres]

  candles_ingest:
    build: .
    command: python -m backtest.candles_backfill --events auto --days 60 --out /data/candles.parquet
    volumes: ["./data:/data"]
    environment: [DB_URL=${DB_URL}, KALSHI_API_KEY=${KALSHI_API_KEY}]
    depends_on: [postgres]

  features:
    build: .
    command: python -m agent.features.run --candles /data/candles.parquet --wx-table wx.minute_obs_5m
    volumes: ["./data:/data"]
    environment: [DB_URL=${DB_URL}]
    depends_on: [candles_ingest]

  pmf_filter:
    build: .
    command: python -m agent.models.filter_loop --city ${CITY}
    environment: [DB_URL=${DB_URL}]
    depends_on: [features]

  signals:
    build: .
    command: python -m agent.signals.accel_loop --city ${CITY}
    environment: [DB_URL=${DB_URL}]
    depends_on: [pmf_filter]

  exec:
    build: .
    command: python -m agent.execution.engine --mode ${TRADE_MODE} --city ${CITY}
    environment: [DB_URL=${DB_URL}, KALSHI_API_KEY=${KALSHI_API_KEY},
                  MAKER_THRESH_CENTS=${MAKER_THRESH_CENTS}, TAKER_THRESH_CENTS=${TAKER_THRESH_CENTS}]
    depends_on: [signals]

volumes:
  pgdata: {{}}
```

---

## 3) Data Contracts (tables / persisted artifacts)

**Candles** `md.candles_1m`  
`ts, market, open_c, high_c, low_c, close_c, volume, event_ticker, city`

**Trades** `md.trades`  
`ts, market, price_c, qty, side`

**Weather 5‑min** `wx.minute_obs_5m`  
`ts, station_id, temp_f, dew_f, rh, wind_mph, conditions`

**Features (minute)** `feat.minute_panel` (wide or long)  
Per bracket: `mid` (YES mid), `v` (velocity), `a` (acceleration), `clv` (close‑location in bar), `spread_est`, `vol_norm`, cross‑bracket deltas, etc.  

**PMF** `pmf.minute`  
`ts, bracket, p_mkt, p_wx, p_fused, p_calibrated`

**Signals** `sig.minute`  
`ts, bracket, ras, hazard, decision, confidence`

**P&L** `bt.pnl`  
`ts, trade_id, market, side, qty, px_c, fee_c, mode, pnl_after_fees`

---

## 4) Feature Set (minute clock)

### 4.1 Kinematics & microstructure (from candles + trades)
- **Mid**: YES mid from implied asks (or simply use close_c / 100 for minute mid proxy).  
- **Velocity / Acceleration**: EWM finite differences or Savitzky–Golay on the last 12–24 bars.  
- **Close‑location value (CLV)**: (close − low) / max(high − low, 1e‑6).  
- **Spread estimate**: moving percentile of (high − low) and occasional REST orderbook samples to anchor regimes.  
- **Volume gates**: minute volume percentile and open‑interest change (if available).

### 4.2 Cross‑bracket coupling
- **Relative Acceleration Score (RAS)** for bracket j:  a_j − mean(a_{j−1}, a_{j+1}).  
- **Lead/lag**: short rolling cross‑correlation in d(m) with adjacent brackets.  
- **Mass conservation**: sum of per‑bracket mid probs ≈ 1; track where mass flows.  
- **Edge proximity**: distance between current running high and bracket interval; favors bins adjacent to the current high.

### 4.3 Weather hazard (5‑minute)
- Maintain running Tmax; compute time‑to‑peak prior from climatology (optional).  
- Short‑term slope nowcast (Kalman or robust regression on last N obs).  
- **Monte Carlo**: simulate rest‑of‑day temps (5‑min granularity) using slope+noise with diurnal envelope; estimate probability of new high and map the Tmax distribution to bracket PMF.  
- Expose hazard(t, Δ) = chance of new high in next Δ minutes and p_wx across brackets.

---

## 5) PMF Fusion & Calibration

### 5.1 Fusion
We have two sources: market‑implied p_mkt (from prices) and weather‑implied p_wx (from MC). We want a coherent vector on the simplex.
- **Logit‑space pooling** (log‑opinion pool): weight by flow intensity and hazard confidence.  
- **Logistic‑normal filter**: treat log‑ratios as Gaussian, update with new evidence, then map back to simplex.

### 5.2 Calibration
- Start with **Platt (sigmoid)** or **Isotonic** per bracket (or grouped), then consider **Beta calibration** if needed.  
- Calibrate on rolling windows; track **Brier** and **ECE**.  
- Keep **per‑city** calibrators; retrain monthly or on drift detection.

---

## 6) Signals & Execution

### 6.1 Entry logic (example gates)
- **Acceleration**: RAS_j > threshold and v_j > 0 (for longs).  
- **CLV**: > 0.6 for buys (< 0.4 for sells).  
- **Hazard**: elevated when we expect upward moves near peak time.  
- **Liquidity**: spread < 5¢; volume percentile high; avoid empty books.

### 6.2 Exit / lifecycle
- Target holding 1–5 minutes unless trend strengthens.  
- Stop on change‑point detection or hazard collapse.  
- Intraday exposure caps per bracket and per city.

### 6.3 Fee‑aware execution
- **Maker first**: post at best; switch to taker **only** when projected next‑bar edge ≥ (fees + conservative slippage).  
- Implement fee formulas as functions; keep schedule in config so you can update without code changes.

---

## 7) Backtesting (candles + trades)

**Minute loop**  
For each minute t:
1) Build features; compute RAS, hazard, p_mkt, p_wx; fuse & calibrate → p(t).  
2) Generate decision; compute expected edge for t+1.  
3) **Fill model**  
   - Maker: filled if limit ∈ [low, close] of minute t+1 **and** volume ≥ qty (use trades if available to pro‑rate).  
   - Taker: filled at close(t) + slippage (spread proxy).  
4) Apply **fees**, update P&L and risk.

**Metrics**  
P&L after fees, turnover, hit‑rate on bracket “winners”, Brier/ECE (before/after calibration), drawdown, maker/taker split.

---

## 8) Shadow & Live

- **Shadow**: in live markets, log hypothetical orders and fills, but do not send. Compare shadow P&L to backtest.  
- **Kill switches**: daily loss cap, cancel‑on‑disconnect, liquidity outage detection.  
- **Promotion**: only flip `TRADE_MODE=live` after sustained shadow profitability and stability.

---

## 9) Suggested Package Layout

```
agent/
  config.py
  data/
    visualcrossing_loop.py     # poll and write to DB (reuse your client)
    kalshi_ws.py               # later, L2 snapshot→delta recorder
  features/
    kinematics.py
    cross_bracket.py
    weather_hazard.py
  models/
    pmf_filter.py
    calibration.py
    monte_carlo.py
  signals/
    accel_signal.py
  execution/
    fees.py
    engine.py
  backtest/
    candles_backfill.py
    trades_pull.py
    features_from_candles.py
    fillsim.py
    run.py
```

---

## 10) Code Stubs (drop‑in)

> These are production‑style, typed stubs. Replace imports to match your repo. Your agent can flesh them out quickly.

### 10.1 Config
```python
# agent/config.py
from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    city: str = os.getenv("CITY", "chicago")
    station_id: str = os.getenv("STATION_ID", "KMDW")
    trade_mode: str = os.getenv("TRADE_MODE", "paper")  # 'paper' or 'live'
    db_url: str = os.getenv("DB_URL", "postgres://user:pass@postgres:5432/kalshi")
    vc_api_key: str = os.getenv("VC_API_KEY", "")
    kalshi_key: str = os.getenv("KALSHI_API_KEY", "")
    ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    mc_paths: int = int(os.getenv("MC_PATHS", "4000"))
    mc_step_minutes: int = 5
    delta_seconds: int = int(os.getenv("DELTA_SECONDS", "20"))
    maker_thresh_cents: float = float(os.getenv("MAKER_THRESH_CENTS", "1.5"))
    taker_thresh_cents: float = float(os.getenv("TAKER_THRESH_CENTS", "3.0"))
    max_daily_loss: float = float(os.getenv("MAX_DAILY_LOSS", "300.0"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
```

### 10.2 Visual Crossing (reuse your client; parameters shown)
```python
# agent/data/visualcrossing_loop.py
import time, httpx, datetime as dt
from typing import Dict, Any
from agent.config import Config

BASE = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

def fetch_station_today(cfg: Config) -> Dict[str, Any]:
    params = {
        "unitGroup": "us",
        "include": "obs,fcst",
        "elements": "datetime,temp,dew,humidity,windspeed,conditions",
        "aggregateMinutes": "5",
        "combinationMethod": "best",
        "maxStations": "1",
        "key": cfg.vc_api_key,
        "contentType": "json",
    }
    url = f"{BASE}/stn:{cfg.station_id}/today"
    r = httpx.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def loop():
    cfg = Config()
    while True:
        js = fetch_station_today(cfg)
        # TODO: upsert obs & forecast into DB
        time.sleep(300)  # 5 minutes
```

### 10.3 Candles backfill (per event preferred)
```python
# backtest/candles_backfill.py
import httpx, time, math
from typing import List, Tuple, Dict
from datetime import datetime, timedelta, timezone

BASE = "https://api.elections.kalshi.com/trade-api/v2"

def get_event_candlesticks(event_ticker: str, start_ts: int, end_ts: int, period_min: int = 1):
    url = f"{BASE}/events/{event_ticker}/candlesticks"
    params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_min}
    r = httpx.get(url, params=params, timeout=30); r.raise_for_status()
    js = r.json()
    return js["market_tickers"], js["market_candlesticks"], js.get("adjusted_end_ts")
```

### 10.4 Feature builder (from candles)
```python
# backtest/features_from_candles.py
import numpy as np
import pandas as pd

def close_location(high_c, low_c, close_c):
    rng = np.maximum(high_c - low_c, 1e-6)
    return (close_c - low_c) / rng

def build_panel(candles_df: pd.DataFrame, window:int=24) -> pd.DataFrame:
    """
    Input columns: ts, market, open_c, high_c, low_c, close_c, volume
    Returns a multiindex columns panel with mid, v, a, clv per market.
    """
    g = []
    for m, df in candles_df.groupby("market"):
        df = df.sort_values("ts").copy()
        df["mid"] = df["close_c"] / 100.0
        df["v"] = df["mid"].diff().ewm(alpha=0.3, adjust=False).mean()
        df["a"] = df["v"].diff().ewm(alpha=0.3, adjust=False).mean()
        df["clv"] = close_location(df["high_c"], df["low_c"], df["close_c"])
        g.append(df.set_index("ts")[["mid","v","a","clv"]].rename(columns=lambda c: (m, c)))
    panel = pd.concat(g, axis=1).sort_index()
    return panel
```

### 10.5 Cross‑bracket features
```python
# agent/features/cross_bracket.py
import numpy as np
import pandas as pd

def relative_accel(row: pd.Series, markets: list[str]) -> dict[str, float]:
    ras = {}
    for i, m in enumerate(markets):
        a = row[(m, "a")]
        nbrs = []
        if i > 0: nbrs.append(row[(markets[i-1], "a")])
        if i < len(markets)-1: nbrs.append(row[(markets[i+1], "a")])
        ras[m] = float(a - (np.mean(nbrs) if nbrs else 0.0))
    return ras
```

### 10.6 PMF fusion (simple logit‑pool)
```python
# agent/models/pmf_filter.py
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()

def fuse_pmf(p_mkt: np.ndarray, p_wx: np.ndarray, w_mkt: float = 0.6, w_wx: float = 0.4) -> np.ndarray:
    p_mkt = np.clip(p_mkt, 1e-6, 1-1e-6); p_mkt /= p_mkt.sum()
    p_wx  = np.clip(p_wx, 1e-6, 1-1e-6);  p_wx  /= p_wx.sum()
    logit_mkt = np.log(p_mkt)  # ignoring (1-p) since multinomial simplex
    logit_wx  = np.log(p_wx)
    return softmax(w_mkt * logit_mkt + w_wx * logit_wx)
```

### 10.7 Calibration wrapper (placeholders)
```python
# agent/models/calibration.py
import numpy as np
from typing import Literal

def apply_calibration(p: np.ndarray, method: Literal["none","platt","isotonic","beta"]="none") -> np.ndarray:
    # placeholder: wire scikit-learn CalibratedClassifierCV or your own fit curves
    return p
```

### 10.8 Fee model
```python
# agent/execution/fees.py
def taker_fee_cents(price_cents:int, contracts:int)->int:
    # fees = round_up(0.07 * C * P * (1-P)), P in dollars
    P = price_cents / 100.0
    fee = 0.07 * contracts * P * (1 - P)
    return int(fee*100 + 0.999)  # cents, rounded up

def maker_fee_cents(price_cents:int, contracts:int)->int:
    # fees = round_up(0.0175 * C * P * (1-P)), where applicable
    P = price_cents / 100.0
    fee = 0.0175 * contracts * P * (1 - P)
    return int(fee*100 + 0.999)
```

### 10.9 Execution gate (fee‑aware maker/taker)
```python
# agent/execution/engine.py
from typing import Dict

def choose_execution(mid_now:float, mid_pred:float, price_cents:int, contracts:int,
                     maker_thresh:float, taker_thresh:float, style:str="maker+taker")->Dict:
    exp_move_cents = int(round((mid_pred - mid_now)*100))
    if "maker" in style and abs(exp_move_cents) >= maker_thresh:
        return {"type":"limit","side":"buy" if exp_move_cents>0 else "sell",
                "price_cents": price_cents, "qty": contracts}
    if "taker" in style and abs(exp_move_cents) >= taker_thresh:
        return {"type":"market","side":"buy" if exp_move_cents>0 else "sell",
                "qty": contracts}
    return {}
```

### 10.10 Backtest loop (minute)
```python
# backtest/run.py
import pandas as pd, numpy as np
from agent.models.pmf_filter import fuse_pmf
from agent.features.cross_bracket import relative_accel

def run_backtest(panel: pd.DataFrame, bins: list[tuple[int,int]], price_now: dict[str,int]):
    markets = sorted({m for m,_ in panel.columns})
    pnl = 0.0
    for ts, row in panel.iterrows():
        p_mkt = np.array([row[(m,"mid")] for m in markets], dtype=float)
        p_mkt = p_mkt / p_mkt.sum()
        # placeholder weather PMF/hazard
        p_wx  = np.ones_like(p_mkt) / len(p_mkt)
        p = fuse_pmf(p_mkt, p_wx, 0.6, 0.4)
        ras = relative_accel(row, markets)
        tkr = max(ras, key=lambda k: ras[k])
        # naive: buy 1 taker if ras>0 and clv>0.6
        if row[(tkr,"a")] > 0 and row[(tkr,"clv")] > 0.6:
            pnl -= (price_now.get(tkr, int(row[(tkr,"mid")]*100))) / 100.0
        # exit next minute at close (toy); plug in fillsim
    return pnl
```

---

## 11) Validation & Risk

- **Unit tests** for fee math, PMF fusion normalization, maker fill rules.  
- **Shadow days**: record decisions & hypothetical fills; verify fee‑aware P&L.  
- **Risk controls**: daily loss cap, per‑bracket exposure, cancel‑on‑disconnect, WS reconnect snapshot.

---

## 12) References (for the agent to consult)

- Kalshi API – Candlesticks (market & event), WebSockets (snapshot→delta), Orderbook semantics; Rate limits.  
- Visual Crossing – Timeline API, sub‑hourly `aggregateMinutes=5`, station `stn:<ID>`.  
- Calibration – Platt/Isotonic/Beta; scikit‑learn docs.  
- (Optional) Microstructure – order‑flow imbalance (OFI), Hawkes‑style intensity for bursts.

> Keep the fee schedule and station mapping in config so you can update without code changes.
