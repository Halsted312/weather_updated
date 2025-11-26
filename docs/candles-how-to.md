Short answer: **yes**—you can (and should) start with Kalshi’s **historical candlestick** and **trades** endpoints to backtest the cross‑bracket acceleration idea before you’ve accumulated your own WebSocket L2 tape. Candles give you 1‑minute OHLCV per bracket; you can engineer robust “acceleration” and cross‑bracket flow features, fuse them with your 5‑minute Visual Crossing station prior, calibrate, and run a credible backtest + shadow test. L2 is still valuable later for queue‑aware fills and finer alpha, but you don’t need to wait for it to begin. ([Kalshi API Documentation][1])

Below is a practical plan, trade‑offs, and **copy‑paste stubs** to stand this up quickly.

---

## What now exists in this repo

* **TimescaleDB warehouse (Docker).** `docker-compose.yml` runs `timescale/timescaledb:2.14.2-pg15`
  so hypertables/compression are available out of the box. The SQLAlchemy bootstrap
  (`infrastructure/database.py`) installs the extension, creates the tables, and promotes
  `market_candles` to a hypertable so historical and live flows share storage.
* **Schema that respects Kalshi’s time weirdness.** `infrastructure/models.py` keeps
  `bucket_start`, `bucket_end`, and a derived `local_close_date` (city timezone lookup lives in
  `core/datetime.py`, copied from the `samples_py_files` helper). This makes “next-day closes” and
  cross-city comparisons sane.
* **Async backfill + live poller.** `pipelines/candles_backfill.py` drives `/candlesticks` with
  cursor-aware chunks, upserts markets, and writes candles via `infrastructure/candles_store.py`
  (ON CONFLICT upserts). `backfill()` hydrates arbitrary ranges, while `live_tail()` keeps the last
  few hours rolling.
* **CLI entrypoints.**

  ```bash
  python -m kalshi_weather_momentum.cli candles-backfill WEATHER.CHIHI \
    --start 2024-06-01T00:00:00Z --end 2024-06-07T23:59:00Z
  python -m kalshi_weather_momentum.cli candles-live WEATHER.CHIHI --lookback-minutes 240
  ```

  Both commands reuse the async REST client, so they work the same in Docker or a venv and they’re
  idempotent (rerunning backfills simply updates overlapping rows).

---

## What you can (and can’t) learn from candles vs WebSocket L2

**Candles (1‑minute OHLCV) – good enough to:**

* Reconstruct minute‑to‑minute **mid** movement per bracket and compute **velocity/acceleration** and cross‑bracket *relative* acceleration (your idea). OHLC is explicitly provided for YES prices. ([Kalshi API Documentation][1])
* Track **volume clustering** and “close‑location” within the minute (close near high ⇒ net buy pressure), a coarse proxy for order‑flow. (OHLC definitions). ([Wikipedia][2])
* Do **coherent probability** modeling across brackets by renormalizing minute close prices to sum ≈ 1 and fusing with a weather prior.

**Candles won’t give you:**

* Queue position, best‑level **order‑flow imbalance** (OFI), or **who moved first** within the minute—those require L2 snapshot→delta. ([Kalshi API Documentation][3])
* **Fill quality** for maker orders. You can still simulate fills from candle high/low + volume (see fill model below), then refine later once you record L2.

**Trades endpoint** fills some gaps (timestamps, price, size for each trade) and is great to **augment** candles if you want intraminute VWAP and signed‑volume heuristics. ([Kalshi API Documentation][4])

---

## Recommended workflow (start today, no L2 history required)

1. **Pull historical 1‑minute candles per bracket.**

   * Use **Get Market Candlesticks** for each bracket *or* the newer event‑level `GET /candlesticks` to pull all brackets of an event in one shot. ([Kalshi API Documentation][1])
   * If you want intraminute granularity, also pull **Get Trades** for the same window. ([Kalshi API Documentation][4])

2. **Engineer minute features that mimic the L2 signals:**

   * **Mid, velocity, acceleration** (EWM derivatives on minute close; optional Savitzky–Golay for smoother acceleration).
   * **Close‑location value:** `(close - low) / (high - low)` ⇒ proxy for buying pressure in that minute.
   * **Cross‑bracket RAS (Relative Acceleration Score):** acceleration of bracket j minus weighted acceleration of its neighbors.
   * **Mass conservation:** sum brackets to ≈1; if not, renormalize the vector of minute closes (YES) to enforce coherence.
   * **Weather hazard gate:** combine with your Visual Crossing 5‑minute station prior to know when late new highs are likely (VC documents sub‑hourly and its normalization/aggregation). ([Visual Crossing][5])

3. **Probability fusion (coherent PMF):**

   * Convert bracket minutes to a probability vector (p^{mkt}); form weather PMF (p^{wx}) via Monte‑Carlo of rest‑of‑day Tmax.
   * Fuse with **log/odds pooling** (logarithmic opinion pool) or in logit space; weight more toward market when momentum/volume spikes, more toward weather when market is quiet. ([Visual Crossing][6])

4. **Calibration:**

   * If probabilities are mis‑calibrated on backtests, apply **Platt scaling** (sigmoid), **Isotonic Regression** (non‑parametric), or **temperature scaling** (logit rescale). Use cross‑validated fits and track Brier/ECE. ([Kalshi API Documentation][4])

5. **Backtest logic (minute clock):**

   * **Entry:** when RAS(j) is high **and** neighbor accelerations are negative **and** close‑location shows persistent buy (or sell) pressure; gate by weather **hazard** (higher hazard ⇒ lower threshold).
   * **Exit:** TP/SL in cents or opposite signal; square up near end‑of‑session or once hazard collapses.
   * **Fees:** net P&L **after** maker/taker math; if you later move to L2, you’ll prefer maker. (Kalshi fee schedule and API docs explain NO/YES reciprocity and orderbook conventions.) ([Kalshi API Documentation][7])

6. **Shadow test live** while you start recording L2 (WebSocket). You can run live in **paper mode** with candles/trades polling at 20 req/s and begin logging fills, then drop in L2 once you’ve collected a week+ of deltas. (WS gives snapshot then incremental deltas.) ([Kalshi API Documentation][8])

---

## How to fetch the history you need (API links)

* **Market Candlesticks** (OHLC for YES): `GET /markets/{ticker}/candlesticks` with `start_ts`, `end_ts`, and interval ⇒ minute bars. ([Kalshi API Documentation][1])
* **Event Candlesticks** (all brackets at once): `GET /candlesticks?event_ticker=...` (see changelog). Use paging if >5k bars. ([Kalshi API Documentation][9])
* **Trades**: `GET /trades` (paginate); filter by market ticker; you’ll get timestamp, price, size. ([Kalshi API Documentation][4])
* **Orderbook (optional for spot checks):** `GET /markets/{ticker}/orderbook` returns **YES bids + NO bids**; asks are implied (YES ask = 100 − NO bid). ([Kalshi API Documentation][7])
* **WebSocket** (for later, to record L2): connect once and subscribe to `orderbook_delta`; you’ll receive **snapshot then deltas**. ([Kalshi API Documentation][8])

With **20 req/s**, you can comfortably backfill many markets/minute bars and still have headroom for trades queries.

---

## Fill modeling without L2 (what to assume)

You can get surprisingly realistic fills from candles + trades:

* **Maker fill rule:** if your limit price is inside **[low, high]** of the *next* minute **and** (close‑location > 0.6 for buys / < 0.4 for sells) **and** minute volume ≥ your size, mark as filled; partial if partially touched.
* **Taker fill rule:** immediate at the minute close (or at open of next minute) + a slippage cushion (1–2¢) to be conservative.
* Add per‑bracket **spread penalty** in illiquid minutes (use `(high−low)` as an upper bound).
* Once you’ve recorded a few days of L2, **recalibrate** these rules to match observed maker vs taker fill rates.

---

## Code stubs (candles‑first backtester)

> These are minimal; plug into your repo. They use candles now, trades optional, and leave a hook for WebSocket later.

### 1) Candles downloader (event‑level preferred)

```python
# candles_dl.py
import httpx, time
from typing import Dict, List

BASE = "https://api.elections.kalshi.com/trade-api/v2"

def get_event_candles(event_ticker: str, start_ts: int, end_ts: int) -> List[dict]:
    """Pull all bracket candlesticks for an event; pages if needed."""
    # API changelog documents GET /candlesticks for an event.  # :contentReference[oaicite:17]{index=17}
    url = f"{BASE}/candlesticks"
    params = {"event_ticker": event_ticker, "start_ts": start_ts, "end_ts": end_ts, "interval_sec": 60}
    out = []
    while True:
        r = httpx.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        out.extend(js["candlesticks"])
        if "cursor" in js and js["cursor"]:
            params["cursor"] = js["cursor"]
        else:
            break
    return out

def get_market_candles(market_ticker: str, start_ts: int, end_ts: int) -> List[dict]:
    url = f"{BASE}/markets/{market_ticker}/candlesticks"  # OHLC for YES prices.  # :contentReference[oaicite:18]{index=18}
    params = {"start_ts": start_ts, "end_ts": end_ts, "interval_sec": 60}
    r = httpx.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()["candlesticks"]
```

### 2) Feature engineering from OHLCV (minute)

```python
# candle_features.py
import numpy as np
import pandas as pd

def close_location(high, low, close):
    rng = np.maximum(high - low, 1e-6)
    return (close - low) / rng

def accel(series, span_v=5, span_a=10):
    v = series.diff().ewm(span=span_v, adjust=False).mean()
    a = v.diff().ewm(span=span_a, adjust=False).mean()
    return v, a

def build_minute_features(df_by_bracket: dict):
    """
    df: per bracket DataFrame with columns: ['ts','open','high','low','close','volume']
    Returns synchronized panel with mid prob (close/100), vel, acc, clv.
    """
    # align on minute index
    idx = sorted(set().union(*[df.ts.values for df in df_by_bracket.values()]))
    feats = {}
    for b, df in df_by_bracket.items():
        d = df.set_index("ts").reindex(idx).ffill()
        p = d["close"] / 100.0
        v, a = accel(p)
        clv = close_location(d["high"], d["low"], d["close"])
        feats[b] = pd.DataFrame({"p": p, "v": v, "a": a, "clv": clv})
    # renormalize probabilities across brackets to enforce sum≈1
    panel = pd.concat({b: f for b, f in feats.items()}, axis=1)
    sump = panel.xs("p", axis=1, level=1).sum(axis=1).replace(0, np.nan)
    for b in feats:
        panel[(b, "p_norm")] = panel[(b, "p")] / sump
    return panel
```

### 3) Cross‑bracket acceleration signal + weather gate

```python
# signals_candles.py
import numpy as np

def ras(panel, bracket_keys):
    """Relative Acceleration Score per minute for each bracket against neighbors."""
    A = {b: panel[(b, "a")] for b in bracket_keys}
    R = {}
    for i, b in enumerate(bracket_keys):
        nbrs = []
        if i > 0:        nbrs.append(A[bracket_keys[i-1]])
        if i < len(bracket_keys)-1: nbrs.append(A[bracket_keys[i+1]])
        nbr_mean = np.mean(nbrs, axis=0) if nbrs else 0.0
        R[b] = A[b] - nbr_mean
    return R

def signal_row(row, ras_j, clv_j, hazard, edge_thresh=0.02):
    # Simple gate: RAS positive, CLV>0.6 (buy pressure), hazard elevated.
    return (ras_j > 0) and (clv_j > 0.6) and (hazard > 0.1)
```

### 4) Backtester (minute clock, fee‑aware)

```python
# backtest_candles.py
def simulate_trades(panel, probs_fused, fee_model, maker=True):
    pnl = 0.0
    positions = { }  # per bracket
    for t in panel.index[:-1]:
        # choose bracket by max RAS etc.; compute expected edge = (p_fused_next - p_now)
        ...
        # maker fill if next minute [low,high] crosses our price and volume sufficient
        # taker fill at close (or next open) with slippage
        pnl += realized_pnl_after_fees
    return pnl
```

---

## When you should switch to L2 (WebSocket) anyway

* Once your candle‑based model backtests well, start recording **WS snapshot→delta** to capture **OFI, queue imbalance and exact microstructure timing**. That will let you:

  * Increase maker usage (much lower fees) with realistic fill estimates.
  * Improve your **acceleration timing** from seconds to hundreds of milliseconds.

Kalshi’s WS docs show exactly how to subscribe once and maintain the book (snapshot then deltas). Use that while still trading off candles/trades; you don’t have to choose one or the other. ([Kalshi API Documentation][8])

---

## Practical pointers

* Prefer **event‑level candlesticks** during backfill to reduce calls and keep brackets aligned. (See changelog for `/candlesticks`.) ([Kalshi API Documentation][9])
* If you need a sanity check on NBBO at certain times, the **orderbook** endpoint tells you YES & NO bids; remember asks are implied (YES ask = 100 − NO bid). ([Kalshi API Documentation][7])
* Keep your **weather prior** in the loop—VC’s sub‑hourly docs explain how they normalize feeder station minutes into your requested 5‑minute grid, which is exactly what you want to compute late‑day “new‑high hazard.” ([Visual Crossing][5])

---

## Bottom line

* **Do it now with candles + trades.** You’ll get 70–90% of the signal you care about (cross‑bracket acceleration with hazard gating), plus credible fee‑aware P&L via a conservative fill model.
* **Record L2 in parallel** and upgrade your fill model + microstructure features when you have a few days of deltas.
* Your 20 req/s is ample for historical pulls and even frequent live polling if you need it. For true HFT‑style execution and maker optimization, L2 will still pay for itself. ([Kalshi API Documentation][10])

If you want, I can extend these stubs into a full `candles_backtest/` module with CLI entrypoints (`download`, `build-features`, `run-backtest`) and a schema for storing event‑level candles per city.

[1]: https://docs.kalshi.com/api-reference/market/get-market-candlesticks?utm_source=chatgpt.com "Get Market Candlesticks - API Documentation"
[2]: https://en.wikipedia.org/wiki/Open-high-low-close_chart?utm_source=chatgpt.com "Open-high-low-close chart"
[3]: https://docs.kalshi.com/websockets/orderbook-updates?utm_source=chatgpt.com "Orderbook Updates - API Documentation"
[4]: https://docs.kalshi.com/api-reference/market/get-trades?utm_source=chatgpt.com "Get Trades - API Documentation"
[5]: https://www.visualcrossing.com/resources/documentation/weather-api/requesting-sub-hourly-weather-data-using-the-weather-api/?utm_source=chatgpt.com "Requesting sub-hourly weather data using the Weather API"
[6]: https://www.visualcrossing.com/resources/documentation/weather-api/weather-api-documentation/?utm_source=chatgpt.com "Weather API Documentation (Legacy endpoints)"
[7]: https://docs.kalshi.com/api-reference/market/get-market-orderbook?utm_source=chatgpt.com "Get Market Orderbook - API Documentation"
[8]: https://docs.kalshi.com/getting_started/quick_start_websockets?utm_source=chatgpt.com "Quick Start: WebSockets - API Documentation"
[9]: https://docs.kalshi.com/changelog?utm_source=chatgpt.com "API Changelog - API Documentation"
[10]: https://docs.kalshi.com/?utm_source=chatgpt.com "Introduction - API Documentation - Kalshi"


## Extended Code Stubs

Absolutely—let’s turn the “candles‑first” approach into a complete, code‑ready module you can drop into your repo and run **today**, while you start recording WebSocket L2 in parallel.

Below you’ll find:

* A **long design doc** `docs/CANDLES_BACKTEST_AGENT.md`
* A **concise README** `backtest/README.md`
* **Extended code stubs** (downloaders, feature builder, PMF fusion, calibration, fill‑sim, backtester runner, and a simple live “paper” loop)
* A minimal **docker‑compose** and **Makefile** snippets

I anchor key claims to Kalshi/Visual Crossing docs where it matters (1‑minute candlesticks, event‑level multi‑candles, trades endpoint, WS snapshot→delta, YES/NO bids semantics, rate limits, VC 5‑minute sub‑hourly). Citations are inline.

---

## `docs/CANDLES_BACKTEST_AGENT.md`

> A practical agent that **does not need WebSocket history** to start. It uses Kalshi **1‑minute candlesticks** (per market or per event / multiple events) and **trade prints** for backtests and shadow trading; later you plug in WS L2 for queue‑aware fills & microstructure signals.

### Scope & data sources

* **Kalshi 1‑minute candlesticks** (YES OHLCV) for each bracket market. You can fetch per‑market or **per‑event (all brackets at once)**; both support **1‑minute** interval. ([Kalshi API Documentation][1])
* **Multiple‑events candlesticks** endpoint pulls aggregated candlesticks for **many events in one call** (cap ~5000 candlesticks per request). This is great for backfilling several cities’ brackets at once. ([Kalshi API Documentation][2])
* **Trades** endpoint returns timestamp, price, size (paginated with `cursor`) to reconstruct intraminute pressure and to **calibrate maker fill** assumptions. ([Kalshi API Documentation][3])
* **Orderbook snapshots** (optional) return **YES and NO bids only** (asks are implied: YES‑ask = 100 − NO‑bid). Use occasionally to estimate typical spreads per price regime. ([Kalshi API Documentation][4])
* **WebSocket** (later): subscribe once; server sends **`orderbook_snapshot` then `orderbook_delta`**; you’ll upgrade fill modeling and add OFI/queue signals. ([Kalshi API Documentation][5])
* **Visual Crossing**: station‑locked Timeline API with **`aggregateMinutes=5`**. Smallest sub‑hourly interval is **5–10 minutes** depending on station; use this to compute the **running high** and a **Monte Carlo** nowcast/hazard. ([Visual Crossing][6])
* **API budgets**: Kalshi tiered rate limits—**Basic 20 reads/sec**, **Advanced 30/sec** (etc.). Plenty for candles/trades backfills. ([Kalshi API Documentation][7])

### Modeling (minute clock)

1. **Market PMF (minute)**

   * Build a vector of bracket “probabilities” from minute closes (YES price / 100), then **renormalize** so brackets sum ≈ 1 (candles are per market; coherence is not enforced on exchange side).
   * Compute **velocity/acceleration** (EWM or Savitzky–Golay) and **Relative Acceleration Score (RAS)**: (a_j - \text{mean}(a_{j\pm1})).

2. **Weather PMF & Hazard**

   * Visual Crossing 5‑minute obs/forecast → **Monte Carlo** rest‑of‑day paths; map to bracket PMF; hazard = chance of a **new high** in next step (first‑passage). (Sub‑hourly request guidance and station cadence noted in VC docs.) ([Visual Crossing][6])

3. **Fusion (coherence + calibration)**

   * Fuse Market vs Weather PMF via **logit‑space pooling** or a **logistic‑normal filter**; keep the vector coherent (sums to 1).
   * Add **post‑hoc calibration** (Platt / Isotonic / Temperature). scikit‑learn provides standard recipes. ([Kalshi API Documentation][8])

4. **Signals**

   * **Cross‑bracket acceleration:** go long bin (j) when its acceleration leads neighbors, minute close‑location signals persistent buy pressure, and hazard is elevated.
   * **Migration:** when PMF center of mass is accelerating upward, bleed exposure from lower bin to (j)+1.
   * **Execution choice:** maker by default; taker only when predicted next‑minute move exceeds (fees + conservative slippage). (You’ll refine this after L2.)

5. **Fill‑sim (candles + trades)**

   * **Maker:** considered filled if your limit is inside `[low, close]` and prints occurred ≤ limit; pro‑rate by execution size at/under your limit. If trades unavailable, use a Brownian‑bridge inside the bar (conservative).
   * **Taker:** fill at minute close + half‑spread estimate (spread from occasional orderbook snapshots & bar stats). (Orderbook asks implied from YES/NO bids.) ([Kalshi API Documentation][4])

6. **Fees**

   * Compute EV **after** maker/taker fees; gate signals accordingly. (Use your current fee schedule in config; rate‑limit info for planning backfills.) ([Kalshi API Documentation][7])

---

## `backtest/README.md`

```
# Candles-First Backtester (Kalshi Weather Brackets)

This module backtests a cross-bracket acceleration strategy using **Kalshi 1-minute candlesticks** and optional **trade prints**; no WebSocket history required. 
Later, you can plug in WS L2 to upgrade fill modeling & microstructure features.

Data:
- Per-market and per-event **1-minute candlesticks** (YES OHLCV).  [docs] 
- **Multiple-events** candlesticks endpoint for batch pulls.        [docs]
- **Trades** endpoint for prints & volumes (paginated w/ cursor).   [docs]
- **Orderbook snapshots** for spread estimation (YES/NO bids only). [docs]
- **Visual Crossing** 5-min station obs/forecast for MC Tmax.       [docs]

[docs] Kalshi candlesticks 1m: https://docs.kalshi.com/api-reference/market/get-market-candlesticks
[docs] Event + multi-event candlesticks: https://docs.kalshi.com/api-reference/events/get-event-candlesticks, https://docs.kalshi.com/api-reference/events/get-event-candlesticks-multiple-events
[docs] Trades: https://docs.kalshi.com/api-reference/market/get-trades
[docs] Orderbook bids only (asks implied): https://docs.kalshi.com/api-reference/market/get-market-orderbook
[docs] Rate limits: https://docs.kalshi.com/getting_started/rate_limits
[docs] VC sub-hourly (aggregateMinutes=5..30): https://www.visualcrossing.com/resources/documentation/weather-api/requesting-sub-hourly-weather-data-using-the-weather-api/

## Quick start
# 1) Backfill last 60 days of 1-min candles for a daily-high series
python -m backtest.candles_backfill --event-tickers KXHIGHCHI_2025-11-19 --days 60

# 2) (optional) Pull trades for active brackets to calibrate maker fills
python -m backtest.trades_pull --markets KXHIGHCHI_83_84,KXHIGHCHI_85_86 --start ... --end ...

# 3) Run backtest with fee-aware fills & calibration
python -m backtest.run --city chicago --calibration isotonic --mode paper

Outputs:
- PnL after fees; maker vs taker split
- Reliability (Brier, ECE) before/after calibration
- Hit-rate for next-bar winners & bracket migration
```

---

## Code (extended stubs)

> These are organized so your agent can flesh out quickly. Replace imports to match your package layout; wire your **existing Visual Crossing** client where noted.

### `backtest/candles_backfill.py`

```python
import argparse, httpx, time, math
from datetime import datetime, timedelta, timezone
import pandas as pd

BASE = "https://api.elections.kalshi.com/trade-api/v2"

def get_event_candles(event_ticker: str, start_ts: int, end_ts: int, period_min: int = 1):
    """
    Per-event endpoint: returns arrays of per-market candlesticks (OHLCV) at 1/60/1440-minute intervals.
    https://docs.kalshi.com/api-reference/events/get-event-candlesticks  # :contentReference[oaicite:11]{index=11}
    """
    url = f"{BASE}/events/{event_ticker}/candlesticks"
    params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_min}
    r = httpx.get(url, params=params, timeout=30); r.raise_for_status()
    js = r.json()
    return js["market_tickers"], js["market_candlesticks"], js.get("adjusted_end_ts")

def get_multi_event_candles(event_tickers: list[str], start_ts: int, end_ts: int, period_min: int = 1):
    """
    Multi-event endpoint (batch fetch): aggregated data across all markets for multiple events.
    Limits total candlesticks (~5000) across events; paginate with adjustedEndTs if needed.
    https://docs.kalshi.com/api-reference/events/get-event-candlesticks-multiple-events  # :contentReference[oaicite:12]{index=12}
    """
    url = f"{BASE}/events/candlesticks"
    params = {
        "event_tickers": ",".join(event_tickers),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": period_min,
    }
    r = httpx.get(url, params=params, timeout=40); r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-tickers", required=True, help="comma-separated event tickers")
    ap.add_argument("--start", help="YYYY-MM-DD", default=None)
    ap.add_argument("--end", help="YYYY-MM-DD", default=None)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", default="data/candles.parquet")
    args = ap.parse_args()

    et = args.event_tickers.split(",")
    if args.start and args.end:
        start_ts = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp())
        end_ts   = int(datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc).timestamp())
    else:
        end_ts   = int(datetime.now(tz=timezone.utc).timestamp())
        start_ts = int((datetime.now(tz=timezone.utc)-timedelta(days=args.days)).timestamp())

    js = get_multi_event_candles(et, start_ts, end_ts, 1)
    # normalize into flat rows: ts, market_tkr, open_c, high_c, low_c, close_c, volume
    rows = []
    for tkr, cands in zip(js["market_tickers"], js["market_candlesticks"]):
        for c in cands:
            rows.append({
                "ts": c["end_ts"], "market": tkr,
                "open_c": c["open_cents"], "high_c": c["high_cents"],
                "low_c": c["low_cents"], "close_c": c["close_cents"],
                "volume": c["volume"]
            })
    df = pd.DataFrame(rows).sort_values(["ts","market"])
    df.to_parquet(args.out)
    print(f"Wrote {len(df):,} rows -> {args.out}")

if __name__ == "__main__":
    main()
```

### `backtest/trades_pull.py`

```python
import argparse, httpx, pandas as pd
from datetime import datetime, timezone

BASE = "https://api.elections.kalshi.com/trade-api/v2"

def get_trades(market_ticker: str, start_ts: int, end_ts: int, limit=1000):
    """
    Trades endpoint (paginated, cursor-based).
    https://docs.kalshi.com/api-reference/market/get-trades  # :contentReference[oaicite:13]{index=13}
    """
    url = f"{BASE}/markets/{market_ticker}/trades"
    out, cursor = [], None
    while True:
        params = {"start_ts": start_ts, "end_ts": end_ts, "limit": limit}
        if cursor: params["cursor"] = cursor
        r = httpx.get(url, params=params, timeout=30); r.raise_for_status()
        js = r.json(); out.extend(js["trades"])
        cursor = js.get("cursor")
        if not cursor: break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", required=True, help="comma-separated market tickers")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="data/trades.parquet")
    args = ap.parse_args()
    start_ts = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc).timestamp())

    rows = []
    for m in args.markets.split(","):
        for t in get_trades(m, start_ts, end_ts):
            rows.append({"market": m, "ts": t["ts"], "price_c": t["price_cents"], "qty": t["quantity"]})
    pd.DataFrame(rows).to_parquet(args.out)

if __name__ == "__main__":
    main()
```

### `features/candle_features.py`

```python
import numpy as np, pandas as pd
from typing import Dict, List

def ewm_diff(x: pd.Series, span: int) -> pd.Series:
    return x.diff().ewm(span=span, adjust=False).mean()

def velocity_acceleration(close_prob: pd.Series, span_v=5, span_a=8) -> tuple[pd.Series,pd.Series]:
    v = ewm_diff(close_prob, span_v)
    a = ewm_diff(v, span_a)
    return v, a

def close_location(df: pd.DataFrame) -> pd.Series:
    rng = (df["high_c"] - df["low_c"]).clip(lower=1e-6)
    return (df["close_c"] - df["low_c"]) / rng

def build_panel(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Input candles: ts, market, open_c, high_c, low_c, close_c, volume.
    Output: MultiIndex columns (market, feature) with p, v, a, clv, and p_norm.
    """
    # compute per-market
    feats = {}
    idx = sorted(candles["ts"].unique())
    for m, dfm in candles.groupby("market"):
        d = dfm.set_index("ts").reindex(idx).ffill()
        p = d["close_c"] / 100.0
        v, a = velocity_acceleration(p)
        clv = close_location(d)
        feats[m] = pd.DataFrame({"p":p, "v":v, "a":a, "clv":clv})
    panel = pd.concat({m: f for m,f in feats.items()}, axis=1)

    # coherence (renormalize)
    ps = panel.xs("p", axis=1, level=1)
    sump = ps.sum(axis=1).replace(0, np.nan)
    for m in feats.keys():
        panel[(m, "p_norm")] = panel[(m,"p")] / sump
    return panel.sort_index()
```

### `models/pmf_fusion.py`

```python
import numpy as np

def softmax(z):
    z = np.asarray(z); z = z - z.max()
    e = np.exp(z); return e / e.sum()

def fuse_pmf(p_mkt: np.ndarray, p_wx: np.ndarray, w_mkt: float, w_wx: float) -> np.ndarray:
    """
    Simple logit-space pooling. Replace later with a logistic-normal UKF.
    """
    p_mkt = np.clip(p_mkt, 1e-9, 1.0); p_wx = np.clip(p_wx, 1e-9, 1.0)
    z = w_mkt * np.log(p_mkt) + w_wx * np.log(p_wx)  # unnormalized log weights
    return softmax(z)

def relative_accel(panel_row, markets: list[str]) -> dict[str,float]:
    A = {m: panel_row[(m,"a")] for m in markets}
    out = {}
    for i, m in enumerate(markets):
        nbrs=[]
        if i>0: nbrs.append(A[markets[i-1]])
        if i<len(markets)-1: nbrs.append(A[markets[i+1]])
        out[m] = A[m] - (np.mean(nbrs) if nbrs else 0.0)
    return out
```

### `models/calibration.py`

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

class Platt:
    def __init__(self): self.lr = LogisticRegression(max_iter=200)
    def fit(self, scores, y): self.lr.fit(scores.reshape(-1,1), y)
    def predict(self, scores): return self.lr.predict_proba(scores.reshape(-1,1))[:,1]

class Iso:
    def __init__(self): self.ir = IsotonicRegression(out_of_bounds="clip")
    def fit(self, scores, y): self.ir.fit(scores, y)
    def predict(self, scores): return self.ir.predict(scores)

def ece(probs, labels, bins=10):
    bins = np.linspace(0,1,bins+1)
    idx = np.digitize(probs, bins)-1
    ece=0.0
    for b in range(len(bins)-1):
        mask = idx==b
        if mask.sum()==0: continue
        ece += np.abs(probs[mask].mean() - labels[mask].mean()) * mask.mean()
    return float(ece)
```

### `models/monte_carlo.py` (keep simple; plug your VC module)

```python
import numpy as np

def mc_tmax(obs_5m: list[tuple[int,float]], fcst_5m: list[tuple[int,float]],
            bins: list[tuple[float,float]], N=4000, rho=0.8, sigma_f=0.6):
    """
    obs_5m, fcst_5m: [(ts,tempF)]
    Return: pmf over bins, hazard (new high next step).
    """
    m_run = max(x for _,x in obs_5m) if obs_5m else -1e9
    F = np.array([x for _,x in fcst_5m], dtype=float)
    K=len(F)
    eps=np.zeros((N,K))
    for k in range(K):
        if k==0: eps[:,k]=np.random.normal(0,sigma_f,N)
        else:    eps[:,k]=rho*eps[:,k-1]+np.sqrt(1-rho**2)*np.random.normal(0,sigma_f,N)
    T = F + eps
    M_future = T.max(1); M = np.maximum(m_run, M_future)
    pmf = [float(((M>=lo)&(M<=hi)).mean()) for lo,hi in bins]
    hazard = float((T[:,0] > m_run).mean())
    return pmf, hazard
```

### `backtest/fillsim.py`

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class Fee:
    maker_rate: float = 0.0175   # example schedule
    taker_rate: float = 0.07

def taker_fee(price_c: int, qty: int, rate=0.07):
    P = price_c/100.0; return np.ceil(100*rate*qty*P*(1-P)).astype(int)

def maker_fee(price_c: int, qty: int, rate=0.0175):
    P = price_c/100.0; return np.ceil(100*rate*qty*P*(1-P)).astype(int)

def taker_fill_price(close_c: int, half_spread_c: int=1):
    return close_c + half_spread_c

def maker_fill(limit_c: int, bar_low_c: int, prints=None):
    if limit_c <= bar_low_c:
        if prints is None: return 1.0
        filled = prints[prints["price_cents"]<=limit_c]["quantity"].sum()
        total  = prints["quantity"].sum()
        return float(filled)/max(1,total)
    return 0.0
```

### `backtest/run.py`

```python
import argparse, pandas as pd, numpy as np
from features.candle_features import build_panel
from models.pmf_fusion import fuse_pmf, relative_accel
from models.monte_carlo import mc_tmax
from models.calibration import Platt, Iso, ece
from backtest.fillsim import maker_fill, taker_fill, maker_fee, taker_fee

def choose_action(row, ras_j, clv_j, hazard, maker_edge_c=2, taker_edge_c=4):
    """Very simple policy: maker if small predicted move; taker if large."""
    if ras_j>0 and clv_j>0.6 and hazard>0.1:
        # predict +X cents next bar from regression/heuristic; stub +2c
        exp_move_c = 2
        if exp_move_c >= taker_edge_c: return "taker"
        if exp_move_c >= maker_edge_c: return "maker"
    return "hold"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candles", default="data/candles.parquet")
    ap.add_argument("--vc_obs", help="parquet of 5-min obs+fcst or hook your module")
    ap.add_argument("--calibration", choices=["none","platt","isotonic"], default="none")
    args=ap.parse_args()

    candles = pd.read_parquet(args.candles)  # ts,market,open_c,high_c,low_c,close_c,volume
    panel = build_panel(candles)
    markets = sorted(candles["market"].unique())

    # Example bins by market naming convention; replace with your lookup
    bins = [(81,82),(83,84),(85,86),(87,88)]
    # Placeholder weather pmf/hazard:
    p_wx = np.array([0.1,0.6,0.2,0.1]); hazard=0.15

    pnl=0.0
    for ts,row in panel.iterrows():
        # Market PMF from normalized close
        p_mkt = np.array([row[(m,"p_norm")] for m in markets])
        p = fuse_pmf(p_mkt, p_wx, w_mkt=0.6, w_wx=0.4)
        RAS = relative_accel(row, markets)
        # Choose a target market (demo picks argmax RAS)
        tkr = max(RAS, key=lambda k: RAS[k])
        clv = row[(tkr,"clv")]
        action = choose_action(row, RAS[tkr], clv, hazard)
        # Fill sim + fee (demo: taker at close; maker at limit = close-1c)
        bar = candles[(candles["ts"]==ts) & (candles["market"]==tkr)].iloc[0]
        if action=="taker":
            fill_px = bar["close_c"]+1; fee = taker_fee(bar["close_c"], 1)
            pnl -= (fill_px + fee) / 100.0   # buy 1
        elif action=="maker":
            limit = bar["close_c"]-1
            frac = maker_fill(limit, bar["low_c"])
            if frac>0:
                fee = maker_fee(limit, 1)
                pnl -= (limit + fee)/100.0 * frac
        # imagine we liquidate next bar at close (demo)
    print("Demo PnL (not real):", round(pnl,2))

if __name__=="__main__":
    main()
```

### Optional: `ingest/orderbook_sample.py` (spread stats)

```python
import httpx

def get_orderbook(market_ticker: str):
    """
    YES/NO bids only; asks implied. Good for estimating spread by price regime.
    https://docs.kalshi.com/api-reference/market/get-market-orderbook  # :contentReference[oaicite:14]{index=14}
    """
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_ticker}/orderbook"
    r = httpx.get(url, timeout=10); r.raise_for_status()
    return r.json()
```

---

## Docker & Make targets (minimal)

**`docker-compose.yml`**

```yaml
version: "3.9"
services:
  backtest:
    build: .
    command: python -m backtest.run --candles /data/candles.parquet --calibration none
    volumes:
      - ./data:/data
    environment:
      - PYTHONUNBUFFERED=1
```

**`Makefile`**

```make
BACKTEST_DAYS?=60
EVENTS?=KXHIGHCHI_2025-11-19

backfill:
	python -m backtest.candles_backfill --event-tickers $(EVENTS) --days $(BACKTEST_DAYS) --out data/candles.parquet

trades:
	python -m backtest.trades_pull --markets $(MARKETS) --start $(START) --end $(END) --out data/trades.parquet

run:
	python -m backtest.run --candles data/candles.parquet --calibration isotonic
```

---

## How this gets you moving **today**

* Everything above runs on **1‑minute** candlesticks/trades. Kalshi documents 1‑minute periods for market and event candles, plus multi‑event aggregation and trades pagination. ([Kalshi API Documentation][1])
* You’ll still ingest Visual Crossing **5‑minute** station data (official sub‑hourly intervals) for the nowcast/hazard. ([Visual Crossing][6])
* Your **rate limit** easily covers backfills and live polling while you spin up a separate process to start **recording WebSocket L2** for future upgrades. ([Kalshi API Documentation][7])
* When L2 is ready, swap the fill‑sim for a **queue‑aware** model and add OFI/imbalance/Hawkes features; the backtest logic and PMF fusion remain unchanged. The WS doc shows how to subscribe to many tickers and process **snapshot→delta** updates. ([Kalshi API Documentation][5])

If you want, I can also add a `city→event_tickers` resolver (pull daily high markets for a city via `GET /events` and their child markets) and a tiny YAML to list your bracket order. But you can already start backfilling **Chicago** with the event ticker(s) you have and be running paper strategies this afternoon.

[1]: https://docs.kalshi.com/api-reference/market/get-market-candlesticks?utm_source=chatgpt.com "Get Market Candlesticks - API Documentation"
[2]: https://docs.kalshi.com/api-reference/events/get-event-candlesticks-multiple-events?utm_source=chatgpt.com "Get Event Candlesticks (Multiple Events)"
[3]: https://docs.kalshi.com/api-reference/market/get-trades?utm_source=chatgpt.com "Get Trades - API Documentation"
[4]: https://docs.kalshi.com/api-reference/market/get-market-orderbook?utm_source=chatgpt.com "Get Market Orderbook - API Documentation"
[5]: https://docs.kalshi.com/websockets/orderbook-updates?utm_source=chatgpt.com "Orderbook Updates - API Documentation"
[6]: https://www.visualcrossing.com/resources/documentation/weather-api/requesting-sub-hourly-weather-data-using-the-weather-api/?utm_source=chatgpt.com "Requesting sub-hourly weather data using the Weather API"
[7]: https://docs.kalshi.com/getting_started/rate_limits?utm_source=chatgpt.com "Rate Limits and Tiers - API Documentation"
[8]: https://docs.kalshi.com/getting_started/quick_start_market_data?utm_source=chatgpt.com "Quick Start: Market Data - API Documentation"
