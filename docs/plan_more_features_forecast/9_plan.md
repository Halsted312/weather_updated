You’re not stuck — Visual Crossing is *designed* to give you mixed granularity, and the right move is to embrace that with a **multi-resolution feature layer**, rather than trying to force everything into 15‑minute grids.

## 1) Why leads 2–7 “missing minutes” is expected (and not a bug)

Visual Crossing’s own docs say sub-hourly **forecast** data is 15-minute minimum and typically only extends **~12–24 hours into the future** (model-dependent).
So in a *historical forecast* setting (basis date = “forecast run stored at midnight UTC”), you should expect:

* **Daily + hourly** data for many days out (full horizon).
* **15‑minute** data only for the near horizon *when VC’s high-res model supports it* (sometimes just lead 0, sometimes lead 1, occasionally more depending on model/location).

Also: VC explicitly says they **do not interpolate hourly forecast data down to sub-hourly** — so when minutes aren’t returned, the API won’t “fill them in for you”.

## 2) The key constraint you should bake into the plan

Historical forecasts are based on “full forecast model runs stored at **midnight UTC** each day.”
So you cannot reconstruct “forecast revisions every 15 minutes throughout the day” historically from VC alone. If you want that dataset, you must **start collecting it going forward** (scheduled polling + storing snapshots).

That means your backtest should treat minute forecasts as:
**“the sub-hourly curve predicted at the basis run time”**, not “intraday updating forecasts”.

## 3) Best way to do BOTH (hourly far out + 15-min near-term) without apples/oranges

### A. Store both resolutions, but *model on compressed features*

Your plan already points the right direction with “compressed, settlement-aligned features”. Keep doing that.

Make a single function that takes a time series at *any* step size and returns the same feature schema:

* `pred_high` (max temp)
* `pred_high_time` (time index of max)
* `peak_window_mean` (11:00–18:00 mean)
* `peak_window_p95`
* `slope_pre_peak`, `slope_post_peak`
* `curve_volatility` (std of first differences)
* `dewpoint_depression_mean` (temp − dew)
* `wind_mean`, `pressure_trend`, `solar_integral`, etc.

Then:

* If minutes exist: compute from minutes.
* Else if only hourly exists: compute from hourly.
* Add flags: `has_minutes`, `step_minutes`.

This keeps one pipeline and avoids “inventing” fake 15‑minute info.

### B. For “drift as day gets closer”, use **daily/high-level drift**, not minute grids

Your “drift_T1_vs_T2 … drift_T1_vs_T3” idea is correct — and it shouldn’t depend on the 15‑minute series. Those drift features come from comparing **daily (or hourly) predicted high** across lead days.

So you can safely:

* pull leads 0–10 (or 0–14) at **daily/hourly**
* pull leads near horizon with minutes when available
* compute drift features entirely from daily/hourly maxima

### C. Consider a *two-regime model* (optional but robust)

If the short-horizon feature set is materially richer (minutes + obs trends), it can be cleaner to train:

* **Long-horizon model** (lead ≥ 2): daily/hourly + macro features
* **Short-horizon model** (lead 0–1): adds minute-curve features + richer obs/market microstructure

Still one pipeline, but a gating switch based on `time_to_event_close` / lead.

## 4) Should you extend forecasts from 7 to 10 days?

Yes, if your goal is **trend/drift stability**. VC supports querying historical forecasts via a **basis date** concept and you can request multiple days forward in one call.
But don’t extend because you want more “15‑minute” — extending mostly helps your **drift/uncertainty features**, not intraday shape.

## 5) Fix the plan’s “basis-batched” backfill assumptions

The nice optimization in the plan is using `forecastBasisDate` with a date range to batch multiple lead days in one request (VC documents this explicitly).
Your current ingestion script is using `forecastBasisDay` (fine, but batching is easier/cleaner with `forecastBasisDate`). 

Two practical tweaks I’d recommend:

1. **Don’t expect minutes for every lead in validation logic**; treat them as “best effort”.
2. **Only request minutes when it’s likely to exist** (near horizon). Since minutes cost 96 records/day at 15-min, that’s expensive at scale. VC confirms the cost math.

## 6) Can you batch multiple cities in one request?

VC *can* do multi-location timeline requests, but their docs warn it’s **limited to single concurrency** and **not intended for loading large datasets into databases**; they recommend multiple single-location requests instead.
So: stick with **single-location requests** and parallelize on your side (with your rate limiter).

## 7) Quick file-level notes (things to keep / adjust)

* Your ingestion loop already gracefully parses minutes only when present (it iterates `hour_data.get("minutes", [])`). That’s exactly what you want. 
* Your minute-query helper is currently **T‑1 specific** (`forecast_basis_date = target_date - 1`). That’s fine for “near-term features”, but for general lead drift you’ll want analogous helpers for **daily/hourly** tables (not minute) or a generalized minute helper that accepts `lead_days`. 
* The *old* station-vs-city script decision logic that triggers on `max_deviation` is indeed noisy (single-point spike). That’s exactly why you should base the decision on settlement-relevant metrics (daily high + peak timing), not “max absolute point diff”. 

## 8) The “finish strong” summary you can hand your agent

**Goal:** one robust pipeline that uses the best available resolution without leaking or hallucinating detail.

**Implementation rule of thumb:**

* Always ingest **daily + hourly** for leads 0–N (7/10/14).
* Ingest **minutes opportunistically** where VC provides it (don’t assume).
* Feature builder computes the same “compressed curve features” from whichever resolution is present + adds `has_minutes/step_minutes`.
* Drift features come from daily/hourly highs across lead days (resolution-independent).
* If intraday updates matter, start collecting live forecast snapshots going forward (VC historical forecasts won’t give you intraday revision history).

(And yes: keeping both station-anchored lat/lon and city-aggregate is defensible even when average gaps are small, *as long as you compress* to a few stable gap features + do ablation tests.)

