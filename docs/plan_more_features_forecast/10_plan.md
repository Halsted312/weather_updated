You’re not stuck between a rock and a hard place — you just have **three different types of information** that need to be handled with a clear contract:

1. **Historical forecasts (what VC *predicted* in the past).**
2. **Historical observations (what actually happened).**
3. **Live streaming forecasts + obs (what you see “right now”).**

If you design the pipeline so that **all three live behind one consistent abstraction**, your backtests and live features can line up cleanly.

Below I’ll lay out:

* A **conceptual model** of the data you want (across leads + resolutions).
* A **concrete plan to give your coding agent**: what to ingest, how, and where to store it.
* How to make sure **training features == live features** as much as possible.

---

## 1. Conceptual model: what you want, precisely

For *each city* and *each target date D* (the Kalshi settlement day), you want to be able to reconstruct:

### A. Historical forecasts (by lead)

Think of a **forecast matrix**:

* Rows = **basis dates** (when the forecast model ran)
* Columns = **target dates** (D, D+1, …, D+14)

From Visual Crossing’s historical forecast API:

* Each basis date **B** stores a **full 15-day forecast** (D = B, B+1, …, B+14) from that run.
* You can query this either as:

  * `forecastBasisDay` with `target_date`, or
  * `forecastBasisDate` with a date range (better for batching).

For each `(basis_date B, target_date D)` pair, you want:

* **Daily** forecast: `tempmax_f`, `tempmin_f`, etc. (`VcForecastDaily`) 
* **Hourly** forecast: hourly temps, humidity, etc. (`VcForecastHourly`) 
* **15-minute** forecast: where available, for near leads, a minute grid (`VcMinuteWeather` with `data_type='historical_forecast'`). 

Crucially:

* **Leads 0–1 (today / tomorrow)** → you often have 15-minute sub-hourly data.
* **Leads 2–14** → you often only have hourly/daily; minutes may not exist. That’s expected.

### B. Historical observations

You already ingest minute/5-minute obs into `VcMinuteWeather` with `data_type='actual_obs'` via your obs ingestion script.
That gives you:

* A **minute-series of actual temps, humidity, etc.** for each station/city.
* Allows you to compute **forecast errors**: `obs − forecast` at hourly or minute resolution.

### C. Live streaming

At live time **T_now**, for a given target date **D**:

* VC forecast API returns a **15-day forecast** (D_now … D_now+14) with daily/hourly, and near-term sub-hourly.
* You’ll be polling regularly (say every 5–15 minutes) and can store:

  * “Live forecast snapshot at time T_now”, plus
  * Current obs (station temp, etc), binned to 1-minute as needed.

But **historically**, you only have **midnight model runs** stored by VC (one basis per day).
So intraday revision history must be **collected by you going forward**, not reconstructed from VC.

---

## 2. Ingestion architecture: what to tell the agent

You already have robust scripts in place:

* `scripts/ingest_vc_hist_forecast_v2.py` – per-city, per-location-type ingest using `forecastBasisDay` and per-target date requests. 
* `scripts/backfill_vc_historical_forecasts.py` – a higher-level backfill driver with adaptive rate limiting and minute parsing, using `forecastBasisDay`. 
* `scripts/validate_15min_ingestion.py` – checks minute counts, basis dates, nulls. 
* `models/data/vc_minute_queries.py` + `scripts/test_query_helpers.py` – query helpers for t−1 minutes and availability checks. 

You don’t need to rip this up; you just need to **lock in exactly what we backfill** and why.

### 2.1 For historical forecasts (backtest)

Tell your agent:

> **Goal**: For each city and each `target_date` in our backtest window, store:
>
> * `VcForecastDaily`: `tempmax_f` etc for lead_days 0–14.
> * `VcForecastHourly`: 24 hourly temps for lead_days 0–14.
> * `VcMinuteWeather (historical_forecast)`: 15-minute temps for lead_days 0–1 (and any extra leads for which VC actually returns minutes).

**Implementation details:**

1. **Use one of the existing scripts as the canonical ingest:**

   * Keep `ingest_vc_hist_forecast_v2.py` as the **per-city/month** ingestion script; it already:

     * Validates `source != 'obs'` (forecast vs obs). 
     * Writes daily, hourly, and minutes into `VcForecastDaily`, `VcForecastHourly`, `VcMinuteWeather`. 
   * Use `--location-type station` for lat/lon at settlement station, and `--location-type city` for city name queries. The v2 script already supports this. 

2. **Backfill leads out to 14 days** (hourly + daily)

   * Keep your `--lead-days` argument in `ingest_vc_hist_forecast_v2.py` and `backfill_vc_historical_forecasts.py`. For backtests, use `0,1,2,…,14` for:

     * `VcForecastDaily` → drift of high temp across 14 days.
     * `VcForecastHourly` → hourly curves across 14 days.
   * Don’t worry if minutes don’t exist for leads ≥2; just let minute ingestion be **best effort**.

3. **Backfill 15-minute only for near horizon**

   * Set `--include-minutes` when backfilling leads 0–1 (or 0–2 if VC provides minute grids there).
   * For lead_days > 2, you can either:

     * Still request minutes but accept there may be none, or
     * Turn off `include_minutes` to save cost.

4. **Optional optimisation:** if you want to cut request count by ~4×, implement the basis-batched script your agent proposed (`backfill_vc_basis_batched.py`) that uses `forecastBasisDate` and multi-day ranges; but this is just an efficiency improvement, not a correctness one. 

### 2.2 For historical observations

Tell your agent:

> Use the existing `ingest_vc_obs_parallel.py` (or similar) to:
>
> * Ingest station obs into `VcMinuteWeather` with `data_type='actual_obs'`.
> * At 5-minute resolution; if we want 1-minute, we’ll forward-fill at query time.

This is already how your obs script behaves: it uses minute history endpoint and writes into `VcMinuteWeather` with all the same weather fields. You don’t need to change schema; just be explicit that `data_type` distinguishes obs vs forecast.

### 2.3 For live streaming

Tell the agent to plan for a **live snapshot table** (or reuse `Vc…` tables with a `data_type='live_forecast'`/`'live_obs'`), e.g.:

* Set up a cron / scheduler that, for each city:

  1. Once per day (e.g., after VC’s midnight UTC run):

     * Calls Timeline (no basis params) to get **current 15-day forecast** with `include=days,hours,minutes`.
     * Writes this snapshot to the same `VcForecastDaily`, `VcForecastHourly`, `VcMinuteWeather` tables with:

       * `data_type = 'forecast'` (not `'historical_forecast'`).
       * `forecast_basis_date = today`.

  2. Every N minutes (for live updates):

     * Polls near-term forecast only (if desired).
     * Writes to a `vc_live_forecast_snapshots` table or appends to `VcMinuteWeather` with `data_type='live_forecast'` and different `source_system`.

This gives you a **clean separation**:

* `data_type='historical_forecast'` → VC midnight model runs for past dates (for backtest).
* `data_type='forecast'`/`'live_forecast'` → live query snapshots going forward.

---

## 3. Feature contract: keep training and live as close as possible

Here’s what I’d have the agent implement in your **feature builder** layer (on top of your existing `forecast.py`, `shape.py`, `station_city.py`):

### 3.1 For drift over 14 days (daily / hourly)

**Training & Live (same logic)**

Define a function:

```python
def get_forecast_path_for_target(session, city_code, target_date, location_type):
    """
    Returns a dict:
      lead_days -> {
          "tempmax_f": ...,
          "hourly_curve": [... 24 values ...]
      }
    using VcForecastDaily/VcForecastHourly with data_type='historical_forecast' (training)
    or data_type='forecast' (live).
    """
```

Implementation details for training:

* For each `lead_days L` from 0..14:

  * Look up `VcForecastDaily` where:

    * `target_date = D`
    * `lead_days = L`
    * `data_type='historical_forecast'`.
  * Derive `basis_date = D - L`.
* Similarly, get the 24 hourly rows from `VcForecastHourly` where:

  * `forecast_basis_date = basis_date`
  * `lead_hours` in `[24*L, 24*(L+1))`.

For live:

* The **current forecast** snapshot is equivalent to `basis_date = today`.
* Derive `lead_days = (target_date - basis_date).days`.
* Use the same tables but with `data_type='forecast'`.

From that dict, compute **compressed drift features**:

* For a single target_date D:

  * `high_TL = [tempmax_f(L) for L in 0..14]`
  * `drift_TL_vs_Tk = tempmax_f(1) - tempmax_f(k)` for k>1
  * `drift_slope = slope of high_TL vs L`
  * `uncertainty_proxy = std(high_TL)`

These require only daily forecasts; they don’t care about 15-minute vs hourly.

### 3.2 For near-horizon shape (15-minute vs hourly)

For **lead 0–1** (today / tomorrow) you want detailed shape features: when does the high occur, is it a spike or plateau, how fast does it climb, etc.

Tell your agent:

> Build a function that:
>
> * Prefers minute-level historical forecast if present for lead 0–1.
> * Falls back to hourly forecast if `VcMinuteWeather` is empty.
> * Always returns the same **feature schema** regardless of resolution.

Example:

```python
def build_intraday_shape_features(
    session, city_code, target_date, lead_days, location_type
):
    """
    Returns a dict of shape features for the intraday forecast curve
    (temp vs time) of (target_date, lead_days), using:
      - VcMinuteWeather (historical_forecast) if available, else
      - VcForecastHourly
    """
```

Internally:

* Query `VcMinuteWeather` where:

  * `forecast_basis_date = target_date - lead_days`
  * `lead_hours` in `[24*L, 24*(L+1))`
  * `data_type='historical_forecast'`.
* If you get ~96 points → treat as 15-minute series (`step_minutes = 15`).
* Else, query `VcForecastHourly` and treat as hourly (`step_minutes = 60`).

Then pass this series into `shape.compute_shape_features` with `step_minutes` to get:

* `shape_plateau_score`, `shape_morning_vs_afternoon`, `shape_peak_time`, etc. 

For **live**, do the same but reading from `data_type='forecast'` or `live_forecast`.

This way:

* Training **never assumes** minutes beyond what VC historically stores.
* Live **benefits** from minutes when available, but passes them through the same feature builder.

### 3.3 For “changing 15-minute forecast over last 2 days”

You only have **historical midnight runs** from VC; you don’t have intraday update history for the past. The only way to truly use “15-minute forecast updates over last 2 days” is to:

* Start collecting *your own* snapshots now (e.g., every 30 minutes), and
* Accept that these features will only have a training history starting **from now onwards**.

So I’d tell the agent:

> For v1, restrict 15-minute features to “shape of the day as predicted by the midnight basis run” (using VC historical forecasts).
> Add a TODO: start logging live forecast snapshots (time, snapshot_id, full 15-day forecast) to a new “forecast_snapshots” table; once we have enough history, we can add features based on “delta between latest snapshot and midnight basis”.

That keeps v1 strictly aligned between training and live.

---

## 4. Long story short – what to tell the agent, in one block

Here’s a paragraph you can literally paste into your plan for the agent:

> **Data & Feature Contract**
>
> 1. **Historical forecasts**
>
>    * For each city/location type (`station` lat/lon, `city` city-name) and each `target_date`, ingest Visual Crossing **historical forecasts** using `forecastBasisDay` (or `forecastBasisDate` for batching) into:
>
>      * `VcForecastDaily` for **lead_days 0..14** (`tempmax_f`, etc.).
>      * `VcForecastHourly` for **lead_days 0..14** (24 hourly temps per lead).
>      * `VcMinuteWeather` with `data_type='historical_forecast'` for **lead_days 0–1** (and any leads where VC actually returns `minutes`), via the existing `parse_minute_forecast` logic.  
>    * Do not try to synthesize minute data where VC doesn’t return it; just let minute rows be absent for leads ≥2 and rely on hourly/daily for those.
> 2. **Historical observations**
>
>    * Use the existing obs ingestion to populate `VcMinuteWeather` with `data_type='actual_obs'` at 5-minute resolution (or better). These are the actual temps used for backtesting and forecast error features.
> 3. **Live forecasts & obs**
>
>    * Introduce `data_type='forecast'` (and/or a small `forecast_snapshots` table) to store **today’s** 15-day forecast from the live VC API, once per day (midnight basis) with `include=days,hours,minutes`.
>    * Optionally log frequent “live forecast snapshots” going forward to support future intraday-update features, but v1 feature engineering should rely only on the midnight basis snapshot so that training and live are aligned.
> 4. **Feature builders**
>
>    * Implement a forecast path builder that, for a given `(city_code, target_date, location_type)`, looks up `VcForecastDaily` / `VcForecastHourly` and returns a compressed set of **lead-based drift features** (vector of predicted highs across lead_days 0..14, drift across leads, slope, etc.).
>    * Implement an intraday shape builder that, for any `(target_date, lead_days)`, prefers `VcMinuteWeather` (15-min curve) if present, and falls back to `VcForecastHourly` (hourly curve) if not. It must expose the same feature schema (peak time, plateau/spike score, peak-window stats) regardless of resolution, with a `step_minutes` flag.
>    * Ensure live code uses the **same builders**, swapping `data_type='historical_forecast'` for `data_type='forecast'` and deriving lead_days from `target_date - today`.

If your agent follows that, you’ll end up with:

* A **consistent historical dataset** for 14-day drift and high-resolution near-term shape.
* A live pipeline that **uses the same abstractions**, so your backtest and tuned parameters match what you actually have in production.
* A clean path to later add truly live “forecast update” features once you’ve collected your own snapshot history.

That’s the balance: honor VC’s constraints (midnight model runs, limited sub-hourly horizon) while designing a feature layer that doesn’t depend on information you can’t reconstruct historically.
