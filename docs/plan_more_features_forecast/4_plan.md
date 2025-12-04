You’re actually *really* close to having exactly what you want already – your schema and ingestion scripts are set up for minute-level data, you’re just not pulling sub-hourly **forecasts** yet or using them in the forecast feature modules.

I’ll break this into:

1. What you already have (and how it fits together)
2. How to extend ingestion to 15-minute **historical forecasts** for *all cities*
3. How to plug 15-min series into your existing **feature modules** (forecast, shape, station/city)
4. Extra feature ideas beyond temperature (humidity, cloud, etc.), using your current tables

---

## 1. What you already have

### 1.1 Schema: you’re already set up for minute forecasts

You have **two generations** of schema:

* Older hourly snapshot table `wx.forecast_snapshot_hourly` created by Alembic rev 005 – per-city, per-hour, per-basis date. 

* New “greenfield” VC schema (rev 007) with:

  * `wx.vc_location` – one row per city + station/city variant (e.g., CHI/KMDW vs Chicago,IL; AUS/KAUS vs Austin,TX), including IANA timezone.
  * `wx.vc_minute_weather` – **minute-level** weather data, designed for obs *and* forecasts / historical forecasts, with:

    * `data_type` ∈ {`actual_obs`, `current_snapshot`, `forecast`, `historical_forecast`}
    * `forecast_basis_date`, `forecast_basis_datetime_utc`, `lead_hours`
    * full weather fields: `temp_f`, `humidity`, `precip_in`, `cloudcover`, `windspeed_mph`, CAPE/CIN, solar, etc.
    * unique index `uq_vc_minute_fcst` specifically for forecast rows (basis_date not null). 
  * `wx.vc_forecast_daily` and `wx.vc_forecast_hourly` – your current daily and hourly snapshot tables (forecast + historical_forecast labeled in `data_type`). 

Then rev 008 adds an `is_forward_filled` boolean to `vc_minute_weather` so you can flag “-77.9°F sentinel replaced by forward-fill” cases. 

**Implication:**
You do **not** need a new table for 15-minute forecasts.
`wx.vc_minute_weather` is explicitly built to hold:

* minute obs (`data_type='actual_obs'`)
* minute forecast snapshots (`'forecast'`)
* minute historical forecast snapshots (`'historical_forecast'`)

across *all* cities.

---

### 1.2 Ingestion: you already do hourly historical forecasts + minute obs

You have three main ingestion paths:

1. **Historical forecast: daily + hourly**

   * `scripts/ingest_vc_historical_forecast_v2.py` (DB-aware) and
   * `scripts/ingest_vc_historical_forecast_parallel.py` (aggressive parallel)
     Both use Timeline API + `forecastBasisDate` to fill `wx.vc_forecast_daily` and `wx.vc_forecast_hourly` with **historical forecasts** (data_type='historical_forecast').

   These are hourly, not minutes, but they already do the “basis date / lead_days” logic you need.

2. **Historical forecast (lat/lon + forecastBasisDay)**

   * `scripts/ingest_vc_hist_forecast_v2.py` is your newer, lat/lon-based ingestion using `forecastBasisDay` so you can say “give me the 0/1/2/3-day-ahead forecast for this target date”. 
   * It writes to the **same** `VcForecastDaily` and `VcForecastHourly` tables, and already supports multiple lead_days (`--lead-days 0,1,2,3`).

3. **Minute-level observations**

   * `scripts/ingest_vc_obs_parallel.py` calls `VisualCrossingClient.fetch_station_history_minutes` and writes into `wx.vc_minute_weather` with `data_type='actual_obs'`.
   * It does clean timezone handling (UTC epoch → IANA local timezone via `ZoneInfo`) and fills lots of fields (temp/humidity/cloudcover/etc.). 

So you already have:

* **T-0..T-6 hourly historical forecasts** for all cities.
* **5-minute obs** stored per station in `vc_minute_weather`.

What you *don’t* yet have is:

* **Sub-hourly forecasts/historical forecasts** in `vc_minute_weather`.
* Forecast feature modules that explicitly understand sub-hourly series (beyond “hour index” assumptions).

---

### 1.3 Feature modules: you’re already halfway there

You have three very relevant feature modules:

1. `shape.py` – shape-of-day features for spike vs plateau: minutes ≥ thresholds, longest run ≥ threshold, morning/afternoon/evening max, 30-minute slopes, etc. The functions are parameterised by `step_minutes`, so they are agnostic to 5 vs 15 minutes. 

2. `station_city.py` – station vs city gap features: current gap, max/mean gap, std, gap trend; it assumes both series share timestamps, which your minute obs ingestion already ensures. 

3. `forecast.py` – T-1 forecast and forecast-vs-observed error features:

   * **Static**: `fcst_prev_max_f`, min/mean/std, percentiles, fraction of forecast max, hour_of_max, `t_forecast_base`.
   * **Dynamic**: error stats (obs − forecast) so far, last hour/3-hour bias, etc.
   * Alignment helper that aggregates 5-min obs up to **hourly** means and compares with hourly forecast. 

All three of these already handle exactly the kind of “shape of curve over time” features we talked about – they’re just wired today to **hourly forecasts + 5-min obs**.

---

## 2. Getting 15-minute historical forecasts (same pipeline, all cities)

### 2.1 What Visual Crossing offers: minutes + basis date

From the VC docs:

* Timeline API now supports **sub-hourly (`minutes`)** as an additional `include` grid.
* Default interval: 15 minutes; you can change via `options=minuteinterval_30`, etc.
* **Historical** sub-hourly: minimum 5–10 minutes, depending on station data.
* **Forecast** sub-hourly: 15-minute minimum, for regions with high-res models; forecast is *not interpolated below 15 min*.
* Historical forecasts use the **basis date / basis day** concept:

  * `forecastBasisDate=YYYY-MM-DD` or
  * `forecastBasisDay=k` to say “k days before the start date ask for the forecast that was made then”.

Crucially, you can **combine** these:

```text
.../timeline/Austin,TX/2025-12-05/2025-12-05
    ?unitGroup=us
    &include=minutes
    &forecastBasisDay=1
    &contentType=json
    &key=YOUR_KEY
```

That gives you:

* For target date 2025-12-05,
* The **15-minute forecast path** for that full day,
* As it looked on the basis run 1 day earlier.

That’s exactly what you want: a 15-minute curve that is consistent with your current hourly basis snapshots.

---

### 2.2 How to plug this into your existing code

Given your current ingestion stack, the cleanest move is:

> **Extend the lat/lon historical forecast script to also write minutes into `wx.vc_minute_weather` with `data_type='historical_forecast'`.**

You already have:

* `ingest_vc_hist_forecast_v2.py` (lat/lon + `forecastBasisDay`) populating `VcForecastDaily` + `VcForecastHourly`. 
* `ingest_vc_obs_parallel.py` parsing minute obs into `VcMinuteWeather` with clean tz logic.

So the move is:

1. **In `fetch_historical_forecast(...)`** (lat/lon version), change:

```python
include = "days,hours"
```

to:

```python
include = "days,hours,minutes"
```

so the JSON has `days[*].hours[*].minutes[*]` grid.

2. Add a **minute parser** analogous to `parse_vc_minute_to_record` in `ingest_vc_obs_parallel.py`, but:

   * `data_type = 'historical_forecast'`
   * `forecast_basis_date = basis_date`
   * `forecast_basis_datetime_utc = basis_datetime_utc`
   * `lead_hours` = floor((minute_dt_utc − basis_datetime_utc) / 3600)

   Example stub:

```python
from zoneinfo import ZoneInfo

def parse_vc_minute_hist_forecast(
    minute_data: dict,
    vc_location_id: int,
    basis_date: date,
    basis_datetime_utc: datetime,
    iana_timezone: str,
) -> dict | None:
    epoch = minute_data.get("datetimeEpoch")
    if not epoch:
        return None

    dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)

    tz_local = ZoneInfo(iana_timezone)
    dt_local_aware = dt_utc.astimezone(tz_local)
    dt_local_naive = dt_local_aware.replace(tzinfo=None)
    tzoffset_minutes = int(dt_local_aware.utcoffset().total_seconds() / 60)

    lead_hours = int((dt_utc - basis_datetime_utc).total_seconds() / 3600)

    return {
        "vc_location_id": vc_location_id,
        "data_type": "historical_forecast",
        "forecast_basis_date": basis_date,
        "forecast_basis_datetime_utc": basis_datetime_utc,
        "lead_hours": lead_hours,
        "datetime_epoch_utc": epoch,
        "datetime_utc": dt_utc,
        "datetime_local": dt_local_naive,
        "timezone": minute_data.get("timezone") or iana_timezone,
        "tzoffset_minutes": tzoffset_minutes,
        "temp_f": minute_data.get("temp"),
        "feelslike_f": minute_data.get("feelslike"),
        "dew_f": minute_data.get("dew"),
        "humidity": minute_data.get("humidity"),
        "precip_in": minute_data.get("precip"),
        "precipprob": minute_data.get("precipprob"),
        "cloudcover": minute_data.get("cloudcover"),
        "windspeed_mph": minute_data.get("windspeed"),
        "windgust_mph": minute_data.get("windgust"),
        "pressure_mb": minute_data.get("pressure"),
        "uvindex": minute_data.get("uvindex"),
        "solarradiation": minute_data.get("solarradiation"),
        "conditions": minute_data.get("conditions"),
        "icon": minute_data.get("icon"),
        "source_system": "vc_timeline_minutes",
        "raw_json": minute_data,
    }
```

This is basically your obs parser with `data_type` and basis fields switched.

3. In `process_month(...)` of `ingest_vc_hist_forecast_v2.py` (lat/lon):

   * After you parse the `days[0]["hours"]` hourly records, also dive into `day_data["hours"][*]["minutes"]` and send each minute to `parse_vc_minute_hist_forecast`. 

   * Collect a `minute_records` list and bulk upsert into `VcMinuteWeather`, using your `uq_vc_minute_fcst` index to de-duplicate.

4. Make sure you **reuse the same `lead_days_list`** you’re already using for hourly (e.g. 0,1,2,3,4,5,6). That ensures you get T-0..T-6 15-min curves *consistent with your hourly basis ingestion*, so the trends across lead days are apples-to-apples.

At that point, for every city, you’ll have:

* `wx.vc_forecast_hourly`: T-0..T-6 hourly historical forecasts.
* `wx.vc_minute_weather` with `data_type='historical_forecast'`: minute-level (5–15 min) curves for each basis/target.

From there, all the 15-min “shape of forecast day” features are just queries on `vc_minute_weather`.

---

## 3. Wiring 15-minute series into your existing feature code

### 3.1 High-temp predictor (this is where most of it should live)

You already have `forecast_static.compute_forecast_static_features()` which takes a flat list of temps for T-1 and computes things like:

* `fcst_prev_max_f`, `fcst_prev_min_f`, mean/std, percentiles
* `fcst_prev_hour_of_max` (currently treating index as hour)
* `t_forecast_base` as rounded max temp used by the rest of the pipeline. 

To adapt this to sub-hourly:

1. **Let it know the step size**:

```python
@register_feature_group("forecast_static")
def compute_forecast_static_features(
    fcst_series: list[float],
    step_minutes: int = 60,
) -> FeatureSet:
    ...
    hour_of_max_index = int(np.argmax(arr))
    hour_of_max = hour_of_max_index * step_minutes / 60.0
```

So if you pass 15-minute temps, `hour_of_max` becomes a **float hour since midnight** (e.g. 14.25 = 14:15).

2. Use T-1 *minute-level* path from `vc_minute_weather` as `fcst_series`:

* Query `vc_minute_weather` where:

  * `data_type = 'historical_forecast'`
  * `forecast_basis_date = D-1`
  * `target_date = D` (via `datetime_local::date`)
* Order by `datetime_local` and pass the `temp_f` column as `fcst_series`, `step_minutes=15`.

This gives your predictor:

* A more precise `fcst_prev_hour_of_max`
* A more accurate `t_forecast_base` (since you’re seeing the rounded true peak of the 15-min curve).

3. You can directly reuse your **shape-of-day machinery** for forecast *too*:

* `compute_shape_features(temps_sofar=fcst_series, timestamps_local_sofar=ts_list, t_base=t_forecast_base, step_minutes=15)` – this will tell you spike vs plateau *in the forecast itself*, not just in obs.

So: yes – most of the 15-min goodness should go into the **high-temp predictor** (ordinal model) first.

### 3.2 Edge module: only pass deltas & summaries

Once the predictor is improved, the edge classifier should just see:

* `forecast_high` and `t_forecast_base` from the improved forecast_static features. 
* Summaries such as:

  * `fcst_prev_hour_of_max`
  * `shape` features from forecast vs from obs (plateau vs spike)
  * `delta_vcmax_fcstmax_sofar` and `fcst_remaining_potential` you already compute. 

You don’t need to pipe the full 15-min forecast series into the edge classifier; let the ordinal model + feature functions digest that into a handful of “is this day trending hotter than expected / where is the high likely to be” numbers.

---

## 4. Feature ideas & checks beyond temperature

You explicitly asked: *“besides temperature, what other features can I put in… humidity, cloud cover, etc.; min to max, over time?”* You’re already storing essentially everything; your schema for both hourly and minutes includes:

* `humidity`, `dew_f`, `cloudcover`, `precip_in`, `precipprob`, `windspeed_mph`, `windgust_mph`, `pressure_mb`, `uvindex`, `solarradiation`, `cape`, `cin`, `degreedays`, etc.

Here are concrete, **low-regret** feature families that fit very naturally into your existing modules.

### 4.1 Static forecast features (for T-k forecasts at any resolution)

Using `vc_forecast_hourly` or your new minute-level forecasts, you can do the same things you currently do for temp, but with other variables:

* **Humidity:**

  * `fcst_prev_humidity_mean`, `fcst_prev_humidity_std`
  * `fcst_prev_humidity_min`, `fcst_prev_humidity_max`
  * `fcst_prev_humidity_range` = max − min

* **Cloud cover & radiative variables:**

  * `fcst_prev_cloudcover_mean`
  * `fcst_prev_cloudcover_min` (clear morning)
  * `fcst_prev_cloudcover_afternoon_mean` (12–17 local) – good for “will the high be fully realized?”
  * `fcst_prev_uvindex_max`, `solarradiation_max`

* **Moisture / stability:**

  * `fcst_prev_dewpoint_mean` (`dew_f`)
  * `fcst_prev_temp_minus_dew_mean` → how “dry” the air is (big gap = low RH).
  * `fcst_prev_cape_max`, `fcst_prev_cin_min` for convective days.

All of these can reuse the `compute_forecast_static_features` pattern – just build a new feature group, e.g. `forecast_static_humidity`, that takes a time series for that variable instead of temp.

### 4.2 Time-of-day / shape features (forecast and obs)

You already do shape-of-day on temps in `shape.py`: minutes ≥ thresholds, max_run, morning/afternoon/evening max, slopes. 

Natural variants:

* **Humidity shape**:

  * `humidity_morning_mean`, `humidity_afternoon_mean` (difference indicates drying).
  * `humidity_min_f_sofar`, `humidity_max_f_sofar`, `humidity_range_sofar`.

* **Cloud shape**:

  * `cloudcover_morning_mean` vs `cloudcover_midday_mean`.
  * `sunny_morning_flag` = 1 if mean cloudcover < X before 11am.

* **Joint features**:

  * `temp_minus_dew_morning` vs `temp_minus_dew_afternoon` (evaporation/drying potential).
  * `uvindex_max_time` (hour of peak insolation) vs `forecast_high_time` (do they align?).

Technically these can all be implemented as small wrappers around `compute_shape_features` style logic but applied to humidity or cloudcover arrays instead of temp.

### 4.3 Forecast error features for humidity/cloudcover

You already have `compute_forecast_error_features` for temp (T-1 hourly forecast vs observed hourly). 

You can replicate that for humidity/cloudcover:

* `humidity_err_mean_sofar` = (obs_humidity − fcst_humidity) mean.
* `humidity_err_last1h`, `humidity_err_last3h_mean` (is a moist boundary layer building faster than expected?).
* Same for cloudcover (is it less cloudy than expected → more heating potential).

Implementation-wise, you can:

* Either add optional inputs to `compute_forecast_error_features` and spit out multiple subtables, or
* Create separate feature groups: `forecast_error_humidity`, `forecast_error_cloud`.

---

## 5. Is 15-min curve + spline to 5-min/1-min “cheating”?

Given the docs:

* Historical sub-hourly: 5–10 minute resolution, station-based, aggregated to your requested `minuteinterval`.
* Forecast sub-hourly: 15-minute minimum; *not* interpolated from hourly.
* Historical forecasts: one basis snapshot per model run (typically once per day at midnight UTC).

So:

* Pulling 15-min **forecast** and then **forward-filling** to 1-minute for alignment with Kalshi candles is totally legitimate – you’re just representing “the latest forecast as of now” as a step function.

* Using sub-hourly **observations** (your existing 5-min `vc_minute_weather`) to compute shape-of-day and plateau/spike features is also legitimate – those are real observations.

What you **can’t** do honestly is pretend that historically you had minute-by-minute *forecast updates* (basis snapshots) during the day – those don’t exist; the provider only stores one forecast run per basis date.

So in your backtests:

* Let the **basis-aligned 15-min curves** represent “what was known at basis time”.
* Use **minute obs** to represent realized intraday shape.
* Build your core features and thresholds around those.

Then in live trading, you’ll have more frequent forecast updates (new model runs), which you can use primarily for:

* extra sanity checks (don’t add new risk if the latest run moves sharply against you),
* logging for future research,
  not as “backtested alpha”.

---

## 6. Concrete to-do list for your codebase

Given all this, here’s the actionable checklist:

1. **Ingestion: 15-min historical forecasts**

   * Extend `fetch_historical_forecast` in `ingest_vc_hist_forecast_v2.py` (lat/lon) to set `include="days,hours,minutes"`. 
   * Write `parse_vc_minute_hist_forecast(...)` modeled on `parse_vc_minute_to_record` in `ingest_vc_obs_parallel.py`, but with `data_type='historical_forecast'` and correct basis fields.
   * Bulk upsert those minutes into `wx.vc_minute_weather` using the existing forecast index (`uq_vc_minute_fcst`).

2. **Forecast static features**

   * Update `compute_forecast_static_features` to accept `step_minutes` and compute `fcst_prev_hour_of_max` in hours, not just an index. 
   * Call it using 15-min T-1 forecast paths from `vc_minute_weather` for each city/day.

3. **Forecast shape features**

   * Reuse `compute_shape_features` in `shape.py` on the T-1 forecast minute series with `step_minutes=15` to get forecast plateau/spike metrics.

4. **Additional variables**

   * Add parallel “static” feature groups for humidity, cloudcover, dewpoint, etc. using the same pattern as `forecast_static`.
   * If you like, add forecast error variants (similar to `forecast_error`) for humidity/cloudcover.

5. **Use in predictor vs edge**

   * Keep **most of the new 15-min info in the high-temp predictor** (the ordinal model).
   * Pass only **summaries/deltas** (forecast_high vs market, remaining potential, shape flags) into the edge classifier.

If you want, next step I can sketch the exact SQL/SQLAlchemy query to pull T-1 15-min series from `vc_minute_weather` for a given city/day so you can wire that into the ordinal trainer, and then derive the new “forecast shape” and “forecast humidity” feature groups.
