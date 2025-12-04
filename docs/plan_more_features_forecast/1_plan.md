Yeah, this is a really good direction.

High-frequency *market* data (1-minute Kalshi candles) + *time-varying forecast* data (15-minute Visual Crossing) is exactly the combo you want: the candles are noisy microstructure, the forecasts are the slow, fundamental driver.

Visually crossing’s docs confirm:

* You get sub-hourly data by adding `include=minutes` to the **Timeline API** request.
* Default interval is **15 minutes**, adjustable via `options=minuteinterval_30` etc; forecast data is **not interpolated below 15 minutes** and is only available sub-hourly in certain regions (CONUS + southern Canada, 12–24h horizon).

So your idea:

> Keep 15-min forecasts as the “real” thing, then forward-fill them to a 1-minute grid so they line up with Kalshi 1-minute candles

…is *exactly* what I’d do. It respects the provider’s resolution while giving you cycle-by-cycle features at candle frequency.

Below is a **Markdown spec + code stubs** you can drop into your repo as something like
`docs/VISUAL_CROSSING_15MIN_FORECAST_PLAN.md` and hand straight to your coding agent.

---

````markdown
# Visual Crossing 15-Minute Forecast Integration Plan

## 0. Goal / Context

We are trading Kalshi daily high-temperature brackets using:

- 1-minute Kalshi candlesticks (bid/ask/last/volume)
- A day-high forecast model (ordinal CatBoost)
- An edge classifier that decides when a mispricing is real enough to trade

Right now, the “weather” side is mostly static at the *daily* level.  
We want to add **time-varying sub-hourly forecast features** from Visual Crossing:

- Use Visual Crossing **Timeline API** with `include=minutes` to pull **15-minute forecast & historical minute data**.   
- Treat the 15-minute series as the truth, and **forward-fill to 1-minute** so it aligns with Kalshi 1-minute candles.
- Build features over time such as:
  - current forecast high of the day vs Kalshi implied temp
  - drift in forecast over last X hours
  - how close the current forecast is to *already observed* temps
  - time since last “new high” in either forecast or observed temps

This document specifies:

1. **Data model & API usage**
2. **Python client design + code stubs**
3. **Alignment logic (15-min → 1-min grid)**
4. **Feature engineering ideas**
5. **Tests & sanity checks**

---

## 1. Visual Crossing API – what we will use

### 1.1 Base API & parameters

We will use the **Timeline Weather API**:   

```text
https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/[location]/[start]/[end]
````

Key parameters:

* `key` – API key (from env var `VISUAL_CROSSING_API_KEY`)
* `unitGroup` – use `"us"` for °F, mph, etc.
* `include` – we *must* include `"minutes"` to get sub-hourly data.
* `contentType` – `"json"` for programmatic use
* (optional) `options=minuteinterval_15` – explicitly request 15-minute interval if needed; default sub-hourly forecast interval is already 15 min.

Example (JSON, one day of minute data for Austin):

```text
https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Austin,TX/2025-06-01/2025-06-01?unitGroup=us&include=minutes&key=YOUR_KEY&contentType=json
```

The Timeline JSON has structure:

```jsonc
{
  "latitude": ...,
  "longitude": ...,
  "timezone": "America/Chicago",
  "days": [
    {
      "datetime": "2025-06-01",
      "tempmax": ...,
      "hours": [
        {
          "datetime": "00:00:00",
          "temp": 75.2,
          "minutes": [
            {"datetime": "00:00:00", "temp": 75.2, ...},
            {"datetime": "00:15:00", "temp": 75.1, ...},
            {"datetime": "00:30:00", "temp": 75.1, ...},
            {"datetime": "00:45:00", "temp": 75.0, ...}
          ]
        },
        ...
      ]
    }
  ]
}
```

By default, minute rows are **15-minute** spaced.

### 1.2 Forecast vs historical

* **Historical minutes**: min interval 5–10 minutes, may be interpolated if you ask for shorter intervals.
* **Forecast minutes**:

  * Minimum interval is **15 minutes**
  * Available sub-hourly only in areas covered by high-resolution models (CONUS + southern Canada)
  * Forecast horizon ~12–24 hours depending on model update schedule

> Important: Forecast sub-hourly data is *not* interpolated below 15 minutes by Visual Crossing.
> If we need 1-minute alignment, we must resample ourselves (forward-fill / step function).

### 1.3 Historical *forecast* vs historical *observations*

If we want “what did the forecast say at time t in the past?”, Visual Crossing offers a **Historical Forecast** addon where you can specify the *basis date* (when the forecast was produced).

For now, this plan assumes:

* **Short-term:**

  * For backtests we approximate with current Timeline sub-hourly data (observations + model) for the intra-day shape.
  * That’s fine for learning how the **realized high** relates to the intraday temp path.

* **Future extension:**

  * If we add the “Historical Forecast” subscription, we can extend this client to fetch forecast-as-of basis date and reconstruct exactly what traders knew at each snapshot.

---

## 2. Python client design

### 2.1 Module & files

Create a new module:

```text
src/visual_crossing/
    __init__.py
    client.py
    minutes_loader.py
    tests/
        test_minutes_fetch.py
        test_minutes_alignment.py
```

### 2.2 Config

Use env var for the API key:

* `VISUAL_CROSSING_API_KEY`

Add to our existing config system if we have one (`config.py`), else read directly from `os.environ`.

### 2.3 `client.py` – basic HTTP client & flattening

**Goal:** simple utility to fetch minute-level JSON and convert to a tidy `DataFrame` with one row per minute.

#### 2.3.1 Code stub

```python
# src/visual_crossing/client.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Any

import requests
import pandas as pd


BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"


@dataclass
class VCConfig:
    api_key: str
    unit_group: str = "us"          # Fahrenheit, mph, etc.
    content_type: str = "json"      # we use JSON for code, CSV if needed
    minute_interval: str | None = None  # e.g. "minuteinterval_15"


def get_default_config() -> VCConfig:
    key = os.environ.get("VISUAL_CROSSING_API_KEY")
    if not key:
        raise RuntimeError("VISUAL_CROSSING_API_KEY not set")
    return VCConfig(api_key=key)


def build_timeline_url(location: str, start: str, end: str | None = None) -> str:
    """
    Build Timeline API URL.

    start, end are strings like "2025-06-01" (local date) or dynamic keywords ("yesterday").
    """
    if end:
        return f"{BASE_URL}/{location}/{start}/{end}"
    else:
        return f"{BASE_URL}/{location}/{start}"


def fetch_minutes_json(
    location: str,
    start: str,
    end: str,
    cfg: VCConfig | None = None,
    extra_params: Mapping[str, Any] | None = None,
) -> dict:
    """
    Fetch sub-hourly data (minutes) as JSON using Timeline API.

    Raises requests.HTTPError on failure.
    """
    cfg = cfg or get_default_config()
    url = build_timeline_url(location, start, end)

    params: dict[str, Any] = {
        "unitGroup": cfg.unit_group,
        "include": "minutes",        # critical
        "key": cfg.api_key,
        "contentType": cfg.content_type,
    }
    if cfg.minute_interval:
        params["options"] = cfg.minute_interval
    if extra_params:
        params.update(extra_params)

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def minutes_json_to_dataframe(payload: dict) -> pd.DataFrame:
    """
    Flatten Timeline JSON with minutes into a pandas DataFrame.

    Output columns (minimum):
      - location
      - tz
      - local_datetime (timezone-naive local time; we will localize later)
      - date (YYYY-MM-DD)
      - hour
      - minute
      - temp, feelslike, dew, humidity, etc. as available

    NOTE: 'minutes' is nested inside each 'hour' in each 'day'
          (see Visual Crossing sub-hourly docs).
    """
    tz = payload.get("timezone")
    address = payload.get("address") or payload.get("resolvedAddress")

    rows: list[dict[str, Any]] = []
    for day in payload.get("days", []):
        day_date = day["datetime"]  # "2025-06-01"
        for hour in day.get("hours", []):
            hour_base = hour.get("datetime", "00:00:00")  # "00:00:00"
            for m in hour.get("minutes", []):
                # minute datetime is "HH:MM:SS" (local clock time)
                minute_time = m["datetime"]               # "00:15:00"
                ts_str = f"{day_date}T{minute_time}"

                row = {
                    "location": address,
                    "tz": tz,
                    "local_datetime_str": ts_str,
                    "date": day_date,
                    "minute_time": minute_time,
                }

                # Copy selected numeric fields from the minute node
                for key in ("temp", "feelslike", "dew", "humidity", "precip", "cloudcover"):
                    if key in m:
                        row[key] = m[key]

                rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["local_datetime"] = pd.to_datetime(df["local_datetime_str"])
    return df
```

---

## 3. Alignment to Kalshi 1-minute candles

### 3.1 Design

We want:

* A **1-minute grid** for each city/day from market open → market close (or entire UTC day).
* For each minute:

  * we have Kalshi candle (bid/ask/last/volume)
  * we have the **latest known Visual Crossing minute forecast** (15-min step function, forward-filled)

Steps:

1. **Fetch minutes** for [event_date, event_date] in local time.
2. Flatten to `df_minutes` with `local_datetime`.
3. Convert to timezone-aware `ts_local` and then to **UTC** (or vice versa) in the *same convention* as your Kalshi candles.
4. Create a full **1-minute index** between `event_start` and `event_end`.
5. Reindex `df_minutes` onto that 1-minute grid with **forward fill**.

### 3.2 Code stub

```python
# src/visual_crossing/minutes_loader.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd
import pytz

from .client import fetch_minutes_json, minutes_json_to_dataframe


def build_1min_forecast_series(
    location: str,
    date: str,              # "YYYY-MM-DD" local
    local_tz: str,          # e.g. "America/Chicago"
) -> pd.DataFrame:
    """
    Fetch 15-min Visual Crossing data for a given local date and
    return a 1-min forward-filled forecast series.

    Output columns:
      - ts_local (timezone-aware)
      - ts_utc
      - temp_fcst
      - feelslike_fcst
      - ... (other fcst fields)
    """
    payload = fetch_minutes_json(location=location, start=date, end=date)
    df_min = minutes_json_to_dataframe(payload)
    if df_min.empty:
        raise RuntimeError(f"No minute data returned for {location} {date}")

    tz = pytz.timezone(local_tz)
    df_min["ts_local"] = df_min["local_datetime"].dt.tz_localize(tz)
    df_min = df_min.sort_values("ts_local")

    # Build 1-min grid for the full local day
    day_start = tz.localize(datetime.fromisoformat(f"{date}T00:00:00"))
    day_end = day_start + timedelta(days=1)
    idx_1min = pd.date_range(start=day_start, end=day_end, freq="1min", inclusive="left")

    df_min = df_min.set_index("ts_local").sort_index()

    # Limit to at most the day
    df_min = df_min.reindex(
        df_min.index[(df_min.index >= day_start) & (df_min.index < day_end)]
    )

    # Forward-fill to full 1-min grid
    df_fcst_1min = df_min.reindex(idx_1min).ffill()

    df_fcst_1min["ts_local"] = df_fcst_1min.index
    df_fcst_1min["ts_utc"] = df_fcst_1min["ts_local"].dt.tz_convert("UTC")

    # Rename key fields
    rename_map = {
        "temp": "temp_fcst",
        "feelslike": "feelslike_fcst",
    }
    df_fcst_1min = df_fcst_1min.rename(columns=rename_map)

    return df_fcst_1min.reset_index(drop=True)
```

### 3.3 Joining to Kalshi candles

Assume you already have a Kalshi 1-minute candles DataFrame:

```python
# columns: ts_utc, bid, ask, last, volume, ...
candles_1m: pd.DataFrame
```

Join:

```python
def join_forecast_to_candles(
    candles_1m: pd.DataFrame,
    fcst_1m: pd.DataFrame,
) -> pd.DataFrame:
    merged = pd.merge_asof(
        candles_1m.sort_values("ts_utc"),
        fcst_1m.sort_values("ts_utc"),
        on="ts_utc",
        direction="backward",    # use latest forecast <= candle time
        tolerance=pd.Timedelta("30min"),  # configurable
    )
    return merged
```

Then wire this into your existing **edge feature pipeline** so each snapshot has:

* `temp_fcst` (current 15-min forecast at that minute)
* `feelslike_fcst`
* potentially forecast high/low features (see next section).

---

## 4. Feature engineering ideas from 15-min forecasts

Once you have a 1-minute series of Visual Crossing forecasts aligned to each Kalshi snapshot, you can add features like:

1. **Forecast high vs market implied temp**

   * Current best guess of day high: `temp_fcst_max_today`
   * Edge: `fcst_max_edge = temp_fcst_max_today - kalshi_implied_high`

2. **Forecast drift**

   * Change in forecast high over last 1h / 3h / since midnight:

     * `fcst_max_delta_1h = temp_fcst_max_today - temp_fcst_max_1h_ago`
     * `fcst_max_delta_3h`, `fcst_max_delta_since_midnight`
   * Same for current forecast temp at current minute.

3. **Observed vs forecast gap**

   * `obs_fcst_gap = forecast_temp_now - observed_temp_now`
   * `obs_max_vs_fcst_max_gap = observed_max_so_far - temp_fcst_max_today`

4. **Phase indicators**

   * Is forecasted high already in the past?

     * `is_past_fcst_peak = (time_now > time_of_forecasted_high)`
   * How many hours until forecasted high time; or since we passed it.

5. **Volatility of forecast**

   * Standard deviation of forecast high over last N hours (how stable is the model’s belief).
   * Count of times the forecast high has changed by ≥1°F today.

All of these can feed both the **ordinal model** and the **edge classifier**.

---

## 5. Tests & sanity checks

Create tests under `src/visual_crossing/tests/`.

### 5.1 `test_minutes_fetch.py`

Goals:

* Verify API connectivity and basic parsing.
* Assert reasonable row counts and monotonic times.

Pseudo-tests:

```python
def test_fetch_single_day_minutes():
    cfg = get_default_config()
    payload = fetch_minutes_json("Austin,TX", "2025-06-01", "2025-06-01", cfg)
    df = minutes_json_to_dataframe(payload)
    assert not df.empty
    # roughly 96 rows for 24h * 4 intervals
    assert 80 <= len(df) <= 120
    # times strictly increasing
    assert df["local_datetime"].is_monotonic_increasing
```

### 5.2 `test_minutes_alignment.py`

Goals:

* Check that 15-min → 1-min resample behaves.

```python
def test_build_1min_forecast_series_shape():
    df_fcst = build_1min_forecast_series("Austin,TX", "2025-06-01", "America/Chicago")
    # 24 * 60 = 1440 minutes per day
    assert len(df_fcst) == 24 * 60
    assert df_fcst["ts_local"].is_monotonic_increasing
    # 15-min values should be constant within each 15-min block
    block = df_fcst.set_index("ts_local").resample("15min")["temp_fcst"].first()
    reconstructed = block.reindex(df_fcst["ts_local"]).ffill().values
    assert (df_fcst["temp_fcst"].values == reconstructed).all()
```

### 5.3 End-to-end join with sample Kalshi candles

You can add a test that:

* Builds a dummy 1-minute Kalshi DF for one day.
* Fetches Visual Crossing 1-minute forecasts.
* Joins them.
* Asserts no nulls in `temp_fcst` for times where we expect coverage (e.g., 06:00–23:00 local).

---

## 6. Strategy notes / what I think

1. **Using 15-minute forecasts is absolutely worth it.**
   The market reacts to updated forecasts; having a time series of those forecasts is the “fundamental” side of microstructure.

2. **Forward-fill to 1-minute is the correct way to align.**
   Visual Crossing explicitly says they *don’t* interpolate forecast below 15 minutes.
   That means any 1-minute interpolation we do is just a representation of “the latest forecast as of now,” not extra information.

3. **You don’t have to abandon 1-minute candles.**
   You can:

   * Keep Kalshi candles at 1-minute resolution.
   * Add 1-minute forecast features via forward-filled 15-minute data.
   * Optionally experiment with 5/15-minute aggregation for modeling, but still fire trades every minute.

4. **Historical forecast vs observations**
   If you later care deeply about “what was the forecast *as of 09:00 yesterday*”, you may want to add the **Historical Forecast addon**. That would be a straightforward extension of the same client (`basisDate` support in the Timeline API).

---

## 7. Next steps for the agent

1. Implement `visual_crossing/client.py` and `visual_crossing/minutes_loader.py` per the stubs above.
2. Wire a new step into the existing Kalshi data pipeline:

   * For each city/day in the edge training set, materialize a 1-minute Visual Crossing forecast series and join it to the candles.
3. Add unit tests for:

   * minute fetch
   * 1-minute resample
   * join alignment
4. Once we have a few days of aligned data, generate quick plots:

   * Meteogram of forecast vs observed temps vs Kalshi implied temp.
   * Use these to refine which forecast-drift / gap features look most predictive.

```

---

If you want, I can also draft a tiny **`scripts/debug_visual_crossing_minutes.py`** that fetches one day for one city and plots a quick meteogram so you can eyeball that the 15-min Visual Crossing path lines up with your observed temps and Kalshi candles.
::contentReference[oaicite:13]{index=13}
```
