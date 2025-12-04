Short version:

* Visual Crossing **does** let you go back and get *historical forecasts*, and you can combine that with **sub-hourly (`minutes`) data**.
* But the historical forecasts are **stored once per day per model run (midnight UTC)**, not every 5–15 minutes – so you **can’t** reconstruct “the forecast as of 09:15, 09:30, 09:45, …” in the past. You only get “the full forecast as of basis date X” (typically once per day).
* You *can* write a debug script to prove this to yourself: loop over basis dates, request `include=minutes`, and compare the minute-level forecast for a given target day. You should see **one forecast per day**, not a new one every 15 minutes.

Below I’ll:

1. Explain what the Historical Forecast API really provides (and its limits).
2. Show how to probe whether 15-minute historical forecasts are actually changing.
3. Give you two concrete debug scripts you can drop into your repo.

---

## 1. What Visual Crossing actually offers for historical forecasts

### 1.1 Basis date: you get one snapshot per day

Visual Crossing’s **Historical Forecast** docs are clear about the granularity:

> “Visual Crossing currently provides the historical forecasts based on the full forecast model runs stored at **midnight UTC each day**.”

You access these via the **Timeline API** with `forecastBasisDate` or `forecastBasisDay`:

```text
.../timeline/London,UK/2023-05-01/2023-05-15
    ?unitGroup=us
    &include=days   # or hours/minutes
    &forecastBasisDate=2023-05-01
    &key=YOUR_KEY
```

* `forecastBasisDate=YYYY-MM-DD` says:
  “Give me the forecast that was generated on this *basis date* (daily model run).”
* `forecastBasisDay=k` is shorthand: basis date = start date − k days.

There is **no time-of-day** in the basis parameter: you can’t ask for “forecast as of 10:15 am” in the past – only “as of the model run on this basis date”.

So your dream of sliding the forecast every 15 minutes *historically* (as if someone captured each rolling update) is not possible with this product: you get one frozen forecast snapshot per day.

### 1.2 Sub-hourly + historical forecast: how they combine

The **sub-hourly Timeline doc** says:

* You add `include=minutes` to get minute-level data.
* Historical data can be as fine as **5–10 minutes**, depending on station availability.
* **Forecast** sub-hourly is available at **15-minute minimum** and **is *not* interpolated below 15 minutes**.

Because the Historical Forecast feature is layered on top of the **same Timeline API** , in practice you can do:

```text
.../timeline/Austin,TX/TARGET_DATE/TARGET_DATE
    ?unitGroup=us
    &include=minutes
    &forecastBasisDate=BASIS_DATE
    &key=YOUR_KEY
```

and you should get **15-minute forecast temps for that target day, as predicted on the BASIS_DATE model run**.

So:

* You **can** get a *15-minute forecast path* for each day, as of each basis date (one snapshot per day).
* You **cannot** get “the forecast as it updated every 15 minutes during the day” historically. At best you get “as of midnight UTC, here’s the 15-min forecast for the whole day”.

This is still useful: for each day d you can have:

* Basis d−3 forecast 15-min series
* Basis d−2 forecast 15-min series
* Basis d−1 forecast 15-min series
* Basis d forecast 15-min series

…which is already a rich feature set for “how did belief about that day evolve over multiple days”. But it won’t match the live 5-minute update cadence you see *today*.

---

## 2. How to empirically check what you get (the experiment you described)

You want to answer:

> “Over the last week, does the forecast for a given target day change every 15 minutes, or only once per day?”

The right experiment:

1. Pick a **city** and **target date** (e.g., Austin, target date = today or yesterday).
2. For each **basis date** in the last 7 days:

   * Call Timeline with `include=minutes&forecastBasisDate=basis_date`.
   * Flatten the `minutes` data for the target day.
3. Compare:

   * For each basis date, what is the **max temp** forecast for that target day?
   * For a specific time (say 15:00 local), what was the forecast temp at that minute as-of each basis date?

If the API supported 15-minute basis resolution, you’d see dozens/hundreds of distinct forecasts per day. In reality you should see **one forecast per basis date** (daily).

Below are two scripts to do exactly that.

---

## 3. Script 1 – Debug historical forecast at minute resolution

Call this `scripts/debug_vc_hist_forecast_minutes.py`.

This:

* Takes `location`, `target_date`, and `days_back`.
* Loops basis dates back `days_back` days.
* For each basis, requests **minute-level** historical forecast for `target_date`.
* Stores, for each basis date:

  * Forecast day max temp (`tempmax_from_minutes`)
  * Forecast temp at a specific time (e.g., 15:00 local)
* Prints a little table and writes a CSV so you can graph it.

```python
#!/usr/bin/env python3
"""
Debug script: Visual Crossing historical forecast at minute resolution.

For a given location + target_date, this script:
  - Loops over basis dates (target_date - k days)
  - Requests Timeline with include=minutes & forecastBasisDate
  - Extracts:
      * tempmax based on minute data
      * temp at a specific local time-of-day
  - Prints a summary table and writes results to CSV.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any

import requests
import pandas as pd


API_BASE = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"


def fetch_hist_fcst_minutes(
    location: str,
    target_date: str,
    basis_date: str,
    unit_group: str = "us",
    api_key: str | None = None,
) -> dict:
    api_key = api_key or os.environ.get("VISUAL_CROSSING_API_KEY")
    if not api_key:
        raise RuntimeError("VISUAL_CROSSING_API_KEY not set")

    url = f"{API_BASE}/{location}/{target_date}/{target_date}"
    params = {
        "unitGroup": unit_group,
        "include": "minutes",
        "forecastBasisDate": basis_date,
        "key": api_key,
        "contentType": "json",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def minutes_payload_to_df(payload: dict) -> pd.DataFrame:
    """Flatten minutes for the single target day."""
    rows: List[Dict[str, Any]] = []
    tz = payload.get("timezone")
    address = payload.get("address") or payload.get("resolvedAddress")

    for day in payload.get("days", []):
        day_date = day["datetime"]  # "YYYY-MM-DD"
        for hour in day.get("hours", []):
            for m in hour.get("minutes", []):
                # minute datetime is "HH:MM:SS"
                minute_time = m["datetime"]
                ts_str = f"{day_date}T{minute_time}"
                row = {
                    "location": address,
                    "tz": tz,
                    "date": day_date,
                    "time": minute_time,
                    "temp": m.get("temp"),
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["local_dt"] = pd.to_datetime(df["date"] + " " + df["time"])
    return df


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: debug_vc_hist_forecast_minutes.py LOCATION TARGET_DATE DAYS_BACK\n"
            "Example: debug_vc_hist_forecast_minutes.py 'Austin,TX' 2025-12-03 7"
        )
        sys.exit(1)

    location = sys.argv[1]
    target_date = sys.argv[2]  # "YYYY-MM-DD"
    days_back = int(sys.argv[3])

    target_dt = datetime.fromisoformat(target_date)

    records = []

    for k in range(days_back + 1):  # 0..days_back
        basis_dt = target_dt - timedelta(days=k)
        basis_date = basis_dt.strftime("%Y-%m-%d")
        print(f"Fetching basis_date={basis_date} -> target_date={target_date}...")

        payload = fetch_hist_fcst_minutes(location, target_date, basis_date)
        df_min = minutes_payload_to_df(payload)

        if df_min.empty:
            print(f"  !! No minute data returned for basis_date={basis_date}")
            continue

        # Estimate tempmax from the minute series
        tempmax_minutes = float(df_min["temp"].max())

        # Example: temp at local 15:00 (3pm) if available
        mask_15 = df_min["local_dt"].dt.strftime("%H:%M") == "15:00"
        temp_at_15 = float(df_min.loc[mask_15, "temp"].iloc[0]) if mask_15.any() else float("nan")

        records.append(
            {
                "basis_date": basis_date,
                "target_date": target_date,
                "tempmax_from_minutes": tempmax_minutes,
                "temp_at_15_local": temp_at_15,
                "n_minutes": len(df_min),
            }
        )

    if not records:
        print("No data collected. Check API key, plan, or parameters.")
        return

    df_summary = pd.DataFrame(records).sort_values("basis_date")
    print("\nHistorical forecast summary:")
    print(df_summary.to_string(index=False))

    out_csv = f"vc_hist_forecast_{location.replace(',', '_')}_{target_date}.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"\nSaved summary to {out_csv}")


if __name__ == "__main__":
    main()
```

**What you should expect:**

* `n_minutes` ~ 96 (24h × 4) for each basis (15-min data).
* `tempmax_from_minutes` and `temp_at_15_local` will vary **by basis date**, but there will be only **one row per basis date**, not per 15-minute basis.

If you try `days_back=7`, you’ll get up to 8 rows (basis=target_date, target_date−1, …, target_date−7) – one forecast snapshot per day.

If you see only a single row or no change across basis dates, that likely means *either*:

* Your plan doesn’t include Historical Forecast, so basisDate is being ignored and you’re just seeing “current best forecast” repeated, or
* You picked a target date too far in the past/future for which historical forecast isn’t available (docs say coverage starts from Jan 1, 2020).

---

## 4. Script 2 – Check sub-hourly *history* vs *forecast*

You also said:

> “I might already be doing hourly historical for the day itself… I need to know if 15-minute interpolation really adds anything.”

So here’s a second script that, for **yesterday**, compares:

* ***Historical observations*** at 15-min (`include=minutes`, no basisDate).
* ***Historical forecast*** for yesterday as of a given basis date (e.g. `forecastBasisDay=1`).

It will let you see:

* How close the midnight forecast was to what actually happened in each 15-minute bin.
* Whether having 15-minute forecast (rather than hourly) meaningfully changes the story.

Call it `scripts/debug_vc_minutes_obs_vs_histfcst.py`:

```python
#!/usr/bin/env python3
"""
Compare Visual Crossing sub-hourly observations vs historical forecasts
for a single day.

- Observations: include=minutes (no forecastBasisDate)
- Historical forecast: include=minutes + forecastBasisDay
"""

import os
from datetime import datetime, timedelta

import requests
import pandas as pd

API_BASE = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"


def fetch_minutes(location, start, end, extra_params=None):
    api_key = os.environ.get("VISUAL_CROSSING_API_KEY")
    if not api_key:
        raise RuntimeError("VISUAL_CROSSING_API_KEY not set")

    url = f"{API_BASE}/{location}/{start}/{end}"
    params = {
        "unitGroup": "us",
        "include": "minutes",
        "key": api_key,
        "contentType": "json",
    }
    if extra_params:
        params.update(extra_params)

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def flatten_minutes(payload):
    rows = []
    for day in payload.get("days", []):
        d = day["datetime"]
        for hour in day.get("hours", []):
            for m in hour.get("minutes", []):
                rows.append(
                    {
                        "date": d,
                        "time": m["datetime"],
                        "temp": m.get("temp"),
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["local_dt"] = pd.to_datetime(df["date"] + " " + df["time"])
    return df


def main():
    location = "Austin,TX"

    # Yesterday local date (approx – adjust for tz if you want)
    today = datetime.utcnow().date()
    target_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Target date: {target_date}")

    # Observations (minutes) for that date
    obs_payload = fetch_minutes(location, target_date, target_date)
    df_obs = flatten_minutes(obs_payload)
    print(f"Obs minutes: {len(df_obs)} rows")

    # Historical forecast as of 1 day before (forecastBasisDay=1)
    fcst_payload = fetch_minutes(
        location,
        target_date,
        target_date,
        extra_params={"forecastBasisDay": 1},
    )
    df_fcst = flatten_minutes(fcst_payload)
    print(f"Hist-forecast minutes: {len(df_fcst)} rows")

    if df_obs.empty or df_fcst.empty:
        print("No data; check API key or plan.")
        return

    # Merge on local time to compare
    df_merge = pd.merge(
        df_obs[["local_dt", "temp"]].rename(columns={"temp": "temp_obs"}),
        df_fcst[["local_dt", "temp"]].rename(columns={"temp": "temp_fcst"}),
        on="local_dt",
        how="inner",
    ).sort_values("local_dt")

    # Basic stats
    df_merge["diff"] = df_merge["temp_fcst"] - df_merge["temp_obs"]
    print(df_merge.head())
    print("\nMean absolute error:", df_merge["diff"].abs().mean())
    print("Max abs error:", df_merge["diff"].abs().max())

    out_csv = f"vc_obs_vs_histfcst_{location.replace(',', '_')}_{target_date}.csv"
    df_merge.to_csv(out_csv, index=False)
    print(f"\nSaved comparison to {out_csv}")


if __name__ == "__main__":
    main()
```

This doesn’t solve the “15-minute basis” problem, but it will tell you:

* How good the midnight 15-minute forecast was for yesterday at each minute.
* Whether sub-hourly actually adds signal vs just using daily/hourly forecast.

---

## 5. What this means for your modeling

Given the docs and what the scripts will show:

* Historical forecasts are **daily snapshots** (basisDate at midnight UTC). You can’t replay the *intraday* refreshes every 15 minutes historically.
* You **can** still:

  * Use sub-hourly forecasts (15-min) *within* each basis snapshot.
  * Get multiple basis snapshots for the same target day (5-day-ahead, 3-day-ahead, 1-day-ahead, etc.).
  * Use that as a powerful feature set for your Kalshi models: “forecast path across days” + “intraday temperature path forecast from midnight”.

For *live trading*, you can absolutely poll the 15-minute forecast going forward (Timeline with `include=minutes`), and you’ll get the new sub-hourly forecast as the models update. For *backtesting*, you’ll have to approximate by:

* Using historical forecasts at daily resolution (basisDate); and
* Using historical observations (sub-hourly) for intra-day realized temps.

The two debug scripts above will give you a concrete feel for what you actually get, so you’re not designing features on assumptions.

If you want, I can next help you:

* Wrap these scripts into a clean `visual_crossing` Python module, and
* Sketch how to pipe “basis_date × target_date × 15-min series” into your existing edge feature pipeline (e.g., as lagged features like “1-day-ahead forecast high” vs realized high).
