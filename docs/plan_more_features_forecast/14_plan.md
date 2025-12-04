Got it — you want very concrete, code-aware instructions your agent can drop in, wired into your existing modules, plus how to test each piece. Let’s layer in *specific* new features that:

* Use the data you’re already ingesting / planning (hourly + 15‑min VC forecasts, multi‑lead daily highs),
* Fit into your existing feature modules (`forecast.py`, `snapshot_builder.py`, `base.py`),
* Are easy to unit test and ablation test.

I’ll give you **4 feature groups**, and for each:

1. What the feature is and why it’s useful.
2. Exact file + function to change or add.
3. Suggested code stub / pattern.
4. How to hook it into `snapshot_builder`.
5. How to test it.

Where I reference existing code, I’ll anchor to line ranges so your agent can find it quickly.

---

## 1) Forecast integer-boundary / rounding-pressure features

### Why

For Kalshi, integer thresholds matter a lot. You *already* compute `fcst_prev_frac_part` in `compute_forecast_static_features`, but:

* It’s asymmetric (max – round(max)) and can go negative.
* You don’t yet have “distance to nearest integer” or a binary “near boundary” flag.

These are cheap, stable features and directly tied to bracket behavior.

### Where

**File:** `models/features/forecast.py`
**Function:** `compute_forecast_static_features` (already registered as `"forecast_static"`).

You already compute:

```python
arr = np.asarray(fcst_series, dtype=np.float64)
max_f = float(arr.max())
...
# Fractional part of max
frac_part = max_f - round(max_f)
...
features = {
    "fcst_prev_max_f": max_f,
    ...
    "fcst_prev_frac_part": frac_part,  # (in your file this line currently uses a bad variable name)
    "fcst_prev_hour_of_max": hour_of_max,
    "t_forecast_base": t_forecast_base,
}
```

### Change / Add

Right after you compute `max_f`, add these:

```python
    # Fractional components for bracket behavior
    raw_frac = max_f - np.floor(max_f)  # in [0, 1)
    distance_to_int = min(raw_frac, 1.0 - raw_frac)  # distance to nearest whole degree
    near_boundary_flag = float(distance_to_int < 0.25)  # within 0.25°F of an integer
```

Then in `features`, make sure you:

* Fix the existing bug (it references `frac` instead of `frac_part` in your snippet),
* Add the new fields:

```python
    features = {
        "fcst_prev_max_f": max_f,
        "fcst_prev_min_f": min_f,
        "fcst_prev_mean_f": mean_f,
        "fcst_prev_std_f": std_f,
        "fcst_prev_q10_f": float(q10),
        "fcst_prev_q25_f": float(q25),
        "fcst_prev_q50_f": float(q50),
        "fcst_prev_q75_f": float(q75),
        "fcst_prev_q90_f": float(q90),

        # Existing “frac part” (keep for backward compat)
        "fcst_prev_frac_part": raw_frac,

        # NEW: more interpretable bracket features
        "fcst_prev_distance_to_int": distance_to_int,
        "fcst_prev_near_boundary_flag": near_boundary_flag,

        "fcst_prev_hour_of_max": hour_of_max,
        "t_forecast_base": t_forecast_base,
    }
```

### Register in base feature list

**File:** `models/features/base.py`
Find `NUMERIC_FEATURE_COLS` and append:

```python
"fcst_prev_distance_to_int",
"fcst_prev_near_boundary_flag",
```

(You already include `fcst_prev_frac_part`; keep that.)

### How to test

* **Unit test:** Create `tests/test_forecast_static.py`:

  ```python
  from models.features.forecast import compute_forecast_static_features

  def test_forecast_integer_boundary_features():
      fs = compute_forecast_static_features([79.2, 82.7, 84.7, 84.7])
      f = fs.to_dict()
      assert "fcst_prev_distance_to_int" in f
      assert "fcst_prev_near_boundary_flag" in f

      max_f = f["fcst_prev_max_f"]
      raw_frac = max_f - int(max_f)
      expected_distance = min(raw_frac, 1.0 - raw_frac)
      assert abs(f["fcst_prev_distance_to_int"] - expected_distance) < 1e-6
  ```

* **End-to-end smoke:** Run `build_snapshot_dataset` for a small Austin window and confirm the new columns show up and are non-null when forecast data exists.

---

## 2) Forecast peak-window & timing from 15‑min / hourly curves

This is where your new 15‑minute VC curves start to matter.

### Why

You want features like:

* Forecasted hour-of-max as a **true float hour** (14.25 for 2:15pm),
* How long the forecast stays near the max (plateau vs sharp spike),
* Eventually: “has the forecasted peak already passed by the current snapshot time?”

We’ll start with **day-static** peak-window features (same for all snapshots on that day). Later you can add snapshot-dependent “has_peak_passed” flags if you want.

### Where

**File:** `models/features/forecast.py`

Add a new feature group:

```python
from datetime import datetime

@register_feature_group("forecast_peak_window")
def compute_forecast_peak_window_features(
    temps_f: list[float],
    timestamps: list[datetime],
    step_minutes: int = 60,
    peak_band_width_f: float = 1.0,
) -> FeatureSet:
    """
    Compress forecast curve (hourly or 15-min) into peak-timing features.

    Assumes temps_f and timestamps are same length and sorted.
    """
    if not temps_f or not timestamps or len(temps_f) != len(timestamps):
        return FeatureSet(name="forecast_peak_window", features={})

    arr = np.asarray(temps_f, dtype=np.float64)
    tmax = float(arr.max())

    idx_max = int(np.argmax(arr))
    ts_max = timestamps[idx_max]

    minutes_since_midnight = (
        ts_max.hour * 60 + ts_max.minute + ts_max.second / 60.0
    )
    hour_of_max_float = minutes_since_midnight / 60.0

    within_band = np.where(arr >= tmax - peak_band_width_f)[0]
    if within_band.size > 0:
        duration_minutes = (within_band[-1] - within_band[0] + 1) * step_minutes
    else:
        duration_minutes = step_minutes

    features = {
        "fcst_peak_temp_f": tmax,
        "fcst_peak_hour_float": hour_of_max_float,
        "fcst_peak_band_width_min": float(duration_minutes),
        "fcst_peak_step_minutes": float(step_minutes),
    }
    return FeatureSet(name="forecast_peak_window", features=features)
```

### Hook it into the pipeline

**File:** `models/data/snapshot_builder.py`

In `build_snapshot_dataset`, after you load `fcst_hourly_df` (and later when you have 15‑min), compute the day-level forecast peak features once per day:

```python
# After:
# fcst_daily = load_historical_forecast_daily(...)
# fcst_hourly_df = load_historical_forecast_hourly(...)
fcst_peak_fs = None
if include_forecast_features and fcst_hourly_df is not None and not fcst_hourly_df.empty:
    tmp = fcst_hourly_df.sort_values("datetime_local").copy()
    temps = tmp["temp_f"].tolist()
    times = pd.to_datetime(tmp["datetime_local"]).tolist()
    step_minutes = int(
        (times[1] - times[0]).total_seconds() / 60.0
    ) if len(times) > 1 else 60

    from models.features.forecast import compute_forecast_peak_window_features
    fcst_peak_fs = compute_forecast_peak_window_features(
        temps_f=temps,
        timestamps=times,
        step_minutes=step_minutes,
    )
else:
    from models.features.base import FeatureSet
    fcst_peak_fs = FeatureSet(name="forecast_peak_window", features={})
```

Now you need to pass `fcst_peak_fs` into `build_single_snapshot`. Easiest pattern: add an optional parameter.

```python
def build_single_snapshot(
    city: str,
    day: date,
    snapshot_hour: int,
    day_obs_df: pd.DataFrame,
    settle_f: int,
    fcst_daily: Optional[dict] = None,
    fcst_hourly_df: Optional[pd.DataFrame] = None,
    include_forecast: bool = True,
    fcst_peak_fs: Optional[FeatureSet] = None,  # NEW
) -> Optional[dict]:
    ...
```

And in the call from `build_snapshot_dataset`:

```python
row = build_single_snapshot(
    city=city_id,
    day=day,
    snapshot_hour=snapshot_hour,
    day_obs_df=day_obs,
    settle_f=settle_f,
    fcst_daily=fcst_daily,
    fcst_hourly_df=fcst_hourly_df,
    include_forecast=include_forecast_features,
    fcst_peak_fs=fcst_peak_fs,
)
```

Inside `build_single_snapshot`, once you’ve assembled `row` with partial‑day, shape, rules, calendar, quality, add:

```python
    if fcst_peak_fs is not None:
        row.update(fcst_peak_fs.to_dict())
```

(Those features are constant per day, so it’s safe to reuse the same FeatureSet across all snapshots for that day.)

### Add to base feature list

`models/features/base.py` → `NUMERIC_FEATURE_COLS`:

```python
"fcst_peak_temp_f",
"fcst_peak_hour_float",
"fcst_peak_band_width_min",
"fcst_peak_step_minutes",
```

### How to test

* Craft a fake forecast curve (e.g., 15‑min temps ramp from 60→90, flat at 90 for 1 hour, then down).

* Call `compute_forecast_peak_window_features` and assert:

  * `fcst_peak_temp_f` == 90,
  * `fcst_peak_hour_float` matches the time of first 90,
  * `fcst_peak_band_width_min` ≈ 60.

* Run `build_snapshot_dataset` for 1–2 days and inspect these new columns; they should be identical across all snapshot hours for the same day.

---

## 3) Forecast drift & volatility across leads (T‑k daily highs)

This is the “how has the forecast for this day moved over the last 7 days?” piece — really important for confidence and risk sizing. Forecast spread/volatility is known to correlate with uncertainty in ensemble systems.

You’re already ingesting multi‑lead daily forecasts into `wx.vc_forecast_daily` (via `ingest_vc_hist_forecast_v2.py`), and the plan doc explicitly calls out drift features.

### Loader: pull multi‑lead daily highs

**File:** `models/data/loader.py`

Add a helper that returns all available leads 0–6 for a given (city, target_date):

```python
def load_historical_forecast_daily_multi(
    session: Session,
    city: str,
    target_date: date,
    max_lead_days: int = 6,
) -> pd.DataFrame:
    """
    Load multi-lead daily forecasts for (city, target_date).

    Returns DataFrame with columns: ['lead_days', 'tempmax_f', ...].
    """
    from src.db.models import VcForecastDaily, VcLocation  # adjust import paths

    q = (
        session.query(
            VcForecastDaily.lead_days,
            VcForecastDaily.tempmax_f,
        )
        .join(VcLocation, VcForecastDaily.vc_location_id == VcLocation.id)
        .filter(
            VcLocation.city_code == city,
            VcForecastDaily.data_type == "historical_forecast",
            VcForecastDaily.target_date == target_date,
            VcForecastDaily.lead_days <= max_lead_days,
        )
        .order_by(VcForecastDaily.lead_days.asc())
    )

    df = pd.read_sql(q.statement, session.bind)
    return df
```

(Your agent can match the actual ORM names/paths.)

### Feature group in `forecast.py`

Add:

```python
@register_feature_group("forecast_drift")
def compute_forecast_drift_features(
    daily_multi_df: pd.DataFrame,
) -> FeatureSet:
    """
    Compress multi-lead daily highs into drift/volatility features.

    Expects columns ['lead_days', 'tempmax_f'].
    """
    if daily_multi_df is None or daily_multi_df.empty:
        return FeatureSet(name="forecast_drift", features={})

    df = daily_multi_df.dropna(subset=["tempmax_f"]).copy()
    if df.empty:
        return FeatureSet(name="forecast_drift", features={})

    df = df.sort_values("lead_days")
    leads = df["lead_days"].to_numpy(dtype=np.float64)
    highs = df["tempmax_f"].to_numpy(dtype=np.float64)

    # Anchor at T-1 if present, else closest-in lead
    mask_t1 = df["lead_days"] == 1
    if mask_t1.any():
        anchor_high = float(highs[mask_t1.argmax()])
    else:
        anchor_high = float(highs[-1])

    deltas = highs - anchor_high

    features = {
        "fcst_drift_num_leads": float(len(highs)),
        "fcst_drift_std_f": float(np.std(highs, ddof=1)) if len(highs) > 1 else 0.0,
        "fcst_drift_max_upside_f": float(np.max(highs) - anchor_high),
        "fcst_drift_max_downside_f": float(anchor_high - np.min(highs)),
        "fcst_drift_mean_delta_f": float(np.mean(deltas)),
    }

    if len(highs) >= 2:
        slope, _ = np.polyfit(leads, highs, deg=1)
        features["fcst_drift_slope_f_per_lead"] = float(slope)
    else:
        features["fcst_drift_slope_f_per_lead"] = 0.0

    return FeatureSet(name="forecast_drift", features=features)
```

### Wire into snapshot builder

In `build_snapshot_dataset` (same file snippet as before):

Just after computing `basis_date` and before the snapshot hour loop:

```python
fcst_drift_fs = None
if include_forecast_features:
    fcst_daily_multi = load_historical_forecast_daily_multi(
        session=session,
        city=city_id,
        target_date=day,
        max_lead_days=6,
    )
    from models.features.forecast import compute_forecast_drift_features
    fcst_drift_fs = compute_forecast_drift_features(fcst_daily_multi)
else:
    from models.features.base import FeatureSet
    fcst_drift_fs = FeatureSet(name="forecast_drift", features={})
```

Then in `build_single_snapshot`, add another optional parameter `fcst_drift_fs` similar to `fcst_peak_fs`, and at the end:

```python
    if fcst_drift_fs is not None:
        row.update(fcst_drift_fs.to_dict())
```

### Register in `base.py`

Add to `NUMERIC_FEATURE_COLS`:

```python
"fcst_drift_num_leads",
"fcst_drift_std_f",
"fcst_drift_max_upside_f",
"fcst_drift_max_downside_f",
"fcst_drift_mean_delta_f",
"fcst_drift_slope_f_per_lead",
```

### Tests

* Unit: build a tiny DataFrame:

  ```python
  df = pd.DataFrame({
      "lead_days": [6, 3, 1],
      "tempmax_f": [84.0, 86.0, 88.0],
  })
  fs = compute_forecast_drift_features(df)
  ```

  Check that `fcst_drift_std_f > 0`, `fcst_drift_max_upside_f == 0`, etc.

* Integration: run `build_snapshot_dataset` for a few days where you know you have leads 0–6; inspect the drift columns and verify they’re constant across snapshots for the same day.

---

## 4) Forecast humidity / cloudcover / dewpoint aggregates (15‑min)

You’ve invested a lot of work to ingest minute-level forecasts with humidity, dew, cloudcover, etc.

Let’s add *simple, robust* day-level aggregates first — they don’t blow up the feature space and are useful for energy balance / heating potential.

### Loader wrapper (reusing your minute query helpers)

You already have `vc_minute_queries.py`.

In `models/data/loader.py`, add:

```python
def load_historical_forecast_15min(
    session: Session,
    city: str,
    target_date: date,
    basis_date: date,
    location_type: str = "station",
) -> pd.DataFrame:
    """
    Load T-1 15-min forecast series for a day.

    This wraps vc_minute_queries.fetch_tminus1_minute_forecast_df().
    """
    from models.data.vc_minute_queries import fetch_tminus1_minute_forecast_df

    df = fetch_tminus1_minute_forecast_df(
        session=session,
        city_code=city,
        target_date=target_date,
        location_type=location_type,
        basis_date=basis_date,
    )
    return df
```

### Feature group in `forecast.py`

Add:

```python
@register_feature_group("forecast_multivar_static")
def compute_forecast_multivar_static_features(
    minute_df: pd.DataFrame,
) -> FeatureSet:
    """
    Day-level aggregates for humidity, cloudcover, dewpoint
    from 15-min T-1 forecast.
    """
    if minute_df is None or minute_df.empty:
        return FeatureSet(name="forecast_multivar_static", features={})

    df = minute_df.copy()
    df["datetime_local"] = pd.to_datetime(df["datetime_local"])
    df["hour"] = df["datetime_local"].dt.hour

    def _stats(series: pd.Series):
        s = series.dropna()
        if s.empty:
            return None, None, None, None
        return float(s.mean()), float(s.min()), float(s.max()), float(s.max() - s.min())

    hum_mean, hum_min, hum_max, hum_range = _stats(df.get("humidity", pd.Series(dtype=float)))
    cc_mean, cc_min, cc_max, cc_range = _stats(df.get("cloudcover", pd.Series(dtype=float)))
    dew_mean, dew_min, dew_max, dew_range = _stats(df.get("dew_f", pd.Series(dtype=float)))

    # Morning vs afternoon humidity
    am = df[df["hour"].between(6, 11)]
    pm = df[df["hour"].between(12, 18)]
    am_hum_mean = float(am["humidity"].dropna().mean()) if not am.empty else None
    pm_hum_mean = float(pm["humidity"].dropna().mean()) if not pm.empty else None

    features = {
        "fcst_humidity_mean": hum_mean,
        "fcst_humidity_min": hum_min,
        "fcst_humidity_max": hum_max,
        "fcst_humidity_range": hum_range,
        "fcst_cloudcover_mean": cc_mean,
        "fcst_cloudcover_min": cc_min,
        "fcst_cloudcover_max": cc_max,
        "fcst_cloudcover_range": cc_range,
        "fcst_dewpoint_mean": dew_mean,
        "fcst_dewpoint_min": dew_min,
        "fcst_dewpoint_max": dew_max,
        "fcst_dewpoint_range": dew_range,
        "fcst_humidity_morning_mean": am_hum_mean,
        "fcst_humidity_afternoon_mean": pm_hum_mean,
    }

    return FeatureSet(name="forecast_multivar_static", features=features)
```

### Wire into snapshot builder

In `build_snapshot_dataset`:

Right after loading hourly forecasts and before the snapshot loop:

```python
fcst_15min_df = None
fcst_multivar_fs = None

if include_forecast_features:
    fcst_15min_df = load_historical_forecast_15min(
        session=session,
        city=city_id,
        target_date=day,
        basis_date=basis_date,
        location_type="station",  # or "city", depending on your choice
    )
    from models.features.forecast import compute_forecast_multivar_static_features
    fcst_multivar_fs = compute_forecast_multivar_static_features(fcst_15min_df)
else:
    from models.features.base import FeatureSet
    fcst_multivar_fs = FeatureSet(name="forecast_multivar_static", features={})
```

Pass `fcst_multivar_fs` into `build_single_snapshot` similarly to the other FeatureSets and merge:

```python
    if fcst_multivar_fs is not None:
        row.update(fcst_multivar_fs.to_dict())
```

### Register in `base.py`

Append to `NUMERIC_FEATURE_COLS`:

```python
"fcst_humidity_mean",
"fcst_humidity_min",
"fcst_humidity_max",
"fcst_humidity_range",
"fcst_cloudcover_mean",
"fcst_cloudcover_min",
"fcst_cloudcover_max",
"fcst_cloudcover_range",
"fcst_dewpoint_mean",
"fcst_dewpoint_min",
"fcst_dewpoint_max",
"fcst_dewpoint_range",
"fcst_humidity_morning_mean",
"fcst_humidity_afternoon_mean",
```

### Tests

* Unit: Build a small DataFrame with known humidity/cloudcover/dew series and check aggregates.
* Integration: Run `build_snapshot_dataset` on 7 days of Austin (where you know 15‑min forecast minute data exists) and inspect null rates and ranges for these columns.

---

## How to hand this to your coding agent

You can give them:

1. This message plus the `logical-moseying-lantern.md` plan (they already have it).
2. A short TODO list:

   1. Implement **Feature Group 1** in `forecast.py` + `base.py` and add a unit test.
   2. Implement **Feature Group 2** in `forecast.py`, wire into `snapshot_builder` (add `fcst_peak_fs` plumbing).
   3. Implement **Feature Group 3** (`load_historical_forecast_daily_multi`, `compute_forecast_drift_features`, plus wiring).
   4. Implement **Feature Group 4** (`load_historical_forecast_15min`, `compute_forecast_multivar_static_features`, plus wiring).
   5. Rebuild a small Austin snapshot dataset and run an ordinal-model training script to confirm everything plugs in.

If you want, once the agent has pushed code for step 1 or 2, you can paste those diffs here and I’ll sanity‑check the feature definitions and whether they match what you’ll have available live.
