### New Modeling Plan   

Here’s a markdown spec you can drop straight into the repo as
`docs/MARKET_CLOCK_TOD_V1_PLAN.md` (or whatever name you like).

---

# Market-Clock TOD v1 – All-Cities Global Model (Daily High)

**Date:** 2025-11-30
**Owner:** Stephen / Kalshi Weather
**Status:** Design approved – implementation pending

---

## 0. Summary

We are adding a **new ordinal CatBoost model** that:

* Predicts **daily highest temperature** for the existing **6 cities** (Chicago, Austin, Denver, LA, Miami, Philly).
* Uses **market-clock time** instead of calendar-day-only time:

  * Time axis is **“minutes since market open”**, where market open is **10:00 local time on D-1** for the event date D.
  * The model is valid from **10:00 D-1** through **market close** (trading ends ~11:59pm local on D).
* Still uses **TMAX for calendar day D** as the label (NWS/CLI high).
* Uses **both**:

  * Physical time-of-day features (hour/minute, calendar encodings), and
  * Market-clock features (minutes/hours since market open, D-1 vs D flags).
* Trains **one global model across all six cities**, using **city as one‑hot numeric features** (no CatBoost categorical features).
* Runs **CatBoost on CPU only**, using **26 threads** for training and inference (`task_type="CPU"`, `thread_count=26`).
* Uses **all daily high markets** for those six cities (no low or other weather markets).

Existing models remain:

* **Baseline / hourly / TOD v1** stay as-is and can be used as baselines and fallbacks.
* TOD v1 is still “same-day partial-day” (10:00–23:45 on D), per-city, without D‑1 context.

This document is the **single source of truth** for implementing **Market-Clock TOD v1**.

---

## 1. Historical Context (What Exists Today)

### 1.1. Models

**Hourly & baseline models**

* Scripts:

  * `scripts/train_all_cities_hourly.py`
  * `scripts/train_all_cities_ordinal.py`
  * `scripts/train_chicago_30min.py`

* Snapshot builder:

  * `models/data/snapshot_builder.py` (hourly snapshots).
* Per-city CatBoost models, `MODEL_VARIANTS["hourly"]` / `"baseline"` in `config/live_trader_config.py`. 

**TOD v1 (same-day, per-city)**

* Script:

  * `scripts/train_tod_v1_all_cities.py`. 
* Dataset builder:

  * `models/data/tod_dataset_builder.py` – builds 15-min snapshots from ~10:00–23:45 on the **event day D only**.
* Features (per snapshot):

  * `models/features/partial_day.py` – partial-day stats on D.
  * `models/features/shape.py` – curve shape/plateau metrics.
  * `models/features/rules.py` – rule meta-features.
  * `models/features/forecast.py` – T‑1 forecast features.
  * `models/features/calendar.py` – time-of-day + calendar encodings.
  * `models/features/quality.py` – missing_fraction, gaps, edge flags.
  * `models/features/base.py` – feature lists, `DELTA_CLASSES`. 
* Label:

  * `delta = settle_f - t_base`, clipped to `DELTA_CLASSES = [-2, …, +10]` (via `compute_delta_target` in `partial_day.py` / `base.py`).
* Integration:

  * `MODEL_VARIANTS["tod_v1"]` in `config/live_trader_config.py`.
  * `LiveInferenceEngine` in `models/inference/live_engine.py` builds TOD snapshots with `cutoff_time`, rounds to 15-min intervals, and calls the per-city models. 

### 1.2. Data & infra

* Observations: `wx.vc_minute_weather` (5-min actual obs).
* Forecasts: `wx.vc_forecast_daily`, `wx.vc_forecast_hourly` (basis-date keyed).
* Settlements: `wx.settlement` (NWS daily TMAX per city).
* Market metadata: `kalshi.markets`, tickers parsed via `parse_event_ticker` etc. 
* Metrics: `models/evaluation/metrics.py` – `compute_all_metrics`, `compute_delta_metrics`, etc. 
* Training uses **CatBoost on CPU** already (no GPU config in training scripts). 

TOD v1 **does not** use D‑1 observations or market-clock timing; it only sees partial-day progression on D.

---

## 2. Design for Market-Clock TOD v1

### 2.1. Semantics

**Event / label**

* Event: Daily **maximum temperature** for calendar day D in the given city.
* Label: `settle_f` = official CLI high (from `wx.settlement` for D).
* Delta: `delta = settle_f - t_base`, clipped to `DELTA_CLASSES` (same helper as TOD v1).

**Market-clock time**

* Market open time for a given daily high market (Chicago example):
  `market_open_time = D-1 at 10:00 local time` (Kalshi weather markets are launched at 10 AM the previous day; individual market pages show “Market open Nov 29 10:00am EST” for the 30th, etc.).
* Trading ends at ~11:59pm local on D (last trading time).

**Time domain for modeling**

* We want the model to be valid from:

  * `D-1 10:00` (market open)
  * through `D 23:59` (or effectively `D 23:55` at 5-min granularity)

**Snapshots**

* Training snapshots will be sampled every **5 minutes** (natural obs cadence) by default:

  * From `market_open_time` (D-1 10:00) to `D 23:55`.
  * ~38 hours × 12 per hour ≈ 456 snapshots per event.
* Inference can run **every minute**:

  * Features use all obs up to the current time;
  * Time features (minutes since open) are continuous numerical inputs.
  * We **do not** need to train a separate model for every possible minute.

### 2.2. Features

At each snapshot time `τ`:

**A. Market-clock features (new)**

* `market_open_time` = `datetime(event_date - 1, 10:00 local)` (or later: read from `kalshi.markets` if available).
* `minutes_since_market_open = max(0, floor((τ - market_open_time) / 60s))`
* `hours_since_market_open = minutes_since_market_open / 60.0`
* `is_d_minus_1 = 1 if τ.date() == D-1 else 0`
* `is_event_day = 1 - is_d_minus_1`

Optional later extensions (not required now, but structure for them):

* `minutes_into_event_day` (0–1440 if τ.date() == D else 0)
* `minutes_since_midnight_dminus1` etc.

**B. Physical time-of-day & calendar features (existing)**

* Reuse `compute_calendar_features(day=event_date, cutoff_time=τ)` from `models/features/calendar.py`:

  * `hour`, `minute`, `minutes_since_midnight`, `hour_sin`, `minute_sin`,
  * `time_of_day_sin`, `time_of_day_cos`,
  * `doy_sin`, `doy_cos`, `week_sin`, `week_cos`, `month`, `is_weekend`. 

Note: `day` is always D (event date), while `cutoff_time` may be D-1 or D; time-of-day features use `cutoff_time`’s hour/minute; day-of-year features are for D (the event day).

**C. Partial-day / shape / forecast / rules (mostly reuse)**

We still compute them using **all obs up to τ**, but now obs span **D-1 10:00 → τ**, not just same-day.

* Partial-day features (`partial_day.py`):

  * `vc_max_f_sofar`, `vc_mean_f_sofar`, percentiles, `t_base = round(vc_max_f_sofar)`, etc. 
* Shape features (`shape.py`):

  * Plateau lengths, slopes, morning/afternoon/evening max, etc.
* Rules (`rules.py`):

  * Rule-based predictions and disagreement flags.
* Forecast features (`forecast.py`):

  * T‑1 daily & hourly forecast for D at basis date D‑1.
  * In v1, we only require T‑1; T‑2 and multiple bases can be a later extension.
* Quality features (`quality.py`), adapted for market-clock:

  * `missing_fraction_since_open`,
  * `max_gap_minutes`,
  * `edge_max_flag`, etc.

**D. City / market identity – one-hot**

We want **one global model** and **city as one-hot numeric features**:

For each snapshot row, we add:

* `city_chicago`, `city_austin`, `city_denver`, `city_los_angeles`, `city_miami`, `city_philadelphia` ∈ {0,1}
* Exactly one is 1.

We can optionally keep `city` string in the dataset (for grouping and debugging), but the model sees only the one-hot columns.

**E. Target**

* `settle_f` (float)
* `t_base` from partial-day features
* `delta = settle_f - t_base`, clipped to `DELTA_CLASSES` (same as TOD v1).

We continue to train **ordinal-ish multiclass** on `delta` class index.

### 2.3. CPU CatBoost setup

* **Processing unit:** CPU only.
* Parameters:

  * `task_type="CPU"`
  * `thread_count=26` (explicitly set to use 26 of 32 cores)
* Otherwise similar to TOD v1:

  * `loss_function="MultiClass"`,
  * `eval_metric="MultiClass"`,
  * `depth` ~6–10,
  * `iterations` ~500–1500,
  * `learning_rate` ~0.03–0.1,
  * `l2_leaf_reg` ~3–10,
  * `border_count` ~128, etc.

Optuna tuning can be reused later; initial v1 can use fixed parameters.

---

## 3. Implementation Checklist

### 3.1. Repo & invariants

* [ ] Branch: `feature/market_clock_tod_v1`.
* [ ] Confirm existing modules compile:

  * `models/data/tod_dataset_builder.py`
  * `models/data/snapshot_builder.py`
  * `models/features/base.py`, `partial_day.py`, `shape.py`, `rules.py`, `forecast.py`, `calendar.py`, `quality.py`
  * `models/data/loader.py`
  * `models/data/splits.py`
  * `models/training/ordinal_trainer.py`
  * `models/inference/live_engine.py`
  * `models/evaluation/metrics.py`
  * `config/live_trader_config.py`
* [ ] Keep TOD v1 training and inference untouched.

### 3.2. New dataset builder – Market-Clock snapshots

**New file:** `models/data/market_clock_dataset_builder.py`

**Function:**

```python
def build_market_clock_snapshot_dataset(
    cities: list[str],
    start_date: date,
    end_date: date,
    session: Session,
    snapshot_interval_min: int = 5,
    market_open_hour: int = 10,
) -> pd.DataFrame:
    ...
```

**Behavior:**

1. Loop over each `city` in `cities`.
2. For each **event_date = D** from `start_date` to `end_date` (inclusive):

   * Define:

     * `market_open = datetime(D.year, D.month, D.day, market_open_hour, 0) - timedelta(days=1)`.
     * `market_close = datetime(D.year, D.month, D.day, 23, 59)` (approx; 23:55 at 5-min granularity).
   * Load **settlement** for D via `load_settlements(session, city_id, D, D)`.
   * Load **observations** for D‑1 and D:

     * `load_vc_observations(session, city_id, D-1, D)` → `obs_df`.
   * Load **T‑1 forecasts** (daily + hourly) for D:

     * `load_historical_forecast_daily(session, city_id, target_date=D, basis_date=D-1)`.
     * `load_historical_forecast_hourly(session, city_id, target_date=D, basis_date=D-1)`.
3. Generate snapshot times `τ` from `market_open` to `market_close` every `snapshot_interval_min` minutes:

   * Use a helper `_generate_snapshot_times(market_open, market_close, snapshot_interval_min)`.
4. For each `τ`:

   * Filter `obs_sofar = obs_df[obs_df["datetime_local"] <= τ]`.

     * If `obs_sofar` is empty, skip this snapshot (or fill with NaNs and mark quality as bad; v1 can just skip).
   * Build `temps_sofar`, `timestamps_sofar` lists.
   * Compute features via a **new helper** `build_market_clock_snapshot_for_training(...)` (see §4.1).
5. Collect all snapshot rows into a list, then `pd.DataFrame`.

**Columns expected in resulting DataFrame:**

* Identity:

  * `city` (string id),
  * `event_date` (date for D),
  * `snapshot_datetime` (τ).
* Market-clock:

  * `minutes_since_market_open`,
  * `hours_since_market_open`,
  * `is_d_minus_1`,
  * `is_event_day`.
* One-hot city:

  * `city_chicago`, `city_austin`, `city_denver`, `city_los_angeles`, `city_miami`, `city_philadelphia`.
* Partial-day / shape / rules / forecast / calendar / quality features (same names as TOD v1, plus any new ones for market-clock).
* Label:

  * `settle_f`, `t_base`, `delta` (int).
* Optionally:

  * `tmax_cli_f`, `tmax_iem_f` for debug.

**Output location for training dataset:**

* `data/market_clock_tod_v1/train_data.parquet`.

### 3.3. New snapshot helper – training-time features

**Location:** `models/data/market_clock_dataset_builder.py`

**Function:**

```python
def build_market_clock_snapshot_for_training(
    city: str,
    event_date: date,
    cutoff_time: datetime,
    temps_sofar: list[float],
    timestamps_sofar: list[datetime],
    fcst_daily: Optional[dict],
    fcst_hourly_df: Optional[pd.DataFrame],
    settle_f: float,
    market_open: datetime,
) -> dict:
    ...
```

**Steps inside:**

1. Compute **market-clock features**:

   ```python
   delta_minutes = max(
       0,
       int((cutoff_time - market_open).total_seconds() // 60),
   )
   minutes_since_market_open = delta_minutes
   hours_since_market_open = minutes_since_market_open / 60.0
   is_d_minus_1 = int(cutoff_time.date() == (event_date - timedelta(days=1)))
   is_event_day = 1 - is_d_minus_1
   ```

2. Call **partial-day features**:

   ```python
   from models.features.partial_day import compute_partial_day_features

   pd_fs = compute_partial_day_features(temps_sofar)
   t_base = pd_fs.t_base  # or pd_fs["t_base"]
   ```

3. Call **shape features**:

   ```python
   from models.features.shape import compute_shape_features

   shape_fs = compute_shape_features(
       temps_sofar=temps_sofar,
       timestamps_sofar=timestamps_sofar,
   )
   ```

4. Call **rule features**:

   ```python
   from models.features.rules import compute_rule_features

   rule_fs = compute_rule_features(
       temps_sofar=temps_sofar,
       timestamps_sofar=timestamps_sofar,
       fcst_daily=fcst_daily,
       fcst_hourly_df=fcst_hourly_df,
   )
   ```

5. Call **forecast features**:

   ```python
   from models.features.forecast import (
       compute_forecast_static_features,
       compute_forecast_error_features,
       compute_forecast_delta_features,
   )

   fcst_static_fs = compute_forecast_static_features(fcst_daily, fcst_hourly_df)
   fcst_error_fs = compute_forecast_error_features(
       temps_sofar=temps_sofar,
       timestamps_sofar=timestamps_sofar,
       fcst_hourly_df=fcst_hourly_df,
   )
   fcst_delta_fs = compute_forecast_delta_features(
       settle_f=settle_f,
       fcst_daily=fcst_daily,
   )
   ```

6. Call **calendar features** (physical time-of-day):

   ```python
   from models.features.calendar import compute_calendar_features

   calendar_fs = compute_calendar_features(
       day=event_date,
       cutoff_time=cutoff_time,
   )
   ```

7. Call **quality features** (market-clock-aware):

   Add a helper in `models/features/quality.py` (see §3.4) or directly calculate:

   ```python
   from models.features.quality import compute_quality_features_market_clock

   quality_fs = compute_quality_features_market_clock(
       timestamps_sofar=timestamps_sofar,
       market_open=market_open,
       step_minutes=5,
   )
   ```

8. Compute **delta**:

   ```python
   from models.features.base import compute_delta_target

   delta_info = compute_delta_target(settle_f=settle_f, vc_max_f_sofar=pd_fs.vc_max_f_sofar)
   # delta_info should include "delta" and potentially "delta_class"
   ```

9. Build **city one-hot**:

   ```python
   def city_one_hot(city: str) -> dict[str, int]:
       return {
           "city_chicago": int(city == "chicago"),
           "city_austin": int(city == "austin"),
           "city_denver": int(city == "denver"),
           "city_los_angeles": int(city == "los_angeles"),
           "city_miami": int(city == "miami"),
           "city_philadelphia": int(city == "philadelphia"),
       }
   ```

10. Combine into a single `dict` row:

    ```python
    row: dict[str, Any] = {
        "city": city,
        "event_date": event_date,
        "snapshot_datetime": cutoff_time,
        "minutes_since_market_open": minutes_since_market_open,
        "hours_since_market_open": hours_since_market_open,
        "is_d_minus_1": is_d_minus_1,
        "is_event_day": is_event_day,
        "settle_f": settle_f,
        "t_base": t_base,
        **city_one_hot(city),
        **pd_fs.to_dict(),
        **shape_fs.to_dict(),
        **rule_fs.to_dict(),
        **fcst_static_fs.to_dict(),
        **fcst_error_fs.to_dict(),
        **fcst_delta_fs.to_dict(),
        **calendar_fs.to_dict(),
        **quality_fs.to_dict(),
        **delta_info,  # includes "delta"
    }
    ```

### 3.4. Quality features adapted for market-clock

**File to modify:** `models/features/quality.py`

Add:

```python
def compute_quality_features_market_clock(
    timestamps_sofar: list[datetime],
    market_open: datetime,
    step_minutes: int = 5,
) -> FeatureSet:
    """
    Compute data quality features using market-clock semantics:
    expected_samples ~ minutes_since_market_open / step_minutes.
    """
    features: dict[str, float] = {}

    n = len(timestamps_sofar)
    if n == 0:
        return FeatureSet(
            missing_fraction_sofar=1.0,
            max_gap_minutes=24 * 60,
            edge_max_flag=0,
        )

    # Expected samples since market open
    last_ts = max(timestamps_sofar)
    minutes_since_open = max(
        0,
        int((last_ts - market_open).total_seconds() // 60),
    )
    expected_samples = max(1, minutes_since_open // step_minutes)

    # Missing fraction (clamped to [0, 1])
    mf = 1.0 - (n / expected_samples)
    missing_fraction_sofar = max(0.0, min(1.0, mf))

    # Max gap
    timestamps_sorted = sorted(timestamps_sofar)
    gaps = [
        (t2 - t1).total_seconds() / 60.0
        for t1, t2 in zip(timestamps_sorted[:-1], timestamps_sorted[1:])
    ]
    max_gap_minutes = max(gaps) if gaps else 0.0

    # Edge max flag: max temp at first or last 10% of window will be computed at partial-day layer,
    # but we can still implement if needed. For now, re-use existing logic via a helper if possible.

    # For v1: set edge_max_flag to 0; implement later if needed.
    edge_max_flag = 0

    return FeatureSet(
        missing_fraction_sofar=missing_fraction_sofar,
        max_gap_minutes=max_gap_minutes,
        edge_max_flag=edge_max_flag,
    )
```

Do **not** remove or break the existing `compute_quality_features` used by TOD v1 and hourly models; this is a separate helper for the market-clock path.

### 3.5. Feature column selection for market-clock model

**File to update:** `models/features/base.py`

Add:

```python
def get_feature_columns_for_market_clock(
    include_forecast: bool = True,
    include_lags: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Return numeric and categorical feature columns for the market-clock TOD model.
    City is encoded via one-hot in numeric features (no categorical features).
    """
    num_cols, cat_cols = get_feature_columns(
        include_forecast=include_forecast,
        include_lags=include_lags,
    )

    # Remove city categorical (we'll use one-hot instead)
    cat_cols = [c for c in cat_cols if c != "city"]

    market_clock_specific = [
        "minutes_since_market_open",
        "hours_since_market_open",
        "is_d_minus_1",
        "is_event_day",
        "city_chicago",
        "city_austin",
        "city_denver",
        "city_los_angeles",
        "city_miami",
        "city_philadelphia",
    ]

    # Deduplicate
    num_cols = list(dict.fromkeys(num_cols + market_clock_specific))

    return num_cols, cat_cols
```

TOD v1 and hourly models continue to use `get_feature_columns` unchanged.

### 3.6. Training script – `train_market_clock_tod_v1.py`

**New file:** `scripts/train_market_clock_tod_v1.py`

**Responsibilities:**

1. Load dataset from `data/market_clock_tod_v1/train_data.parquet`.
2. Prepare features & labels.
3. Use day-grouped CV across all cities.
4. Train **one CatBoost model** on CPU with 26 threads.
5. Save model and metadata to `models/saved/market_clock_tod_v1/`.

**Outline:**

```python
#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime

import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from models.features.base import (
    get_feature_columns_for_market_clock,
    DELTA_CLASSES,
)
from models.data.splits import DayGroupedTimeSeriesSplit

DATA_PATH = Path("data/market_clock_tod_v1/train_data.parquet")
MODEL_DIR = Path("models/saved/market_clock_tod_v1")
MODEL_PATH = MODEL_DIR / "ordinal_catboost_market_clock_tod_v1.pkl"
METADATA_PATH = MODEL_DIR / "training_metadata.json"


def prepare_data() -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    df = pd.read_parquet(DATA_PATH)

    # event_date used for grouping
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date

    num_cols, cat_cols = get_feature_columns_for_market_clock(
        include_forecast=True,
        include_lags=True,
    )
    feature_cols = num_cols + cat_cols

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataset: {missing}")

    X = df[feature_cols]

    delta_to_idx = {d: i for i, d in enumerate(DELTA_CLASSES)}
    y = df["delta"].map(delta_to_idx).astype(int).values

    return df, y, feature_cols, cat_cols


def get_cv_splits(df: pd.DataFrame):
    cv = DayGroupedTimeSeriesSplit(n_splits=5, gap_days=0)
    # Group by event_date (all cities pooled)
    for train_idx, val_idx in cv.split(df, group_key="event_date"):
        yield train_idx, val_idx


def train() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df, y, feature_cols, cat_cols = prepare_data()
    cat_feature_indices: list[int] = []  # No categorical features in this variant

    params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "task_type": "CPU",
        "thread_count": 26,
        "depth": 8,
        "learning_rate": 0.05,
        "iterations": 1000,
        "l2_leaf_reg": 5.0,
        "border_count": 128,
        "random_strength": 1.0,
        "early_stopping_rounds": 100,
        "verbose": 100,
    }

    model = CatBoostClassifier(**params)

    splits = list(get_cv_splits(df))
    train_idx, val_idx = splits[-1]

    X = df[feature_cols]
    train_pool = Pool(
        data=X.iloc[train_idx],
        label=y[train_idx],
        cat_features=cat_feature_indices,
    )
    val_pool = Pool(
        data=X.iloc[val_idx],
        label=y[val_idx],
        cat_features=cat_feature_indices,
    )

    model.fit(train_pool, eval_set=val_pool)

    model.save_model(str(MODEL_PATH))

    # Basic metrics
    val_pred = model.predict(val_pool).flatten().astype(int)
    idx_to_delta = {i: d for i, d in enumerate(DELTA_CLASSES)}
    y_val_delta = np.vectorize(idx_to_delta.get)(y[val_idx])
    y_pred_delta = np.vectorize(idx_to_delta.get)(val_pred)

    delta_acc = float((y_val_delta == y_pred_delta).mean())
    delta_mae = float(np.abs(y_val_delta - y_pred_delta).mean())

    metadata = {
        "model_variant": "market_clock_tod_v1",
        "created_at": datetime.utcnow().isoformat(),
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "delta_classes": DELTA_CLASSES,
        "params": params,
        "validation_metrics": {
            "delta_accuracy": delta_acc,
            "delta_mae": delta_mae,
        },
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Validation Δ-accuracy: {delta_acc:.3f}")
    print(f"Validation Δ-MAE: {delta_mae:.3f}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    train()
```

### 3.7. Config & inference integration

**File to update:** `config/live_trader_config.py`

Add new variant:

```python
MODEL_VARIANTS = {
    "baseline": {...},
    "hourly": {...},
    "tod_v1": {...},
    "market_clock_tod_v1": {
        "folder_suffix": "_market_clock_tod_v1",
        "filename": "ordinal_catboost_market_clock_tod_v1.pkl",
        "snapshot_hours": None,  # No hour-snapping; use exact cutoff_time
        "requires_snapping": False,
        "snapshot_interval_min": 5,  # Training spacing; inference can be every minute
    },
}
```

Allow switching:

```python
ORDINAL_MODEL_VARIANT = "market_clock_tod_v1"
```

**File to update:** `models/inference/live_engine.py`

Add:

1. **Helper** to compute market open:

   ```python
   from datetime import datetime, time, timedelta

   def get_market_open_time(city: str, event_date: date) -> datetime:
       # For now: 10:00 local on D-1.
       # Later: can be replaced with DB/WebSocket-driven open time per ticker.
       return datetime.combine(event_date - timedelta(days=1), time(10, 0))
   ```

2. **When predicting**, for the `"market_clock_tod_v1"` variant:

   * Determine `market_open = get_market_open_time(city, event_date)`.
   * Use **actual current time** as `cutoff_time` (no snapping to 5 or 15):

     * The model sees continuous `minutes_since_market_open`.
     * Last obs in `temps_sofar` will still only update every 5 minutes, which is fine.

   Example integration sketch:

   ```python
   def predict(self, city, event_date, session, current_time=None, force_refresh=False):
       ...
       model_variant = self.model_variant  # from config

       if current_time is None:
           current_time = datetime.now(tz=LOCAL_TZ)  # ensure local tz

       if model_variant == "market_clock_tod_v1":
           market_open = get_market_open_time(city, event_date)
           cutoff_time = current_time
           snapshot_hour = None  # unused

           # Load obs and T-1 forecasts for both D-1 and D
           data = load_inference_data_market_clock(
               city_id=city,
               event_date=event_date,
               cutoff_time=cutoff_time,
               market_open=market_open,
               session=session,
           )
           temps_sofar = data["temps_sofar"]
           timestamps_sofar = data["timestamps_sofar"]
           fcst_daily = data["fcst_daily"]
           fcst_hourly_df = data["fcst_hourly_df"]

           features = build_market_clock_snapshot_for_inference(
               city=city,
               event_date=event_date,
               cutoff_time=cutoff_time,
               temps_sofar=temps_sofar,
               timestamps_sofar=timestamps_sofar,
               fcst_daily=fcst_daily,
               fcst_hourly_df=fcst_hourly_df,
               market_open=market_open,
           )

           # Convert features dict to DataFrame row
           X = pd.DataFrame([features])

           model = self.models["market_clock_tod_v1"]
           delta_probs = model.predict_proba(X)[0]
           ...
       else:
           # Existing baseline/hourly/tod_v1 paths remain unchanged
           ...
   ```

3. **New helper `build_market_clock_snapshot_for_inference`** should mirror the training helper, but without label and with only features. It can live either in `models/data/market_clock_dataset_builder.py` or in `models/data/snapshot_builder.py`. Use the **same feature logic** as training to avoid train–test skew.

### 3.8. Inference data loader for market-clock

**File to update or extend:** `models/data/loader.py`

Add:

```python
def load_inference_data_market_clock(
    city_id: str,
    event_date: date,
    cutoff_time: datetime,
    market_open: datetime,
    session: Session,
) -> dict:
    """
    Load observations and forecasts for market-clock inference.
    Observations: from D-1 10:00 up to cutoff_time.
    Forecasts: T-1 daily/hourly for event_date.
    """
    # Determine date range for observations
    obs_start_date = (event_date - timedelta(days=1))
    obs_end_date = event_date

    obs_df = load_vc_observations(session, city_id, obs_start_date, obs_end_date)
    obs_df = obs_df.sort_values("datetime_local")
    obs_df = obs_df[obs_df["datetime_local"] >= market_open]
    obs_sofar = obs_df[obs_df["datetime_local"] <= cutoff_time]

    temps_sofar = obs_sofar["temp_f"].tolist()
    timestamps_sofar = obs_sofar["datetime_local"].tolist()

    fcst_daily = load_historical_forecast_daily(
        session=session,
        city_id=city_id,
        target_date=event_date,
        basis_date=event_date - timedelta(days=1),
    )
    fcst_hourly_df = load_historical_forecast_hourly(
        session=session,
        city_id=city_id,
        target_date=event_date,
        basis_date=event_date - timedelta(days=1),
    )

    return {
        "temps_sofar": temps_sofar,
        "timestamps_sofar": timestamps_sofar,
        "fcst_daily": fcst_daily,
        "fcst_hourly_df": fcst_hourly_df,
    }
```

TOD v1’s `load_inference_data` remains unchanged.

### 3.9. Testing & evaluation

**New script:** `scripts/test_market_clock_tod_v1_inference.py`

* [ ] Load `LiveInferenceEngine` with `ORDINAL_MODEL_VARIANT = "market_clock_tod_v1"`.
* [ ] For each city, pick a handful of historical event dates:

  * A hot summer day,
  * A cold winter day,
  * A borderline/noisy transition day.
* [ ] For each (city, D), call `predict()` at:

  * `D-1 10:01`,
  * `D-1 14:00`,
  * `D-1 20:00`,
  * `D 06:00`,
  * `D 12:00`,
  * `D 18:00`,
  * `D 23:30`.
* [ ] Assert:

  * No exceptions.
  * `delta_probs` sum ~1 (within small tolerance).
  * `delta_hat` ∈ `DELTA_CLASSES`.
  * `minutes_since_market_open` ≥ 0 and behaves as expected.

**Training evaluation:**

* [ ] Use `models/evaluation/metrics.py` to compute:

  * Δ accuracy, Δ MAE, within-1, within-2 rates.
  * Settlement accuracy / MAE (reconstruct temp via `t_base + delta_hat`).
  * Calibration metrics (ECE).
* [ ] Compare:

  * TOD v1 per-city models vs Market-Clock TOD v1 global on the **same backtest period**:

    * Use last N days of the dataset, or a fixed date range.
* [ ] Write a small report to `docs/MARKET_CLOCK_TOD_V1_RESULTS.md`.

---

## 4. Notes on Sampling Interval vs Continuous Time

* We **train** on discrete snapshots (default **5-minute** interval) for each market and city.
* The model sees **continuous “minutes_since_market_open”** as a numeric feature.
* At **inference**, we can:

  * Run predictions every **minute**,
  * Build features using all obs up to that minute,
  * Fill `minutes_since_market_open` as the exact minute difference.

The CatBoost model doesn’t need separate training for “every minute”; the continuous numeric feature plus dense 5-min training snapshots give it enough information to interpolate across intermediate minutes.

---

