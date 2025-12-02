Here’s how I’d fold **“yesterday’s forecast” + forecast-vs-actual deltas** into the intraday models, with all the no-lookahead constraints baked in.

I’ll give you:

* A quick confirmation of **what Visual Crossing can do** (hourly / sub-hourly, historical forecast).
* A **feature block** you can bolt onto your existing intraday feature builder.
* How this changes the **Model 1 (logistic)** and **Model 2 (CatBoost+Optuna)** setups (mostly: add columns, same training protocol).

I’ll keep this in one markdown file you can drop into the repo as something like:

> `FORECAST_FEATURES_EXTENSION.md`

---

````markdown
# FORECAST_FEATURES_EXTENSION.md
Using T-1 Visual Crossing Forecasts + Forecast vs Actual Deltas

This document describes how to extend the intraday Δ-models to use:

1. **The forecast issued the day before** (T-1) for the current day, from Visual Crossing.
2. **Live forecast error features**: the delta between that forecast and the actual Visual Crossing observations coming in during the day.

We do this without **any lookahead**:

- Training: we reconstruct the forecast that was *actually issued* on the previous day (T-1) using Visual Crossing’s **Historical Forecast** product.   
- Intraday: at any time τ on day D, we only use:
  - The forecast generated on basis date ≤ D-1.
  - Observations from day D with `datetime_local ≤ τ`.
  - No data from later in the day.

Everything plugs into the **same two models** as before:

- **Model 1** – Multinomial logistic Δ-model (elastic net + Platt).   
- **Model 2** – CatBoost Δ-model (CatBoostClassifier + Optuna + Platt).   

---

## 1. What Visual Crossing Can Provide (Forecast Granularity & Historical Forecasts)

### 1.1 Forecast granularity

From Visual Crossing docs:

- The **Timeline Weather API** can return:
  - **Daily & hourly** forecasts for up to 15 days.   
  - **Sub-hourly “minute-level” data** when you include `minutes` and a `minuteinterval` option (e.g., 15-minute intervals).   

- For **sub-hourly forecast**:
  - Minimum time interval is **15 minutes**.
  - Historical data can go down to **5–10 minutes** depending on station, but **forecast** is *not* interpolated below 15 minutes.   

So:

> You can absolutely get **hourly** and **15-minute** forecasts for a given day from Visual Crossing.  
> For forecast, you **cannot** get true 5-minute data; that resolution is only for historical obs in some cases.

For your use case, hourly is already enough; 15-minute forecasts are a nice bonus for short-range (0–24h) prediction.

### 1.2 Historical forecasts (what people actually saw yesterday)

Visual Crossing explicitly supports **historical forecast data** via the Timeline API and “basis date”:   

- You can query:
  - A **target date** D (the day you care about).
  - A **basis date** B (e.g. D-1), which is the day the forecast was *issued*.
- The API returns **what the 15-day forecast looked like on basis date B**, including hourly (and, in some cases, sub-hourly) forecasts for D.

That’s exactly what you want for “what did Visual Crossing think yesterday about today’s high?”

---

## 2. New Feature Block: T-1 Forecast & Forecast vs Actual Deltas

We’ll extend the **intraday snapshot dataset** (per `(city, day, snapshot_hour)`) with a new group of features.

At a high level, for each `(city, day)`:

- Use VC Historical Forecast to get **forecast series** for day D, as issued on T-1.
  - Call it `fcst_temp_tminus1[h]` for each hour (or 15-minute slot) of day D.
- On day D, as actual VC obs come in:
  - `obs_temp[h]` up to current time τ.
- Labels:
  - `T_settle` = NWS/CLI daily high.
  - Δ-target as before: `delta(τ) = T_settle − t_base(τ)`.

### 2.1 Forecast-only features (known before the day starts)

These exist **even at 00:00** on D, since they’re fully determined by the `basis_date = D-1` forecast:

Let `fcst_series` = all hourly (or 15-min) temps for D from the T-1 forecast.

Examples:

```python
import numpy as np

def forecast_only_features(fcst_series: list[float]) -> dict:
    if not fcst_series:
        return {}
    arr = np.asarray(fcst_series, dtype=float)
    q10, q25, q50, q75, q90 = np.percentile(arr, [10, 25, 50, 75, 90])

    max_f = float(arr.max())
    idx_max = int(arr.argmax())

    return {
        "fcst_prev_max_f": max_f,
        "fcst_prev_min_f": float(arr.min()),
        "fcst_prev_mean_f": float(arr.mean()),
        "fcst_prev_std_f": float(arr.std(ddof=1)),
        "fcst_prev_q10_f": float(q10),
        "fcst_prev_q25_f": float(q25),
        "fcst_prev_q50_f": float(q50),
        "fcst_prev_q75_f": float(q75),
        "fcst_prev_q90_f": float(q90),
        "fcst_prev_frac_part": max_f - round(max_f),
        "fcst_prev_hour_of_max": idx_max,  # assuming hourly forecast starts at hour 0
    }
````

You can also define a **forecast-only baseline**:

```python
def forecast_baseline(max_f: float) -> dict:
    t_forecast_base = int(round(max_f))
    return {
        "t_forecast_base": t_forecast_base,
    }
```

Later, the model will see both `t_base(τ)` (from actual VC obs so far) and `t_forecast_base` (from T-1 forecast) and can learn which is more reliable at different times of day / weather regimes.

### 2.2 Intraday forecast vs actual deltas (using obs **up to τ**)

For a snapshot at local time τ on day D:

* Let `fcst_series_so_far` = all forecast temps for hours (or minutes) ≤ τ.
* Let `obs_series_so_far`  = all actual VC temps for times ≤ τ.

We align them by time index (e.g., hourly):

```python
def forecast_error_features(fcst_series_so_far: list[float],
                            obs_series_so_far: list[float]) -> dict:
    import numpy as np
    n = min(len(fcst_series_so_far), len(obs_series_so_far))
    if n == 0:
        return {}

    fc = np.asarray(fcst_series_so_far[:n], dtype=float)
    ob = np.asarray(obs_series_so_far[:n], dtype=float)

    err = ob - fc  # positive = actual warmer than forecast

    return {
        "err_mean_so_far": float(err.mean()),
        "err_std_so_far": float(err.std(ddof=1)),
        "err_max_pos_so_far": float(err.max()),
        "err_max_neg_so_far": float(err.min()),
        "err_abs_mean_so_far": float(np.abs(err).mean()),
        # last few hours error trend (if enough samples)
        "err_last1h": float(err[-1]) if n >= 1 else None,
        "err_last3h_mean": float(err[-3:].mean()) if n >= 3 else None,
    }
```

These features answer questions like:

> “Is today trending consistently hotter than yesterday’s forecast?”
> “Is the forecast over- or under-shooting, and by how much, as of τ?”

This is *extremely* informative near the end of the day when the difference between VC forecast and actual obs indicates bias.

### 2.3 Combining forecast & obs baselines

You now have three “baseline” notions:

1. `t_forecast_base` – from T-1 forecast max.
2. `t_base(τ)` – from **actual VC observations up to τ** (our existing baseline).
3. Optionally: `t_forecast_max_so_far(τ)` – max of forecast up to τ (useful if the forecast expects a late-evening spike you haven’t seen yet).

You can expose all three; the model can learn relationships like:

* Early in the day, `t_forecast_base` is more informative.
* By 20:00, `t_base(τ)` + forecast error stats dominate.

You still keep Δ-target defined relative to one of them (I’d keep it as before: `Δ = T_settle − t_base(τ)`), and let the model use forecast information as features.

### 2.4 Updated intraday feature row

For each `(city, day, snapshot_hour)` you now have:

* **Label**: `delta(τ)` as before.
* **Existing features**: partial-day VC stats, shape, rules, snapshot hour, lags, etc.
* **New forecast features**:

  * From T-1 forecast:

    * `fcst_prev_max_f`, `fcst_prev_min_f`, `fcst_prev_mean_f`, `fcst_prev_std_f`, `fcst_prev_qXX_f`, `fcst_prev_frac_part`, `fcst_prev_hour_of_max`, `t_forecast_base`.
  * From forecast vs actual errors (up to τ):

    * `err_mean_so_far`, `err_std_so_far`, `err_max_pos_so_far`, `err_max_neg_so_far`, `err_abs_mean_so_far`, `err_last1h`, `err_last3h_mean`.

---

## 3. Changes to the Modeling Setup

The **model structures don’t need to change**. We are just giving them more information:

* For **Model 1 (logistic Δ-model)**:

  * Extend the `num_cols` list in `build_X_y_intraday` to include the forecast features.
* For **Model 2 (CatBoost Δ-model)**:

  * Extend `feature_cols` similarly; CatBoost will happily consume them.

Everything else stays:

* Same **TimeSeriesSplit** for CV (snapshot rows sorted by day+hour).
* Same **CalibratedClassifierCV(method='sigmoid')** for Platt scaling on top of each base classifier.
* Same **train/test split by day** (no future days in training).

### 3.1 Where to plug forecast features into Model 1 (logistic)

In `build_X_y_intraday(df)` from `intraday_common_setup.py`, extend `num_cols`:

```python
num_cols = [
    # existing partial-day VC features...
    "vc_max_f_so_far", "vc_min_f_so_far", "vc_mean_f_so_far", "vc_std_f_so_far",
    "vc_q10_f_so_far", "vc_q25_f_so_far", "vc_q50_f_so_far", "vc_q75_f_so_far", "vc_q90_f_so_far",
    "vc_frac_part_so_far",
    "t_base", "abs_delta",
    "minutes_ge_base", "minutes_ge_base_p1", "minutes_ge_base_m1",
    "max_run_ge_base", "max_run_ge_base_p1", "max_run_ge_base_m1",
    "max_minus_second_max",
    "max_morning_f_so_far", "max_afternoon_f_so_far", "max_evening_f_so_far",
    "slope_max_30min_up_so_far", "slope_max_30min_down_so_far",
    "pred_max_round_so_far", "pred_ceil_max_so_far", "pred_floor_max_so_far",
    "pred_plateau_20min_so_far", "pred_ignore_singletons_so_far", "pred_c_first_so_far",
    "err_max_round_so_far", "err_ceil_max_so_far", "err_floor_max_so_far",
    "err_plateau_20min_so_far", "err_ignore_singletons_so_far", "err_c_first_so_far",
    "range_pred_rules_so_far", "num_distinct_preds_so_far",

    # NEW: forecast-only features
    "fcst_prev_max_f", "fcst_prev_min_f", "fcst_prev_mean_f", "fcst_prev_std_f",
    "fcst_prev_q10_f", "fcst_prev_q25_f", "fcst_prev_q50_f", "fcst_prev_q75_f", "fcst_prev_q90_f",
    "fcst_prev_frac_part",
    "fcst_prev_hour_of_max",
    "t_forecast_base",

    # NEW: forecast-vs-actual error features up to τ
    "err_mean_so_far", "err_std_so_far", "err_max_pos_so_far", "err_max_neg_so_far",
    "err_abs_mean_so_far", "err_last1h", "err_last3h_mean",

    # snapshot & calendar & lags
    "snapshot_hour", "snapshot_hour_sin", "snapshot_hour_cos",
    "doy_sin", "doy_cos", "week_sin", "week_cos",
    "month", "is_weekend",
    "settle_f_lag1", "settle_f_lag2", "settle_f_lag7",
    "vc_max_f_lag1", "vc_max_f_lag7", "delta_vcmax_lag1",

    # quality
    "num_samples_so_far",
]
```

The rest of `model1_intraday_logistic.py` remains valid; you just get more powerful features feeding the same regularized, calibrated classifier.

### 3.2 Where to plug forecast features into Model 2 (CatBoost+Optuna)

For CatBoost, you are **not** using sklearn preproc; you pass raw `X_train` DataFrame directly:

```python
all_X, _, _ = build_X_y_intraday(df)   # now includes the new forecast cols
feature_cols = all_X.columns.tolist()

X_train = df_train[feature_cols]
 X_test  = df_test[feature_cols]
```

CatBoost will see the new columns and will naturally exploit them; no other change required.

Your Optuna objective, CatBoost wrapper, and calibration logic remain unchanged – they just operate on a higher-dimensional feature space.

---

## 4. How This Changes the “Walk Forward” Behavior

With these forecast features, the **intraday probability distribution at any time τ** is now conditioned on:

1. What VC thought *yesterday* (full forecast shape + T-1 forecast high).
2. How wrong/right that forecast has been **so far today** (partial forecast error features).
3. What the actual temps today have done up to τ (partial-day max, plateau, slopes, etc.).
4. Seasonality, lags, and city identity (unchanged).

### 4.1 Early in the day (morning)

* Obs: limited; actual high may not have occurred yet.
* Forecast: carries most of the information.
* Model will lean heavily on `fcst_prev_max_f`, `t_forecast_base`, and climatology.

This is exactly where the previous-day forecast helps: it gives you a prior distribution over possible daily highs before the actual max occurs.

### 4.2 Mid/late day (afternoon / evening)

* Obs: you have seen most of the day’s shape.
* Forecast: still useful, but forecast vs actual error features tell you how biased it is.
* Model will gradually shift weight from forecast-only to obs+error.

Your **plateau heuristic** (“after it’s plateaued for 2 hours and then dropped >1σ, run the model”) can be implemented as a trigger to call the model with full partial-day features; but the model itself is agnostic – it just sees a feature vector, including plateau stats and forecast error.

### 4.3 End-of-day (e.g. 22:00–23:50)

* Obs: very close to daily max; `t_base(τ)` is close or equal to `T_settle`.
* Forecast: error stats seal the deal; if forecast said 88°F but you saw 91°F plateaued for 3 hours, the model should be very confident about 90+ brackets.

At this point, the Δ-distribution is typically tightly concentrated (e.g., P(Δ=0) ~ 0.85, P(Δ=±1) small), which is perfect for aggressive last-minute trading if market prices lag that certainty.

---

## 5. No-Lookahead Guarantees with Forecast Features

Using T-1 forecast is still safe:

* The **basis date** for forecasts is D-1 (or earlier), not D or D+1.
* Historical forecasts via the Timeline API with basis date guarantee that you are using the forecast that actually existed yesterday, not a re-run with future data.

At intraday time τ:

* You may use:

  * `fcst_prev_*` for the entire day (it was known since T-1).
  * `err_*_so_far`, built only from times ≤ τ where both forecast and obs exist.
* You may **not** include anything based on obs after τ (we haven’t done that).
* Time splits between train and test are still by **calendar day**, not snapshot. Snapshots on test days are never used in training, and calibration CV splits only within training days in time order.

So the models remain **strictly causal** from the perspective of each day and each snapshot time.

---

## 6. Summary for the Agent

1. **Extend the intraday feature builder** to:

   * Retrieve T-1 **historical forecast** series per `(city, day)` via VC Timeline API with basis date.
   * Generate:

     * `fcst_prev_*` features (full-day forecast shape).
     * `t_forecast_base` = rounded forecast high.
   * At each snapshot τ, align T-1 forecast vs partial-day obs to create:

     * `err_mean_so_far`, `err_std_so_far`, `err_max_pos_so_far`, `err_max_neg_so_far`, `err_abs_mean_so_far`, `err_last1h`, `err_last3h_mean`.

2. **Add these features** to:

   * `num_cols` in `build_X_y_intraday` for Model 1.
   * `feature_cols` for Model 2’s CatBoost configuration.

3. **Re-train Model 1 & Model 2** using the same:

   * Train/test day split (e.g. train ≤ 2024-12-31, test ≥ 2025-01-01).
   * `TimeSeriesSplit` for CV.
   * `CalibratedClassifierCV(method='sigmoid')` for Platt scaling.

4. **Compare against old versions**:

   * Δ accuracy/MAE vs previous models.
   * Settlement accuracy/MAE per snapshot.
   * Brier scores & reliability curves for several thresholds (e.g. T ≥ 80, 85, 90, 95°F).

If the forecast features work as expected, you should see:

* Much better **early-day** discrimination (when obs are sparse).
* Better **calibration** in the tails (close calls near bracket boundaries).
* A model that smoothly blends “what VC thought yesterday” with “what actually happened so far today”.
