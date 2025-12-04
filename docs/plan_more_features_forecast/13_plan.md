You’re right to be skeptical about splitting into three totally separate models. Given your current code, you’re *already* very close to the “one model, continuous in time” setup you want — you just haven’t fully exploited it yet.

Let me unpack what the research coder is saying, how it fits with your existing code, and how to get **one model** that works from D-1 10:00 all the way through the end of D, using both day-before and same-day obs.

---

## 1. You already have the “one model, all times” infrastructure

You actually have two dataset builders that do precisely what you’re asking for:

### a) `snapshot_builder.py` (fixed hourly snapshots) 

* It builds a dataset with rows like `(city, day, snapshot_hour)` and features computed “as of” that snapshot hour.

* Snapshot hours: `[10, 12, 14, 16, 18, 20, 22, 23]`.

* Each row is:

  > “What do we know about this day **so far** at this time?”
  > (obs so far, T-1 forecast, shape, rules, calendar, etc.)

* Then the **ordinal delta model** (via `OrdinalDeltaTrainer`) is trained on all those snapshots, one model across all times of day. 

This is exactly the “one model, many information states” pattern — just on an hourly grid instead of 5/15 minutes.

### b) `tod_dataset_builder.py` (arbitrary 5/15-minute snapshots) 

This is even closer to what you just described:

* `build_tod_snapshot_dataset` generates snapshots every `snapshot_interval_min` minutes between `day_start_hour` and `day_end_hour`.
  Example with 15-minute intervals: `10:00, 10:15, 10:30, …, 23:45` → **56 snapshots per day**.
* For each snapshot it:

  * Filters obs up to that timestamp.
  * Adds T-1 forecast (`fcst_daily`, `fcst_hourly_df`).
  * Computes partial-day features, shape, rules, calendar, quality, **then builds one row per (city, day, snapshot_timestamp)**.
  * Computes `delta = settle_f - t_base` as your label.

In other words: you already have a **continuous-time snapshot dataset builder** that supports exactly the “every 15 minutes I want a prediction for the final high” story. You then train a *single ordinal CatBoost* model (like your TOD v1 / market-clock models) on that dataset.

So in terms of mechanics, you do **not** need three different models; the “pre-obs / partial-obs / near-final” states are just different rows in the same dataset, and the model sees indicators of those states via:

* `snapshot_timestamp`, `snapshot_hour`,
* market-clock features (minutes since D-1 10:00, is D-1 vs D, etc.),
* quality features like `num_samples_sofar`, expected vs actual obs.

---

## 2. Why the research coder suggested multiple models (and why I’d *start* with one)

The research coder’s “three model types” idea — `pre_obs_model`, `partial_obs_model`, `near_final_model` — is basically a **manual mixture-of-experts by information regime**:

* Pre-obs: mostly forecast, almost no obs, more uncertainty.
* Partial: some obs, some forecast error, mid-day.
* Near-final: almost full obs, forecast error known, very confident.

This *can* help if:

* The data distribution is wildly different between regimes, and
* You have enough data to train each model robustly.

But there are well-known trade-offs in splitting models like this:

* You cut your data into 3 chunks → less data per model → more risk of overfit, especially in rare regimes (e.g., weird pre-obs behaviors).
* You complicate your inference pipeline (have to choose a model each time).
* You lose the ability for a *single model* to learn smooth behavior across time (e.g., how confidence ramps up from D-1 evening to D mid-day).

Your **current codebase is already designed for the “one model, multiple snapshots” approach**:

* `tod_dataset_builder` builds many snapshots at fine granularity. 
* `MarketClockOrdinalTrainer` and `OrdinalDeltaTrainer` are already global models that learn from snapshots across cities and times.

So my suggestion is:

> Stick with **one model**, trained on rich snapshot features across the whole 36-hour window, and encode “information state” as features — not as separate models.

If you later see strong empirical evidence that separate models per regime outperform, you can revisit — but your current architecture doesn’t push you into three-model land at all.

---

## 3. How to bake research coder’s ideas into your **one-model** pipeline

You already have:

* `build_dataset` and `build_market_clock_dataset` which build snapshots from D-1 10:00 to D 23:55 with a configurable interval (5 minutes in production).
* `build_snapshot` / `build_snapshot_for_inference` that enforce “only data before cutoff_time” and compute all features. 
* Global models like `MarketClockOrdinalTrainer` that train on all snapshots across time.

To incorporate the research coder’s ideas **without** multiple models:

### 3.1 Treat “information state” as features

You can add (or ensure you use) features like:

* `is_d_minus_1`: 1 if cutoff_time is on D-1, 0 if on D.
* `minutes_since_market_open`: difference between cutoff_time and D-1 10:00.
* `minutes_to_end_of_day`: D 23:59 − cutoff_time.
* `obs_coverage`: `num_samples_sofar / expected_samples` (already implied by `compute_quality_features`).

The model will naturally learn:

* “At D-1 10:00 with no obs, predictions are fuzzy.”
* “At D 17:00 with many obs and big positive forecast error, Δ tends to be positive and small variability.”

In fact, `tod_dataset_builder` already stores `snapshot_timestamp`, and your market-clock variant uses “minutes since market open” as features.

### 3.2 Add forecast drift and boundary distance as *new features* (not new models)

Many of the coder’s “HIGH priority quick wins” (drift, distance to integer boundary, station-city gap) can be layered onto the existing snapshot pipeline:

* Multi-horizon forecasts: You already load them in `build_dataset` via `load_multi_horizon_forecasts` when `include_multi_horizon=True`. 
* At each snapshot, compute:

  * `fcst_high_T1`, `fcst_high_T2`, … from `fcst_multi`.
  * `drift_T1_T3 = T1 − T3`, `drift_slope`, `forecast_volatility` (std of T0..T6).
  * `distance_to_boundary = min(frac_part, 1 − frac_part)` for the current forecast high — how close we are to the integer threshold.

All of those are features you can add to the snapshot row; the same ordinal CatBoost model will train on them. No need to split models.

### 3.3 Use D-1 evening obs in the model (you already do)

You said:

> “If the evening the day before is higher than the forecast was, then maybe next day it will too.”

With `time_window="market_clock"`, you **already** include obs from D−1 as context for predictions on D:

* In `build_dataset`, when `time_window="market_clock"`, `_get_window` uses `get_market_clock_window(event_date)` → D-1 10:00 to D 23:55.
* You also load obs from D-1 and D:

  ```python
  d_minus_1 = event_date - timedelta(days=1)
  day_obs_df = obs_df[
      (obs_df["obs_date"] == event_date) | (obs_df["obs_date"] == d_minus_1)
  ].copy()
  ```

So when the snapshot time is, say, D 11:00, `temps_sofar` includes everything from D-1 10:00 onward. That means:

* If D-1 evening was way above the T-1 forecast, your **partial_day/shape/forecast_error** features will see that and the model can learn patterns like “days following big warm overshoots tend to overshoot again”.

You don’t need a second “pre-obs model” — you just need to be sure your snapshot interval (e.g., 5 or 15 minutes) covers the D-1 portion you care about and that your features use that extended window.

---

## 4. How this all connects to your current Δ-model training

You already have:

* `OrdinalDeltaTrainer` and `MarketClockOrdinalTrainer` as **one-model ordinal regressors** over Δ = settle − t_base.
* Those trainers treat all snapshot rows equally; time is just another feature.

This is exactly the single-model architecture you want:

* The **dataset** (from `build_dataset` / `build_tod_snapshot_dataset`) encodes time-of-day, how many obs you’ve seen, whether T0 forecast exists, etc.
* The **model** is global over all these snapshots, and “learns” how the predictive signal changes as time advances.

The research coder’s suggestion of three models is essentially manual “mixture of experts” by regime. There are MoE architectures in the literature that can do this automatically, but they add complexity.

Given your current stack:

* You already *pool across cities* in some variants (global market-clock),
* You already sample many snapshot times,
* You already have rich features that indicate how much information you have at each time,

…a **single ordinal CatBoost** is the right baseline. If later you want to experiment with regime-specific models, you can — but I wouldn’t start there.

---

## 5. How to tell the research coder what you want

If you want a concise instruction to them that captures your preference:

> **Direction: Single model, full market-clock window**
>
> 1. We will continue to use a **single ordinal CatBoost Δ-model** trained on snapshots from D−1 10:00 through D 23:55 (market-clock window) using `build_dataset` or `build_tod_snapshot_dataset` with a snapshot interval of 5–15 minutes.
> 2. At each snapshot time, features must use **only data available before that cutoff**, as already enforced in `build_snapshot` and `compute_snapshot_features`. 
> 3. Instead of separate models for pre-obs / partial-obs / near-final, we will:
>
>    * Add or expose features that encode “information state”: minutes since market open, is D−1 vs D, obs coverage, etc.
>    * Add forecast drift features (multi-horizon) and distance-to-boundary features as additional columns in the snapshot dataset.
> 4. The model will learn behavior across time; we’ll evaluate its Δ performance (within-1/within-2, calibration) stratified by time-of-day (e.g., D−1 morning vs D noon vs D evening) to ensure it’s behaving sensibly in each regime.
> 5. If, after that, we find a glaring gap (e.g., pre-obs regime underperforms badly and doesn’t improve with more features), then we can explore a second model just for that regime — but the default is a **single model** for simplicity and consistency.

That keeps you aligned with your intuition: continuous decisions, one model, and heavy use of the D-1 evening behavior as context — without an artificial “bright line” at midnight.

If you want, next step I can help design *specific* additional features (the drift, integer boundary, peak window, solar-cloud interaction) in terms of your existing feature modules (`models/features/forecast.py`, `partial_day.py`, `shape.py`) so you can hand over concrete TODOs to the agent.
