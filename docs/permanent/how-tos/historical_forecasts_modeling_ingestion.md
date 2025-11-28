### VC historical granularity   

* Yes, Visual Crossing *does* support sub-hourly **forecasts** and **historical** data.
* Forecasts: default hourly, but you can get **15-minute forecast intervals** (for CONUS + S. Canada) via `include=minutes` and `minuteinterval` in the Timeline API.
* Historical: sub-hourly with **5–10 minute** resolution via the same Timeline API options.
* You can also get **historical forecasts** via `forecastBasisDate` / `forecastBasisDay` to reconstruct “what the forecast looked like yesterday for today” (you’re already using this).

So yes: you *should* use VC’s forecast (daily & sub-hourly) as features, and also use the **delta between forecast and actuals** intraday.

Below I’ll show:

1. What VC actually provides (granularity + historical forecast).
2. How to wire **forecast features** into the feature set we designed.
3. How to do it **without leakage** both for end-of-day and intraday predictions.
4. How this changes the modeling (for both your multinomial/logistic and CatBoost pipelines).

---

## 1. What Visual Crossing gives you (forecast granularity)

### 1.1. Default forecast resolution

Visual Crossing’s Timeline Weather API:

* Gives a **15-day forecast** per location.
* All forecast data is computed at the **hourly level**, and daily metrics (max/min temperature) are aggregated from hourly values.

So out of the box, you get hourly forecast temps `temp` and daily high `tempmax`.

### 1.2. Sub-hourly / minute-level data

The newer Timeline API supports **sub-hourly** data:

* You request it via `include=minutes` in the Timeline API.
* Historical sub-hourly data: minimum interval **5–10 minutes**, depending on station data and plan.
* Forecast sub-hourly data:

  * **Minimum interval 15 minutes**, with `minuteinterval_15` etc.
  * Coverage: currently **continental US + southern Canada** (perfect for your six US cities).
  * Sub-hourly forecasts typically extend **12–24 hours into the future** (short-term high-res window).

That matches your mental model: you can get forecast temp at 15-min resolution for today and near-term hours, and history at 5–10min resolution, so you can align forecast vs obs on a near-5min grid.

---

## 2. Historical forecast (so we *don’t* look into the future)

You’re already using VC’s historical forecast feature:

* Historical forecasts are the forecasts that were **actually issued on a past “basis date”** for some “target date”.
* VC stores full forecast model runs at **midnight UTC** each day and exposes them via `forecastBasisDate` or `forecastBasisDay`.

Example from docs:

```text
...&include=days&forecastBasisDate=2023-05-01
```

or

```text
...&include=days&forecastBasisDay=5  # "5-day-ahead" forecast
```

So for training you can reconstruct:

* “What did the **D-1 forecast** think the day’s max would be?”
* “What did the **D-3 forecast** think?”
* And even the **full hourly/sub-hourly forecast path** that existed as of that basis date.

This is perfect for a **no-leakage feature**: everything the model sees is *exactly* what a trader would have seen at the time.

---

## 3. How to integrate forecast into the feature set

Let’s layer it on top of what we already designed.

### 3.1. New tables / data you’ll want

In addition to your `wx.vc_minute_weather` (actuals) and `wx.settlement_kalshi` (settlement):

1. **Hourly/sub-hourly forecast table** (historical forecast):

   Something like:

   ```sql
   CREATE TABLE wx.vc_fcst_subhourly (
       city            TEXT NOT NULL,
       basis_date_utc  DATE NOT NULL,      -- forecastBasisDate
       target_day      DATE NOT NULL,      -- day being forecast
       datetime_local  TIMESTAMPTZ NOT NULL,
       temp_f          DOUBLE PRECISION,
       -- other fcst elements if you want
       PRIMARY KEY (city, basis_date_utc, target_day, datetime_local)
   );
   ```

   You populate this via Timeline API queries with `include=minutes` and `forecastBasisDate` for historical backfill, and for production use you just call the forecast API each day without the `forecastBasisDate` (today’s basis).

2. **Daily forecast summary table** (optional but nice):

   ```sql
   CREATE TABLE wx.vc_fcst_daily (
       city            TEXT NOT NULL,
       basis_date_utc  DATE NOT NULL,
       target_day      DATE NOT NULL,
       fcst_tmax_f     DOUBLE PRECISION,    -- daily high from forecast
       fcst_tmin_f     DOUBLE PRECISION,
       fcst_tmax_hour  SMALLINT,           -- forecasted hour of max (local)
       PRIMARY KEY (city, basis_date_utc, target_day)
   );
   ```

Build these using Visual Crossing’s historical forecast API with `forecastBasisDate` and include `days` and `hours/minutes` as needed.

---

### 3.2. Static “prior” forecast features (basis day, not time-of-day)

For each `(city, day)` you pick **one basis forecast** to act as your prior:

* Example: the forecast issued at **00:00 UTC on day D-1** for day D (`forecastBasisDate = D-1` and `target_day = D`).

Static features (same for all cutoffs on that day):

* `fcst_tmax_f_prev` – D-1 forecast’s daily high.
* `fcst_tmin_f_prev` – D-1 forecast’s daily low (optional).
* `fcst_tmax_hour_prev` – forecasted hour of max temp (local).
* `fcst_tmax_f_prev_lead1`, `fcst_tmax_f_prev_lead2`, etc. if you include multiple lead times (D-1, D-2, etc.).

These are “prior beliefs” about the day’s high before any observations arrive.

You **don’t** use `settle_f - fcst_tmax_f_prev` as a feature (that’s the label); but you can compute historical forecast bias offline to see whether there’s systematic under/over-forecast, and eventually build lagged bias features across *previous* days (e.g., 7-day rolling fcst bias per city *up to day D-1*).

---

### 3.3. Dynamic forecast–actual delta features (intraday)

This is what you asked about explicitly: “the delta between the previous forecasted temperatures and the actuals coming in.”

For a given `(city, day, cutoff_time)`:

1. Let `A(t)` be actual VC obs at sub-hourly or 5-min resolution up to time t.
2. Let `F(t)` be the **historical forecast path** (from basis D-1) at the same resolution for day D.

At cutoff `τ` (local):

* **Instantaneous error at current time**:

  * `fcst_temp_now = F(τ_rounded)` (e.g., nearest 15-min or hour).
  * `actual_temp_now = A(τ_rounded)` (you have this).
  * `delta_temp_now = actual_temp_now - fcst_temp_now`.

* **Error statistics so far** (over all times `t ≤ τ`):

  * `mean_error_sofar = mean(A(t) - F(t))`.
  * `rmse_error_sofar = sqrt(mean((A(t) - F(t))^2))`.
  * `max_over_fcst_sofar = max(A(t) - F(t) over t ≤ τ)` (how far above fcst we’ve been).
  * `min_over_fcst_sofar = min(A(t) - F(t) over t ≤ τ)`.

* **Forecasted max vs realized so far**:

  * `fcst_tmax_f_prev` (from D-1 forecast).
  * `delta_vcmax_fcstmax_sofar = vc_max_f_sofar - fcst_tmax_f_prev`.
  * `delta_now_vs_fcst_max = actual_temp_now - fcst_tmax_f_prev`.

* **Forecast’s view of remaining potential**:

  * `fcst_max_remaining = max(F(t) for t > τ)` – what D-1 forecast thought might still happen *after* now.
  * `fcst_max_sofar = max(F(t) for t ≤ τ)` – how high forecast expected we’d be by now.

These give the model a very rich picture of:

* “Are we tracking above/below the forecast?”
* “Is there still forecasted upside later today?”
* “Have we already overshot the forecast by a lot?”

And it’s all **time-consistent**: you only use historical forecast data from basis D-1 and actuals up to τ. There’s no future leakage.

---

### 3.4. How this changes the feature set

On top of the features we already designed (max/min/quantiles, rule outputs, shape features, lags, calendar, quality), add:

**Static forecast priors (per day):**

* `fcst_tmax_f_prev`
* `fcst_tmin_f_prev`
* `fcst_tmax_hour_prev`
* (Optionally: `fcst_tmax_f_prev_lead2`, `fcst_tmax_f_prev_lead3` if you want multi-lead info.)

**Forecast path features (for the whole day):**

* `fcst_mean_f_day` = mean forecast temp for day D.
* `fcst_std_f_day` = std of forecast temps.
* `fcst_q90_f_day` (90th percentile forecast temp).
* `fcst_daily_range = fcst_tmax_f_prev - fcst_tmin_f_prev`.

**Dynamic forecast–actual deltas (per cutoff):**

* `delta_temp_now = actual_temp_now - fcst_temp_now`.
* `mean_error_sofar`.
* `rmse_error_sofar`.
* `max_over_fcst_sofar`, `min_over_fcst_sofar`.
* `delta_vcmax_fcstmax_sofar = vc_max_f_sofar - fcst_tmax_f_prev`.
* `delta_now_vs_fcst_max`.
* `fcst_max_remaining`.
* `fcst_max_sofar`.

You can easily add these into the same feature builder that constructs the base, rule, and shape features.

---

## 4. Modeling changes with forecast features

### 4.1. Conceptual impact

Before forecast features:

* Your model is basically “learn settlement from VC obs + shape + rules.”
* You get ~65% exact and ~97% within 1°F for Chicago with obs alone.

With forecast features:

* You’re effectively doing **Bayesian updating**:

  * Prior: `fcst_tmax_f_prev` and the whole forecast path.
  * Likelihood: how actuals deviate from forecast as the day unfolds.
* The model learns patterns like:

  * “If D-1 forecast was 88°F but by 2 pm actual is already 90°F and residuals are strongly positive, bump the expected settle up.”
  * “If the forecast said 100°F but actuals are consistently 3°F below and the peak is forecast later, maybe the settle ends up closer to 98°F.”

This should:

* Reduce MAE on the **residual** part (beyond VC obs alone).
* Make the multi-class or CatBoost model **sharper** and more stable, especially on borderline days.

### 4.2. No-leakage training design with forecasts

Nothing changes about the CV design:

* You still:

  * Build rows `(city, day, cutoff_time)` with features built from:

    * **historical forecast as of D-1**,
    * **actuals up to cutoff_time**, and
    * no data beyond that time.

* You still:

  * Use **day-based time-series splits** where test days come after train days,
  * Keep all cutoffs for a given day in the same fold.

This guarantees:

* No future days leak,
* No future-of-day leaks via forecast or actual series.

### 4.3. Integration into Model 1 (Multinomial Logistic)

For the multinomial logistic pipeline:

* Just expand your `feature_cols` to include all `fcst_*` and `delta_*` features.
* The logit model will learn:

  * How much to trust the prior forecast vs actual VC obs,
  * When large forecast residuals mean the day will “run hot” or “cool.”

If you want to go more “stats nerd”:

* You can explicitly model the **residual**:

  [
  \Delta = T_\text{settle} - \text{fcst_tmax_f_prev}
  ]

  as a multi-class variable (e.g. Δ ∈ {−4…+4}), but you don’t *have* to. Including `fcst_tmax_f_prev` and its deltas as features in the full-temperature model already gives you most of that benefit.

### 4.4. Integration into Model 2 (CatBoost + Optuna)

For CatBoost:

* Add all forecast features directly; CatBoost eats wide feature sets happily.
* `city` and maybe `month` remain categorical; all `fcst_*/delta_*` features are numeric.

Your Optuna objective doesn’t change:

* It still does day-grouped time-series CV.
* Now the hyperparameter search is simply optimizing over a richer feature set.

You can even:

* Train a **CatBoost regression** on `settle_f` (or on the residual `settle_f - fcst_tmax_f_prev`),
* Or stay with `CatBoostClassifier(loss_function='MultiClass')`.

Either way, forecast features should tighten the distribution.

---

## 5. Intraday usage with forecast features

Now for the “walk forward during the day” part.

### 5.1. At each cutoff time τ:

For live trading on day D at local time τ:

1. **Forecast input**:

   * Use the current forecast time series that you’ve ingested earlier (equivalent to D-1 historical forecast in backtests, or “the latest run you choose” in production).
   * For a clean apples-to-apples backtest, pick a fixed basis (e.g., midnight D-1).

2. **Observation input**:

   * Pull obs from `wx.vc_minute_weather` up to `datetime_local <= τ`.

3. **Feature builder**:

   * Compute base/rule/shape features from obs-so-far.
   * Add static forecast priors (`fcst_tmax_f_prev`, etc.).
   * Add dynamic forecast–actual deltas (`delta_temp_now`, `rmse_error_sofar`, etc.).

4. **Model prediction**:

   * Feed into multinomial logistic or CatBoost model.
   * Get a full distribution `P(T = t | x(τ))`.
   * Convert into bracket probabilities by summing over temps `t` in each bracket range.

### 5.2. Fixed evaluation times vs streaming

You have two operational modes:

1. **Fixed cutoffs** (e.g., 20:00 local each day):

   * Train/evaluate only on those cutoffs.
   * Very clean for backtests and probability calibration.

2. **Streaming predictions** (e.g., hourly from 10:00–22:00):

   * Train on multiple cutoffs per day (as we described).
   * Evaluate performance as a function of time-of-day (which you can encode in features).
   * Use this to define:

     * “Safe windows” when probabilities are stable enough,
     * Or dynamic sizing rules based on current time-of-day and forecast–actual error pattern.

The forecast–actual delta features make **early-day predictions much more informed**, because the model knows whether the day is already warmer/colder than the forecast expected.

---

## 6. TL;DR for your agent

If you want something concrete to tell Claude now:

* **Add ingestion** for Visual Crossing **historical forecasts**:

  * Use `forecastBasisDate` (or `forecastBasisDay`) with Timeline API to create:

    * `wx.vc_fcst_subhourly` (15-min forecast temps for D given basis D-1),
    * `wx.vc_fcst_daily` (daily forecast highs/lows per basis).

* **Extend the feature builder** to:

  * Join these forecast tables for each `(city, day)`.
  * Compute:

    * Static prior features: `fcst_tmax_f_prev`, `fcst_tmax_hour_prev`, etc.
    * Dynamic forecast–actual deltas at each cutoff: `delta_temp_now`, `mean_error_sofar`, `rmse_error_sofar`, `fcst_max_remaining`, etc.

* **Update both models** (multinomial logistic and CatBoost+Optuna) to include these new columns in `feature_cols`.

* **Keep the same day-grouped time-series CV**, making sure:

  * Forecast features are always computed from a basis date ≤ day,
  * Obs features use only data ≤ cutoff_time,
  * Folds split by day in time order.

That gets you a **forecast-aware, intraday, non-leaky probability model** that’s basically the professional version of what you were sketching: VC forecast as prior, VC obs as likelihood, plus your rule logic, all fused by a proper calibrated ML layer.
