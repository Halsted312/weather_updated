Got it — you want something you can drop into the same planning / design doc that spells out:

* what the **linear model** will look like (features + label),
* how we’ll **train & validate** it, and
* what **live sanity checks** we’ll use to keep it honest.

Here’s a self-contained section you can paste into your doc and reference later.

---

## Linear Model Strategy Design (for `open_maker_linear_model`)

Once the `curve_gap` and `linear_model` strategies are wired into the codebase, we’ll add a first, simple **probability model** that predicts whether our chosen bracket will settle YES. This is classic “statistical post-processing” of weather forecasts: learn a mapping from forecast + market features → probability of the event. National meteorological services do exactly this (Model Output Statistics / MOS) to correct raw forecasts using linear/logistic models.

### 1. Problem setup

We frame this as a **binary classification** at a specific decision time (initially, market open or a fixed “pre-high” time):

* **Input**: a feature vector (x_t) for each (city, event_date) at decision time (t).

* **Label**:

  [
  y = \begin{cases}
  1 &\text{if the selected bracket resolves YES (correct bin)} \
  0 &\text{otherwise}
  \end{cases}
  ]

* **Model**: start with a **logistic regression / Elastic Net**:

  [
  \mathrm{logit}(P(y=1|x)) = \beta^\top x
  ]

  This is simple, fast, and well-understood; later we can upgrade to gradient boosting or CatBoost (a standard choice for post-processing ensemble weather forecasts).

The model’s output is a **probability** (p_{\text{model}}) that the chosen bracket will win. For trading, we compare it to the market-implied probability (p_{\text{mkt}} = \text{price}/1.0) and trade only when (p_{\text{model}} - p_{\text{mkt}}) is large enough.

---

### 2. Feature set

We’ll build one feature vector per (city, event_date) at a chosen decision time. The feature set can evolve, but v1 will include:

#### 2.1 Forecast features (Visual Crossing)

Using `wx.forecast_snapshot` (daily) and `wx.forecast_snapshot_hourly` (hourly):

* **Daily high forecasts**:

  * `tempmax_t0`: forecast high for today (target_date) from yesterday’s basis (lead_days=1).
  * `tempmax_t1`, `tempmax_t2`: forecast highs for next 2 days (lead_days=2,3).
  * Derived:

    * `delta_t1_t0 = tempmax_t1 - tempmax_t0`
    * `delta_t2_t0 = tempmax_t2 - tempmax_t0`
    * `trend_3d` = slope of (t0, t1, t2).

* **Hourly curve shape** for today (basis = yesterday):

  * `predicted_high_hour_prev`: hour-of-day of max forecast temp for the target date.
  * `morning_mean_fcst`, `afternoon_mean_fcst`: mean forecast temp over 06–12 and 12–18 local.
  * `curve_steepness_am`, `curve_steepness_pm`: temp change per hour from morning to afternoon.

* **Weather condition features** for the day:

  * `humidity_fcst` (daily average or at decision time),
  * `windspeed_fcst`,
  * `precip_fcst` (expected rainfall),
  * `fog_fcst` (binary/indicator if VC conditions contain fog),
  * `cloudcover_fcst`,
  * `thunderstorm_flag` if conditions include storms, etc.
    These are standard predictors in MOS / post-processing systems for temperature and severe weather.

#### 2.2 Market features (Kalshi)

Using `kalshi.markets` + `kalshi.candles_1m` at/near decision time:

* **Bracket metadata**:

  * `bin_index` (0..5 after sorting by strike),
  * `floor_strike`, `cap_strike`,
  * `bin_width = cap_strike - floor_strike`,
  * `delta_temp_to_floor = tempmax_t0 - floor_strike`,
  * `delta_temp_to_cap = cap_strike - tempmax_t0`.

* **Price ladder** at decision time:

  * `price_center`: mid/last for chosen bracket.
  * `price_neighbour_down`, `price_neighbour_up`.
  * Derived:

    * `spread_center = ask_center - bid_center`,
    * `skew_up = price_up - price_center`,
    * `skew_down = price_center - price_down`.

* **Volume/liquidity** (if available):

  * recent trade count or volume in center and neighbour bins over last N minutes.

These features tell us how much the **market already agrees** with our weather-based view, which is key for edge filtering.

#### 2.3 Observation vs forecast features (for intraday decision times)

If the decision time is intraday (e.g. 2h before predicted high), add:

* `T_obs(t)`: observed temp at decision time (VC minute obs, aggregated over 15 min).
* `T_fcst(t)`: forecast temp at decision time (from hourly `temp_fcst_f`).
* `delta_obs_fcst = T_obs(t) - T_fcst(t)`.
* `slope_obs_1h`: °F/hour slope over last hour.
* `err_mean_1d`: mean forecast error for this city over trailing N days (from settlement vs forecast).

These are the same ingredients used in the `curve_gap` heuristic, just being fed into the model as numeric features.

---

### 3. Label definition

For each (city, event_date, decision_time):

* Use `wx.settlement.tmax_final` and the same bracket resolver we use in the strategies.
* Define:

  ```text
  y_bracket_win = 1 if the bracket we *would* choose at decision time eventually resolves YES
                  0 otherwise
  ```

You can also define multi-class labels (“which bin wins?”), but v1 will focus on “will the center (forecast) bin win?” since that’s directly tied to the open_maker strategies.

---

### 4. Training & validation protocol

Time series / forecast data must be evaluated **chronologically**, not with random k-folds. Best practice is a **rolling origin** or fixed time split.

We’ll do both:

#### 4.1 First cut: single train/test split

* Choose a window (e.g. 2022-01-01 → 2025-11-26).

* Split chronologically:

  * Train: earliest 70% of days.
  * Test: latest 30% of days.

* Train logistic / Elastic Net on train, then evaluate on test.

Metrics:

* **Log loss** / negative log likelihood.
* **Brier score** (mean squared error of predicted probabilities).
* Calibration curve / reliability diagram (is p=0.7 really ~70% correct?).

#### 4.2 Optional: rolling origin evaluation

Once v1 is working, add a rolling origin evaluation (e.g., Hyndman’s rolling forecasting origin):

* Train on first N months, evaluate next M days.
* Roll window forward and repeat.
* Average scores across “origins” to check stability over time.

This is especially helpful to ensure the model isn’t just lucky in one particular year.

---

### 5. Calibration & probability sanity checks

We want **probabilities** from the linear model that are:

* **Well calibrated** (e.g., 70% predictions win ~70% of the time), and
* **Sharp** enough to be useful for trading.

After training:

1. Plot **calibration curves** on the test set:

   * bin predicted probabilities,
   * compare predicted vs empirical frequencies.

2. If needed, apply a simple calibration method:

   * For smaller datasets (<~10k samples per city), **Platt scaling** (logistic calibration) often works well.
   * For larger datasets or more complex miscalibration, **isotonic regression** on top of the linear model’s scores can correct non-sigmoid distortions.

3. Use **Brier score** and reliability diagrams to compare “raw vs calibrated” probabilities; pick the version with better calibration on hold-out test.

---

### 6. Live sanity-check suite

Before and during live trading, we’ll maintain a small suite of checks to detect overfitting or regime shifts:

1. **Rolling performance vs backtest**

   * Maintain rolling windows (e.g. last 30/60 live days) of:

     * win rate,
     * P&L per trade,
     * Sharpe_daily,
     * calibration (e.g. predicted p deciles vs realized frequencies).
   * Alert if live metrics drift far from backtest expectations (e.g. win rate drops from 70% → 50% over 60 days).

2. **Edge realization check**

   * For each trade, record:

     * `p_model`, `p_mkt`, and realized outcome.
   * Group trades by `p_model` bins (e.g. 0.6–0.7, 0.7–0.8, …) and monitor:

     * does the empirical win rate in each bin stay close to the mean predicted `p_model` in that bin? (Basic calibration check.)

3. **Feature drift monitoring**

   * Track summary stats of key features (e.g., distribution of `tempmax_t0`, `delta_obs_fcst`, bracket prices) over live periods.
   * If distributions shift significantly vs training (e.g., due to a change in Kalshi’s participant mix or a change in VC forecast bias), retrain the linear model using the most recent rolling window (e.g., last 12–18 months).

4. **Shut-off logic**

   * Define hard rules to **stop trading** or cut size when:

     * Sharpe_daily over the last N days falls below a threshold (e.g. 0 or negative),
     * calibration error exceeds a tolerance (e.g., Brier score much worse than baseline),
     * or abnormal behaviour (e.g. repeated filled trades at prices the backtest rarely saw).

These checks are relatively simple to implement and will act as guardrails so the linear model doesn’t quietly drift into a regime where it “looks smart but trades dumb.”

---

That should give you a nice, referenceable section in the doc for:

* what the **first linear model** will look like,
* how we’ll **train / validate / calibrate** it,
* and how we’ll **monitor live performance** to keep overfitting in check.

