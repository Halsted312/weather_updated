# Modeling features + heuristics: next steps
You’ve got:

* Settlements solid (CLI/CF6 pipeline). 
* Minute-level obs from VC humming. 
* Markets + candles (with proper float strikes + source column). 
* Health checks showing full coverage. 

So the right next move is to point the agent at **feature engineering + heuristics**, not more plumbing.

You said you like the idea of a **midnight-based heuristic** that:

* Starts from the **midnight forecast high** for today,
* Uses the **shape of the next ~5 days’ forecast highs** (is the whole pattern trending up/down?),
* Then tracks **how today’s forecast curve evolves during the day** (1–2h before close, compare actual temp vs original curve),
* And uses all of that to tune the “true” high estimate and decide which bracket to buy/hold/cancel.

Here’s a clean reply you can paste to the agent that sets this up as the next work package.

---

**Reply to agent (you can paste this):**

Awesome – data layer looks great and the daemons are ready. Let’s move on to feature engineering and a first heuristic backtest.

I want to start with a **midnight-based heuristic** that:

* Uses the **midnight forecast high for today**,
* Incorporates the **next ~5 days of forecast highs** (path/trend), and
* Tracks **intraday deviations** between actual temps and the original midnight forecast curve (e.g., 1–2 hours before market close).

Please implement the following in stages.

---

### 1. Feature view: midnight forecast + 5-day path

Create a feature view or materialized view, e.g. `feature.midnight_forecast_path`, with one row per `(city, event_date)` that composes information from:

* `wx.forecast_snapshot` – daily historical forecasts (we already have basis_date + target_date).
* `wx.settlement` – `tmax_final` for that city/date (label). 

For each `(city, event_date)`:

* Let `basis_date = event_date` (midnight-of-day forecast; lead_days = 0).

* Features:

  * `tempmax_t0` = forecast high for today at basis_date (lead_days = 0).
  * `tempmax_t1`..`tempmax_t5` = forecast highs for days 1..5 ahead (lead_days = 1..5).
  * `trend_next_5d` = simple trend features, e.g.:

    * slope of `tempmax_t0..t5`,
    * max/min over next 5 days,
    * `tempmax_t0 - min(tempmax_t1..t5)` and `max(tempmax_t1..t5) - tempmax_t0`.
  * Optionally: previous day’s forecast for today (basis_date = event_date-1, lead_days=1) to capture how the forecast shifted coming into midnight.

* Label:

  * `tmax_final` from `wx.settlement`.
  * `delta_midnight = tmax_final - tempmax_t0`.

This view is the basis for both:

* a simple **Option-1 “midnight forecast edge” backtest**, and
* parameters for the midnight heuristic (e.g., learn how to bias `tempmax_t0` using next-5-day trend).

---

### 2. Feature view: intraday deviations vs midnight curve

Create another feature view/materialized view, e.g. `feature.intraday_temp_vs_midnight`, with one row per `(city, event_date, ts_utc)` on a coarse grid (5-minute or 15-minute, your call) during the trading day.

Inputs:

* `wx.minute_obs` – real 5-minute observations. 
* `wx.forecast_snapshot_hourly` – 72-hour hourly forecast curves for each basis_date.
* `wx.forecast_snapshot` – daily forecast high at basis_date (for context).
* `wx.settlement` – `tmax_final` (label). 

For each row `(city, event_date, ts_utc)`:

* Time features:

  * `ts_local`,
  * minutes since local midnight,
  * minutes to market close / settlement cut-off.

* Observed temp:

  * `temp_obs_f` from `wx.minute_obs`.

* Forecast curve:

  * Map `ts_local` to the **midnight basis_date’s hourly forecast** from `wx.forecast_snapshot_hourly` and interpolate to the minute grid.
  * `temp_fcst_midnight_f` = forecast temp at this time from the midnight forecast.
  * Optionally, `temp_fcst_latest_f` if we ever want to use the latest intraday forecast basis (for now, we can stick to midnight).

* Deviations:

  * `err_temp = temp_obs_f - temp_fcst_midnight_f`.
  * Rolling stats: last 1 hour and 3 hours:

    * `err_temp_roll_mean_1h`, `err_temp_roll_std_1h`,
    * `temp_obs_roll_slope_1h` (trend of obs),
    * `temp_fcst_midnight_roll_slope_1h` (trend of forecast curve).

* Forecast high context:

  * `tempmax_t0` from the previous view (midnight forecast high).
  * `delta_obs_vs_high = temp_obs_f - tempmax_t0`.
  * `delta_now = estimated_high_now - tempmax_t0` once we define an intraday estimator.

* Label:

  * `tmax_final` and optionally an indicator of which bin settled.

We don’t need every 5-minute row for the heuristic initially – we can focus on key decision times:

* **Midnight**,
* **2–3 hours after sunrise**,
* **1–2 hours before market close / settlement**.

But building this generic view makes later ML much easier.

---

### 3. Heuristic definition + backtest harness

Now implement a first **parameterized heuristic** that we can later tune (e.g. with Optuna). The idea:

**3.1 Midnight adjustment heuristic (pre-open)**

At midnight for each `(city, event_date)`:

* Start with `tempmax_t0` (midnight forecast high).

* Compute a **trend-adjusted high**:

  * `T_adj_midnight = tempmax_t0 + α * (mean(tempmax_t1..t5) - tempmax_t0) + β * (max(tempmax_t1..t5) - min(tempmax_t1..t5))`

  where `α` and `β` are tunable parameters (initially small).

* Convert `T_adj_midnight` into bracket probabilities (simple error model or even a degenerate distribution at `round(T_adj_midnight)` for v1).

* Compare to Kalshi brackets and prices at a defined decision time (e.g., first candle after listing):

  * Join to `kalshi.markets` (brackets) + `kalshi.candles_1m` (prices at midnight or first trading minute). 
  * Compute implied market `p_mkt_yes` per bin vs your model `p_model_yes` from `T_adj_midnight`.
  * If `p_model_yes - p_mkt_yes > θ_edge` (another parameter) and volume conditions met, simulate a buy in that bin and hold to settlement.

**3.2 Intraday update heuristic (last 1–2 hours)**

Closer to settlement (e.g., 1–2 hours before close):

* Use `feature.intraday_temp_vs_midnight` to build a simple **intraday adjusted high**:

  * For a late decision time `t_close`:

    * Feature set from that row:

      * `temp_obs_f(t_close)`,
      * `temp_fcst_midnight_f(t_close)`,
      * rolling error and slopes.

    * Define a simple linear correction:

      * `T_adj_close = tempmax_t0 + γ * (temp_obs_f(t_close) - temp_fcst_midnight_f(t_close)) + δ * temp_obs_roll_slope_1h`

      where `γ` and `δ` are tunable.

* Again convert `T_adj_close` to per-bin probabilities and compare to prices at `t_close`:

  * This can be used as a **rebalancing / exit** rule:

    * If you’re long a bin that now looks overvalued vs `T_adj_close`, exit or reduce.
    * If another bin now looks underpriced, rotate.

**3.3 Backtest harness**

* Implement a backtest engine that:

  * Loops over historical days in the continuous window.
  * For each `(city, event_date)`:

    * Runs the **midnight decision** (and logs trades).
    * Runs the **intraday decision** at the chosen times (and logs adjustments).
  * Uses final `tmax_final` to compute P&L per trade and per day.

* Store trades into `sim.trade` and runs into `sim.run` (you already have those tables).

* Expose the heuristic parameters `(α, β, γ, δ, θ_edge, etc.)` in a config object so we can plug Optuna in later.

---

### 4. Priority

For now, please:

1. Implement `feature.midnight_forecast_path` and `feature.intraday_temp_vs_midnight` as views/materialized views.
2. Implement the midnight + intraday heuristic in a backtest module (e.g., `backtest/heuristic_midnight_intraday.py`) with tunable parameters but **fixed default values**.
3. Run a first backtest over the **continuous 6-city window** and output:

   * Daily P&L per city,
   * Hit rate (fraction of days where T_adj was in the correct bin),
   * Distribution of errors (`tmax_final - T_adj_midnight`, `tmax_final - T_adj_close`).

Once that’s working, we can talk about how to plug in Optuna and/or add more ML around this heuristic.

---

That should steer the agent into exactly what you want: a **midnight-anchored, path-aware heuristic** that uses:

* The next 5–7 days’ forecast highs as context, and
* Intraday deviations vs the original midnight curve,

and gets you all the way to a backtestable, parameterized trading strategy.
Please confirm when you’ve implemented the feature views and the backtest harness, and share the first backtest results.