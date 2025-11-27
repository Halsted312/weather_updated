Yeah, this is looking very strong, and you’re at the point where it makes sense to:

* add the “obs vs forecast curve” heuristic as a **third strategy**,
* sketch a **fourth strategy** that layers a simple linear model on top of your features,
* and define a **live testing plan** with ~$20 bets, orders + cancels, and contract limits in mind.

I’ll give you two big chunks:

1. A **copy-paste prompt for your coder** to implement Strategy 3 (curve-gap heuristic) and Strategy 4 (linear model).
2. A **concrete live-testing plan** using your existing infrastructure, with $20 sizing and realistic use of the Kalshi order API (create/cancel, limits, etc.).

---

## 1. Prompt for the coder – Strategy 3 & Strategy 4

You can paste this into the VS Code agent as-is (or tweak wording).

---

### New Strategies Overview

We already have:

* `open_maker_base` – “maker at open, hold to settlement.”
* `open_maker_next_over` – same entry, with an **exit** heuristic based on neighbour price near the predicted high.

I’d like to add:

* **Strategy 3: `open_maker_curve_gap`**
  – uses **minute-level temps vs forecast hourly curve** to decide whether to *shift up* the effective bracket (or at least adjust our belief).

* **Strategy 4: `open_maker_linear_model`**
  – uses a **simple linear / logistic model** on engineered features (including fog, wind, rain, etc.) to produce a probabilistic forecast at decision time, and trades only when model edge vs market price is big enough.

We’ll keep Strategy 4 initially **linear** (scikit-learn logistic / Elastic Net). Later we can swap in CatBoost / GBDT if we like.

The architecture in `open_maker.core` and `open_maker/strategies` is solid; please treat the strategies package as canonical and use the registry.

---

### Strategy 3: `open_maker_curve_gap`

**Intuition**

At some decision time τ (e.g. 2h before predicted high), compare:

* `T_obs(τ)` – observed temp from **minute-level VC Obs** (`wx.minute_obs`) aggregated to that time.
* `T_fcst(τ)` – forecast temp at τ from **VC historical hourly forecast** (`wx.forecast_snapshot_hourly` with `temp_fcst_f`).

If `T_obs(τ)` is **significantly above** the forecast curve (and trending up), that’s evidence the true high will be higher than the original forecast. We can:

* upgrade our internal high estimate by one bin (or more), and
* (optionally) re-allocate part of our position into the next bin up.

For now, implement `curve_gap` as a **pure entry filter / adjustment** (no intraday trades yet), to keep things simple and testable.

#### 3.1 Parameters (`CurveGapParams`)

Create a new strategy class and params:

```python
# open_maker/strategies/curve_gap.py

from dataclasses import dataclass
from .base import StrategyBase, StrategyParamsBase

@dataclass
class CurveGapParams(StrategyParamsBase):
    entry_price_cents: float = 30.0
    temp_bias_deg: float = 1.0
    basis_offset_days: int = 1
    bet_amount_usd: float = 100.0

    # decision timing, minutes relative to yesterday's predicted-high hour
    decision_offset_min: int = -180  # e.g. -180 = 3h before predicted high

    # obs vs forecast thresholds
    delta_obs_fcst_min_deg: float = 1.5  # T_obs - T_fcst must be >= this
    slope_min_deg_per_hour: float = 0.5  # 1h slope of obs must be >= this

    # shift permissions
    max_shift_bins: int = 1             # allow shifting up by at most this many bins
```

* `basis_offset_days` and `temp_bias_deg` work as in base.
* `decision_offset_min` uses the same mechanism as `next_over`:

  * compute yesterday’s predicted high hour via `get_predicted_high_hour`,
  * convert that + offset to a UTC decision time via `compute_decision_time_utc`.
* `delta_obs_fcst_min_deg` and `slope_min_deg_per_hour` are thresholds for “we are above the curve and heating up.”

#### 3.2 Strategy flow

1. **Entry time** – same as base:

   * At market open (10am ET previous day, using `listed_at`), select the **center bracket** based on `tempmax_fcst_f` (lead_days 1) + bias.
   * Compute `entry_price_cents` and position size exactly as `open_maker_base`.
   * For now, we always post an order (we’ll add “no-trade days” later).

2. **Decision time τ (curve check)**

   For `CurveGapStrategy.decide`:

   * Use `get_predicted_high_hour(session, city, event_date)` with basis_date = event_date-1 (already implemented).

   * Compute `decision_time_utc` with `decision_offset_min` (same as in `next_over`).

   * Pull **minute obs** from `wx.minute_obs`:

     ```sql
     SELECT ts_utc, temp_f
     FROM wx.minute_obs
     WHERE loc_id = :station
       AND ts_utc BETWEEN :decision_time_utc - '60 min'::interval
                       AND :decision_time_utc
     ORDER BY ts_utc;
     ```

   * From `wx.forecast_snapshot_hourly` (basis_date = event_date-1), compute:

     * `T_fcst(τ)` – forecast temp at decision time via interpolation from hourly `temp_fcst_f`.

   * Compute:

     * `T_obs(τ)` – average of obs temps over the last 15 minutes before τ.
     * `delta_obs_fcst = T_obs(τ) - T_fcst(τ)`
     * `slope_obs_1h` – linear slope of `temp_f` over the last hour (in °F/hour).

   * If either:

     * too few obs points (e.g. < 3), or
     * no forecast point,
     * → **skip adjustment**, return `action="hold"`.

3. **Adjustment rule**

   If:

   ```python
   delta_obs_fcst >= params.delta_obs_fcst_min_deg
   and slope_obs_1h >= params.slope_min_deg_per_hour:
   ```

   then:

   * Compute which bin index `i` we originally chose for the event (center bin) from `context.all_brackets` sorted by strike.
   * Compute a new index `j = min(i + params.max_shift_bins, max_idx)` (shift up one bin; we can tune `max_shift_bins` later to allow 2).
   * Keep your **bet size fixed** but recompute P&L **as if the entire position is in bin `j`** instead of `i`.

   In this first version of `curve_gap` we’re not actually simulating a mid-day sell-and-buy (that’s more complicated); we’re saying:

   > “If the obs vs forecast gap and slope were present at τ, then the *correct* decision in hindsight would have been to choose the higher bin.”

   That gives us a pure “did we move to the right bin?” heuristic to evaluate before we model actual intraday trading.

   Implementation wise, in `CurveGapStrategy.decide(context, candles_df)` you can:

   * return `action="hold", override_bracket=j`,
   * and the P&L engine in `run_strategy` can check `override_bracket` when resolving.

   Or more simply: in `CurveGapStrategy`, override `context.chosen_bracket` when deciding, and rely on the base P&L logic.

4. **Registration**

   In `open_maker/strategies/__init__.py`, register:

   ```python
   register_strategy("open_maker_curve_gap", CurveGapStrategy, CurveGapParams)
   ```

5. **Optuna tuning**

   In `open_maker/optuna_tuner.py`:

   * Add `"open_maker_curve_gap"` to `--strategy` choices.

   * For its search space:

     ```python
     entry_price_cents     ∈ {30, 35, 40}
     temp_bias_deg         ∈ Uniform(-2.0, 2.0)
     basis_offset_days     ∈ {1}  # keep simple at first
     decision_offset_min   ∈ {-360, -300, -240, -180, -120}
     delta_obs_fcst_min_deg ∈ Uniform(1.0, 3.0)
     slope_min_deg_per_hour ∈ Uniform(0.2, 1.0)
     max_shift_bins         ∈ {1, 2}
     ```

   * Use `sharpe_daily` as the metric, with the same time-based train/test split.

---

### Strategy 4: `open_maker_linear_model`

**Intuition**

Take the engineered features you already have:

* Forecast side:

  * `tempmax_t0, t1, t2`
  * predicted high hour, daily trends, 3–5 day path
  * VC fields: humidity, cloudcover, precip, windspeed, fog, etc.

* Market side:

  * initial mid/ask/bid for each bracket at/near open (using your candles),
  * bracket index, distance from forecast temp to bin boundaries,
  * price ladder shape (neighbour prices).

* Obs side (for intraday decisions later):

  * delta obs vs forecast at decision time, slopes, volatility.

Then fit a **simple logistic / Elastic Net model**:

[
\text{logit}(P(\text{our bracket wins} | \text{features})) = \beta^\top x
]

Use that probability `p_model` vs market implied `p_mkt = price / 1.0` to decide:

* whether to trade at all, and
* how much to size (e.g. constant $20 if margin of edge > threshold).

#### 4.1 Training script

Add a new module:

* `open_maker/train_linear_model.py`

This script should:

1. Build a **training dataset** from historical data at a fixed decision point (e.g. open):

   For each `(city, event_date)`:

   * Compute features `x`:

     * forecast: `tempmax_t0, tempmax_t1, tempmax_t2, bias-adjusted T, etc.`
     * weather extras from VC daily/hourly: `humidity, windspeed, precip, fog, cloudcover`
     * bracket meta: `bin_index`, `floor_strike`, `cap_strike`, `delta_temp_to_floor`, `delta_temp_to_cap`
     * market: initial price of our bin & neighbours, simple spreads.

   * Label `y = 1` if our *chosen* bracket (center or adjusted) won (using `wx.settlement` + resolver), else `0`.

2. Split data **chronologically** into train/test.

3. Fit a logistic regression / Elastic Net using scikit-learn (or similar):

   ```python
   from sklearn.linear_model import LogisticRegressionCV

   model = LogisticRegressionCV(
       Cs=10, penalty="l2", solver="lbfgs", cv=TimeSeriesSplit(...),
       scoring="neg_log_loss", max_iter=1000
   )
   model.fit(X_train, y_train)
   ```

4. Save:

   * `model.coef_`, `model.intercept_`,
   * feature list / scaler parameters,
     to a JSON or pickle file e.g. `models/open_maker_linear_v1.pkl`.

#### 4.2 Strategy implementation

Add `open_maker/strategies/linear_model.py`:

* `LinearModelParams`:

  ```python
  @dataclass
  class LinearModelParams(StrategyParamsBase):
      entry_price_cents: float = 30.0
      basis_offset_days: int = 1
      bet_amount_usd: float = 20.0
      edge_threshold: float = 0.05  # p_model - p_mkt must be >= this
  ```

* Strategy flow at open:

  1. Compute features `x` exactly as in training.

  2. Load the trained linear model from disk.

  3. Compute `p_model = model.predict_proba(x)[0,1]`.

  4. Let `p_mkt = entry_price_cents / 100.0`.

  * If `p_model - p_mkt < edge_threshold`:
    → **no-trade** (skip this event; return `action="no_trade"` and P&L=0).
  * Else:
    → execute the *same* entry as base/curve_gap (maker at `entry_price_cents`), hold to settlement.

* Register as:

  ```python
  register_strategy("open_maker_linear_model", LinearModelStrategy, LinearModelParams)
  ```

We don’t need Optuna for this yet; the main heavy lifting is training the model and picking `edge_threshold`. Eventually we can add an Optuna tuner that adjusts `edge_threshold` + maybe a temp_bias to improve Sharpe.

---

## 2. Live testing plan with $20, REST + WebSocket, cancels, and limits

Now, how to actually *use* these strategies live with the data you’re streaming.

### 2.1 Constraints & limits

* **Position limits / accountability**: Kalshi filings and analysis show per-contract limits around $25,000 or 25,000 contracts per strike for many retail contracts.

  At $20/trade, with prices 30–50¢, you’re talking about 40–70 contracts per trade — *tiny* relative to those limits, so you’re absolutely safe there.

* With such small size, you can **reasonably assume you’re not moving the market**, especially in liquid brackets; you’re a small maker in the book, not the entire market.

### 2.2 Live trader skeleton

Add a new module, e.g.:

* `open_maker/live_trader.py`

This script should:

1. Subscribe to **market_lifecycle_v2** over WebSocket (you already have `market_open_listener.py`):

   * Detect when a new weather event opens:

     * `series_ticker in {KXHIGHCHI, KXHIGHAUS, ...}`
     * `state == "open"` in `market_lifecycle_v2` payload.

2. On an `open` event for an event_ticker:

   * Immediately:

     * Fetch the **forecast snapshot** (`wx.forecast_snapshot`) for `target_date` and appropriate `basis_date` using the same logic as the strategy.
     * Build a `TradeContext` using the same data loaders you use in backtest (markets, settlement table for sanity, etc.).
     * Call your chosen strategy’s **entry decision**:

       ```python
       strategy = StrategyCls(params)
       context = build_trade_context_for_live(...)
       decision = strategy.decide_on_open(context)
       ```

       For now, `decide_on_open` can be:

       * For base/curve_gap: always “enter” center bin at `entry_price_cents`.
       * For `linear_model`: only enter when `p_model - p_mkt >= edge_threshold`.

   * If strategy says **no trade** (edge too small): skip.

   * If strategy says **trade**:

     * Compute `num_contracts` from `bet_amount_usd = 20.0` and `entry_price_cents`.
     * Place a maker **limit order** via REST:

       ```http
       POST /portfolio/orders
       {
         "ticker": "KXHIGHCHI-25NOV28-B41.5",
         "side": "yes",
         "price": entry_price_cents,
         "size": num_contracts,
         "timeInForce": "GTC",
         "cancelOnPause": true
       }
       ```

       Using the `Create Order` endpoint from Kalshi’s API docs.

3. Track orders and cancels:

   * Store `order_id` and `event_ticker` in a local table (e.g. `sim.live_orders`).

   * Periodically:

     * Check via REST `GET /portfolio/orders/{order_id}` whether you’re filled, partially filled, or still resting.

   * If you want to cancel (e.g. if the strategy indicates at decision time “this is bad now” or if it’s too close to market close):

     * Call `DELETE /portfolio/orders/{order_id}` using the Cancel Order endpoint.

   * If your exit strategy decides to **flatten**:

     * Place a **taker order** on the NO side at the bid (or cross the spread) via REST and then record realized P&L.

4. Intraday checks

For the curve_gap and next_over style heuristics:

* Once per minute (or every 5 minutes) during trading hours:

  * For any open positions:

    * Compute decision time logic (e.g. 2h before yesterday’s predicted high for today), and only check when current time is at/after that time.
    * Use your **minute obs** + forecast to compute the additional signals (curve gap, slope).
    * If the strategy says “exit”, issue cancels and/or closing trades.

In v1, I’d keep live trading extremely simple:

* Use **only base strategy** with the tuned (30¢, +1.1°F) parameters,
* at **$20 per trade** per city per day,
* and **no exits**, just pure buy-and-hold,
* for a couple of weeks, while monitoring:

  * fills (time to fill vs open),
  * slippage (were you really filled at your entry_price?),
  * whether the order book seems to move because of your orders.

Once that looks sane, you can:

* turn on `open_maker_next_over` live with small size,
* then use `curve_gap` and later the linear model as more advanced live filters.

---

### 2.3 Summary of live-testing steps

1. **Hard-code a very safe live size**, e.g.:

   ```python
   bet_amount_usd_live = 20.0
   max_contracts_per_trade = 75  # safety cap
   ```

2. **Run base strategy live**:

   * Use best params from backtest (30c, +1.1°F, basis 1).
   * One trade per city/event/day, $20 each.

3. **Record everything**:

   * Order IDs, timestamps, fills, P&L, bid/ask at time of fill.

4. **Compare live P&L + hit rate** over a modest sample (say 50–100 live trades) to your backtest expectations.

If live tracking looks even in the same ballpark (say, win rate ~60–70% and P&L roughly linear in edge), then it’s worth rolling out:

* next_over (exit heuristic),
* curve_gap (obs vs forecast adjustment),
* and the linear_model strategy.

At that point, the CatBoost / ML model can be a *later upgrade* – the big gains now are more likely to come from:

* realistic fills,
* risk controls (edge filters, size scaling),
* and using the full VC feature set (fog, wind, precip) via a simple model, rather than a huge LLM or a heavy GBDT setup.

If you’d like, once your coder has `curve_gap` + `linear_model` wired up, we can design the exact feature set and label for the first linear model training run, and then define a small suite of “live sanity checks” to ensure the model isn’t just overfitting noise.
