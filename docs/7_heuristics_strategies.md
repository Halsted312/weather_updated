## Heuristics and strategies for Open Maker

Given the results you’re seeing, I’d treat the current `open_maker` as **Strategy 0** and then layer new heuristics on top in a structured way, exactly like you suggested.


---

## Message to backend agent

We’re ready to add a family of heuristics on top of the open_maker framework and compare them cleanly.

### High-level architecture

Please refactor `open_maker` so that:

* We can plug in multiple strategies, each with:

  * a **strategy_id** string (e.g. `"open_maker_base"`, `"open_maker_next_over"`, `"open_maker_divvy"`),
  * its own parameter dataclass,
  * its own decision logic,
* And the backtest engine can:

  * run **one or more strategies** over the same date range,
  * write trades to `sim.trade` with `strategy_id`,
  * write runs to `sim.run` with `strategy_id` and parameters,
  * compute and print **Sharpe metrics** for each strategy.

Concretely:

1. In `open_maker/core.py`, extract the current logic into a generic runner:

   ```python
   def run_strategy(
       strategy_id: str,
       params: StrategyParamsBase,
       cities: list[str],
       start_date: date,
       end_date: date,
   ) -> OpenMakerResult:
       ...
   ```

   where `StrategyParamsBase` is a base class, and each strategy subclass defines its own fields.

2. Keep the current single-bin “buy & hold at open” logic as `strategy_id="open_maker_base"`.

3. Add new strategy implementations as separate functions and param dataclasses, but reuse the same plumbing for forecast lookup, market metadata, candles, fees, etc.

---

## Strategy 0 – `"open_maker_base"` (already implemented)

**Description:**
On market open (10am ET previous day), use Visual Crossing `tempmax` for the target date plus a bias, select the center bracket, post a maker order at fixed YES price `p_entry` (e.g. 45–50¢), assume it eventually fills at that price, hold to maturity. No exits.

**Status:** implemented as the current `open_maker.core` behaviour.

Please:

* Treat this as **`strategy_id = "open_maker_base"`** for reporting.
* Ensure `OpenMakerResult.summary()` already prints:

  * ROI,
  * `sharpe_per_trade`,
  * `sharpe_daily`,
  * `pnl_std_per_trade`,
  * `pnl_std_daily`.

---

## Strategy 1 – `"open_maker_next_over"`

*(simple exit heuristic based on a neighbour bin at fixed time)*

**Idea:**
Same entry as `base`, but add a **one-time exit check** at a known decision time (e.g. 11:00 local, or `2h before predicted high`) based on neighbouring bracket’s price. The simplest version:

> If at decision time, the **bin one step above us** (i+1) is trading ≥ 50¢ and our bin is still ≤ 30¢, treat that as “market thinks the high is higher than we do” and **exit at current bid**, even if it’s a loss. Otherwise, hold to settlement.

Later we can generalize to “bin two away”, or symmetric checks up/down.

### Implementation details

1. **Parameters (`NextOverParams`)**

   Add a new dataclass:

   ```python
   @dataclass
   class NextOverParams(StrategyParamsBase):
       entry_price_cents: int         # e.g. 45 or 50
       temp_bias_deg: float           # e.g. +/- 2F
       basis_offset_days: int         # 1 or 0
       bet_amount_usd: float          # e.g. 100 or 200
       decision_mode: str             # 'fixed_time' or 'pre_high'
       decision_hour_local: float     # used if decision_mode == 'fixed_time', e.g. 11.0
       # Exit rule thresholds:
       neighbor_side: str             # 'up' or 'down'; start with 'up'
       neighbor_idx_offset: int       # 1 or 2; start with 1
       neighbor_price_min_c: int      # e.g. 50 -> 50c
       our_price_max_c: int           # e.g. 30 -> 30c
   ```

2. **Decision time**

   Two options (controlled by `decision_mode`):

   * `decision_mode='fixed_time'`:
     Convert `decision_hour_local` (e.g. 11.0 = 11:00) into a UTC timestamp for `(city, event_date)` and use the candle closest to that time.

   * `decision_mode='pre_high'`:
     Use the predicted high hour from the forecast dataset (you already store this in `feature.midnight_forecast_path` and/or can recompute from `wx.forecast_snapshot_hourly`), compute `T_decision = T_pred_high - 2h`, and use the candle closest to that time.

   For v1, a **fixed 11:00 local** is fine; we can add `pre_high` once the mechanics work.

3. **Exit rule**

   During simulation, after entry:

   * Determine your **bin index** `i` (center bracket) from `market_df` sorted by strike.
   * Determine neighbour index:

     * if `neighbor_side='up'`, `j = i + neighbor_idx_offset`
     * if `neighbor_side='down'`, `j = i - neighbor_idx_offset`
   * If `j` is out of range, skip the heuristic (hold).

   At decision time, for those two tickers:

   * Query `kalshi.candles_1m` for both brackets in a small window around `T_decision` (e.g. ±5 minutes) and take the last candle before or at decision time.

   Let:

   * `our_price_c = yes_bid_c` (or mid) for our bin,
   * `neighbor_price_c = yes_bid_c` (or last) for neighbour at that time.

   Exit condition:

   ```python
   if neighbor_price_c >= params.neighbor_price_min_c and our_price_c <= params.our_price_max_c:
       # exit at our current bid
       exit_price_c = our_price_c
       settle based on exit_price instead of final settlement
   else:
       hold to maturity
   ```

   P&L for exited trades = (exit_price - entry_price) * contracts – maker/taker fees (if any on exit; for now assume crossing spread as taker).

4. **Strategy identity & backtest**

   * Implement this decision logic in `run_strategy("open_maker_next_over", params, ...)`.
   * Write trades with `strategy_id="open_maker_next_over"` and a `decision_type="exit_next_over"`.

5. **Optuna integration**

   In `open_maker/optuna_tuner.py`:

   * Add a new “strategy_type” flag or separate entry point for this strategy.

   * For `NextOverParams`, search over:

     ```python
     entry_price_cents     ∈ {40, 45, 50}
     temp_bias_deg         ∈ Uniform(-2.0, 2.0)
     basis_offset_days     ∈ {1, 0}
     neighbor_idx_offset   ∈ {1, 2}
     neighbor_price_min_c  ∈ {40, 45, 50}
     our_price_max_c       ∈ {20, 25, 30}
     decision_hour_local   ∈ {11.0, 13.0, 15.0}
     ```

   * Use **`sharpe_daily`** as the objective metric for this strategy.

---

## Strategy 2 – `"open_maker_divvy"`

*(split bet across center and neighbours with softmax weights)*

**Idea:**
At open, instead of betting 100% on one bin, split the bet across the center bin and its immediate neighbours:

* `w_center` on the forecast bin,
* `w_lower` on bin `i-1`,
* `w_upper` on bin `i+1`,

with `w_center + w_lower + w_upper = 1`, `w ≥ 0`. Edge cases (top/bottom) re-normalize.

We let **Optuna tune** the weights (via softmax) to maximize Sharpe.

### Implementation details

1. **Parameters (`DivvyParams`)**

   ```python
   @dataclass
   class DivvyParams(StrategyParamsBase):
       entry_price_cents: int      # 40, 45, 50
       temp_bias_deg: float        # +/- a few degrees
       basis_offset_days: int      # 1 or 0
       bet_amount_usd: float       # total per event
       # raw allocation params, to be softmaxed:
       a_center: float
       a_lower: float
       a_upper: float
   ```

2. **Convert raw params → weights via softmax**

   In `open_maker.utils`:

   ```python
   def softmax_weights(a_center, a_lower, a_upper):
       exps = np.exp([a_center, a_lower, a_upper])
       denom = exps.sum()
       return exps[0] / denom, exps[1] / denom, exps[2] / denom
   ```

   (Use Python’s `math.exp` if you don’t want numpy.)

3. **Allocate per event**

   After choosing the **center index** `i`, compute `w_center, w_lower, w_upper` and then handle edges:

   * If `i == 0` (lowest bin): set `w_lower = 0`, renormalize center+upper.
   * If `i == last_idx`: set `w_upper = 0`, renormalize center+lower.

   Then:

   ```python
   stake_center = w_center * bet_amount_usd
   stake_lower  = w_lower * bet_amount_usd  # 0 if i == 0
   stake_upper  = w_upper * bet_amount_usd  # 0 if i == last_idx
   ```

   For each allocated bin, compute `num_contracts` and P&L independently, then sum for the event.

4. **No exit in v1**

   Start with no exit heuristic (just buy & hold for all three positions). We can combine this later with an exit heuristic if needed, but first see what pure diversification does.

5. **Strategy identity**

   * Implement as `strategy_id="open_maker_divvy"`.
   * Record each component trade with the correct ticker and P&L, but group them via `run_id` and `strategy_id`.

6. **Optuna integration**

   In the Optuna tuner:

   * For `DivvyParams`, add to the search space:

     ```python
     entry_price_cents    ∈ {40, 45, 50}
     temp_bias_deg        ∈ Uniform(-2.0, 2.0)
     basis_offset_days    ∈ {1, 0}
     a_center             ∈ Uniform(-2.0, 2.0)
     a_lower              ∈ Uniform(-2.0, 2.0)
     a_upper              ∈ Uniform(-2.0, 2.0)
     ```

   * Again, use `sharpe_daily` as the primary objective.

---

## Strategy 3 – `"open_maker_divvy_exit"`

*(optional later: divvy + neighbour-based exit)*

Once Strategies 1 and 2 are working and we have results, we can combine them:

* Use the **diversified allocation** at open (center + neighbours),
* Add a **relative price exit** (like in Strategy 1) two hours before predicted high.

This is “Part 8” in your plan – I’m fine leaving this as a documented “future strategy” for now; let’s get `base`, `next_over`, and `divvy` solid first.

---

## Reporting & comparison

Finally, please update the CLI and reporting so that:

* I can run all three strategies over the same window and compare:

  ```bash
  python -m open_maker.core --all-cities --days 180 \
    --strategy open_maker_base \
    --strategy open_maker_next_over \
    --strategy open_maker_divvy
  ```

* And get a summary table like:

  ```
  Strategy               Trades   Win%   ROI    Sharpe_daily
  open_maker_base        919      59.6%  19.3%  0.433
  open_maker_next_over   880      57.2%  15.1%  0.600
  open_maker_divvy       910      62.0%  17.8%  0.700
  ```

* Each strategy’s trades are stored in `sim.trade` with an appropriate `strategy_id` and `decision_type` so we can slice and dice them later in SQL.

Once you’ve got `base`, `next_over`, and `divvy` implemented and tunable with Optuna (using a time-based train/test split and `sharpe_daily` as the objective), we can look at the results and decide which combination is worth turning into live logic – and where to add more sophisticated rules (like your “two brackets away” exit) without overcomplicating things.
