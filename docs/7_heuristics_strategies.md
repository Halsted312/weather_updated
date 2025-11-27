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



## Further answers to your specific questions
---

Thanks, this is heading exactly where I want. Here are my answers and how I’d like to proceed.

---

## 1. Implementation scope: stage it, but design for multiple strategies

Let’s **start with base + next_over**, then add divvy on top of that:

* **Option B:**

  * Implement `open_maker_base` (which we already have), plus the new **“next_over” exit strategy** first.
  * Make sure the architecture supports multiple strategies cleanly (via `strategy_id`, separate param dataclasses, shared execution engine).
  * Once that’s solid and we see metrics for *both* strategies side by side, then we add the **divvy** strategy (`center + neighbours`) as a third one.

So: **don’t implement all 3 in one shot.**
Do **base + next_over** first, but build it so adding `open_maker_divvy` is just “drop in another strategy class”.

---

## 2. CLI behaviour: keep backward compatible default

I want this to **continue to work** as it does now:

```bash
python -m open_maker.core --all-cities --days 90
```

So:

* If **no** `--strategy` is supplied, default to `open_maker_base` (the current behaviour).
* If `--strategy` is supplied, run one or more named strategies:

  ```bash
  # Single strategy
  python -m open_maker.core --all-cities --days 90 --strategy open_maker_next_over

  # Multiple strategies in one run
  python -m open_maker.core --all-cities --days 90 \
    --strategy open_maker_base \
    --strategy open_maker_next_over
  ```

That way we can compare strategies side by side without breaking existing scripts/notebooks.

---

## 3. Candle data coverage & price source

We should assume that **`yes_bid_c` is NOT always present** in every minute candle (very thin markets, no resting bid at that instant, etc.). The docs show yes_bid/yes_ask as core price fields for markets and candlesticks, but they don’t guarantee they’re populated every minute.

So from the start, please build in a **robust fallback chain** for the price you use at decision time:

For **our own exit / sell decision** (we’re trying to *hit the bid* to get out):

1. **First choice:** `yes_bid_c` (best bid)
2. If `yes_bid_c` is null but both `yes_bid_c` and `yes_ask_c` are available in nearby candles:

   * Compute a simple **mid**: `(yes_bid_c + yes_ask_c)/2` as a fallback approximation.
3. If neither bid nor ask is usable:

   * Fall back to `close_c` (last traded price) as a final fallback.
4. If everything is missing in a reasonable window around the decision time:

   * Don’t apply the exit rule for that trade; just hold to settlement and log a warning.

For **neighbour bracket prices** used in the **“next_over”** trigger (we’re using this as a *signal*, not a fill price):

* Use `yes_bid_c` if available; otherwise `close_c` (we just need relative levels).

Make this behaviour explicit in code and comments, so we can revisit it if we see weird behaviour in very thin minutes.

---

## 4. Decision time: tune “minutes to estimated high” via Optuna

I like the idea of **tuning how far before the predicted high** we run the exit heuristic.

Concretely for `open_maker_next_over`:

### 4.1 How to define the decision time

Per `(city, event_date)`:

1. Use the **hourly forecast curves** from Visual Crossing for the **target date** and for **basis_date = event_date - 1** (yesterday’s midnight run). You already store this in `wx.forecast_snapshot_hourly`.

2. From that basis, extract:

   * `predicted_high_hour_prev` – the **hour-of-day** for the **max temp for event_date** in yesterday’s run. That’s our “last time we thought we had a high temp (the day before)” reference.

3. Define a tunable integer `decision_offset_min` (negative values mean “before the predicted high”):

   ```python
   decision_time_local = predicted_high_hour_prev + decision_offset_min / 60.0
   ```

   For example:

   * `predicted_high_hour_prev = 15.0` (3pm local),
   * `decision_offset_min = -120` → decision at 13:00 (1pm local).

4. Convert `decision_time_local` to UTC using the city’s timezone (which you already have in `cities.py`), then pick the nearest candle at or before that time for both our bracket and the neighbour.

### 4.2 Optuna search space (fine-grained, 150–200 trials)

Given we want “exact, gradual” control, not huge jumps:

* `decision_offset_min`:

  * Let’s start with a range like **120 to 360 minutes before the predicted high**, in reasonably fine steps:

    ```python
    decision_offset_min ∈ {-360, -330, -300, -270, -240, -210, -180, -150, -120}
    ```

  * That’s 2–6 hours before expected high; more than that and you’re basically back to “midday” or earlier.

Given this plus other parameters, **150–200 trials** is a sensible Optuna budget.

---

## 5. Strategies & parameters: what to implement now

### 5.1 Strategy `"open_maker_base"` – unchanged, baseline

* Already implemented; no changes beyond what you’ve just done.

### 5.2 Strategy `"open_maker_next_over"` – first exit heuristic

Keep this as the first incremental heuristic:

**Entry:**

* Same as base: pick center bin via forecast + temp_bias_deg, post maker at `entry_price_cents` (like 45c), assume we get filled.

**Decision time (tuned):**

* Use `predicted_high_hour_prev` from **yesterday’s** forecast curve (basis_date = event_date-1).
* `decision_offset_min` tunable parameter (negative minutes before that time).

**Exit rule (simple version):**

* At `decision_time`:

  * Let `i` = center bin index (0..5).
  * Let `j = i + 1` (bin immediately above).

* Fetch prices at decision time:

  * `price_center_c` = our bin’s `yes_bid_c` (or mid/close fallback).
  * `price_up_c` = neighbour bin’s `yes_bid_c` (or close fallback).

* Exit rule (thresholds tunable):

  ```python
  if price_up_c >= neighbor_price_min_c and price_center_c <= our_price_max_c:
      # Sell our entire position at price_center_c (assume we hit the bid)
      exit_price_c = price_center_c
      # P&L = (exit_price_c - entry_price_c) * contracts - exit fees
  else:
      hold to maturity
  ```

**Parameters to tune in Optuna (`NextOverParams`):**

* `entry_price_cents` ∈ {40, 45, 50}
* `temp_bias_deg` ∈ [-2.0, +2.0] (continuous)
* `basis_offset_days` ∈ {1, 0}
* `decision_offset_min` ∈ {-360, -330, -300, -270, -240, -210, -180, -150, -120}
* `neighbor_price_min_c` ∈ {40, 45, 50}
* `our_price_max_c` ∈ {20, 25, 30}

**Optuna objective:**

* Use **`sharpe_daily`** as the primary objective (train), with time-based train/test split (70/30).
* After finding best params on train, evaluate once on test and print both ROI & Sharpe.

### 5.3 Strategy `"open_maker_divvy"` – to follow after next_over is solid

Once base + next_over are working and debugged, then:

* Implement `"open_maker_divvy"` with the softmax allocation weights as discussed earlier:

  * Raw params `a_center`, `a_lower`, `a_upper` tuned by Optuna.
  * Convert to weights `(w_center, w_lower, w_upper)` via softmax.
  * Allocate a fixed total `bet_amount_usd` across center and neighbours.
  * No exit logic in v1; pure buy & hold.

This is **Part 8b**, after we’re confident `"next_over"` behaves as expected.

---

## 6. Summary of answers to your specific questions

1. **Implementation scope:**

   * Use **Option B**: implement `open_maker_next_over` first, keeping the architecture multi-strategy-ready. Add `open_maker_divvy` afterwards.

2. **CLI:**

   * Keep current behaviour as default: if `--strategy` is not provided, run `open_maker_base`.
   * Allow multiple `--strategy` flags to run several strategies in one go.

3. **Candle coverage:**

   * Assume `yes_bid_c` is **not 100% present**; implement price selection with robust fallbacks:

     * Our exit / fill price: `yes_bid_c → mid (if we have both bid/ask) → close_c → skip exit if nothing is usable nearby`.
     * Neighbour signal: `yes_bid_c → close_c`.

4. **Decision-time tuning:**

   * Add `decision_offset_min` param (minutes relative to **yesterday’s predicted high hour for today**, from hourly forecast) and tune it with Optuna over a discrete set in roughly `[-360, -120]` with 150–200 trials.

Once you’ve got `"open_maker_next_over"` wired up with these parameters and the Sharpe-based Optuna objective, we can look at the train/test results and a few debug trades, then move on to `"open_maker_divvy"` and more sophisticated combinations.
