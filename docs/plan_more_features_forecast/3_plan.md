You’re thinking about exactly the right issues here:

1. *Where in the pipeline do these weather features belong?*
2. *Is it honest to backtest using “once-a-day” historical forecasts while planning to trade on live 15-min updates?*

Let’s tackle both.

---

## 1. Where should sub-hourly forecast features live?

You effectively have **two models**:

1. **High-temperature predictor (ordinal / CatBoost)**

   * Tries to predict the day’s high temperature (or bracket distribution) from weather info.

2. **Edge module / edge classifier**

   * Takes “model forecast vs market” and decides if there is a tradable mispricing.

### My recommendation

* **Primary weather signal → high-temp predictor.**
  Anything that’s about the *physics / meteorology* (Visual Crossing forecasts, climatology, recent actual temps) belongs primarily upstream in the **high-temp model**. That’s the model whose job is “what is the actual high likely to be?”

* **Relative & timing features → edge module.**
  The edge module should mostly see:

  * Output of your high-temp model: `E[high]`, uncertainty, bracket probabilities.
  * Market state: prices, spreads, volume, time to close.
  * A *small* number of **derived weather–market deltas**, e.g.

    * `forecast_high_now − market_implied_high_now`
    * `Δ forecast_high since basis run`
    * `Δ forecast_high vs yesterday’s forecast`

So:

> **Yes**, put sub-hourly forecast information into the **predictor** first.
> Then pass only **summaries/differences** into the edge module so you don’t end up re-implementing a weather model inside the classifier.

That keeps the architecture clean:

* Layer 1: “What will actually happen?”
* Layer 2: “Given what I think will happen and what the market thinks, is this edge real?”

---

## 2. Is backtesting with once-daily historical forecasts “apples vs oranges”?

This is the more subtle part.

### What Visual Crossing actually gives you historically

From their historical forecast docs:

> “Visual Crossing currently provides the historical forecasts based on the **full forecast model runs stored at midnight UTC each day**.”

That’s the **basis forecast**—one snapshot per model run (per day). You can ask:

```text
.../timeline/Austin,TX/targetDate/targetDate
  ?forecastBasisDate=YYYY-MM-DD
  &include=minutes
  &key=...
```

and you will get *sub-hourly (15-minute) forecast values* for `targetDate` as of that **single basis run**. Combining with the sub-hourly doc:

* Sub-hourly forecast data is available at a **minimum 15-minute interval**.
* Forecast data is **not interpolated below 15 minutes**.

So you can do:

* For each **basis date** (one per day), fetch a **15-minute forecast path** for the entire target day.

What you **cannot** reconstruct is:

> “The forecast as of 09:15, 09:30, 09:45, … for that day.”

Because the provider only keeps **daily snapshots** of the forecast (midnight runs), not every intra-day update.

So if you:

* Use historical forecast + `include=minutes` to get a **15-min path**, then
* Forward-fill it to 1-minute to align with Kalshi candles,

you’re really modeling:

> “The forecast path for the day as it looked at **midnight UTC (or previous day)**.”

Live, if you hit Timeline every 5–15 minutes, you’ll see:

* The midnight run’s path at the start of the day,
* Then updates when a new high-res model run is ingested (maybe 2–4×/day, not every 15 minutes).

Your backtest will **not** have those intraday revisions.

### Is that “apples vs oranges”?

Kind of, yes – they are **two different information processes**:

1. **Historical backtest** (using basisDate):

   * Forecast information updates **once per day** (midnight UTC).
   * Within that day you see a 15-minute forecast path, but it’s fixed.

2. **Live trading** (polling Timeline minutely):

   * Forecast information updates when model runs update (a few times per day).
   * Within a day, the 15-minute path **can change mid-day**.

So if you **pretend** in your backtest that the minute-by-minute live changes existed historically, you’re giving yourself information that the model wouldn’t have had.

Concretely:

* Backtest with basisDate-only = you’re trading as if you take the midnight forecast, never look again.
* Live with 15-min polling = you’re trading with a strictly richer information set.

So:

> Using midnight-only historical forecasts to approximate *live* intraday update behaviour is **not a perfect backtest** – it **understates** the information you will have live, and the *patterns of forecast changes* won’t match.

But that doesn’t mean the backtest is useless. It just means you have to be clear what you’re simulating.

---

## 3. What’s reasonable vs not reasonable

### Reasonable / honest backtest

Design the model around what you can reconstruct:

1. **Choose a single “vantage time” per day** for the predictor
   e.g., “forecast as of basis date D−1” or “as of D at 00:00 local”.

2. For each day:

   * Use `forecastBasisDate` (or `forecastBasisDay`) to get the **15-minute forecast path** for that day from that run.
   * Collapse this into features for your **high-temp model**:

     * forecasted daily high
     * time of forecast high
     * intraday shape stats (e.g., how sharp is the rise, how early in the day the high is expected, etc.).

3. Train your **ordinal high-temp predictor** using only those basis-aligned features (plus climatology, actual history, etc.). That is fully backtestable.

4. Build your **edge classifier** from:

   * predicted high / bracket probabilities from that predictor,
   * Kalshi prices & microstructure,
   * maybe changes **across basis days** (how did the forecast for this day change from D−3 → D−2 → D−1 → D).

This gives you a **clean, time-consistent backtest**: every decision uses only what was knowable at the chosen basis time.

### Questionable backtest

* Try to **simulate intraday forecast revisions** historically by:

  * Taking the midnight basis 15-min path,
  * Pretending it updates every 15 minutes during the day in some synthetic way.

The danger is:

* You’ll design features (e.g., “Δ forecast over last hour”) that **cannot be grounded** in actual historical forecast behaviour.
* Live data will have patterns of revisions driven by model update schedule / assimilation that your synthetic path doesn’t share.

So I would **not** rely heavily on synthetic intraday updates for anything like Sharpe tuning, threshold selection, etc. That’s where apples vs oranges will bite.

---

## 4. So where do these 15-minute features go, concretely?

Given these constraints, a reasonable architecture is:

### 4.1 High-temperature predictor (layer 1)

Use **historical forecast features that are basis-aligned**:

* For each day, for each basis date you care about (e.g., D−2, D−1, D):

  * Fetch `include=minutes` with `forecastBasisDate`.
  * Derive features such as:

    * `fcst_high_basis_Dm1`, `fcst_high_basis_D`, etc.
    * `fcst_high_diff_D_vs_Dm1`
    * `expected_time_of_high_basis_D`
    * simple shape stats from the 15-min curve (e.g., slope in morning hours).

* Train your ordinal model to predict actual high using those features + historical obs.

That’s physically meaningful and fully backtestable.

### 4.2 Edge detection (layer 2)

Once the predictor can say “given basis-time data, the high is ~X±σ”:

* At decision time within day D (say 10:00, 13:00, etc.) use:

  * **Predictor output** conditioned on basis-aligned forecast info (maybe plus actual temps observed so far).
  * **Market state** at that minute.
  * Derived deltas such as:

    * `forecast_high_basis - market_implied_high_now`
    * `basis_to_basis forecast drift` (D−2→D−1→D).

What I would **not** try to do is:

* Pretend you had a different full forecast run every 15 minutes historically; the data provider doesn’t support that.

### 4.3 For live trading

When you’re live:

* You *will* have more information than the backtest (because you’ll see intraday forecast updates).
* That’s okay if you treat intraday 15-min changes as **conservative extras**, not as the primary source of your edge calibration.

Examples of “safe extra live features” that don’t ruin the backtest story:

* If the new live forecast high moves **against** your position (e.g., forecast drops after you buy a high bracket), you can **tighten risk or skip new trades**. That’s risk management, not extra alpha you’re claiming to have backtested.

* If live forecast moves **in your favour**, you can log that for research, but don’t assume the magnitude of that incremental alpha is as robust as the basis-aligned backtest.

---

## 5. Direct answers to your questions

> **“Wouldn’t I want these features put into the Python predictor high-temperature code before the edge detection?”**

Yes:

* Put **most of the Visual Crossing forecast info (daily+basis+15-min path)** into the **high-temp predictor**.
* Pass **summaries and deltas** into the edge module so it knows “forecast vs market”.

> **“Would it be fair and robust to backtest with historical forecasts that only come out once per day at midnight and then interpolate 15 minutes across the day?”**

* **Fair and robust for modeling “forecast as of basis time”** → **Yes**, as long as you are explicit that:

  * You’re modeling “what did we know as of the midnight (or previous day) run?”
  * The intraday 15-min path you use is **fixed for that basis run**.

* **Fair for modeling live, rolling 5–15 min forecast updates** → **No**, that’s apples vs oranges.

  * Historical basis snapshots are once per day, not once per 15 minutes.

> **“Those are almost two different things, right?”**

Exactly:

* **Backtestable thing**: daily forecast snapshots (with 15-min structure inside each snapshot).
* **Live thing**: sequence of those snapshots over the day as model runs update.

So design your **core models and edge thresholds** around the first, and treat the second as future upside / risk management, not as something you can fully validate historically.

If you’d like, next step I can sketch a concrete feature schema for the high-temp predictor like:

```text
fcst_high_basis_Dm2
fcst_high_basis_Dm1
fcst_high_basis_D
fcst_high_diff_D_vs_Dm1
fcst_high_diff_Dm1_vs_Dm2
fcst_high_time_basis_D (in hours after midnight)
fcst_temp_morning_avg (06:00–12:00 basis D)
...
```

and how to wire that cleanly into your existing ordinal trainer.
