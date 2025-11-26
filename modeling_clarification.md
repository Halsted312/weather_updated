

## How Kalshi’s *daily high temperature* markets actually work (and why this matters)

* **Source of truth** — Markets settle off **NWS Daily Climate Reports (CLI)** (or equivalent NWS climate products). The help center states: weather contracts “settle based on the final climate report issued by the National Weather Service…typically released the following morning.” ([Kalshi Help Center][1])

* **Rulebooks** — The CFTC‑filed rulebooks specify the product mechanics:

  * **HIGHTEMP** (“highest temperature at <station>…”): Source Agency = **NWS**; settlement off the set of **Daily Climate Reports**; **Last Trading Time** often specified as **11:59 PM ET** and (for some listings) may be **the day prior** to the event date. **Expiration time** typically **10:00 AM ET**; determination may be delayed in edge cases. (See Appendix A lines on Last Trading Date/Time and Expiration.) 
  * **DVTEMP** (Death Valley variant) shows a different pattern: **Last Trading Date = the event date**, **LTT 11:59 PM PT**; still NWS Daily Climate Report is the underlying. (Their Appendix A links to the exact NWS CLI URL.) 
  * **Takeaway:** **Do not hardcode “intraday on event day.”** Different weather products use different LTT rules. Always pull timing from the API for each market instance.

* **Programmatic timings** — The Markets API returns `open_time`, `close_time`, `expected_expiration_time`, `expiration_time`, and `early_close_condition`. Use **`close_time`** (UTC) as the actual last tradable timestamp for your backtests and live logic. ([Kalshi API Documentation][2])

* **Trading hours generally** — Exchange is 24/7 except maintenance windows; still, each market has its own **close_time**. ([Kalshi Help Center][3])

* **Fees** — Maker/taker formulas (ceil to the next cent) are in the fee schedule: **taker** `ceil(0.07 * C * P * (1−P))`, **maker** `ceil(0.0175 * C * P * (1−P))`. Build with the PDF schedule that matches your data period. ([Kalshi][4])

> **Implication for strategy & modeling:**
> “Intraday trading” means *trading minute‑by‑minute while the market is open prior to `close_time`*, not necessarily while the physical temperature is evolving on the event day. For some cities/products, you’ll trade **all day up to 23:59 (local/ET)**; for others, **trading stops the prior night**. Your execution and feature windows must be keyed to `close_time` from the API.

---

## What to optimize: probability vs. temperature?

Pros do both, but **your edge is almost always in calibrated probabilities** at *decision time* (minutes before `close_time`), not in raw temperature point forecasts:

* **For bin (range) markets**, trading is about **Pr(YES | info at time t)** vs. effective price (fees+slippage+spread).
* A **TMAX regressor** can be useful if you transform it to **bin probabilities** (via estimated distribution CDF), but a direct **classifier per bin** with calibration is usually simpler to deploy, and integrates execution features (spread, microstructure) that a pure weather regressor ignores.

Given your rich 1‑min/5‑min market + weather data, the most robust approach is:

1. **Classification to calibrated `p̂_yes(t)`** for each bin (or one‑vs‑all), **time‑aware** (feature includes `minutes_to_close`).
2. **Kelly‑sized execution** with explicit microstructure controls (spread, slippage, volume).
3. **Optional** TMAX distribution model (CatBoostRegressor w/ quantiles) later to improve tails.

---

## Train/test split: day holdouts vs. within‑day

Use **both**, but in different places:

* **Model training & calibration:** **Group by event day** and do **rolling walk‑forward** (e.g., train on prior 42 days → validate 7 days; step 1 day). Each *row* is a **snapshot at minute t**. This prevents leakage across days but lets the model learn from intra‑day snapshots in history.
* **Backtesting:** simulate a **minute‑bar execution** from market `open_time` to `close_time` using only features available at each minute **t**; no peeking past t.
* **Intraday vs. hold‑to‑settlement P&L:** keep **one backtester** with a **strategy interface** so you can switch between:

  * **Hold‑to‑settlement** (enter before `close_time`, P&L at settlement), and
  * **Intraday P&L** (enter/exit pre‑close to ride swings).
    Your current harness can be extended (see stubs below).

---

## Answers to your agent’s clarifying questions (recommended choices)

1. **Implementation order:** Ridge baseline → Lasso → **CatBoost** classifier (CPU). Each added only after prior is validated end‑to‑end.
2. **Quantile regression:** **Later.** Start with CatBoostClassifier; add quantile CatBoostRegressor (for TMAX distribution) once the classifier+calibration+execution loop is green.
3. **Opinion pooling (market vs. model):** **Phase 2**—get calibrated model working first, then add log‑odds pooling weight **w** estimated on the *train window* by minimizing log‑loss.
4. **City scope:** Start with **Chicago** (KMDW) to simplify debugging; then all cities.
5. **Backtester structure:** Keep one unified harness with a **Strategy** plugin (your simple buy/hold becomes `BuyHoldStrategy`; the Kelly‑sized ML strategy becomes `ModelKellyStrategy`).
6. **Walk‑forward params:** Configurable; start with **train=42 days, test=7 days, step=1 day**; rolling window. (You can also try 28/7 if you need faster cycles.)
7. **Spread & slippage controls:** Use **max spread 3¢** and **slippage 1¢** as defaults; allow overrides. Wide spreads can be exploitable, but the expected edge must cover **half‑spread + slippage + fees**; we gate entries accordingly. (Maker flow can relax this later.) Microstructure on Kalshi does matter. ([Karl Whelan][5])

---

## Features pros actually lean on (beyond what you listed)

**Market microstructure (per minute):**

* Mid, best bid/ask, **half‑spread**, spread % of price.
* Quote momentum: Δmid(1/5/15m), Δyes_bid, Δyes_ask; **RSI‑style oscillator** over yes_bid.
* **Order‑flow imbalance** (trades hitting yes vs. no), rolling buy/sell volume ratio, **price impact** proxies.
* **Order book** depth if you have it (levels 1–5); else rolling **liquidity** from the API fields.
* Time since last trade; volatility of last 15 minutes.

**Weather nowcasting (aligned to 5‑min obs → ffill≤4m into 1‑min):**

* Current temp, **Δtemp slopes** (5/15/30/60 min); **rolling max in last 30/60/120 min**.
* **Distance‑to‑bin edges**: `temp_to_floor = floor_strike - temp_now`, `temp_to_cap = cap_strike - temp_now` (and absolute distance), **monotone** w.r.t. P(YES) for ‘greater/less’.
* **Diurnal context**: minutes since sunrise/sunset, solar elevation angle bucket, cloud cover proxy, wind, humidity, pressure trend (if available).
* **Forecast anchors**: from your provider (if accessible) or a simple prior: rolling **day‑ahead forecast** vs. nowcast error residuals.

**Calendar/time features**

* `minutes_to_close`, hour‑of‑day (local), day‑of‑week, recent volatility regime dummies.

Tie monotone constraints to physics‑guided features:

* For **‘greater’** bins: P(YES) **increases** as `temp_now` ↑ and `minutes_to_close` ↓; P(YES) **decreases** as `temp_to_floor` ↑ (further below threshold).
* For **‘less’** bins: monotonicities flip accordingly.
* For **‘between’** bins: monotonic with **|distance to interval|** (further from interval → lower probability).

---

## Execution & risk rules (your edits folded in)

* **Entry gate (taker):** only enter if
  `edge_cents ≥ τ_open + half_spread_cents + slippage_cents + fee_cents`
  with defaults **τ_open = 3¢**, **half‑spread = spread/2**, **slippage = 1¢**, fees from schedule. ([Kalshi][4])
* **Exit hysteresis:** require `edge_cents ≤ τ_close` (**0.5–1.5¢**) to exit; `τ_close < τ_open`.
* **Max spread:** **3¢** default; configurable; *allow* entries on wider spreads only if **edge** covers the extra spread buffer (i.e., auto‑raise τ_open by the extra half‑spread).
* **Sizing:** Fractional Kelly on **effective price** (fees+slippage). Use **alpha‑Kelly**, e.g., α∈{0.1, 0.25, 0.5}, to control risk (this scales the Kelly fraction; not a probability threshold).
* **City/day/side risk cap:** **≤10% of bankroll** per (city, day, side).
* **Concurrency:** **max 3 bins per city**; if a 4th signal appears, **replace weakest edge**.

---

## Modeling stack and Optuna tuning (CPU‑only CatBoost + Ridge/Lasso)

**Why CatBoost first?** Strong with mixed feature types, handles monotone constraints, robust on modest data. No GPU needed.

### Step 0 — Dataset builder (minute snapshots)

Each minute t before `close_time`, create a snapshot row with features from market+weather up to t and label = **final YES/NO** for that bin. Group by **event day** and **city** for CV.

> **Important:** For weather markets whose LTT is the **prior day**, your snapshots stop at that `close_time`. The physical temperatures evolve *after* the market is closed; your model is forecasting settlement using forecasts/nowcasts **before** the day occurs. For markets with LTT at the **end of the event day**, you can incorporate day‑of observations up to `close_time`. Use API `close_time` to avoid leakage for each market. ([Kalshi API Documentation][2])

---

## Code you can drop in

> Below are **lean stubs** wired to your conventions (prices in **cents**, times in UTC, daily group CV, Optuna 15 trials per step). They assume your existing `db`, `backtest`, and `fees` modules.

### `models/kelly.py`

```python
# models/kelly.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class FeeParams:
    taker_mult: float = 0.07     # from fee schedule
    maker_mult: float = 0.0175
    round_up: bool = True

def taker_fee_cents(price_cents: int, contracts: int, fees: FeeParams) -> int:
    # ceil(0.07 * C * P * (1 - P)) in CENTS; P in dollars
    p = price_cents / 100.0
    raw = 0.07 * contracts * p * (1.0 - p) * 100.0
    return int(raw) if raw == int(raw) else int(raw) + 1

def effective_yes_entry_cents(price_cents: int, slippage_cents: int, fees: FeeParams, maker: bool=False, contracts:int=1) -> int:
    fee = taker_fee_cents(price_cents, contracts, fees) if not maker else int((0.0175 * (price_cents/100.0) * (1-(price_cents/100.0)) * 100)+0.9999)
    return price_cents + slippage_cents + fee

def break_even_prob(price_cents_eff: int) -> float:
    # With unit $1 payoff, ignoring exit fees on settlement
    return min(0.9999, max(0.0001, price_cents_eff / 100.0))

def kelly_fraction_yes(p_hat: float, price_cents_eff: int, alpha: float = 0.25) -> float:
    """
    Kelly for binary contract with $1 payout and entry cost y_eff (dollars).
    f* = (p − y_eff) / (1 − y_eff). We scale by alpha for risk control.
    """
    y_eff = price_cents_eff / 100.0
    edge = p_hat - y_eff
    denom = (1.0 - y_eff)
    f_star = 0.0 if denom <= 0 else (edge / denom)
    return max(0.0, alpha * f_star)

def gate_entry(edge_cents: float, spread_cents: int, slippage_cents: int, fee_cents: int, tau_open_cents: int = 3) -> bool:
    half_spread = spread_cents / 2.0
    needed = tau_open_cents + half_spread + slippage_cents + fee_cents
    return edge_cents >= needed

def gate_exit(edge_cents: float, tau_close_cents: float = 1.0) -> bool:
    return edge_cents <= tau_close_cents
```

### `risk/manager.py`

```python
# risk/manager.py
from collections import defaultdict

class RiskManager:
    def __init__(self, bankroll_cents: int, city_day_side_limit_pct: float = 0.10, max_bins_per_city: int = 3):
        self.bankroll_cents = bankroll_cents
        self.limit_pct = city_day_side_limit_pct
        self.max_bins_per_city = max_bins_per_city
        self.city_day_side_exposure = defaultdict(int)  # key: (city, date, side) -> cents
        self.city_positions = defaultdict(set)          # key: city -> set of active tickers

    def can_open(self, city: str, day_key: str, side: str, additional_exposure_cents: int) -> bool:
        key = (city, day_key, side)
        cap = int(self.bankroll_cents * self.limit_pct)
        if self.city_day_side_exposure[key] + additional_exposure_cents > cap:
            return False
        if len(self.city_positions[city]) >= self.max_bins_per_city:
            return False
        return True

    def register_open(self, city: str, day_key: str, side: str, ticker: str, exposure_cents: int):
        self.city_day_side_exposure[(city, day_key, side)] += exposure_cents
        self.city_positions[city].add(ticker)

    def register_close(self, city: str, day_key: str, side: str, ticker: str, exposure_cents: int):
        self.city_day_side_exposure[(city, day_key, side)] = max(0, self.city_day_side_exposure[(city, day_key, side)] - exposure_cents)
        if ticker in self.city_positions[city]:
            self.city_positions[city].remove(ticker)

    def replace_weakest_if_needed(self, city: str, candidate_edge: float, current_edges_by_ticker: dict) -> str | None:
        """If at limit, evict weakest edge in city for a stronger candidate."""
        if len(self.city_positions[city]) < self.max_bins_per_city:
            return None
        weakest = min(current_edges_by_ticker.items(), key=lambda kv: kv[1], default=(None, None))
        if weakest[0] is None:
            return None
        if candidate_edge > weakest[1]:
            return weakest[0]  # signal to close weakest
        return None
```

### `models/catboost_model.py` (CPU; monotone constraints; Optuna 15 trials per walk‑forward step)

```python
# models/catboost_model.py
from __future__ import annotations
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
import optuna

def train_catboost_with_optuna(
    X, y, groups, feature_names, monotone_features: dict[str, int],  # name -> +1/-1
    n_trials: int = 15, random_state: int = 42
):
    # Build monotone vector aligned to feature order
    mono_vec = [monotone_features.get(f, 0) for f in feature_names]

    def objective(trial: optuna.Trial):
        params = {
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
            "random_seed": random_state,
            "verbose": False,
            "monotone_constraints": mono_vec,
            "bootstrap_type": "MVS",
            "auto_class_weights": "Balanced",  # can also try "SqrtBalanced" or fixed weights
            "border_count": trial.suggest_int("border_count", 64, 254),
            "early_stopping_rounds": 200
        }

        gkf = GroupKFold(n_splits=5)
        losses = []
        for train_idx, val_idx in gkf.split(X, y, groups):
            train_pool = Pool(X[train_idx], y[train_idx], feature_names=feature_names)
            val_pool   = Pool(X[val_idx],   y[val_idx],   feature_names=feature_names)
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool, verbose=False)
            # Probabilities and logloss
            p = model.predict_proba(val_pool)[:,1]
            eps = 1e-12
            ll = -np.mean(y[val_idx]*np.log(p+eps) + (1-y[val_idx])*np.log(1-p+eps))
            losses.append(ll)
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Refit on full (train) data with best params
    best = study.best_params
    best["loss_function"] = "Logloss"
    best["eval_metric"] = "Logloss"
    best["random_seed"] = random_state
    best["verbose"] = False
    best["monotone_constraints"] = [monotone_features.get(f, 0) for f in feature_names]
    best["bootstrap_type"] = "MVS"
    best["auto_class_weights"] = "Balanced"
    best["early_stopping_rounds"] = 200

    full_pool = Pool(X, y, feature_names=feature_names)
    model = CatBoostClassifier(**best)
    model.fit(full_pool, verbose=False)

    return model, study
```

### `models/calibration.py` (isotonic if enough data; else Platt)

```python
# models/calibration.py
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

def calibrate_classifier(estimator, X_cal, y_cal):
    n = len(y_cal)
    if n >= 1000:
        cal = CalibratedClassifierCV(base_estimator=estimator, method="isotonic", cv="prefit")
    else:
        cal = CalibratedClassifierCV(base_estimator=estimator, method="sigmoid", cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal
```

### Ridge/Lasso helpers (tuned with Optuna; mirrors CatBoost flow)

```python
# models/logistic_models.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import optuna

def train_logreg_optuna(X, y, groups, l1_ratio=None, n_trials=15, random_state=42):
    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 1e+2, log=True)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
        if l1_ratio is None:
            penalty = "l2"
            solver = "lbfgs"
        else:
            penalty = "elasticnet"
            solver = "saga"
        
        gkf = GroupKFold(n_splits=5)
        losses = []
        for tr, va in gkf.split(X, y, groups):
            lr = LogisticRegression(
                C=C, class_weight=class_weight,
                penalty=penalty, solver=solver, max_iter=5000,
                l1_ratio=l1_ratio, random_state=random_state, n_jobs=-1
            )
            lr.fit(X[tr], y[tr])
            p = lr.predict_proba(X[va])[:,1]
            eps = 1e-12
            ll = -np.mean(y[va]*np.log(p+eps) + (1-y[va])*np.log(1-p+eps))
            losses.append(ll)
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    lr = LogisticRegression(
        C=best["C"], class_weight=best.get("class_weight", None),
        penalty=("l2" if l1_ratio is None else "elasticnet"),
        solver=("lbfgs" if l1_ratio is None else "saga"),
        l1_ratio=l1_ratio, max_iter=5000, n_jobs=-1
    )
    lr.fit(X, y)
    return lr, study
```

### Opinion pooling (log‑odds weighted)

```python
# models/pooling.py
import numpy as np

def log_odds_pool(p_model: float, p_market: float, w: float) -> float:
    """p* = sigmoid( w*logit(p_model) + (1-w)*logit(p_market) )"""
    eps = 1e-6
    p_model = np.clip(p_model, eps, 1-eps)
    p_market = np.clip(p_market, eps, 1-eps)
    lm = np.log(p_model/(1-p_model))
    lk = np.log(p_market/(1-p_market))
    z = w*lm + (1-w)*lk
    return 1.0/(1.0+np.exp(-z))
```

### Strategy plug‑in and backtester glue

Create a strategy interface and a model‑driven implementation that respects your risk rules and gating:

```python
# backtest/strategy.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class Strategy(ABC):
    @abstractmethod
    def on_minute(self, ctx: Dict[str, Any]) -> list[Dict]:
        """
        ctx: {
          'timestamp', 'city', 'event_date', 'ticker', 
          'yes_bid', 'yes_ask', 'spread_cents', 'features', 
          'p_market', 'p_model', 'p_pooled', 'price_cents'
        }
        Return list of trade intents:
          [{'action':'BUY'|'SELL', 'contracts':int, 'ticker':str}]
        """
        ...
```

```python
# backtest/model_strategy.py
from models.kelly import effective_yes_entry_cents, break_even_prob, kelly_fraction_yes, gate_entry, gate_exit
from risk.manager import RiskManager

class ModelKellyStrategy(Strategy):
    def __init__(self, bankroll_cents:int, fees, tau_open=3, tau_close=1, slippage=1, max_spread=3, alpha=0.25):
        self.fees = fees
        self.tau_open = tau_open
        self.tau_close = tau_close
        self.slippage = slippage
        self.max_spread = max_spread
        self.alpha = alpha
        self.risk = RiskManager(bankroll_cents=bankroll_cents)

    def on_minute(self, ctx):
        acts = []
        spread = ctx['spread_cents']
        if spread > self.max_spread:
            # allow only if edge covers extra spread implicitly via gate_entry
            pass
        yes_price = ctx['yes_ask']  # taker buy
        fee_cents = effective_yes_entry_cents(yes_price, self.slippage, self.fees, maker=False, contracts=1) - yes_price
        p_hat = ctx['p_pooled']  # or p_model
        y_eff = yes_price + self.slippage + fee_cents
        edge_cents = max(0.0, 100.0*p_hat - y_eff)

        if gate_entry(edge_cents, spread, self.slippage, fee_cents, tau_open_cents=self.tau_open):
            # Kelly sizing (bounded by risk manager)
            f = kelly_fraction_yes(p_hat, yes_price + self.slippage + fee_cents, alpha=self.alpha)
            contracts_budget = max(0, int(f * (self.risk.bankroll_cents/100.0) / (yes_price/100.0)))
            exposure_cents = contracts_budget * yes_price
            city = ctx['city']; day_key = str(ctx['event_date'])
            if contracts_budget > 0 and self.risk.can_open(city, day_key, 'long_yes', exposure_cents):
                self.risk.register_open(city, day_key, 'long_yes', ctx['ticker'], exposure_cents)
                acts.append({'action':'BUY','contracts':contracts_budget,'ticker':ctx['ticker']})

        # For exits pre‑close, evaluate gate_exit() and generate SELL intents if holding.
        return acts
```

> You can wire this into your existing backtester by adding a `--strategy model_kelly` option that routes minute bars through the strategy’s `on_minute()` and uses your `Portfolio` to execute with fees.

---

## Implementation plan (what to ask your agent to code next)

1. **Timing correctness**

   * Use **Kalshi API** `close_time` and `expiration_time` for each market to build the minute snapshot schedule and to define when trading is allowed. **Do not hardcode the day‑of vs day‑before rule**; markets vary. ([Kalshi API Documentation][2])

2. **Feature pipeline**

   * Build `features/engineering.py` to join 1‑min market candles with ffilled 5‑min weather, add the feature sets listed above, and output `X, y, groups (event_day)` and `feature_names`.
   * Include `minutes_to_close` (from `close_time`), **distance‑to‑bin**, and your momentum/volatility features.

3. **Baselines then CatBoost**

   * Implement **Ridge** (L2) and **Lasso** (ElasticNet with l1_ratio ~ 1.0) baselines with **Optuna 15 trials each window** (C, class_weight), then calibration (isotonic or Platt depending on N) using `models/logistic_models.py` + `models/calibration.py`.
   * Implement **CatBoostClassifier** (CPU) with **monotone constraints** on physics‑guided features and **Optuna 15 trials per window** using `models/catboost_model.py`.

4. **Opinion pooling (phase 2)**

   * Add log‑odds pooling `p_pooled = pool(p_model, p_market, w)` where **w** is tuned on the train window (min log‑loss). Keep this switchable (`--pooling on/off`).

5. **Unified backtester**

   * Extend `backtest.py` into a **strategy harness** with `BuyHoldStrategy` and `ModelKellyStrategy`.
   * Add **gating** (τ_open, τ_close, max_spread, slippage) and the **RiskManager** (10% / city‑day‑side; max 3 bins per city; replace weakest).

6. **Walk‑forward evaluation**

   * Configuration: `train_days=42`, `test_days=7`, `step_days=1`, `optuna_trials=15`.
   * Metrics: **Sharpe**, **max DD**, **turnover**, **fee spend**, **Brier**, **log‑loss**, **calibration curves**.
   * Save per‑window models, calibration params, Optuna best params.

---

## A few helper snippets

**Pulling market times (per series)**

```python
# data/kalshi_times.py
import requests, pandas as pd

def fetch_markets(series_ticker: str, api_base="https://api.elections.kalshi.com/trade-api/v2/markets"):
    params = {"series_ticker": series_ticker, "limit": 1000}
    j = requests.get(api_base, params=params, timeout=20).json()
    rows = []
    for m in j.get("markets", []):
        rows.append({
            "ticker": m["ticker"],
            "event_ticker": m["event_ticker"],
            "open_time": m["open_time"],
            "close_time": m["close_time"],
            "expiration_time": m["expiration_time"],
            "strike_type": m.get("strike_type"),
            "floor_strike": m.get("floor_strike"),
            "cap_strike": m.get("cap_strike")
        })
    return pd.DataFrame(rows)
```

**Distance‑to‑bin features**

```python
def add_bin_distance_feats(df):
    # expects temp_now column (°F), floor_strike, cap_strike, strike_type
    df["temp_to_floor"] = df["floor_strike"].where(df["floor_strike"].notna(), df["temp_now"]) - df["temp_now"]
    df["temp_to_cap"] = df["cap_strike"] - df["temp_now"]
    df["dist_to_interval"] = 0.0
    mask_between = df["strike_type"] == "between"
    df.loc[mask_between, "dist_to_interval"] = (
        df.loc[mask_between, ["temp_to_floor","temp_to_cap"]]
        .apply(lambda r: 0 if (r["temp_to_floor"]<=0<=r["temp_to_cap"]) else min(abs(r["temp_to_floor"]), abs(r["temp_to_cap"])), axis=1)
    )
    return df
```

**Monotone constraints mapping for CatBoost**

```python
feature_names = ["minutes_to_close","temp_now","temp_to_floor","temp_to_cap","dist_to_interval","yes_bid","yes_ask","spread_cents","mom_5m","mom_15m"]
monotone = {
    # For 'greater' bins (example): p_yes increases as temp_now↑, minutes_to_close↓, temp_to_floor↓
    "temp_now": +1,
    "minutes_to_close": -1,
    "temp_to_floor": -1,
    # Dist to interval is generally negative monotone for 'between'
    "dist_to_interval": -1
}
```

---

## Why the max spread & slippage defaults are sensible

* **Spread & microstructure**—On Kalshi, the maker–taker design & fees can distort pricing. Entering on wide spreads without a proven edge is a common source of negative expectancy (you pay half‑spread + slippage + fees with no compensation). Start strict (**3¢ max spread**; **1¢ slippage**). As your edge grows, conditionally allow wider spreads **only when edge_cents still exceeds the higher gate**. ([Karl Whelan][5])

---

## Final checklist for the agent

1. **Wire API times**: all feature snapshots and backtests must respect `close_time` per market (no trading after it). ([Kalshi API Documentation][2])
2. **Implement dataset builder**: minute snapshots, ffilled 5‑min weather, physics‑guided features, `minutes_to_close`, bin distances.
3. **Train**: Ridge (L2), Lasso (L1/elastic), then CatBoost (CPU), each with **Optuna 15 trials** per walk‑forward step; calibrate (isotonic if ≥1k, else Platt).
4. **Pooling**: add optional log‑odds pooling with market price.
5. **Strategy harness**: add `ModelKellyStrategy` with gating (τ_open=3¢, τ_close=1¢), slippage=1¢, max_spread=3¢; Risk caps (10% per city‑day‑side; 3 bins/city; replace weakest).
6. **Metrics & reports**: Sharpe, max DD, turnover, fees, calibration diagnostics, source‑of‑truth comparisons.

---

### Sources

* **Help Center:** Weather markets settle off NWS daily climate reports; typically next morning. ([Kalshi Help Center][1])
* **Rulebooks:** HIGHTEMP and DVTEMP contract terms (NWS source, trading/expiration definitions, LTT and expiration times). 
* **Trading hours:** Exchange 24/7 with maintenance windows; individual markets still governed by their `close_time`. ([Kalshi Help Center][3])
* **Fee formulas:** Kalshi fee schedule (maker/taker). ([Kalshi][4])
* **API fields:** `open_time`, `close_time`, `expiration_time`, strikes, etc., for markets. ([Kalshi API Documentation][2])
* **Microstructure context:** Maker–taker and fee interactions affect realized expectancy. ([Kalshi][6])

---


[1]: https://help.kalshi.com/markets/popular-markets/weather-markets?utm_source=chatgpt.com "Weather Markets"
[2]: https://docs.kalshi.com/api-reference/market/get-markets "Get Markets - API Documentation"
[3]: https://help.kalshi.com/trading/what-are-trading-hours "What are trading hours? | Kalshi Help Center"
[4]: https://kalshi.com/docs/kalshi-fee-schedule.pdf?utm_source=chatgpt.com "Fee Schedule for Oct 2025"
[5]: https://www.karlwhelan.com/Papers/Kalshi.pdf?utm_source=chatgpt.com "The Economics of the Kalshi Prediction Market"
[6]: https://news.kalshi.com/p/makers-and-takers?utm_source=chatgpt.com "Makers and Takers"


## Further Clarifications:


Thanks for the clarifications — here’s how I’d like to proceed for the **Ridge baseline** and overall structure.

---

### 1. Feature scope for Ridge baseline

Let’s do this in **two phases**:

1. **Phase 1 (baseline): minimal but strong feature set**
   Use ~10–15 features to validate the full pipeline (data, training, calibration, backtest, risk manager):

   * Market:

     * `yes_mid`, `yes_bid`, `yes_ask`, `spread_cents`
     * `minutes_to_close`
   * Weather (from 5-min VC, ffilled ≤4m):

     * `temp_now`
     * `temp_to_floor` / `temp_to_cap` (distance-to-bracket)
   * Calendar:

     * `hour_of_day_local`
     * `day_of_week`

2. **Phase 2 (expanded): full feature set**
   After the minimal Ridge model produces reasonable Sharpe backtests end-to-end, expand to the richer set:

   * Rolling returns & std (1/5/15/30 min)
   * Volume/volatility features
   * Weather slopes / rolling max/min (30/60/120 min)
   * Diurnal (mins since sunrise / to sunset, etc.)

**So: start minimal for Ridge to validate the machinery, but design the feature code in a way that we can easily switch on the richer feature set later.**

---

### 2. Bracket handling

Let’s keep it **simple and structured**:

* **Train separate models per bracket *type***, not a single model for all types:

  * One Ridge model for **`greater`** bins (≥ floor).
  * One Ridge model for **`less`** bins (< cap).
  * One Ridge model for **`between`** bins ([floor, cap]).
* Within each type, include the actual **numeric floor/cap (or distance-to-floor/cap)** as features, so the model can generalize across different thresholds for that type.

Reasons:

* Behavior is qualitatively different between “≥ B”, “< B”, and “[A, B]” markets.
* This also plays nicely with the later **monotone constraints** on CatBoost (distance-to-threshold features behave differently per type).
* One big “all types mixed” model is harder to reason about and harder to debug early.

So: **separate models per type for Ridge; later we mirror the same pattern for CatBoost.**

---

### 3. Data validation first

Yes, absolutely — do a **quick data validation pass before you wire the full ML pipeline.** Specifically:

* Confirm we have **1-minute candles** for Chicago in the DB (for the chosen date range).
* Confirm we have **5-minute weather** data (Visual Crossing) aligned on timestamps and that **ffill ≤4 minutes** works as expected.
* Confirm that `close_time` and `expiration_time` are present and non-null for each market.
* Validate **time zones**:

  * Market times in UTC.
  * Mapping to local time (`America/Chicago` for Chicago) is correct.
  * `date_local` for settlement matches the CLI/GHCND day definition.

You can do this with a **simple script** (no modeling yet), and log:

* Counts of rows per day.
* Sample minute windows for a couple of days.
* A few `close_time` vs `minutes_to_close` examples.

Aim for ~1 hour on this; it will save a lot of debugging later.

---

### 4. Output artifacts per walk-forward window

For each **walk-forward train/test window**, please save:

1. **Model artifacts**

   * Ridge model: pickled `CalibratedClassifierCV` object (with underlying ridge).
   * Lasso model: same structure.
   * CatBoost model: CatBoost model + calibrated wrapper.

2. **Optuna artefacts**

   * `study.best_params` and at least **best value** (log-loss).
   * You don’t need full study blobs at first, but storing `best_params` and log-loss in a JSON/YAML per window is very useful.

3. **Predictions & features (test window)**

   * Per minute per market: CSV/Parquet with columns like
     `timestamp, city, ticker, strike_type, floor_strike, cap_strike, p_model, p_market, p_pooled(optional), yes_bid, yes_ask, spread_cents, minutes_to_close, tmax_final, true_label`.

4. **Backtest summaries**

   * JSON with: net P&L, Sharpe, max drawdown, turnover, fee spend, number of trades, etc.

These become your **audit trail** and allow us to revisit any window without retraining.

---

### 5. Backtest integration

Use **one unified backtester** with strategy plugins:

* Keep `run_backtest.py` as the main entrypoint.
* Add a `--strategy` flag with options:

  * `--strategy buyhold` → current simple buy-and-hold implementation.
  * `--strategy model_kelly` → model-driven, Kelly-sized strategy.

Under the hood:

* Create a `Strategy` interface (`backtest/strategy.py`) and move buy-and-hold logic into `BuyHoldStrategy`.
* Add `ModelKellyStrategy` that:

  * Uses the model probabilities.
  * Applies the **gate_entry/gate_exit** logic.
  * Applies the **RiskManager** (10% per city-day-side, max 3 bins per city).

This keeps code DRY and lets us reuse all the portfolio / summary logic.

---

### 6. Train/test splitting & Optuna frequency (critical)

You’re 100% right: **we do NOT want to run Optuna “per minute.”** That would be insane.

Instead:

* For backtesting, use **walk-forward by day**:

  * Example:

    * Train window: 42 days of minute snapshots.
    * Test window: next 7 days of minute snapshots.
    * Step forward by 7 days (or 1 day if you want high resolution).
  * For each train window **once**, run Optuna (≈15 trials) to tune Ridge/Lasso/CatBoost hyperparams; then:

    * Fit model on the train window.
    * Evaluate over the whole 7-day test window — no further tuning inside the test days.

* For live / shadow mode, **reuse the latest tuned parameters** for the rest of the day; you can retune:

  * Once per day (overnight), or
  * Once every few hours if you want, but not per minute.

So: **Optuna per window (train segment)**, not per minute.

---

## Short answer to agent (ready to paste)

Here’s a concise response you can send directly:

---

**Answers to your clarifying questions:**

1. **Feature scope for Ridge baseline**
   Start **minimal** to validate the full pipeline, then expand:

   * Phase 1 (baseline Ridge): ~10–15 strong features:

     * Market: mid, bid, ask, spread, minutes-to-close
     * Weather: current temp, distance-to-floor/cap
     * Time: hour-of-day, day-of-week
   * Phase 2: add full microstructure (rolls, momentum, volatility) and richer weather/diurnal features once the baseline end-to-end flow is working.

2. **Bracket handling**
   Train **separate Ridge models per bracket type**, not one unified model:

   * One model for `greater` bins, one for `less`, one for `between`.
   * Within each type, include floor/cap (or distance-to-threshold) as features so the model can generalize across brackets of that type.

3. **Data validation script**
   Yes, please do a quick validation pass **before** building the Ridge pipeline:

   * Verify 1-minute candles for Chicago exist and look sane for the chosen date range.
   * Verify 5-minute weather data exists and joins correctly via ffill ≤4 minutes.
   * Verify `close_time` and `expiration_time` are present and correct from the `markets` table.
   * Verify timezone handling: UTC storage, correct conversion to local (`America/Chicago`) and `date_local`.

4. **Artifacts per walk-forward window**
   Save all of the following for each train/test window:

   * Pickled calibrated model objects (Ridge, then later Lasso and CatBoost).
   * Optuna best params + best log-loss in a small JSON/YAML.
   * Predictions CSV/Parquet for the test window (per minute per market, with p_model, p_market, etc.).
   * Backtest summary JSON (P&L, Sharpe, max DD, fees, turnover).

5. **Backtest integration**
   Extend the **existing `run_backtest.py`** with a strategy plugin:

   * Add a `--strategy` flag (e.g., `buyhold` vs `model_kelly`).
   * Move current logic into a `BuyHoldStrategy`.
   * Create a `ModelKellyStrategy` that uses model probabilities + Kelly sizing + the risk manager.
     This keeps a single harness and avoids duplicate backtest code.

6. **Walk-forward and Optuna frequency**

   * For backtests, do **walk-forward by day**, not by minute: e.g., 42 train days → 7 test days, step forward.
   * Run **Optuna once per train window** (≈15 trials per model), then evaluate the trained model across the entire test window.
   * For live/shadow mode, reuse the latest tuned parameters for the rest of the day; we can retune once per day or per few hours if needed, but never per minute.

Once the Ridge baseline is trained, calibrated, and backtested with this setup, we’ll:

* Add Lasso with the same pattern.
* Then switch on CatBoost + monotone constraints + Optuna in the same walk-forward framework.
* After that, we can add opinion pooling and (optionally) the CatBoost quantile/TMAX route.
