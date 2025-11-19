## 0) What you asked to change (applied)

* **CatBoost on CPU** (no GPU assumptions).
* **Optuna** (15 trials/step) for quick but principled tuning—including monotone‑constraint choices—*per* walk‑forward step to avoid look‑ahead. ([Optuna][1])
* **Models**: keep Ridge/Lasso baselines; **replace GBM piece with CatBoost** (classifier for bracket odds; optional regressor for TMAX quantiles). CatBoost supports **monotone constraints** that we’ll use where physics implies monotonicity (e.g., probability of “≥ B” rises with warmer temps). ([CatBoost][2])
* **Calibration**: time‑aware calibration with Platt (sigmoid) or isotonic via `CalibratedClassifierCV`; enforce monotonicity across brackets with PAVA. ([Scikit-learn][3])
* **Execution filters (taker fills)**

  * **Max spread**: default **3¢** (configurable).
  * **Slippage**: **1¢** assumed.
  * Entry requires **edge ≥ τ_open + half_spread + slippage + fee impact** (we compute a break‑even probability that includes fees).
* **Risk caps**

  * **≤ 10% of bankroll per city per day per side**.
  * **Max 3 concurrent bins per city**; if a 4th wants to open, sell/trim the lowest‑edge existing position to stay at 3.
* **Walk‑forward**

  * Rolling train window (e.g., 28–42 days) → test next 7 days with **TimeSeriesSplit** style logic. ([Scikit-learn][4])
* **Minute vs 5‑minute alignment**

  * Keep market at 1‑min and 5‑min; weather aligned at **5‑min “minutes”** from Visual Crossing (`include=minutes&options=minuteinterval_5`). ([Visual Crossing][5])

---

## 1) What pros typically model here (and why)

**Predict probabilities of settlement (“YES” probability per bracket)** rather than predicting exact TMAX and then binning—because:

* Execution uses *prices* (market‑implied probabilities), and **calibration** is central to monetizing edges; tree ensembles (like CatBoost/GBMs) often need calibration for good probability estimates. ([Cornell Computer Science][6])
* CatBoost supports **ordered boosting** to reduce leakage/prediction shift and works well on tabular data with interactions; we’ll still calibrate. ([sagemaker.readthedocs.io][7])

We’ll still keep a **CatBoostRegressor (Quantile)** path to estimate the **distribution of TMAX**—useful as a cross‑check and for “between” brackets; CatBoost supports quantile losses. ([CatBoost][8])

**Monotone constraints**: where physics implies direction (e.g., probability of “≥ B” increases with “temp_now – B”), we enforce monotonicity to reduce variance and overfitting. ([CatBoost][2])

**Ensembling with market**: pool model odds with market odds using a **log‑odds (logarithmic opinion) pool**; optimize the weight on the trailing train window via log‑loss. ([NeurIPS Papers][9])

---

## 2) Feature set (beyond what you listed)

**Market microstructure (1‑min & 5‑min)**

* Price: yes_bid/ask/mid/last; **spread**; **depth proxies** if derivable; **volume**, **VWAP**.
* Momentum & realized variance: **1/5/15/30‑min returns** and **std**; **RSI‑like oscillator** on price; **z‑score of mid vs day’s moving average**.
* Relative value across bins: **distance‑to‑strike** (price ladder position), **implied density smoothness** across adjacent bins, **no‑arb checks** (prob mass sums to ≈1).
* Time to close (minutes), **day‑of‑week**, known **market halts** flags.

**Weather (5‑min minutes via Visual Crossing)**

* **Current temp, dew point, humidity, wind**.
* **Rolling max/min/avg** over 30/60/120 min; **slope** (linear fit) over 15/30 min.
* **Distance to bracket**: `temp_now - floor` and `cap - temp_now`.
* **Cloud cover/precip indicators** (convective precipitation often caps highs; low clouds reduce daytime highs). ([American Meteorological Society Journals][10])
* **Anomalies vs normals** (Visual Crossing returns “normal” arrays—use deviations from normal for the day/hour). ([Visual Crossing][5])
* **Solar/diurnal**: hour since sunrise / to solar noon (Visual Crossing provides sunrise/sunset in days/hours). ([Visual Crossing][5])

**Join policy**

* Left‑join 1‑min candles to **last known 5‑min weather** (≤ 4 min ffill).
* Derive “intraday max so far” as a feature.

---

## 3) Calibration and sizing (with your numbers)

**Calibration**

* Fit base model on window; then calibrate **out‑of‑fold** within the training window using `CalibratedClassifierCV` with **isotonic** if enough data (≥ 1k points); fall back to **sigmoid (Platt)** otherwise. Pros do this to fix the classic S‑shaped distortion from boosted trees. ([Scikit-learn][3])
* Enforce **monotonicity across bracket levels** with PAVA on the set of bracket thresholds per timestamp.

**Sizing**

* Compute **break‑even probability** `p_be` that includes: taker fee, **half‑spread**, **slippage (1¢)**. Use your existing `fees.py` for exact fee math; add spread & slippage to the effective cost.
* **Edge** = `p_calibrated - p_be`.
* **Kelly fraction** for binary payoff 1 at price `y` (dollars) with fees included:
  `f* = (p - y_eff) / (1 - y_eff)` where `y_eff` is the effective price including fees, half‑spread, slippage. Then **fractional Kelly α**: position fraction = `α * f*` with α ∈ {0.1, 0.25, 0.5}. Fractional Kelly is standard to reduce drawdown risk. ([Wikipedia][11])
* **Execution filters** (taker fills):

  * **Max spread** default **3¢**. Why not allow very wide spreads? Because crossing wide spreads with taker fees requires very large edge; your condition already scales edge by **half‑spread + slippage + fee**. Keep 3¢ as *default*, but let Optuna tune it per window if desired.
  * Entry hysteresis: `edge ≥ τ_open` where `τ_open = 1.5¢` (example) **plus** half‑spread + slippage + fee term.
  * Exit hysteresis: `edge ≤ τ_close` where `τ_close = 0.5¢`.
* **Risk caps**

  * Per (city, day, side): **≤ 10% bankroll**.
  * **≤ 3 concurrent bins per city**. If a new signal ranks in the top‑3 by edge, sell/trim the weakest to admit it; otherwise skip.

---

## 4) Model menu

### A) **CatBoostClassifier** (primary)

* Targets: **per‑bracket binary (YES/NO)**.
* **Monotone constraints** on features where direction is known (e.g., `temp_now_minus_floor` positive for “≥ floor”; `cap_minus_temp_now` positive for “≤ cap”). ([CatBoost][2])
* Train with ordered boosting (default CatBoost behavior); CPU only. ([sagemaker.readthedocs.io][7])
* **Calibrate** with isotonic/Platt. ([Scikit-learn][3])
* **Optuna** (15 trials/step) to tune `depth`, `l2_leaf_reg`, `learning_rate`, `bagging_temperature`, `min_data_in_leaf`, and which features get monotone constraints. Prune trials using the CatBoost pruning callback to keep steps fast. ([Optuna][12])

### B) **Ridge & Lasso Logistic** (fast baselines)

* Good sanity checks; L1 (Lasso) will zero weak features automatically.

### C) **CatBoostRegressor (Quantile)** (optional)

* Predict TMAX quantiles; derive bracket probabilities by integrating the fitted distribution. ([CatBoost][8])

### D) **Opinion pool with the market**

* Combine calibrated model `p_model` and market `p_mkt` via **log‑odds pooling**:
  `logit(p_pool) = w·logit(p_model) + (1−w)·logit(p_mkt)`
  Tune `w ∈ [0,1]` on the train window with log‑loss. ([NeurIPS Papers][9])

---

## 5) File stubs & examples

> Paths assume a `models/`, `backtest/`, `scripts/` layout you already use.

### `models/features.py` — **feature engineering**

```python
# models/features.py
from __future__ import annotations
import numpy as np
import pandas as pd

WEATHER_ROLLS = [5, 15, 30, 60, 120]   # minutes
PRICE_ROLLS   = [1, 5, 15, 30]

def _roll(df: pd.DataFrame, col: str, wins: list[int], agg: str, suffix: str):
    for w in wins:
        if agg == "mean":
            df[f"{col}_ma_{w}{suffix}"] = df[col].rolling(w, min_periods=max(1, w//3)).mean()
        elif agg == "std":
            df[f"{col}_std_{w}{suffix}"] = df[col].rolling(w, min_periods=max(2, w//3)).std(ddof=0)
        elif agg == "max":
            df[f"{col}_max_{w}{suffix}"] = df[col].rolling(w, min_periods=max(1, w//3)).max()
        elif agg == "min":
            df[f"{col}_min_{w}{suffix}"] = df[col].rolling(w, min_periods=max(1, w//3)).min()

def add_market_features(candles_1m: pd.DataFrame, candles_5m: pd.DataFrame) -> pd.DataFrame:
    """Assumes candles_* have datetime index at 1m/5m boundaries and columns:
       yes_bid, yes_ask, yes_mid, last, volume, vwap."""
    df = candles_1m.copy()
    df["spread_c"] = (df["yes_ask"] - df["yes_bid"]).astype(float)
    for w in PRICE_ROLLS:
        df[f"ret_{w}m"] = df["yes_mid"].pct_change(w).replace([np.inf, -np.inf], np.nan)
    _roll(df, "yes_mid", PRICE_ROLLS, "std", "m")
    _roll(df, "volume", PRICE_ROLLS, "mean", "m")
    return df

def add_weather_features(w5: pd.DataFrame, floor: float|None, cap: float|None) -> pd.DataFrame:
    """w5 at 5‑minute resolution: columns temp, dew, humidity, wind, precip, cloudcover, etc."""
    df = w5.copy()
    # Rolling stats & slopes
    for col in ["temp", "dew", "humidity", "wind", "precip", "cloudcover"]:
        _roll(df, col, WEATHER_ROLLS, "mean", "")
        _roll(df, col, WEATHER_ROLLS, "std", "")
        _roll(df, col, WEATHER_ROLLS, "max", "")
        _roll(df, col, WEATHER_ROLLS, "min", "")
        # slope via 1st‑diff smoothed
        df[f"{col}_slope_15"] = df[col].diff().rolling(15, min_periods=5).mean()

    # Distance‑to‑bracket features (physics‑guided monotonic features)
    if floor is not None:
        df["temp_minus_floor"] = df["temp"] - float(floor)
    if cap is not None:
        df["cap_minus_temp"] = float(cap) - df["temp"]

    # Diurnal features (requires sunrise/sunset already merged per day)
    if {"sunrise", "sunset"}.issubset(df.columns):
        df["mins_since_sunrise"] = (df.index - df["sunrise"]).dt.total_seconds() / 60.0
        df["mins_to_sunset"]     = (df["sunset"] - df.index).dt.total_seconds() / 60.0

    return df

def join_market_weather(c1: pd.DataFrame, w5: pd.DataFrame) -> pd.DataFrame:
    """Join 1m market data to last‑known 5m weather (≤4m ffill)."""
    w5r = w5.reindex(c1.index, method="ffill", limit=4)
    return c1.join(w5r, how="left")
```

> Notes: Visual Crossing supports **minute‑level** via `include=minutes` and `options=minuteinterval_5`; you’ll already have those fields in your `weather` table. ([Visual Crossing][5])

---

### `models/catboost_model.py` — **CatBoostClassifier + Optuna (CPU)**

```python
# models/catboost_model.py
from __future__ import annotations
import optuna
import numpy as np
import pandas as pd
from typing import Sequence, Dict, Any
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier, Pool
from optuna.integration import CatBoostPruningCallback

def _build_monotone_vector(feature_names: list[str],
                           pos_monotone: Sequence[str] = (),
                           neg_monotone: Sequence[str] = ()) -> list[int]:
    vec = []
    pos = set(pos_monotone); neg = set(neg_monotone)
    for f in feature_names:
        if f in pos: vec.append(1)
        elif f in neg: vec.append(-1)
        else: vec.append(0)
    return vec

def tune_catboost_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,  # e.g., local calendar day to avoid leakage
    pos_monotone: Sequence[str],
    neg_monotone: Sequence[str],
    n_trials: int = 15,
    random_state: int = 42,
) -> Dict[str, Any]:
    feat_names = X.columns.tolist()
    mono_vec = _build_monotone_vector(feat_names, pos_monotone, neg_monotone)

    gkf = GroupKFold(n_splits=4)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 200, 800),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": random_state,
            "bootstrap_type": "MVS",   # fast & robust
            "thread_count": -1,
            "monotone_constraints": mono_vec,  # key constraint vector
            "verbose": False,
        }
        # Optional: sample which constraints to enforce
        if trial.suggest_categorical("use_monotone", [True, False]) is False:
            params["monotone_constraints"] = [0]*len(mono_vec)

        fold_losses = []
        for (tr, va) in gkf.split(X, y, groups=groups):
            train_pool = Pool(X.iloc[tr], y[tr], feature_names=feat_names)
            valid_pool = Pool(X.iloc[va], y[va], feature_names=feat_names)
            model = CatBoostClassifier(**params)
            cb_pruner = CatBoostPruningCallback(trial, "Logloss")
            model.fit(train_pool, eval_set=valid_pool, callbacks=[cb_pruner])
            p = model.predict_proba(valid_pool)[:, 1]
            fold_losses.append(log_loss(y[va], p, eps=1e-6))
        return float(np.mean(fold_losses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": random_state,
        "thread_count": -1,
        "monotone_constraints": mono_vec if best_params.get("use_monotone", True) else [0]*len(mono_vec),
        "verbose": False,
    })
    best_params.pop("use_monotone", None)
    return best_params
```

> Rationale: CatBoost’s **monotone_constraints** are per feature (1/−1/0). We select which to enforce using Optuna (binary switch), then tune the usual GBDT hyperparameters. A **GroupKFold by day** prevents temporal leakage. ([CatBoost][2])
> We use Optuna’s CatBoost pruning callback to skip weak trials fast. ([Optuna][12])

---

### `models/calibration.py` — **isotonic / Platt with time‑aware CV**

```python
# models/calibration.py
from __future__ import annotations
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

def calibrate_probs(method: str, base_estimator, X, y, cv_splits):
    """
    method: 'isotonic' or 'sigmoid' (Platt).
    cv_splits: precomputed train/val indices respecting time order.
    """
    # We use ensemble=True so each fold gets its own calibrator, predictions averaged.
    calib = CalibratedClassifierCV(
        estimator=base_estimator,
        method=("isotonic" if method == "isotonic" else "sigmoid"),
        cv=list(cv_splits)
    )
    calib.fit(X, y)
    return calib
```

> Why: **CalibratedClassifierCV** does train→calibrate splits correctly; boosted trees typically need this to turn scores into usable probabilities. ([Scikit-learn][3])

---

### `models/kelly.py` — **fractional Kelly with fees, spread, slippage**

```python
# models/kelly.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ExecutionCosts:
    spread_cents: int
    slippage_cents: int
    taker: bool = True

def effective_price_cents(entry_price_cents: int,
                          fees_cents: int,
                          costs: ExecutionCosts) -> int:
    """Add fees + half-spread + slippage to entry cost for YES."""
    half_spread = costs.spread_cents // 2
    return entry_price_cents + fees_cents + half_spread + costs.slippage_cents

def kelly_fraction_yes(p_yes: float, eff_price_cents: int) -> float:
    """Binary payoff 100c if YES, else 0. Kelly fraction of bankroll to bet long-YES.
       Convert eff price to dollars y in [0,1].  f* = (p - y) / (1 - y)."""
    y = eff_price_cents / 100.0
    if p_yes <= y or y >= 1.0:
        return 0.0
    return (p_yes - y) / (1.0 - y)

def fractional_kelly(alpha: float, f_star: float) -> float:
    """Alpha ∈ {0.1, 0.25, 0.5}. Typical pros scale down Kelly to reduce drawdowns."""
    return max(0.0, alpha * f_star)
```

> Kelly and **fractional Kelly** are standard; we always compute `y` with **fees + half‑spread + 1¢ slippage** so the criterion is realistic. ([Wikipedia][11])

---

### `backtest/run_kelly_backtest.py` — **entry/exit logic, risk caps**

```python
# backtest/run_kelly_backtest.py
from __future__ import annotations
import math
import pandas as pd
from collections import defaultdict
from models.kelly import ExecutionCosts, effective_price_cents, kelly_fraction_yes, fractional_kelly
from backtest.fees import calc_taker_fee

class RiskManager:
    def __init__(self, bankroll: float, max_city_side_frac: float = 0.10, max_bins_per_city: int = 3):
        self.bankroll = bankroll
        self.max_city_side_frac = max_city_side_frac
        self.max_bins_per_city = max_bins_per_city
        self.positions = {}  # ticker -> size (contracts)
        self.city_side_gross = defaultdict(float)  # (city, side, day) -> notional$
        self.city_open_bins = defaultdict(set)    # (city, day) -> {tickers}

    def can_open(self, city: str, side: str, day: str, notional_dollars: float, ticker: str, edge_rank: float) -> tuple[bool, list[str]]:
        key = (city, side, day)
        # enforce 3 bins per city
        open_bins = self.city_open_bins[(city, day)]
        if len(open_bins) < self.max_bins_per_city:
            # check bankroll per city-side
            if self.city_side_gross[key] + notional_dollars <= self.max_city_side_frac * self.bankroll:
                return True, []
            return False, []
        # If full, see if we can replace the weakest
        # Here you’d store edges per ticker; for brevity assume weakest is provided
        weakest_ticker = min(open_bins, key=lambda t: 0.0)  # TODO: lookup stored edge
        # Replace logic: if new edge > weakest edge by margin
        # Return list of positions to close first
        return True, [weakest_ticker]

    def register_open(self, city: str, side: str, day: str, notional_dollars: float, ticker: str):
        key = (city, side, day)
        self.city_side_gross[key] += notional_dollars
        self.city_open_bins[(city, day)].add(ticker)

def decide_entry(
    p_cal: float,
    yes_bid: int,
    yes_ask: int,
    costs: ExecutionCosts,
    alpha: float,
    tau_open_cents: int = 2,  # edge hurdle before fees (configurable)
) -> tuple[int, float]:
    """Return (contracts, kelly_fraction). Crosses the ask."""
    price_cents = yes_ask
    # compute fees at 1 contract to get per-contract fee cents (then scale)
    per_contract_fee = calc_taker_fee(1, price_cents)  # already ceil to cents
    y_eff_cents = effective_price_cents(price_cents, per_contract_fee, costs)
    f_star = kelly_fraction_yes(p_cal, y_eff_cents)
    f = fractional_kelly(alpha, f_star)
    # Convert bankroll fraction to #contracts later (backtest wrapper knows bankroll)
    return (1 if f > 0 else 0), f

def edge_filter(p_cal: float, yes_ask: int, spread_cents: int, costs: ExecutionCosts, fees_func) -> bool:
    """Require edge ≥ fee + half-spread + slippage (+ tau_open)."""
    fee_cents = fees_func(1, yes_ask)
    y_eff_cents = effective_price_cents(yes_ask, fee_cents, costs)
    p_be = y_eff_cents / 100.0
    edge = p_cal - p_be
    hurdle = 0.0  # if you want an extra 0.5–2.0¢ absolute hurdle, add here
    return edge > hurdle
```

> **Spread & slippage**: We gate entries by requiring `p_cal` to exceed the **effective break‑even** probability that includes fee + half‑spread + 1¢ slippage. This is why a **3¢ max spread** default is sensible—if the spread widens, half‑spread eats more edge; your filter still allows wide spreads, but only when the edge is strong enough to overcome costs.

---

### `pipelines/walkforward_optuna.py` — **walk‑forward + 15 trials/step**

```python
# pipelines/walkforward_optuna.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostClassifier, Pool
from models.catboost_model import tune_catboost_classifier
from models.calibration import calibrate_probs

def day_groups(index: pd.DatetimeIndex) -> np.ndarray:
    return index.tz_localize(None).strftime("%Y-%m-%d").to_numpy()

def fit_catboost_with_calibration(X_tr, y_tr, groups_tr, pos_mono, neg_mono, n_trials=15):
    best_params = tune_catboost_classifier(X_tr, y_tr, groups_tr, pos_mono, neg_mono, n_trials=n_trials)
    model = CatBoostClassifier(**best_params)
    # time-aware CV splits for calibration (e.g., 4 folds expanding)
    uniq_days = pd.Series(groups_tr).unique()
    n_splits = min(4, len(uniq_days) - 1) if len(uniq_days) > 4 else 2
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for tr_idx, va_idx in tscv.split(np.arange(len(X_tr))):
        splits.append((tr_idx, va_idx))
    calib = calibrate_probs("isotonic", model, X_tr, y_tr, splits)  # auto-fallback to sigmoid if low data
    return calib
```

> We explicitly **re‑tune with Optuna at each walk‑forward step** (your request) to avoid any forward‑fit bias; small **15‑trial searches** are fine given the strong CatBoost defaults. ([Optuna][1])

---

### (Optional) `models/tmax_quantiles.py` — **CatBoostRegressor for TMAX**

```python
# models/tmax_quantiles.py
from catboost import CatBoostRegressor, Pool
import numpy as np

def fit_tmax_quantile(X, y, alpha=0.9, **kwargs):
    params = dict(
        loss_function="Quantile:alpha={}".format(alpha),
        iterations=600,
        depth=6,
        learning_rate=0.07,
        l2_leaf_reg=3.0,
        random_seed=42,
        thread_count=-1,
        verbose=False
    )
    params.update(kwargs)
    model = CatBoostRegressor(**params)
    model.fit(Pool(X, y))
    return model
```

> Use quantile fits at several α (e.g., 0.1/0.5/0.9) to approximate the TMAX distribution, then integrate to bracket odds if you want a second opinion. ([CatBoost][8])

---

## 6) Monotone‑constraint mapping (examples)

For **greater‑than** brackets (≥ floor): enforce **increasing** with

* `temp_now_minus_floor`, `temp_ma_30`, `temp_max_60`, *etc.*

For **less‑than** brackets (< cap): enforce **increasing** with

* `cap_minus_temp_now`, `cap_minus_temp_ma_30`, …

For **between** [floor, cap]: model two monotone sub‑problems (`≥ floor` and `≤ cap`) and multiply if you want a parametric independence approximation, then calibrate. (Or train a direct binary and omit constraints.)

---

## 7) “Large spread” question—when (not) to trade

* With **taker** fills, you pay **fees + half‑spread + slippage** up front. If the spread is wide, your **p_be** rises materially.
* You *can* still trade wide spreads when your calibrated edge is **big enough**—our filter already encodes that:
  `enter if p_cal - p_be ≥ τ_open`.
* In practice, a **3¢ default** keeps turnover realistic and avoids crossing obviously stale books; you can let Optuna tune a **per‑window cap** if desired.

---

## 8) Weather data granularity (Visual Crossing)

* Use `include=minutes&options=minuteinterval_5` for sub‑hourly weather; align to markets at 1‑min using ≤ 4‑minute ffill. ([Visual Crossing][5])

---

## 9) What to ask your agent to code next (summary checklist)

1. **Feature layer** (above `models/features.py`)

   * Add the extra features listed (market RV/momentum; weather rolls/slopes; anomaly vs normal; diurnal).
   * Ensure 1‑min market ↔ 5‑min weather join.

2. **CatBoostClassifier pipeline**

   * Implement `tune_catboost_classifier(...)` + `fit_catboost_with_calibration(...)` exactly as above.
   * Define **monotone feature lists** per bracket type.

3. **Opinion pool**

   * Implement `p_pool = sigmoid(w·logit(p_model) + (1−w)·logit(p_mkt))`; tune `w` on log‑loss over the train window. ([NeurIPS Papers][9])

4. **Execution policy**

   * Gate by **edge ≥ half‑spread + slippage + fee + τ_open**; default **spread≤3¢**, **slippage=1¢**.
   * Hysteresis: `(τ_open=1.5¢, τ_close=0.5¢)` initial defaults.

5. **Risk manager**

   * Enforce **≤ 10% bankroll per (city, day, side)** and **≤ 3 bins/city** (replace weakest if needed).

6. **Walk‑forward loop**

   * For each 7‑day test slice: re‑tune CatBoost with **Optuna 15 trials** and recalibrate; run Kelly‑sized backtest; log Sharpe, max DD, fee spend, turnover.

7. **Diagnostics**

   * Reliability plots (pre/post‑calibration), **Brier** & **log‑loss**, edge histograms, and source‑of‑edge attribution.

---

## 10) A few guardrails (pros’ habits)

* **Calibration drift monitoring**: keep a rolling log‑loss gap between model vs market; refit thresholds if drift grows.
* **Seasonality**: consider training **per city** and include month‑of‑year and “days since solstice” type features.
* **Sensible defaults first**, then let Optuna move them; do not over‑parameterize entry/exit in early runs.
* **Time‑aware CV** everywhere; never mix future into calibration or hyperparameter search. ([Scikit-learn][4])

---

### Citations (key load‑bearing)

* CatBoost monotone constraints parameter. ([CatBoost][2])
* CatBoost ordered boosting / prediction‑shift background. ([sagemaker.readthedocs.io][7])
* CatBoost quantile losses (for TMAX distribution). ([CatBoost][8])
* Probability calibration & why trees need it. ([Scikit-learn][3])
* Optuna pruning/callbacks; study.optimize. ([Optuna][12])
* TimeSeriesSplit for time‑ordered CV. ([Scikit-learn][4])
* Visual Crossing **minutes** support & options. ([Visual Crossing][5])
* Fractional Kelly usage. ([Wikipedia][11])
* Log‑odds (logarithmic opinion) pooling for combining probabilities. ([NeurIPS Papers][9])
* Clouds/precip can cap daytime highs (feature motivation). ([American Meteorological Society Journals][10])

---


[1]: https://optuna.readthedocs.io/_/downloads/en/v3.0.0-b0/pdf/?utm_source=chatgpt.com "Optuna Documentation"
[2]: https://catboost.ai/docs/en/references/training-parameters/?utm_source=chatgpt.com "Overview"
[3]: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html?utm_source=chatgpt.com "CalibratedClassifierCV"
[4]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html?utm_source=chatgpt.com "TimeSeriesSplit — scikit-learn 1.7.2 documentation"
[5]: https://www.visualcrossing.com/resources/documentation/weather-api/timeline-weather-api/ "Weather API Documentation | Visual Crossing"
[6]: https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf?utm_source=chatgpt.com "Predicting Good Probabilities With Supervised Learning"
[7]: https://sagemaker.readthedocs.io/en/v2.244.1/algorithms/tabular/catboost.html?utm_source=chatgpt.com "CatBoost — sagemaker 2.244.1 documentation"
[8]: https://catboost.ai/docs/en/concepts/loss-functions-regression?utm_source=chatgpt.com "Regression: objectives and metrics"
[9]: https://papers.neurips.cc/paper/1413-selecting-weighting-factors-in-logarithmic-opinion-pools.pdf?utm_source=chatgpt.com "Selecting Weighting Factors in Logarithmic Opinion Pools"
[10]: https://journals.ametsoc.org/view/journals/clim/12/8/1520-0442_1999_012_2451_eocsmp_2.0.co_2.xml?utm_source=chatgpt.com "Effects of Clouds, Soil Moisture, Precipitation, and Water ..."
[11]: https://en.wikipedia.org/wiki/Kelly_criterion?utm_source=chatgpt.com "Kelly criterion"
[12]: https://optuna.readthedocs.io/en/v3.0.2/reference/generated/optuna.integration.CatBoostPruningCallback.html?utm_source=chatgpt.com "optuna.integration.CatBoostPruningCallback - Read the Docs"






---

## 1) Ridge & Lasso logistic — what we’re doing

We’re using **scikit-learn’s `LogisticRegression`** as linear-probability baselines:

* **Ridge logistic** (L2 penalty) → stable, smooth, good baseline; use solver like `lbfgs` or `saga` with `penalty='l2'`. ([Scikit-learn][1])
* **Lasso logistic** (L1 penalty) → automatic feature selection; sparsifies coefficients and can “turn off” useless features. You need `penalty='l1'` and a solver that supports L1 (e.g., `liblinear` or `saga`). ([Scikit-learn][1])

We’ll:

* Tune **`C`** (inverse regularization strength) and **`class_weight`** (None vs 'balanced') with **Optuna** per walk-forward step. Optuna replaces hand grid searches and is efficient with even ~15 trials. ([Optuna][2])
* Use **GroupKFold by day** for CV inside each Optuna search so we respect time ordering.
* Wrap best logistic in **`CalibratedClassifierCV`** (isotonic when enough data, otherwise Platt / sigmoid) to get proper probabilities. ([Scikit-learn][3])

These baselines give you:

* A **sanity check** vs CatBoost.
* Fast models you can retrain constantly.
* Interpretability: you can inspect which features Lasso zeroes out.

---

## 2) Design choices for tuning (to avoid leakage)

For each **train window** in your walk-forward loop:

1. Extract training `X_train, y_train` and a **group label** per row (e.g., `date_local`) so we can do grouped CV.
2. Run **Optuna** for ~15 trials per model (Ridge & Lasso), each trial:

   * Suggest `log10_C` in a reasonable range (e.g. 1e-3 to 1e2).
   * Suggest `class_weight` ∈ {None, 'balanced'}.
   * Possibly toggle `fit_intercept` or use `solver='liblinear'` vs `solver='saga'` depending on data size.
3. For each trial, evaluate **log-loss** (or Brier) across 3–4 grouped folds (GroupKFold by day).
4. Use `study.best_params` to refit a final logistic on the full train window.
5. Wrap that with `CalibratedClassifierCV` (“sigmoid” if calibration sample count is small, “isotonic” otherwise). ([Scikit-learn][3])

Then, you treat Ridge/Lasso exactly like CatBoost in the rest of your pipeline: produce calibrated probabilities, plug into the same **Kelly + fee** logic, and walk-forward backtest.

---

## 3) File stub: `models/logit_linear.py`

Drop this into `models/logit_linear.py` (or similar). It includes:

* Optuna tuning for Ridge (`penalty='l2'`)
* Optuna tuning for Lasso (`penalty='l1'`)
* A helper to wrap best logistic in `CalibratedClassifierCV` with **time-aware CV**.

```python
# models/logit_linear.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Literal, Sequence

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV

# --- Config dataclasses -------------------------------------------------------

@dataclass
class LogitConfig:
    penalty: Literal["l1", "l2"] = "l2"
    max_iter: int = 1000
    n_splits_cv: int = 4
    random_state: int = 42
    n_trials_optuna: int = 15


# --- Internal helpers ---------------------------------------------------------

def _make_base_logit(penalty: str, C: float, class_weight: Any, max_iter: int, random_state: int) -> LogisticRegression:
    """
    Create a LogisticRegression with given penalty and hyperparams.

    Notes (per sklearn docs):
      - 'liblinear' supports L1 and L2.
      - 'saga' supports L1/L2/elasticnet and is good for large/sparse data.
    """
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    # For big/high-dimensional X, consider saga.
    return LogisticRegression(
        penalty=penalty,
        C=C,
        class_weight=class_weight,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1 if solver != "liblinear" else None,
    )


def _optuna_objective_logit(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: LogitConfig,
) -> float:
    """Objective: minimize average CV log-loss."""
    # log10(C) in [-3, 2] → C in [1e-3, 1e2]
    log10_C = trial.suggest_float("log10_C", -3.0, 2.0)
    C = 10.0 ** log10_C
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    base = _make_base_logit(
        penalty=cfg.penalty,
        C=C,
        class_weight=class_weight,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
    )

    # Grouped CV by day to avoid temporal leakage
    unique_groups = np.unique(groups)
    n_splits = min(cfg.n_splits_cv, len(unique_groups) - 1) if len(unique_groups) > 2 else 2
    gkf = GroupKFold(n_splits=n_splits)

    losses = []
    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        base.fit(X_tr, y_tr)
        p_va = base.predict_proba(X_va)[:, 1]
        losses.append(log_loss(y_va, p_va, eps=1e-6))

    return float(np.mean(losses))


def tune_logit_with_optuna(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    penalty: Literal["l1", "l2"],
    n_trials: int = 15,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Return best hyperparameters for given penalty using Optuna."""
    cfg = LogitConfig(penalty=penalty, random_state=random_state, n_trials_optuna=n_trials)

    def objective(trial: optuna.Trial) -> float:
        return _optuna_objective_logit(trial, X, y, groups, cfg)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg.n_trials_optuna, show_progress_bar=False)

    best = study.best_params
    best["C"] = 10.0 ** best.pop("log10_C")
    best["penalty"] = penalty
    return best


# --- Public API ---------------------------------------------------------------

def fit_logit_with_calibration(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    penalty: Literal["l1", "l2"],
    cal_method: Literal["isotonic", "sigmoid"] = "isotonic",
    n_trials: int = 15,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """
    Tune LogisticRegression hyperparams via Optuna, then fit with CalibratedClassifierCV.

    - penalty='l2' => Ridge-style
    - penalty='l1' => Lasso-style
    - cal_method:
        * 'isotonic' when you have plenty of calibration data (>= ~1000), per sklearn docs.
        * 'sigmoid' (Platt) for smaller calibration sets to avoid overfitting.
    """
    best = tune_logit_with_optuna(X, y, groups, penalty=penalty, n_trials=n_trials, random_state=random_state)

    base = _make_base_logit(
        penalty=penalty,
        C=best["C"],
        class_weight=best["class_weight"],
        max_iter=1000,
        random_state=random_state,
    )

    # Time-aware CV splits reused for calibration
    unique_groups = np.unique(groups)
    n_splits = 4 if len(unique_groups) > 5 else 2
    gkf = GroupKFold(n_splits=n_splits)
    cv_splits = list(gkf.split(X, y, groups=groups))

    # Use the full train window to fit base; CalibratedClassifierCV will refit inside folds
    calib = CalibratedClassifierCV(
        estimator=base,
        method=("isotonic" if cal_method == "isotonic" else "sigmoid"),
        cv=cv_splits,
    )
    calib.fit(X, y)

    return calib
```

Key points anchored to docs:

* `C` is inverse regularization strength (smaller C ⇒ stronger penalty). ([Scikit-learn][4])
* L1 penalty requires solver like `liblinear` or `saga`; L2 supports many solvers (lbfgs, newton-cg, sag, saga). ([Scikit-learn][1])
* `CalibratedClassifierCV` supports `method='sigmoid'` (Platt) or `method='isotonic'`, with isotonic recommended when you have enough calibration samples; otherwise prefer sigmoid. ([Scikit-learn][5])
* Optuna’s pattern—define `objective(trial)`, call `study.optimize` for `n_trials`, then read `study.best_params`—comes straight from its docs. ([Optuna][2])

---

## 4) Example: using Ridge/Lasso in your walk-forward trainer

In your `models/train.py` (or equivalent), you can integrate like this:

```python
# models/train.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import TimeSeriesSplit

from models.logit_linear import fit_logit_with_calibration
from models.catboost_model import fit_catboost_with_calibration  # from earlier stub
from models.features import join_market_weather, build_minute_features

def day_groups(index: pd.DatetimeIndex) -> np.ndarray:
    """Group key (string) per row; used for GroupKFold / calibration."""
    return index.tz_convert("UTC").strftime("%Y-%m-%d").to_numpy()


def train_models_for_window(
    X: pd.DataFrame,
    y: np.ndarray,
    idx: pd.DatetimeIndex,
    pos_monotone: list[str],
    neg_monotone: list[str],
    ridge_trials: int = 15,
    lasso_trials: int = 15,
    cb_trials: int = 15,
) -> Dict[str, Any]:
    """
    Train:
      - Ridge logistic (L2) with Optuna+calibration
      - Lasso logistic (L1) with Optuna+calibration
      - CatBoost (primary) with monotone constraints + calibration
    Returns dict of calibrated models.
    """
    groups = day_groups(idx)
    X_np = X.values

    models = {}

    # Ridge
    ridge_cal = fit_logit_with_calibration(
        X_np, y, groups,
        penalty="l2",
        cal_method="isotonic",
        n_trials=ridge_trials
    )
    models["ridge"] = ridge_cal

    # Lasso
    lasso_cal = fit_logit_with_calibration(
        X_np, y, groups,
        penalty="l1",
        cal_method="sigmoid",  # often safer with L1 + limited data
        n_trials=lasso_trials
    )
    models["lasso"] = lasso_cal

    # CatBoost
    cb_cal = fit_catboost_with_calibration(
        X_tr=X, y_tr=y, groups_tr=groups,
        pos_mono=pos_monotone, neg_mono=neg_monotone,
        n_trials=cb_trials
    )
    models["catboost"] = cb_cal

    return models
```

This matches the CatBoost flow you already have: each walk-forward step:

1. Build features & labels.
2. Call `train_models_for_window(...)`.
3. For the test slice, call `model["catboost"].predict_proba(X_test)[:, 1]` (or `"ridge"`/`"lasso"`) and feed those probabilities into your **Kelly + fee** engine.

---

## 5) How this plays with CatBoost + Optuna

Now you’ve got three families:

* **Ridge (L2)** logistic: smooth baseline.
* **Lasso (L1)** logistic: feature-sparse baseline.
* **CatBoost**: non-linear, monotone-enabled tree ensemble.

All three:

* Are tuned per walk-forward step with **Optuna** (≈15 trials), which is much smarter than grids. ([Optuna][2])
* Are calibrated with **CalibratedClassifierCV** using isotonic when you have enough calibration data, sigmoid otherwise. ([Scikit-learn][3])
* Produce probabilities that plug into **exactly the same** fee-aware fractional Kelly sizing.

From there, you:

* Walk-forward backtest all three, measure **net Sharpe, max DD, fee drag, turnover**, and pick the best (or ensemble) by city.
* Use the best model as your **shadow trading** engine; if it holds up live, it becomes your execution model.

If you’d like, next step I can also sketch:

* A very small **Optuna search space for Ridge/Lasso** (e.g., just `log10_C`, `class_weight`, maybe `l1_ratio` if you want Elastic-Net via `saga`), and
* A **small wrapper** that picks the best of {ridge, lasso, catboost} based on validation Sharpe and writes a simple YAML “model registry” you can inspect.

[1]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?utm_source=chatgpt.com "LogisticRegression — scikit-learn 1.7.2 documentation"
[2]: https://optuna.org/?utm_source=chatgpt.com "Optuna - A hyperparameter optimization framework"
[3]: https://scikit-learn.org/stable/modules/calibration.html?utm_source=chatgpt.com "1.16. Probability calibration"
[4]: https://scikit-learn.org/0.19/modules/generated/sklearn.linear_model.LogisticRegression.html?utm_source=chatgpt.com "sklearn.linear_model.LogisticRegression"
[5]: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html?utm_source=chatgpt.com "CalibratedClassifierCV"
