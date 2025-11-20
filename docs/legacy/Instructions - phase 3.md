
## Quick pass on the code you attached

**Overall:** the validation + dataset builder scaffolding is solid and ready for model training. A couple of nits and one likely typo to fix before we proceed.

### 1) `scripts/validate_ml_data.py`

* ✅ Good: you switched to `CAST(:param AS type)` instead of `:param::type` (avoids SQLAlchemy/Postgres clash). You also validate candles/weather coverage and UTC handling. 
* **Suggestion:** add an explicit check that `minutes_to_close` never goes negative and is computed in the *station’s local timezone window* (DST gotchas). Kalshi weather markets settle off the **NWS Daily Climate Report** and use *local standard time* during DST, so the day boundary is slightly shifted; make sure your label logic matches that convention. ([Kalshi Help Center][1])

### 2) `scripts/validate_data.py`

* Looks fine as a light sanity‑check companion. Consider consolidating repeated queries with the ML validator to keep a single source of truth. 

### 3) `tests/test_joins.py`

* **Likely typo:** mapping contains `KXHIGHAUST` / `KXHIGHLA`; these should be `KXHIGHAUS` / `KXHIGHLAX` to match the series tickers you’ve been using elsewhere. Fix before training so your series filters don’t accidentally miss data. 
* **Timezone note:** if you use any `datetime.utcnow()` calls in tests, make them timezone‑aware (`datetime.now(timezone.utc)`) so the tests mimic production code paths. 

---

## Should you “continue”? Yes—here’s the precise order

To balance **speed** (you said backtests shouldn’t take all day) with **correctness**, I’d proceed exactly like this:

1. **Phase 3 Ridge baseline with calibration (this unlocks everything else).**

   * Ridge logistic per bracket type (greater / less / between).
   * Grouped CV **by day** to prevent leakage. ([Scikit-learn][2])
   * 15 Optuna trials **per walk‑forward window** (fast mode). ([Optuna][3])
   * Calibrate probabilities: **isotonic** if ≥1k calibration points, else **Platt/sigmoid**. ([Scikit-learn][4])

2. **Walk‑forward trainer (42→7 days, step 7)** with artifacts saved per window (model pickle, Optuna JSON, predictions CSV, window metadata).

3. **Model‑driven strategy hook** in the backtester: fee‑aware effective price, edge gate (≥3¢), slippage=1¢, **quarter‑Kelly (α=0.25)** sizing, and your risk limits (≤10% bankroll per city‑day‑side; soft cap on concurrent bins). Kelly is the right sizing framework here; it maximizes long‑run log growth, and fractional Kelly controls risk. ([Wikipedia][5])

4. **A/B backtest**: simple 3‑bin cap allocator (baseline) vs “multinomial” allocator later. Keep the simple allocator **now** for speed and clean debugging. We can flip in the multinomial allocator after we have baseline Sharpe.

5. (Optional backlog) **CatBoost with monotone constraints** once Ridge is stable, with monotonicity on physics‑guided features (e.g., distance‑to‑floor for “greater” should not *decrease* p(YES) as temperature rises). ([CatBoost][6])

> **Why these choices:**
> • GroupKFold by day is the correct guard against temporal leakage when multiple 1‑min samples share a day‑level outcome. ([Scikit-learn][2])
> • CalibratedClassifierCV is the standard way to make classifier probabilities usable in trading; use **isotonic** with enough data, otherwise sigmoid. ([Scikit-learn][4])
> • Kelly + fee‑aware breakeven is essential on Kalshi because **fees are a function of price** (expected earnings) and are rounded up to the next cent. We size only when **edge after fees + spread + slippage** is positive. ([Kalshi][7])
> • Weather markets settle from the **NWS Daily Climate Report** the next morning; ensure label windows align to NWS day boundaries (DST nuance). ([Kalshi Help Center][1])

---

## Drop‑in code: Ridge + calibration + walk‑forward

> These stubs assume your minimal feature set from `ml/dataset.py` (10 features), your day‑grouping, and your fee functions (you’ve already implemented costs/fees). Plug them in as new files and wire them to your existing CLI.

### `ml/logit_linear.py` — Ridge with Optuna + calibration

```python
# ml/logit_linear.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
import optuna  # 15 trials per window

@dataclass
class RidgeTuned:
    base_params: Dict[str, Any]
    calib_method: str  # 'isotonic' or 'sigmoid'
    coef_: Optional[np.ndarray] = None  # populated after fit

def _mk_groups_kfold(n_splits: int = 4) -> GroupKFold:
    # groups are day indices from your dataset builder
    return GroupKFold(n_splits=n_splits)

def tune_ridge_with_optuna(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_trials: int = 15, seed: int = 42
) -> RidgeTuned:
    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        # class_weight helps deal with skew; try None or balanced
        cw = trial.suggest_categorical("class_weight", [None, "balanced"])
        # stronger L2 near extremes can be helpful
        penalty = "l2"
        solver = "lbfgs"

        clf = LogisticRegression(
            C=C, penalty=penalty, solver=solver, max_iter=2000,
            class_weight=cw, n_jobs=1
        )

        # out-of-fold probabilities for Brier/log-loss
        oof = cross_val_predict(
            clf, X, y, groups=groups, cv=_mk_groups_kfold(),
            method="predict_proba"
        )[:, 1]

        # score: Brier (lower is better)
        return brier_score_loss(y, oof)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    # Choose calibration: isotonic if enough data; else sigmoid (Platt)
    calib_method = "isotonic" if len(y) >= 1000 else "sigmoid"  # sklearn guidance
    return RidgeTuned(base_params=best, calib_method=calib_method)

def fit_ridge_calibrated(
    X: np.ndarray, y: np.ndarray, tuned: RidgeTuned, groups: np.ndarray, seed: int = 42
) -> CalibratedClassifierCV:
    base = LogisticRegression(
        **tuned.base_params,
        penalty="l2", solver="lbfgs", max_iter=2000, n_jobs=1
    )
    # GroupKFold CV for calibration too (disjoint folds by day)
    calib = CalibratedClassifierCV(
        estimator=base, method=tuned.calib_method, cv=_mk_groups_kfold()
    )
    calib.fit(X, y, groups=groups)
    # save coefficients for reporting
    try:
        tuned.coef_ = calib.base_estimator_.coef_.copy()
    except Exception:
        pass
    return calib

def evaluate_probs(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "brier": float(brier_score_loss(y_true, p)),
        "logloss": float(log_loss(y_true, p, eps=1e-6)),
    }

def export_meta(tuned: RidgeTuned) -> str:
    return json.dumps({"base_params": tuned.base_params, "calibration": tuned.calib_method})
```

**Why this way:** `CalibratedClassifierCV` implements both **isotonic** and **sigmoid** calibration; isotonic is preferred with ample data (≥~1k) to avoid overfitting; otherwise sigmoid (Platt) is safer. ([Scikit-learn][4])

### `ml/train_walkforward.py` — walk‑forward (42→7) with artifacts

```python
# ml/train_walkforward.py
from __future__ import annotations
import os, json, pathlib
from datetime import date, timedelta
import numpy as np
import pandas as pd

from ml.dataset import build_training_dataset  # your builder
from ml.logit_linear import tune_ridge_with_optuna, fit_ridge_calibrated, evaluate_probs, export_meta

def iter_windows(start_date: date, end_date: date, train_days: int = 42, test_days: int = 7, step_days: int = 7):
    cur = start_date
    while cur + timedelta(days=train_days + test_days - 1) <= end_date:
        train_start = cur
        train_end   = cur + timedelta(days=train_days - 1)
        test_start  = train_end + timedelta(days=1)
        test_end    = test_start + timedelta(days=test_days - 1)
        yield (train_start, train_end, test_start, test_end)
        cur += timedelta(days=step_days)

def train_city_bracket_walkforward(
    city: str, bracket_type: str, start_date: date, end_date: date,
    out_dir: str = "models/trained", n_trials: int = 15
):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    for (tr_s, tr_e, te_s, te_e) in iter_windows(start_date, end_date):
        # build datasets
        X_tr, y_tr, g_tr, meta_tr = build_training_dataset(city, tr_s, tr_e, bracket_type)
        X_te, y_te, g_te, meta_te = build_training_dataset(city, te_s, te_e, bracket_type)

        # tune on train
        tuned = tune_ridge_with_optuna(X_tr, y_tr, g_tr, n_trials=n_trials)
        model = fit_ridge_calibrated(X_tr, y_tr, tuned, g_tr)

        # predict test
        p_te = model.predict_proba(X_te)[:, 1]
        metrics = evaluate_probs(y_te, p_te)

        # persist artifacts
        win_dir = pathlib.Path(out_dir) / city / bracket_type / f"window_{tr_s:%Y%m%d}_{tr_e:%Y%m%d}"
        win_dir.mkdir(parents=True, exist_ok=True)
        np.save(win_dir / "model.pkl.npy", model)  # quick&dirty; replace with joblib if you prefer
        (win_dir / "best_params.json").write_text(export_meta(tuned))
        pd.DataFrame({
            "timestamp": meta_te["timestamp"],
            "market_ticker": meta_te["market_ticker"],
            "event_date": meta_te["event_date"],
            "p_model": p_te,
            "y": y_te,
        }).to_csv(win_dir / f"preds_{te_s:%Y%m%d}_{te_e:%Y%m%d}.csv", index=False)
        (win_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
```

### `backtest/model_strategy.py` — fee‑aware, Kelly‑sized signals

```python
# backtest/model_strategy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import math

from models.costs import (
    taker_fee_cents,
    effective_yes_entry_cents,
    effective_no_entry_cents,
)

@dataclass
class StrategyConfig:
    tau_open_cents: int = 3     # open if edge >= 3¢ after costs
    tau_close_cents: int = 1    # close if edge <= 1¢
    slippage_cents: int = 1
    alpha_kelly: float = 0.25   # quarter Kelly
    max_spread_cents: int = 3
    max_bankroll_pct_city_day_side: float = 0.10

def breakeven_prob_from_entry(entry_cents: int) -> float:
    # YES entry breakeven ≈ entry / 100 (payout is 100¢)
    return entry_cents / 100.0

def kelly_fraction_yes(p_model: float, y_eff: float) -> float:
    # f* = (p - y_eff) / (1 - y_eff)
    num = p_model - y_eff
    den = 1.0 - y_eff
    return 0.0 if den <= 0 else num / den

def choose_side_and_size(
    yes_bid: int, yes_ask: int, p_model: float, cfg: StrategyConfig, bankroll_cents: int
) -> Dict:
    spread = yes_ask - yes_bid
    if spread > cfg.max_spread_cents:
        return {"action": "NONE", "reason": "spread"}

    # YES path
    y_entry = effective_yes_entry_cents(yes_bid, yes_ask, cfg.slippage_cents)
    p_be_yes = breakeven_prob_from_entry(y_entry)
    edge_yes_cents = 100 * p_model - y_entry

    # NO path
    n_entry = effective_no_entry_cents(yes_bid, yes_ask, cfg.slippage_cents)
    p_be_no = breakeven_prob_from_entry(n_entry)  # applies to (1 - p)
    edge_no_cents = 100 * (1 - p_model) - n_entry

    if max(edge_yes_cents, edge_no_cents) < cfg.tau_open_cents:
        return {"action": "NONE", "reason": "edge"}

    if edge_yes_cents >= edge_no_cents:
        f_star = max(0.0, kelly_fraction_yes(p_model, p_be_yes))
        side = "BUY_YES"
        p_eff = p_be_yes
        edge_c = edge_yes_cents
        entry_cents = y_entry
    else:
        # Kelly on NO uses p'=1-p against NO-entry cost
        p_no = 1.0 - p_model
        f_star = max(0.0, kelly_fraction_yes(p_no, p_be_no))
        side = "BUY_NO"
        p_eff = p_be_no
        edge_c = edge_no_cents
        entry_cents = n_entry

    f_star = min(f_star, 1.0)
    frac = cfg.alpha_kelly * f_star

    # bankroll cap per city-day-side (10%): convert to contracts
    cap_dollars = (cfg.max_bankroll_pct_city_day_side * bankroll_cents) / 100.0
    # 1 contract pays 1 dollar per 100¢; approximate contracts by dollar exposure
    max_contracts_cap = max(1, int(cap_dollars))  # crude cap; integrate your RiskManager for precise caps

    # contracts sized by fraction of bankroll dollar exposure
    contracts = max(1, min(max_contracts_cap, int(frac * cap_dollars)))
    return {
        "action": side,
        "contracts": contracts,
        "edge_cents": int(edge_c),
        "entry_cents": int(entry_cents),
        "p_model": float(p_model),
        "p_eff": float(p_eff),
    }
```

> **Why fee‑aware gating + fractional Kelly?**
> Kalshi’s fee is **rounded up** and **priced on expected earnings** (`0.07 * C * P * (1-P)` for takers; maker 0.0175), so your breakeven probability **is not just `ask/100`**; it’s `effective_entry / 100`. Only act when the **edge after costs** exceeds a threshold, then scale with fractional Kelly to manage drawdowns. ([Kalshi][7])

---

## How to run (fast mode)

1. **Single window sanity pass (Chicago, BETWEEN)**

```bash
python -m ml.train_walkforward \
  --city chicago \
  --bracket between \
  --start 2025-09-01 --end 2025-10-31 \
  --tr-days 42 --te-days 7 --step 7 \
  --n-trials 15
```

2. **Model‑driven backtest using your existing harness**
   Either add `--strategy model_kelly` to your current `run_backtest.py`, or create `run_model_backtest.py` that loads the per‑window predictions and calls `backtest/model_strategy.py` for minute‑by‑minute decisions (taker fills, slippage=1¢, spread≤3¢).

---

## A few final details to keep you “pro‑grade”

* **DST boundaries & NWS day definition:** NWS climate reports can use *local standard time* during DST, meaning the “day” for the official high can span **01:00 to 00:59** the next local day. Align label windows to NWS, not midnight UTC. Your settlement sync already points at NWS’ Daily Climate Report as the source of truth. ([Kalshi Help Center][1])
* **Grouped CV by day** is appropriate for your 1‑min rows -> 1 day outcome structure (non‑overlapping groups). ([Scikit-learn][2])
* **Calibration choice**: isotonic can overfit with small calibration sets; use sigmoid (Platt) when <~1000 points in the window. ([Scikit-learn][4])
* **Optuna cadence**: you don’t need to re‑tune every minute. Tune **once per walk‑forward window** (or **hourly** in shadow trading), cache the best params, and reuse intra‑window; this keeps backtests fast while still adapting. ([Optuna][3])
* **Future (after Ridge is stable):**
  • **CatBoost with monotone constraints** on “distance to floor/cap” features (increasing/decreasing as physics dictates). ([CatBoost][6])
  • **Opinion pooling** (log‑odds blend of model vs market price), tuning the blend weight by log‑loss on train windows. ([NeurIPS Papers][8])
  • **Multinomial Kelly** across between‑bins (sum‑to‑1 PMF), but do it only after you’ve baselined Sharpe with the simple allocator.

---

## Answering your implicit “how do weather markets work” & why this matches

* Kalshi’s **weather markets** on daily highs settle **the next morning** off the **final NWS Daily Climate Report** (CLI), not phone app temps; DST is handled as above. Your settlement pipeline (CLI → CF6 → IEM_CF6 → GHCND → VC) is exactly aligned with this. ([Kalshi Help Center][1])
* Your fee module matches Kalshi’s **Oct 2025 fee schedule** (taker 0.07, maker 0.0175, ceiling to next cent; no settlement fee). The backtester’s cost model needs to use these formulas **for each trade**. ([Kalshi][7])

---

## What to tell the agent (copy/paste)

> **Go ahead and start Phase 3 Ridge baseline now.**
>
> 1. Implement `ml/logit_linear.py` exactly as above (Optuna 15 trials, GroupKFold by day, isotonic≥1000 else sigmoid).
> 2. Implement `ml/train_walkforward.py` and run Chicago BETWEEN for 2025‑09‑01 to 2025‑10‑31 (42→7 windows, step 7). Save artifacts per window.
> 3. Add `backtest/model_strategy.py` as above. In the backtester, add a `--strategy model_kelly` path that loads per‑window predictions, computes fee‑aware breakeven, gates on 3¢ edge, uses slippage=1¢, spread≤3¢, and sizes with **α=0.25 Kelly**, capped at **10% bankroll per city‑day‑side**.
> 4. Ensure labels and `minutes_to_close` follow **NWS day boundaries** (DST nuance).
> 5. After the Ridge walk‑forward finishes, run the model‑driven backtest and report **net Sharpe, max DD, turnover, fee spend**.
> 6. Only then, queue CatBoost with **monotone constraints** and the multinomial allocator as Phase 6.

---

### Why I’m comfortable “continuing” now

* Your data validation & dataset builder already show the pieces are hooked up (candles, weather, metadata). 
* The light code edits above (series ticker typos, timezone awareness) are trivial and won’t block training. 
* The modeling stack and training protocol follow the standard playbook for **probability models that will be traded**: group‑aware CV, proper calibration, fee‑aware sizing, and walk‑forward evaluation. ([Scikit-learn][2])

---

---

## 1) Quick code review & fixes (on the files you attached)

### `scripts/validate_ml_data.py`

**What’s solid**

* Coverage checks for candles and weather, sample join test, and grouped-by-day reporting look good. The casting fix (`CAST(:param AS date)`) avoids the `:param::type` bug. 
* The 1‑minute vs. 5‑minute fallback is sensible. 

**Tighten these**

1. **Strike presence count can overcount markets**
   You currently do `COUNT(floor_strike) + COUNT(cap_strike)` for “markets_with_strikes”. A single market may have *both* strikes (between), making counts > total markets. Use a boolean OR:

```sql
SELECT
  COUNT(*)                        AS total_markets,
  COUNT(close_time)               AS markets_with_close_time,
  COUNT(*) FILTER (
    WHERE floor_strike IS NOT NULL OR cap_strike IS NOT NULL
  )                               AS markets_with_strikes
FROM markets
WHERE ticker LIKE 'KXHIGHCHI%'
  AND close_time >= :cutoff_date;
```

(Replace the existing query in `validate_market_metadata`.) 

2. **Hardcoded city (Chicago/KMDW) → parameterize**
   Turn `'KXHIGHCHI%'` and `'KMDW'` into params (or derive from your `dim_city` table) so you can reuse this script for other cities. 

3. **Coverage denominator**
   `expected_total = 1440*days*num_locations` assumes the last 60 days align with full 24h days for all stations. For a precise percentage, compute days from `MIN(ts_utc)::date`…`MAX(ts_utc)::date` and multiply by *active* locations in that span. (Not a blocker—just be aware.) 

4. **Timezone text**
   The script asserts candles and weather are UTC. That’s consistent with your DB usage; just keep local conversions in feature engineering only (which you’re already doing). 

Everything else I see in that script looks consistent with the plan and your DB schema.

> **Verdict:** No blocking issues. Make the strike-count fix, parametrize city/station, and proceed.

---

## 2) What to tell the agent now (concise directive)

> **Continue to Phase 3–5 exactly as planned; implement the Ridge baseline + calibration and the model‑driven backtest.**
>
> * **Train scope:** Chicago **between** brackets first (most samples), then add **greater** and **less**.
> * **Features:** Start with the current minimal 10 features; keep the dataset builder flexible to add more later.
> * **Tuning:** Optuna **15 trials per walk‑forward window**.
> * **Calibration:** Use `CalibratedClassifierCV`—**isotonic** if ≥1,000 calibration points, **sigmoid/Platt** otherwise. (This is straight from scikit‑learn guidance.) ([Scikit-learn][1])
> * **Walk‑forward:** 42d train → 7d test, **step 7d**. Group splits **by day** to avoid leakage; if you need a built‑in splitter reference, sklearn’s `TimeSeriesSplit` shows the pattern for ordered data (we’ll still group by event day). ([Scikit-learn][2])
> * **Execution filters:** taker fills, **max spread = 3¢**, slippage **1¢**, entry hysteresis **3¢** / exit **1¢** (configurable).
> * **Risk:** ≤10% bankroll per (city, day, side). **Max 3 concurrent bins per city** for the first baseline; we’ll implement the multinomial allocator later for A/B.
> * **Artifacts per window:** pickle(model), JSON(best params), CSV(predictions), YAML(window metadata), HTML (calibration plots).
> * **A/B path ready:** Add a flag `--allocator={simple,multinomial}` but ship **simple** first; multinomial Kelly next.
> * **Probability calibration references (for implementation details):** sklearn docs on calibration (isotonic vs. Platt). ([Scikit-learn][3])

---

## 3) Drop‑in code (robust, minimal deps)

Below are self‑contained stubs that follow the plan, slot into your repo, and use the exact practices we discussed (Optuna per window, calibration, saved artifacts, Kelly‑sized strategy). They reference your existing DB and backtest structure.

### `ml/logit_linear.py` — Ridge logistic + Optuna + calibration

```python
# ml/logit_linear.py
from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
import joblib
import optuna

# ---------- Data wrapper for clarity ----------
@dataclass
class WindowData:
    X_train: np.ndarray
    y_train: np.ndarray
    groups_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta_test: pd.DataFrame  # timestamps, market_ticker, etc.

@dataclass
class TrainResult:
    best_params: Dict
    calib_method: str
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    model_path: str
    preds_path: str
    params_path: str

# ---------- Tuning ----------
def _ridge_search_space(trial: optuna.Trial) -> Dict:
    # C is inverse regularization; search log scale
    C = trial.suggest_float("C", 1e-3, 1e+3, log=True)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    return {"C": C, "class_weight": class_weight}

def tune_ridge_with_optuna(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 4, n_trials: int = 15, seed: int = 42
) -> Dict:
    """
    GroupKFold by event day; objective is mean log loss across folds.
    """
    gkf = GroupKFold(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = _ridge_search_space(trial)
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                penalty="l2", solver="liblinear", max_iter=2000, **params, random_state=seed
            )),
        ])

        losses: List[float] = []
        for train_idx, valid_idx in gkf.split(X, y, groups):
            pipe.fit(X[train_idx], y[train_idx])
            p = pipe.predict_proba(X[valid_idx])[:,1]
            losses.append(log_loss(y[valid_idx], p, eps=1e-6))
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", study_name="ridge_logit")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

# ---------- Fit + calibration ----------
def fit_ridge_with_calibration(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, best_params: Dict, seed: int = 42
) -> CalibratedClassifierCV:
    """
    Split train into model-train and calibration folds by day (no leakage).
    Use isotonic if ample calibration points (>=1000), else sigmoid (Platt).
    """
    # 20% of days for calibration
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    (train_idx, calib_idx) = next(gss.split(X, y, groups))
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_cal, y_cal = X[calib_idx], y[calib_idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2", solver="liblinear", max_iter=5000, random_state=seed, **best_params
        ))
    ])
    pipe.fit(X_tr, y_tr)

    method = "isotonic" if len(y_cal) >= 1000 else "sigmoid"  # sklearn guidance
    # scikit‑learn’s CalibratedClassifierCV docs: isotonic needs many samples
    # to avoid overfitting; sigmoid=Platt otherwise.
    # https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
    calibrated = CalibratedClassifierCV(pipe, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated, method

# ---------- Evaluate & persist ----------
def evaluate_binary(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "log_loss": float(log_loss(y_true, p, eps=1e-6)),
        "brier": float(brier_score_loss(y_true, p)),
    }

def train_one_window(
    win: WindowData, artifacts_dir: str, tag: str, n_trials: int = 15, seed: int = 42
) -> TrainResult:
    os.makedirs(artifacts_dir, exist_ok=True)
    best = tune_ridge_with_optuna(win.X_train, win.y_train, win.groups_train, n_trials=n_trials, seed=seed)
    model, calib_method = fit_ridge_with_calibration(win.X_train, win.y_train, win.groups_train, best, seed=seed)

    # Train metrics via cross-validated calibration split already computed implicitly
    # Here we just evaluate on the calibration distribution as proxy:
    # (Optional) Could return cv metrics from tuning loop instead.
    # Test metrics
    p_test = model.predict_proba(win.X_test)[:,1]
    test_metrics = evaluate_binary(win.y_test, p_test)

    # Persist
    model_path = os.path.join(artifacts_dir, f"ridge_{tag}.pkl")
    joblib.dump(model, model_path)
    params_path = os.path.join(artifacts_dir, f"ridge_params_{tag}.json")
    with open(params_path, "w") as f:
        json.dump({"best_params": best, "calibration": calib_method}, f, indent=2)

    preds = win.meta_test.copy()
    preds["p_model"] = p_test
    preds_path = os.path.join(artifacts_dir, f"ridge_preds_{tag}.csv")
    preds.to_csv(preds_path, index=False)

    return TrainResult(
        best_params=best,
        calib_method=calib_method,
        train_metrics={"cv_metric": None},
        test_metrics=test_metrics,
        model_path=model_path,
        preds_path=preds_path,
        params_path=params_path,
    )
```

**Why this design:**

* **Calibration choice** follows sklearn guidance—*isotonic* is powerful but can overfit on small calibration sets; use *sigmoid/Platt* otherwise. ([Scikit-learn][1])
* Grouped splits by **event day** prevent temporal leakage. `TimeSeriesSplit` is a good reference for ordered data, but here we explicitly group by day because your labels resolve at day level. ([Scikit-learn][2])

---

### `ml/train_walkforward.py` — 42→7 day windows, per bracket, per city

```python
# ml/train_walkforward.py
from __future__ import annotations
import os
from datetime import date, timedelta
from typing import Tuple, Iterable, Literal
import pandas as pd
import numpy as np
from ml.logit_linear import WindowData, train_one_window
from ml.dataset import build_training_dataset  # you already have this

Bracket = Literal["between", "greater", "less"]

def windows(start: date, end: date, train_days: int = 42, test_days: int = 7, step_days: int = 7
) -> Iterable[Tuple[date, date, date, date]]:
    cur = start
    while True:
        train_start = cur
        train_end = cur + timedelta(days=train_days-1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days-1)
        if test_end > end: break
        yield train_start, train_end, test_start, test_end
        cur = cur + timedelta(days=step_days)

def train_city_bracket_walkforward(
    city: str,
    bracket: Bracket,
    start: date,
    end: date,
    outdir: str = "models/trained",
    n_trials: int = 15,
):
    os.makedirs(outdir, exist_ok=True)
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows(start, end)):
        # --- build train ---
        X_tr, y_tr, g_tr, meta_tr = build_training_dataset(
            city=city, start_date=tr_s, end_date=tr_e, bracket_type=bracket
        )
        # --- build test ---
        X_te, y_te, g_te, meta_te = build_training_dataset(
            city=city, start_date=te_s, end_date=te_e, bracket_type=bracket
        )
        win = WindowData(
            X_train=X_tr, y_train=y_tr, groups_train=g_tr,
            X_test=X_te, y_test=y_te, meta_test=meta_te
        )
        tag = f"{city}_{bracket}_{tr_s.strftime('%Y%m%d')}_{te_e.strftime('%Y%m%d')}"
        art_dir = os.path.join(outdir, city, bracket, f"win_{tr_s:%Y%m%d}_{te_e:%Y%m%d}")
        res = train_one_window(win, art_dir, tag, n_trials=n_trials)
        print(f"[{i}] {city}/{bracket} {tr_s}→{te_e}  "
              f"test brier={res.test_metrics['brier']:.4f}  logloss={res.test_metrics['log_loss']:.4f}")
```

---

### `backtest/model_strategy.py` — model‑driven signals with fees, spread, slippage, Kelly

```python
# backtest/model_strategy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd

from models.costs import (
    effective_yes_entry_cents, effective_no_entry_cents
)
# If your cost functions live elsewhere, adjust import.

@dataclass
class ExecParams:
    max_spread_cents: int = 3
    slippage_cents: int = 1
    tau_open_cents: int = 3   # entry hysteresis
    tau_close_cents: int = 1  # exit hysteresis
    alpha_kelly: float = 0.25 # risk scaling

def breakeven_prob_yes(yes_bid: int, yes_ask: int, slippage: int = 1) -> float:
    y_eff = effective_yes_entry_cents(yes_bid, yes_ask, slippage=slippage)
    return y_eff / 100.0

def breakeven_prob_no(yes_bid: int, yes_ask: int, slippage: int = 1) -> float:
    n_eff = effective_no_entry_cents(yes_bid, yes_ask, slippage=slippage)
    return n_eff / 100.0

def kelly_fraction_yes(p: float, y_eff: float) -> float:
    # Kelly for 0–100 payoff in cents: f* = (p - y_eff) / (1 - y_eff)
    # See Kelly criterion (binary payoff; b=1) and adapt to price-in-cents payoff geometry
    # https://en.wikipedia.org/wiki/Kelly_criterion
    if p <= y_eff: 
        return 0.0
    return (p - y_eff) / max(1e-9, 1.0 - y_eff)

def edge_cents_yes(p: float, yes_bid: int, yes_ask: int, slippage: int = 1) -> float:
    y_eff = effective_yes_entry_cents(yes_bid, yes_ask, slippage=slippage) / 100.0
    ev_yes = 1.0 * p - y_eff
    return ev_yes * 100.0

def edge_cents_no(p: float, yes_bid: int, yes_ask: int, slippage: int = 1) -> float:
    n_eff = effective_no_entry_cents(yes_bid, yes_ask, slippage=slippage) / 100.0
    ev_no = 1.0 * (1.0 - p) - n_eff
    return ev_no * 100.0

class ModelKellyStrategy:
    """
    Compute signal from model probability and order book snapshot.
    Designed to plug into your backtest loop.
    """

    def __init__(self, exec_params: ExecParams):
        self.params = exec_params

    def signal_for_row(self, row: pd.Series) -> Optional[Dict]:
        """
        row must contain: yes_bid_close, yes_ask_close, p_model, city, event_date, market_ticker.
        """
        bid, ask = int(row["yes_bid_close"]), int(row["yes_ask_close"])
        spread = max(0, ask - bid)
        if spread > self.params.max_spread_cents:
            return None

        p = float(row["p_model"])
        # Choose cheaper side after costs
        e_yes = edge_cents_yes(p, bid, ask, slippage=self.params.slippage_cents)
        e_no  = edge_cents_no (p, bid, ask, slippage=self.params.slippage_cents)

        # Entry rule with hysteresis
        take_yes = e_yes >= self.params.tau_open_cents and e_yes >= e_no
        take_no  = e_no  >= self.params.tau_open_cents and e_no  >  e_yes
        if not (take_yes or take_no):
            return None

        if take_yes:
            y_eff = breakeven_prob_yes(bid, ask, self.params.slippage_cents)
            f_star = self.params.alpha_kelly * kelly_fraction_yes(p, y_eff)
            side = "BUY_YES"
            edge = e_yes
        else:
            n_eff = breakeven_prob_no(bid, ask, self.params.slippage_cents)
            # Kelly for NO mirrors YES once you flip p->(1-p) and price->(100-price).
            f_star = self.params.alpha_kelly * kelly_fraction_yes(1.0 - p, n_eff)
            side = "BUY_NO"
            edge = e_no

        if f_star <= 0.0:
            return None

        # Convert fraction to contracts sizing in your portfolio code (position limits apply there)
        return {
            "action": side,
            "kelly_frac": float(f_star),
            "edge_cents": float(edge),
            "spread_cents": int(spread),
        }
```

> **Note on Kelly**: the simple binary‑payoff Kelly fraction is (f^* = p - \frac{1-p}{b}). For YES contracts with a 0/100 payoff and deterministic price‑in‑cents entry, the equivalent fraction reduces to (f^* = \frac{p - y_{\text{eff}}}{1 - y_{\text{eff}}}), which is what we use (then apply a fractional Kelly multiplier ( \alpha)). ([Wikipedia][4])

---

### `scripts/run_train_walkforward.py` — CLI to train Ridge walk‑forward

```python
# scripts/run_train_walkforward.py
import argparse
from datetime import datetime
from ml.train_walkforward import train_city_bracket_walkforward

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="chicago")
    ap.add_argument("--bracket", choices=["between", "greater", "less"], default="between")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--trials", type=int, default=15)
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date()
    train_city_bracket_walkforward(
        city=args.city, bracket=args.bracket, start=start, end=end, n_trials=args.trials
    )

if __name__ == "__main__":
    main()
```

---

## 4) Why these choices (citations)

* **Calibration (isotonic vs. Platt/sigmoid).** Scikit‑learn recommends isotonic when you have enough calibration data (≳1,000 points); otherwise use Platt’s sigmoid to avoid overfitting. Your per‑window minute‑level data for “between” will usually exceed that threshold; for “greater/less” it may not, so we switch automatically. ([Scikit-learn][1])

* **Walk‑forward validation** with grouped splits by event day prevents leakage and mirrors deployment. `TimeSeriesSplit` is a good reference for ordered splits; in finance there’s also purged/embargoed CV for label look‑ahead issues if you later need to go beyond day‑level grouping. ([Scikit-learn][2])

* **Kelly sizing** is the right theoretical target for long‑run log‑wealth growth; we use **fractional Kelly** (α=0.25) to control variance and apply **fees/spread/slippage** in the break‑even probability. ([Wikipedia][4])

* **CatBoost monotone constraints** (for later) are documented and easy to add when you move to tree models; we keep Ridge first for speed and stability. ([CatBoost][5])

* **Optuna** is appropriate for quick per‑window tuning with a small trial budget (15) to avoid overfitting/hours‑long runs. ([Optuna][6])

---

## 5) Should you tell it to continue?

**Yes.** With the small strike‑count fix and city/station parameterization in the validation script, you’re ready to proceed exactly as the agent proposed:

1. **Phase 3:** Ridge logistic + Optuna + calibration (use the code stubs above).
2. **Phase 4:** Walk‑forward trainer (42→7, step 7) saving models/preds/metadata per window.
3. **Phase 5:** Model‑driven Kelly strategy with your execution filters and risk limits, using the provided `ModelKellyStrategy`.

That gets you to a trustworthy, fee‑aware, **calibrated** baseline with real walk‑forward backtests. After we have those numbers, we can A/B test the multinomial allocator and/or move to CatBoost with monotone constraints.

If you want a single message to paste into the agent, use this:

---

### 📋 Paste‑to‑agent directive

> **Proceed with Phase 3–5 (Ridge baseline + calibration + model backtest).**
> **Fix first:** In `validate_ml_data.py` replace the “markets_with_strikes” query with an OR‑count and parameterize city/station (remove hardcoded `'KXHIGHCHI%'` and `'KMDW'`).
> **Then implement:**
>
> 1. **Ridge logistic with Optuna (15 trials/window)** using `ml/logit_linear.py` (attached). GroupKFold by event day; return best params.
> 2. **Calibration:** wrap with `CalibratedClassifierCV`; choose **isotonic if N_cal ≥ 1000**, else **sigmoid/Platt**. Save calibrated model per window with params JSON and predictions CSV.
> 3. **Walk‑forward trainer:** add `ml/train_walkforward.py` (attached). Run 42→7 days, step 7. Start with **Chicago, between brackets** only, then add greater/less.
> 4. **Model strategy:** add `backtest/model_strategy.py` (attached). Taker fills; **max spread=3¢, slippage=1¢, τ_open=3¢, τ_close=1¢, α=0.25 fractional Kelly**.
> 5. **CLI:** add `scripts/run_train_walkforward.py` (attached). Example:
>
>    ```
>    python scripts/run_train_walkforward.py --city chicago --bracket between --start 2025-08-01 --end 2025-10-31 --trials 15
>    ```
> 6. **Artifacts per window:** save `ridge.pkl`, `ridge_params.json`, `ridge_preds.csv`, and a small YAML with window metadata.
> 7. **Backtest:** wire `ModelKellyStrategy` into the existing backtest harness via a `--strategy model_kelly` switch (or add a new `run_model_backtest.py`) and run the per‑window simulations.
> 8. **Reports:** produce per‑window and aggregate metrics (Brier, log‑loss, Sharpe, max DD, turnover, fee spend). Plot reliability curves.

---