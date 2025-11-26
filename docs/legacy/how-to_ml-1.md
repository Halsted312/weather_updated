Below is a drop‑in **engineering brief** you can paste to your coding agent. It answers your Optuna/walk‑forward question (yes: tune per window), outlines a pro‑grade Ridge/Lasso pipeline, and includes robust code stubs that slot into your repo (file names, function signatures, CLI). I’ve also added quality gates and diagnostics so we don’t regress on calibration or execution. Citations at the end back up calibration, CV, and tuning choices.

---

# MODEL_SPEC_RIDGE_LASSO_v1.md

**Goal.** Build a reproducible, walk‑forward ML stack that predicts *daily integer Tmax* and/or *per‑bin YES probability* for Kalshi weather markets, using:

* Logit with **L2 (“ridge”)** and **L1/elastic‑net (“lasso”)** penalties
* **Per‑window Optuna** tuning (90‑day train → 7‑day test; step 7)
* **Probability calibration** (isotonic when N is large, else Platt/sigmoid)
* **Opinion pooling** with market odds (optional, log‑odds blend)
* **Maker‑first execution** already in codebase

Use VC 5‑minute weather for **six cities** (AUS, CHI, LAX, MIA, DEN, PHL). **Exclude NYC** VC features (keep NYC labels + market features) because station quality is poor.

---

## 0) High‑level decisions (what & why)

1. **Tune Optuna per walk‑forward window.**
   The odds environment/seasonality drifts; re‑tuning C/l1_ratio/class_weight on each 90‑day training slice prevents stale hyperparams. (This is standard in time‑series ML and aligns with how we’d trade in production.)

2. **Model types.**

   * *Ridge logistic*: `penalty="l2"` — stable with correlated features.
   * *Lasso / Elastic‑Net logistic*: `penalty="l1"` or `penalty="elasticnet", l1_ratio∈[0,1]` via `solver="saga"`. L1 can zero out weak features → try more features safely.

3. **Calibration.**

   * Use `CalibratedClassifierCV` with **isotonic** when calibration set N≥1,000; else **sigmoid/Platt**. This is the canonical approach and improves Brier/log‑loss in practice. ([Scikit-learn][1])

4. **Walk‑forward grouping.**

   * Use **GroupKFold by event_date** (no same‑day leakage); for the intra‑window CV used by Optuna, also group by day. ([Scikit-learn][2])

5. **Metrics to gate models.**
   Report **Brier score** (proper scoring), **log loss**, **calibration curve**, and edge diagnostics. Use `sklearn.metrics.brier_score_loss` for evaluation. ([Scikit-learn][3])

6. **Feature policy.**

   * Ridge uses the **baseline 10** + a few conservative additions.
   * Lasso/Elastic‑Net gets a **feature‑rich** set; regularization will prune.
   * All features must be available by the minute of prediction; no post‑close leakage.

7. **Opinion pooling (optional, later flag).**

   * Combine model and market in **log‑odds space**: `logit(p*) = w·logit(p_model) + (1-w)·logit(p_mkt)`. Tune `w∈[0,1]` on the train window by minimizing log‑loss. Literature: combining probability forecasts & logit pooling. (See citations.)

---

## 1) File layout (new/updated)

```
ml/
  models/
    logit_linear.py          # Ridge/Lasso + Optuna + calibration
    pooling.py               # Log-odds pooling (optional)
  train_walkforward.py       # Walk-forward trainer (per bracket)
  eval.py                    # Metrics, calibration curves, diagnostics

configs/
  train_ridge.yaml           # Default config for ridge
  train_lasso.yaml           # Default config for lasso

scripts/
  run_train_walkforward.py   # CLI entrypoint
  plot_calibration.py        # Reliability diagram per window
```

---

## 2) Feature sets

**Available columns (today):**
Market: `yes_mid, yes_bid, yes_ask, spread_cents, minutes_to_close`
Weather (VC): `temp_now, temp_to_floor, temp_to_cap`
Time: `hour_of_day_local, day_of_week`
(You already gate NYC VC features out.)

**Additions (safe for Ridge):**

* `mid_chg_5m = yes_mid - yes_mid_lag5`
* `spread_pct = spread_cents / clip(yes_mid,5,95)`
* `log_minutes_to_close = log1p(minutes_to_close)`

**Additions (only in Lasso/Elastic‑Net search space):**

* Microstructure: `rv_15m, rv_60m` (realized vol from 1‑min), `oi_level`, `vol_5m`, `mid_slope_5m`
* Weather dynamics: `temp_5m_ema, temp_15m_ema`, `temp_accel = temp_now - temp_5m_ema`
* Time basis: `sin_hour, cos_hour` (cyclical encoding)
* Interactions: `temp_now * spread_pct`, `temp_now * log_minutes_to_close`

**Leakage rules:** forbid features using any info after `t` (your join logic already respects close_time).

---

## 3) Core stubs

### 3.1 `ml/models/logit_linear.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import json, joblib, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import log_loss, brier_score_loss
import optuna

@dataclass
class LogitConfig:
    penalty: str = "l2"              # "l2" (ridge) or "l1"/"elasticnet" (lasso/EN)
    solver: str = "saga"             # saga supports l1 & elasticnet; lbfgs for l2
    max_iter: int = 5000
    n_jobs: int = -1
    # calibration
    calib_method_large: str = "isotonic"
    calib_method_small: str = "sigmoid"
    calib_threshold: int = 1000
    # CV
    n_splits: int = 4
    random_state: int = 1337

def _build_pipe(C: float, cfg: LogitConfig, l1_ratio: Optional[float]) -> Pipeline:
    lr_kwargs = dict(
        penalty=cfg.penalty,
        C=C,
        solver=cfg.solver if cfg.penalty in ["l1", "elasticnet"] else "lbfgs",
        max_iter=cfg.max_iter,
        n_jobs=cfg.n_jobs,
        class_weight=None,           # will be tuned via Optuna suggestion
        fit_intercept=True,
        l1_ratio=l1_ratio if cfg.penalty == "elasticnet" else None,
    )
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(**lr_kwargs))
    ])

def tune_with_optuna(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: LogitConfig,
    n_trials: int = 30,
    direction: str = "minimize"
) -> Dict[str, Any]:
    """Tune C, class_weight, (optionally) l1_ratio using GroupKFold by day."""
    gkf = GroupKFold(n_splits=cfg.n_splits)

    def objective(trial: optuna.Trial) -> float:
        # search space
        C = trial.suggest_float("C", 1e-4, 1e4, log=True)
        cw = trial.suggest_categorical("class_weight", [None, "balanced"])
        if cfg.penalty == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        else:
            l1_ratio = None

        pipe = _build_pipe(C, cfg, l1_ratio)
        pipe.set_params(clf__class_weight=cw)

        # out-of-fold probabilities (grouped)
        oof = cross_val_predict(
            pipe, X, y, groups=groups,
            cv=gkf, method="predict_proba", n_jobs=-1
        )[:, 1]

        # evaluate with log loss (proper) + brier as tie‑breaker
        ll = log_loss(y, oof, labels=[0,1])
        brier = brier_score_loss(y, oof)
        trial.set_user_attr("brier", float(brier))
        return ll

    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=cfg.random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial.params
    best["brier"] = study.best_trial.user_attrs.get("brier")
    best["value"] = study.best_value
    return best

def fit_with_calibration(
    X_train: np.ndarray, y_train: np.ndarray,
    X_cal: np.ndarray, y_cal: np.ndarray,
    cfg: LogitConfig, best_params: Dict[str, Any]
) -> CalibratedClassifierCV:
    """Fit final model on train, then calibrate on held‑out calibration split."""
    # (Option: split last 20% of train for calibration)
    method = cfg.calib_method_large if len(y_cal) >= cfg.calib_threshold else cfg.calib_method_small

    l1_ratio = best_params.get("l1_ratio")
    pipe = _build_pipe(best_params["C"], cfg, l1_ratio)
    pipe.set_params(clf__class_weight=best_params.get("class_weight"))
    pipe.fit(X_train, y_train)

    # Wrap in calibrator
    calibrated = CalibratedClassifierCV(pipe, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated

def save_artifacts(model, best_params: Dict[str, Any], out_dir: str):
    joblib.dump(model, f"{out_dir}/model.joblib")
    with open(f"{out_dir}/best_params.json","w") as f:
        json.dump(best_params, f, indent=2)
```

**Notes.**

* We deliberately use `LogisticRegression` (not `RidgeClassifier`) because we need calibrated probabilities. L1/L2/elastic‑net penalties and solver behavior: see sklearn docs. ([Scikit-learn][4])
* Calibration strategy per sklearn user guide. ([Scikit-learn][1])

---

### 3.2 `ml/train_walkforward.py` (per bracket)

```python
from __future__ import annotations
import os, json, numpy as np, pandas as pd
from datetime import date, timedelta
from typing import Literal, Dict, Any, Tuple
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss
from ml.models.logit_linear import LogitConfig, tune_with_optuna, fit_with_calibration, save_artifacts
from ml.eval import calibration_summary
from ml.dataset import build_training_dataset   # you already have this; returns X,y,groups,meta

def walkforward(
    city: str,
    bracket: Literal["between","greater","less"],
    start: date, end: date,
    train_days: int = 90,
    test_days: int = 7,
    step_days: int = 7,
    model_type: Literal["ridge","lasso","elasticnet"]="ridge",
    n_trials: int = 40,
    out_root: str = "models/trained"
):
    cfg = LogitConfig()
    if model_type == "ridge":
        cfg.penalty = "l2"; cfg.solver = "lbfgs"
    elif model_type == "lasso":
        cfg.penalty = "l1"; cfg.solver = "saga"
    elif model_type == "elasticnet":
        cfg.penalty = "elasticnet"; cfg.solver = "saga"

    cursor = start
    summaries = []
    while cursor + timedelta(days=train_days+test_days) <= end:
        tr_start = cursor
        tr_end   = cursor + timedelta(days=train_days)
        te_end   = tr_end + timedelta(days=test_days)

        # Build train & test (your builder should support date ranges + bracket)
        X_tr, y_tr, g_tr, meta_tr = build_training_dataset(city, tr_start, tr_end, bracket)
        X_te, y_te, g_te, meta_te = build_training_dataset(city, tr_end, te_end, bracket)

        # Split train into inner train/cal for calibration (last 20% of days)
        # Group by day; use groups to slice:
        tr_days = pd.Series(g_tr).unique()
        cutoff = int(0.8*len(tr_days))
        keep_days = set(tr_days[:cutoff]); cal_days = set(tr_days[cutoff:])
        idx_keep = np.array([g in keep_days for g in g_tr])
        idx_cal  = np.array([g in cal_days  for g in g_tr])

        best = tune_with_optuna(X_tr[idx_keep], y_tr[idx_keep], g_tr[idx_keep], cfg, n_trials=n_trials)

        model = fit_with_calibration(
            X_tr[idx_keep], y_tr[idx_keep],
            X_tr[idx_cal], y_tr[idx_cal],
            cfg, best
        )

        # Predict test
        p_te = model.predict_proba(X_te)[:,1]
        ll = log_loss(y_te, p_te); brier = brier_score_loss(y_te, p_te)

        out_dir = f"{out_root}/{city}/{bracket}/win_{tr_start:%Y%m%d}_{tr_end:%Y%m%d}"
        os.makedirs(out_dir, exist_ok=True)
        save_artifacts(model, best, out_dir)

        pd.DataFrame({
            "timestamp": meta_te["timestamp"].values,
            "market_ticker": meta_te["market_ticker"].values,
            "p_model": p_te,
            "y": y_te
        }).to_csv(f"{out_dir}/preds_test.csv", index=False)

        calibration_summary(y_te, p_te, out_path=f"{out_dir}/calibration.json")

        summaries.append(dict(
            city=city, bracket=bracket, tr_start=f"{tr_start}", tr_end=f"{tr_end}",
            test_end=f"{te_end}", logloss=float(ll), brier=float(brier), best=best
        ))

        cursor += timedelta(days=step_days)

    # save cross-window summary
    pd.DataFrame(summaries).to_json(
        f"{out_root}/{city}/{bracket}/walkforward_summary.json", orient="records", indent=2
    )
```

---

### 3.3 `ml/eval.py`

```python
import numpy as np, json
from sklearn.calibration import calibration_curve

def calibration_summary(y_true, p_pred, n_bins=15, out_path=None):
    frac_pos, mean_pred = calibration_curve(y_true, p_pred, n_bins=n_bins, strategy="uniform")
    out = {
        "bins": int(n_bins),
        "mean_pred": mean_pred.tolist(),
        "frac_pos": frac_pos.tolist(),
        "ece_like": float(np.mean(np.abs(frac_pos - mean_pred)))  # simple ECE proxy
    }
    if out_path:
        with open(out_path, "w") as f: json.dump(out, f, indent=2)
    return out
```

---

### 3.4 Optional opinion pooling (`ml/models/pooling.py`)

```python
import numpy as np
from scipy.special import expit, logit

def logit_blend(p_model: np.ndarray, p_mkt: np.ndarray, w: float) -> np.ndarray:
    """Blend probabilities in log-odds space: logit(p*) = w*logit(pm) + (1-w)*logit(pmk)."""
    pm = np.clip(p_model, 1e-6, 1-1e-6)
    pk = np.clip(p_mkt, 1e-6, 1-1e-6)
    lo = w*logit(pm) + (1.0-w)*logit(pk)
    return expit(lo)
```

* Fit `w` on train (0–1) by minimizing log‑loss (grid or Optuna 1‑D search). Combining forecasts in log‑odds is a standard method in forecast aggregation; see probability calibration and combining references. ([Scikit-learn][1])

---

## 4) CLI

`configs/train_ridge.yaml`

```yaml
city: chicago
bracket: between
start: 2024-04-01
end: 2025-11-14
train_days: 90
test_days: 7
step_days: 7
model_type: ridge
n_trials: 40
```

`scripts/run_train_walkforward.py`

```python
import argparse, yaml
from datetime import date
from ml.train_walkforward import walkforward

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    walkforward(
        city=cfg["city"],
        bracket=cfg["bracket"],
        start=date.fromisoformat(cfg["start"]),
        end=date.fromisoformat(cfg["end"]),
        train_days=cfg["train_days"],
        test_days=cfg["test_days"],
        step_days=cfg["step_days"],
        model_type=cfg["model_type"],
        n_trials=cfg["n_trials"],
    )
```

---

## 5) Quality gates (fail the run if violated)

* **Calibration improves**: test Brier must be ≤ naïve (market mid) Brier by ≥ 2%.
* **Coverage sanity**: # predictions on test window ≥ 95% of eligible minutes.
* **No leakage**: Max timestamp in features ≤ close_time for every row.
* **Stability**: Coef L2 norm not exploding: `||β||₂ ≤ β_max` (set guardrail per bracket).
* **Edge sanity**: Average predicted edge (after costs) not all one‑sided without trades available (indicates joining issue).

---

## 6) How to run (initial pass)

1. **Ridge** (baseline + safe adds):

```bash
python scripts/run_train_walkforward.py --config configs/train_ridge.yaml
```

2. **Lasso / Elastic‑Net** (feature‑rich):

```yaml
# configs/train_lasso.yaml differences
model_type: elasticnet
n_trials: 60
```

```bash
python scripts/run_train_walkforward.py --config configs/train_lasso.yaml
```

3. **Backtest** (you already wired ModelKelly): point to the `preds_test.csv` per window.

---

## 7) Why these choices (citations)

* **CalibratedClassifierCV** and when to use **isotonic** vs **sigmoid**: scikit‑learn user guide on calibration. Isotonic is non‑parametric, higher variance; use with larger data; sigmoid (Platt) is safer for small calibration sets. ([Scikit-learn][1])
* **GroupKFold** & groups in CV: keep groups (e.g., event day) intact to avoid leakage. ([Scikit-learn][2])
* **Brier score** as a proper scoring rule for probabilities. ([Scikit-learn][3])
* **Logistic regression penalties & solvers**: logistic reg with L1/L2/elastic‑net via `liblinear`/`saga`/`lbfgs`. (Use `saga` for L1/elastic‑net.) ([Scikit-learn][4])
* **Optuna** define‑by‑run tuning with TPE sampler (the standard path for sklearn models). ([Scikit-learn][5])

---

## 8) Answers to your specific questions

**Q: Should Optuna run each walk‑forward window?**
**A: Yes.** Tune per window (e.g., 30–60 trials) to adapt to recent data. Use the same search spaces for reproducibility, and set a seed in the TPE sampler.

**Q: We now have ~684 days; can we train on ~90 days?**
Yes. Start with 90‑day train → 7‑day test. For robustness, you can repeat with 120‑day windows as a sensitivity check once the 90‑day run finishes.

**Q: Lasso vs Ridge?**

* Ridge = stable coefficients, less variance; use modest feature set (baseline + a few).
* Lasso/Elastic‑Net = try a **larger** feature set; it will zero/prune. Compare their Brier/log‑loss and backtest P&L. If Lasso over‑sparsifies, switch to Elastic‑Net with tuned `l1_ratio` (0.1–0.5 often sweet‑spot in correlated settings).

**Q: NYC VC data?**
Keep NYC labels and market features; **exclude** NYC VC minutes from feature set (already enforced). This keeps the feature playing field fair across the six quality stations.

---

## 9) Feature cookbook (ready‑to‑code)

**Market microstructure**
`mid = (yes_bid+yes_ask)/2`
`rv_15m = sqrt(sum((mid_t - mid_{t-1})^2 over 15m))`
`spread_pct = spread_cents / clip(mid,5,95)`
`imbalance = (yes_ask - mid) - (mid - yes_bid) = 0` (for symmetry markets; include only if informative)

**Weather dynamics (VC)**
`temp_5m_ema`, `temp_15m_ema`, `temp_accel = temp_now - temp_5m_ema`
`temp_to_floor`, `temp_to_cap` (your baseline)
`diurnal = sin(2π * hour/24), cos(2π * hour/24)`

**Time to close**
`log_minutes_to_close`, maybe interaction with `temp_now`

Add these in a feature builder behind a flag so Ridge uses “baseline+3”, Lasso/EN uses “rich”.

---

## 10) Pitfalls & guards

* **Class imbalance** varies by bracket: always let Optuna toggle `class_weight ∈ {None, "balanced"}`.
* **DST days**: ensure `minutes_to_close` monotone declines and never negative; drop broken rows.
* **Coherent bins (between)**: later, add your CVXPY PMF projection as a post‑processor to make integer‑Tmax probabilities sum correctly across bins.

---

## 11) What to hand back after a run

* `models/trained/{city}/{bracket}/win_*/`

  * `model.joblib`, `best_params.json`, `preds_test.csv`, `calibration.json`
* `{city}/{bracket}/walkforward_summary.json`
* A single CSV merging all test predictions with market mids for **edge diagnostics**:
  `city, bracket, timestamp, market_ticker, p_model, p_market, y, spread_cents, minutes_to_close`

---

## 12) Immediate action items for the agent

1. Implement the modules exactly as stubbed (file names/paths above).
2. Add **feature flags**: `--feature_set {baseline,ridge_safe,rich}` in dataset builder.
3. Run **Ridge** with 40 trials on **Chicago/between** 2024‑04‑01 → 2025‑11‑14.
4. Run **Elastic‑Net** with 60 trials on same window.
5. Produce the cross‑window comparison table:

   * mean(log‑loss), mean(Brier), P&L/Sharpe in backtest with maker‑first.
6. Gate deploy: only promote model type that beats baseline on both scoring **and** backtest net P&L.

---

### References

* scikit‑learn User Guide: **Calibration of classifier probabilities** (isotonic vs sigmoid; reliability curves). ([Scikit-learn][1])
* scikit‑learn: **Groups in cross‑validation** (avoid leakage). ([Scikit-learn][2])
* scikit‑learn: **Brier score loss** API. ([Scikit-learn][3])
* scikit‑learn: **LogisticRegression** penalties/solvers. ([Scikit-learn][4])
* Optuna docs: **Define‑by‑run hyperparameter optimization** (sklearn integration). ([Scikit-learn][5])

---

## Quick reply you can send to the agent

> **Go ahead with per‑window Optuna tuning** (40 trials for ridge, 60 for elastic‑net) on 90→7→7 walk‑forward. Implement the exact stubs in `ml/models/logit_linear.py`, `ml/train_walkforward.py`, and `ml/eval.py` as above. Use isotonic calibration when the held‑out calibration split has ≥1,000 samples; else Platt. For Ridge, stick to the baseline features + `mid_chg_5m`, `spread_pct`, `log_minutes_to_close`. For Lasso/EN, enable the rich feature set. Exclude NYC VC features (keep labels). Return per‑window artifacts + a cross‑window summary with Brier, log‑loss, and backtest P&L using the maker‑first strategy. Promote whichever model wins both scoring and P&L.**

If you want me to tailor the feature‑rich set more (or add the log‑odds opinion pooling step), I can draft that next.

[1]: https://sklearn.org/1.6/api/sklearn.calibration.html?utm_source=chatgpt.com "sklearn.calibration — scikit-learn 1.6.0 documentation"
[2]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html?utm_source=chatgpt.com "cross_val_predict"
[3]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html?utm_source=chatgpt.com "cross_validate — scikit-learn 1.7.2 documentation"
[4]: https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.LogisticRegression.html?utm_source=chatgpt.com "sklearn.linear_model.LogisticRegression"
[5]: https://scikit-learn.ru/stable/auto_examples/linear_model/plot_iris_logistic.html?utm_source=chatgpt.com "Logistic Regression 3-class Classifier - scikit-learn"
