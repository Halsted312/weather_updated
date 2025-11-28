
# DELTA_TEMP_INTRADAY_MODELS.md  
Intraday Δ-Models for Kalshi Temperature Settlement (Logistic + CatBoost+Optuna)

This doc defines a **full intraday modeling pipeline** to estimate a **distribution over the final daily high** (the NWS/CLI / Kalshi settlement temperature) for each `(city, day)` as the day unfolds.

We define and implement **two models**:

1. **Model 1 – Multinomial Logistic Δ-Model (Elastic Net + Platt Calibration)**  
   - One global multi-class model on  
     `Δ = T_settle − T_base`,  
     where `T_base` is a deterministic baseline from Visual Crossing partial-day temps.  
   - Trained on **intraday snapshots**, calibrated with `CalibratedClassifierCV(method="sigmoid")` (Platt scaling).   

2. **Model 2 – CatBoost Δ-Model (CatBoostClassifier + Optuna + Platt Calibration)**  
   - Same target Δ, using a **CatBoost multiclass classifier** with categorical features and non-linear interactions.   
   - Hyperparameters tuned with **Optuna** (10 trials to start), then wrapped in a Platt-calibrated classifier.   

Both models:

- Use **time-aware** splits (`TimeSeriesSplit`) to avoid looking into the future.   
- Have a clear **train/test split by day** (no training on test days).
- Produce **calibrated probabilities** that you can directly compare to Kalshi-implied probabilities (via Brier score, calibration curves, etc.).   

The twist in this doc vs earlier versions:  
> **All features are computed from *partial-day* data only.**  
At any intraday time `τ`, the model only sees VC data up to `τ`, not the full shape of the day.

---

## 0. Intraday Framing: Snapshots, Not Full Days

### 0.1 What is a “snapshot”?

A **snapshot** is:

> A view of `(city, day)` at some local time `τ` (e.g., 10:00, 13:00, 18:00, 21:00), using only Visual Crossing observations up to `τ`.

For each snapshot we know:

- The **final settled high** `T_settle` (from NWS/IEM/CLI) – the label.
- All VC 5-min temps **up to τ**, but **not after**.

We then define:

- `T_base(τ)` = a deterministic baseline derived from partial-day data up to `τ`  
  (e.g., `round(max(temp_so_far))`).
- `Δ(τ) = T_settle − T_base(τ)`.

The model learns:

\[
P\big(\Delta(\tau) = d \mid \text{features up to } \tau\big), \quad d \in \{-2,-1,0,+1,+2\}.
\]

From that, you get:

- `P(T_settle = t | τ)` = `P(Δ = t − T_base(τ) | τ)`  
- `P(T_settle ≥ K | τ)` or `P(a ≤ T_settle ≤ b | τ)` by summing over `t`.

### 0.2 Snapshot grid

To keep things simple and statistically sane:

- Define a fixed set of **snapshot times** in local time per day, e.g.:

  ```python
  SNAPSHOT_HOURS = [10, 12, 14, 16, 18, 20, 22, 23]  # local hours
````

* For each `(city, day)` and each `h` in `SNAPSHOT_HOURS`:

  * If there is enough VC data up to `h` (say ≥ 4 hours), create a snapshot.
  * Compute features using **only VC data with `datetime_local < h:00`**.

This gives you a rich intraday dataset of `(city, day, snapshot_hour)` rows, each with the same label `T_settle` but different partial-day predictors.

At inference time, you:

* At, say, 21:55 local, compute the features up to 21:55, set `snapshot_hour = 21`, and call the same model.

---

## 1. Intraday Feature Engineering (Partial-Day)

We’ll construct a **feature row per snapshot**:

```text
city, day, snapshot_hour, settle_f, t_base(τ), delta(τ) = settle_f - t_base(τ), ...
```

### 1.1 Base stats from partial-day VC temps

Let `temps_so_far` be all 5-min VC temps up to `τ` (in local time). We compute:

* `vc_max_f_so_far`
* `vc_min_f_so_far`
* `vc_mean_f_so_far`
* `vc_std_f_so_far`
* `vc_q10_f_so_far`, `vc_q25_f_so_far`, `vc_q50_f_so_far`, `vc_q75_f_so_far`, `vc_q90_f_so_far`
* `vc_frac_part_so_far = vc_max_f_so_far − round(vc_max_f_so_far)`
* `num_samples_so_far`

Skeleton:

```python
import numpy as np

def base_stats_partial(temps_so_far: list[float]) -> dict:
    if not temps_so_far:
        return {}
    arr = np.asarray(temps_so_far, dtype=float)
    q10, q25, q50, q75, q90 = np.percentile(arr, [10, 25, 50, 75, 90])
    max_f = float(arr.max())
    return {
        "vc_max_f_so_far": max_f,
        "vc_min_f_so_far": float(arr.min()),
        "vc_mean_f_so_far": float(arr.mean()),
        "vc_std_f_so_far": float(arr.std(ddof=1)),
        "vc_q10_f_so_far": float(q10),
        "vc_q25_f_so_far": float(q25),
        "vc_q50_f_so_far": float(q50),
        "vc_q75_f_so_far": float(q75),
        "vc_q90_f_so_far": float(q90),
        "vc_frac_part_so_far": max_f - round(max_f),
        "num_samples_so_far": int(arr.size),
    }
```

### 1.2 Partial-day base rule and Δ

For each snapshot:

* `t_base = round(vc_max_f_so_far)`  (or your chosen deterministic rule).
* `delta = settle_f − t_base`  (target for the Δ model).

```python
def delta_target(settle_f: int, vc_max_f_so_far: float) -> dict:
    t_base = int(round(vc_max_f_so_far))
    delta = int(settle_f - t_base)
    return {
        "t_base": t_base,
        "delta": delta,
        "abs_delta": abs(delta),
    }
```

### 1.3 Partial-day plateau / shape features

We still care about “spike vs plateau” but only using temps up to `τ`.

Assume 5-min cadence and we have `temps_so_far` and `timestamps_local_so_far`.

```python
def shape_partial(temps_so_far: list[float],
                  timestamps_local_so_far: list,
                  t_base: int,
                  step_minutes: int = 5) -> dict:
    import numpy as np

    if not temps_so_far:
        return {}

    arr = np.asarray(temps_so_far, dtype=float)
    n = arr.size
    feats = {}

    # Plateaus around base
    def minutes_ge(thresh: float) -> int:
        return int((arr >= thresh).sum() * step_minutes)

    feats["minutes_ge_base"]    = minutes_ge(t_base)
    feats["minutes_ge_base_p1"] = minutes_ge(t_base + 1)
    feats["minutes_ge_base_m1"] = minutes_ge(t_base - 1)

    def max_run_ge(thresh: float) -> int:
        best = run = 0
        for t in arr:
            if t >= thresh:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best * step_minutes

    feats["max_run_ge_base"]    = max_run_ge(t_base)
    feats["max_run_ge_base_p1"] = max_run_ge(t_base + 1)
    feats["max_run_ge_base_m1"] = max_run_ge(t_base - 1)

    # spike vs plateau: difference between max and 2nd max
    top3 = sorted(arr, reverse=True)[:3]
    second = top3[1] if len(top3) > 1 else top3[0]
    feats["max_minus_second_max"] = float(top3[0] - second)

    # morning vs afternoon vs evening behavior so far
    if timestamps_local_so_far:
        hours = np.array([ts.hour for ts in timestamps_local_so_far])
        feats["max_morning_f_so_far"] = float(arr[hours < 12].max()) if (hours < 12).any() else None
        feats["max_afternoon_f_so_far"] = float(arr[(hours >= 12) & (hours < 17)].max()) if ((hours >= 12) & (hours < 17)).any() else None
        feats["max_evening_f_so_far"] = float(arr[hours >= 17].max()) if (hours >= 17).any() else None

    # 30-min slopes so far
    window = 30 // step_minutes
    if n >= window + 1:
        diffs = arr[window:] - arr[:-window]
        feats["slope_max_30min_up_so_far"] = float(diffs.max())
        feats["slope_max_30min_down_so_far"] = float(diffs.min())
    else:
        feats["slope_max_30min_up_so_far"] = None
        feats["slope_max_30min_down_so_far"] = None

    return feats
```

### 1.4 Intraday rule meta-features (partial-day)

Your deterministic rules (max_of_rounded, plateau_20min, c_first, etc.) can be applied to `temps_so_far` just like to full-day temps.

```python
from analysis.temperature.rules import ALL_RULES  # adapt import

def rule_features_partial(temps_so_far: list[float],
                          settle_f: int) -> dict:
    feats = {}
    preds = []
    for name, fn in ALL_RULES.items():
        try:
            pred = fn(temps_so_far)
        except Exception:
            pred = None
        feats[f"pred_{name}_so_far"] = pred
        if pred is not None:
            preds.append(pred)
            feats[f"err_{name}_so_far"] = pred - settle_f
        else:
            feats[f"err_{name}_so_far"] = None

    if preds:
        feats["range_pred_rules_so_far"] = max(preds) - min(preds)
        feats["num_distinct_preds_so_far"] = len(set(preds))
        feats["disagree_flag_so_far"] = int(len(set(preds)) > 1)
    else:
        feats["range_pred_rules_so_far"] = None
        feats["num_distinct_preds_so_far"] = None
        feats["disagree_flag_so_far"] = None

    return feats
```

### 1.5 Lag & calendar features (unchanged)

Lag and seasonality features don’t depend on how far into the day we are; they’re per `(city, day)`:

* `settle_f_lag1`, `settle_f_lag2`, `settle_f_lag7`
* `vc_max_f_lag1`, `vc_max_f_lag7`
* `delta_vcmax_lag1 = vc_max_f_today_so_far − vc_max_f_lag1`
* `doy_sin`, `doy_cos`, `week_sin`, `week_cos`
  (day-of-year and week-of-year encoded as sin/cos).
  These cyclical encodings are standard for time-related models in sklearn.

You can compute these **after** building the snapshot table by grouping on `(city, day)` and using `.shift()`.

### 1.6 Context and quality features

* Multi-variable weather if available (dewpoint, RH, wind, precip, clouds) – aggregated **up to τ**.
* Quality flags (missing fraction, max gap so far, etc.).
* Snapshot-specific features like `snapshot_hour` (and sin/cos of it).

```python
def snapshot_calendar_features(day, snapshot_hour: int) -> dict:
    import math
    doy = day.timetuple().tm_yday
    week = day.isocalendar()[1]
    return {
        "snapshot_hour": snapshot_hour,
        "snapshot_hour_sin": math.sin(2 * math.pi * snapshot_hour / 24.0),
        "snapshot_hour_cos": math.cos(2 * math.pi * snapshot_hour / 24.0),
        "doy_sin": math.sin(2 * math.pi * doy / 365.25),
        "doy_cos": math.cos(2 * math.pi * doy / 365.25),
        "week_sin": math.sin(2 * math.pi * week / 52.0),
        "week_cos": math.cos(2 * math.pi * week / 52.0),
        "month": day.month,
        "is_weekend": int(day.weekday() >= 5),
    }
```

---

## 2. Snapshot Dataset Construction (No Forward-Looking)

We now build a **snapshot-level feature table**:

```text
city, day, snapshot_hour, settle_f, t_base(τ), delta(τ), ... features ...
```

Key constraints:

* For each snapshot `(city, day, τ)`:

  * Use only VC obs with `datetime_local < τ`.
  * Do *not* use any `T_settle`-derived info beyond the fact that `T_settle` is the label.
* Train/test split is by `day`, not by snapshot → **no training on future days**.
* For CV inside training, we use **time-based folds** (expanding window).

Skeleton builder:

```python
# scripts/build_temp_features_intraday.py

from datetime import datetime, time
import pandas as pd

SNAPSHOT_HOURS = [10, 12, 14, 16, 18, 20, 22, 23]  # local hours

def build_intraday_feature_rows(day_series_iter, vc_loader):
    """
    day_series_iter: yields DaySeries(city, day, settle_f, etc.)
    vc_loader(city, day) -> DataFrame with VC minute obs including datetime_local, temp_f, ...
    """
    rows = []
    for ds in day_series_iter:
        df_vc = vc_loader(ds.city, ds.day).sort_values("datetime_local")
        for h in SNAPSHOT_HOURS:
            cutoff = datetime(ds.day.year, ds.day.month, ds.day.day, h, 0, 0)
            df_so_far = df_vc[df_vc["datetime_local"] < cutoff]
            if df_so_far.empty:
                continue

            temps_so_far = df_so_far["temp_f"].tolist()
            timestamps_local_so_far = df_so_far["datetime_local"].tolist()

            base = base_stats_partial(temps_so_far)
            delta_info = delta_target(ds.settle_f, base["vc_max_f_so_far"])
            shape = shape_partial(temps_so_far, timestamps_local_so_far, delta_info["t_base"])
            rules = rule_features_partial(temps_so_far, ds.settle_f)
            cal  = snapshot_calendar_features(ds.day, h)
            # add context + quality if needed

            row = {
                "city": ds.city,
                "day": ds.day,
                "snapshot_hour": h,
                "settle_f": ds.settle_f,
                **base,
                **delta_info,
                **shape,
                **rules,
                **cal,
            }
            rows.append(row)

    df_snap = pd.DataFrame(rows)
    # Add lags by (city, day)
    df_snap = add_lag_and_calendar_features_by_day(df_snap)
    df_snap.to_parquet("reports/temperature_features_intraday.parquet", index=False)
    return df_snap
```

`add_lag_and_calendar_features_by_day` can reuse the earlier lag logic, but now applied at the day level and broadcast to each snapshot of that day.

---

## 3. Common Modeling Setup (Intraday Snapshot Data)

We assume `reports/temperature_features_intraday.parquet` exists.

```python
# intraday_common_setup.py

import pandas as pd
from datetime import date
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

FEATURE_PATH_INTRADAY = "reports/temperature_features_intraday.parquet"

def load_intraday_table() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_PATH_INTRADAY)
    # sort by (day, snapshot_hour, city) to preserve temporal order
    df = df.sort_values(["day", "snapshot_hour", "city"]).reset_index(drop=True)
    # optional: restrict Δ to [-2, 2]
    df = df[df["delta"].between(-2, 2)]
    return df


def make_train_test_split(df: pd.DataFrame, cutoff: date):
    train_mask = df["day"] < cutoff
    df_train = df[train_mask].reset_index(drop=True)
    df_test  = df[~train_mask].reset_index(drop=True)
    return df_train, df_test


def build_X_y_intraday(df: pd.DataFrame):
    y = df["delta"]  # multiclass target

    num_cols = [
        "vc_max_f_so_far", "vc_min_f_so_far", "vc_mean_f_so_far", "vc_std_f_so_far",
        "vc_q10_f_so_far", "vc_q25_f_so_far", "vc_q50_f_so_far", "vc_q75_f_so_far", "vc_q90_f_so_far",
        "vc_frac_part_so_far",
        "t_base", "abs_delta",  # t_base is computed but we won't use delta as a feature
        "minutes_ge_base", "minutes_ge_base_p1", "minutes_ge_base_m1",
        "max_run_ge_base", "max_run_ge_base_p1", "max_run_ge_base_m1",
        "max_minus_second_max",
        "max_morning_f_so_far", "max_afternoon_f_so_far", "max_evening_f_so_far",
        "slope_max_30min_up_so_far", "slope_max_30min_down_so_far",
        # rule preds and errors
        "pred_max_round_so_far", "pred_ceil_max_so_far", "pred_floor_max_so_far",
        "pred_plateau_20min_so_far", "pred_ignore_singletons_so_far", "pred_c_first_so_far",
        "err_max_round_so_far", "err_ceil_max_so_far", "err_floor_max_so_far",
        "err_plateau_20min_so_far", "err_ignore_singletons_so_far", "err_c_first_so_far",
        "range_pred_rules_so_far", "num_distinct_preds_so_far",
        # snapshot & calendar
        "snapshot_hour", "snapshot_hour_sin", "snapshot_hour_cos",
        "doy_sin", "doy_cos", "week_sin", "week_cos",
        "month", "is_weekend",
        # lags (added later)
        "settle_f_lag1", "settle_f_lag2", "settle_f_lag7",
        "vc_max_f_lag1", "vc_max_f_lag7", "delta_vcmax_lag1",
        # quality (optional)
        "num_samples_so_far",
    ]

    cat_cols = ["city"]

    X = df[num_cols + cat_cols]

    preproc = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return X, y, preproc


def make_time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    # Uses ordered rows (day, snapshot_hour) for CV, avoiding future-to-past leakage
    return TimeSeriesSplit(n_splits=n_splits)
```

---

## 4. **Model 1** – Multinomial Logistic Δ-Model (Elastic Net + Platt Scaling)

### 4.1 Why this model?

* Multinomial logistic regression is an established method for multi-class problems and supports ℓ2 / elastic net regularization in scikit-learn.
* Wrapping it in `CalibratedClassifierCV(method="sigmoid")` implements **Platt scaling** for probability calibration, recommended when data is moderately sized and distortions are roughly sigmoid-shaped.
* Combined with `TimeSeriesSplit`, this gives **time-aware, calibrated intraday probabilities** without leakage.

### 4.2 Training & evaluation script

```python
# model1_intraday_logistic.py

from datetime import date
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    brier_score_loss,
    classification_report,
)

from intraday_common_setup import (
    load_intraday_table,
    make_train_test_split,
    build_X_y_intraday,
    make_time_series_cv,
)


def train_and_eval_intraday_logistic(cutoff: date):
    # 1) Load and split
    df = load_intraday_table()
    df_train, df_test = make_train_test_split(df, cutoff=cutoff)

    X_train, y_train, preproc = build_X_y_intraday(df_train)
    X_test,  y_test,  _       = build_X_y_intraday(df_test)

    # 2) Base multinomial logistic with elastic net penalty
    base_lr = LogisticRegression(
        multi_class="multinomial",
        solver="saga",          # needed for elasticnet penalty 
        penalty="elasticnet",
        l1_ratio=0.5,           # mix of L1/L2, tune later
        C=1.0,                  # inverse regularization strength
        max_iter=4000,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("pre", preproc),
            ("clf", base_lr),
        ]
    )

    # 3) Platt calibration on TRAIN ONLY with time-series CV
    tscv = make_time_series_cv(n_splits=5)
    calibrated_lr = CalibratedClassifierCV(
        base_estimator=pipe,
        method="sigmoid",   # Platt scaling
        cv=tscv,
    )  # scikit-learn supports this directly 

    calibrated_lr.fit(X_train, y_train)

    # 4) Evaluate on holdout TEST set (no days from future)
    y_pred = calibrated_lr.predict(X_test)
    proba  = calibrated_lr.predict_proba(X_test)  # shape: (n_samples, n_classes)

    acc_delta = accuracy_score(y_test, y_pred)
    mae_delta = mean_absolute_error(y_test, y_pred)
    print(f"[Model1] Δ accuracy (test): {acc_delta:.4f}")
    print(f"[Model1] Δ MAE (test):      {mae_delta:.4f}")
    print("\nΔ classification report (test):")
    print(classification_report(y_test, y_pred))

    # 5) Settlement-level metrics at snapshot level
    df_test_eval = df_test.copy()
    df_test_eval["delta_pred"] = y_pred
    df_test_eval["t_pred"] = df_test_eval["t_base"] + df_test_eval["delta_pred"]
    t_mae = mean_absolute_error(df_test_eval["settle_f"], df_test_eval["t_pred"])
    t_acc = (df_test_eval["settle_f"] == df_test_eval["t_pred"]).mean()
    print(f"[Model1] Settlement accuracy (per-snapshot, test): {t_acc:.4f}")
    print(f"[Model1] Settlement MAE (°F, per-snapshot, test):  {t_mae:.4f}")

    # 6) Bracket event calibration example: T_settle >= K
    K = 90
    d_classes = np.sort(y_train.unique())
    t_base_test   = df_test_eval["t_base"].to_numpy()
    t_settle_test = df_test_eval["settle_f"].to_numpy()

    p_bracket = []
    for i in range(len(df_test_eval)):
        probs_i = proba[i]          # Δ distribution
        t_base_i = t_base_test[i]
        T_values = t_base_i + d_classes
        p = probs_i[T_values >= K].sum()
        p_bracket.append(p)

    p_bracket = np.array(p_bracket)
    y_bracket = (t_settle_test >= K).astype(int)

    brier = brier_score_loss(y_bracket, p_bracket)  # Brier measures calibration for binary events 
    print(f"[Model1] Brier score for event T>= {K} (test): {brier:.4f}")

    prob_true, prob_pred = calibration_curve(y_bracket, p_bracket, n_bins=10)
    calib_df = pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true})
    calib_df.to_csv("reports/model1_intraday_calibration_T_ge_90.csv", index=False)
    print("Saved calibration curve points to reports/model1_intraday_calibration_T_ge_90.csv")

    # 7) Save model
    import joblib
    joblib.dump(calibrated_lr, "models/model1_intraday_logistic_delta_calibrated.pkl")
    print("Saved Model1 to models/model1_intraday_logistic_delta_calibrated.pkl")


if __name__ == "__main__":
    # e.g., train on days < 2025-01-01, test on 2025+ days
    cutoff_day = date(2025, 1, 1)
    train_and_eval_intraday_logistic(cutoff=cutoff_day)
```

> **Agent instructions for Model 1**
>
> 1. Build `temperature_features_intraday.parquet` via `build_temp_features_intraday.py` first.
> 2. Run `model1_intraday_logistic.py` with a sensible `cutoff_day`.
> 3. Capture:
>
>    * Δ accuracy & MAE on test.
>    * Settlement accuracy & MAE.
>    * Brier score + calibration curve CSV for T ≥ 90°F.
> 4. Save the model `models/model1_intraday_logistic_delta_calibrated.pkl` for later use in live trading code.

---

## 5. **Model 2** – CatBoost Δ-Model (CatBoostClassifier + Optuna + Platt)

### 5.1 Why this model?

* **CatBoostClassifier** supports multiclass classification with `MultiClass` loss and native categorical handling.
* CatBoost is strong on tabular data and can model non-linearities and interactions without manual feature crosses.
* Optuna provides efficient Bayesian optimization of hyperparameters with few trials.
* We wrap CatBoost in `CalibratedClassifierCV(method="sigmoid")` to correct probability distortions.

### 5.2 Dependencies

```bash
pip install catboost optuna
```

(Per CatBoost and Optuna docs. )

### 5.3 CatBoost + Optuna + Platt skeleton

```python
# model2_intraday_catboost_optuna.py

from datetime import date
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    brier_score_loss,
)

from catboost import CatBoostClassifier, Pool
import optuna

from intraday_common_setup import (
    load_intraday_table,
    make_train_test_split,
    build_X_y_intraday,
    make_time_series_cv,
)


def create_catboost_pool(X_df: pd.DataFrame, y: pd.Series):
    """
    X_df: raw features including 'city' as categorical.
    """
    cat_features = [X_df.columns.get_loc("city")]  # index of 'city'
    pool = Pool(data=X_df, label=y, cat_features=cat_features)
    return pool, cat_features


def make_ts_folds_for_optuna(X_df: pd.DataFrame, y: pd.Series, n_splits: int = 3):
    """
    TimeSeriesSplit folds over snapshot index within TRAIN.
    Ensures we train on earlier rows and validate on later rows. 
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_idx, valid_idx in tscv.split(X_df):
        folds.append((train_idx, valid_idx))
    return folds


def objective_catboost(trial: optuna.Trial,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       folds):
    """
    Optuna objective: minimize mean multi-class logloss across time-series folds.
    """
    params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 8.0, log=True),
        "iterations": trial.suggest_int("iterations", 150, 400),
        "random_seed": 42,
        "verbose": False,
    }

    X_values = X_train.reset_index(drop=True)
    y_values = y_train.reset_index(drop=True)

    losses = []
    for train_idx, valid_idx in folds:
        X_tr = X_values.iloc[train_idx]
        y_tr = y_values.iloc[train_idx]
        X_va = X_values.iloc[valid_idx]
        y_va = y_values.iloc[valid_idx]

        pool_tr, cat_feats = create_catboost_pool(X_tr, y_tr)
        pool_va, _        = create_catboost_pool(X_va, y_va)

        model = CatBoostClassifier(**params)
        model.fit(pool_tr, eval_set=pool_va)

        preds_proba = model.predict_proba(pool_va)  # multi-class probs 

        # manual logloss
        eps = 1e-15
        y_true = y_va.to_numpy()
        # assume y is encoded as 0..C-1 classes
        p = preds_proba[np.arange(len(y_true)), y_true]
        p = np.clip(p, eps, 1 - eps)
        loss = -np.mean(np.log(p))
        losses.append(loss)

    return float(np.mean(losses))


def tune_catboost_with_optuna(X_train: pd.DataFrame,
                              y_train: pd.Series,
                              n_trials: int = 10):
    folds = make_ts_folds_for_optuna(X_train, y_train, n_splits=3)

    def objective(trial):
        return objective_catboost(trial, X_train, y_train, folds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:", study.best_trial.number)
    print("Best logloss:", study.best_trial.value)
    print("Best params:", study.best_trial.params)

    best_params = study.best_trial.params
    final_params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "random_seed": 42,
        "verbose": False,
        **best_params,
    }

    X_train_reset = X_train.reset_index(drop=True)
    y_train_reset = y_train.reset_index(drop=True)
    pool_train, cat_feats = create_catboost_pool(X_train_reset, y_train_reset)

    final_model = CatBoostClassifier(**final_params)
    final_model.fit(pool_train)

    return final_model, cat_feats, study


def train_and_eval_intraday_catboost(cutoff: date, n_trials: int = 10):
    # 1) Load & split
    df = load_intraday_table()
    df_train, df_test = make_train_test_split(df, cutoff=cutoff)

    # Use the same feature set as Model 1, but we skip sklearn preproc for CatBoost.
    all_X, _, _ = build_X_y_intraday(df)  # to get column list
    feature_cols = all_X.columns.tolist()

    X_train = df_train[feature_cols]
    y_train = df_train["delta"]
    X_test  = df_test[feature_cols]
    y_test  = df_test["delta"]

    # 2) Hyperparam tuning on TRAIN ONLY
    cb_model, cat_features, study = tune_catboost_with_optuna(
        X_train, y_train, n_trials=n_trials
    )

    # 3) Wrap CatBoost in Platt calibration (TRAIN ONLY)
    class CatBoostWrapper:
        def __init__(self, base_model, feature_cols, cat_features):
            self.base_model = base_model
            self.feature_cols = feature_cols
            self.cat_features = cat_features

        def fit(self, X, y):
            X_df = pd.DataFrame(X, columns=self.feature_cols)
            pool_tr, _ = create_catboost_pool(X_df, y)
            self.base_model.fit(pool_tr)
            return self

        def predict_proba(self, X):
            X_df = pd.DataFrame(X, columns=self.feature_cols)
            pool = Pool(data=X_df, cat_features=self.cat_features)
            return self.base_model.predict_proba(pool)

        def predict(self, X):
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)

    base_wrapper = CatBoostWrapper(cb_model, feature_cols, cat_features)

    tscv = make_time_series_cv(n_splits=5)
    calibrated_cb = CalibratedClassifierCV(
        base_estimator=base_wrapper,
        method="sigmoid",  # Platt scaling 
        cv=tscv,
    )

    calibrated_cb.fit(X_train, y_train)

    # 4) Evaluate on TEST set
    y_pred = calibrated_cb.predict(X_test)
    proba  = calibrated_cb.predict_proba(X_test)

    acc_delta = accuracy_score(y_test, y_pred)
    mae_delta = mean_absolute_error(y_test, y_pred)
    print(f"[Model2] Δ accuracy (test): {acc_delta:.4f}")
    print(f"[Model2] Δ MAE (test):      {mae_delta:.4f}")

    df_test_eval = df_test.copy()
    df_test_eval["delta_pred"] = y_pred
    df_test_eval["t_pred"] = df_test_eval["t_base"] + df_test_eval["delta_pred"]
    t_mae = mean_absolute_error(df_test_eval["settle_f"], df_test_eval["t_pred"])
    t_acc = (df_test_eval["settle_f"] == df_test_eval["t_pred"]).mean()
    print(f"[Model2] Settlement accuracy (per-snapshot, test): {t_acc:.4f}")
    print(f"[Model2] Settlement MAE (°F, per-snapshot, test):  {t_mae:.4f}")

    # Bracket calibration
    K = 90
    d_classes = np.sort(y_train.unique())
    t_base_test   = df_test_eval["t_base"].to_numpy()
    t_settle_test = df_test_eval["settle_f"].to_numpy()

    p_bracket = []
    for i in range(len(df_test_eval)):
        probs_i = proba[i]
        t_base_i = t_base_test[i]
        T_values = t_base_i + d_classes
        p = probs_i[T_values >= K].sum()
        p_bracket.append(p)

    p_bracket = np.array(p_bracket)
    y_bracket = (t_settle_test >= K).astype(int)

    brier = brier_score_loss(y_bracket, p_bracket)
    print(f"[Model2] Brier score for event T>= {K} (test): {brier:.4f}")

    prob_true, prob_pred = calibration_curve(y_bracket, p_bracket, n_bins=10)
    calib_df = pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true})
    calib_df.to_csv("reports/model2_intraday_calibration_T_ge_90.csv", index=False)
    print("Saved calibration curve points to reports/model2_intraday_calibration_T_ge_90.csv")

    # 5) Save model & study
    import joblib
    joblib.dump(calibrated_cb, "models/model2_intraday_catboost_delta_calibrated.pkl")
    joblib.dump(study, "models/model2_intraday_optuna_study.pkl")
    print("Saved Model2 and Optuna study.")

    return calibrated_cb, study


if __name__ == "__main__":
    cutoff_day = date(2025, 1, 1)
    train_and_eval_intraday_catboost(cutoff=cutoff_day, n_trials=10)
```

> **Agent instructions for Model 2**
>
> 1. Install `catboost` and `optuna`.
> 2. Run `model2_intraday_catboost_optuna.py` with `cutoff_day` matching Model 1 and `n_trials=10`.
> 3. Log:
>
>    * Best Optuna parameters & logloss.
>    * Δ & T_settle metrics on test.
>    * Brier score + calibration CSV.
> 4. Save the calibrated CatBoost model and Optuna study under `models/`.

---

## 6. How to Use These Models Live (Intraday)

### 6.1 When to call the model?

Because training is on snapshots at fixed hours, you have at least three options:

1. **Fixed-time snapshots** (simple & robust):

   * Decide that trading decisions will be made at e.g. 10:00, 13:00, 16:00, 19:00, 22:00 local.
   * Train models on exactly those `SNAPSHOT_HOURS`.
   * Live: at each such time, compute features with data up to now and call the model.

2. **Near-end-of-day model**:

   * Focus on snapshots at 22:00 / 23:00 / 23:50 local, where the high is almost always known.
   * Strongest accuracy and narrower uncertainty bands for last-minute trading.

3. **Heuristic triggers** (more advanced):

   * E.g., run the model whenever:

     * Current max has been stable (plateau) for ≥ 2 hours, and
     * Last hour’s temps are ≥ 1 std dev below that max.
   * In code: implement this heuristic on `temps_so_far` and call the model once the triggers fire.

You don’t need to *train* a separate model for each pattern; the model sees `snapshot_hour`, shape features, and plateau stats and learns how informative partial days are at different times.

### 6.2 Walk-forward usage pattern

For a given `(city, day)`:

1. As VC 5-min obs stream in, keep updating:

   * `temps_so_far`
   * Partial-day stats & plateau metrics
   * `t_base(τ)`

2. At time `τ` (for which you want a probability distribution):

   * Build a feature row exactly as you did for training (but only with data up to `τ`).
   * Run through the chosen model (e.g., CatBoost Δ-model).
   * Get `p(Δ=d | τ)` for d ∈ {−2,…,+2}.
   * Convert to `P(T_settle = t | τ)` and then to bracket probabilities.

3. Compare `P_model(bracket)` to `P_market(bracket)` and size positions accordingly.

The **no-lookahead guarantee** is baked into:

* How features are constructed (only `temps_so_far`),
* The daily split on train vs test days, and
* The use of `TimeSeriesSplit` for intratrain calibration & CV, which is designed for time series and avoids training on future data when evaluating on past.

---

## 7. Comparison & Metrics

Once both models have been run:

1. Compare on the *same test set*:

   * **Δ accuracy / MAE**.
   * **Settlement accuracy / MAE** at snapshot level.
2. For key bracket events (e.g., T ≥ 80, 85, 90, 95°F):

   * Compute Brier scores, which directly measure calibration quality.
   * Plot reliability curves using the `calibration_curve` outputs.
3. Examine **performance vs snapshot_hour**:

   * Group test rows by `snapshot_hour` and compute metrics per hour.
   * Expect:

     * **Higher uncertainty** (worse metrics) early in the day.
     * **Sharper, more accurate** predictions near late evening.

Winner model is not necessarily the one with best accuracy; for trading, you care most about:

* Probability calibration quality (Brier, reliability plots),
* Stability / robustness across cities and time,
* How well it identifies days where the distribution is tightly peaked vs spread out (for sizing).

---

## 8. No-Leakage Checklist (Intraday)

* [ ] Features at snapshot `τ` use only VC obs with `datetime_local < τ`.
* [ ] Train/test split is by `day` (e.g., train: 2022–2024, test: 2025), never mixing days.
* [ ] Calibration (`CalibratedClassifierCV`) & Optuna CV are run **only on training data** with `TimeSeriesSplit` (temporal order preserved).
* [ ] No  `shuffle=True` anywhere in time splits.
* [ ] When analyzing performance, test metrics are computed *only* on the holdout days.

If all boxes are ticked, you have:

* Two intraday Δ-models (logistic & CatBoost+Optuna),
* A clean, partial-day feature factory with no forward-looking leakage,
* Calibrated probability distributions you can query at any time of day to back your Kalshi trading decisions.
