# Temperature Settlement Modeling Plan

This doc defines **two full modeling pipelines** you can hand to your coding agent:

- **Model 1 (my pick):**  
  Regularized **multinomial logistic regression** over discrete temperatures with **time-series CV** and **probability calibration**.

- **Model 2 (your pick):**  
  **CatBoostClassifier + Optuna** with time-aware CV and 10 trials, giving a strong non-linear baseline.

Both are designed to:

- Use **rich features** (including your rule-based outputs).  
- Avoid **data leakage** (no training on future days; no using future-of-day information in intraday features).   
- Produce a **full probability distribution over integer temps**, so you can price **all brackets** from a single model.  
- Support **intraday use**: you can get a distribution at multiple times during the day, not just at midnight.

---

## 0. Global Design: How We Build the Dataset

### 0.1. Units of data

We’ll build a dataset indexed by:

- `city`
- `day`  (local calendar date)
- `cutoff_time` (local time-of-day when we make the prediction)

Each row is:

> “At time `cutoff_time` on this `city`+`day`, using only information available up to that time, what will the **final daily max temperature** (settlement) be?”

This gives you:

- **End-of-day** predictions (e.g., cutoff `23:50` for backtesting/validation).  
- **Intraday** predictions (e.g., hourly cutoffs: 10:00, 12:00, 14:00, …) using **partial-day features** only.

The **target** for all rows is the same:

- `settle_f` = integer °F daily high from your settlement table (proxy for Kalshi’s CLI-based settlement).

### 0.2. No look-ahead: partial-day feature logic

For each `(city, day, cutoff_time)`:

- Only use VC 5-min obs with `datetime_local <= cutoff_time`.
- Do **not** use any obs after the cutoff.
- Do not use “shape features” that depend on the future (e.g., “temp goes down 1 std after plateau”) — instead use only what you’ve seen **so far** (e.g., “last 2 hours are flat”, “last temp is below max_so_far”, etc.).

This is how we ensure **no future information leaks into features**.

### 0.3. Time-series cross-validation, grouped by day

Key constraints:

- You cannot train on **future days** and test on earlier days.  
- You should not train on any part of day D if you’re evaluating on another part of day D (to avoid “same-day leakage” via labels).  

So we:

1. Define the sorted list of **unique days** per city:  
   `unique_days = sorted(df['day'].unique())`.

2. Create **blocked time-series folds** on these `unique_days` using a custom splitter:

   - For fold k:
     - `train_days` = earliest days up to some index.
     - `test_days` = the next block of days.
   - All intraday rows for a given day go to the **same** side (train or test).

This is essentially a **grouped time-series split** — recommended for time-ordered data to avoid leakage.   

---

## 1. Feature Engineering

Below is a feature set you can compute for each `(city, day, cutoff_time)`. Some features require per-city/day aggregation; others use per-day/per-cutoff truncated series.

Assume you have:

- VC 5-min obs table: `wx.vc_minute_weather` with at least:
  - `city`, `datetime_local`, `temp_f`, and optionally `dewpoint_f`, `rh`, `wind_mph`, `precip_in`, `cloud_pct`.
- Settlement table: `wx.settlement_kalshi`:
  - `city`, `day`, `tmax_f` (this is your `settle_f`).

### 1.1. Per-row inputs

For each training sample we want:

```text
city          : categorical
day           : date (for lags & calendar)
cutoff_time   : time-of-day (how far into the day we are)
temps_f_sofar : list of VC 5-min temps up to cutoff (local)
settle_f      : integer °F (daily max, target)
````

You already have the 5-min series and settlement; the new part is multiple `cutoff_time`s per day. Pick something like:

* Every hour: local times `00:00, 01:00, ..., 23:00`, or
* Only relevant windows: e.g., `10:00–23:00` every hour.

---

### 1.2. Base temperature stats (truncated to cutoff)

Given `temps = temps_f_sofar`:

* `vc_max_f_sofar` = max(temps)
* `vc_min_f_sofar` = min(temps)
* `vc_mean_f_sofar` = mean(temps)
* `vc_std_f_sofar` = std dev
* Quantiles: `vc_q10_f_sofar`, `vc_q25_f_sofar`, `vc_q50_f_sofar`, `vc_q75_f_sofar`, `vc_q90_f_sofar`.

Python snippet:

```python
import numpy as np

def base_stats_features(temps: np.ndarray) -> dict:
    q10, q25, q50, q75, q90 = np.percentile(temps, [10, 25, 50, 75, 90])
    return {
        "vc_max_f_sofar": float(temps.max()),
        "vc_min_f_sofar": float(temps.min()),
        "vc_mean_f_sofar": float(temps.mean()),
        "vc_std_f_sofar":  float(temps.std(ddof=1)),
        "vc_q10_f_sofar":  float(q10),
        "vc_q25_f_sofar":  float(q25),
        "vc_q50_f_sofar":  float(q50),
        "vc_q75_f_sofar":  float(q75),
        "vc_q90_f_sofar":  float(q90),
    }
```

---

### 1.3. Rule-based meta-features (using truncated series)

You already implemented rules like `max_round`, `max_of_rounded`, `floor_max`, `ceil_max`, `plateau_20min`, `ignore_singletons`, `c_first` for the *full-day* series.

For intraday, just apply them to `temps_f_sofar` instead:

```python
from analysis.temperature.rules import ALL_RULES  # existing

def rule_features(temps: list[float]) -> dict:
    feats = {}
    preds = {}
    for name, fn in ALL_RULES.items():
        pred = fn(temps)
        feats[f"pred_{name}"] = pred
        preds[name] = pred

    # Base rule: max_of_rounded if available, else max_round
    base_pred = preds.get("max_of_rounded") or preds.get("max_round")
    feats["base_pred"] = base_pred

    if base_pred is not None and len(temps) > 0:
        vc_max = max(temps)
        delta = vc_max - base_pred
        feats["delta_vc_base"] = float(delta)
        feats["abs_delta_vc_base"] = float(abs(delta))
    else:
        feats["delta_vc_base"] = None
        feats["abs_delta_vc_base"] = None

    valid_preds = [p for p in preds.values() if p is not None]
    if valid_preds:
        feats["range_pred_rules"] = max(valid_preds) - min(valid_preds)
        feats["disagree_flag"] = int(len(set(valid_preds)) > 1)
        feats["num_distinct_preds"] = len(set(valid_preds))
    else:
        feats["range_pred_rules"] = None
        feats["disagree_flag"] = None
        feats["num_distinct_preds"] = None

    return feats
```

These rule outputs are powerful **meta-features**. They embody your prior reasoning and give CatBoost/logistic very informative inputs.

---

### 1.4. Shape-of-day-so-far features (spike vs plateau, timing, slopes)

Given truncated `temps` and matching `timestamps_local` (up to cutoff):

```python
def shape_features(temps: np.ndarray, timestamps_local: list) -> dict:
    feats = {}
    n = len(temps)
    if n == 0:
        return feats

    step_minutes = 5
    vc_max = temps.max()
    base_pred = int(round(vc_max))  # or use rule_features["base_pred"]

    def minutes_ge(thresh: float) -> int:
        return int((temps >= thresh).sum() * step_minutes)

    def max_run_ge(thresh: float) -> int:
        best = run = 0
        for t in temps:
            if t >= thresh:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best * step_minutes

    feats["minutes_ge_base"]    = minutes_ge(base_pred)
    feats["minutes_ge_base_p1"] = minutes_ge(base_pred + 1)
    feats["minutes_ge_base_m1"] = minutes_ge(base_pred - 1)

    feats["max_run_ge_base"]    = max_run_ge(base_pred)
    feats["max_run_ge_base_p1"] = max_run_ge(base_pred + 1)
    feats["max_run_ge_base_m1"] = max_run_ge(base_pred - 1)

    # spike vs plateau
    top3 = sorted(temps, reverse=True)[:3]
    feats["max_minus_second_max"] = float(
        top3[0] - (top3[1] if len(top3) > 1 else top3[0])
    )

    # time of max (local)
    if timestamps_local:
        idx_max = int(temps.argmax())
        t_max = timestamps_local[idx_max]
        feats["hour_of_max_local"] = t_max.hour
        feats["minute_of_max_local"] = t_max.minute
    else:
        feats["hour_of_max_local"] = None
        feats["minute_of_max_local"] = None

    # morning/afternoon/evening max so far
    if timestamps_local:
        hours = np.array([ts.hour for ts in timestamps_local])
        feats["max_morning_f"] = float(temps[hours < 12].max()) if (hours < 12).any() else None
        feats["max_afternoon_f"] = float(temps[(hours >= 12) & (hours < 17)].max()) if ((hours >= 12) & (hours < 17)).any() else None
        feats["max_evening_f"] = float(temps[hours >= 17].max()) if (hours >= 17).any() else None

    # slopes: 30-min up & down
    window = 30 // step_minutes  # e.g. 6 for 5-min data
    if n >= window + 1:
        diffs = temps[window:] - temps[:-window]
        feats["slope_max_30min_up"] = float(diffs.max())
        feats["slope_max_30min_down"] = float(diffs.min())
    else:
        feats["slope_max_30min_up"] = None
        feats["slope_max_30min_down"] = None

    return feats
```

All of this is **cutoff-aware**: it only sees data up to the current time.

---

### 1.5. Lag & calendar features

Use **previous days’ outcomes** (which are known in reality) and cyclical time encodings.

Once you’ve built a per-(city, day, cutoff_time) dataframe, you can compute:

```python
def add_lag_and_calendar_features(df):
    # df indexed by city, day, cutoff_time and sorted.
    df = df.sort_values(["city", "day", "cutoff_time"]).copy()

    # daily lags: use one row per day at a reference cutoff, or merge from a daily table
    # More robust: join in a daily-level features table for lags.
    # But simple version for now: lag settle_f by city+day (same across cutoffs).
    df["settle_f_lag1"] = df.groupby("city")["settle_f"].transform(
        lambda s: s.groupby(df["day"]).transform("first").shift(1)
    )

    # Calendar encodings
    doy = df["day"].dt.dayofyear
    week = df["day"].dt.isocalendar().week.astype(int)

    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["week_sin"] = np.sin(2 * np.pi * week / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * week / 52.0)

    df["month"] = df["day"].dt.month
    df["is_weekend"] = (df["day"].dt.weekday >= 5).astype(int)

    # time-of-day encodings
    # convert cutoff_time -> minutes since midnight
    minutes_since_midnight = df["cutoff_time"].dt.hour * 60 + df["cutoff_time"].dt.minute
    df["tod_sin"] = np.sin(2 * np.pi * minutes_since_midnight / (24 * 60))
    df["tod_cos"] = np.cos(2 * np.pi * minutes_since_midnight / (24 * 60))

    return df
```

Cyclical encodings like this are standard for time features and prevent discontinuities at boundaries.

---

### 1.6. Multi-variable weather context (optional)

If VC provides humidity, dewpoint, wind, precip, cloud, etc., you can compute truncated-day aggregates exactly like temps. Climate/forecast literature shows multi-variable features often improve extreme temp prediction (humidity, cloud, wind) :

* `vc_max_dewpoint_f_sofar`
* `vc_mean_rh_sofar`, `vc_max_rh_sofar`
* `vc_mean_wind_mph_sofar`, `vc_max_wind_mph_sofar`
* `vc_total_precip_in_sofar`
* `vc_mean_cloud_sofar`

(Use the same pattern as `base_stats_features`.)

---

### 1.7. Data-quality / missingness features

These help the model understand when VC is less trustworthy.

```python
def quality_features(temps: np.ndarray, timestamps_local: list, expected_samples_per_day=288):
    feats = {}
    n = len(temps)
    feats["num_samples_sofar"] = n

    if expected_samples_per_day:
        feats["missing_fraction_sofar"] = float(
            max(0, expected_samples_per_day - n) / expected_samples_per_day
        )
    else:
        feats["missing_fraction_sofar"] = None

    if timestamps_local:
        times = np.array([t.timestamp() for t in timestamps_local])
        if len(times) > 1:
            diffs = np.diff(times)
            feats["max_gap_minutes"] = int(diffs.max() / 60)
        else:
            feats["max_gap_minutes"] = 0

        # Flag if max so far is within 30 minutes of start or current cutoff
        idx_max = int(temps.argmax())
        t_max = timestamps_local[idx_max]
        start = timestamps_local[0]
        end = timestamps_local[-1]
        edge = (t_max - start).total_seconds() <= 1800 or (end - t_max).total_seconds() <= 1800
        feats["edge_max_flag"] = int(edge)
    else:
        feats["max_gap_minutes"] = None
        feats["edge_max_flag"] = None

    return feats
```

---

## 2. Day-grouped Time-Series Cross-Validation (No Leakage)

We want a reusable split:

* Based on **days**, not individual intraday rows.
* Respect **time order**: training days < test days.
* Optionally include a **gap** between train and test to protect against autocorrelation near the boundary.

Example splitter:

```python
import numpy as np
from typing import Iterator, Tuple

def day_time_series_splits(df, n_splits: int = 5, gap_days: int = 0
                           ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_idx, test_idx) for df with columns ['city','day'].
    Splits on unique days in sorted order, grouping all rows of a day together.
    """
    unique_days = np.array(sorted(df["day"].unique()))
    n_days = len(unique_days)
    if n_splits >= n_days:
        raise ValueError("Not enough days for n_splits")

    # define folds as contiguous day blocks
    fold_sizes = np.full(n_splits, n_days // n_splits, dtype=int)
    fold_sizes[: (n_days % n_splits)] += 1
    day_indices = np.arange(n_days)

    current = 0
    fold_boundaries = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        fold_boundaries.append((start, stop))
        current = stop

    for k, (start, stop) in enumerate(fold_boundaries):
        # training uses all days before current test block (minus gap)
        test_day_idx = day_indices[start:stop]
        train_day_idx = day_indices[: max(0, start - gap_days)]

        train_days = set(unique_days[train_day_idx])
        test_days = set(unique_days[test_day_idx])

        train_idx = df.index[df["day"].isin(train_days)].to_numpy()
        test_idx  = df.index[df["day"].isin(test_days)].to_numpy()

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        yield train_idx, test_idx
```

This is analogous to `TimeSeriesSplit` but at **day resolution** and with grouping.

---

## 3. Model 1 – Multinomial Logistic Regression (My Pick)

### 3.1. Why this model

* We want a **full discrete distribution** over temperatures (e.g. 40–110°F).
* **Multinomial logistic regression** gives us a direct model of `P(T = t | x)` for all t, in one model.
* It’s linear in features, easy to regularize (L2 / elastic net), and often quite well-calibrated by default.

Later you can upgrade to an **ordinal model** if you want to exploit the ordered nature of temperatures, but multinomial is a clean first step.

### 3.2. Target and label space

Define:

* `settle_f` ∈ [T_min, T_max], e.g., 30–110°F.
* Map each temp to a **class index**:

```python
temp_values = np.arange(T_min, T_max + 1)
temp_to_idx = {t: i for i, t in enumerate(temp_values)}
df["y_class"] = df["settle_f"].map(temp_to_idx)
```

Class index `k` corresponds to temperature `temp_values[k]`.

### 3.3. Feature matrix

Pick a rich yet manageable set (you can expand as you like):

```python
feature_cols = [
    # base stats
    "vc_max_f_sofar",
    "vc_min_f_sofar",
    "vc_mean_f_sofar",
    "vc_std_f_sofar",
    "vc_q10_f_sofar",
    "vc_q25_f_sofar",
    "vc_q50_f_sofar",
    "vc_q75_f_sofar",
    "vc_q90_f_sofar",

    # rule-based
    "pred_max_of_rounded",
    "pred_max_round",
    "pred_floor_max",
    "pred_ceil_max",
    "pred_plateau_20min",
    "pred_ignore_singletons",
    "pred_c_first",
    "base_pred",
    "delta_vc_base",
    "abs_delta_vc_base",
    "range_pred_rules",
    "num_distinct_preds",
    "disagree_flag",

    # shape & timing
    "minutes_ge_base",
    "minutes_ge_base_p1",
    "minutes_ge_base_m1",
    "max_run_ge_base",
    "max_run_ge_base_p1",
    "max_run_ge_base_m1",
    "max_minus_second_max",
    "hour_of_max_local",
    "minute_of_max_local",
    "max_morning_f",
    "max_afternoon_f",
    "max_evening_f",
    "slope_max_30min_up",
    "slope_max_30min_down",

    # lag & calendar
    "settle_f_lag1",
    "vc_max_f_lag1",          # if you precompute daily max lag
    "delta_vcmax_lag1",
    "doy_sin",
    "doy_cos",
    "week_sin",
    "week_cos",
    "month",
    "is_weekend",
    "tod_sin",
    "tod_cos",

    # quality
    "num_samples_sofar",
    "missing_fraction_sofar",
    "max_gap_minutes",
    "edge_max_flag",

    # context (if available)
    "vc_max_dewpoint_f_sofar",
    "vc_mean_rh_sofar",
    "vc_max_rh_sofar",
    "vc_mean_wind_mph_sofar",
    "vc_max_wind_mph_sofar",
    "vc_total_precip_in_sofar",
    "vc_mean_cloud_sofar",

    # categorical
    "city",
]
```

Split into `num_cols` and `cat_cols` for preprocessing:

```python
num_cols = [c for c in feature_cols if c not in ["city"]]
cat_cols = ["city"]
```

### 3.4. Pipeline and CV

Use scikit-learn’s `ColumnTransformer` + `LogisticRegression` with `multi_class='multinomial'`.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

preproc = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

logreg = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",   # or 'saga' if you want elasticnet
    C=1.0,
    max_iter=1000,
)

pipe = Pipeline(steps=[
    ("pre", preproc),
    ("clf", logreg),
])
```

**Time-series CV with no leakage:**

```python
X = df[feature_cols]
y = df["y_class"]

fold_results = []
for fold_id, (train_idx, test_idx) in enumerate(day_time_series_splits(df, n_splits=5, gap_days=1)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipe.fit(X_train, y_train)

    # Predictions
    proba = pipe.predict_proba(X_test)
    y_pred_class = proba.argmax(axis=1)

    # Metrics
    from sklearn.metrics import accuracy_score, mean_absolute_error, log_loss

    # convert class to actual temp
    temps_pred = temp_values[y_pred_class]
    temps_true = temp_values[y_test.to_numpy()]

    acc = accuracy_score(y_test, y_pred_class)
    mae = mean_absolute_error(temps_true, temps_pred)
    ll  = log_loss(y_test, proba)

    fold_results.append({"fold": fold_id, "acc": acc, "mae": mae, "log_loss": ll})
```

### 3.5. Calibration

Logistic regression is often reasonably calibrated, but you can still refine it with `CalibratedClassifierCV`, which implements **Platt scaling** (sigmoid) or **isotonic regression** in a CV-safe way.

For time series, do **not** use random CV; instead:

* Reserve a **final block of days** as a calibration set.
* Fit the logistic model on earlier days.
* Fit a 1D calibrator (Platt or isotonic) on the predicted scores vs true labels of the calibration block.

Example (simple multi-class Platt-like calibration per temperature threshold is more complex; you can start with binary “≥K” calibration per bracket you care about).

---

## 4. Model 2 – CatBoost + Optuna (Your Pick)

### 4.1. Why this model

* CatBoost handles **categoricals natively** and is strong on tabular data with many features.
* It supports **time-series oriented CV** and grouping (GroupId) to keep groups in same fold.
* Optuna provides efficient hyperparameter search with pruning.

We’ll use:

* `CatBoostClassifier` with `loss_function='MultiClass'`.
* Optuna with **10 trials** (you can scale up later).
* Our **day-grouped time-series splits** for evaluation inside the Optuna objective.

### 4.2. Preparing the data for CatBoost

CatBoost likes:

* A numeric matrix for most features.
* A list of column indices for categorical features (`city`, maybe `month`, etc.).

We can keep `city` and maybe `month` as categorical, and everything else numeric.

```python
from catboost import CatBoostClassifier, Pool
import optuna
import numpy as np

# Build X_catboost as a plain DataFrame
catboost_cols = feature_cols  # same list as before
X_cb = df[catboost_cols].copy()
y_cb = df["y_class"].values

# Categorical columns indices
cat_features = [X_cb.columns.get_loc("city"), X_cb.columns.get_loc("month")]
```

Fill NaNs appropriately (CatBoost can handle some missing, but it’s good to be explicit).

---

### 4.3. Optuna objective with day-based CV

We’ll define an Optuna objective that:

* Suggests CatBoost hyperparams.
* For each time-series fold:

  * Trains CatBoost on training days.
  * Evaluates on test days.
* Returns the **mean log loss** or **MAE** across folds.

```python
def catboost_objective(trial: optuna.Trial):
    params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "iterations": 1000,
        "early_stopping_rounds": 50,
        "verbose": False,
        "random_seed": 42,
    }

    fold_losses = []

    for fold_id, (train_idx, test_idx) in enumerate(day_time_series_splits(df, n_splits=5, gap_days=1)):
        X_train, X_test = X_cb.iloc[train_idx], X_cb.iloc[test_idx]
        y_train, y_test = y_cb[train_idx], y_cb[test_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        test_pool  = Pool(X_test,  y_test,  cat_features=cat_features)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=test_pool, use_best_model=True)

        proba = model.predict_proba(test_pool)
        # log loss
        from sklearn.metrics import log_loss
        loss = log_loss(y_test, proba)
        fold_losses.append(loss)

        # Optional: implement pruning
        trial.report(loss, fold_id)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(fold_losses))
```

### 4.4. Running Optuna with 10 trials

```python
study = optuna.create_study(
    direction="minimize",
    study_name="catboost_temp_multiclass",
)

study.optimize(catboost_objective, n_trials=10)
print("Best trial:", study.best_trial.number)
print("Best params:", study.best_trial.params)
print("Best loss:", study.best_trial.value)
```

After that, fit a **final CatBoost model** on the full training period (e.g., up to some cutoff date) using the best params, and validate on a held-out test period.

---

### 4.5. Calibration for CatBoost

CatBoost’s probabilities are often decent but can be overconfident. You can:

* Reserve a **final block of days** as a calibration set.
* Fit CatBoost on earlier days.
* Fit per-class or per-threshold calibration (Platt or isotonic) on calibration set predictions vs true labels.

Pragmatically, for **bracket-level** decisions:

* For each threshold K you care about, define event `E_K = 1[settle_f ≥ K]`.
* On calibration set:

  * Compute model’s implied `P(E_K)` = sum of probs over classes ≥ K.
  * Use `CalibratedClassifierCV` or custom isotonic regression to fit `P_calibrated(E_K)` from `P_model(E_K)`.

You don’t need to recalibrate the whole multi-class distribution at once; calibrating a handful of key thresholds is enough for trading.

---

## 5. Intraday Use: How to Get Distributions Throughout the Day

### 5.1. Training on multiple cutoffs per day

To support intraday predictions properly:

1. Choose a set of cutoff times per day, e.g.:

   ```python
   CUTOFF_TIMES = [
       "10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "22:00"
   ]
   ```

2. For each `(city, day, cutoff_time)`:

   * Truncate VC obs to `datetime_local <= cutoff_time`.
   * Compute all features **from that truncated series**.
   * Label is the final `settle_f` for that day.

3. Build one big dataset with these multiple rows per day and the `cutoff_time` (also encoded as `tod_sin/tod_cos`).

As long as the feature builder never touches obs after `cutoff_time`, there is **no future leakage** in features.

The **only time leakage risk** is in cross-validation, which we solved by splitting on days (all rows of a day go into the same fold, and folds respect time order).

### 5.2. At run time

For a live day `D`, current local time `τ`:

1. Pull VC obs up to time `τ`.
2. Compute features in the exact same way as in training (truncate, aggregate, rules, shape, lags).
3. Pass feature vector into your chosen model (logistic or CatBoost).
4. Get the probability vector over temps `P(T = t | x(τ))`.
5. Sum to get **bracket probabilities**.

You’ll see:

* Early in the day (say 10:00), the distribution is wide and often below the eventual max.
* As the day progresses and temps approach / pass the max, the distribution sharpens.

You can quantify this by computing metrics vs time-of-day: e.g., AUC & Brier for ≥90°F at each cutoff.

### 5.3. Fixed evaluation time vs dynamic “peak detection”

Two complementary strategies:

1. **Fixed decision time(s)**:

   * Decide “we will place trades at 20:00 local” (or 22:00) for each city.
   * Train & evaluate specifically on those cutoffs.
   * This is clean and easy to backtest.

2. **Dynamic heuristic (“peak likely passed”)**:

   * Use features like:

     * `vc_max_f_sofar`,
     * current temp vs `vc_max_f_sofar` (e.g., `current_temp < vc_max_f_sofar - 1°F`),
     * `slope_max_30min_down`, etc.
   * Train a small classifier to predict “has daily max already occurred?” using only past data and the eventual max time as the target.
   * When that classifier says “yes” with high confidence, you can trigger the full temperature distribution prediction.

For a **first implementation**, I’d recommend:

* Use both models at a **fixed set of cutoffs** (e.g., every hour from noon to 22:00).
* Evaluate metrics vs time-of-day and choose one or two “sweet spots” where probability forecasts are sharp and stable.
* Later, add the dynamic peak-detection layer if needed.

---

## 6. What to Tell Your Coding Agent (Claude)

You can paste the following high-level instructions:

> 1. Implement a **feature builder** that, for each `(city, day, cutoff_time)`, truncates VC 5-min observations up to that cutoff and computes:
>
>    * Base stats (`vc_max_f_sofar`, `vc_mean_f_sofar`, quantiles, etc.)
>    * All existing rule outputs (`pred_max_of_rounded`, `pred_plateau_20min`, etc.) on the truncated series, plus `base_pred`, `delta_vc_base`, `range_pred_rules`, etc.
>    * Shape-of-day features (`minutes_ge_base`, `max_run_ge_base`, `max_minus_second_max`, `hour_of_max_local`, slopes, etc.).
>    * Lag features (at least `settle_f_lag1`, `vc_max_f_lag1`).
>    * Calendar encodings (`doy_sin/cos`, `week_sin/cos`, `month`, `is_weekend`).
>    * Time-of-day encodings (`tod_sin/cos`).
>    * Data-quality features (`num_samples_sofar`, `missing_fraction_sofar`, `max_gap_minutes`, `edge_max_flag`).
>    * Optional multi-variable VC context (dewpoint, RH, wind, precip, cloud).
> 2. Build a dataset with rows for multiple **cutoff times per day** (e.g., 10:00, 12:00, …, 22:00 local), including:
>
>    * `city`, `day`, `cutoff_time`, all features above,
>    * `settle_f` from `wx.settlement_kalshi`.
> 3. Implement a **day-based time-series splitter** (`day_time_series_splits`) that:
>
>    * Gets sorted unique days,
>    * Creates 5 folds where each test block is a contiguous block of days,
>    * Ensures all rows of a given day go to the same side (train/test),
>    * Optionally inserts a gap of 1 day between train and test.
> 4. **Model 1 (Multinomial Logistic)**:
>
>    * Use scikit-learn’s `LogisticRegression(multi_class='multinomial')` in a `Pipeline` with:
>
>      * `ColumnTransformer` for numeric + one-hot city.
>    * Use the day-based splitter for CV and report:
>
>      * Top-1 accuracy on `settle_f`,
>      * MAE in °F,
>      * log loss.
>    * (Optional) Add a separate calibration step using a hold-out block of days and Platt scaling or isotonic on the probabilities for key temperature thresholds.
> 5. **Model 2 (CatBoost + Optuna)**:
>
>    * Build a CatBoost-ready matrix using the same features.
>    * Mark `city` (and optionally `month`) as categorical.
>    * Use Optuna with an objective that:
>
>      * Runs CatBoost on each fold from the day-based splitter,
>      * Returns mean log loss across folds,
>      * Uses `iterations=1000`, `early_stopping_rounds=50`, and the suggested hyperparameters (depth, learning_rate, l2_leaf_reg, etc.).
>    * Run with `n_trials=10` initially.
>    * Fit a final CatBoost model with the best params on all training days, evaluate on a held-out test period.
> 6. Compare the two models on:
>
>    * Top-1 temp accuracy,
>    * MAE,
>    * Log loss,
>    * Calibration (Brier scores / reliability curves) for selected brackets (e.g., ≥80, ≥85, ≥90, ≥95°F).
>    * Do this **for each cutoff time** to understand how performance evolves through the day.
> 7. Once results look good, expose a simple function:
>
>    * `predict_temp_distribution(city, current_day, current_time)` which:
>
>      * Builds features from live VC obs up to `current_time`,
>      * Calls the chosen model,
>      * Returns a dict `{temp: prob}` and helper methods for bracket probabilities.

From there you’ll have:

* A **time-consistent**, non-leaky feature pipeline.
* Two strong models (logistic & CatBoost) with rich features.
* Clean metrics & calibration for both, comparable across cutoffs.
* A path to intraday probability distributions you can actually trade on.

