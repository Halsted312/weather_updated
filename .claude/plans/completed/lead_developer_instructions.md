
1. **High-level decisions** (for you + the coding agent’s questions).
2. **Concrete strategy + step‑by‑step plan** you can hand to the agent, including calibration + visualization folder.

I’ll keep this focused on the **edge classifier** (where the money is made), and treat the **ordinal model** as a separate, upstream probability engine you can improve over time.

---

## 1. Decisions up front

### 1.1 Threshold tuning: A vs B?

Given your new goal (“optimize Sharpe/PnL inside Optuna”) the answer is:

> **Go with Option A:** *always* tune `decision_threshold` inside Optuna for any objective where the threshold matters.

Concretely, that means in `EdgeClassifier._create_optuna_objective()` we always allow Optuna to sample `decision_threshold` whenever the objective is about trading (filtered precision, F1, Sharpe, mean PnL, etc.). 

For a pure diagnostic mode like `optuna_metric='auc'`, threshold doesn’t affect the score, so we can either:

* not sample it at all, **or**
* let Optuna sample it but know it won’t change the AUC score (only the persisted threshold for later use).

But for the mode you *care about* (Sharpe/PnL) there’s no ambiguity: **threshold must be part of the search space**.

Also: grid search becomes redundant, so `_tune_decision_threshold()` can be removed or only kept for a legacy/debug mode.

---

### 1.2 What is the “modeling phase” goal?

Given all the project docs and code you’ve got:

* Ordinal model (delta) is already fairly sophisticated and tuned with Optuna using AUC/logloss. 
* Feature backfill and wiring (lags, multi-horizon drift, etc.) are well-understood and mostly an ingestion / ETL task. 
* Edge classifier is the **last mile** between all that and actual dollars. 

So I’d define the current modeling phase like this:

1. **Primary modeling goal right now**

   * Build a **robust, jointly tuned edge classifier** (per city) where:

     * Base CatBoost hyperparams **+** calibration method **+** decision threshold are tuned together.
     * Objective is **strategy-level** (Sharpe / mean PnL) on a validation window.

2. **Secondary modeling goals (in order)**

   * Calibrate the **ordinal model’s multi-class probabilities** cleanly (for bracket pricing & implied temps). 
   * Once the pipeline is solid in one city (Austin or Chicago), **roll to all 6 cities** using the same framework.
   * After that: new features, further tuning, fancy calibrators (Dirichlet, etc.).

So: yes, train all 6 edge classifiers eventually – but first get the **joint Optuna+calibration+Sharpe loop working cleanly on one city**, with good visualization of calibration and PnL.

---

### 1.3 Baseline & target metrics

You already have an example Austin edge classifier with crazy-good metrics: AUC ≈ 0.9998, filtered win rate ≈ 99.86% on test. 

That’s a strong *sanity check* but also a yellow flag for possible:

* Label simplification (pnl=±1 regardless of price), and/or
* Temporal leakage (random splits mixing days).

So for the new scheme I’d think in terms of:

* **Baseline**:

  * Current edge classifier metrics (`test_auc`, `filtered_win_rate`) per city.
* **Targets** (research phase):

  * Maintain or improve **validation Sharpe / mean PnL** while:

    * using clean **time-based splits** by day (no leakage, per your datetime reference). 
    * avoiding silly solutions like trading once per year.

We don’t need a numeric Sharpe target yet; the key is: **are we still profitable and stable when we evaluate on a truly out-of-sample period with no parameter re-tuning?**

---

### 1.4 Live vs research?

Given:

* You’re still actively changing the modeling/code pipeline.
* Edge classifier design is mid-refactor. 

I would treat this as **research/backtesting** phase with an eye toward live soon, which implies:

* Optimize for **robustness and interpretability** over squeezing another 0.01 of AUC.
* Add logging + plots so you can trust the thing before real money goes in.

---

## 2. Strategy: “All-in-one Optuna” for the edge classifier

Let’s zoom into the edge classifier, since that’s where you want joint tuning of:

* CatBoost hyperparams
* **Calibration method** (Platt vs isotonic; later PyCalib / Dirichlet)
* Decision threshold
* And evaluate with **Sharpe / PnL** as the Optuna objective.

You already have:

* `EdgeClassifier` using CatBoost + Optuna, with objectives `auc`, `filtered_precision`, `f1`. 
* A training script that builds edge data from ordinal model outputs + Kalshi prices + settlements, and labels each signal with `pnl` ∈ {+1, −1}.

We’ll extend that.

### 2.1 Data splitting for trading-oriented optimization

Use your existing `df_signals` (edges with non-`"no_trade"` signals and PnL) as the base dataset. 

For each city:

1. **Group by day** and sort by `day` and `snapshot_time`.
2. Split days into:

   * `train_days` (earliest ~60%)
   * `val_days` (next ~20%)
   * `test_days` (latest ~20%).

Then:

* `X_train, y_train` = edges from `train_days`; `y_train = 1[pnl > 0]`.
* `X_val, y_val` = edges from `val_days`.
* `X_test, y_test` = edges from `test_days` (never touched during Optuna).

This replaces the current random `train_test_split` in `EdgeClassifier.train` when we’re in “trading mode.” 

---

### 2.2 Inside one Optuna trial (edge classifier)

For a new **trading-mode Optuna metric**, say `optuna_metric='sharpe'` or `'mean_pnl'`, you can structure the objective like this:

1. **Sample hyperparams** in the trial:

   * CatBoost params: depth, iterations, learning_rate, l2_leaf_reg, etc. (you already do this). 
   * **Calibration method**:

     ```python
     calibration_method = trial.suggest_categorical(
         "calibration_method", ["none", "sigmoid", "isotonic"]
     )
     ```

     * `"sigmoid"` = Platt scaling (logistic).
     * `"isotonic"` = non-parametric isotonic regression. ([Scikit-learn][1])
   * **Decision threshold**:

     ```python
     decision_threshold = trial.suggest_float("decision_threshold", 0.5, 0.99)
     ```
   * (Optional later) gating on `abs_edge` or `confidence`, e.g. `min_abs_edge` hyperparam.

2. **Build base model + calibrator pipeline**

   Simplest robust path with scikit‑learn:

   ```python
   from sklearn.calibration import CalibratedClassifierCV

   base_estimator = CatBoostClassifier(
       loss_function="Logloss",
       eval_metric="AUC",
       random_seed=self.random_state,
       verbose=False,
       **catboost_params,
   )

   if calibration_method == "none":
       model = base_estimator
   else:
       model = CalibratedClassifierCV(
           estimator=base_estimator,
           method=calibration_method,
           cv=3  # internal K-fold on the *training* data
       )
   ```

   * `CalibratedClassifierCV` handles Platt (`"sigmoid"`) or isotonic regression internally, training on a cross‑validated split of the **training** data so calibration uses held-out folds. ([Scikit-learn][2])

3. **Fit model on `X_train, y_train`**

   * Just call `model.fit(X_train, y_train)`.
   * Inside, if calibration is used, scikit-learn fits base CatBoost on folds and uses out-of-fold predictions to fit the calibrators. ([Scikit-learn][1])

4. **Get calibrated probabilities on validation set**

   ```python
   y_val_proba = model.predict_proba(X_val)[:, 1]  # P(edge wins)
   ```

5. **Simulate trades and compute Sharpe or mean PnL**

   * Use your **actual PnL labels** from backtest (`pnl` column) for each edge in the validation set. 

   * Decide which edges to trade with the trial’s `decision_threshold`:

     ```python
     trade_mask = y_val_proba >= decision_threshold
     val_trades = y_val[trade_mask]  # where y_val is 1/0 or use pnl_val[trade_mask]
     pnl_trades = pnl_val[trade_mask]  # raw ±1 or true PnL if you extend it
     n_trades = len(pnl_trades)
     ```

   * If `n_trades < min_trades_for_metric`, return a large negative penalty (like you already do for filtered_precision). 

   * Compute **objective**:

     * For **mean PnL per trade**:

       ```python
       score = pnl_trades.mean()
       ```

       (Or scale by some penalty on too many/too few trades.)

     * For **Sharpe per trade** (non-annualized):

       ```python
       import numpy as np
       if np.std(pnl_trades) == 0:
           return -1e6
       sharpe = np.mean(pnl_trades) / np.std(pnl_trades)
       score = sharpe
       ```

       This is consistent with standard Sharpe definition (mean excess return / std) used in trading. ([QuantStart][3])

   * Return this `score` from the objective.

With your hardware (ThreadRipper + 5090 + 128GB), this kind of Nested‑ish pipeline per trial is totally doable.

---

### 2.3 After Optuna finishes (edge classifier)

Once Optuna has found `best_params` including `calibration_method` and `decision_threshold`:

1. **Rebuild model** with those params:

   * Again, `CalibratedClassifierCV(CatBoostClassifier(**best_cat_params), method=best_cal_method, cv=3)`

2. **Fit on `X_train ∪ X_val`** (all pre‑test edges).

3. **Evaluate on `X_test`**:

   * Compute calibrated probabilities.
   * Apply `decision_threshold`.
   * Compute:

     * Test Sharpe / mean PnL,
     * Test AUC (for diagnostics),
     * Test filtered win rate,
     * Trade count.

4. **Save full pipeline + metadata**:

   * Persist `CalibratedClassifierCV` (it’s picklable) in your existing `edge_classifier.pkl` along with feature list and `decision_threshold`.
   * Enhanced JSON metadata (city, date ranges, best params incl. calibration & threshold, Sharpe, etc.), as outlined in the handoff file. 

This gives you exactly what you wanted: **one Optuna loop** optimizes *CatBoost + calibration + decision rule* for a **trading metric**.

---

## 3. Calibration strategy details (and where ordinal fits)

You asked specifically about Platt vs isotonic, and when to calibrate.

### 3.1 For the edge classifier (binary)

* Use **CalibratedClassifierCV** with:

  * `method="sigmoid"` (Platt) for stability when you don’t have tons of edge samples.
  * `method="isotonic"` for more flexible, non-linear calibration when you have lots of data. ([Scikit-learn][1])

Given your data volume of edges per city is decent (tens of thousands of samples, per the metrics JSON).

* You can safely let Optuna choose between `"sigmoid"` and `"isotonic"` with a **Sharpe/PnL objective**, as in §2.2.

**When to calibrate?**
Exactly as in the plan above: **inside each trial, on the training partition**, using `CalibratedClassifierCV` so all calibrator training is done without touching the validation set. ([Scikit-learn][2])

---

### 3.2 For the ordinal model / bracket probabilities (multi-class)

For the **ordinal delta model**, you already convert K−1 binary `P(delta ≥ k)` into class probabilities and renormalize. 

That’s already pretty close to calibrated multi-class output, but you can go further:

* First, continue tuning the ordinal model with a proper scoring rule (AUC/logloss cross‑val). 
* Then **optionally add a second calibration stage** on top of the resulting class probabilities.

Options for multi-class calibration:

1. **One-vs-rest calibration** (use scikit-learn):

   * Treat each class as “this class vs others”, calibrate those probabilities, then renormalize so they sum to 1.
   * This is what sklearn effectively does for multi-class in `CalibratedClassifierCV`. ([Scikit-learn][1])

2. **More advanced: PyCalib / Dirichlet calibration**:

   * **PyCalib**: a dedicated library for assessing and calibrating probabilistic classifiers, including multi-class. ([GitHub][4])
   * **Dirichlet calibration** (`dirichletcal`): natively multi-class calibration shown to outperform simple temperature scaling in many setups. ([GitHub][5])

Given complexity, I’d **start by calibrating the edge classifier** (binary) as above, and treat multi-class ordinal calibration as a second wave. It doesn’t have to be in the same Optuna loop that optimizes Sharpe for edge; you can treat it as a separate “forecast quality” tuning problem.

---

## 4. Visualizations / outputs folder

You asked for a root-level folder to hold code for calibration and metric visualization. I’d recommend:

### 4.1 Folder structure

At project root:

```text
visualizations/
├── __init__.py
├── calibration_plots.py
├── edge_reports.py
└── ordinal_reports.py
```

Optionally also:

```text
reports/
└── edge/
    └── {city}/
        ├── edge_calibration_{city}.png
        ├── edge_pnl_vs_threshold_{city}.png
        └── edge_sharpe_vs_threshold_{city}.png
```

### 4.2 `visualizations/calibration_plots.py`

Functions like:

* `plot_reliability_diagram(y_true, y_proba, n_bins=10, title=None, save_path=None)`

  * Use `sklearn.calibration.calibration_curve` to produce a reliability diagram (mean predicted vs empirical frequency). ([Scikit-learn][6])
* `plot_calibration_histogram(y_proba, n_bins=20, ...)`
* `show_brier_logloss(y_true, y_proba)` for quick text metrics.

If you want to use **PyCalib** here for richer metrics:

* Functions from PyCalib to compute expected calibration error, reliability, and to apply more advanced calibration methods. ([classifier-calibration.github.io][7])

### 4.3 `visualizations/edge_reports.py`

* Load:

  * `models/saved/{city}/edge_classifier.pkl`
  * `models/saved/{city}/edge_classifier.json`
  * `models/saved/{city}/edge_training_data.parquet` (edge dataset).

Produce:

* Calibration curves on **test set** for:

  * Raw CatBoost vs calibrated pipeline probabilities.
* PnL/Sharpe diagnostics:

  * `PnL vs threshold` curve (grid across thresholds).
  * `Sharpe vs threshold`.
  * Distribution of per-trade PnL.
* Maybe a simple “strategy equity curve” across test dates.

This is where you might integrate PyCalib’s evaluation functions too.

### 4.4 `visualizations/ordinal_reports.py`

* Similar reliability diagrams and Brier/logloss for the ordinal delta model’s bracket probabilities. 

---

## 5. Putting it all together: what to tell the coding agent

Here’s a concise (but technically precise) summary you can hand them:

1. **Threshold tuning decision**

   * Adopt **Option A**: for any trading-specific objective (`filtered_precision`, new `sharpe`, `mean_pnl`), include `decision_threshold` in the Optuna search space.
   * Remove or deprecate the separate grid search `_tune_decision_threshold()` path; tuning should live inside Optuna.

2. **New Optuna objectives for edge classifier**

   * Add support for `optuna_metric="sharpe"` and/or `optuna_metric="mean_pnl"`.
   * Inside `_create_optuna_objective`, for these metrics:

     * Build `CalibratedClassifierCV(CatBoostClassifier(**params), method=trial.choice(["sigmoid", "isotonic", "none"]), cv=3)`. ([Scikit-learn][1])
     * Fit on a **time-based train split by day** (use the same `df_signals` you already build in `scripts/train_edge_classifier.py`).
     * Get calibrated probabilities on validation days and compute the trial’s objective:

       * Choose trades where `p >= decision_threshold`.
       * Use actual `pnl` for those edges to compute mean PnL or Sharpe.
       * Penalize trials where number of trades falls below `min_trades_for_metric`.

3. **Final training & saving**

   * After Optuna, rebuild the **best calibrator + CatBoost** pipeline and fit on train+val.
   * Evaluate metrics (Sharpe, mean PnL, AUC, filtered win rate) on a **held-out test window**.
   * Save:

     * The full `CalibratedClassifierCV` pipeline in `edge_classifier.pkl`.
     * Rich JSON metadata (`edge_classifier.json`) including:

       * city, date range, n_trials, calibration_method, decision_threshold,
       * test Sharpe, mean PnL, AUC, filtered win rate, trade count, feature_importance.

4. **Visualization folder**

   * Add a root-level `visualizations/` package with:

     * `calibration_plots.py` – reliability diagrams, calibration curves, Brier/logloss summaries using sklearn, with optional PyCalib integration. ([Scikit-learn][6])
     * `edge_reports.py` – functions to:

       * Load edge dataset + model.
       * Plot calibration curves on test set.
       * Plot PnL/Sharpe vs threshold and equity curves.
     * `ordinal_reports.py` – similar plots for the ordinal model’s bracket probabilities.

   * Update the training scripts (e.g., `04_train_edge_classifier.py`) to optionally call these report functions and drop PNGs into `reports/edge/{city}/`.

5. **Phase prioritization**

   * Phase 1: Implement the new edge classifier joint tuning (CatBoost + calibration + threshold, `optuna_metric='sharpe'`) for **one city** (Austin or Chicago).
   * Phase 2: Once stable, roll out to all 6 cities using the same pipeline.
   * Phase 3: Add multi-class calibration (PyCalib or Dirichlet) on top of the ordinal model’s bracket probabilities as a separate optimization problem.

---
