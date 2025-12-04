
1. **Near drop-in changes to `EdgeClassifier`** to add:

   * `optuna_metric="sharpe"` / `"mean_pnl"`
   * joint tuning of **CatBoost + calibration method + decision_threshold**
   * Sharpe/mean-PnL objective inside `_create_optuna_objective`

2. **A small `visualizations/` package** with:

   * reliability / calibration plots
   * PnL/Sharpe vs threshold plots for the edge classifier

3. **Minimal wiring changes** to `04_train_edge_classifier.py` and `scripts/train_edge_classifier.py`.

I’ll keep everything aligned with your existing code / docs.

---

## 1. Updated `EdgeClassifier` with Sharpe + calibration

File: `models/edge/classifier.py` 

Key ideas:

* **Inputs**: still a DataFrame with `pnl` and all edge features.
* **Targets**:

  * `y = 1[pnl > 0]` for classification metrics.
  * `pnl` array (±1, or real PnL later) for Sharpe/mean-PnL.
* **Optuna search space**:

  * CatBoost tree params (unchanged).
  * `calibration_method ∈ {none, sigmoid, isotonic}`.
  * `decision_threshold` when metric depends on it.
* **Objective**:

  * `auc` → ROC AUC.
  * `filtered_precision` / `f1` → as before.
  * `mean_pnl` → average PnL per traded edge.
  * `sharpe` → mean(PnL) / std(PnL) on traded edges.
* **Final model**:

  * If `calibration_method = "none"`: plain `CatBoostClassifier`.
  * Else: `CalibratedClassifierCV(CatBoostClassifier, method=..., cv=3)` from sklearn.

### 1.1 Imports

At the top of `classifier.py`, add:

```python
from sklearn.calibration import CalibratedClassifierCV
```

(Leave the CatBoost imports; we’ll still use them.)

---

### 1.2 Updated `_create_optuna_objective` (with Sharpe + calibration)

Replace your existing `_create_optuna_objective` with something like:

```python
    def _create_optuna_objective(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        pnl_val: Optional[np.ndarray] = None,  # NEW: raw PnL on validation set
    ):
        """Create Optuna objective function for hyperparameter tuning.

        Supports:
          - optimize_metric='auc'
          - optimize_metric='filtered_precision'
          - optimize_metric='f1'
          - optimize_metric='mean_pnl'
          - optimize_metric='sharpe'
        """

        def objective(trial: optuna.Trial) -> float:
            # --- CatBoost hyperparameters ---
            bootstrap_type = trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            )

            params = {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "random_seed": self.random_state,
                "verbose": False,
                # Tree structure
                "depth": trial.suggest_int("depth", 3, 8),
                "iterations": trial.suggest_int("iterations", 50, 300),
                # Learning
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", 0.1, 10.0, log=True
                ),
                # Regularization
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 30),
                "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
                # Bootstrap
                "bootstrap_type": bootstrap_type,
            }

            if bootstrap_type == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float(
                    "bagging_temperature", 0.0, 2.0
                )
            else:  # Bernoulli or MVS
                params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

            # --- NEW: calibration method hyperparameter ---
            calibration_method = trial.suggest_categorical(
                "calibration_method", ["none", "sigmoid", "isotonic"]
            )

            # --- NEW: decision threshold hyperparameter for trading metrics ---
            trial_threshold = None
            if self.optimize_metric in {
                "filtered_precision",
                "f1",
                "mean_pnl",
                "sharpe",
            }:
                trial_threshold = trial.suggest_float(
                    "decision_threshold", 0.5, 0.99
                )

            # --- Build model + calibrator ---
            base_model = CatBoostClassifier(**params)

            if calibration_method == "none":
                model = base_model
            else:
                # CalibratedClassifierCV will fit CatBoost internally on folds
                model = CalibratedClassifierCV(
                    estimator=base_model,
                    method=calibration_method,
                    cv=3,
                )

            # Fit on training data (numpy arrays)
            model.fit(X_train, y_train)

            # Predict probabilities on validation set
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # --- Compute objective ---
            metric = self.optimize_metric

            if metric == "auc":
                # Classic model diagnostic
                return float(roc_auc_score(y_val, y_pred_proba))

            if metric == "filtered_precision":
                mask = y_pred_proba >= trial_threshold
                trades = int(mask.sum())
                if trades < self.min_trades_for_metric:
                    return -1e6
                precision = float(y_val[mask].mean()) if trades > 0 else 0.0
                return precision

            if metric == "f1":
                preds = (y_pred_proba >= trial_threshold).astype(int)
                return float(f1_score(y_val, preds))

            if metric in {"mean_pnl", "sharpe"}:
                if pnl_val is None:
                    # Fallback: approximate pnl as +1/-1 using y_val
                    pnl_vector = (2 * y_val - 1).astype(float)
                else:
                    pnl_vector = pnl_val.astype(float)

                mask = y_pred_proba >= trial_threshold
                trades = int(mask.sum())
                if trades < self.min_trades_for_metric:
                    return -1e6

                trade_pnl = pnl_vector[mask]
                if trade_pnl.size == 0:
                    return -1e6

                mean_pnl = float(trade_pnl.mean())

                if metric == "mean_pnl":
                    return mean_pnl

                # Sharpe per-trade: mean / std
                std_pnl = float(trade_pnl.std())
                if std_pnl == 0.0:
                    return -1e6
                sharpe = mean_pnl / std_pnl
                return sharpe

            # Default fallback = AUC
            return float(roc_auc_score(y_val, y_pred_proba))

        return objective
```

Notes:

* This is intentionally **agnostic** to your PnL scaling: right now you use `pnl ∈ {+1, −1}`, but if later you swap to dollar PnL, Sharpe will still work.
* CalibratedClassifierCV handles both Platt (`"sigmoid"`) and isotonic calibration.
* Threshold is **jointly tuned** with model + calibration for all trading metrics.

---

### 1.3 Updated `train()` to pass `pnl` and use `calibration_method` in final model

Still in `EdgeClassifier` (same file). 

#### (a) Build `pnl` arrays and pass into `_create_optuna_objective`

Inside `train()`, just after you compute `y` and `X`, add:

```python
        # Raw PnL vector (can be ±1 or real PnL; used for Sharpe/mean_pnl)
        pnl = df_valid[target_col].astype(float).values
```

Then, when you split:

```python
        if shuffle:
            X_trainval, X_test, y_trainval, y_test, pnl_trainval, pnl_test = train_test_split(
                X,
                y,
                pnl,
                test_size=test_size,
                random_state=self.random_state,
                shuffle=True,
            )
            X_train, X_val, y_train, y_val, pnl_train, pnl_val = train_test_split(
                X_trainval,
                y_trainval,
                pnl_trainval,
                test_size=val_size / (1 - test_size),
                random_state=self.random_state,
                shuffle=True,
            )
        else:
            # Time-ordered split (for debugging; consider day-grouped split later)
            n = len(X)
            train_end = int(n * (1 - val_size - test_size))
            val_end = int(n * (1 - test_size))
            X_train, y_train, pnl_train = X[:train_end], y[:train_end], pnl[:train_end]
            X_val, y_val, pnl_val = (
                X[train_end:val_end],
                y[train_end:val_end],
                pnl[train_end:val_end],
            )
            X_test, y_test, pnl_test = X[val_end:], y[val_end:], pnl[val_end:]
```

Then:

```python
        objective = self._create_optuna_objective(
            X_train, y_train, X_val, y_val, pnl_val=pnl_val
        )
```

#### (b) Respect `calibration_method` when building final model

After Optuna:

```python
        self.best_params = study.best_params
        logger.info(f"Best trial score: {study.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        # Pull out tuned threshold (for non-AUC metrics)
        if "decision_threshold" in self.best_params:
            self.decision_threshold = float(self.best_params["decision_threshold"])

        # NEW: calibration_method from best params (default = 'none')
        calib_method = self.best_params.get("calibration_method", "none")

        # Separate CatBoost params from meta-params
        catboost_params = {
            k: v
            for k, v in self.best_params.items()
            if k not in {"decision_threshold", "calibration_method"}
        }

        base_params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": self.random_state,
            "verbose": False,
            **catboost_params,
        }

        base_model = CatBoostClassifier(**base_params)

        if calib_method == "none":
            self.model = base_model
        else:
            self.model = CalibratedClassifierCV(
                estimator=base_model,
                method=calib_method,
                cv=3,
            )

        # Final fit on train set
        self.model.fit(X_train, y_train)
```

We no longer need CatBoost `Pool` here – X is already a numeric matrix with no categorical features.

#### (c) Threshold tuning fallback for AUC mode

Keep your existing threshold grid search for **AUC-only** optimization:

```python
        tuned_info = None
        if (
            tune_threshold
            and "decision_threshold" not in self.best_params
            and self.optimize_metric == "auc"
        ):
            # Only for pure AUC mode
            y_val_proba = self.model.predict_proba(X_val)[:, 1]
            best_thr, best_precision, best_trades = self._tune_decision_threshold(
                y_true=y_val,
                y_proba=y_val_proba,
            )
            tuned_info = {
                "best_threshold": best_thr,
                "val_precision": best_precision,
                "val_trades": best_trades,
            }
            self.decision_threshold = best_thr
```

Everything below (test metrics, save, etc.) can remain the same, except you might want to rename `"Best trial AUC"` logging to `"Best trial score"` since with Sharpe you’re not maximizing AUC anymore.

The JSON you already emit (e.g. in Austin’s model) will now also contain `calibration_method` inside `best_params`. 

---

## 2. Visualizations package

Create a new folder at repo root:

```text
visualizations/
├── __init__.py
├── calibration_plots.py
└── edge_reports.py
```

You can add more over time, but these two files will cover what you need now.

---

### 2.1 `visualizations/calibration_plots.py`

Goal: reliability diagrams + basic calibration metrics for any binary classifier (edge classifier, or even the ordinal thresholds if you want).

```python
# visualizations/calibration_plots.py

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


def plot_reliability_diagram(
    y_true,
    y_proba,
    n_bins: int = 10,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """Reliability diagram (calibration curve) for binary probabilities.

    Args:
        y_true: 1D array-like of true labels {0,1}
        y_proba: 1D array-like of predicted probabilities
        n_bins: number of bins
        title: optional plot title
        save_path: if provided, save PNG here
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_pred, frac_pos, marker="o", linestyle="", label="Empirical")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(title or "Reliability Diagram")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig, ax


def summarize_calibration(y_true, y_proba) -> dict:
    """Compute basic calibration metrics (Brier, log loss)."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    eps = 1e-15
    y_proba_clipped = np.clip(y_proba, eps, 1 - eps)

    return {
        "brier": float(brier_score_loss(y_true, y_proba_clipped)),
        "log_loss": float(log_loss(y_true, y_proba_clipped)),
    }
```

Later, if you want PyCalib extras (ECE, advanced calibrators), you can add them here – but this is enough to start.

---

### 2.2 `visualizations/edge_reports.py`

Goal: take a trained edge classifier + edge dataset, and generate:

* Calibration curves on **test** set.
* PnL / Sharpe vs threshold curves.

```python
# visualizations/edge_reports.py

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.edge.classifier import EdgeClassifier
from .calibration_plots import plot_reliability_diagram, summarize_calibration


def load_edge_model_and_data(city: str) -> tuple[EdgeClassifier, pd.DataFrame]:
    """Helper: load edge classifier + edge_training_data for a city."""
    base = Path(f"models/saved/{city}")
    model_path = base / "edge_classifier"
    data_path = base / "edge_training_data.parquet"

    clf = EdgeClassifier()
    clf.load(model_path)

    df_edge = pd.read_parquet(data_path)
    # Filter to signals only, like training script
    df_signals = df_edge[df_edge["signal"] != "no_trade"].copy()

    return clf, df_signals


def plot_edge_calibration_for_city(
    city: str,
    save_dir: Optional[Path] = None,
    test_fraction: float = 0.2,
):
    """Calibration plots for edge classifier on a hold-out slice.

    This does a simple time-based split (last test_fraction of rows as test).
    You can upgrade this to day-based grouping if you like.
    """
    clf, df = load_edge_model_and_data(city)

    df = df.sort_values("day").reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1 - test_fraction))
    df_test = df.iloc[test_start:]

    y_true = (df_test["pnl"] > 0).astype(int).values
    y_proba = clf.predict(df_test)

    metrics = summarize_calibration(y_true, y_proba)

    title = f"{city.title()} edge classifier – calibration (Brier={metrics['brier']:.4f})"
    if save_dir is None:
        save_dir = Path(f"visualizations/edge/{city}")
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / "edge_calibration.png"

    plot_reliability_diagram(y_true, y_proba, n_bins=15, title=title, save_path=fig_path)

    return {
        "metrics": metrics,
        "plot_path": fig_path,
    }


def plot_pnl_sharpe_vs_threshold(
    df_signals: pd.DataFrame,
    proba: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
    min_trades: int = 10,
    save_path: Optional[Path] = None,
):
    """Plot mean PnL and Sharpe as functions of decision_threshold."""
    y_pnl = df_signals["pnl"].astype(float).values
    proba = np.asarray(proba)

    if thresholds is None:
        thresholds = np.linspace(0.5, 0.99, 25)

    mean_pnls = []
    sharpes = []
    trade_counts = []

    for thr in thresholds:
        mask = proba >= thr
        trades = int(mask.sum())
        trade_counts.append(trades)

        if trades < min_trades:
            mean_pnls.append(np.nan)
            sharpes.append(np.nan)
            continue

        pnl_trades = y_pnl[mask]
        mean_pnls.append(float(pnl_trades.mean()))
        std = float(pnl_trades.std())
        if std == 0:
            sharpes.append(np.nan)
        else:
            sharpes.append(float(mean_pnls[-1] / std))

    thresholds = np.asarray(thresholds)
    mean_pnls = np.asarray(mean_pnls)
    sharpes = np.asarray(sharpes)
    trade_counts = np.asarray(trade_counts)

    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.plot(thresholds, mean_pnls, marker="o", label="Mean PnL per trade")
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel("Mean PnL per trade")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, sharpes, marker="x", linestyle="--", label="Sharpe")
    ax2.set_ylabel("Sharpe (per trade)")

    # Combine legends
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)

    return {
        "thresholds": thresholds,
        "mean_pnls": mean_pnls,
        "sharpes": sharpes,
        "trade_counts": trade_counts,
    }


def edge_report_for_city(
    city: str,
    save_dir: Optional[Path] = None,
    test_fraction: float = 0.2,
):
    """Generate calibration + PnL/Sharpe vs threshold plots for a city."""
    clf, df_signals = load_edge_model_and_data(city)

    df_signals = df_signals.sort_values("day").reset_index(drop=True)
    n = len(df_signals)
    test_start = int(n * (1 - test_fraction))
    df_test = df_signals.iloc[test_start:]

    proba_test = clf.predict(df_test)

    if save_dir is None:
        save_dir = Path(f"visualizations/edge/{city}")
    save_dir.mkdir(parents=True, exist_ok=True)

    cal = plot_edge_calibration_for_city(city, save_dir, test_fraction)

    pnl_sharpe = plot_pnl_sharpe_vs_threshold(
        df_signals=df_test,
        proba=proba_test,
        save_path=save_dir / "edge_pnl_sharpe_vs_threshold.png",
    )

    return {
        "calibration": cal,
        "pnl_sharpe": pnl_sharpe,
        "save_dir": save_dir,
    }
```

Use this after training:

```bash
PYTHONPATH=. python -c "from visualizations.edge_reports import edge_report_for_city; edge_report_for_city('austin')"
```

And you’ll get:

* `visualizations/edge/austin/edge_calibration.png`
* `visualizations/edge/austin/edge_pnl_sharpe_vs_threshold.png`

…plus a little dict of metrics if you call it from a notebook.

---

## 3. Wiring into your existing scripts

You already have:

* `models/pipeline/04_train_edge_classifier.py` as a thin wrapper. 
* `scripts/train_edge_classifier.py` as the main training script.

They already accept `--optuna-metric` with choices `["auc", "filtered_precision", "f1"]`. You just need to add `sharpe` / `mean_pnl`.

### 3.1 CLI change: allow Sharpe / mean_pnl

In **both**:

* `models/pipeline/04_train_edge_classifier.py` and
* `scripts/train_edge_classifier.py`

update the argparse for `--optuna-metric`:

```python
    parser.add_argument(
        "--optuna-metric",
        type=str,
        default="filtered_precision",
        choices=["auc", "filtered_precision", "f1", "mean_pnl", "sharpe"],
        help="Optuna objective metric (default: filtered_precision)",
    )
```

Everything else in those scripts can remain unchanged; `EdgeClassifier` now understands the new metrics.

---

## 4. A quick mental checklist for you / the agent

Pulling it all together:

1. **EdgeClassifier**

   * Add `CalibratedClassifierCV` import.
   * Replace `_create_optuna_objective` with the Sharpe/mean-PnL version above.
   * Update `train()` to:

     * track `pnl` vector,
     * split `pnl` alongside X and y,
     * pass `pnl_val` into `_create_optuna_objective`,
     * build final model using `calibration_method` and `decision_threshold` from `best_params`.

2. **Scripts**

   * Extend `--optuna-metric` choices to include `"mean_pnl"` and `"sharpe"` in both training entrypoints.

3. **Visualizations**

   * Create `visualizations/calibration_plots.py` and `visualizations/edge_reports.py` as above.
   * Optionally call `edge_report_for_city(city)` at the end of `04_train_edge_classifier.py` so every training run leaves PNGs in `visualizations/edge/{city}/`.

4. **Time-based splits (optional next tweak)**

   * For now, `shuffle=True` random splits are okay while we’re just wiring Sharpe in.
   * When you’re ready, you can replace the `train_test_split` logic in `EdgeClassifier.train` with a **day-grouped time split**, reusing `DayGroupedTimeSeriesSplit` like you do in `OrdinalDeltaTrainer`.

---

