Below is a **drop-in replacement** for `EdgeClassifier.train` that:

* Uses **`DayGroupedTimeSeriesSplit` grouped by `day`** to define train/val.
* Holds out the **last block of days as test**.
* Assumes the `_create_optuna_objective(...)` you and I discussed earlier (with `pnl_val` and Sharpe/mean-PnL support).
* Works with the existing `prepare_features()` and `predict()`.

I’ll also show the **extra import** you need.

---

## 1. Import DayGroupedTimeSeriesSplit

At the top of `models/edge/classifier.py`, add:

```python
from models.data.splits import DayGroupedTimeSeriesSplit
```

(You’re already using this in `OrdinalDeltaTrainer`.)

---

## 2. New `train` method using `DayGroupedTimeSeriesSplit` by `day`

Paste this over your current `EdgeClassifier.train` in `classifier.py`:

```python
    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "pnl",
        val_size: float = 0.15,
        test_size: float = 0.15,
        shuffle: bool = False,
        tune_threshold: bool = True,
        n_splits: int = 5,
    ) -> dict:
        """Train edge classifier with Optuna using day-grouped time splits.

        This version:
          - Respects the project datetime conventions (see DATETIME_AND_API_REFERENCE.md)
          - Uses `DayGroupedTimeSeriesSplit` grouped by the `day` column
          - Holds out the last block of days as a true test set
          - Uses time-ordered CV for the train/val division

        Args:
            df: DataFrame with edge data (must have target_col, features, and 'day')
            target_col: Column with outcome (pnl > 0 = edge was real)
            val_size: Fraction of days for validation (on the train+val side)
            test_size: Fraction of days for final test (last block of days)
            shuffle: Ignored (kept for API compatibility; splits are time-based)
            tune_threshold: Whether to optionally grid-tune threshold in AUC mode
            n_splits: Number of folds for DayGroupedTimeSeriesSplit (used to pick the last fold as train/val)

        Returns:
            Dict with training metrics and best parameters.
            Also sets `self.model`, `self.best_params`, `self.decision_threshold`,
            and `self.train_metrics`.
        """
        logger.info(f"Training EdgeClassifier with {self.n_trials} Optuna trials")
        logger.info("Using day-grouped time splits (DayGroupedTimeSeriesSplit)")

        # Prepare features
        df_prep = self.prepare_features(df)

        # Filter to rows with valid outcome
        df_valid = df_prep[df_prep[target_col].notna()].copy()
        if df_valid.empty:
            raise ValueError("No rows with valid target in EdgeClassifier.train()")

        # Require 'day' column for grouped time splits
        if "day" not in df_valid.columns:
            raise ValueError(
                "EdgeClassifier.train with day-grouped splits requires a 'day' column. "
                "Make sure your edge DataFrame includes 'day' (event date) as defined "
                "in DATETIME_AND_API_REFERENCE.md."
            )

        # Sort by day (and snapshot_time if available) for deterministic splits
        sort_cols = ["day"]
        if "snapshot_time" in df_valid.columns:
            sort_cols.append("snapshot_time")
        df_valid = df_valid.sort_values(sort_cols).reset_index(drop=True)

        # Create binary target: 1 if pnl > 0 (edge was real), 0 otherwise
        y = (df_valid[target_col] > 0).astype(int).values
        pnl = df_valid[target_col].astype(float).values

        # Feature matrix
        X = df_valid[self.feature_cols].values
        X = np.nan_to_num(X, nan=0.0)

        # --- Day-based outer split: train+val vs test (by unique days) ---

        unique_days = df_valid["day"].unique()
        n_days = len(unique_days)
        if n_days < 3:
            raise ValueError(
                f"Not enough distinct days ({n_days}) to create train/val/test splits."
            )

        # Compute test / val days as fractions of total days
        n_test_days = max(1, int(round(test_size * n_days)))
        n_val_days = max(1, int(round(val_size * n_days)))

        if n_test_days + n_val_days >= n_days:
            # Ensure we always have at least 1 train day
            n_val_days = max(1, min(n_val_days, n_days - n_test_days - 1))

        test_days = unique_days[-n_test_days:]
        trainval_days = unique_days[: n_days - n_test_days]

        logger.info(
            f"Day splits (by unique 'day'): "
            f"total_days={n_days}, train+val_days={len(trainval_days)}, test_days={len(test_days)}"
        )

        trainval_mask = df_valid["day"].isin(trainval_days)
        test_mask = df_valid["day"].isin(test_days)

        df_trainval = df_valid[trainval_mask].reset_index(drop=True)
        df_test = df_valid[test_mask].reset_index(drop=True)

        X_trainval = df_trainval[self.feature_cols].values
        y_trainval = (df_trainval[target_col] > 0).astype(int).values
        pnl_trainval = df_trainval[target_col].astype(float).values

        X_test = df_test[self.feature_cols].values
        y_test = (df_test[target_col] > 0).astype(int).values
        pnl_test = df_test[target_col].astype(float).values

        # --- Inner split: train vs val using DayGroupedTimeSeriesSplit on train+val days ---

        groups = df_trainval["day"].values
        cv = DayGroupedTimeSeriesSplit(n_splits=n_splits)

        # Take the LAST split: earlier days -> train, later days -> val
        # (You can also do CV inside _create_optuna_objective if you want k-fold averaging)
        splits = list(cv.split(X_trainval, groups=groups))
        if not splits:
            raise ValueError("DayGroupedTimeSeriesSplit produced no splits (check n_splits vs number of days)")

        train_idx, val_idx = splits[-1]
        X_train, y_train, pnl_train = (
            X_trainval[train_idx],
            y_trainval[train_idx],
            pnl_trainval[train_idx],
        )
        X_val, y_val, pnl_val = (
            X_trainval[val_idx],
            y_trainval[val_idx],
            pnl_trainval[val_idx],
        )

        logger.info(
            f"Row-wise split (time-based): "
            f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )
        logger.info(f"Class balance - train: {y_train.mean():.1%} positive")

        # --- Run Optuna study on train/val split ---

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )

        # NOTE: assumes _create_optuna_objective signature includes pnl_val
        objective = self._create_optuna_objective(
            X_train, y_train, X_val, y_val, pnl_val=pnl_val
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info(f"Starting Optuna optimization with {self.n_trials} trials")
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=False,
        )

        self.best_params = study.best_params
        logger.info(f"Best trial score ({self.optimize_metric}): {study.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        # Extract decision threshold (if tuned)
        if "decision_threshold" in self.best_params:
            self.decision_threshold = float(self.best_params["decision_threshold"])

        calib_method = self.best_params.get("calibration_method", "none")
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

        # Fit final model on TRAIN+VAL (still only pre-test data)
        X_fit = np.vstack([X_train, X_val])
        y_fit = np.concatenate([y_train, y_val])

        self.model.fit(X_fit, y_fit)

        # --- Evaluate on test set (true hold-out days) ---

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.decision_threshold).astype(int)

        test_auc = roc_auc_score(y_test, y_pred_proba)
        test_accuracy = accuracy_score(y_test, y_pred)

        # Calculate win rate when model predicts "trade"
        trade_mask = y_pred == 1
        if trade_mask.sum() > 0:
            filtered_win_rate = y_test[trade_mask].mean()
            n_trades_recommended = int(trade_mask.sum())
            pnl_trades = pnl_test[trade_mask]
            mean_pnl_trades = float(pnl_trades.mean())
            std_pnl_trades = float(pnl_trades.std())
            if std_pnl_trades > 0:
                sharpe_trades = mean_pnl_trades / std_pnl_trades
            else:
                sharpe_trades = 0.0
        else:
            filtered_win_rate = 0.0
            n_trades_recommended = 0
            mean_pnl_trades = 0.0
            sharpe_trades = 0.0

        # Baseline win rate (without filtering)
        baseline_win_rate = float(y_test.mean())
        mean_pnl_all = float(pnl_test.mean())

        self.train_metrics = {
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "optuna_metric": self.optimize_metric,
            "best_optuna_score": float(study.best_value),
            "test_auc": float(test_auc),
            "test_accuracy": float(test_accuracy),
            "baseline_win_rate": baseline_win_rate,
            "filtered_win_rate": float(filtered_win_rate),
            "n_trades_recommended": int(n_trades_recommended),
            "decision_threshold": float(self.decision_threshold),
            "calibration_method": calib_method,
            "mean_pnl_all_edges": mean_pnl_all,
            "mean_pnl_trades": mean_pnl_trades,
            "sharpe_trades": sharpe_trades,
        }

        logger.info(f"Test AUC: {test_auc:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.1%}")
        logger.info(f"Baseline win rate: {baseline_win_rate:.1%}")
        logger.info(
            f"Filtered win rate: {filtered_win_rate:.1%} "
            f"(n_trades={n_trades_recommended})"
        )
        logger.info(f"Mean PnL (all edges): {mean_pnl_all:.4f}")
        logger.info(f"Mean PnL (trades): {mean_pnl_trades:.4f}")
        logger.info(f"Sharpe (trades): {sharpe_trades:.4f}")

        return self.train_metrics
```

### Notes on this design

* **`val_size` and `test_size` are now fractions of *days***, not rows. That’s usually what you want for time series.
* `DayGroupedTimeSeriesSplit` is used on the **train+val days** to get a time-respecting train/val split; we pick the **last fold** so validation is on the most recent pre-test days (closest to your backtest). This mirrors how you use it for the ordinal model.
* The code assumes you’ve updated `_create_optuna_objective` to accept `pnl_val` and support `mean_pnl` / `sharpe` as we discussed earlier.

