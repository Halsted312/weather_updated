"""
Edge Classifier: ML-based edge quality filter for Kalshi weather trading.

This classifier predicts whether a detected edge signal will be profitable.
It uses CatBoost with Optuna hyperparameter tuning and optional calibration
to filter edge signals before placing trades.

Key features:
- Joint optimization of CatBoost + calibration + decision threshold
- Trading metrics: Sharpe, mean_pnl (not just AUC)
- Time-based splits with DayGroupedTimeSeriesSplit (prevents leakage)
- Calibration via sklearn's CalibratedClassifierCV
"""

import logging
from pathlib import Path
from typing import Dict, Literal, Optional

import joblib
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Optuna for hyperparameter tuning
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Import day-grouped time series split
from models.data.splits import DayGroupedTimeSeriesSplit

logger = logging.getLogger(__name__)


# Helper class for CalibratedClassifierCV with day-grouped splits
# Must be at module level to be picklable
class DayAwareCV:
    """Custom CV that uses DayGroupedTimeSeriesSplit.

    Wraps DayGroupedTimeSeriesSplit for use with CalibratedClassifierCV.
    Must be at module level (not nested in a method) to be picklable.
    """

    def __init__(self, df_reference: pd.DataFrame, n_splits: int = 3):
        """Initialize with reference DataFrame containing 'day' column."""
        self.cv = DayGroupedTimeSeriesSplit(n_splits=n_splits)
        self.df_reference = df_reference

    def split(self, X, y=None, groups=None):
        """Generate train/val splits using day grouping."""
        for train_idx, val_idx in self.cv.split(self.df_reference):
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        return self.cv.n_splits


# Available maker_fill_prob values for Optuna tuning
MULTI_MFP_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


class EdgeClassifier:
    """Binary classifier for edge quality prediction.

    Predicts P(edge will be profitable | edge features) using CatBoost
    with optional calibration and threshold optimization.

    Attributes:
        model: Trained CatBoost or CalibratedClassifierCV model
        feature_cols: List of feature column names
        best_params: Best hyperparameters from Optuna
        decision_threshold: Probability threshold for trading (0-1)
        train_metrics: Training/validation metrics dict
    """

    def __init__(
        self,
        n_trials: int = 80,
        optimize_metric: Literal["auc", "filtered_precision", "f1", "mean_pnl", "sharpe"] = "sharpe",
        min_trades_for_metric: int = 10,
        n_jobs: int = -1,
        random_state: int = 42,
        decision_threshold: float = 0.5,
        tune_mfp: bool = False,
    ):
        """Initialize EdgeClassifier.

        Args:
            n_trials: Number of Optuna trials for hyperparameter tuning
            optimize_metric: Metric to optimize
                - "auc": ROC AUC (diagnostic)
                - "filtered_precision": Win rate when model says trade
                - "f1": F1 score
                - "mean_pnl": Average PnL per trade
                - "sharpe": Mean PnL / Std PnL (per-trade Sharpe)
            min_trades_for_metric: Minimum trades required for valid metric
            n_jobs: Number of parallel jobs (-1 = all cores)
            random_state: Random seed for reproducibility
            decision_threshold: Initial decision threshold (may be overridden by Optuna)
            tune_mfp: Whether to tune maker_fill_prob via Optuna (requires multi-MFP cached data)

        Note:
            If optimize_metric is a trading metric (sharpe, mean_pnl, filtered_precision, f1),
            Optuna will tune decision_threshold and override this initial value.
            For diagnostic metrics (auc), this value will be used as-is.
        """
        self.n_trials = n_trials
        self.optimize_metric = optimize_metric
        self.min_trades_for_metric = min_trades_for_metric
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.tune_mfp = tune_mfp

        # Model and metadata (populated during training)
        self.model = None
        self.feature_cols = None
        self.best_params = {}
        self.decision_threshold = decision_threshold
        self.best_maker_fill_prob = 0.4  # Default, may be tuned
        self.train_metrics = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction.

        Args:
            df: DataFrame with edge data

        Returns:
            DataFrame with prepared features
        """
        # For now, just return as-is (features already computed by training script)
        # Could add feature engineering here if needed
        return df.copy()

    def _create_optuna_objective(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        pnl_val: Optional[np.ndarray] = None,
        multi_mfp_pnl_val: Optional[Dict[float, np.ndarray]] = None,
    ):
        """Create Optuna objective function for hyperparameter tuning.

        Args:
            X_train: Training features (numeric)
            y_train: Training labels (1/0)
            X_val: Validation features
            y_val: Validation labels
            df_train: Training DataFrame with 'day' column
            df_val: Validation DataFrame with 'day' column
            pnl_val: Validation PnL values (for Sharpe/mean_pnl objectives)
            multi_mfp_pnl_val: Dict mapping maker_fill_prob -> validation PnL array
                               (for tune_mfp mode)

        Returns:
            Objective function for Optuna
        """

        def objective(trial: optuna.Trial) -> float:
            # --- Maker fill prob hyperparameter (if tuning) ---
            if self.tune_mfp and multi_mfp_pnl_val:
                mfp_idx = trial.suggest_int("maker_fill_prob_idx", 0, len(MULTI_MFP_VALUES) - 1)
                selected_mfp = MULTI_MFP_VALUES[mfp_idx]
                pnl_vector_for_trial = multi_mfp_pnl_val[selected_mfp]
                y_train_for_trial = (pnl_vector_for_trial > 0).astype(int)
                # Use Y from the selected mfp's P&L column for validation
                y_val_for_eval = y_val  # Features don't change, only target interpretation
                pnl_val_for_eval = pnl_vector_for_trial
            else:
                y_train_for_trial = y_train
                y_val_for_eval = y_val
                pnl_val_for_eval = pnl_val
            # --- CatBoost hyperparameters ---
            bootstrap_type = trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            )

            params = {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "random_seed": self.random_state,
                "verbose": False,
                # Tree structure - optimized for edge dataset (~10k-50k rows, ~100-200 features)
                "depth": trial.suggest_int("depth", 4, 8),
                "iterations": trial.suggest_int("iterations", 200, 1000),
                # Learning - narrower range for stability on noisy P&L labels
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.02, 0.15, log=True
                ),
                # Regularization - stronger for noisy labels
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", 1.0, 30.0, log=True
                ),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
                "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 0.9),
                # Bootstrap
                "bootstrap_type": bootstrap_type,
            }

            if bootstrap_type == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float(
                    "bagging_temperature", 0.0, 2.0
                )
            else:  # Bernoulli or MVS
                params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

            # --- NEW: Calibration method hyperparameter ---
            calibration_method = trial.suggest_categorical(
                "calibration_method", ["none", "sigmoid", "isotonic"]
            )

            # --- NEW: Decision threshold for trading metrics ---
            trial_threshold = None
            if self.optimize_metric in {"filtered_precision", "f1", "mean_pnl", "sharpe"}:
                # Constrain to reasonable range to prevent extreme thresholds
                trial_threshold = trial.suggest_float("decision_threshold", 0.55, 0.85)

            # --- Build model + calibrator ---
            base_model = CatBoostClassifier(**params)

            if calibration_method == "none":
                model = base_model
            else:
                # Use day-grouped time series split for calibration CV
                # CRITICAL: Default cv=3 would use random K-fold (leakage!)
                cv = DayGroupedTimeSeriesSplit(n_splits=3, gap_days=0)

                model = CalibratedClassifierCV(
                    estimator=base_model,
                    method=calibration_method,  # "sigmoid" or "isotonic"
                    cv=cv,
                )

            # Fit on training data
            if calibration_method == "none":
                model.fit(X_train, y_train)
            else:
                # Use module-level DayAwareCV (picklable!)
                cv = DayAwareCV(df_reference=df_train, n_splits=3)
                model = CalibratedClassifierCV(
                    estimator=base_model,
                    method=calibration_method,
                    cv=cv,
                )
                model.fit(X_train, y_train)

            # Predict probabilities on validation set
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # --- Compute objective ---
            metric = self.optimize_metric

            if metric == "auc":
                # Classic model diagnostic
                return float(roc_auc_score(y_val_for_eval, y_pred_proba))

            if metric == "filtered_precision":
                mask = y_pred_proba >= trial_threshold
                trades = int(mask.sum())
                if trades < self.min_trades_for_metric:
                    return -1e6
                precision = float(y_val_for_eval[mask].mean()) if trades > 0 else 0.0
                # Penalize if too few trades even if precision is good
                # Encourage more trades by penalizing extreme selectivity
                if trades < 50:
                    precision *= (trades / 50.0)  # Soft penalty
                return precision

            if metric == "f1":
                preds = (y_pred_proba >= trial_threshold).astype(int)
                return float(f1_score(y_val_for_eval, preds, zero_division=0.0))

            if metric in {"mean_pnl", "sharpe"}:
                if pnl_val_for_eval is None:
                    # Fallback: approximate pnl as +1/-1 using y_val
                    pnl_vector = (2 * y_val_for_eval - 1).astype(float)
                else:
                    pnl_vector = pnl_val_for_eval.astype(float)

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
            return float(roc_auc_score(y_val_for_eval, y_pred_proba))

        return objective

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "pnl",
        val_size: float = 0.15,
        test_size: float = 0.15,
        shuffle: bool = False,
        tune_threshold: bool = True,
        cv_splits: int = 5,
    ) -> dict:
        """Train edge classifier with Optuna using day-grouped time splits.

        CRITICAL: shuffle=True is FORBIDDEN to prevent data leakage.
        All snapshots from a given day must stay together in the same split.

        Args:
            df: DataFrame with edge data (must have 'day', target_col, and features)
            target_col: Column with outcome (pnl > 0 = edge was profitable)
            val_size: Fraction of days for validation
            test_size: Fraction of days for test (held-out)
            shuffle: MUST be False (raises error if True)
            tune_threshold: Whether to tune decision threshold
            cv_splits: Number of CV splits for DayGroupedTimeSeriesSplit

        Returns:
            Dict with training metrics and best parameters

        Raises:
            ValueError: If shuffle=True (prevents leakage)
            ValueError: If 'day' column missing
        """
        if shuffle:
            raise ValueError(
                "shuffle=True is not allowed for EdgeClassifier. "
                "Time-based splits are required to prevent data leakage. "
                "All snapshots from a given day must stay together."
            )

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
                "Make sure your edge DataFrame includes 'day' (event date)."
            )

        # Sort by day (and snapshot_time if available) for deterministic splits
        sort_cols = ["day"]
        if "snapshot_time" in df_valid.columns:
            sort_cols.append("snapshot_time")
        df_valid = df_valid.sort_values(sort_cols).reset_index(drop=True)

        # Detect feature columns (exclude metadata and non-numeric columns)
        exclude_cols = {"day", "snapshot_time", "pnl", "signal", "settlement_temp",
                       "tmax_final", "city", "event_date", "cutoff_time",
                       # Exclude realistic P&L metadata columns (strings and booleans)
                       "pnl_gross", "fee_usd", "entry_price_cents", "trade_role",
                       "trade_side", "trade_action", "target_bracket", "bracket_won",
                       "trade_won", "ev_cents"}
        self.feature_cols = [c for c in df_valid.columns if c not in exclude_cols]

        logger.info(f"Using {len(self.feature_cols)} features: {self.feature_cols[:5]}...")

        # Create binary target: 1 if pnl > 0 (edge was profitable), 0 otherwise
        y = (df_valid[target_col] > 0).astype(int).values
        pnl = df_valid[target_col].astype(float).values

        # Feature matrix
        X = df_valid[self.feature_cols].values
        X = np.nan_to_num(X, nan=0.0)

        # --- Day-based outer split: train+val vs test ---
        unique_days = sorted(df_valid["day"].unique())
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
            f"Day splits: "
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

        # --- Inner split: train vs val using DayGroupedTimeSeriesSplit ---
        cv = DayGroupedTimeSeriesSplit(n_splits=cv_splits)
        splits = list(cv.split(df_trainval))

        if not splits:
            raise ValueError("DayGroupedTimeSeriesSplit produced no splits")

        # Take the LAST split: earlier days → train, later days → val
        train_idx, val_idx = splits[-1]

        df_train = df_trainval.iloc[train_idx].copy()
        df_val = df_trainval.iloc[val_idx].copy()

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

        # CRITICAL LEAKAGE CHECKS
        train_days_set = set(df_train["day"].unique())
        val_days_set = set(df_val["day"].unique())
        test_days_set = set(df_test["day"].unique())

        # Check 1: No same-day overlap between splits
        assert train_days_set.isdisjoint(val_days_set), "Train/val day overlap detected!"
        assert train_days_set.isdisjoint(test_days_set), "Train/test day overlap detected!"
        assert val_days_set.isdisjoint(test_days_set), "Val/test day overlap detected!"

        # Check 2: Temporal ordering
        train_days_list = sorted(train_days_set)
        val_days_list = sorted(val_days_set)
        test_days_list = sorted(test_days_set)

        assert train_days_list[-1] < val_days_list[0], "Train days not before val days!"
        assert val_days_list[-1] < test_days_list[0], "Val days not before test days!"

        # Check 3: No settlement in features
        assert "settlement_temp" not in self.feature_cols, "settlement_temp leaked into features!"
        assert "tmax_final" not in self.feature_cols, "tmax_final leaked into features!"

        logger.info(f"✓ Leakage checks PASSED")
        logger.info(
            f"Row-wise split: "
            f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )
        logger.info(f"Class balance - train: {y_train.mean():.1%} positive")

        # --- Run Optuna study on train/val split ---
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter tuning")

        # Extract multi-MFP P&L columns if they exist and tune_mfp is enabled
        multi_mfp_pnl_val = None
        if self.tune_mfp:
            multi_mfp_pnl_val = {}
            for mfp_val in MULTI_MFP_VALUES:
                col_name = f"pnl_mfp_{int(mfp_val * 100):02d}"
                if col_name in df_val.columns:
                    multi_mfp_pnl_val[mfp_val] = df_val[col_name].fillna(0).astype(float).values
                else:
                    logger.warning(f"Missing column {col_name} for tune_mfp mode")
            if not multi_mfp_pnl_val:
                logger.error("No multi-MFP P&L columns found! Use --multi-mfp to generate them first.")
                self.tune_mfp = False
                multi_mfp_pnl_val = None
            else:
                logger.info(f"Loaded {len(multi_mfp_pnl_val)} maker_fill_prob P&L columns for tuning")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )

        objective = self._create_optuna_objective(
            X_train, y_train, X_val, y_val, df_train, df_val,
            pnl_val=pnl_val,
            multi_mfp_pnl_val=multi_mfp_pnl_val,
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

        # Extract maker_fill_prob (if tuned)
        if "maker_fill_prob_idx" in self.best_params:
            mfp_idx = self.best_params["maker_fill_prob_idx"]
            self.best_maker_fill_prob = MULTI_MFP_VALUES[mfp_idx]
            logger.info(f"Best maker_fill_prob: {self.best_maker_fill_prob:.1%}")

        calib_method = self.best_params.get("calibration_method", "none")
        catboost_params = {
            k: v
            for k, v in self.best_params.items()
            if k not in {"decision_threshold", "calibration_method", "maker_fill_prob_idx"}
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
            # Use module-level DayAwareCV (picklable!)
            cv = DayAwareCV(df_reference=df_trainval, n_splits=3)
            self.model = CalibratedClassifierCV(
                estimator=base_model,
                method=calib_method,
                cv=cv,
            )

        # Fit final model on train+val (still only pre-test data)
        logger.info("Fitting final model on train+val combined...")
        self.model.fit(X_trainval, y_trainval)

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

        # Get feature importance
        # For CalibratedClassifierCV, feature importance is in the base estimator
        feature_importance = {}
        try:
            if isinstance(self.model, CalibratedClassifierCV):
                # For calibrated models, get importance from base estimator
                # Note: Multiple calibrators exist (one per CV fold)
                # We'll use the first calibrator's base model
                if hasattr(self.model, 'calibrated_classifiers_'):
                    base_estimator = self.model.calibrated_classifiers_[0].estimator
                    importances = base_estimator.feature_importances_
                else:
                    importances = None
            else:
                # For uncalibrated CatBoost
                importances = self.model.feature_importances_

            if importances is not None:
                feature_importance = {
                    feat: float(imp)
                    for feat, imp in zip(self.feature_cols, importances)
                }
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            feature_importance = {}

        self.train_metrics = {
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "n_test_total": int(len(X_test)),  # Alias for compatibility
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
            "feature_importance": feature_importance,
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

        # LEAKAGE INDICATOR WARNINGS
        if test_auc > 0.95:
            logger.warning(
                f"⚠️ Test AUC = {test_auc:.4f} > 0.95 suggests possible data leakage! "
                "Real markets should not be this predictable."
            )
        if filtered_win_rate > 0.90:
            logger.warning(
                f"⚠️ Filtered win rate = {filtered_win_rate:.1%} > 90% is unrealistic! "
                "Check for data leakage."
            )

        return self.train_metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for edge data.

        Args:
            df: DataFrame with edge features

        Returns:
            Array of probabilities P(edge will be profitable)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        df_prep = self.prepare_features(df)
        X = df_prep[self.feature_cols].values
        X = np.nan_to_num(X, nan=0.0)

        y_proba = self.model.predict_proba(X)[:, 1]
        return y_proba

    def save(self, path: str, city: Optional[str] = None, train_metrics: Optional[dict] = None):
        """Save trained model and metadata.

        Args:
            path: Base path (without extension) for saving model
                  Will create: {path}.pkl and {path}.json
            city: City name (for metadata)
            train_metrics: Training metrics dict (uses self.train_metrics if None)
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save model as pickle
        pkl_path = path_obj.with_suffix(".pkl")
        joblib.dump(
            {
                "model": self.model,
                "feature_cols": self.feature_cols,
                "best_params": self.best_params,
                "decision_threshold": self.decision_threshold,
            },
            pkl_path,
        )
        logger.info(f"Saved model to {pkl_path}")

        # Save metadata as JSON
        metrics = train_metrics if train_metrics is not None else self.train_metrics

        metadata = {
            "trained_at": pd.Timestamp.now().isoformat(),
            "city": city,
            "feature_cols": self.feature_cols,
            "best_params": self.best_params,
            "train_metrics": metrics,
            "decision_threshold": float(self.decision_threshold),
            "n_trials": self.n_trials,
            "optimize_metric": self.optimize_metric,
        }

        json_path = path_obj.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {json_path}")

    def load(self, path: str):
        """Load trained model from disk.

        Args:
            path: Base path (without extension) to load from
        """
        path_obj = Path(path)
        pkl_path = path_obj.with_suffix(".pkl")

        if not pkl_path.exists():
            raise FileNotFoundError(f"Model file not found: {pkl_path}")

        data = joblib.load(pkl_path)

        self.model = data["model"]
        self.feature_cols = data["feature_cols"]
        self.best_params = data.get("best_params", {})
        self.decision_threshold = data.get("decision_threshold", 0.5)

        logger.info(f"Loaded model from {pkl_path}")
        logger.info(f"Features: {len(self.feature_cols)}, threshold: {self.decision_threshold:.2f}")
