#!/usr/bin/env python3
"""
Train Market-Clock TOD v1 Global Ordinal CatBoost Model.

This trains a single global model across all 6 cities using market-clock
time features (minutes since market open at D-1 10:00).

Key differences from per-city TOD v1:
- Global model (all cities pooled)
- Market-clock features (minutes_since_market_open, is_d_minus_1, etc.)
- City one-hot encoding (not categorical)
- CPU CatBoost with 26 threads

Usage:
    # Train on smoke dataset (fixed params, no Optuna)
    .venv/bin/python scripts/train_market_clock_tod_v1.py \\
        --input data/market_clock_tod_v1/train_data_smoke.parquet

    # Train on full dataset with Optuna (v1.1)
    .venv/bin/python scripts/train_market_clock_tod_v1.py \\
        --input data/market_clock_tod_v1/train_data.parquet \\
        --use-optuna --trials 30

    # Train with custom test days
    .venv/bin/python scripts/train_market_clock_tod_v1.py \\
        --input data/market_clock_tod_v1/train_data.parquet \\
        --test-days 90
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

# Add project root to path (scripts/training/core/ -> 3 levels up)
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    raise ImportError("catboost required. Install with: pip install catboost")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from models.features.base import DELTA_CLASSES
from models.evaluation.metrics import compute_delta_metrics, compute_ordinal_metrics
from models.data.splits import DayGroupedTimeSeriesSplit


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output directory
MODEL_DIR = Path("models/saved/market_clock_tod_v1")

# Fixed CatBoost params for v1 (CPU, 26 threads)
DEFAULT_CATBOOST_PARAMS = {
    "task_type": "CPU",
    "thread_count": 26,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "depth": 8,
    "learning_rate": 0.05,
    "iterations": 1000,
    "l2_leaf_reg": 5.0,
    "border_count": 128,
    "min_data_in_leaf": 10,
    "random_strength": 0.5,
    "random_seed": 42,
    "verbose": False,
}


class MarketClockOrdinalTrainer:
    """Ordinal regression trainer for market-clock global model.

    Uses all-threshold binary classifiers with CPU CatBoost.
    """

    def __init__(
        self,
        catboost_params: Optional[dict] = None,
        n_trials: int = 0,
        cv_splits: int = 3,
        verbose: bool = False,
        delta_min_range: tuple[int, int] = (-14, -8),
        delta_max_range: tuple[int, int] = (8, 14),
        calibrate: str = 'none',  # 'none', 'isotonic', or 'sigmoid'
    ):
        """Initialize trainer.

        Args:
            catboost_params: CatBoost parameters (defaults to CPU 26-thread config)
            n_trials: Optuna trials (0 = use fixed params)
            cv_splits: Number of CV splits for tuning
            verbose: Show progress
            delta_min_range: Optuna search range for min delta (low, high)
            delta_max_range: Optuna search range for max delta (low, high)
            calibrate: Calibration method - 'none', 'isotonic', or 'sigmoid' (Platt)
        """
        self.catboost_params = catboost_params or DEFAULT_CATBOOST_PARAMS.copy()
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.verbose = verbose
        self.delta_min_range = delta_min_range
        self.delta_max_range = delta_max_range
        self.calibrate = calibrate

        # Set during training
        self.thresholds: list[int] = []
        self._min_delta: Optional[int] = None
        self._max_delta: Optional[int] = None
        self._delta_classes: Optional[list[int]] = None
        self._feature_cols: list[str] = []

        self.classifiers: dict[int, Any] = {}
        self.calibrators: dict[int, Any] = {}  # Calibrators per threshold
        self.best_params: dict = {}
        self.best_delta_range: Optional[tuple[int, int]] = None
        self.study: Optional[Any] = None

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Get feature columns by auto-discovering from the dataset.

        Uses all numeric columns except metadata/label columns.
        Excludes columns that are >95% constant (no predictive value).

        IMPORTANT: Two types of err_* features exist:
        - SAFE (forecast-based): err_mean_sofar, err_std_sofar, etc. - compare obs vs T-1 forecast
        - LEAKY (rule-based): err_{rule}_sofar - compare rule_pred vs settle_f (target leakage!)

        We INCLUDE forecast-based err_* but EXCLUDE rule-based err_* features.
        """
        # Metadata and label columns to exclude
        exclude_cols = {
            'city', 'day', 'event_date', 'cutoff_time',
            'delta', 'settle_f', 't_base', 't_forecast_base',
        }

        # Rule-based err_* features that use settle_f (TARGET LEAKAGE!)
        # These are computed as (rule_prediction - settle_f) in models/features/rules.py
        # At inference time, we don't know settle_f, so these can't be computed
        leaky_rule_err_features = {
            'err_max_round_sofar',
            'err_max_of_rounded_sofar',
            'err_ceil_max_sofar',
            'err_floor_max_sofar',
            'err_plateau_20min_sofar',
            'err_ignore_singletons_sofar',
            'err_c_first_sofar',
        }

        # Find all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        # Exclude leaky rule-based err_* features
        excluded_leaky = [c for c in feature_cols if c in leaky_rule_err_features]
        if excluded_leaky:
            logger.warning(f"Excluding {len(excluded_leaky)} leaky rule-based err_* features: {excluded_leaky}")
        feature_cols = [c for c in feature_cols if c not in leaky_rule_err_features]

        # Log which err_* features are being INCLUDED (forecast-based, safe)
        safe_err_features = [c for c in feature_cols if c.startswith('err_')]
        if safe_err_features:
            logger.info(f"Including {len(safe_err_features)} safe forecast-based err_* features: {safe_err_features}")

        # Remove columns that are >95% constant (no predictive value)
        valid_features = []
        excluded_constant = []
        for col in feature_cols:
            if col in df.columns:
                mode_pct = df[col].value_counts(normalize=True, dropna=False).iloc[0]
                if mode_pct < 0.95:
                    valid_features.append(col)
                else:
                    excluded_constant.append(col)

        if excluded_constant:
            logger.info(f"Excluded {len(excluded_constant)} constant features: {excluded_constant[:5]}...")

        logger.info(f"Auto-discovered {len(valid_features)} features from dataset")
        return valid_features

    def _create_base_model(self) -> CatBoostClassifier:
        """Create base CatBoost classifier."""
        params = self.catboost_params.copy()
        if self.best_params:
            params.update(self.best_params)

        # Handle bootstrap-specific params
        bootstrap_type = params.get("bootstrap_type", "Bayesian")
        if bootstrap_type != "Bayesian":
            params.pop("bagging_temperature", None)
        else:
            params.pop("subsample", None)

        return CatBoostClassifier(**params)

    def _tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y_raw: pd.Series,
        df: pd.DataFrame,
    ) -> None:
        """Run Optuna hyperparameter search including delta range tuning.

        Args:
            X: Feature matrix
            y_raw: RAW (unclipped) delta values
            df: Full DataFrame for CV splitting (needs 'day' column)
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default parameters")
            return

        logger.info(f"Starting Optuna tuning with {self.n_trials} trials")
        logger.info(f"Delta range search: min in {self.delta_min_range}, max in {self.delta_max_range}")

        cv = DayGroupedTimeSeriesSplit(n_splits=self.cv_splits)

        # Reset indices to ensure iloc works correctly with KFold
        X = X.reset_index(drop=True)
        y_raw = y_raw.reset_index(drop=True)
        df = df.reset_index(drop=True)

        def objective(trial: optuna.Trial) -> float:
            # Delta range hyperparameters
            delta_min = trial.suggest_int("delta_min", self.delta_min_range[0], self.delta_min_range[1])
            delta_max = trial.suggest_int("delta_max", self.delta_max_range[0], self.delta_max_range[1])

            # Clip deltas to the suggested range
            y = y_raw.clip(lower=delta_min, upper=delta_max)

            # CatBoost hyperparameters
            bootstrap_type = trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli"]
            )

            params = {
                "task_type": "CPU",
                "thread_count": 26,
                "depth": trial.suggest_int("depth", 4, 10),
                "iterations": trial.suggest_int("iterations", 300, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "border_count": trial.suggest_int("border_count", 32, 254),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
                "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
                "bootstrap_type": bootstrap_type,
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "random_seed": 42,
                "verbose": False,
            }

            if bootstrap_type == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float(
                    "bagging_temperature", 0.0, 1.0
                )
            else:
                params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)

            # Use middle threshold for tuning (within the clipped range)
            tuning_threshold = (delta_min + delta_max) // 2
            y_binary = (y >= tuning_threshold).astype(int)

            within2_scores = []
            for train_idx, val_idx in cv.split(df):
                X_tr = X.iloc[train_idx]
                y_tr_binary = y_binary.iloc[train_idx]
                y_tr_full = y.iloc[train_idx]
                X_va = X.iloc[val_idx]
                y_va_binary = y_binary.iloc[val_idx]
                y_va_full = y.iloc[val_idx]

                if len(y_va_binary.unique()) < 2:
                    continue

                # Train a simple model for this fold to get predictions
                model = CatBoostClassifier(**params)
                train_pool = Pool(X_tr, y_tr_binary)
                val_pool = Pool(X_va, y_va_binary)

                model.fit(
                    train_pool,
                    eval_set=val_pool,
                    early_stopping_rounds=50,
                )

                # Get predictions using expected value from thresholds
                proba = model.predict_proba(val_pool)[:, 1]

                # Convert probability to expected delta
                # P(delta >= threshold) = proba => expected ~ threshold + (proba - 0.5) * range
                delta_range = delta_max - delta_min
                y_pred = tuning_threshold + (proba - 0.5) * delta_range * 0.5
                y_pred = np.round(y_pred).astype(int)
                y_pred = np.clip(y_pred, delta_min, delta_max)

                # Compute within-2 accuracy (key metric for bracket trading)
                within2 = (np.abs(y_va_full.values - y_pred) <= 2).mean()
                within2_scores.append(within2)

            return float(np.mean(within2_scores)) if within2_scores else 0.0

        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
        )

        # Extract best params (separating delta range from model params)
        best = self.study.best_params.copy()
        self.best_delta_range = (best.pop("delta_min"), best.pop("delta_max"))
        self.best_params = best

        logger.info(f"Best delta range: [{self.best_delta_range[0]}, {self.best_delta_range[1]}]")
        logger.info(f"Best model params: {self.best_params}")
        logger.info(f"Best Within-2: {self.study.best_value:.4f}")

    def train(
        self,
        df_train: pd.DataFrame,
        df_val: Optional[pd.DataFrame] = None,
    ) -> "MarketClockOrdinalTrainer":
        """Train ordinal model.

        Args:
            df_train: Training DataFrame with features and 'delta' column (raw, unclipped)
            df_val: Optional validation DataFrame

        Returns:
            Self
        """
        logger.info(f"Training market-clock ordinal model on {len(df_train)} samples")

        # Get feature columns
        self._feature_cols = self._get_feature_columns(df_train)
        logger.info(f"Using {len(self._feature_cols)} features")

        X_train = df_train[self._feature_cols].copy()
        y_train_raw = df_train["delta"].copy()

        # Fill NaN with median for numeric features
        for col in X_train.columns:
            if X_train[col].isna().any():
                X_train[col] = X_train[col].fillna(X_train[col].median())

        # Log raw delta range
        raw_min, raw_max = int(y_train_raw.min()), int(y_train_raw.max())
        logger.info(f"Raw delta range in data: [{raw_min}, {raw_max}]")

        # Optuna tuning if requested (finds best delta range + model params)
        if self.n_trials > 0:
            self._tune_hyperparameters(X_train, y_train_raw, df_train)

        # Determine final delta range
        if self.best_delta_range is not None:
            self._min_delta, self._max_delta = self.best_delta_range
            logger.info(f"Using Optuna-optimized delta range: [{self._min_delta}, {self._max_delta}]")
        else:
            # Use middle of search range as default
            self._min_delta = (self.delta_min_range[0] + self.delta_min_range[1]) // 2
            self._max_delta = (self.delta_max_range[0] + self.delta_max_range[1]) // 2
            logger.info(f"Using default delta range: [{self._min_delta}, {self._max_delta}]")

        # Clip deltas to the final range
        y_train = y_train_raw.clip(lower=self._min_delta, upper=self._max_delta)

        # Set up thresholds and classes
        self.thresholds = list(range(self._min_delta + 1, self._max_delta + 1))
        self._delta_classes = list(range(self._min_delta, self._max_delta + 1))

        logger.info(f"Training {len(self.thresholds)} threshold classifiers")

        # Prepare validation data (also clip to delta range)
        X_val, y_val = None, None
        if df_val is not None:
            X_val = df_val[self._feature_cols].copy()
            for col in X_val.columns:
                if X_val[col].isna().any():
                    X_val[col] = X_val[col].fillna(X_train[col].median())
            y_val = df_val["delta"].clip(lower=self._min_delta, upper=self._max_delta)

        # Train threshold classifiers
        for k in self.thresholds:
            y_binary = (y_train >= k).astype(int)
            pos_rate = y_binary.mean()

            # Handle extreme imbalance
            if pos_rate > 0.995 or pos_rate < 0.005:
                logger.warning(
                    f"Threshold {k}: extreme imbalance (pos_rate={pos_rate:.4f}), "
                    "using constant predictor"
                )
                self.classifiers[k] = {"type": "constant", "prob": float(pos_rate)}
                continue

            if pos_rate < 0.02 or pos_rate > 0.98:
                logger.warning(f"Threshold {k}: highly imbalanced ({pos_rate:.1%} positive)")

            model = self._create_base_model()
            train_pool = Pool(X_train, y_binary)

            if X_val is not None:
                y_val_binary = (y_val >= k).astype(int)
                val_pool = Pool(X_val, y_val_binary)
                model.fit(
                    train_pool,
                    eval_set=val_pool,
                    early_stopping_rounds=100,
                )
            else:
                model.fit(train_pool)

            self.classifiers[k] = model

            if self.verbose:
                logger.info(f"Threshold {k} trained (pos_rate={pos_rate:.1%})")

        logger.info(f"Trained {len(self.classifiers)} classifiers")

        # Calibration (requires validation set)
        if self.calibrate != 'none' and X_val is not None:
            logger.info(f"Calibrating threshold classifiers with {self.calibrate} method...")
            self._calibrate_classifiers(X_val, y_val)
        elif self.calibrate != 'none':
            logger.warning("Calibration requested but no validation set provided - skipping")

        return self

    def _calibrate_classifiers(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """Calibrate threshold classifiers using validation set.

        For each threshold k, fit a calibrator on P(delta >= k) predictions.
        """
        for k in self.thresholds:
            clf = self.classifiers[k]

            # Skip constant predictors
            if isinstance(clf, dict) and clf.get("type") == "constant":
                continue

            # Get raw probabilities from classifier
            y_binary = (y_val >= k).astype(int)
            proba_raw = clf.predict_proba(X_val)[:, 1]

            # Fit calibrator
            if self.calibrate == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(proba_raw, y_binary)
            else:  # sigmoid / Platt
                # Use logistic regression on log-odds for Platt scaling
                calibrator = SklearnLogisticRegression(solver='lbfgs', max_iter=1000)
                calibrator.fit(proba_raw.reshape(-1, 1), y_binary)

            self.calibrators[k] = calibrator

            if self.verbose:
                logger.info(f"Calibrated threshold {k}")

        logger.info(f"Calibrated {len(self.calibrators)} classifiers")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Returns:
            Array of shape (n_samples, n_classes) with P(delta=k) for each class
        """
        X = df[self._feature_cols].copy()
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        n = len(X)
        n_classes = len(self._delta_classes)

        # Get P(delta >= k) for each threshold
        exceedance_probs = np.zeros((n, len(self.thresholds)))

        for i, k in enumerate(self.thresholds):
            clf = self.classifiers[k]
            if isinstance(clf, dict) and clf["type"] == "constant":
                exceedance_probs[:, i] = clf["prob"]
            else:
                raw_proba = clf.predict_proba(X)[:, 1]

                # Apply calibration if available
                if k in self.calibrators:
                    calibrator = self.calibrators[k]
                    if self.calibrate == 'isotonic':
                        exceedance_probs[:, i] = calibrator.predict(raw_proba)
                    else:  # sigmoid
                        exceedance_probs[:, i] = calibrator.predict_proba(
                            raw_proba.reshape(-1, 1)
                        )[:, 1]
                else:
                    exceedance_probs[:, i] = raw_proba

        # Enforce monotonicity: P(delta >= k) should decrease with k
        for i in range(1, len(self.thresholds)):
            exceedance_probs[:, i] = np.minimum(
                exceedance_probs[:, i],
                exceedance_probs[:, i-1]
            )

        # Convert to class probabilities
        proba = np.zeros((n, n_classes))

        # P(delta = min_class) = 1 - P(delta >= min_class + 1)
        proba[:, 0] = 1.0 - exceedance_probs[:, 0]

        # P(delta = k) = P(delta >= k) - P(delta >= k+1)
        for i in range(len(self.thresholds) - 1):
            proba[:, i+1] = exceedance_probs[:, i] - exceedance_probs[:, i+1]

        # P(delta = max_class) = P(delta >= max_class)
        proba[:, -1] = exceedance_probs[:, -1]

        # Clip to [0, 1] and normalize
        proba = np.clip(proba, 0, 1)
        row_sums = proba.sum(axis=1, keepdims=True)
        proba = proba / np.maximum(row_sums, 1e-10)

        return proba

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict delta class (expected value)."""
        proba = self.predict_proba(df)
        expected = (proba * np.array(self._delta_classes)).sum(axis=1)
        return np.round(expected).astype(int)

    def save(self, path: Path) -> None:
        """Save model to file."""
        joblib.dump({
            "classifiers": self.classifiers,
            "calibrators": self.calibrators,
            "calibrate": self.calibrate,
            "thresholds": self.thresholds,
            "min_delta": self._min_delta,
            "max_delta": self._max_delta,
            "delta_classes": self._delta_classes,
            "feature_cols": self._feature_cols,
            "catboost_params": self.catboost_params,
            "best_params": self.best_params,
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "MarketClockOrdinalTrainer":
        """Load model from file."""
        data = joblib.load(path)
        trainer = cls()
        trainer.classifiers = data["classifiers"]
        trainer.calibrators = data.get("calibrators", {})
        trainer.calibrate = data.get("calibrate", "none")
        trainer.thresholds = data["thresholds"]
        trainer._min_delta = data["min_delta"]
        trainer._max_delta = data["max_delta"]
        trainer._delta_classes = data["delta_classes"]
        trainer._feature_cols = data["feature_cols"]
        trainer.catboost_params = data.get("catboost_params", {})
        trainer.best_params = data.get("best_params", {})
        return trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Market-Clock TOD v1 Global Model'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input parquet dataset path'
    )

    parser.add_argument(
        '--test-days',
        type=int,
        default=30,
        help='Days to hold out for test set (default: 30)'
    )

    parser.add_argument(
        '--use-optuna',
        action='store_true',
        help='Enable Optuna hyperparameter tuning (v1.1)'
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=30,
        help='Optuna trials (default: 30)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: models/saved/market_clock_tod_v1/)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show verbose output'
    )

    parser.add_argument(
        '--calibrate',
        choices=['none', 'isotonic', 'sigmoid'],
        default='none',
        help='Calibration method: none, isotonic, or sigmoid/platt (default: none)'
    )

    parser.add_argument(
        '--feature-importance',
        action='store_true',
        help='Save feature importance report after training'
    )

    # Delta range bounds for Optuna search
    parser.add_argument(
        '--delta-min-range',
        type=str,
        default='-14,-8',
        help='Optuna search range for delta_min as "low,high" (default: -14,-8)'
    )

    parser.add_argument(
        '--delta-max-range',
        type=str,
        default='8,14',
        help='Optuna search range for delta_max as "low,high" (default: 8,14)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    logger.info("=" * 80)
    logger.info("TRAINING MARKET-CLOCK TOD V1 GLOBAL MODEL")
    logger.info("=" * 80)
    logger.info(f"Input: {input_path}")
    logger.info(f"Test days: {args.test_days}")
    logger.info(f"Optuna: {args.use_optuna} (trials={args.trials})")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Convert event_date to datetime for splitting
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])

    # Use 'day' column for splitting if available (for lag feature compat)
    if 'day' not in df.columns and 'event_date' in df.columns:
        df['day'] = df['event_date']

    # Time-based train/test split
    test_cutoff = df['event_date'].max() - pd.Timedelta(days=args.test_days)
    df_train = df[df['event_date'] <= test_cutoff].copy()
    df_test = df[df['event_date'] > test_cutoff].copy()

    # Guard: ensure we have training data
    if len(df_train) == 0:
        logger.error("No training data after time split – reduce --test-days or expand date range.")
        return 1

    logger.info(f"\nTrain: {len(df_train)} rows ({df_train['event_date'].nunique()} days)")
    logger.info(f"Test: {len(df_test)} rows ({df_test['event_date'].nunique()} days)")

    # City breakdown
    if 'city' in df_train.columns:
        logger.info("\nCity breakdown (train):")
        for city in sorted(df_train['city'].unique()):
            city_rows = len(df_train[df_train['city'] == city])
            logger.info(f"  {city}: {city_rows:,} rows")

    # Parse delta range arguments
    delta_min_range = tuple(int(x) for x in args.delta_min_range.split(','))
    delta_max_range = tuple(int(x) for x in args.delta_max_range.split(','))
    logger.info(f"Delta range search: min in {delta_min_range}, max in {delta_max_range}")

    # Train model
    n_trials = args.trials if args.use_optuna else 0
    trainer = MarketClockOrdinalTrainer(
        n_trials=n_trials,
        verbose=args.verbose,
        delta_min_range=delta_min_range,
        delta_max_range=delta_max_range,
        calibrate=args.calibrate,
    )
    logger.info(f"Calibration: {args.calibrate}")

    trainer.train(df_train, df_test if len(df_test) > 0 else None)

    # Evaluate
    if len(df_test) > 0:
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION")
        logger.info("=" * 80)

        y_true = df_test['delta'].values
        y_pred = trainer.predict(df_test)
        proba = trainer.predict_proba(df_test)

        delta_metrics = compute_delta_metrics(y_true, y_pred)
        # Pass explicit classes for correct rank ordering in ordinal metrics
        ordinal_metrics = compute_ordinal_metrics(
            y_true, proba, classes=trainer._delta_classes
        )

        logger.info(f"\nTest Metrics:")
        logger.info(f"  Accuracy: {delta_metrics['delta_accuracy']:.1%}")
        logger.info(f"  MAE: {delta_metrics['delta_mae']:.3f}")
        logger.info(f"  Within-1: {delta_metrics['within_1_rate']:.1%}")
        logger.info(f"  Within-2: {delta_metrics['within_2_rate']:.1%}")
        logger.info(f"  Ordinal Loss: {ordinal_metrics['ordinal_loss']:.4f}")

        # Per-city metrics if applicable
        if 'city' in df_test.columns and df_test['city'].nunique() > 1:
            logger.info("\nPer-city metrics:")
            for city in sorted(df_test['city'].unique()):
                city_mask = df_test['city'] == city
                city_y_true = y_true[city_mask]
                city_y_pred = y_pred[city_mask]
                city_metrics = compute_delta_metrics(city_y_true, city_y_pred)
                logger.info(
                    f"  {city}: Acc={city_metrics['delta_accuracy']:.1%}, "
                    f"MAE={city_metrics['delta_mae']:.3f}, "
                    f"W1={city_metrics['within_1_rate']:.1%}"
                )
    else:
        delta_metrics = {}
        ordinal_metrics = {}
        logger.warning("No test data - skipping evaluation")

    # Save model
    logger.info("\n" + "=" * 80)
    logger.info("SAVING MODEL")
    logger.info("=" * 80)

    model_path = output_dir / "ordinal_catboost_market_clock_tod_v1.pkl"
    trainer.save(model_path)

    # Save metadata
    from datetime import datetime
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "model_variant": "market_clock_tod_v1",
        "model_type": "global",
        "calibration": args.calibrate,
        "n_calibrators": len(trainer.calibrators),
        "delta_range": [trainer._min_delta, trainer._max_delta],
        "delta_range_search": {
            "min_range": list(delta_min_range),
            "max_range": list(delta_max_range),
        },
        "delta_range_optimized": trainer.best_delta_range is not None,
        "n_thresholds": len(trainer.thresholds),
        "n_features": len(trainer._feature_cols),
        "n_optuna_trials": n_trials,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "n_train_days": df_train['event_date'].nunique() if 'event_date' in df_train else 0,
        "n_test_days": df_test['event_date'].nunique() if 'event_date' in df_test else 0,
        "cities": sorted(df['city'].unique().tolist()) if 'city' in df else [],
        "metrics": {
            "accuracy": delta_metrics.get('delta_accuracy'),
            "mae": delta_metrics.get('delta_mae'),
            "within_1": delta_metrics.get('within_1_rate'),
            "within_2": delta_metrics.get('within_2_rate'),
            "ordinal_loss": ordinal_metrics.get('ordinal_loss'),
        },
        "best_params": trainer.best_params,
        "feature_cols": trainer._feature_cols,
    }

    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved metadata to {metadata_path}")

    # Save feature columns
    features_path = output_dir / "feature_columns.txt"
    with open(features_path, 'w') as f:
        for col in trainer._feature_cols:
            f.write(f"{col}\n")
    logger.info(f"Saved feature columns to {features_path}")

    # Feature importance (aggregate across threshold classifiers)
    if args.feature_importance:
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE IMPORTANCE")
        logger.info("=" * 80)

        importance_agg = {}
        n_classifiers = 0

        for k, clf in trainer.classifiers.items():
            if isinstance(clf, dict) and clf.get("type") == "constant":
                continue  # Skip constant predictors

            n_classifiers += 1
            imp = clf.get_feature_importance()
            for i, col in enumerate(trainer._feature_cols):
                importance_agg[col] = importance_agg.get(col, 0) + imp[i]

        # Average and sort
        importance_avg = {k: v / n_classifiers for k, v in importance_agg.items()}
        importance_sorted = sorted(importance_avg.items(), key=lambda x: x[1], reverse=True)

        # Log top 20
        logger.info(f"\nTop 20 features (avg across {n_classifiers} classifiers):")
        for i, (col, imp) in enumerate(importance_sorted[:20], 1):
            logger.info(f"  {i:2d}. {col}: {imp:.2f}")

        # Save to file
        importance_path = output_dir / "feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(dict(importance_sorted), f, indent=2)
        logger.info(f"\nSaved feature importance to {importance_path}")

    # Save train/test data references
    df_train.to_parquet(output_dir / "train_data.parquet", index=False)
    if len(df_test) > 0:
        df_test.to_parquet(output_dir / "test_data.parquet", index=False)
    logger.info(f"Saved train/test data to {output_dir}")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
