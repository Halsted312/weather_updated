---
plan_id: ml-framework-intraday-delta-models
created: 2025-11-28
status: completed
priority: high
agent: kalshi-weather-quant
---

# ML Framework for Intraday Temperature Settlement Prediction

## Objective

Create a modular ML framework to train, evaluate, and serve intraday Δ-models that predict `Δ = T_settle - T_base` (deviation from partial-day baseline) for Kalshi temperature settlement.

## Context

- **Data ingestion** is being handled by another agent (historical forecasts backfill)
- **Target**: Predict discrete temperature delta classes `Δ ∈ {-2, -1, 0, +1, +2}`
- **Models**: Multinomial Logistic (Elastic Net + Platt) and CatBoost (Optuna + Platt)
- **Features**: 50+ features across 6 categories, computed from partial-day data only (no lookahead)
- **Existing infrastructure**: `analysis/temperature/rules.py` has deterministic rules to import

## Directory Structure

```
models/
├── __init__.py
├── README.md                          # Framework overview and usage guide
│
├── features/                          # Feature engineering (pure functions)
│   ├── __init__.py                    # Exports ALL_FEATURES registry
│   ├── base.py                        # FeatureSet dataclass, composition utilities
│   ├── partial_day.py                 # Base stats from VC obs up to τ
│   ├── shape.py                       # Plateau, spike, slope features
│   ├── rules.py                       # Wraps analysis/temperature/rules.py
│   ├── forecast.py                    # T-1 forecast + forecast error deltas
│   ├── calendar.py                    # Snapshot hour, doy, lags
│   └── quality.py                     # Missing fraction, gaps, edge flags
│
├── data/                              # Data loading and dataset construction
│   ├── __init__.py
│   ├── loader.py                      # DB queries for training and inference
│   ├── snapshot_builder.py            # Build snapshot-level feature table
│   └── splits.py                      # Train/test splits, TimeSeriesSplit CV
│
├── training/                          # Model training pipelines
│   ├── __init__.py
│   ├── base_trainer.py                # Abstract base class for trainers
│   ├── logistic_trainer.py            # Model 1: Logistic + Elastic Net + Platt
│   └── catboost_trainer.py            # Model 2: CatBoost + Optuna + Platt
│
├── evaluation/                        # Metrics and evaluation
│   ├── __init__.py
│   ├── metrics.py                     # Accuracy, MAE, Brier score, calibration
│   ├── evaluator.py                   # Run full evaluation suite
│   └── reports.py                     # Generate evaluation reports
│
├── inference/                         # Live inference
│   ├── __init__.py
│   ├── predictor.py                   # Load model, compute features, predict
│   └── probability.py                 # Convert Δ probs to bracket probs
│
├── saved/                             # Trained model artifacts
│   └── .gitkeep
│
└── reports/                           # Evaluation outputs
    └── .gitkeep
```

## File Specifications

### 1. features/base.py (~100 lines)
```python
"""
Base utilities for feature engineering.

Defines FeatureSet dataclass and composition utilities for building
feature vectors from partial-day observations.
"""

@dataclass
class FeatureSet:
    """Container for a named group of features."""
    name: str
    features: dict[str, float | int | None]

def compose_features(*feature_sets: FeatureSet) -> dict[str, Any]:
    """Merge multiple FeatureSets into one dict."""
    ...

# Registry pattern for all feature functions
ALL_FEATURE_GROUPS: dict[str, Callable] = {}
```

### 2. features/partial_day.py (~150 lines)
```python
"""
Partial-day observation features.

Computes base statistics from Visual Crossing temps observed up to
snapshot time τ. No future data used.

Features:
- vc_max_f_sofar, vc_min_f_sofar, vc_mean_f_sofar, vc_std_f_sofar
- vc_q10_f_sofar through vc_q90_f_sofar (percentiles)
- vc_frac_part_sofar (fractional part of max, for rounding behavior)
- num_samples_sofar
- t_base (rounded max = baseline for Δ target)
"""

def compute_partial_day_features(
    temps_sofar: list[float],
) -> FeatureSet:
    """Compute base stats from temps observed up to τ."""
    ...

def compute_delta_target(
    settle_f: int,
    vc_max_f_sofar: float,
) -> dict[str, int]:
    """Compute t_base and delta target."""
    ...
```

### 3. features/shape.py (~200 lines)
```python
"""
Shape-of-day features for spike vs plateau detection.

Analyzes the temporal pattern of temperatures to distinguish:
- Sustained plateaus (high confidence in current max)
- Brief spikes (may see higher later)

Features:
- minutes_ge_base, minutes_ge_base_p1, minutes_ge_base_m1
- max_run_ge_base, max_run_ge_base_p1, max_run_ge_base_m1
- max_minus_second_max (spike indicator)
- max_morning_f_sofar, max_afternoon_f_sofar, max_evening_f_sofar
- slope_max_30min_up_sofar, slope_max_30min_down_sofar
"""

def compute_shape_features(
    temps_sofar: list[float],
    timestamps_local_sofar: list[datetime],
    t_base: int,
    step_minutes: int = 5,
) -> FeatureSet:
    """Compute plateau and shape features from partial-day data."""
    ...
```

### 4. features/rules.py (~150 lines)
```python
"""
Rule-based meta-features.

Wraps deterministic rules from analysis/temperature/rules.py and
computes:
- Predictions from each rule applied to partial-day data
- Errors vs settlement (for training) or vs t_base (for inference)
- Disagreement signals between rules

Features:
- pred_{rule}_sofar for each rule
- err_{rule}_sofar for each rule
- range_pred_rules_sofar, num_distinct_preds_sofar, disagree_flag_sofar
"""

from analysis.temperature.rules import ALL_RULES

def compute_rule_features(
    temps_sofar: list[float],
    settle_f: int | None = None,  # None for inference
) -> FeatureSet:
    """Apply all rules to partial-day temps, compute meta-features."""
    ...
```

### 5. features/forecast.py (~200 lines)
```python
"""
T-1 forecast and forecast-vs-actual error features.

Uses the forecast issued on day D-1 for target day D, plus
real-time comparison with actual observations.

Static features (known at start of day):
- fcst_prev_max_f, fcst_prev_min_f, fcst_prev_mean_f, fcst_prev_std_f
- fcst_prev_q10_f through fcst_prev_q90_f
- fcst_prev_frac_part, fcst_prev_hour_of_max
- t_forecast_base (rounded forecast high)

Dynamic features (computed as obs come in):
- err_mean_sofar, err_std_sofar (forecast vs actual bias)
- err_max_pos_sofar, err_max_neg_sofar (overshoot/undershoot)
- err_abs_mean_sofar, err_last1h, err_last3h_mean
- delta_vcmax_fcstmax_sofar
"""

def compute_forecast_static_features(
    fcst_series: list[float],  # T-1 forecast temps for day D
) -> FeatureSet:
    """Compute features from yesterday's forecast (no obs needed)."""
    ...

def compute_forecast_error_features(
    fcst_series_sofar: list[float],
    obs_series_sofar: list[float],
) -> FeatureSet:
    """Compute forecast-vs-actual deltas up to τ."""
    ...
```

### 6. features/calendar.py (~120 lines)
```python
"""
Calendar and temporal features.

Encodes time-of-day, day-of-year, and lag features.

Features:
- snapshot_hour, snapshot_hour_sin, snapshot_hour_cos
- doy_sin, doy_cos, week_sin, week_cos
- month, is_weekend
- settle_f_lag1, settle_f_lag2, settle_f_lag7
- vc_max_f_lag1, vc_max_f_lag7
- delta_vcmax_lag1
"""

def compute_calendar_features(
    day: date,
    snapshot_hour: int,
) -> FeatureSet:
    """Compute calendar/time encoding features."""
    ...

def compute_lag_features(
    df: pd.DataFrame,  # Full dataset with prior days
    city: str,
    day: date,
) -> FeatureSet:
    """Compute lag features from historical data."""
    ...
```

### 7. features/quality.py (~80 lines)
```python
"""
Data quality features.

Flags potential data quality issues that may affect predictions.

Features:
- missing_fraction_sofar
- max_gap_minutes (largest gap in observation series)
- edge_max_flag (max near start/end of data window)
"""

def compute_quality_features(
    temps_sofar: list[float],
    timestamps_sofar: list[datetime],
    expected_samples: int,
) -> FeatureSet:
    """Compute data quality indicators."""
    ...
```

### 8. data/loader.py (~200 lines)
```python
"""
Data loading from database.

Provides unified interface for loading:
- Historical data (for training): VC obs + settlements + forecasts
- Current data (for inference): VC obs up to now + yesterday's forecast

Both use the same feature functions - only the data source differs.
"""

def load_training_data(
    cities: list[str],
    start_date: date,
    end_date: date,
    session: Session,
) -> pd.DataFrame:
    """Load historical obs + settlements for training."""
    ...

def load_inference_data(
    city: str,
    target_date: date,
    cutoff_time: datetime,
    session: Session,
) -> dict:
    """Load current obs + T-1 forecast for live inference."""
    ...

def load_historical_forecast(
    city: str,
    target_date: date,
    basis_date: date,
    session: Session,
) -> list[float]:
    """Load T-1 forecast series for a given day."""
    ...
```

### 9. data/snapshot_builder.py (~250 lines)
```python
"""
Snapshot dataset construction.

Builds the training dataset of (city, day, snapshot_hour) rows
with all features computed from partial-day data.

Key constraint: Features at snapshot τ use ONLY data with
datetime_local < τ (no lookahead).
"""

SNAPSHOT_HOURS = [10, 12, 14, 16, 18, 20, 22, 23]

def build_snapshot_dataset(
    cities: list[str],
    start_date: date,
    end_date: date,
    session: Session,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build full snapshot-level feature table for training."""
    ...

def build_single_snapshot(
    city: str,
    day: date,
    snapshot_hour: int,
    temps_sofar: list[float],
    timestamps_sofar: list[datetime],
    fcst_series: list[float],
    settle_f: int,
) -> dict:
    """Build feature row for one snapshot."""
    ...
```

### 10. data/splits.py (~100 lines)
```python
"""
Train/test splitting utilities.

Implements day-based temporal splits to avoid lookahead leakage.
All snapshots from a day go to the same fold.
"""

def train_test_split_by_date(
    df: pd.DataFrame,
    cutoff_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by calendar date (train < cutoff, test >= cutoff)."""
    ...

def make_time_series_cv(
    n_splits: int = 5,
    gap_days: int = 1,
) -> TimeSeriesSplit:
    """Create time-series CV splitter with optional gap."""
    ...
```

### 11. training/base_trainer.py (~150 lines)
```python
"""
Abstract base class for model trainers.

Defines common interface for training, calibration, and saving.
"""

class BaseTrainer(ABC):
    """Base trainer with common training workflow."""

    @abstractmethod
    def train(self, X_train, y_train) -> Any:
        """Train the base model."""
        ...

    def calibrate(self, model, X_train, y_train) -> CalibratedClassifierCV:
        """Apply Platt scaling calibration."""
        ...

    def save(self, model, path: Path) -> None:
        """Save trained model with metadata."""
        ...

    def load(self, path: Path) -> Any:
        """Load trained model."""
        ...
```

### 12. training/logistic_trainer.py (~200 lines)
```python
"""
Model 1: Multinomial Logistic Δ-Model.

Elastic Net regularized logistic regression with Platt calibration.
Good baseline with interpretable coefficients.
"""

class LogisticDeltaTrainer(BaseTrainer):
    """Train multinomial logistic model for Δ prediction."""

    def __init__(
        self,
        l1_ratio: float = 0.5,
        C: float = 1.0,
        max_iter: int = 4000,
    ):
        ...

    def train(self, X_train, y_train) -> Pipeline:
        """Train logistic model with preprocessing."""
        ...

    def get_feature_importance(self) -> pd.DataFrame:
        """Extract coefficient magnitudes per feature."""
        ...
```

### 13. training/catboost_trainer.py (~280 lines)
```python
"""
Model 2: CatBoost Δ-Model with Optuna tuning.

Gradient boosting with native categorical support and
Bayesian hyperparameter optimization.
"""

class CatBoostDeltaTrainer(BaseTrainer):
    """Train CatBoost model with Optuna hyperparameter search."""

    def __init__(
        self,
        n_trials: int = 10,
        cv_splits: int = 3,
    ):
        ...

    def tune_hyperparameters(
        self,
        X_train,
        y_train,
    ) -> dict:
        """Run Optuna study to find best params."""
        ...

    def train(self, X_train, y_train) -> CatBoostClassifier:
        """Train with best hyperparameters."""
        ...

    def get_feature_importance(self) -> pd.DataFrame:
        """Extract CatBoost feature importances."""
        ...
```

### 14. evaluation/metrics.py (~150 lines)
```python
"""
Evaluation metrics for Δ-models.

Computes classification metrics, probabilistic calibration metrics,
and bracket-level performance.
"""

def delta_accuracy(y_true, y_pred) -> float:
    """Exact match accuracy on Δ classes."""
    ...

def delta_mae(y_true, y_pred) -> float:
    """Mean absolute error on Δ."""
    ...

def settlement_accuracy(df_eval: pd.DataFrame) -> float:
    """Accuracy on final T_settle = t_base + delta_pred."""
    ...

def bracket_brier_score(
    proba: np.ndarray,
    t_base: np.ndarray,
    t_settle: np.ndarray,
    threshold: int,
    delta_classes: np.ndarray,
) -> float:
    """Brier score for P(T >= threshold) event."""
    ...

def calibration_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Generate calibration curve data points."""
    ...
```

### 15. evaluation/evaluator.py (~200 lines)
```python
"""
Full evaluation suite for trained models.

Runs comprehensive evaluation and generates reports.
"""

class ModelEvaluator:
    """Evaluate trained Δ-model on test data."""

    def __init__(self, model, df_test: pd.DataFrame):
        ...

    def evaluate_delta(self) -> dict:
        """Compute Δ-level metrics."""
        ...

    def evaluate_settlement(self) -> dict:
        """Compute settlement-level metrics."""
        ...

    def evaluate_brackets(
        self,
        thresholds: list[int] = [80, 85, 90, 95],
    ) -> dict:
        """Compute Brier scores for key bracket thresholds."""
        ...

    def evaluate_by_snapshot_hour(self) -> pd.DataFrame:
        """Stratify metrics by snapshot hour."""
        ...

    def full_evaluation(self) -> dict:
        """Run all evaluations, return combined results."""
        ...
```

### 16. inference/predictor.py (~180 lines)
```python
"""
Live inference predictor.

Loads trained model and computes predictions from current data.
"""

class DeltaPredictor:
    """Load model and predict Δ distribution for live data."""

    def __init__(self, model_path: Path):
        """Load saved model."""
        ...

    def predict(
        self,
        city: str,
        target_date: date,
        cutoff_time: datetime,
        session: Session,
    ) -> dict:
        """
        Predict Δ distribution for (city, date) at cutoff_time.

        Returns:
            {
                't_base': int,
                'delta_probs': {-2: 0.01, -1: 0.05, 0: 0.80, 1: 0.12, 2: 0.02},
                'predicted_delta': int,
                'features': dict,  # For debugging/logging
            }
        """
        ...
```

### 17. inference/probability.py (~120 lines)
```python
"""
Convert Δ probabilities to bracket probabilities.

Translates model output into actionable trading signals.
"""

def delta_probs_to_temp_probs(
    delta_probs: dict[int, float],
    t_base: int,
) -> dict[int, float]:
    """Convert P(Δ=d) to P(T=t) for each temperature."""
    ...

def temp_probs_to_bracket_prob(
    temp_probs: dict[int, float],
    bracket_floor: int,
    bracket_cap: int | None = None,
) -> float:
    """Compute P(floor <= T < cap) from temperature distribution."""
    ...

def compute_all_bracket_probs(
    delta_probs: dict[int, float],
    t_base: int,
    thresholds: list[int],
) -> dict[str, float]:
    """Compute P(T >= K) for multiple thresholds."""
    ...
```

## Implementation Order

### Phase 1: Core Infrastructure
1. `features/base.py` - FeatureSet and composition
2. `features/partial_day.py` - Base stats (most fundamental)
3. `features/calendar.py` - Time features
4. `data/loader.py` - DB queries

### Phase 2: Feature Expansion
5. `features/shape.py` - Plateau/spike detection
6. `features/rules.py` - Rule meta-features
7. `features/quality.py` - Quality flags
8. `features/forecast.py` - Forecast features (depends on forecast data being available)

### Phase 3: Dataset & Training
9. `data/snapshot_builder.py` - Build training dataset
10. `data/splits.py` - CV utilities
11. `training/base_trainer.py` - Abstract trainer
12. `training/logistic_trainer.py` - Model 1

### Phase 4: Evaluation & Model 2
13. `evaluation/metrics.py` - All metrics
14. `evaluation/evaluator.py` - Evaluation harness
15. `training/catboost_trainer.py` - Model 2

### Phase 5: Inference
16. `inference/predictor.py` - Live prediction
17. `inference/probability.py` - Bracket conversion
18. `evaluation/reports.py` - Report generation

## Dependencies to Add

```toml
# pyproject.toml additions
catboost = "^1.2"
optuna = "^3.4"
```

## Key Design Principles

1. **Pure feature functions**: Take data in, return features out. No DB coupling inside feature code.

2. **No lookahead**: Features at snapshot τ use ONLY data with `datetime_local < τ`.

3. **DRY**: Import rules from `analysis/temperature/rules.py`, don't duplicate.

4. **Files under 300 lines**: Split functionality across focused modules.

5. **Registry pattern**: `ALL_FEATURE_GROUPS` dict for easy iteration and composition.

6. **Documented**: Each file has module docstring explaining purpose, features produced.

7. **Type hints**: All functions fully typed.

8. **Testable**: Pure functions are easy to unit test with mock data.

## Testing Strategy

- Unit tests for each feature function with known inputs/outputs
- Integration test: Build small snapshot dataset, verify no NaN leakage
- Model test: Train on subset, verify predictions are in expected range
- Temporal test: Verify train/test split respects date boundaries

## Comparison Output File

A key deliverable is `models/reports/model_comparison.csv` (and `.md` summary) containing:
- **Classification**: Accuracy, off-by-1 rate, off-by-2+ rate
- **Regression**: MAE, RMSE on settlement temp
- **Probabilistic**: Log loss, Brier scores for key thresholds (80, 85, 90, 95°F)
- **Calibration**: Reliability curve data points
- **By snapshot hour**: All metrics stratified by time-of-day
- **By model**: Side-by-side comparison of Logistic vs CatBoost

This allows quick comparison of which model performs better and at what times.

## Implementation Notes

- **Start with Chicago only** (KMDW) - other cities being loaded by parallel agent
- **Incremental**: Get Chicago working end-to-end before expanding to other cities

## Success Criteria

- [ ] All feature functions implemented and documented
- [ ] Snapshot dataset builds without errors for Chicago
- [ ] Model 1 (logistic) trains and produces calibrated probabilities
- [ ] Model 2 (CatBoost) trains with Optuna tuning
- [ ] Comparison report generated with all metrics
- [ ] Evaluation reports show metrics by snapshot hour
- [ ] Inference works: load model, query current data, get bracket probs
- [ ] Models saved to `models/saved/` with metadata

## Sign-off Log

### 2025-11-28 Session 3 (Final - Training & Validation Complete)
**Status**: Completed - Models trained, validated, and saved

**Critical Fixes This Session**:
- ✅ **Fixed Feature Leakage**: Removed 7 `err_{rule}_sofar` features that used `settle_f` (the target). These had -0.969 correlation with delta and caused CatBoost to achieve 100% "accuracy" via leakage.
- ✅ **Fixed CatBoost sklearn Interface**: Added `get_params()`, `set_params()`, and `_estimator_type = "classifier"` to `CatBoostCalibratedWrapper` for sklearn compatibility with `CalibratedClassifierCV`.
- ✅ **Fixed CatBoost Bootstrap Params**: Made `bootstrap_type` a hyperparameter with conditional params (`bagging_temperature` for Bayesian, `subsample` for Bernoulli).

**Feature Engineering Improvements**:
- ✅ Removed useless features (constant or 99%+ same value):
  - `minutes_ge_base_p1`, `max_run_ge_base_p1` (constant at 0)
  - `disagree_flag_sofar` (99.4% constant at 1)
  - `max_gap_minutes` (99.6% constant at 5)
- ✅ Removed redundant features:
  - `pred_max_round_sofar`, `pred_max_of_rounded_sofar` (identical to `t_base`)
  - `pred_c_first_sofar`, `pred_ignore_singletons_sofar` (low differentiation)
- ✅ Added 4 new derived features with high correlation to delta (no leakage):
  - `obs_fcst_max_gap` (0.654 corr) - upside potential
  - `hours_until_fcst_max` (0.547 corr) - time until expected max
  - `above_fcst_flag` (-0.562 corr) - already exceeded forecast
  - `day_fraction` (-0.653 corr) - proportion of day elapsed
- ✅ Expanded DELTA_CLASSES from [-2, +2] to [-2, +10] (13 classes) to handle early-morning snapshots

**Null Handling & Scaling**:
- ✅ Structural nulls (time-based features like `max_afternoon_f_sofar`, `max_evening_f_sofar`) now filled with `vc_max_f_sofar` instead of median
- ✅ Added `RobustScaler` (median/IQR-based) to `LogisticDeltaTrainer` for proper L1/L2 regularization with skewed features
- ✅ Kept median imputation for random nulls (lag features ~1.4%)

**CatBoost Hyperparameter Expansion** (30 trials):
- Added: `min_data_in_leaf`, `random_strength`, `colsample_bylevel`
- Added conditional: `bootstrap_type` with `bagging_temperature` (Bayesian) or `subsample` (Bernoulli)

**Final Model Results (Chicago, train < 2025-06-01)**:

| Model | Accuracy | MAE | ECE | Best For |
|-------|----------|-----|-----|----------|
| CatBoost v3 | 53.2% | 1.05 | 0.026 | Most hours, calibration |
| Logistic v3 | 46.4% | 0.99 | 0.033 | 10am, interpretability |

Best CatBoost params: depth=7, iterations=469, learning_rate=0.032, l2_leaf_reg=1.09, min_data_in_leaf=19, random_strength=1.73, colsample_bylevel=0.96, bootstrap_type=Bayesian, bagging_temperature=0.44

**Files Modified**:
- `models/features/base.py` - Updated NUMERIC_FEATURE_COLS, DELTA_CLASSES, added comments
- `models/training/base_trainer.py` - Added structural null handling, derived feature computation
- `models/training/logistic_trainer.py` - Added RobustScaler, updated imports
- `models/training/catboost_trainer.py` - Fixed sklearn interface, expanded hyperparameters

**Saved Artifacts**:
- `models/saved/catboost_chicago_v3.pkl` + `.json` metadata
- `models/saved/logistic_chicago_v3.pkl` + `.json` metadata

**Decision on Platt Scaling**: Kept Platt scaling - it calibrates raw probabilities into trustworthy probabilities for trading decisions. The probabilities need to be well-calibrated (ECE < 0.03) for bracket probability calculations.

**Next Steps for Future Sessions**:
1. Train on all 6 cities once data ingestion completes
2. Build inference pipeline for live predictions
3. Integrate with open_maker strategies for bracket selection
4. Add hour-specific models or ensemble approach
5. Consider ordinal regression (CORN) for better ordinal handling

---

### 2025-11-28 Session 2 (Completion)
**Status**: Completed - All core modules implemented

**Completed this session**:
- ✅ All 17+ Python files implemented across 5 sub-packages
- ✅ Features modules: base.py, partial_day.py, calendar.py, shape.py, rules.py, quality.py, forecast.py
- ✅ Data modules: loader.py, snapshot_builder.py, splits.py
- ✅ Training modules: base_trainer.py, logistic_trainer.py, catboost_trainer.py
- ✅ Evaluation modules: metrics.py, evaluator.py, reports.py
- ✅ Inference modules: predictor.py, probability.py
- ✅ All __init__.py files with proper exports
- ✅ README.md documentation
- ✅ Dependencies added to pyproject.toml (catboost, optuna, joblib)
- ✅ All imports verified working

**Files created (17 core + 8 init/config)**:
```
models/
├── __init__.py
├── README.md
├── features/
│   ├── __init__.py
│   ├── base.py (143 lines)
│   ├── partial_day.py (141 lines)
│   ├── calendar.py (157 lines)
│   ├── shape.py (177 lines)
│   ├── rules.py (136 lines)
│   ├── quality.py (89 lines)
│   └── forecast.py (229 lines)
├── data/
│   ├── __init__.py
│   ├── loader.py (219 lines)
│   ├── snapshot_builder.py (175 lines)
│   └── splits.py (157 lines)
├── training/
│   ├── __init__.py
│   ├── base_trainer.py (164 lines)
│   ├── logistic_trainer.py (175 lines)
│   └── catboost_trainer.py (256 lines)
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py (344 lines)
│   ├── evaluator.py (265 lines)
│   └── reports.py (291 lines)
├── inference/
│   ├── __init__.py
│   ├── predictor.py (262 lines)
│   └── probability.py (301 lines)
├── saved/
│   └── .gitkeep
└── reports/
    └── .gitkeep
```

**Ready for next steps**:
1. Run end-to-end training test with Chicago data
2. Generate first comparison report
3. Tune hyperparameters with more Optuna trials once data is fully loaded

---

### 2025-11-28 Session 1
**Status**: In progress - user approved

**Design decisions confirmed**:
- Top-level `models/` directory
- Descriptive file names (not numbered)
- Features split by category (6 files)
- Import rules from `analysis/temperature/rules.py`
- DB query for live inference (not streaming buffer)
- `models/saved/` for artifacts, `models/reports/` for outputs
- Comparison output file for model evaluation
- Start with Chicago (KMDW) only

**User approved**: Ready to implement
