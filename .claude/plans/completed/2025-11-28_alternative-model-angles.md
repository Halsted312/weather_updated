---
plan_id: alternative-model-angles
created: 2025-11-28
status: in_progress
priority: medium
agent: kalshi-weather-quant
---

# Alternative Modeling Approaches for Temperature Settlement

## Objective

Explore alternative model architectures that complement the existing delta-model framework, reusing the same features and evaluation infrastructure while providing different conceptual angles.

## Context

**What we have (completed in ml-framework-intraday-delta-models plan)**:
- Δ-model: predicts `delta = settle_f - t_base` as discrete classes [-2, +10]
- CatBoost with Optuna (53.2% accuracy, 1.05 MAE, 0.026 ECE)
- Logistic with Elastic Net + RobustScaler (46.4% accuracy, 0.99 MAE)
- Platt calibration, time-series CV with day grouping
- ~60 features (partial day stats, shape, rules, forecast, calendar, quality)

**Problem with current approach**:
- Multinomial classification treats delta classes as **unordered categories**
- Predicting delta=+5 when true delta=+6 is penalized the same as predicting delta=-2
- The model doesn't explicitly know that -2 < -1 < 0 < +1 < ... < +10
- Early-hour predictions (10am-2pm) are particularly challenging

**Opportunity**:
- Ordinal regression respects the natural ordering of delta
- Different targets (residual from forecast vs residual from obs max) may be easier to predict
- Hour-specific models could capture time-varying behavior better
- Same features and evaluation can validate which angle works best

## Candidate Approaches

### 1. Ordinal Regression (CORN/CORAL) - Primary Focus

**Concept**: Instead of multinomial P(delta=k), model cumulative probabilities P(delta >= k).

**Why it matters**:
- Delta is inherently ordered: -2 < -1 < 0 < +1 < +2 < ... < +10
- Ordinal models impose monotonicity: P(delta >= k) >= P(delta >= k+1)
- Predictions naturally respect "close is better than far"
- Direct computation of bracket probabilities: P(T >= threshold) = P(delta >= threshold - t_base)

**Implementation options**:
1. **CORN (Conditional Ordinal Regression Network)**: sklearn-compatible via `coral_pytorch` or manual implementation
2. **All-Threshold Model**: Train K-1 binary classifiers for delta >= {-1, 0, 1, ..., 10}
3. **Proportional Odds Logistic**: `statsmodels.miscmodels.ordinal_model.OrderedModel`
4. **CatBoost with ordinal loss**: Custom loss or multi-output with monotonicity

**Expected benefit**: Lower MAE (respects ordering), potentially better calibration for tail events.

### 2. Forecast Residual Model

**Concept**: Instead of `delta = settle - t_base`, predict `residual = settle - fcst_prev_max_f`.

**Why it matters**:
- T-1 forecast is often closer to settlement than t_base (especially early in day)
- The residual captures "forecast error" which may have different patterns
- At 10am, `t_base` might be 85°F when settlement is 93°F (delta=+8)
- But forecast might be 92°F, so residual is only +1 (easier to predict)

**Implementation**:
- Change target computation in `_prepare_features` or add parallel target
- Train same models on new target
- Compare: which target is more predictable at each snapshot hour?

### 3. Hour-Stratified Models

**Concept**: Train separate models for different snapshot hours.

**Why it matters**:
- 10am model sees very different data than 10pm model
- Feature importance varies: early hours rely on forecast, late hours rely on observed max
- Current single model uses `snapshot_hour` as feature, but can't capture structural differences

**Implementation**:
- Cluster hours: Morning (10-12), Afternoon (14-16), Evening (18-20), Night (22-23)
- Train 4 specialized models
- Ensemble: route to appropriate model based on hour
- Compare: combined performance vs single global model

### 4. Direct Temperature Model (Lower Priority)

**Concept**: Predict `settle_f` directly instead of delta.

**Why**: Avoids the t_base dependency, but may need temperature binning or wider class range.

## Proposed Implementation Order

1. **Ordinal Regression** (primary) - Different model architecture, same target
2. **Forecast Residual** (secondary) - Same models, different target
3. **Hour-Stratified** (tertiary) - Same models, partitioned training
4. **Comparison report** - Evaluate all approaches on same test set

## Tasks

### Phase 1: Ordinal Regression Framework ✅
- [x] Research ordinal regression options in sklearn/statsmodels/pytorch
- [x] Implement `OrdinalDeltaTrainer` extending `BaseTrainer`
- [x] Add ordinal loss: All-Threshold approach with CatBoost/Logistic base
- [x] Ensure compatibility with existing `_prepare_features` and evaluation

### Phase 2: Training & Evaluation ✅
- [x] Train ordinal model on Chicago data (same train/test split)
- [x] Compare metrics: accuracy, MAE, off-by-1 rate, off-by-2+ rate
- [x] Compare calibration: ordinal loss, cumulative accuracy
- [x] Compare by snapshot hour

### Phase 3: Forecast Residual Target ✅
- [x] Add `residual` target column: `settle_f - fcst_prev_max_f`
- [x] Train Ordinal CatBoost on residual target
- [x] Evaluate: is residual more predictable than delta? **NO - delta is better**

### Phase 4: Hour-Stratified Models (Deferred)
- [ ] Define hour clusters (Morning/Afternoon/Evening/Night)
- [ ] Train separate CatBoost models per cluster
- [ ] Build ensemble predictor that routes by hour
- [ ] Compare vs global model

### Phase 5: Comprehensive Comparison ✅
- [x] Generate comparison report: all models × all hours × all metrics
- [x] Identify best approach for each scenario
- [x] Document findings and recommendations

## Files to Create/Modify

| Action | Path | Notes |
|--------|------|-------|
| CREATE | `models/training/ordinal_trainer.py` | Ordinal regression trainer |
| MODIFY | `models/features/base.py` | Add residual target option |
| MODIFY | `models/training/base_trainer.py` | Support multiple targets |
| CREATE | `models/training/hour_stratified.py` | Hour-based ensemble |
| MODIFY | `models/evaluation/metrics.py` | Add ordinal-specific metrics |
| CREATE | `models/reports/model_comparison_v2.md` | Comprehensive comparison |

## Technical Details

### Ordinal Regression Implementation

**Option A: All-Threshold Binary Classifiers**
```python
# For each threshold k in {-1, 0, 1, ..., 10}:
# Train binary classifier for P(delta >= k)
# At inference: P(delta = k) = P(delta >= k) - P(delta >= k+1)

class AllThresholdOrdinal:
    def __init__(self, base_classifier_factory):
        self.classifiers = {}  # threshold -> classifier

    def fit(self, X, y):
        for k in range(-1, 11):  # thresholds
            y_binary = (y >= k).astype(int)
            self.classifiers[k] = clone(base_classifier_factory()).fit(X, y_binary)

    def predict_proba(self, X):
        # Get P(delta >= k) for each threshold
        probs_ge = {k: clf.predict_proba(X)[:, 1] for k, clf in self.classifiers.items()}
        # Convert to P(delta = k)
        proba = np.zeros((len(X), 13))  # 13 classes
        for i, k in enumerate(range(-2, 11)):
            if k == -2:
                proba[:, i] = 1 - probs_ge[-1]
            elif k == 10:
                proba[:, i] = probs_ge[10]
            else:
                proba[:, i] = probs_ge[k] - probs_ge[k+1]
        return np.clip(proba, 0, 1)  # Handle monotonicity violations
```

**Option B: Proportional Odds (statsmodels)**
```python
from statsmodels.miscmodels.ordinal_model import OrderedModel

model = OrderedModel(y, X, distr='logit')  # or 'probit'
result = model.fit(method='bfgs')
proba = result.predict()  # Returns P(Y=k) for each class
```

### Forecast Residual Target

```python
# In base_trainer._prepare_features or snapshot_builder:
if "fcst_prev_max_f" in df.columns:
    df["residual"] = df["settle_f"] - df["fcst_prev_max_f"]
else:
    df["residual"] = df["delta"]  # Fallback to delta if no forecast
```

### Evaluation Metrics for Ordinal

Add to `models/evaluation/metrics.py`:
```python
def ordinal_mae(y_true, y_pred):
    """MAE respecting ordinal nature."""
    return np.abs(y_true - y_pred).mean()

def off_by_n_rate(y_true, y_pred, n=1):
    """Fraction of predictions within n classes of true."""
    return (np.abs(y_true - y_pred) <= n).mean()

def cumulative_accuracy(y_true, proba, classes):
    """Accuracy of P(Y >= k) predictions for each threshold."""
    results = {}
    for k in classes[1:]:  # Skip lowest class
        y_binary = (y_true >= k).astype(int)
        p_ge_k = proba[:, classes >= k].sum(axis=1)
        pred_binary = (p_ge_k >= 0.5).astype(int)
        results[f"acc_ge_{k}"] = (y_binary == pred_binary).mean()
    return results
```

## Completion Criteria

- [x] Ordinal model trained and evaluated
- [x] At least one alternative target (residual) evaluated
- [x] Comparison table with all models × metrics × hours
- [x] Clear recommendation for which approach to use when
- [x] Code integrates cleanly with existing evaluation framework

## Results Summary

### Model Comparison - Chicago Test Set (66 days, 528 samples)

| Model | Accuracy | MAE | Within 1 | Within 2 | Ordinal Loss |
|-------|----------|-----|----------|----------|--------------|
| **Ordinal CatBoost** | **57.4%** | **0.65** | **86.6%** | **95.5%** | **2.03** |
| Ordinal Logistic | 50.8% | 0.72 | 86.2% | 94.7% | 2.63 |
| CatBoost v3 (multinomial) | 53.2% | 1.05 | 79.7% | N/A | 10.81 |
| Logistic v3 (multinomial) | 46.4% | 0.99 | 80.3% | N/A | 9.68 |

### Key Findings

1. **Ordinal beats Multinomial**: Ordinal CatBoost improves over Multinomial CatBoost by:
   - +4.2% accuracy (57.4% vs 53.2%)
   - -38% MAE (0.65 vs 1.05)
   - +6.9% within-1 accuracy (86.6% vs 79.7%)

2. **CatBoost beats Logistic as base**: Ordinal CatBoost beats Ordinal Logistic:
   - +6.6% accuracy
   - -10% MAE

3. **Delta is better than Residual target**:
   - Delta target: 57.4% accuracy, 0.65 MAE
   - Residual target: 39.0% accuracy, 1.18 MAE
   - Reason: Delta shrinks as observations accumulate (t_base approaches settlement)

4. **By-Hour Performance**: Ordinal CatBoost wins 6/8 hours on accuracy:
   - Logistic slightly better at very early hours (10-12)
   - CatBoost dominates afternoon through night (14-23)

### Recommendation

**Use Ordinal CatBoost as the primary model** for delta prediction:
- Best overall accuracy and MAE
- Respects ordinal nature of delta classes
- Direct P(delta >= k) computation for bracket probabilities
- Strong performance across most hours

## Sign-off Log

### 2025-11-28 - Plan Created
**Status**: Draft - ready for implementation

**Context**:
Just completed the ml-framework plan with CatBoost (53.2% acc) and Logistic (46.4% acc) on Chicago data. The models treat delta as unordered categories. This plan explores ordinal regression and alternative targets that may better capture the structured nature of the prediction problem.

**Key insight**:
Delta is ordinal (-2 < -1 < 0 < ... < +10) but we're treating it as categorical. Ordinal regression should:
1. Improve MAE by penalizing "far" predictions more than "close" ones
2. Provide more natural P(delta >= k) for bracket probability computation
3. Potentially improve calibration for extreme events

**Next steps**:
1. Research ordinal regression implementations (statsmodels vs sklearn-compatible)
2. Start with All-Threshold approach (most flexible, can use CatBoost as base)
3. Implement `OrdinalDeltaTrainer`

**Blockers**: None

### 2025-11-28 - Implementation Complete
**Status**: 90% complete - core experiments done

**Completed**:
- ✅ Implemented `OrdinalDeltaTrainer` with All-Threshold approach
- ✅ Supports both CatBoost and Logistic base classifiers
- ✅ Added ordinal metrics to `models/evaluation/metrics.py`
- ✅ Trained and evaluated Ordinal CatBoost (57.4% acc, 0.65 MAE)
- ✅ Trained and evaluated Ordinal Logistic (50.8% acc, 0.72 MAE)
- ✅ Tested forecast residual target (worse than delta - 39% acc)
- ✅ Generated comprehensive comparison table

**Files Created/Modified**:
- `models/training/ordinal_trainer.py` - New ordinal regression trainer
- `models/training/__init__.py` - Added exports
- `models/evaluation/metrics.py` - Added ordinal metrics
- `models/saved/ordinal_catboost_chicago_v1.pkl` - Trained model
- `models/saved/ordinal_logistic_chicago_v1.pkl` - Trained model
- `models/saved/ordinal_catboost_residual_chicago_v1.pkl` - Residual model

**Key Result**: Ordinal CatBoost is the best model with 38% lower MAE than multinomial

**Remaining**:
- Hour-stratified models (deferred - single model performs well)
- Train on all 6 cities (optional - core approach validated)

**Blockers**: None
