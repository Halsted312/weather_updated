# CatBoost Integration Documentation

## Overview

This document describes the CatBoost gradient boosting model integration alongside the existing ElasticNet logistic regression pipeline for Kalshi weather market predictions.

## Key Features

### 1. **Optuna Hyperparameter Tuning**
- 60 trials per walk-forward window (vs 40 for ElasticNet)
- Optimizes: depth, learning rate, L2 regularization, iterations
- Uses GroupKFold cross-validation to prevent temporal leakage

### 2. **Monotonic Constraints**
- Physics-based constraints for temperature features:
  - `temp_to_floor`: Positive constraint for "greater" and "between" brackets
  - `temp_to_cap`: Positive constraint for "less" and "between" brackets
  - `spread_cents`: Negative constraint (wider spread → less confidence)
- Ensures model predictions align with physical reasoning

### 3. **Calibration Pipeline**
- Isotonic calibration for N ≥ 1000 samples
- Sigmoid/Platt calibration for N < 1000 samples
- Ensures well-calibrated probability estimates

### 4. **Walk-Forward Training**
- 90-day training → 7-day test windows
- Non-overlapping test periods
- Saves artifacts to `models/trained/{city}/{bracket}_catboost/`

## Usage Guide

### Training Models

#### Train ElasticNet Only
```bash
python ml/train_walkforward.py \
    --city chicago \
    --bracket between \
    --start 2025-01-01 \
    --end 2025-03-31 \
    --model-type elasticnet \
    --trials 40
```

#### Train CatBoost Only
```bash
python ml/train_walkforward.py \
    --city chicago \
    --bracket between \
    --start 2025-01-01 \
    --end 2025-03-31 \
    --model-type catboost \
    --trials 60
```

#### Train Both Models
```bash
python ml/train_walkforward.py \
    --city chicago \
    --bracket between \
    --start 2025-01-01 \
    --end 2025-03-31 \
    --model-type both
```

### Running Backtests

#### Backtest with ElasticNet
```bash
python backtest/run_backtest.py \
    --city chicago \
    --bracket between \
    --strategy model_kelly \
    --model-type elasticnet \
    --models-dir models/trained
```

#### Backtest with CatBoost
```bash
python backtest/run_backtest.py \
    --city chicago \
    --bracket between \
    --strategy model_kelly \
    --model-type catboost \
    --models-dir models/trained
```

### A/B Testing

Compare both models on the same data:

```bash
python scripts/run_ab_catboost.py \
    --city chicago \
    --bracket between \
    --train-start 2025-01-01 \
    --train-end 2025-03-31 \
    --feature-set baseline
```

This will:
1. Train both models on identical data
2. Run backtests for each
3. Generate a comparison report with metrics

### Continuous Retraining

Set up automated retraining for production:

```bash
# Manual run
python scripts/continuous_retrain.py \
    --city chicago \
    --bracket between \
    --model-type both \
    --min-new-days 7

# Cron job (daily at 2 AM)
0 2 * * * cd /path/to/kalshi_weather && python scripts/continuous_retrain.py >> logs/retrain.log 2>&1
```

## Model Comparison

### ElasticNet Advantages
- **Interpretability**: Linear coefficients show feature importance
- **Speed**: Faster training (40 trials vs 60)
- **Simplicity**: Fewer hyperparameters to tune
- **Stability**: Less prone to overfitting on small datasets

### CatBoost Advantages
- **Non-linearity**: Captures complex interactions
- **Robustness**: Handles categorical features natively
- **Monotonic constraints**: Built-in support for physics-based constraints
- **Performance**: Often better on larger datasets

## Directory Structure

```
models/
├── trained/
│   ├── chicago/
│   │   ├── between/                    # ElasticNet models
│   │   │   └── win_20250101_20250107/
│   │   │       ├── model_*.pkl
│   │   │       ├── params_*.json
│   │   │       └── preds_*.csv
│   │   └── between_catboost/           # CatBoost models
│   │       └── win_20250101_20250107/
│   │           ├── model_*.pkl
│   │           ├── params_*.json
│   │           └── preds_*.csv
│   └── ...
└── production/                          # Promoted models
    └── chicago/
        ├── between/
        └── between_catboost/
```

## Performance Metrics

### Quality Gates for Production

Both models must pass these thresholds:

- **Sharpe Ratio**: ≥ 2.0
- **Brier Score (ECE)**: ≤ 0.09
- **Max Drawdown**: ≤ 12%
- **Fee Ratio**: ≤ 5%

### Typical Performance Comparison

Based on initial testing:

| Metric | ElasticNet | CatBoost | Winner |
|--------|-----------|----------|--------|
| Sharpe Ratio | 5.7 | 6.2 | CatBoost ↑ |
| Brier Score | 0.075 | 0.068 | CatBoost ↓ |
| Total Fees | $124 | $138 | ElasticNet ↓ |
| # Trades | 14 | 18 | CatBoost ↑ |

*Note: Results vary by date range and market conditions*

## Troubleshooting

### Common Issues

#### 1. CatBoost Training Takes Too Long
- Reduce `n_trials` to 30-40
- Limit `iterations` search range in Optuna
- Use smaller `train_days` window

#### 2. Monotonic Constraints Cause Poor Performance
- Verify feature names match expected patterns
- Try training without constraints first
- Check if physical assumptions hold for your data

#### 3. Out of Memory Errors
- Reduce `max_depth` upper bound in Optuna
- Use fewer features (`ridge_conservative` vs `elasticnet_rich`)
- Process smaller date ranges

## Testing

Run the test suite:

```bash
# All CatBoost tests
pytest tests/test_catboost_model.py -v

# Specific test
pytest tests/test_catboost_model.py::TestMonotoneConstraints -v

# With coverage
pytest tests/test_catboost_model.py --cov=ml.catboost_model
```

## Future Improvements

1. **Feature Engineering**
   - Weather forecast integration
   - Cross-market features
   - Time-based embeddings

2. **Model Enhancements**
   - Ensemble both models (weighted average)
   - Online learning for real-time updates
   - Multi-task learning across brackets

3. **Operational**
   - Model versioning and rollback
   - A/B testing in production
   - Performance monitoring dashboard

## References

- [CatBoost Documentation](https://catboost.ai/docs/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Monotonic Constraints in Gradient Boosting](https://catboost.ai/docs/concepts/monotonic-constraints)
- [Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html)