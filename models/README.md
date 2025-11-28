# ML Framework for Temperature Settlement Prediction

This directory contains the machine learning framework for predicting Kalshi
temperature settlement using intraday Visual Crossing observations.

## Overview

The framework predicts **Δ = T_settle - T_base**, where:
- `T_settle` = Final NWS/Kalshi daily high (integer °F)
- `T_base` = Rounded maximum observed temperature so far at snapshot time τ

By predicting the deviation from a partial-day baseline, the models can
generate probability distributions at any time during the day.

## Directory Structure

```
models/
├── features/          # Feature engineering (pure functions)
│   ├── base.py        # FeatureSet dataclass, composition utilities
│   ├── partial_day.py # Base stats from VC obs up to τ
│   ├── shape.py       # Plateau, spike, slope features
│   ├── rules.py       # Wraps analysis/temperature/rules.py
│   ├── forecast.py    # T-1 forecast + forecast error features
│   ├── calendar.py    # Snapshot hour, doy, lags
│   └── quality.py     # Missing fraction, gaps, edge flags
│
├── data/              # Data loading and dataset construction
│   ├── loader.py      # DB queries for training and inference
│   ├── snapshot_builder.py  # Build snapshot-level feature table
│   └── splits.py      # Train/test splits, TimeSeriesSplit CV
│
├── training/          # Model training pipelines
│   ├── base_trainer.py     # Abstract base class
│   ├── logistic_trainer.py # Model 1: Logistic + Elastic Net + Platt
│   └── catboost_trainer.py # Model 2: CatBoost + Optuna + Platt
│
├── evaluation/        # Metrics and evaluation
│   ├── metrics.py     # Accuracy, MAE, Brier score, calibration
│   ├── evaluator.py   # Run full evaluation suite
│   └── reports.py     # Generate comparison reports
│
├── inference/         # Live inference
│   ├── predictor.py   # Load model, compute features, predict
│   └── probability.py # Convert Δ probs to bracket probs
│
├── saved/             # Trained model artifacts (.pkl files)
└── reports/           # Evaluation outputs (CSV, MD files)
```

## Models

### Model 1: Multinomial Logistic (Elastic Net + Platt)
- Regularized logistic regression for Δ ∈ {-2, -1, 0, +1, +2}
- Interpretable coefficients
- Platt scaling for probability calibration

### Model 2: CatBoost + Optuna + Platt
- Gradient boosting with native categorical handling
- Bayesian hyperparameter optimization via Optuna
- Platt scaling for probability calibration

## Key Design Principles

1. **Pure feature functions**: Take data in, return features out. No DB coupling.
2. **No lookahead**: Features at snapshot τ use ONLY data with datetime_local < τ.
3. **Registry pattern**: ALL_FEATURE_GROUPS dict for easy composition.
4. **Chicago first**: Initial development focused on KMDW station.

## Quick Start

```python
# Training
from models.data.snapshot_builder import build_snapshot_dataset
from models.training.logistic_trainer import LogisticDeltaTrainer

df = build_snapshot_dataset(cities=['chicago'], start_date=..., end_date=...)
trainer = LogisticDeltaTrainer()
model = trainer.train(X_train, y_train)
trainer.save(model, 'models/saved/logistic_chicago_v1.pkl')

# Inference
from models.inference.predictor import DeltaPredictor

predictor = DeltaPredictor('models/saved/logistic_chicago_v1.pkl')
result = predictor.predict(city='chicago', target_date=today, cutoff_time=now)
print(result['delta_probs'])  # {-2: 0.01, -1: 0.05, 0: 0.80, 1: 0.12, 2: 0.02}
```

## Output Files

After training and evaluation, find results in:
- `models/saved/` - Trained model .pkl files with metadata
- `models/reports/model_comparison.csv` - Side-by-side model metrics
- `models/reports/model_comparison.md` - Human-readable summary
- `models/reports/calibration_*.csv` - Reliability curve data
