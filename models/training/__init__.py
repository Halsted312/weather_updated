"""
Model training pipelines for temperature Δ-models.

This package contains trainers for the two main model types:
1. Multinomial Logistic (Elastic Net + Platt calibration)
2. CatBoost (Optuna tuning + Platt calibration)

Modules:
    base_trainer: Abstract base class with common training workflow
    logistic_trainer: Model 1 - Logistic regression trainer
    catboost_trainer: Model 2 - CatBoost with Optuna trainer
"""
