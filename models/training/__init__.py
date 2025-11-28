"""
Model training pipelines for temperature Δ-models.

This package contains trainers for three model architectures:
1. Multinomial Logistic (Elastic Net + Platt calibration)
2. CatBoost (Optuna tuning + Platt calibration)
3. Ordinal Regression (All-threshold binary classifiers)

Modules:
    base_trainer: Abstract base class with common training workflow
    logistic_trainer: Model 1 - Logistic regression trainer
    catboost_trainer: Model 2 - CatBoost with Optuna trainer
    ordinal_trainer: Model 3 - Ordinal regression (CatBoost or Logistic base)
"""

from models.training.base_trainer import BaseTrainer
from models.training.logistic_trainer import LogisticDeltaTrainer, train_logistic_model
from models.training.catboost_trainer import CatBoostDeltaTrainer, train_catboost_model
from models.training.ordinal_trainer import OrdinalDeltaTrainer, train_ordinal_model

__all__ = [
    "BaseTrainer",
    "LogisticDeltaTrainer",
    "train_logistic_model",
    "CatBoostDeltaTrainer",
    "train_catboost_model",
    "OrdinalDeltaTrainer",
    "train_ordinal_model",
]
