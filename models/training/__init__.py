"""
Model training pipelines for temperature Δ-models.

This package contains trainers for two model architectures:
1. CatBoost (Optuna tuning + Platt calibration)
2. Ordinal Regression (All-threshold binary classifiers)

Modules:
    base_trainer: Abstract base class with common training workflow
    catboost_trainer: Model 2 - CatBoost with Optuna trainer
    ordinal_trainer: Model 3 - Ordinal regression (CatBoost or Logistic base)

Note: LogisticDeltaTrainer was archived to legacy/models/training/
"""

from models.training.base_trainer import BaseTrainer
from models.training.catboost_trainer import CatBoostDeltaTrainer, train_catboost_model
from models.training.ordinal_trainer import OrdinalDeltaTrainer, train_ordinal_model

__all__ = [
    "BaseTrainer",
    "CatBoostDeltaTrainer",
    "train_catboost_model",
    "OrdinalDeltaTrainer",
    "train_ordinal_model",
]
