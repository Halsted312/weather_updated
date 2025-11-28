"""
ML Framework for Intraday Temperature Settlement Prediction.

This package provides tools for training, evaluating, and serving
Δ-models that predict temperature settlement deviations for Kalshi
weather markets.

Subpackages:
    features: Feature engineering modules (pure functions)
    data: Data loading and dataset construction
    training: Model training pipelines
    evaluation: Metrics and evaluation utilities
    inference: Live prediction and probability conversion

Usage:
    from models.features import compute_all_features
    from models.training import LogisticDeltaTrainer, CatBoostDeltaTrainer
    from models.inference import DeltaPredictor
"""

__version__ = "0.1.0"
