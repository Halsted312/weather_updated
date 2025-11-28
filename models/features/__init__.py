"""
Feature engineering modules for temperature Δ-models.

This package contains pure functions that compute features from
partial-day Visual Crossing observations. All functions are designed
to work identically for training (historical data) and inference (live data).

Modules:
    base: FeatureSet dataclass and composition utilities
    partial_day: Base statistics from VC temps up to snapshot time
    shape: Plateau, spike, and slope features
    rules: Rule-based meta-features (wraps analysis/temperature/rules.py)
    forecast: T-1 forecast and forecast-vs-actual error features
    calendar: Time encoding and lag features
    quality: Data quality indicators

Usage:
    from models.features import compute_all_features
    features = compute_all_features(temps_sofar, timestamps_sofar, ...)
"""

from models.features.base import FeatureSet, compose_features, ALL_FEATURE_GROUPS

__all__ = [
    "FeatureSet",
    "compose_features",
    "ALL_FEATURE_GROUPS",
]
