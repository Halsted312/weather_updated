"""
Live inference predictor for temperature Δ-models.

This module handles loading trained models and generating predictions
from current Visual Crossing observations. It's designed for real-time
trading decisions.

Usage:
    >>> from models.inference.predictor import DeltaPredictor
    >>> predictor = DeltaPredictor('models/saved/logistic_chicago_v1.pkl')
    >>> result = predictor.predict(
    ...     city='chicago',
    ...     target_date=date.today(),
    ...     cutoff_time=datetime.now(),
    ...     session=session,
    ... )
    >>> print(result['delta_probs'])  # {-2: 0.01, -1: 0.05, 0: 0.80, 1: 0.12, 2: 0.02}
    >>> print(result['t_base'])  # 92
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from models.data.loader import load_inference_data
from models.data.snapshot_builder import build_snapshot_for_inference
from models.features.base import DELTA_CLASSES
from models.inference.probability import (
    delta_probs_to_dict,
    compute_bracket_probabilities,
)

logger = logging.getLogger(__name__)


class DeltaPredictor:
    """Load trained model and predict Δ distribution for live data.

    This class encapsulates the full inference pipeline:
    1. Load trained model from disk
    2. Query current observations from database
    3. Compute features from partial-day data
    4. Generate Δ probability distribution
    5. Convert to bracket probabilities for trading

    Attributes:
        model: Loaded trained model
        model_path: Path to model file
        metadata: Model metadata (if available)
    """

    def __init__(self, model_path: Path):
        """Load saved model.

        Args:
            model_path: Path to saved model (.pkl file)
        """
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path)
        logger.info(f"Loaded model from {self.model_path}")

        # Try to load metadata
        self.metadata = {}
        metadata_path = self.model_path.with_suffix(".json")
        if metadata_path.exists():
            import json
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)

    def predict(
        self,
        city: str,
        target_date: date,
        cutoff_time: datetime,
        session: Session,
    ) -> dict:
        """Predict Δ distribution for (city, date) at cutoff_time.

        Queries current observations, computes features, and generates
        a probability distribution over Δ classes.

        Args:
            city: City identifier (e.g., 'chicago')
            target_date: The day we're predicting settlement for
            cutoff_time: Local datetime - only use obs before this time
            session: Database session for querying observations

        Returns:
            Dict with:
                t_base: Baseline temperature (rounded partial-day max)
                delta_probs: Dict mapping Δ to probability
                predicted_delta: Most likely Δ value
                predicted_settle: t_base + predicted_delta
                confidence: Probability of predicted delta
                features: Feature dict (for debugging/logging)
                bracket_probs: P(T >= K) for common thresholds
        """
        # Load current data
        inference_data = load_inference_data(
            city_id=city,
            target_date=target_date,
            cutoff_time=cutoff_time,
            session=session,
        )

        temps_sofar = inference_data["temps_sofar"]
        timestamps_sofar = inference_data["timestamps_sofar"]

        if not temps_sofar:
            raise ValueError(f"No observations found for {city} on {target_date}")

        # Build feature row
        snapshot_hour = cutoff_time.hour
        features = build_snapshot_for_inference(
            city=city,
            day=target_date,
            snapshot_hour=snapshot_hour,
            temps_sofar=temps_sofar,
            timestamps_sofar=timestamps_sofar,
            fcst_daily=inference_data.get("fcst_daily"),
            fcst_hourly_df=inference_data.get("fcst_hourly_df"),
        )

        t_base = features["t_base"]

        # Convert to DataFrame for model
        df = pd.DataFrame([features])

        # Get predictions
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(df)[0]
        else:
            proba = None

        if hasattr(self.model, "predict"):
            predicted_delta = int(self.model.predict(df)[0])
        elif proba is not None:
            predicted_delta = DELTA_CLASSES[np.argmax(proba)]
        else:
            raise ValueError("Model must have predict() or predict_proba()")

        # Build result
        result = {
            "city": city,
            "target_date": target_date,
            "cutoff_time": cutoff_time,
            "snapshot_hour": snapshot_hour,
            "t_base": t_base,
            "predicted_delta": predicted_delta,
            "predicted_settle": t_base + predicted_delta,
            "features": features,
        }

        # Add probability information if available
        if proba is not None:
            delta_probs = delta_probs_to_dict(proba, DELTA_CLASSES)
            result["delta_probs"] = delta_probs
            result["confidence"] = delta_probs.get(predicted_delta, 0.0)

            # Compute bracket probabilities
            bracket_probs = compute_bracket_probabilities(
                delta_probs, t_base, thresholds=[80, 85, 90, 95]
            )
            result["bracket_probs"] = bracket_probs
        else:
            result["delta_probs"] = {predicted_delta: 1.0}
            result["confidence"] = 1.0
            result["bracket_probs"] = {}

        return result

    def predict_from_temps(
        self,
        city: str,
        target_date: date,
        snapshot_hour: int,
        temps_sofar: list[float],
        timestamps_sofar: list[datetime],
        fcst_daily: Optional[dict] = None,
        fcst_hourly_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Predict from provided temperature data (no DB query).

        Useful when you already have the temperature data in memory.

        Args:
            city: City identifier
            target_date: Target date
            snapshot_hour: Hour of snapshot
            temps_sofar: List of observed temperatures
            timestamps_sofar: Corresponding timestamps
            fcst_daily: Optional T-1 daily forecast
            fcst_hourly_df: Optional T-1 hourly forecast DataFrame

        Returns:
            Same structure as predict()
        """
        if not temps_sofar:
            raise ValueError("No temperatures provided")

        # Build feature row
        features = build_snapshot_for_inference(
            city=city,
            day=target_date,
            snapshot_hour=snapshot_hour,
            temps_sofar=temps_sofar,
            timestamps_sofar=timestamps_sofar,
            fcst_daily=fcst_daily,
            fcst_hourly_df=fcst_hourly_df,
        )

        t_base = features["t_base"]
        df = pd.DataFrame([features])

        # Get predictions
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(df)[0]
            delta_probs = delta_probs_to_dict(proba, DELTA_CLASSES)
            predicted_delta = DELTA_CLASSES[np.argmax(proba)]
        else:
            predicted_delta = int(self.model.predict(df)[0])
            delta_probs = {predicted_delta: 1.0}

        return {
            "city": city,
            "target_date": target_date,
            "snapshot_hour": snapshot_hour,
            "t_base": t_base,
            "predicted_delta": predicted_delta,
            "predicted_settle": t_base + predicted_delta,
            "delta_probs": delta_probs,
            "confidence": delta_probs.get(predicted_delta, 0.0),
            "bracket_probs": compute_bracket_probabilities(delta_probs, t_base),
            "features": features,
        }

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_path": str(self.model_path),
            "metadata": self.metadata,
            "delta_classes": DELTA_CLASSES,
        }


def load_predictor(model_path: Path) -> DeltaPredictor:
    """Convenience function to load a predictor.

    Args:
        model_path: Path to saved model

    Returns:
        DeltaPredictor instance
    """
    return DeltaPredictor(model_path)
