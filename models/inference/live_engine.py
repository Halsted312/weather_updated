"""
Live Inference Engine for WebSocket Trading

Pre-loads all city models at startup and provides fast prediction
for live order book updates.
"""

import logging
import numpy as np
import pandas as pd
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo
from sqlalchemy import text

from models.training.ordinal_trainer import OrdinalDeltaTrainer
from models.features.pipeline import SnapshotContext, compute_snapshot_features
from models.features.calendar import compute_lag_features
from models.features.base import DELTA_CLASSES
from models.data.loader import load_full_inference_data
from models.inference.probability import (
    delta_probs_to_dict,
    expected_settlement,
    settlement_std,
    confidence_interval
)
from config import live_trader_config as config

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Complete prediction result with uncertainty metrics"""
    city: str
    event_date: date
    bracket_probs: Dict[str, float]  # ticker → P(win)
    t_base: int  # Current max observed temp (rounded)
    expected_settle: float  # E[settlement temp]
    settlement_std: float  # Std of settlement prediction
    ci_90_low: int  # 90% CI lower bound
    ci_90_high: int  # 90% CI upper bound
    timestamp: datetime  # When prediction was made
    snapshot_hour: int  # Which hour snapshot was used


class LiveInferenceEngine:
    """
    Pre-loads ordinal models for all cities and provides fast bracket
    probability predictions for live trading.

    Includes caching to avoid re-running inference on every tick.
    """

    def __init__(self, inference_cooldown_sec: float = 30.0):
        self.models: Dict[str, OrdinalDeltaTrainer] = {}
        self.model_metadata: Dict[str, dict] = {}
        self.inference_cooldown_sec = inference_cooldown_sec

        # Get variant config
        self.variant = config.ORDINAL_MODEL_VARIANT
        self.variant_config = config.MODEL_VARIANTS[self.variant]

        # Prediction cache: (city, event_date) → PredictionResult
        self.prediction_cache: Dict[Tuple[str, date], PredictionResult] = {}

        self._load_all_models()

    def _load_all_models(self):
        """Load all 6 city models at startup - variant aware"""
        logger.info(f"Loading ordinal models (variant={self.variant}) for all cities...")

        for city in config.CITIES:
            # Build path based on variant
            folder = f"{city}{self.variant_config['folder_suffix']}"
            model_path = config.MODEL_DIR / folder / self.variant_config['filename']

            if not model_path.exists():
                logger.warning(f"Model not found for {city} at {model_path}")
                continue

            try:
                trainer = OrdinalDeltaTrainer()
                trainer.load(model_path)
                self.models[city] = trainer

                # Store metadata
                self.model_metadata[city] = {
                    'path': str(model_path),
                    'delta_range': getattr(trainer, '_metadata', {}).get('delta_range', [-12, 12]),
                    'n_classifiers': len(trainer.classifiers) if hasattr(trainer, 'classifiers') else 24,
                }

                logger.info(f"✓ Loaded {city} model: {self.model_metadata[city]['n_classifiers']} classifiers")

            except Exception as e:
                logger.error(f"✗ Failed to load {city} model: {e}")
                raise

        if len(self.models) == 0:
            raise RuntimeError("No models loaded! Cannot proceed.")

        logger.info(f"Successfully loaded {len(self.models)}/6 city models")

    def predict(
        self,
        city: str,
        event_date: date,
        session,
        current_time: Optional[datetime] = None,
        force_refresh: bool = False
    ) -> Optional[PredictionResult]:
        """
        Get bracket win probabilities with uncertainty metrics.

        Uses caching to avoid re-running inference on every tick.

        Args:
            city: City identifier ('chicago', 'austin', etc.)
            event_date: Event settlement date
            session: Database session
            current_time: Optional override for "now" (default: use actual current time)
            force_refresh: Force recomputation (ignore cache)

        Returns:
            PredictionResult with bracket_probs, settlement_std, CI, etc.
            Or None if prediction fails validation
        """
        if city not in self.models:
            raise ValueError(f"No model loaded for {city}")

        # Check cache first
        cache_key = (city, event_date)
        if not force_refresh and cache_key in self.prediction_cache:
            cached = self.prediction_cache[cache_key]
            time_since = (datetime.now() - cached.timestamp).total_seconds()

            if time_since < self.inference_cooldown_sec:
                logger.debug(
                    f"Using cached prediction for {city} {event_date} "
                    f"({time_since:.1f}s ago)"
                )
                return cached

        # Need to compute fresh prediction
        if current_time is None:
            city_tz = ZoneInfo(config.CITY_TIMEZONES[city])
            current_time = datetime.now(city_tz)

        # Get snapshot parameters based on variant
        cutoff_time, snapshot_hour = self._get_snapshot_params(current_time)

        # Load ALL data for inference (matches training pipeline exactly)
        try:
            data = load_full_inference_data(
                city_id=city,
                event_date=event_date,
                cutoff_time=cutoff_time,
                session=session,
            )
        except ValueError as e:
            # STRICT: load_full_inference_data raises on missing data
            logger.error(f"Failed to load inference data for {city} {event_date}: {e}")
            return None

        temps_sofar = data["temps_sofar"]
        timestamps_sofar = data["timestamps_sofar"]

        if len(temps_sofar) < config.MIN_OBSERVATIONS:
            logger.warning(
                f"{city} {event_date}: Only {len(temps_sofar)} observations "
                f"(need {config.MIN_OBSERVATIONS})"
            )
            if config.REQUIRE_MIN_OBSERVATIONS:
                return None

        # Build SnapshotContext (same as training pipeline)
        ctx = SnapshotContext(
            city=city,
            event_date=event_date,
            cutoff_time=cutoff_time,
            window_start=data["window_start"],
            temps_sofar=temps_sofar,
            timestamps_sofar=timestamps_sofar,
            obs_df=data["obs_df"],
            fcst_daily=data["fcst_daily"],
            fcst_hourly_df=data["fcst_hourly_df"],
            fcst_multi=data["fcst_multi"],
            candles_df=data["candles_df"],
            city_obs_df=data["city_obs_df"],
            more_apis=data["more_apis"],
            obs_t15_mean_30d_f=data["obs_t15_mean"],
            obs_t15_std_30d_f=data["obs_t15_std"],
            settle_f=None,  # Inference mode - no settlement
        )

        # Compute features using UNIFIED pipeline (same as training!)
        try:
            features = compute_snapshot_features(ctx, include_labels=False)
        except Exception as e:
            logger.error(f"Feature building failed for {city} {event_date}: {e}")
            return None

        # Add lag features using existing compute_lag_features()
        lag_df = data.get("lag_data")
        if lag_df is not None and not lag_df.empty:
            try:
                lag_fs = compute_lag_features(lag_df, city, event_date)
                features.update(lag_fs.to_dict())

                # Compute delta_vcmax_lag1 = today's max so far - yesterday's max
                vc_max_f_lag1 = features.get("vc_max_f_lag1")
                vc_max_f_sofar = features.get("vc_max_f_sofar")
                if vc_max_f_lag1 is not None and vc_max_f_sofar is not None:
                    features["delta_vcmax_lag1"] = vc_max_f_sofar - vc_max_f_lag1

                logger.debug(f"{city}: Added {len(lag_fs)} lag features")
            except Exception as e:
                logger.warning(f"{city}: Could not compute lag features: {e}")

        # VALIDATION: STRICT feature parity check - RAISE on any mismatch
        expected_cols = set(self.models[city].numeric_cols + self.models[city].categorical_cols)
        actual_cols = set(features.keys())

        missing = expected_cols - actual_cols
        if missing:
            raise ValueError(
                f"{city}: Missing {len(missing)} features. "
                f"First 10: {sorted(missing)[:10]}. "
                "Inference pipeline not aligned with training."
            )

        # Check for high null rates (>1%) - configurable via STRICT_FEATURE_VALIDATION
        features_df_check = pd.DataFrame([features])
        present_cols = list(expected_cols & actual_cols)
        null_rates = features_df_check[present_cols].isna().mean()
        high_null_cols = null_rates[null_rates > 0.01]

        if len(high_null_cols) > 0:
            if getattr(config, 'STRICT_FEATURE_VALIDATION', True):
                raise ValueError(
                    f"{city}: Null rate >1% in {len(high_null_cols)} columns. "
                    f"Columns: {dict(high_null_cols)}. "
                    "Check data loading - all sources must be present."
                )
            else:
                logger.warning(
                    f"{city}: {len(high_null_cols)} columns have >1% null rate. "
                    f"Proceeding anyway (STRICT_FEATURE_VALIDATION=False). "
                    f"CatBoost handles NaN gracefully."
                )

        # Get t_base from features
        t_base = features.get("t_base", round(max(temps_sofar)) if temps_sofar else 0)

        # Run model prediction
        try:
            features_df = pd.DataFrame([features])
            delta_probs_array = self.models[city].predict_proba(features_df)  # Shape: (1, 25) for 25 delta classes
            delta_probs_array = delta_probs_array[0]  # Get first row

            # Convert to dict for probability.py utilities
            delta_probs_dict = delta_probs_to_dict(delta_probs_array)

            # Compute settlement statistics using probability.py
            expected_settle = expected_settlement(delta_probs_dict, t_base)
            std = settlement_std(delta_probs_dict)
            ci_low, ci_high = confidence_interval(delta_probs_dict, t_base, level=0.9)

            # Check model confidence
            if config.REQUIRE_MODEL_CONFIDENCE:
                if std > config.MAX_MODEL_STD_DEGF:
                    logger.warning(
                        f"{city} {event_date}: Model uncertainty too high "
                        f"(std={std:.2f}°F > {config.MAX_MODEL_STD_DEGF}°F)"
                    )
                    return None

                ci_span = ci_high - ci_low
                if ci_span > config.MAX_MODEL_CI_SPAN_DEGF:
                    logger.warning(
                        f"{city} {event_date}: Confidence interval too wide "
                        f"({ci_span}°F > {config.MAX_MODEL_CI_SPAN_DEGF}°F)"
                    )
                    return None

        except Exception as e:
            logger.error(f"Model prediction failed for {city} {event_date}: {e}")
            return None

        # Map delta probabilities to bracket probabilities
        bracket_probs = self._delta_to_bracket_probs(
            delta_probs_array, city, event_date, session, t_base
        )

        # Create result
        result = PredictionResult(
            city=city,
            event_date=event_date,
            bracket_probs=bracket_probs,
            t_base=t_base,
            expected_settle=expected_settle,
            settlement_std=std,
            ci_90_low=ci_low,
            ci_90_high=ci_high,
            timestamp=datetime.now(),
            snapshot_hour=snapshot_hour if snapshot_hour else cutoff_time.hour
        )

        # Cache result
        self.prediction_cache[cache_key] = result

        logger.info(
            f"{city} {event_date}: E[settle]={expected_settle:.1f}°F, "
            f"std={std:.2f}°F, 90%CI=[{ci_low},{ci_high}]°F, "
            f"{len(bracket_probs)} brackets"
        )

        return result

    def _get_snapshot_params(self, current_time: datetime) -> Tuple[datetime, Optional[int]]:
        """Get snapshot parameters - always use 5-min intervals like training.

        Training uses 5-minute market-clock snapshots (D-1 10:00 → D 23:55).
        We replicate this at inference time for feature parity.
        """
        # Floor to 5-minute intervals (same as training pipeline)
        interval_min = 5
        total_minutes = current_time.hour * 60 + current_time.minute
        floored_minutes = (total_minutes // interval_min) * interval_min

        cutoff_time = current_time.replace(
            hour=floored_minutes // 60,
            minute=floored_minutes % 60,
            second=0,
            microsecond=0
        )

        # Return None for snapshot_hour (not used with 5-min intervals)
        return cutoff_time, None

    def _delta_to_bracket_probs(
        self,
        delta_probs: np.ndarray,
        city: str,
        event_date: date,
        session,
        t_base: int
    ) -> Dict[str, float]:
        """
        Convert delta probabilities to bracket win probabilities.

        Delta classes are [-12, -11, ..., 0, ..., +11, +12] (25 classes).
        delta = settled_temp - t_base (current max observed)

        Uses DELTA_CLASSES imported from models.features.base for consistency.
        """
        # Get markets for this city/event
        markets = self._get_markets(session, city, event_date)
        if markets.empty:
            logger.warning(f"{city} {event_date}: No markets found in database")
            return {}

        # Use global DELTA_CLASSES from models.features.base (imported at top)
        # DELTA_CLASSES = [-12, -11, ..., 0, ..., +11, +12] (25 classes)

        bracket_probs = {}

        for _, market in markets.iterrows():
            ticker = market['ticker']
            strike_type = market['strike_type']
            floor_strike = market.get('floor_strike')
            cap_strike = market.get('cap_strike')

            prob = 0.0

            for i, delta in enumerate(DELTA_CLASSES):
                if i >= len(delta_probs):
                    break  # Safety check

                settled_temp = t_base + delta

                # Check if this settled temp wins this bracket
                if strike_type == 'less':
                    # Wins if temp <= cap
                    if cap_strike is not None and settled_temp <= cap_strike:
                        prob += delta_probs[i]

                elif strike_type == 'greater':
                    # Wins if temp >= floor + 1
                    if floor_strike is not None and settled_temp >= floor_strike + 1:
                        prob += delta_probs[i]

                elif strike_type == 'between':
                    # Wins if floor <= temp <= cap
                    if (floor_strike is not None and cap_strike is not None and
                        floor_strike <= settled_temp <= cap_strike):
                        prob += delta_probs[i]

            # Only include brackets with meaningful probability
            if prob >= config.MIN_BRACKET_PROB:
                bracket_probs[ticker] = float(prob)

        return bracket_probs

    def _get_markets(self, session, city: str, event_date: date) -> pd.DataFrame:
        """Query Kalshi markets for this city/event"""
        query = text("""
            SELECT ticker, strike_type, floor_strike, cap_strike
            FROM kalshi.markets
            WHERE city = :city AND event_date = :event_date
        """)

        result = session.execute(query, {'city': city, 'event_date': event_date})

        rows = []
        for row in result:
            rows.append({
                'ticker': row[0],
                'strike_type': row[1],
                'floor_strike': row[2],
                'cap_strike': row[3],
            })

        return pd.DataFrame(rows)

