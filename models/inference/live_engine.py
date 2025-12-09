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
from models.data.snapshot_builder import build_snapshot_for_inference
from models.data.loader import load_inference_data
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
                    'delta_range': getattr(trainer, '_metadata', {}).get('delta_range', [-2, 10]),
                    'n_classifiers': len(trainer.classifiers) if hasattr(trainer, 'classifiers') else 12,
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

        # Load all inference data (observations + T-1 forecasts)
        try:
            inference_data = load_inference_data(
                city_id=city,
                target_date=event_date,
                cutoff_time=cutoff_time,
                session=session,
            )
            temps_sofar = inference_data["temps_sofar"]
            timestamps_sofar = inference_data["timestamps_sofar"]
            fcst_daily = inference_data.get("fcst_daily")
            fcst_hourly_df = inference_data.get("fcst_hourly_df")
        except Exception as e:
            logger.error(f"Failed to load inference data for {city} {event_date}: {e}")
            return None

        if len(temps_sofar) < config.MIN_OBSERVATIONS:
            logger.warning(
                f"{city} {event_date}: Only {len(temps_sofar)} observations "
                f"(need {config.MIN_OBSERVATIONS})"
            )
            if config.REQUIRE_MIN_OBSERVATIONS:
                return None

        # Get t_base (current max observed temp, rounded)
        t_base = round(max(temps_sofar)) if temps_sofar else 0

        # Build features
        try:
            features = build_snapshot_for_inference(
                city=city,
                day=event_date,
                temps_sofar=temps_sofar,
                timestamps_sofar=timestamps_sofar,
                cutoff_time=cutoff_time,  # Primary parameter
                snapshot_hour=snapshot_hour,  # Backward compat
                fcst_daily=fcst_daily,
                fcst_hourly_df=fcst_hourly_df,
            )
        except Exception as e:
            logger.error(f"Feature building failed for {city} {event_date}: {e}")
            return None

        # Run model prediction
        try:
            features_df = pd.DataFrame([features])
            delta_probs_array = self.models[city].predict_proba(features_df)  # Shape: (1, 13)
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
            snapshot_hour=snapshot_hour
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
        """Get snapshot parameters based on model variant"""

        if not self.variant_config['requires_snapping']:
            # TOD model: use exact timestamp floored to interval
            interval_min = config.TOD_SNAPSHOT_INTERVAL_MIN
            total_minutes = current_time.hour * 60 + current_time.minute
            floored_minutes = (total_minutes // interval_min) * interval_min

            cutoff_time = current_time.replace(
                hour=floored_minutes // 60,
                minute=floored_minutes % 60,
                second=0,
                microsecond=0
            )
            return cutoff_time, None

        else:
            # Baseline/hourly: snap to nearest training hour
            snapshot_hour = min(
                self.variant_config['snapshot_hours'],
                key=lambda x: abs(x - current_time.hour)
            )
            cutoff_time = current_time.replace(hour=snapshot_hour, minute=0, second=0, microsecond=0)
            return cutoff_time, snapshot_hour

    def _get_observations(
        self,
        session,
        city: str,
        event_date: date,
        cutoff: datetime
    ) -> Tuple[List[float], List[datetime]]:
        """
        Query observations from wx.vc_minute_weather up to cutoff time.

        Returns:
            (temps_sofar, timestamps_sofar)
        """
        city_code = config.CITY_CODES[city]

        query = text("""
            SELECT vm.datetime_local, vm.temp_f
            FROM wx.vc_minute_weather vm
            JOIN wx.vc_location vl ON vm.vc_location_id = vl.id
            WHERE vl.city_code = :city_code
              AND DATE(vm.datetime_local) = :target_date
              AND vm.temp_f IS NOT NULL
              AND vm.datetime_local <= :cutoff
            ORDER BY vm.datetime_local
        """)

        result = session.execute(
            query,
            {
                'city_code': city_code,
                'target_date': event_date,
                'cutoff': cutoff
            }
        )

        temps = []
        timestamps = []
        for row in result:
            timestamps.append(row[0])
            temps.append(row[1])

        return temps, timestamps

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

        Delta classes are [-2, -1, 0, 1, ..., 10] for most cities.
        delta = settled_temp - t_base (current max observed)
        """
        # Get markets for this city/event
        markets = self._get_markets(session, city, event_date)
        if markets.empty:
            logger.warning(f"{city} {event_date}: No markets found in database")
            return {}

        # Delta classes (global range)
        DELTA_CLASSES = list(range(-2, 11))  # [-2, -1, 0, 1, ..., 10]

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
            FROM kalshi.market
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

    def _compute_std(self, delta_probs: np.ndarray) -> float:
        """Compute standard deviation of delta distribution"""
        DELTA_CLASSES = np.array(range(-2, 11))
        mean = np.sum(DELTA_CLASSES * delta_probs)
        variance = np.sum(((DELTA_CLASSES - mean) ** 2) * delta_probs)
        return np.sqrt(variance)
