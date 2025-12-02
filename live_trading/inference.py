"""
Inference wrapper integrating ordinal model and edge classifier.

Combines:
- models/inference/live_engine.py (ordinal temperature predictions)
- models/edge/implied_temp.py (market-implied temperature)
- models/edge/detector.py (edge detection)
- models/edge/classifier.py (ML filtering)

Into a single interface for trading decisions.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from sqlalchemy.orm import Session

from models.inference.live_engine import LiveInferenceEngine, PredictionResult
from models.edge.implied_temp import compute_market_implied_temp, MarketImpliedResult
from models.edge.detector import detect_edge, EdgeResult, EdgeSignal, select_bracket_for_signal
from models.edge.classifier import EdgeClassifier
import numpy as np

logger = logging.getLogger(__name__)


def _build_bracket_candles_from_snapshot(brackets: list[dict]) -> dict[str, pd.DataFrame]:
    """
    Convert simple bracket dict list to format expected by compute_market_implied_temp.

    Args:
        brackets: List of dicts with ticker, yes_bid, floor_strike, cap_strike

    Returns:
        Dict mapping bracket label to single-row DataFrame with yes_bid_close
    """
    bracket_candles = {}

    for bracket in brackets:
        # Get bracket label from strikes
        floor = bracket.get('floor_strike')
        cap = bracket.get('cap_strike')

        if floor is None and cap is None:
            continue

        # Create label
        if floor is None:
            label = f"<{int(cap)}"
        elif cap is None:
            label = f">{int(floor)}"
        else:
            label = f"{int(floor)}-{int(cap)}"

        # Get price (use yes_bid as probability proxy)
        yes_bid = bracket.get('yes_bid', 50)

        # Create single-row DataFrame
        bracket_candles[label] = pd.DataFrame([{
            'bucket_start': datetime.now(),
            'yes_bid_close': yes_bid
        }])

    return bracket_candles


@dataclass
class EdgeDecision:
    """Complete edge analysis and trading decision."""

    # Temperature analysis
    forecast_implied_temp: float
    market_implied_temp: float
    edge_degf: float
    signal: str  # 'buy_high', 'buy_low', 'no_trade'

    # ML classifier
    edge_classifier_prob: float
    should_trade: bool

    # Recommendation
    recommended_bracket: Optional[str]
    recommended_side: Optional[str]  # 'yes' or 'no'
    recommended_action: Optional[str]  # 'buy' or 'sell'

    # Explanation
    reason: str

    # Uncertainty metrics
    forecast_uncertainty: float
    market_uncertainty: float
    combined_uncertainty: float

    # Metadata
    prediction_result: Optional[PredictionResult]  # Full ordinal prediction
    edge_result: Optional[EdgeResult]  # Full edge detection result


class InferenceWrapper:
    """
    Wrapper combining ordinal prediction and edge detection for live trading.

    Integrates:
    1. LiveInferenceEngine for temperature predictions
    2. Market-implied temperature calculation
    3. Edge detection
    4. Edge classifier ML filtering
    """

    def __init__(self):
        """Initialize inference wrapper."""
        # Load ordinal models via existing live engine
        self.live_engine = LiveInferenceEngine()
        logger.info(f"Loaded ordinal models for cities: {list(self.live_engine.models.keys())}")

        # Edge classifiers loaded lazily per city
        self.edge_classifiers: Dict[str, EdgeClassifier] = {}

    def _get_model_path(self, city: str, model_type: str) -> Path:
        """
        Central model path resolver (shim for future reorganization).

        Args:
            city: City identifier
            model_type: 'ordinal' or 'edge_classifier'

        Returns:
            Path to model file
        """
        base = Path("models/saved")

        if model_type == "ordinal":
            return base / city / "ordinal_catboost_optuna.pkl"
        elif model_type == "edge_classifier":
            return base / city / "edge_classifier.pkl"
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _get_edge_classifier(self, city: str) -> EdgeClassifier:
        """
        Lazy-load edge classifier for city.

        Args:
            city: City identifier

        Returns:
            Loaded EdgeClassifier instance
        """
        if city not in self.edge_classifiers:
            model_path = self._get_model_path(city, "edge_classifier")

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Edge classifier not found for {city} at {model_path}. "
                    f"Train it with: python scripts/train_edge_classifier.py --city {city}"
                )

            classifier = EdgeClassifier()
            classifier.load(model_path)
            self.edge_classifiers[city] = classifier

            logger.info(f"Loaded edge classifier for {city} from {model_path}")

        return self.edge_classifiers[city]

    def _build_edge_features(
        self,
        edge: float,
        prediction: PredictionResult,
        market_snapshot: Dict[str, Any],
        current_time: datetime,
    ) -> pd.DataFrame:
        """
        Build feature DataFrame for edge classifier.

        Args:
            edge: Edge magnitude (forecast - market implied)
            prediction: Ordinal prediction result
            market_snapshot: Current market state
            current_time: Evaluation timestamp

        Returns:
            Single-row DataFrame with edge features
        """
        # Extract market spread
        best_bid = market_snapshot.get('best_bid', 0)
        best_ask = market_snapshot.get('best_ask', 100)
        spread = best_ask - best_bid

        # Compute temporal features
        snapshot_hour = current_time.hour
        # hours_to_event_close approximated (markets close around midnight local)
        hours_to_close = 24 - snapshot_hour  # Simplified

        # Extract from original features if available in market_snapshot
        obs_fcst_max_gap = market_snapshot.get('obs_fcst_max_gap', 0.0)
        fcst_remaining_potential = market_snapshot.get('fcst_remaining_potential', 0.0)
        temp_volatility_30min = market_snapshot.get('temp_volatility_30min', 0.0)
        minutes_since_market_open = market_snapshot.get('minutes_since_market_open', 0)

        # Determine signal (needed by prepare_features to derive is_buy_low)
        signal_str = "buy_low" if edge < 0 else "buy_high" if edge > 0 else "no_trade"

        # Build feature dict
        # Note: 'signal' not a final feature, but prepare_features uses it to derive 'is_buy_low'
        features = {
            'edge': edge,
            'abs_edge': abs(edge),
            'signal': signal_str,  # Used by prepare_features, not final feature
            'confidence': abs(edge) / 1.5 if edge != 0 else 0,
            'is_buy_low': 1 if edge < 0 else 0,
            'forecast_uncertainty': prediction.settlement_std,
            'snapshot_hour': snapshot_hour,
            'hours_to_event_close': hours_to_close,
            'minutes_since_market_open': minutes_since_market_open,
            'obs_fcst_max_gap': obs_fcst_max_gap,
            'fcst_remaining_potential': fcst_remaining_potential,
            'temp_volatility_30min': temp_volatility_30min,
            'market_bid_ask_spread': spread,
        }

        return pd.DataFrame([features])

    def evaluate_edge(
        self,
        city: str,
        event_date: date,
        market_snapshot: Dict[str, Any],
        session: Session,
        edge_threshold_degf: float = 1.5,
        confidence_threshold: float = 0.5,
        current_time: Optional[datetime] = None,
    ) -> EdgeDecision:
        """
        Run complete edge analysis pipeline.

        Steps:
        1. Get ordinal temperature prediction
        2. Compute market-implied temperature from bracket prices
        3. Detect edge (forecast vs market)
        4. Run edge classifier for ML filtering
        5. Generate trading decision

        Args:
            city: City identifier
            event_date: Event settlement date
            market_snapshot: Current market state with bracket prices
            session: Database session for data loading
            edge_threshold_degf: Minimum edge for signal (degrees F)
            confidence_threshold: ML probability threshold for trading
            current_time: Optional timestamp override

        Returns:
            EdgeDecision with complete analysis and trading recommendation
        """
        if current_time is None:
            current_time = datetime.now()

        # 1. Get ordinal prediction
        prediction = self.live_engine.predict(
            city=city,
            event_date=event_date,
            session=session,
            current_time=current_time
        )

        if prediction is None:
            return self._no_trade_decision(
                reason="No ordinal prediction available (failed validation or missing data)"
            )

        # 2. Compute market-implied temperature
        bracket_data = market_snapshot.get('brackets', [])
        if not bracket_data:
            return self._no_trade_decision(
                reason="No market data available",
                prediction_result=prediction
            )

        try:
            # Convert bracket list to DataFrame dict format
            bracket_candles = _build_bracket_candles_from_snapshot(bracket_data)

            market_result = compute_market_implied_temp(
                bracket_candles=bracket_candles,
                tail_extension=5.0
            )
        except Exception as e:
            logger.error(f"Failed to compute market-implied temp: {e}")
            return self._no_trade_decision(
                reason=f"Market-implied calculation failed: {e}",
                prediction_result=prediction
            )

        # 3. Detect edge
        forecast_implied = prediction.expected_settle
        market_implied = market_result.implied_temp

        edge_result = detect_edge(
            forecast_implied=forecast_implied,
            market_implied=market_implied,
            forecast_uncertainty=prediction.settlement_std,
            market_uncertainty=market_result.uncertainty,
            threshold=edge_threshold_degf,
            min_confidence=0.0  # We'll use ML classifier for confidence
        )

        edge_degf = edge_result.edge
        signal = edge_result.signal

        # Short-circuit if no edge detected
        if signal == EdgeSignal.NO_TRADE:
            return EdgeDecision(
                forecast_implied_temp=forecast_implied,
                market_implied_temp=market_implied,
                edge_degf=edge_degf,
                signal=signal.value,
                edge_classifier_prob=0.0,
                should_trade=False,
                recommended_bracket=None,
                recommended_side=None,
                recommended_action=None,
                reason=f"Edge too small ({edge_degf:.2f}°F < {edge_threshold_degf}°F threshold)",
                forecast_uncertainty=prediction.settlement_std,
                market_uncertainty=market_result.uncertainty,
                combined_uncertainty=edge_result.combined_uncertainty,
                prediction_result=prediction,
                edge_result=edge_result,
            )

        # 4. Run edge classifier
        try:
            classifier = self._get_edge_classifier(city)
            features_df = self._build_edge_features(
                edge=edge_degf,
                prediction=prediction,
                market_snapshot=market_snapshot,
                current_time=current_time
            )

            edge_probs = classifier.predict(features_df)
            edge_prob = float(edge_probs[0])

        except Exception as e:
            logger.error(f"Edge classifier failed for {city}: {e}", exc_info=True)
            return self._no_trade_decision(
                reason=f"Edge classifier error: {e}",
                forecast_implied_temp=forecast_implied,
                market_implied_temp=market_implied,
                edge_degf=edge_degf,
                signal=signal.value,
                prediction_result=prediction,
                edge_result=edge_result
            )

        # 5. Trading decision
        should_trade = edge_prob >= confidence_threshold

        # Select bracket if trading
        recommended_bracket = None
        recommended_side = None
        recommended_action = "buy"  # Default to buying

        if should_trade:
            # Get available brackets from market snapshot
            available_brackets = [
                (b['floor_strike'], b['cap_strike'])
                for b in bracket_data
                if b.get('floor_strike') and b.get('cap_strike')
            ]

            bracket = select_bracket_for_signal(
                signal=signal,
                forecast_temp=forecast_implied,
                available_brackets=available_brackets
            )

            if bracket:
                # Find ticker for this bracket
                for b in bracket_data:
                    if (b.get('floor_strike'), b.get('cap_strike')) == bracket:
                        recommended_bracket = b.get('ticker')
                        # For BUY_HIGH/BUY_LOW we're buying YES on the selected bracket
                        recommended_side = "yes"
                        break

        reason_parts = [
            f"Edge {edge_degf:+.2f}°F ({signal.value})",
            f"classifier prob {edge_prob:.3f}",
        ]

        if should_trade:
            reason_parts.append(f"TRADE: {recommended_bracket or 'TBD'}")
        else:
            reason_parts.append(f"SKIP (prob {edge_prob:.3f} < threshold {confidence_threshold:.3f})")

        return EdgeDecision(
            forecast_implied_temp=forecast_implied,
            market_implied_temp=market_implied,
            edge_degf=edge_degf,
            signal=signal.value,
            edge_classifier_prob=edge_prob,
            should_trade=should_trade,
            recommended_bracket=recommended_bracket,
            recommended_side=recommended_side,
            recommended_action=recommended_action,
            reason=" | ".join(reason_parts),
            forecast_uncertainty=prediction.settlement_std,
            market_uncertainty=market_result.uncertainty,
            combined_uncertainty=edge_result.combined_uncertainty,
            prediction_result=prediction,
            edge_result=edge_result,
        )

    def _no_trade_decision(
        self,
        reason: str,
        forecast_implied_temp: float = 0.0,
        market_implied_temp: float = 0.0,
        edge_degf: float = 0.0,
        signal: str = "no_trade",
        prediction_result: Optional[PredictionResult] = None,
        edge_result: Optional[EdgeResult] = None,
    ) -> EdgeDecision:
        """
        Helper to create a no-trade decision with given reason.

        Args:
            reason: Explanation for no-trade
            (other args): Optional values to include

        Returns:
            EdgeDecision with should_trade=False
        """
        return EdgeDecision(
            forecast_implied_temp=forecast_implied_temp,
            market_implied_temp=market_implied_temp,
            edge_degf=edge_degf,
            signal=signal,
            edge_classifier_prob=0.0,
            should_trade=False,
            recommended_bracket=None,
            recommended_side=None,
            recommended_action=None,
            reason=reason,
            forecast_uncertainty=prediction_result.settlement_std if prediction_result else 0.0,
            market_uncertainty=0.0,
            combined_uncertainty=0.0,
            prediction_result=prediction_result,
            edge_result=edge_result,
        )
