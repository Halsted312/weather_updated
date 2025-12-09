"""
City scanner engine for multi-city edge detection.

Scans all enabled cities, evaluates edges, and ranks trading opportunities
by expected value. Supports both ML classifier-based and threshold-only modes.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

from sqlalchemy.orm import Session

from live_trading.config import TradingConfig
from live_trading.inference import InferenceWrapper, EdgeDecision
from live_trading.market_data import get_market_snapshot
from live_trading.websocket.handler import WebSocketHandler
from live_trading.websocket.order_book import OrderBookManager
from live_trading.websocket.market_state import MarketStateTracker
from src.trading.fees import find_best_trade

logger = logging.getLogger(__name__)


@dataclass
class CityOpportunity:
    """A trading opportunity for a city/event."""

    # Identity
    city: str
    event_date: date
    ticker: str

    # Edge analysis
    edge_degf: float
    signal: str  # 'buy_high', 'buy_low', 'no_trade'

    # ML confidence
    edge_classifier_prob: float
    inference_mode: str  # 'classifier' or 'threshold'

    # Market data
    yes_bid: int
    yes_ask: int
    spread: int

    # Temperature
    forecast_temp: float
    market_implied_temp: float

    # Trade recommendation
    recommended_side: str  # 'yes' or 'no'
    recommended_action: str  # 'buy' or 'sell'
    recommended_price: int
    ev_per_contract: float
    role: str  # 'maker' or 'taker'

    # Full decision object
    decision: EdgeDecision

    # Metadata
    reason: str
    timestamp: datetime


class CityScannerEngine:
    """
    Multi-city scanner with adaptive inference.

    Automatically uses:
    - ML classifier (if available and config allows)
    - Threshold-only (fallback or by config)

    Ranks opportunities by expected value.
    """

    def __init__(
        self,
        config: TradingConfig,
        inference: InferenceWrapper,
        ws_handler: Optional[WebSocketHandler] = None,
        order_book_mgr: Optional[OrderBookManager] = None,
        market_state_tracker: Optional[MarketStateTracker] = None,
    ):
        """
        Initialize scanner engine.

        Args:
            config: Trading configuration
            inference: InferenceWrapper with ordinal models + edge detection
            ws_handler: WebSocket handler for real-time data (optional)
            order_book_mgr: Order book manager for market data (optional)
            market_state_tracker: Market state tracker for metadata (optional)
        """
        self.config = config
        self.inference = inference
        self.ws_handler = ws_handler
        self.order_book_mgr = order_book_mgr
        self.market_state_tracker = market_state_tracker

        # Auto-detect available classifiers
        self.cities_with_classifiers = self._detect_available_classifiers()
        logger.info(f"Scanner initialized. Classifiers available: {self.cities_with_classifiers}")
        logger.info(f"Inference mode: {config.inference_mode}")

    def _detect_available_classifiers(self) -> List[str]:
        """
        Scan models/saved/ for cities with trained edge classifiers.

        Returns:
            List of city names with available classifiers
        """
        available = []
        base = Path("models/saved")

        for city in self.config.enabled_cities:
            classifier_path = base / city / "edge_classifier.pkl"
            if classifier_path.exists():
                available.append(city)
                logger.info(f"✓ Classifier found: {city}")
            else:
                logger.debug(f"○ No classifier: {city} (will use threshold)")

        return available

    def should_use_classifier(self, city: str) -> bool:
        """
        Determine if we should use ML classifier for this city.

        Logic:
        - threshold_only: Never use classifier
        - classifier_only: Only if classifier available (skip city otherwise)
        - adaptive: Use classifier if available, threshold otherwise

        Args:
            city: City identifier

        Returns:
            True if should use ML classifier, False for threshold-only
        """
        if self.config.inference_mode == "threshold_only":
            return False

        has_classifier = city in self.cities_with_classifiers

        if self.config.inference_mode == "classifier_only":
            return has_classifier

        # adaptive mode (default)
        return has_classifier

    def _get_target_dates(self, custom_dates: Optional[List[date]] = None) -> List[date]:
        """
        Get list of target event dates to scan.

        Args:
            custom_dates: Optional custom date list

        Returns:
            List of dates to scan (today + tomorrow by default)
        """
        if custom_dates:
            return custom_dates

        # Default: scan today and tomorrow
        today = date.today()
        tomorrow = today + timedelta(days=1)
        return [today, tomorrow]

    def _get_market_snapshot(self, city: str, event_date: date) -> Optional[Dict[str, Any]]:
        """
        Get current market snapshot for city/event.

        Uses order book manager if available, otherwise returns None.

        Args:
            city: City identifier
            event_date: Event settlement date

        Returns:
            Market snapshot dict with brackets, or None if unavailable
        """
        if not self.order_book_mgr or not self.market_state_tracker:
            logger.warning(
                f"No order book manager or market state tracker, "
                f"cannot get market snapshot for {city} {event_date}"
            )
            return None

        # Use shared utility to get market snapshot
        return get_market_snapshot(
            order_book_mgr=self.order_book_mgr,
            market_state_tracker=self.market_state_tracker,
            city=city,
            event_date=event_date,
        )

    def _build_opportunity(
        self,
        city: str,
        event_date: date,
        decision: EdgeDecision,
        market_snapshot: Dict[str, Any],
        current_time: datetime,
    ) -> Optional[CityOpportunity]:
        """
        Build CityOpportunity from edge decision.

        Args:
            city: City identifier
            event_date: Event settlement date
            decision: EdgeDecision from inference
            market_snapshot: Market data
            current_time: Evaluation timestamp

        Returns:
            CityOpportunity if trade is viable, None otherwise
        """
        if not decision.should_trade:
            return None

        # Get bracket data for recommended bracket
        ticker = decision.recommended_bracket
        if not ticker:
            logger.warning(f"No recommended bracket in decision for {city} {event_date}")
            return None

        # Find bracket in market snapshot
        bracket_data = None
        for bracket in market_snapshot.get('brackets', []):
            if bracket.get('ticker') == ticker:
                bracket_data = bracket
                break

        if not bracket_data:
            logger.warning(f"Bracket {ticker} not found in market snapshot")
            return None

        yes_bid = bracket_data.get('yes_bid', 0)
        yes_ask = bracket_data.get('yes_ask', 100)

        # Use find_best_trade to get optimal trade
        side, action, price, ev, role = find_best_trade(
            model_prob=decision.edge_classifier_prob,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            min_ev_cents=self.config.min_ev_per_contract_cents,
            maker_fill_prob=0.6  # Configurable
        )

        if ev <= 0:
            logger.debug(f"Skip {city} {event_date}: EV={ev:.2f}¢ <= 0")
            return None

        # Determine inference mode
        inference_mode = "classifier" if self.should_use_classifier(city) else "threshold"

        return CityOpportunity(
            city=city,
            event_date=event_date,
            ticker=ticker,
            edge_degf=decision.edge_degf,
            signal=decision.signal,
            edge_classifier_prob=decision.edge_classifier_prob,
            inference_mode=inference_mode,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            spread=yes_ask - yes_bid,
            forecast_temp=decision.forecast_implied_temp,
            market_implied_temp=decision.market_implied_temp,
            recommended_side=side,
            recommended_action=action,
            recommended_price=price,
            ev_per_contract=ev,
            role=role,
            decision=decision,
            reason=decision.reason,
            timestamp=current_time,
        )

    async def scan_all_cities(
        self,
        session: Session,
        target_dates: Optional[List[date]] = None,
        current_time: Optional[datetime] = None,
    ) -> List[CityOpportunity]:
        """
        Scan all enabled cities for trading opportunities.

        Process:
        1. For each (city, event_date):
           - Get market snapshot
           - Run edge evaluation
           - If should_trade: calculate EV and create opportunity
        2. Sort by EV descending
        3. Return ranked list

        Args:
            session: Database session
            target_dates: Optional list of dates to scan (default: today + tomorrow)
            current_time: Optional timestamp override

        Returns:
            List of CityOpportunity sorted by EV (best first)
        """
        if current_time is None:
            current_time = datetime.now()

        dates = self._get_target_dates(target_dates)
        opportunities = []

        logger.info(f"\n{'='*70}")
        logger.info(f"SCANNING {len(self.config.enabled_cities)} cities × {len(dates)} dates")
        logger.info(f"Inference mode: {self.config.inference_mode}")
        logger.info(f"{'='*70}\n")

        for city in self.config.enabled_cities:
            # Skip if classifier_only mode and city has no classifier
            if self.config.inference_mode == "classifier_only":
                if not self.should_use_classifier(city):
                    logger.debug(f"Skip {city}: No classifier (classifier_only mode)")
                    continue

            for event_date in dates:
                try:
                    logger.info(f"\n[{city.upper()}] {event_date}")
                    logger.info(f"  {'─'*60}")

                    # Get market snapshot
                    market_snapshot = self._get_market_snapshot(city, event_date)
                    if not market_snapshot:
                        logger.warning(f"  ⚠ No market data available")
                        continue

                    # Log bracket data from WebSocket
                    brackets = market_snapshot.get('brackets', [])
                    logger.info(f"  📊 {len(brackets)} brackets from WebSocket:")
                    for bracket in brackets[:8]:  # Show first 8
                        floor = bracket.get('floor_strike')
                        cap = bracket.get('cap_strike')
                        bid = bracket.get('yes_bid', 0)
                        ask = bracket.get('yes_ask', 100)
                        spread = ask - bid

                        if floor is None:
                            label = f"≤{int(cap)}°F"
                        elif cap is None:
                            label = f"≥{int(floor)+1}°F"
                        else:
                            label = f"{int(floor)}-{int(cap)}°F"

                        logger.info(f"     {label:12} bid={bid:2}¢ ask={ask:2}¢ spread={spread}¢")

                    # Run edge evaluation
                    decision = self.inference.evaluate_edge(
                        city=city,
                        event_date=event_date,
                        market_snapshot=market_snapshot,
                        session=session,
                        current_time=current_time,
                    )

                    # Log decision details
                    logger.info(f"\n  🎯 EDGE ANALYSIS:")
                    logger.info(f"     Forecast:  {decision.forecast_implied_temp:.1f}°F")
                    logger.info(f"     Market:    {decision.market_implied_temp:.1f}°F")
                    logger.info(f"     Edge:      {decision.edge_degf:+.1f}°F")
                    logger.info(f"     Signal:    {decision.signal}")

                    use_ml = self.should_use_classifier(city)
                    logger.info(f"     Mode:      {'ML Classifier' if use_ml else 'Threshold-only'}")

                    if use_ml:
                        logger.info(f"     ML Prob:   {decision.edge_classifier_prob:.1%}")
                    logger.info(f"     Decision:  {'TRADE ✓' if decision.should_trade else 'NO TRADE ✗'}")
                    logger.info(f"     Reason:    {decision.reason}")

                    # Build opportunity if viable
                    opportunity = self._build_opportunity(
                        city=city,
                        event_date=event_date,
                        decision=decision,
                        market_snapshot=market_snapshot,
                        current_time=current_time,
                    )

                    if opportunity:
                        opportunities.append(opportunity)
                        logger.info(f"\n  ✅ OPPORTUNITY FOUND:")
                        logger.info(f"     Ticker:    {opportunity.ticker}")
                        logger.info(f"     Rec Side:  {opportunity.recommended_side.upper()} {opportunity.recommended_action.upper()}")
                        logger.info(f"     Price:     {opportunity.recommended_price}¢ ({opportunity.role})")
                        logger.info(f"     EV:        ${opportunity.ev_per_contract/100:.2f}/contract")
                        logger.info(f"     Spread:    {opportunity.spread}¢")
                    else:
                        logger.info(f"  ⊘ No viable opportunity (EV too low or no trade signal)")

                except Exception as e:
                    logger.error(f"Error scanning {city} {event_date}: {e}", exc_info=True)
                    continue

        # Sort by EV descending
        opportunities.sort(key=lambda x: x.ev_per_contract, reverse=True)

        # Summary
        logger.info(f"\n{'='*70}")
        logger.info(f"SCAN COMPLETE: Found {len(opportunities)} opportunities")
        if opportunities:
            best = opportunities[0]
            logger.info(f"BEST: {best.city.upper()} {best.event_date} - EV=${best.ev_per_contract/100:.2f}")
        logger.info(f"{'='*70}\n")

        return opportunities

    def get_opportunity_summary(self, opportunities: List[CityOpportunity]) -> Dict[str, Any]:
        """
        Generate summary statistics for opportunities.

        Args:
            opportunities: List of opportunities

        Returns:
            Dict with summary stats
        """
        if not opportunities:
            return {
                'count': 0,
                'total_ev': 0.0,
                'avg_ev': 0.0,
                'best_city': None,
                'inference_modes': {},
            }

        total_ev = sum(opp.ev_per_contract for opp in opportunities)
        avg_ev = total_ev / len(opportunities)

        # Count by inference mode
        mode_counts = {}
        for opp in opportunities:
            mode = opp.inference_mode
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        return {
            'count': len(opportunities),
            'total_ev': total_ev / 100.0,  # Convert to dollars
            'avg_ev': avg_ev / 100.0,
            'best_city': opportunities[0].city if opportunities else None,
            'best_ev': opportunities[0].ev_per_contract / 100.0 if opportunities else 0.0,
            'inference_modes': mode_counts,
        }
