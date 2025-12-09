"""
Edge-based live trading daemon for Kalshi weather markets.

Main entry point for the edge trading system. Coordinates:
- WebSocket connection for real-time market data
- Minute-by-minute edge evaluation
- Order placement with maker→taker conversion
- Position tracking and risk management
- Comprehensive decision logging
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from zoneinfo import ZoneInfo

from live_trading.config import TradingConfig
from live_trading.inference import InferenceWrapper, EdgeDecision
from live_trading.market_data import get_market_snapshot
from live_trading.order_manager import OrderManager, VolumeTracker, PendingOrder
from live_trading.daily_loss_tracker import DailyLossTracker
from live_trading.websocket.handler import WebSocketHandler, KalshiAuth
from live_trading.websocket.order_book import OrderBookManager, MarketState
from live_trading.websocket.market_state import MarketStateTracker
from live_trading.db.session_logger import SessionLogger
from live_trading.utils import (
    get_current_weather_day,
    get_next_weather_day,
    is_market_open,
    CITY_TIMEZONES
)

from src.kalshi.client import KalshiClient
from src.config.settings import get_settings
from src.db.connection import get_db_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/edge_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global shutdown flag
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, requesting shutdown...")
    shutdown_requested = True


class EdgeTrader:
    """
    Main trading daemon.

    Coordinates all components for edge-based live trading.
    """

    def __init__(self, config: TradingConfig, dry_run: bool = True):
        """
        Initialize edge trader.

        Args:
            config: Trading configuration
            dry_run: If True, log decisions but don't place orders
        """
        self.config = config
        self.dry_run = dry_run

        # Settings
        settings = get_settings()

        # Core components
        self.inference = InferenceWrapper()
        self.kalshi_client = KalshiClient(
            api_key=settings.kalshi_api_key,
            private_key_path=settings.kalshi_private_key_path,
            base_url=settings.kalshi_base_url
        )
        self.order_manager = OrderManager(self.kalshi_client, config)
        self.volume_tracker = VolumeTracker(config.volume_lookback_minutes)
        self.position_tracker = DailyLossTracker(config)
        self.session_logger = SessionLogger()

        # WebSocket components
        auth = KalshiAuth(settings.kalshi_api_key, settings.kalshi_private_key_path)
        ws_url = self._get_ws_url(settings.kalshi_base_url)
        self.ws_handler = WebSocketHandler(ws_url, auth)
        self.order_book_manager = OrderBookManager()
        self.market_state_tracker = MarketStateTracker()

        # Register WebSocket handlers
        self.ws_handler.register_handler("ticker", self.order_book_manager.handle_ticker)
        self.ws_handler.register_handler("fill", self._on_fill)
        self.ws_handler.register_handler("trade", self._on_trade)
        self.ws_handler.register_handler("market_lifecycle_v2", self.market_state_tracker.handle_market_lifecycle)

        # Session tracking
        self.session_id: Optional[UUID] = None

        logger.info(
            f"EdgeTrader initialized "
            f"(dry_run={dry_run}, cities={config.enabled_cities}, "
            f"aggressiveness={config.aggressiveness:.2f})"
        )

    def _get_ws_url(self, base_url: str) -> str:
        """Derive WebSocket URL from REST base URL."""
        ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = ws_url.replace("/trade-api/v2", "/trade-api/ws/v2")
        return ws_url

    async def run(self) -> None:
        """
        Main trading loop.

        Steps:
        1. Start session logging
        2. Start all background tasks (WebSocket, order manager)
        3. Subscribe to market data
        4. Run minute-by-minute evaluation loop
        5. Graceful shutdown
        """
        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            # Start session
            self.session_id = self.session_logger.start_session(self.config, self.dry_run)

            # Start background tasks
            logger.info("Starting background tasks...")
            ws_task = asyncio.create_task(self.ws_handler.start())
            await asyncio.sleep(2)  # Give WebSocket time to connect

            await self.order_manager.start()

            # Subscribe to channels
            logger.info("Subscribing to market data...")
            await self._subscribe_to_markets()

            # Main evaluation loop
            logger.info("Starting main evaluation loop...")
            await self._trading_loop()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", exc_info=True)

        finally:
            # Cleanup
            await self._shutdown()

    async def _subscribe_to_markets(self) -> None:
        """Subscribe to required WebSocket channels."""
        # Get series tickers for enabled cities
        series_map = {
            'chicago': 'KXHIGHCHI',
            'austin': 'KXHIGHAUS',
            'denver': 'KXHIGHDEN',
            'los_angeles': 'KXHIGHLAX',
            'miami': 'KXHIGHMIA',
            'philadelphia': 'KXHIGHPHI',
        }

        series_tickers = [series_map[city] for city in self.config.enabled_cities]

        # Subscribe to ticker (bid/ask updates)
        await self.ws_handler.subscribe(
            channels=["ticker"],
            series_tickers=series_tickers
        )

        # Subscribe to fills (our order fills)
        await self.ws_handler.subscribe(channels=["fill"])

        # Subscribe to trades (for volume tracking)
        # Note: trades channel doesn't accept series_tickers filter, use no filter (all markets)
        await self.ws_handler.subscribe(channels=["trades"])

        # Subscribe to market lifecycle (open/close events)
        await self.ws_handler.subscribe(
            channels=["market_lifecycle_v2"],
            series_tickers=series_tickers
        )

        logger.info(f"Subscribed to channels for {series_tickers}")

    async def _trading_loop(self) -> None:
        """
        Main evaluation loop: check for edges every minute.

        For each enabled city:
        1. Get current and next-day event dates
        2. Get market snapshot
        3. Run edge evaluation
        4. Log decision
        5. Execute trade if should_trade=True
        """
        while not shutdown_requested:
            for city in self.config.enabled_cities:
                try:
                    await self._evaluate_city(city)
                except Exception as e:
                    logger.error(f"Error evaluating {city}: {e}", exc_info=True)

            # Wait 60 seconds before next evaluation
            await asyncio.sleep(60)

        logger.info("Trading loop stopped")

    async def _evaluate_city(self, city: str) -> None:
        """
        Evaluate trading opportunities for a city.

        Args:
            city: City to evaluate
        """
        # Get event dates (today and tomorrow)
        current_day = get_current_weather_day(city)
        next_day = get_next_weather_day(city)
        event_dates = [current_day, next_day]

        for event_date in event_dates:
            # Skip if market closed
            if not is_market_open(city, event_date):
                continue

            # Get market snapshot
            market_snapshot = self._get_market_snapshot(city, event_date)
            if not market_snapshot:
                logger.debug(f"No market data for {city} {event_date}")
                continue

            # Run edge evaluation
            with get_db_session() as session:
                decision = self.inference.evaluate_edge(
                    city=city,
                    event_date=event_date,
                    market_snapshot=market_snapshot,
                    session=session,
                    edge_threshold_degf=self.config.edge_threshold_degf,
                    confidence_threshold=self.config.effective_confidence_threshold
                )

            # Log decision FIRST (always, even if no trade)
            decision_id = self.session_logger.log_decision(
                session_id=self.session_id,
                city=city,
                event_date=event_date,
                edge_decision=decision,
                market_snapshot=market_snapshot,
                order_id=None  # Will update if order placed
            )

            # Execute trade if decision says to
            if decision.should_trade and not self.dry_run:
                order_id = await self._execute_trade(city, event_date, decision, decision_id)

                # Update decision with order_id (if order was placed)
                if order_id:
                    self.session_logger.update_decision_with_order(decision_id, order_id)

            # Log summary
            if decision.should_trade:
                logger.info(
                    f"[{city.upper()} {event_date}] TRADE: "
                    f"{decision.signal} edge={decision.edge_degf:+.2f}°F "
                    f"prob={decision.edge_classifier_prob:.3f} "
                    f"bracket={decision.recommended_bracket} "
                    f"{'[DRY-RUN]' if self.dry_run else '[LIVE]'}"
                )
            else:
                logger.debug(
                    f"[{city.upper()} {event_date}] NO TRADE: {decision.reason}"
                )

    def _get_market_snapshot(self, city: str, event_date: date) -> Optional[Dict]:
        """
        Get current market snapshot for city/event.

        Args:
            city: City
            event_date: Event date

        Returns:
            Dict with market data or None
        """
        # Use shared utility for market snapshot retrieval
        return get_market_snapshot(
            order_book_mgr=self.order_book_manager,
            market_state_tracker=self.market_state_tracker,
            city=city,
            event_date=event_date,
        )

    async def _execute_trade(
        self,
        city: str,
        event_date: date,
        decision: EdgeDecision,
        decision_id: UUID
    ) -> Optional[UUID]:
        """
        Execute a trade based on edge decision.

        Args:
            city: City
            event_date: Event date
            decision: EdgeDecision with should_trade=True

        Returns:
            order_id if order placed, None otherwise
        """
        ticker = decision.recommended_bracket
        if not ticker:
            logger.warning("Cannot execute trade: no recommended bracket")
            return None

        # Check position limits
        can_trade, reason = self.position_tracker.can_open_position(city, event_date)
        if not can_trade:
            logger.info(f"Position limit check failed: {reason}")
            return None

        # Size position (use existing risk utilities)
        from src.trading.risk import PositionSizer
        sizer = PositionSizer(
            bankroll_usd=self.config.bankroll_usd,
            kelly_fraction=self.config.effective_kelly_fraction,
            max_bet_usd=self.config.max_bet_per_trade_usd,
        )

        # Estimate edge for sizing
        edge = decision.edge_classifier_prob - 0.5  # Probability above 50%

        size_result = sizer.size_position(
            model_prob=decision.edge_classifier_prob,
            price_cents=50,  # TODO: Get actual entry price from market snapshot
            model_std_degf=decision.forecast_uncertainty
        )

        num_contracts = size_result.num_contracts
        if num_contracts <= 0:
            logger.info("Position sizer returned 0 contracts")
            return None

        # Compute maker timeout based on volume
        volume = self.volume_tracker.get_recent_volume(ticker)
        maker_timeout_sec = self.config.compute_maker_timeout_sec(volume)

        # Place order
        try:
            # Get entry price (simplified: use ask for buy)
            market_state = self.order_book_manager.get_market_state(ticker)
            if not market_state:
                logger.warning(f"No market state for {ticker}")
                return None

            entry_price = market_state.yes_ask

            # Create order via Kalshi API
            result = self.kalshi_client.create_order(
                ticker=ticker,
                side=decision.recommended_side or "yes",
                action=decision.recommended_action or "buy",
                count=num_contracts,
                order_type="limit",
                yes_price=entry_price if decision.recommended_side == "yes" else None,
                no_price=entry_price if decision.recommended_side == "no" else None,
                client_order_id=f"edge-{city}-{event_date}-{uuid4().hex[:8]}"
            )

            order_id_str = result.get("order", {}).get("order_id")
            if not order_id_str:
                logger.error("Order creation failed: no order_id in response")
                return None

            order_id = UUID(order_id_str)

            # Track pending order
            pending_order = PendingOrder(
                order_id=order_id,
                ticker=ticker,
                city=city,
                event_date=str(event_date),
                side=decision.recommended_side or "yes",
                action=decision.recommended_action or "buy",
                num_contracts=num_contracts,
                maker_price_cents=entry_price,
                placed_at=datetime.now(),
                maker_timeout_sec=maker_timeout_sec
            )

            self.order_manager.track_order(pending_order)

            # NOTE: Position tracking moved to _on_fill (Bug #5 fix)
            # Positions should be added when filled, not when placed

            # Log to database with real Kalshi order_id and decision_id
            self.session_logger.log_order(
                order_id=order_id,  # Real Kalshi order_id
                session_id=self.session_id,
                decision_id=decision_id,  # Link to decision
                city=city,
                event_date=event_date,
                ticker=ticker,
                bracket_label="",  # TODO: Parse from ticker
                side=decision.recommended_side or "yes",
                action=decision.recommended_action or "buy",
                num_contracts=num_contracts,
                maker_price_cents=entry_price,
                notional_usd=num_contracts * entry_price / 100.0,
                volume_at_order=volume,
                maker_timeout_used_sec=maker_timeout_sec
            )

            logger.info(
                f"Order placed: {order_id} "
                f"{num_contracts}x {ticker} @ {entry_price}¢ "
                f"(timeout={maker_timeout_sec}s, volume={volume})"
            )

            return order_id

        except Exception as e:
            logger.error(f"Failed to execute trade: {e}", exc_info=True)
            return None

    async def _on_fill(self, message: dict) -> None:
        """
        Handle fill notification from WebSocket.

        Args:
            message: Fill message from WebSocket
        """
        msg_payload = message.get("msg", {})
        order_id_str = msg_payload.get("order_id")

        if order_id_str:
            order_id = UUID(order_id_str)
            self.order_manager.on_fill(order_id, msg_payload)

            # Get order details from database
            from live_trading.db.models import TradingOrder
            with get_db_session() as db:
                order = db.query(TradingOrder).filter_by(order_id=order_id).first()

                if order:
                    # Add position now that order is filled (Bug #5 fix)
                    self.position_tracker.add_position(
                        ticker=order.ticker,
                        city=order.city,
                        event_date=order.event_date,
                        side=order.side,
                        num_contracts=order.num_contracts,
                        entry_price_cents=msg_payload.get("yes_price") or msg_payload.get("no_price") or order.maker_price_cents
                    )

                    # Update order status in database
                    self.session_logger.update_order_status(
                        order_id=order_id,
                        new_status="filled",
                        note="Filled via WebSocket notification",
                        final_fill_price_cents=msg_payload.get("yes_price") or msg_payload.get("no_price"),
                        is_taker_fill=msg_payload.get("is_taker", False)
                    )

                    logger.info(f"Fill received: {order_id} (position added)")
                else:
                    logger.warning(f"Fill for unknown order: {order_id}")

    async def _on_trade(self, message: dict) -> None:
        """
        Handle public trade message for volume tracking.

        Args:
            message: Trade message from WebSocket
        """
        msg_payload = message.get("msg", {})
        ticker = msg_payload.get("market_ticker")
        count = msg_payload.get("count", 0)

        if ticker and count:
            self.volume_tracker.add_trade(ticker, count)

    async def _shutdown(self) -> None:
        """Graceful shutdown: stop all components."""
        logger.info("Shutting down...")

        # Stop background tasks
        await self.order_manager.stop()
        await self.ws_handler.stop()

        # End session
        if self.session_id:
            self.session_logger.end_session(self.session_id, status="stopped")

        logger.info("Shutdown complete")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Edge-based live trading daemon for Kalshi weather markets"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (default: dry-run)"
    )
    parser.add_argument(
        "--city",
        type=str,
        help="Single city to trade (overrides config enabled_cities)"
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("config/edge_trader.json"),
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(args.log_level)

    # Load config
    config = TradingConfig.from_json(args.config_file)

    # Override city if specified
    if args.city:
        config.enabled_cities = [args.city]

    dry_run = not args.live

    # Validate config
    errors = config.validate()
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    # Display configuration
    logger.info("=" * 60)
    logger.info("Edge Trader Configuration:")
    logger.info(f"  Mode: {'LIVE TRADING' if not dry_run else 'DRY-RUN'}")
    logger.info(f"  Cities: {config.enabled_cities}")
    logger.info(f"  Aggressiveness: {config.aggressiveness:.2f}")
    logger.info(f"  Confidence threshold: {config.effective_confidence_threshold:.3f}")
    logger.info(f"  Kelly fraction: {config.effective_kelly_fraction:.3f}")
    logger.info(f"  Max bet: ${config.max_bet_per_trade_usd:.0f}")
    logger.info(f"  Max positions: {config.max_total_positions}")
    logger.info("=" * 60)

    # Confirm if live trading
    if not dry_run:
        logger.warning("=" * 60)
        logger.warning("LIVE TRADING ENABLED - REAL ORDERS WILL BE PLACED")
        logger.warning("=" * 60)
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            logger.info("Cancelled by user")
            sys.exit(0)

    # Run trader
    trader = EdgeTrader(config, dry_run=dry_run)
    asyncio.run(trader.run())


if __name__ == "__main__":
    main()
