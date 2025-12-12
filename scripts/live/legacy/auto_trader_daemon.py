#!/usr/bin/env python3
"""
Automatic trading daemon.

Continuously scans all 6 cities and executes trades automatically
with safety limits.
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from live_trading.config import TradingConfig
from live_trading.scanner import CityScannerEngine
from live_trading.incremental_executor import IncrementalOrderExecutor
from live_trading.daily_loss_tracker import DailyLossTracker
from live_trading.inference import InferenceWrapper
from live_trading.order_manager import OrderManager
from live_trading.websocket.handler import WebSocketHandler, KalshiAuth
from live_trading.websocket.order_book import OrderBookManager
from live_trading.websocket.market_state import MarketStateTracker
from src.kalshi.client import KalshiClient
from src.config.settings import get_settings
from src.db.connection import get_db_session

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

shutdown_requested = False


async def _load_market_metadata(order_book_mgr, market_state_tracker, enabled_cities):
    """Load active market metadata from database into trackers."""
    from src.db.connection import get_db_session
    from src.db.models import KalshiMarket
    from live_trading.websocket.market_state import MarketMetadata
    from live_trading.websocket.order_book import MarketState
    from datetime import date, timedelta, datetime

    with get_db_session() as session:
        start_date = date.today()
        end_date = start_date + timedelta(days=7)

        markets = session.query(KalshiMarket).filter(
            KalshiMarket.event_date >= start_date,
            KalshiMarket.event_date <= end_date,
            KalshiMarket.status == 'active',
            KalshiMarket.city.in_(enabled_cities)
        ).all()

        logger.info(f"Loading {len(markets)} active markets...")

        for market in markets:
            metadata = MarketMetadata(
                market_ticker=market.ticker,
                event_ticker=market.ticker.split('-')[0],
                series_ticker=f"KXHIGH{market.city.upper()[:3]}",
                strike_type=market.strike_type,
                floor_strike=market.floor_strike,
                cap_strike=market.cap_strike,
            )
            market_state_tracker.markets[market.ticker] = metadata

            market_state = MarketState(
                ticker=market.ticker,
                last_update=datetime.now(),
                yes_bid=50,
                yes_ask=50,
            )
            order_book_mgr.markets[market.ticker] = market_state
            order_book_mgr.ticker_to_city[market.ticker] = market.city

        logger.info(f"✓ Loaded {len(markets)} markets")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, requesting shutdown...")
    shutdown_requested = True


class AutoTradingDaemon:
    """Automatic trading daemon with safety limits."""

    def __init__(
        self,
        scanner: CityScannerEngine,
        executor: IncrementalOrderExecutor,
        loss_tracker: DailyLossTracker,
        config: TradingConfig,
        dry_run: bool = True
    ):
        self.scanner = scanner
        self.executor = executor
        self.loss_tracker = loss_tracker
        self.config = config
        self.dry_run = dry_run

    async def run(self):
        """Main daemon loop."""
        logger.info("🤖 Auto trading daemon started")
        logger.info(f"Config: max_bet=${self.config.max_bet_per_trade_usd}, max_loss=${self.config.max_daily_loss_usd}")
        logger.info(f"Enabled cities: {self.config.enabled_cities}")
        logger.info(f"Dry run: {self.dry_run}")

        scan_interval = 60  # seconds

        while not shutdown_requested:
            try:
                # 1. Check circuit breaker
                if self.loss_tracker.should_pause_trading(cooldown_minutes=5):
                    status = self.loss_tracker.get_circuit_breaker_status()
                    logger.warning(
                        f"🔴 Circuit breaker active: "
                        f"Daily P&L ${status['daily_pnl_usd']:.2f}"
                    )
                    await asyncio.sleep(300)  # Wait 5 minutes
                    continue

                # 2. Scan for opportunities
                logger.info("Scanning all cities...")
                with get_db_session() as session:
                    opportunities = await self.scanner.scan_all_cities(session)

                if not opportunities:
                    logger.info("No opportunities found")
                    await asyncio.sleep(scan_interval)
                    continue

                # 3. Filter by position limits
                tradeable = []
                for opp in opportunities:
                    can_trade, reason = self.loss_tracker.can_trade(opp.city, opp.event_date)
                    if can_trade:
                        tradeable.append(opp)
                    else:
                        logger.debug(f"Skip {opp.city} {opp.event_date}: {reason}")

                if not tradeable:
                    logger.info("No tradeable opportunities (limits)")
                    await asyncio.sleep(scan_interval)
                    continue

                # 4. Take best opportunity
                best = tradeable[0]
                logger.info(
                    f"🎯 Best: {best.city.upper()} {best.event_date} "
                    f"edge={best.edge_degf:+.1f}°F prob={best.edge_classifier_prob:.0%} "
                    f"EV=${best.ev_per_contract/100:.2f} [{best.inference_mode}]"
                )

                # 5. Execute trade
                if not self.dry_run:
                    await self._execute_trade(best)
                else:
                    logger.info("[DRY RUN] Would trade")

                # 6. Log status
                self.loss_tracker.log_circuit_breaker_status()

                # 7. Wait before next scan
                await asyncio.sleep(scan_interval)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(scan_interval)

        logger.info("Auto trading daemon stopped")

    async def _execute_trade(self, opp):
        """Execute trade with max bet limit."""
        amount_usd = self.config.max_bet_per_trade_usd

        # Plan execution
        chunks = self.executor.plan_incremental_entry(
            ticker=opp.ticker,
            target_usd=amount_usd,
            side=opp.recommended_side,
            action=opp.recommended_action,
            yes_bid=opp.yes_bid,
            yes_ask=opp.yes_ask,
        )

        logger.info(f"Executing {len(chunks)} chunks totaling ${amount_usd:.2f}")

        # Execute without confirmation (automatic)
        placed = await self.executor.execute_incremental_order(
            ticker=opp.ticker,
            city=opp.city,
            event_date=opp.event_date,
            side=opp.recommended_side,
            action=opp.recommended_action,
            yes_bid=opp.yes_bid,
            yes_ask=opp.yes_ask,
            chunks=chunks,
            confirm_each=False,
        )

        # Add position
        if placed:
            total_contracts = sum(chunk.num_contracts for _, chunk in placed)
            self.loss_tracker.add_position(
                ticker=opp.ticker,
                city=opp.city,
                event_date=opp.event_date,
                side=opp.recommended_side,
                num_contracts=total_contracts,
                entry_price_cents=opp.recommended_price,
            )
            logger.info(f"✅ Executed trade: {len(placed)} orders, {total_contracts} contracts")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Automatic trading daemon")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--live',
        action='store_true',
        help='Live trading mode (place real orders)'
    )
    mode_group.add_argument(
        '--dry-run',
        dest='dry_run',
        action='store_true',
        default=True,
        help='Dry run mode - no real orders (DEFAULT)'
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/auto_trader.json'),
        help='Config file path'
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    if args.config.exists():
        config = TradingConfig.from_json(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = TradingConfig()

    # Ensure all 6 cities enabled
    if len(config.enabled_cities) < 6:
        config.enabled_cities = [
            "chicago", "austin", "denver",
            "los_angeles", "miami", "philadelphia"
        ]

    # Validate
    errors = config.validate()
    if errors:
        logger.error(f"Config errors: {errors}")
        return 1

    # Determine dry run mode
    if args.live:
        dry_run = False
        logger.warning("🔴 LIVE TRADING MODE - Real orders will be placed!")
    else:
        dry_run = True
        logger.info("⚠ DRY RUN MODE - No real orders will be placed")

    # Initialize components
    settings = get_settings()

    kalshi_client = KalshiClient(
        api_key=settings.kalshi_api_key,
        private_key_path=settings.kalshi_private_key_path,
        base_url=settings.kalshi_base_url
    )

    inference = InferenceWrapper()
    order_manager = OrderManager(kalshi_client, config)
    loss_tracker = DailyLossTracker(config)

    # WebSocket
    auth = KalshiAuth(settings.kalshi_api_key, settings.kalshi_private_key_path)
    ws_url = "wss://trading-api.kalshi.com/trade-api/ws/v2"

    ws_handler = WebSocketHandler(ws_url, auth)
    order_book_mgr = OrderBookManager()
    market_state_tracker = MarketStateTracker()

    # Register WebSocket handlers (ticker channel provides bid/ask/volume from WS)
    ws_handler.register_handler("ticker", order_book_mgr.handle_ticker)

    # Start background tasks
    await order_manager.start()
    asyncio.create_task(ws_handler.start())
    await asyncio.sleep(3)  # Let WebSocket connect

    # Subscribe to market data channels
    logger.info("Subscribing to market data...")
    series_map = {
        'chicago': 'KXHIGHCHI',
        'austin': 'KXHIGHAUS',
        'denver': 'KXHIGHDEN',
        'los_angeles': 'KXHIGHLAX',
        'miami': 'KXHIGHMIA',
        'philadelphia': 'KXHIGHPHI',
    }
    series_tickers = [series_map[city] for city in config.enabled_cities if city in series_map]

    await ws_handler.subscribe(channels=["ticker"], series_tickers=series_tickers)
    await ws_handler.subscribe(channels=["market_lifecycle_v2"], series_tickers=series_tickers)
    logger.info(f"Subscribed to {len(series_tickers)} series")

    # Load market metadata from database
    logger.info("Loading market metadata from database...")
    await _load_market_metadata(order_book_mgr, market_state_tracker, config.enabled_cities)

    # Initialize scanner and executor
    scanner = CityScannerEngine(
        config=config,
        inference=inference,
        ws_handler=ws_handler,
        order_book_mgr=order_book_mgr,
        market_state_tracker=market_state_tracker
    )

    executor = IncrementalOrderExecutor(
        order_manager=order_manager,
        config=config,
        order_book_mgr=order_book_mgr
    )

    # Create and run daemon
    daemon = AutoTradingDaemon(
        scanner=scanner,
        executor=executor,
        loss_tracker=loss_tracker,
        config=config,
        dry_run=dry_run
    )

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        await daemon.run()
    finally:
        logger.info("Cleaning up...")
        await ws_handler.stop()
        await order_manager.stop()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
