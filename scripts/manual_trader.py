#!/usr/bin/env python3
"""
Manual trading CLI entry point.

Interactive terminal for scanning all 6 cities and manually executing trades.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from live_trading.config import TradingConfig
from live_trading.scanner import CityScannerEngine
from live_trading.incremental_executor import IncrementalOrderExecutor
from live_trading.inference import InferenceWrapper
from live_trading.position_tracker import PositionTracker
from live_trading.order_manager import OrderManager
from live_trading.websocket.handler import WebSocketHandler, KalshiAuth
from live_trading.websocket.order_book import OrderBookManager
from live_trading.websocket.market_state import MarketStateTracker
from live_trading.ui.manual_cli import ManualTradingCLI
from src.kalshi.client import KalshiClient
from src.config.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


async def _load_market_metadata(order_book_mgr, market_state_tracker, enabled_cities):
    """Load active market metadata from database into trackers."""
    from src.db.connection import get_db_session
    from src.db.models import KalshiMarket
    from live_trading.websocket.market_state import MarketMetadata
    from live_trading.websocket.order_book import MarketState
    from datetime import date, timedelta, datetime

    with get_db_session() as session:
        # Load markets for next 7 days
        start_date = date.today()
        end_date = start_date + timedelta(days=7)

        markets = session.query(KalshiMarket).filter(
            KalshiMarket.event_date >= start_date,
            KalshiMarket.event_date <= end_date,
            KalshiMarket.status == 'active',
            KalshiMarket.city.in_(enabled_cities)
        ).all()

        logger.info(f"Loading {len(markets)} active markets from database...")

        for market in markets:
            # Populate MarketStateTracker with metadata
            metadata = MarketMetadata(
                market_ticker=market.ticker,
                event_ticker=market.ticker.split('-')[0],  # Approximate
                series_ticker=f"KXHIGH{market.city.upper()[:3]}",
                strike_type=market.strike_type,
                floor_strike=market.floor_strike,
                cap_strike=market.cap_strike,
            )
            market_state_tracker.markets[market.ticker] = metadata

            # Populate OrderBookManager with initial state (prices will come from WS)
            market_state = MarketState(
                ticker=market.ticker,
                last_update=datetime.now(),
                yes_bid=50,  # Default, will be updated by WS
                yes_ask=50,
            )
            order_book_mgr.markets[market.ticker] = market_state
            order_book_mgr.ticker_to_city[market.ticker] = market.city

        logger.info(f"✓ Loaded {len(markets)} markets into trackers")
        logger.info(f"  Markets by city: {[(m.city, m.event_date) for m in markets[:5]]}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Manual trading CLI")

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
        help='Config file path (optional)'
    )
    parser.add_argument(
        '--cities',
        nargs='+',
        help='Cities to enable (default: all 6)'
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    if args.config and args.config.exists():
        config = TradingConfig.from_json(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = TradingConfig()
        logger.info("Using default config")

    # Override cities if specified
    if args.cities:
        config.enabled_cities = args.cities

    # Set all 6 cities by default
    if config.enabled_cities == ["chicago"]:  # Default from config
        config.enabled_cities = [
            "chicago", "austin", "denver",
            "los_angeles", "miami", "philadelphia"
        ]

    # Validate config
    errors = config.validate()
    if errors:
        logger.error(f"Config validation failed: {errors}")
        return 1

    logger.info(f"Config: {config}")

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
    position_tracker = PositionTracker(config)

    # WebSocket components (for real-time market data)
    auth = KalshiAuth(settings.kalshi_api_key, settings.kalshi_private_key_path)
    ws_url = "wss://trading-api.kalshi.com/trade-api/ws/v2"  # From settings if available

    ws_handler = WebSocketHandler(ws_url, auth)
    order_book_mgr = OrderBookManager()
    market_state_tracker = MarketStateTracker()

    # Register WebSocket handlers (ticker channel provides bid/ask/volume)
    ws_handler.register_handler("ticker", order_book_mgr.handle_ticker)

    # Load market metadata from database FIRST (don't wait for WebSocket)
    logger.info("Loading market metadata from database...")
    await _load_market_metadata(order_book_mgr, market_state_tracker, config.enabled_cities)

    # TODO: WebSocket for live price updates (skipping for now - hangs on subscribe)
    # Using DB market metadata with default prices (50¢ bid/ask)
    # WebSocket integration can be added later
    logger.info("⚠ Skipping WebSocket (using DB prices) - TODO: fix WebSocket subscribe hanging")

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

    # Run CLI
    cli = ManualTradingCLI(
        scanner=scanner,
        executor=executor,
        position_tracker=position_tracker,
        dry_run=dry_run
    )

    try:
        await cli.run()
    finally:
        # Cleanup
        logger.info("Shutting down...")
        await ws_handler.stop()
        await order_manager.stop()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
