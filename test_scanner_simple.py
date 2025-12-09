#!/usr/bin/env python3
"""Simple scanner test without WebSocket or CLI."""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from live_trading.config import TradingConfig
from live_trading.scanner import CityScannerEngine
from live_trading.inference import InferenceWrapper
from live_trading.websocket.order_book import OrderBookManager
from live_trading.websocket.market_state import MarketStateTracker
from src.db.connection import get_db_session

async def test_scanner():
    """Test scanner with minimal setup."""
    print("Initializing...")

    # Config
    config = TradingConfig()
    config.enabled_cities = ['chicago']  # Just one city for speed
    print(f"✓ Config: {config.enabled_cities}")

    # Inference
    print("Loading inference models...")
    inference = InferenceWrapper()
    print(f"✓ Loaded models for: {list(inference.live_engine.models.keys())}")

    # Managers (no WebSocket)
    order_book_mgr = OrderBookManager()
    market_state_tracker = MarketStateTracker()

    # Load markets from DB
    print("Loading markets from DB...")
    from scripts.manual_trader import _load_market_metadata
    await _load_market_metadata(order_book_mgr, market_state_tracker, config.enabled_cities)
    print(f"✓ Loaded {len(order_book_mgr.markets)} markets")

    # Create scanner
    scanner = CityScannerEngine(
        config=config,
        inference=inference,
        ws_handler=None,
        order_book_mgr=order_book_mgr,
        market_state_tracker=market_state_tracker
    )
    print(f"✓ Scanner created")

    # Scan
    print("\nScanning...")
    with get_db_session() as session:
        opportunities = await scanner.scan_all_cities(session)

    print(f"\n✅ Scan complete: {len(opportunities)} opportunities")
    for opp in opportunities[:3]:
        print(f"  {opp.city.upper()}: EV=${opp.ev_per_contract/100:.2f}")

if __name__ == "__main__":
    asyncio.run(test_scanner())
