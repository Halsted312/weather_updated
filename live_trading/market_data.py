"""
Shared market data utilities.

Centralizes market snapshot retrieval logic for consistency across
all trading components.

IMPORTANT: Market data comes from WebSocket (unlimited rate), NOT REST API.
The WebSocket ticker channel provides real-time bid/ask/volume updates that
are tracked by OrderBookManager. This avoids consuming precious REST API quota
(28 req/sec) which is reserved for data ingestion.
"""

import logging
from datetime import date, datetime
from typing import Optional, Dict, Any

from live_trading.websocket.order_book import OrderBookManager
from live_trading.websocket.market_state import MarketStateTracker

logger = logging.getLogger(__name__)


def get_market_snapshot(
    order_book_mgr: OrderBookManager,
    market_state_tracker: MarketStateTracker,
    city: str,
    event_date: date,
) -> Optional[Dict[str, Any]]:
    """
    Get current market snapshot for city/event.

    Builds a snapshot with:
    - Bracket list (ticker, yes_bid, yes_ask, floor_strike, cap_strike)
    - Best overall bid/ask across all brackets
    - Timestamp

    Args:
        order_book_mgr: Order book manager with current prices
        market_state_tracker: Market state tracker with metadata
        city: City identifier
        event_date: Event settlement date

    Returns:
        Dict with market snapshot, or None if no markets available
    """
    # Get all markets for this city/event
    markets = order_book_mgr.get_markets_for_city(city, event_date)

    if not markets:
        logger.debug(f"No markets found for {city} {event_date}")
        return None

    # Build bracket data for market-implied calculation
    brackets = []
    best_bid = 0
    best_ask = 100

    for market in markets:
        # Get bracket metadata from market state tracker
        metadata = market_state_tracker.get_market_metadata(market.ticker)

        floor_strike = None
        cap_strike = None

        if metadata and metadata.floor_strike is not None:
            floor_strike = metadata.floor_strike
        if metadata and metadata.cap_strike is not None:
            cap_strike = metadata.cap_strike

        brackets.append({
            'ticker': market.ticker,
            'yes_bid': market.yes_bid,
            'yes_ask': market.yes_ask,
            'floor_strike': floor_strike,
            'cap_strike': cap_strike,
        })

        # Track best overall bid/ask
        if market.yes_bid > best_bid:
            best_bid = market.yes_bid
        if market.yes_ask < best_ask:
            best_ask = market.yes_ask

    return {
        'city': city,
        'event_date': event_date,
        'brackets': brackets,
        'best_bid': best_bid,
        'best_ask': best_ask,
        'timestamp': datetime.now()
    }
