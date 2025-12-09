"""
Order book state management from WebSocket ticker and orderbook channels.

Handles:
- Ticker channel updates (bid/ask/volume)
- Orderbook snapshot + delta (optional, for full depth)
- Market state tracking per ticker
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Current state of a Kalshi market."""

    ticker: str
    last_update: datetime

    # Best bid/ask (from ticker channel)
    yes_bid: int = 0      # cents
    yes_ask: int = 100    # cents
    no_bid: int = 0
    no_ask: int = 100

    # Volume and open interest
    volume: int = 0
    open_interest: int = 0
    dollar_volume: int = 0

    # Price history (last N updates)
    price_history: List[tuple[datetime, int]] = field(default_factory=list)
    max_history: int = 100

    @property
    def yes_mid(self) -> float:
        """Yes mid-price in cents."""
        return (self.yes_bid + self.yes_ask) / 2.0

    @property
    def yes_spread(self) -> int:
        """Yes bid-ask spread in cents."""
        return self.yes_ask - self.yes_bid

    @property
    def market_prob(self) -> float:
        """Market-implied probability from yes mid-price."""
        return self.yes_mid / 100.0

    def update_from_ticker(self, msg: dict) -> None:
        """
        Update state from ticker message.

        Args:
            msg: Ticker message payload
        """
        self.yes_bid = msg.get("yes_bid", self.yes_bid)
        self.yes_ask = msg.get("yes_ask", self.yes_ask)
        self.no_bid = msg.get("no_bid", self.no_bid)
        self.no_ask = msg.get("no_ask", self.no_ask)

        self.volume = msg.get("volume", self.volume)
        self.open_interest = msg.get("open_interest", self.open_interest)
        self.dollar_volume = msg.get("dollar_volume", self.dollar_volume)

        # Track price history
        price = msg.get("price")
        if price is not None:
            self.price_history.append((datetime.now(), price))
            # Trim to max_history
            if len(self.price_history) > self.max_history:
                self.price_history = self.price_history[-self.max_history:]

        self.last_update = datetime.now()


class OrderBookManager:
    """
    Manages order book state for all tracked markets.

    Primarily uses ticker channel for simplicity and reliability.
    Can be extended to support full orderbook_delta if needed.
    """

    def __init__(self):
        """Initialize order book manager."""
        self.markets: Dict[str, MarketState] = {}  # ticker → MarketState
        self.ticker_to_city: Dict[str, str] = {}   # ticker → city (for lookup)

    async def handle_ticker(self, message: dict) -> None:
        """
        Handle ticker channel message.

        Args:
            message: WebSocket ticker message
        """
        msg_payload = message.get("msg", {})
        ticker = msg_payload.get("market_ticker")

        if not ticker:
            logger.warning("Ticker message missing market_ticker")
            return

        # Get or create market state
        if ticker not in self.markets:
            self.markets[ticker] = MarketState(
                ticker=ticker,
                last_update=datetime.now()
            )

        # Update from ticker
        self.markets[ticker].update_from_ticker(msg_payload)

        logger.debug(
            f"Ticker update: {ticker} "
            f"bid={self.markets[ticker].yes_bid} "
            f"ask={self.markets[ticker].yes_ask}"
        )

    def get_market_state(self, ticker: str) -> Optional[MarketState]:
        """
        Get current market state for ticker.

        Args:
            ticker: Market ticker

        Returns:
            MarketState or None if not tracked
        """
        return self.markets.get(ticker)

    def get_markets_for_city(self, city: str, event_date: date) -> List[MarketState]:
        """
        Get all markets for a city and event date.

        Args:
            city: City identifier
            event_date: Event date

        Returns:
            List of MarketState objects
        """
        # Parse tickers to find matches
        # Format: KXHIGHCHI-25DEC01-B33.5
        city_codes = {
            'chicago': 'CHI',
            'austin': 'AUS',
            'denver': 'DEN',
            'los_angeles': 'LA',
            'miami': 'MIA',
            'philadelphia': 'PHI',
        }

        city_code = city_codes.get(city, city.upper()[:3])
        date_str = event_date.strftime("%y%b%d").upper()  # 25DEC09 (YYMMMDD format)

        markets = []
        for ticker, state in self.markets.items():
            if f"KXHIGH{city_code}" in ticker and date_str in ticker:
                markets.append(state)

        return markets

    def get_best_quotes(self, ticker: str) -> Optional[dict]:
        """
        Get best bid/ask for ticker.

        Args:
            ticker: Market ticker

        Returns:
            Dict with bid/ask or None
        """
        state = self.get_market_state(ticker)
        if not state:
            return None

        return {
            "ticker": ticker,
            "yes_bid": state.yes_bid,
            "yes_ask": state.yes_ask,
            "spread": state.yes_spread,
            "mid": state.yes_mid,
            "volume": state.volume,
            "last_update": state.last_update,
        }
