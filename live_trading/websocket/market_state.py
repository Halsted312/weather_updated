"""
Market metadata tracking from lifecycle channels.

Tracks market open/close times, settlement status, and event metadata
from market_lifecycle_v2 and event_lifecycle channels.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MarketMetadata:
    """Metadata for a Kalshi market from lifecycle events."""

    market_ticker: str
    event_ticker: str
    series_ticker: str

    # Lifecycle timestamps (epoch seconds UTC)
    open_ts: Optional[int] = None
    close_ts: Optional[int] = None
    expected_expiration_ts: Optional[int] = None

    # Market details
    strike_type: Optional[str] = None  # "less", "between", "greater"
    floor_strike: Optional[float] = None
    cap_strike: Optional[float] = None

    # Status
    status: str = "unknown"  # created, open, closed, settled
    last_event_type: Optional[str] = None

    # Descriptive
    name: Optional[str] = None
    title: Optional[str] = None

    @property
    def is_open(self) -> bool:
        """Check if market is currently open for trading."""
        return self.status == "open"

    @property
    def open_time_utc(self) -> Optional[datetime]:
        """Convert open_ts to datetime."""
        if self.open_ts:
            return datetime.fromtimestamp(self.open_ts)
        return None

    @property
    def close_time_utc(self) -> Optional[datetime]:
        """Convert close_ts to datetime."""
        if self.close_ts:
            return datetime.fromtimestamp(self.close_ts)
        return None


class MarketStateTracker:
    """
    Tracks market metadata from lifecycle channels.

    Subscribes to:
    - market_lifecycle_v2: open/close times, settlement
    - event_lifecycle: event creation and updates
    """

    def __init__(self):
        """Initialize market state tracker."""
        self.markets: Dict[str, MarketMetadata] = {}  # market_ticker → metadata
        self.events: Dict[str, dict] = {}  # event_ticker → event data

    async def handle_market_lifecycle(self, message: dict) -> None:
        """
        Handle market_lifecycle_v2 message.

        Args:
            message: WebSocket market_lifecycle_v2 message
        """
        msg_payload = message.get("msg", {})
        ticker = msg_payload.get("market_ticker")

        if not ticker:
            logger.warning("Market lifecycle message missing market_ticker")
            return

        event_type = msg_payload.get("event_type", "unknown")

        # Get or create metadata
        if ticker not in self.markets:
            self.markets[ticker] = MarketMetadata(
                market_ticker=ticker,
                event_ticker=msg_payload.get("event_ticker", ""),
                series_ticker=msg_payload.get("series_ticker", ""),
            )

        meta = self.markets[ticker]

        # Update from message
        meta.last_event_type = event_type
        meta.open_ts = msg_payload.get("open_ts", meta.open_ts)
        meta.close_ts = msg_payload.get("close_ts", meta.close_ts)

        # Parse additional metadata if present
        additional = msg_payload.get("additional_metadata", {})
        if additional:
            meta.strike_type = additional.get("strike_type", meta.strike_type)
            meta.floor_strike = additional.get("floor_strike", meta.floor_strike)
            meta.cap_strike = additional.get("cap_strike", meta.cap_strike)
            meta.name = additional.get("name", meta.name)
            meta.title = additional.get("title", meta.title)
            meta.expected_expiration_ts = additional.get("expected_expiration_ts", meta.expected_expiration_ts)

        # Update status based on event
        if event_type == "created":
            meta.status = "created"
        elif event_type in ("open", "activated"):
            meta.status = "open"
        elif event_type in ("closed", "deactivated"):
            meta.status = "closed"
        elif event_type == "settled":
            meta.status = "settled"

        logger.info(
            f"Market lifecycle: {ticker} {event_type} "
            f"(status={meta.status}, open_ts={meta.open_ts})"
        )

    async def handle_event_lifecycle(self, message: dict) -> None:
        """
        Handle event_lifecycle message.

        Args:
            message: WebSocket event_lifecycle message
        """
        msg_payload = message.get("msg", {})
        event_ticker = msg_payload.get("event_ticker")

        if not event_ticker:
            logger.warning("Event lifecycle message missing event_ticker")
            return

        # Store event data
        self.events[event_ticker] = msg_payload

        logger.debug(f"Event lifecycle: {event_ticker}")

    def get_market_metadata(self, ticker: str) -> Optional[MarketMetadata]:
        """
        Get metadata for a market.

        Args:
            ticker: Market ticker

        Returns:
            MarketMetadata or None
        """
        return self.markets.get(ticker)

    def get_open_markets(self) -> List[MarketMetadata]:
        """
        Get all currently open markets.

        Returns:
            List of MarketMetadata for open markets
        """
        return [m for m in self.markets.values() if m.is_open]

    def get_markets_for_series(self, series_ticker: str) -> List[MarketMetadata]:
        """
        Get all markets for a series.

        Args:
            series_ticker: Series ticker (e.g., "KXHIGHCHI")

        Returns:
            List of MarketMetadata
        """
        return [
            m for m in self.markets.values()
            if m.series_ticker == series_ticker
        ]
