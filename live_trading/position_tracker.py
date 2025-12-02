"""
Position tracking with limit enforcement.

Tracks open positions by ticker, by city, and globally.
Enforces position limits at multiple levels.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Set, Tuple, Optional

from live_trading.config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position."""

    ticker: str
    city: str
    event_date: date
    side: str  # "yes" or "no"
    num_contracts: int
    entry_price_cents: int
    opened_at: datetime


class PositionTracker:
    """
    Tracks open positions and enforces limits.

    Limits enforced:
    - Max positions per city/event
    - Max total open positions
    - Max daily loss
    """

    def __init__(self, config: TradingConfig):
        """
        Initialize position tracker.

        Args:
            config: Trading configuration with limits
        """
        self.config = config

        # Position tracking
        self.positions: Dict[str, Position] = {}  # ticker → Position
        self.city_event_positions: Dict[Tuple[str, date], Set[str]] = defaultdict(set)  # (city, event_date) → {tickers}

        # P&L tracking
        self.daily_pnl_cents: Dict[date, int] = defaultdict(int)  # date → pnl_cents
        self.total_pnl_cents: int = 0

    def add_position(
        self,
        ticker: str,
        city: str,
        event_date: date,
        side: str,
        num_contracts: int,
        entry_price_cents: int
    ) -> None:
        """
        Add a new position.

        Args:
            ticker: Market ticker
            city: City
            event_date: Event date
            side: "yes" or "no"
            num_contracts: Number of contracts
            entry_price_cents: Entry price in cents
        """
        position = Position(
            ticker=ticker,
            city=city,
            event_date=event_date,
            side=side,
            num_contracts=num_contracts,
            entry_price_cents=entry_price_cents,
            opened_at=datetime.now()
        )

        self.positions[ticker] = position
        self.city_event_positions[(city, event_date)].add(ticker)

        logger.info(
            f"Position added: {ticker} {num_contracts}x @ {entry_price_cents}¢ "
            f"(total positions: {len(self.positions)})"
        )

    def remove_position(self, ticker: str, pnl_cents: int = 0) -> None:
        """
        Remove a position (closed/settled).

        Args:
            ticker: Ticker to remove
            pnl_cents: P&L for this position (positive = profit)
        """
        if ticker not in self.positions:
            logger.warning(f"Attempted to remove non-existent position: {ticker}")
            return

        position = self.positions.pop(ticker)

        # Remove from city/event tracking
        key = (position.city, position.event_date)
        if ticker in self.city_event_positions[key]:
            self.city_event_positions[key].remove(ticker)

        # Update P&L
        today = date.today()
        self.daily_pnl_cents[today] += pnl_cents
        self.total_pnl_cents += pnl_cents

        logger.info(
            f"Position removed: {ticker} P&L=${pnl_cents/100:.2f} "
            f"(total P&L: ${self.total_pnl_cents/100:.2f})"
        )

    def has_position(self, ticker: str) -> bool:
        """Check if we have an open position for ticker."""
        return ticker in self.positions

    def get_position_count(self) -> int:
        """Get total number of open positions."""
        return len(self.positions)

    def get_position_count_for_city_event(self, city: str, event_date: date) -> int:
        """Get number of positions for a specific city/event."""
        return len(self.city_event_positions.get((city, event_date), set()))

    def can_open_position(
        self,
        city: str,
        event_date: date,
        check_daily_loss: bool = True
    ) -> Tuple[bool, str]:
        """
        Check if new position can be opened based on limits.

        Args:
            city: City for new position
            event_date: Event date for new position
            check_daily_loss: Whether to check daily loss limit

        Returns:
            (can_open, reason)
        """
        # Check global position limit
        if self.get_position_count() >= self.config.max_total_positions:
            return False, f"Max total positions reached ({self.config.max_total_positions})"

        # Check per-city-event limit
        city_event_count = self.get_position_count_for_city_event(city, event_date)
        if city_event_count >= self.config.max_positions_per_city:
            return False, f"Max positions for {city}/{event_date} reached ({self.config.max_positions_per_city})"

        # Check daily loss limit
        if check_daily_loss:
            today = date.today()
            daily_pnl = self.daily_pnl_cents.get(today, 0)

            if daily_pnl < 0:  # Negative P&L = loss
                daily_loss_usd = abs(daily_pnl) / 100.0
                if daily_loss_usd >= self.config.max_daily_loss_usd:
                    return False, f"Daily loss limit reached (${daily_loss_usd:.2f} >= ${self.config.max_daily_loss_usd})"

        return True, "OK"

    def reset_daily_pnl(self) -> None:
        """Reset daily P&L (call at start of new trading day)."""
        # Keep last 7 days of history
        cutoff_date = date.today() - timedelta(days=7)
        self.daily_pnl_cents = {
            d: pnl
            for d, pnl in self.daily_pnl_cents.items()
            if d >= cutoff_date
        }

    def get_daily_pnl_usd(self, day: Optional[date] = None) -> float:
        """
        Get P&L for a specific day in USD.

        Args:
            day: Date to query (default: today)

        Returns:
            P&L in USD (positive = profit)
        """
        if day is None:
            day = date.today()

        pnl_cents = self.daily_pnl_cents.get(day, 0)
        return pnl_cents / 100.0

    def get_total_pnl_usd(self) -> float:
        """Get total P&L across all time in USD."""
        return self.total_pnl_cents / 100.0
