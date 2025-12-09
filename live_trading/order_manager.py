"""
Order manager with maker→taker conversion.

Tracks pending limit orders and converts them to market orders
after a volume-weighted timeout period expires.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, Optional, Deque, Tuple
from uuid import UUID, uuid4

from live_trading.config import TradingConfig
from src.trading.fees import classify_liquidity_role

logger = logging.getLogger(__name__)


@dataclass
class PendingOrder:
    """Represents a pending limit order awaiting fill or timeout."""

    order_id: UUID
    ticker: str
    city: str
    event_date: str
    side: str  # "yes" or "no"
    action: str  # "buy" or "sell"
    num_contracts: int
    maker_price_cents: int
    placed_at: datetime
    maker_timeout_sec: int

    @property
    def timeout_at(self) -> datetime:
        """Time when maker order should convert to taker."""
        return self.placed_at + timedelta(seconds=self.maker_timeout_sec)

    @property
    def should_convert_to_taker(self) -> bool:
        """Check if timeout has been reached."""
        return datetime.now() >= self.timeout_at

    @property
    def age_seconds(self) -> int:
        """Seconds since order was placed."""
        return int((datetime.now() - self.placed_at).total_seconds())


class OrderManager:
    """
    Manages order lifecycle with maker→taker conversion.

    Features:
    - Track all pending limit orders
    - Background task checks for timeouts
    - Converts to market order after timeout
    - Handles fill notifications
    """

    def __init__(self, kalshi_client, config: TradingConfig):
        """
        Initialize order manager.

        Args:
            kalshi_client: KalshiClient instance for order operations
            config: Trading configuration
        """
        self.client = kalshi_client
        self.config = config

        self.pending_orders: Dict[UUID, PendingOrder] = {}
        self._timeout_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start background timeout checker task."""
        if self._timeout_task is not None:
            logger.warning("OrderManager already started")
            return

        self._running = True
        self._timeout_task = asyncio.create_task(self._timeout_checker_loop())
        logger.info("OrderManager started (timeout checker active)")

    async def stop(self) -> None:
        """Stop background task."""
        self._running = False

        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass

            self._timeout_task = None

        logger.info("OrderManager stopped")

    def track_order(self, order: PendingOrder) -> None:
        """
        Add order to tracking.

        Args:
            order: PendingOrder to track
        """
        self.pending_orders[order.order_id] = order
        logger.info(
            f"Tracking order {order.order_id}: "
            f"{order.action} {order.side} @ {order.maker_price_cents}¢ "
            f"(timeout in {order.maker_timeout_sec}s)"
        )

    def place_order(
        self,
        ticker: str,
        side: str,
        action: str,
        num_contracts: int,
        price_cents: int,
        city: str,
        event_date: date,
        yes_bid: int,
        yes_ask: int,
        maker_timeout_sec: Optional[int] = None,
    ) -> UUID:
        """
        Place an order and track it if maker.

        This is the centralized order placement method that:
        - Generates standardized client_order_id
        - Calls Kalshi API to create order
        - Automatically tracks maker orders for timeout conversion
        - Returns order_id for reference

        Args:
            ticker: Market ticker
            side: 'yes' or 'no'
            action: 'buy' or 'sell'
            num_contracts: Number of contracts
            price_cents: Price in cents
            city: City for tracking
            event_date: Event date for tracking
            yes_bid: Current best YES bid (for role detection)
            yes_ask: Current best YES ask (for role detection)
            maker_timeout_sec: Optional timeout override

        Returns:
            UUID of created order

        Raises:
            Exception if order creation fails
        """
        # Generate standardized client_order_id
        client_order_id = f"om-{city}-{event_date}-{uuid4().hex[:8]}"

        # Determine if this will be maker or taker
        role = classify_liquidity_role(side, action, price_cents, yes_bid, yes_ask)
        is_maker = (role == "maker")

        # Place order via Kalshi API
        result = self.client.create_order(
            ticker=ticker,
            side=side,
            action=action,
            count=num_contracts,
            order_type="limit",
            yes_price=price_cents if side == "yes" else None,
            no_price=price_cents if side == "no" else None,
            client_order_id=client_order_id,
        )

        # Extract order_id from response
        order_id_str = result.get("order", {}).get("order_id")
        if not order_id_str:
            raise ValueError(f"No order_id in API response: {result}")

        order_id = UUID(order_id_str)

        logger.info(
            f"Order placed: {order_id} ({role}) "
            f"{action} {side} {num_contracts}x @ {price_cents}¢"
        )

        # Track maker orders for timeout conversion
        if is_maker:
            if maker_timeout_sec is None:
                # Use default timeout from config
                maker_timeout_sec = self.config.maker_timeout_base_seconds

            pending_order = PendingOrder(
                order_id=order_id,
                ticker=ticker,
                city=city,
                event_date=str(event_date),
                side=side,
                action=action,
                num_contracts=num_contracts,
                maker_price_cents=price_cents,
                placed_at=datetime.now(),
                maker_timeout_sec=maker_timeout_sec,
            )
            self.track_order(pending_order)

        return order_id

    def on_fill(self, order_id: UUID, fill_data: dict) -> None:
        """
        Handle fill notification from WebSocket.

        Args:
            order_id: Order that was filled
            fill_data: Fill message payload
        """
        if order_id in self.pending_orders:
            order = self.pending_orders.pop(order_id)
            logger.info(
                f"Order filled (removed from tracking): {order_id} "
                f"(age={order.age_seconds}s)"
            )
        else:
            logger.debug(f"Fill for non-tracked order: {order_id}")

    def on_cancel(self, order_id: UUID) -> None:
        """
        Handle order cancellation.

        Args:
            order_id: Order that was cancelled
        """
        if order_id in self.pending_orders:
            order = self.pending_orders.pop(order_id)
            logger.info(f"Order cancelled (removed from tracking): {order_id}")

    async def _timeout_checker_loop(self) -> None:
        """
        Background task: check for maker timeouts every 10 seconds.

        Converts any orders that have exceeded their maker timeout
        into market (taker) orders.
        """
        while self._running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                # Check all pending orders for timeout
                orders_to_convert = [
                    order
                    for order in self.pending_orders.values()
                    if order.should_convert_to_taker
                ]

                for order in orders_to_convert:
                    await self._convert_to_taker(order)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Timeout checker error: {e}", exc_info=True)

    async def _convert_to_taker(self, order: PendingOrder) -> None:
        """
        Convert maker order to taker (market order).

        Steps:
        1. Cancel the limit order
        2. Update DB status
        3. Submit a market order for the same position

        Args:
            order: PendingOrder to convert
        """
        try:
            logger.info(
                f"Converting to taker: {order.order_id} "
                f"(timeout reached after {order.age_seconds}s)"
            )

            # Cancel the limit order
            try:
                self.client.cancel_order(str(order.order_id))
                logger.info(f"Cancelled maker order: {order.order_id}")
            except Exception as e:
                logger.warning(f"Failed to cancel order {order.order_id}: {e}")
                # Continue anyway - may already be filled

            # Submit market order (taker)
            result = self.client.create_order(
                ticker=order.ticker,
                side=order.side,
                action=order.action,
                count=order.num_contracts,
                order_type="market",  # Taker order
            )

            new_order_id = result.get("order", {}).get("order_id")
            logger.info(
                f"Taker order placed: {new_order_id} "
                f"(converted from {order.order_id})"
            )

            # Remove from tracking
            if order.order_id in self.pending_orders:
                del self.pending_orders[order.order_id]

            # Note: DB status will be updated when fill arrives via WebSocket

        except Exception as e:
            logger.error(
                f"Failed to convert order {order.order_id} to taker: {e}",
                exc_info=True
            )

    def get_pending_count(self) -> int:
        """Get number of pending orders."""
        return len(self.pending_orders)

    def get_pending_for_ticker(self, ticker: str) -> list[PendingOrder]:
        """
        Get pending orders for a specific ticker.

        Args:
            ticker: Market ticker

        Returns:
            List of PendingOrder objects
        """
        return [
            order
            for order in self.pending_orders.values()
            if order.ticker == ticker
        ]


class VolumeTracker:
    """
    Track recent trading volume for timeout calculation.

    Maintains a rolling window of trade volume per ticker.
    High volume → longer maker timeout (more likely to fill)
    Low volume → shorter timeout (convert to taker faster)
    """

    def __init__(self, lookback_minutes: int = 30):
        """
        Initialize volume tracker.

        Args:
            lookback_minutes: How far back to look for volume
        """
        self.lookback = timedelta(minutes=lookback_minutes)
        self.volumes: Dict[str, Deque[Tuple[datetime, int]]] = {}  # ticker → [(ts, contracts)]

    def add_trade(self, ticker: str, contracts: int, trade_time: Optional[datetime] = None) -> None:
        """
        Record a trade.

        Args:
            ticker: Market ticker
            contracts: Number of contracts traded
            trade_time: Trade timestamp (default: now)
        """
        if trade_time is None:
            trade_time = datetime.now()

        if ticker not in self.volumes:
            self.volumes[ticker] = deque()

        self.volumes[ticker].append((trade_time, contracts))

        # Prune old entries
        self._prune_old_trades(ticker)

    def get_recent_volume(self, ticker: str) -> int:
        """
        Get total volume in lookback window.

        Args:
            ticker: Market ticker

        Returns:
            Total contracts traded in window
        """
        if ticker not in self.volumes:
            return 0

        # Prune old entries first
        self._prune_old_trades(ticker)

        # Sum contracts
        total = sum(contracts for _, contracts in self.volumes[ticker])
        return total

    def _prune_old_trades(self, ticker: str) -> None:
        """
        Remove trades outside lookback window.

        Args:
            ticker: Market ticker
        """
        if ticker not in self.volumes:
            return

        cutoff = datetime.now() - self.lookback

        # Remove old entries from left of deque
        while self.volumes[ticker] and self.volumes[ticker][0][0] < cutoff:
            self.volumes[ticker].popleft()

    def compute_maker_timeout(
        self,
        ticker: str,
        config: TradingConfig
    ) -> int:
        """
        Compute maker timeout based on recent volume.

        Args:
            ticker: Market ticker
            config: Trading configuration

        Returns:
            Timeout in seconds
        """
        volume = self.get_recent_volume(ticker)

        # Use config's built-in method
        return config.compute_maker_timeout_sec(volume)
