"""
Incremental order executor for handling large trades.

Splits large orders into smaller chunks to respect liquidity constraints
and executes them sequentially with optional user confirmation.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple
from datetime import datetime, date

from live_trading.config import TradingConfig
from live_trading.order_manager import OrderManager
from live_trading.websocket.order_book import OrderBookManager
from src.trading.fees import classify_liquidity_role, taker_fee_total, maker_fee_total

logger = logging.getLogger(__name__)


@dataclass
class OrderChunk:
    """A single chunk of a larger order."""

    chunk_index: int
    total_chunks: int
    price_cents: int
    num_contracts: int
    cost_usd: float
    is_maker: bool
    fee_cents: int

    def __str__(self) -> str:
        role = "Maker" if self.is_maker else "Taker"
        return (
            f"Chunk {self.chunk_index}/{self.total_chunks}: "
            f"{self.num_contracts} contracts @ {self.price_cents}¢ "
            f"(${self.cost_usd:.2f}, {role}, fee={self.fee_cents}¢)"
        )


class IncrementalOrderExecutor:
    """
    Executor for splitting and executing large orders incrementally.

    Features:
    - Splits orders into chunks based on max bet size
    - Sequential execution with delays
    - Optional per-chunk confirmation
    - Callback support for UI updates
    """

    def __init__(
        self,
        order_manager: OrderManager,
        config: TradingConfig,
        order_book_mgr: Optional[OrderBookManager] = None,
    ):
        """
        Initialize incremental executor.

        Args:
            order_manager: OrderManager for centralized order placement
            config: Trading configuration
            order_book_mgr: Optional order book manager for liquidity checks
        """
        self.order_manager = order_manager
        self.config = config
        self.order_book_mgr = order_book_mgr

    def plan_incremental_entry(
        self,
        ticker: str,
        target_usd: float,
        side: str,
        action: str,
        yes_bid: int,
        yes_ask: int,
    ) -> List[OrderChunk]:
        """
        Plan how to split a large order into chunks.

        Args:
            ticker: Market ticker
            target_usd: Total amount to invest
            side: 'yes' or 'no'
            action: 'buy' or 'sell'
            yes_bid: Current YES bid price (cents)
            yes_ask: Current YES ask price (cents)

        Returns:
            List of OrderChunk objects describing execution plan
        """
        chunks = []
        remaining_usd = target_usd
        max_chunk_usd = self.config.max_bet_per_trade_usd
        chunk_index = 1

        # Determine price based on action
        # For buy: use ask (worst price)
        # For sell: use bid (worst price)
        if action == "buy":
            price_cents = yes_ask if side == "yes" else (100 - yes_bid)
        else:  # sell
            price_cents = yes_bid if side == "yes" else (100 - yes_ask)

        # Calculate total chunks needed
        total_chunks = int((target_usd + max_chunk_usd - 0.01) / max_chunk_usd)

        while remaining_usd > 0.01:  # Avoid floating point issues
            # Size this chunk
            chunk_usd = min(remaining_usd, max_chunk_usd)

            # Calculate contracts for this chunk
            # cost = num_contracts × price / 100
            num_contracts = int(chunk_usd / (price_cents / 100.0))

            if num_contracts <= 0:
                break  # Can't afford even 1 contract

            # Actual cost
            cost_usd = num_contracts * price_cents / 100.0

            # Determine if maker or taker
            role = classify_liquidity_role(side, action, price_cents, yes_bid, yes_ask)
            is_maker = (role == "maker")

            # Calculate fee
            if is_maker:
                fee_cents = maker_fee_total(price_cents, num_contracts)
            else:
                fee_cents = taker_fee_total(price_cents, num_contracts)

            chunks.append(OrderChunk(
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                price_cents=price_cents,
                num_contracts=num_contracts,
                cost_usd=cost_usd,
                is_maker=is_maker,
                fee_cents=fee_cents,
            ))

            remaining_usd -= cost_usd
            chunk_index += 1

        logger.info(
            f"Planned {len(chunks)} chunks for ${target_usd:.2f} "
            f"(max ${max_chunk_usd:.2f} per chunk)"
        )

        return chunks

    async def execute_incremental_order(
        self,
        ticker: str,
        city: str,
        event_date: date,
        side: str,
        action: str,
        yes_bid: int,
        yes_ask: int,
        chunks: List[OrderChunk],
        confirm_each: bool = False,
        callback: Optional[Callable[[str, OrderChunk], bool]] = None,
        delay_sec: float = 0.5,
    ) -> List[Tuple[str, OrderChunk]]:
        """
        Execute order chunks sequentially using OrderManager.

        Args:
            ticker: Market ticker
            city: City identifier
            event_date: Event settlement date
            side: 'yes' or 'no'
            action: 'buy' or 'sell'
            yes_bid: Current YES bid price
            yes_ask: Current YES ask price
            chunks: List of OrderChunk objects to execute
            confirm_each: If True, require confirmation for each chunk
            callback: Optional callback(message, chunk) → should_continue
            delay_sec: Delay between chunks (seconds)

        Returns:
            List of (order_id, chunk) tuples for successfully placed orders
        """
        placed_orders = []

        for chunk in chunks:
            try:
                # Confirmation check
                if confirm_each and callback:
                    message = f"Place {chunk}?"
                    should_continue = await callback(message, chunk)
                    if not should_continue:
                        logger.info(f"User cancelled at chunk {chunk.chunk_index}")
                        break

                # Log placement
                logger.info(f"Placing {chunk}")

                # Place order via OrderManager (centralized!)
                order_id = self.order_manager.place_order(
                    ticker=ticker,
                    side=side,
                    action=action,
                    num_contracts=chunk.num_contracts,
                    price_cents=chunk.price_cents,
                    city=city,
                    event_date=event_date,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                )

                placed_orders.append((str(order_id), chunk))
                logger.info(f"✓ Placed order {order_id} for chunk {chunk.chunk_index}")

                # Callback for UI update
                if callback and not confirm_each:
                    await callback(f"Placed {chunk}", chunk)

                # Delay before next chunk
                if chunk.chunk_index < chunk.total_chunks:
                    await asyncio.sleep(delay_sec)

            except Exception as e:
                logger.error(f"Failed to place chunk {chunk.chunk_index}: {e}", exc_info=True)
                # Continue with remaining chunks or stop?
                # For now, continue
                continue

        logger.info(
            f"Completed: {len(placed_orders)}/{len(chunks)} chunks placed "
            f"({len(chunks) - len(placed_orders)} failed/cancelled)"
        )

        return placed_orders

    def get_total_cost(self, chunks: List[OrderChunk]) -> Tuple[float, int]:
        """
        Calculate total cost and fees for chunks.

        Args:
            chunks: List of OrderChunk objects

        Returns:
            (total_cost_usd, total_fee_cents)
        """
        total_cost = sum(chunk.cost_usd for chunk in chunks)
        total_fee_cents = sum(chunk.fee_cents for chunk in chunks)
        return total_cost, total_fee_cents

    def estimate_fill_probability(self, ticker: str, is_maker: bool) -> float:
        """
        Estimate probability of maker order filling.

        Uses order book data if available, otherwise returns default.

        Args:
            ticker: Market ticker
            is_maker: Whether order is maker (limit) or taker (market)

        Returns:
            Fill probability (0.0 - 1.0)
        """
        if not is_maker:
            return 1.0  # Taker always fills (in theory)

        # TODO: Use order book depth to estimate fill probability
        # For now, return conservative default
        return 0.6

    def summary(self, chunks: List[OrderChunk]) -> str:
        """
        Generate human-readable summary of execution plan.

        Args:
            chunks: List of OrderChunk objects

        Returns:
            Multi-line summary string
        """
        if not chunks:
            return "No chunks to execute"

        total_cost, total_fee_cents = self.get_total_cost(chunks)
        total_contracts = sum(chunk.num_contracts for chunk in chunks)

        maker_chunks = sum(1 for chunk in chunks if chunk.is_maker)
        taker_chunks = len(chunks) - maker_chunks

        summary_lines = [
            f"Execution Plan: {len(chunks)} chunk(s)",
            f"  Total contracts: {total_contracts}",
            f"  Total cost: ${total_cost:.2f}",
            f"  Total fees: {total_fee_cents}¢ (${total_fee_cents/100:.2f})",
            f"  Maker chunks: {maker_chunks}",
            f"  Taker chunks: {taker_chunks}",
        ]

        return "\n".join(summary_lines)
