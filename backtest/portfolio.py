#!/usr/bin/env python3
"""
Portfolio simulator for Kalshi weather markets.

Tracks positions, executes trades with fees, settles markets, and calculates P&L.
Fee-aware and handles the full lifecycle from entry to settlement.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Literal, Tuple
from dataclasses import dataclass, field
import pandas as pd

from backtest.fees import taker_fee_cents, maker_fee_cents

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """
    Represents a position in a single market.

    YES contracts are the only tradeable side (NO implied as 100¢ - YES).
    """
    market_ticker: str
    contracts: int = 0          # Number of YES contracts held (can be negative for short)
    avg_entry_price: float = 0.0  # Average entry price in cents
    realized_pnl_cents: int = 0   # Cumulative realized P&L in cents
    fees_paid_cents: int = 0      # Cumulative fees paid in cents

    def is_flat(self) -> bool:
        """Check if position is flat (no contracts)."""
        return self.contracts == 0


@dataclass
class Trade:
    """Record of a single trade execution."""
    timestamp: datetime
    market_ticker: str
    side: Literal["buy", "sell"]
    contracts: int
    price_cents: int
    fee_type: Literal["taker", "maker"]
    fee_cents: int
    pnl_cents: Optional[int] = None  # Only set on exit trades
    # Diagnostic fields for edge analysis
    p_model: Optional[float] = None  # Model probability [0, 1]
    p_market: Optional[float] = None  # Market probability (mid / 100)
    edge_cents: Optional[float] = None  # Expected edge in cents
    spread_cents: Optional[int] = None  # Bid-ask spread
    time_to_close_minutes: Optional[float] = None  # Minutes until market close


@dataclass
class Settlement:
    """Record of a market settlement."""
    timestamp: datetime
    market_ticker: str
    result: Literal["YES", "NO"]
    contracts: int
    avg_entry_price: float
    payout_cents: int
    pnl_cents: int


@dataclass
class LimitOrder:
    """Pending limit order (maker)."""
    order_id: int
    timestamp: datetime
    market_ticker: str
    side: Literal["buy", "sell"]
    contracts: int
    limit_price_cents: int  # Max buy price or min sell price
    # Diagnostic fields
    p_model: Optional[float] = None
    p_market: Optional[float] = None
    edge_cents: Optional[float] = None
    spread_cents: Optional[int] = None
    time_to_close_minutes: Optional[float] = None


class Portfolio:
    """
    Portfolio simulator with position tracking and P&L calculation.

    Handles:
    - Cash management (starting capital)
    - Position tracking (per market)
    - Trade execution with fees (taker/maker)
    - Market settlement (YES=100¢, NO=0¢)
    - P&L calculation (realized + unrealized)
    """

    def __init__(self, initial_cash_cents: int = 10_000_00):
        """
        Initialize portfolio.

        Args:
            initial_cash_cents: Starting cash in cents (default: $10,000)
        """
        self.initial_cash_cents = initial_cash_cents
        self.cash_cents = initial_cash_cents
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.settlements: List[Settlement] = []
        self.open_limit_orders: List[LimitOrder] = []
        self._next_order_id = 1

        logger.info(f"Portfolio initialized with ${initial_cash_cents/100:,.2f}")

    def get_position(self, market_ticker: str) -> Position:
        """Get position for a market (creates if doesn't exist)."""
        if market_ticker not in self.positions:
            self.positions[market_ticker] = Position(market_ticker=market_ticker)
        return self.positions[market_ticker]

    def execute_trade(
        self,
        timestamp: datetime,
        market_ticker: str,
        side: Literal["buy", "sell"],
        contracts: int,
        price_cents: int,
        fee_type: Literal["taker", "maker"] = "taker",
        # Diagnostic fields for edge analysis
        p_model: Optional[float] = None,
        p_market: Optional[float] = None,
        edge_cents: Optional[float] = None,
        spread_cents: Optional[int] = None,
        time_to_close_minutes: Optional[float] = None,
    ) -> bool:
        """
        Execute a trade (buy or sell YES contracts).

        Args:
            timestamp: Trade timestamp
            market_ticker: Market identifier
            side: "buy" or "sell"
            contracts: Number of contracts (positive)
            price_cents: Execution price in cents (0-100)
            fee_type: "taker" or "maker"
            p_model: Model probability estimate [0, 1] (for diagnostics)
            p_market: Market probability from mid-price (for diagnostics)
            edge_cents: Expected edge in cents (for diagnostics)
            spread_cents: Bid-ask spread (for diagnostics)
            time_to_close_minutes: Minutes until market close (for diagnostics)

        Returns:
            True if trade executed, False if insufficient cash
        """
        if contracts <= 0:
            raise ValueError(f"Contracts must be positive, got {contracts}")

        # Calculate fee
        if fee_type == "taker":
            fee = taker_fee_cents(contracts, price_cents)
        else:
            fee = maker_fee_cents(contracts, price_cents)

        # Calculate cash impact
        contract_cost = contracts * price_cents

        if side == "buy":
            total_cost = contract_cost + fee
            if total_cost > self.cash_cents:
                logger.warning(
                    f"Insufficient cash for buy: need {total_cost/100:.2f}, "
                    f"have {self.cash_cents/100:.2f}"
                )
                return False

            self.cash_cents -= total_cost

            # Update position
            pos = self.get_position(market_ticker)

            # Update average entry price (weighted average)
            if pos.contracts > 0:
                # Adding to long position
                total_contracts = pos.contracts + contracts
                pos.avg_entry_price = (
                    (pos.avg_entry_price * pos.contracts + price_cents * contracts)
                    / total_contracts
                )
                pos.contracts = total_contracts
            elif pos.contracts < 0:
                # Covering short position
                if contracts <= abs(pos.contracts):
                    # Partial cover - realize P&L
                    pnl = contracts * (pos.avg_entry_price - price_cents)
                    pos.realized_pnl_cents += pnl
                    pos.contracts += contracts
                    if pos.contracts == 0:
                        pos.avg_entry_price = 0.0
                else:
                    # Full cover + go long - realize P&L on short, reset for long
                    cover_contracts = abs(pos.contracts)
                    pnl = cover_contracts * (pos.avg_entry_price - price_cents)
                    pos.realized_pnl_cents += pnl

                    # Remaining contracts start new long position
                    remaining = contracts - cover_contracts
                    pos.contracts = remaining
                    pos.avg_entry_price = price_cents
            else:
                # New long position
                pos.contracts = contracts
                pos.avg_entry_price = price_cents

            pos.fees_paid_cents += fee

        else:  # sell
            proceeds = contract_cost - fee
            self.cash_cents += proceeds

            # Update position
            pos = self.get_position(market_ticker)

            if pos.contracts > 0:
                # Closing/reducing long position
                if contracts <= pos.contracts:
                    # Partial or full close - realize P&L
                    pnl = contracts * (price_cents - pos.avg_entry_price)
                    pos.realized_pnl_cents += pnl
                    pos.contracts -= contracts
                    if pos.contracts == 0:
                        pos.avg_entry_price = 0.0
                else:
                    # Close long + go short - realize P&L on long, reset for short
                    long_contracts = pos.contracts
                    pnl = long_contracts * (price_cents - pos.avg_entry_price)
                    pos.realized_pnl_cents += pnl

                    # Remaining contracts start new short position
                    remaining = contracts - long_contracts
                    pos.contracts = -remaining
                    pos.avg_entry_price = price_cents
            elif pos.contracts < 0:
                # Adding to short position
                total_contracts = abs(pos.contracts) + contracts
                pos.avg_entry_price = (
                    (pos.avg_entry_price * abs(pos.contracts) + price_cents * contracts)
                    / total_contracts
                )
                pos.contracts = -total_contracts
            else:
                # New short position
                pos.contracts = -contracts
                pos.avg_entry_price = price_cents

            pos.fees_paid_cents += fee

        # Record trade
        # Calculate P&L if closing/reducing position
        trade_pnl = None
        if side == "sell" and market_ticker in self.positions:
            # We just realized some P&L
            trade_pnl = pos.realized_pnl_cents

        trade = Trade(
            timestamp=timestamp,
            market_ticker=market_ticker,
            side=side,
            contracts=contracts,
            price_cents=price_cents,
            fee_type=fee_type,
            fee_cents=fee,
            pnl_cents=trade_pnl,
            # Diagnostic fields
            p_model=p_model,
            p_market=p_market,
            edge_cents=edge_cents,
            spread_cents=spread_cents,
            time_to_close_minutes=time_to_close_minutes,
        )
        self.trades.append(trade)

        logger.debug(
            f"Trade: {side.upper()} {contracts} {market_ticker} @ {price_cents}¢ "
            f"({fee_type} fee: {fee}¢), cash: ${self.cash_cents/100:,.2f}"
        )

        return True

    def place_limit_order(
        self,
        timestamp: datetime,
        market_ticker: str,
        side: Literal["buy", "sell"],
        contracts: int,
        limit_price_cents: int,
        # Diagnostic fields
        p_model: Optional[float] = None,
        p_market: Optional[float] = None,
        edge_cents: Optional[float] = None,
        spread_cents: Optional[int] = None,
        time_to_close_minutes: Optional[float] = None,
    ) -> int:
        """
        Place a limit order (maker).

        Args:
            timestamp: Order placement time
            market_ticker: Market identifier
            side: "buy" or "sell"
            contracts: Number of contracts
            limit_price_cents: Max buy price or min sell price
            Diagnostic fields: same as execute_trade

        Returns:
            order_id for tracking/cancellation
        """
        order = LimitOrder(
            order_id=self._next_order_id,
            timestamp=timestamp,
            market_ticker=market_ticker,
            side=side,
            contracts=contracts,
            limit_price_cents=limit_price_cents,
            p_model=p_model,
            p_market=p_market,
            edge_cents=edge_cents,
            spread_cents=spread_cents,
            time_to_close_minutes=time_to_close_minutes,
        )
        self.open_limit_orders.append(order)
        self._next_order_id += 1

        logger.debug(
            f"Limit order placed: {side.upper()} {contracts} {market_ticker} @ ≤{limit_price_cents}¢ "
            f"(order_id={order.order_id})"
        )

        return order.order_id

    def check_limit_fills(
        self,
        timestamp: datetime,
        market_ticker: str,
        bid_cents: int,
        ask_cents: int,
    ) -> List[int]:
        """
        Check if any pending limit orders can fill based on current market prices.

        Fill logic:
        - BUY limit fills if ask <= limit_price (we can buy at our price or better)
        - SELL limit fills if bid >= limit_price (we can sell at our price or better)

        Args:
            timestamp: Current time
            market_ticker: Market to check
            bid_cents: Current best bid
            ask_cents: Current best ask

        Returns:
            List of filled order_ids
        """
        filled_order_ids = []

        for order in list(self.open_limit_orders):  # Copy list to allow modification
            if order.market_ticker != market_ticker:
                continue

            filled = False
            fill_price = 0

            if order.side == "buy":
                # BUY limit fills if ask <= limit (we can buy at our price or better)
                if ask_cents <= order.limit_price_cents:
                    filled = True
                    fill_price = order.limit_price_cents  # Fill at limit (maker)

            else:  # sell
                # SELL limit fills if bid >= limit (we can sell at our price or better)
                if bid_cents >= order.limit_price_cents:
                    filled = True
                    fill_price = order.limit_price_cents  # Fill at limit (maker)

            if filled:
                # Execute as maker trade
                success = self.execute_trade(
                    timestamp=timestamp,
                    market_ticker=order.market_ticker,
                    side=order.side,
                    contracts=order.contracts,
                    price_cents=fill_price,
                    fee_type="maker",
                    p_model=order.p_model,
                    p_market=order.p_market,
                    edge_cents=order.edge_cents,
                    spread_cents=order.spread_cents,
                    time_to_close_minutes=order.time_to_close_minutes,
                )

                if success:
                    filled_order_ids.append(order.order_id)
                    self.open_limit_orders.remove(order)
                    logger.debug(f"Limit order filled: order_id={order.order_id} @ {fill_price}¢ (maker)")

        return filled_order_ids

    def cancel_limit_order(self, order_id: int) -> bool:
        """
        Cancel a pending limit order.

        Returns:
            True if order was found and canceled
        """
        for order in self.open_limit_orders:
            if order.order_id == order_id:
                self.open_limit_orders.remove(order)
                logger.debug(f"Limit order canceled: order_id={order_id}")
                return True
        return False

    def cancel_all_limit_orders(self, market_ticker: Optional[str] = None):
        """
        Cancel all pending limit orders, optionally filtered by market.

        Args:
            market_ticker: If provided, only cancel orders for this market
        """
        if market_ticker:
            self.open_limit_orders = [
                o for o in self.open_limit_orders if o.market_ticker != market_ticker
            ]
            logger.debug(f"Canceled all limit orders for {market_ticker}")
        else:
            count = len(self.open_limit_orders)
            self.open_limit_orders.clear()
            logger.debug(f"Canceled {count} limit orders")

    def settle_market(
        self,
        timestamp: datetime,
        market_ticker: str,
        result: Literal["YES", "NO"],
    ) -> Optional[Settlement]:
        """
        Settle a market at expiration.

        YES contracts pay 100¢, NO contracts pay 0¢ (no settlement fee).

        Args:
            timestamp: Settlement timestamp
            market_ticker: Market identifier
            result: "YES" or "NO"

        Returns:
            Settlement record if position existed, None otherwise
        """
        if market_ticker not in self.positions:
            return None

        pos = self.positions[market_ticker]

        if pos.is_flat():
            return None

        # Calculate settlement payout
        if result == "YES":
            payout_per_contract = 100
        else:
            payout_per_contract = 0

        # Positive contracts = long YES (collect payout if YES wins)
        # Negative contracts = short YES (pay payout if YES wins)
        payout_cents = pos.contracts * payout_per_contract

        # Calculate P&L
        # For long: P&L = payout - (avg_entry_price * contracts)
        # For short: P&L = (avg_entry_price * contracts) - payout
        cost_basis_cents = int(pos.avg_entry_price * abs(pos.contracts))

        if pos.contracts > 0:
            # Long position
            settlement_pnl = payout_cents - cost_basis_cents
        else:
            # Short position
            settlement_pnl = cost_basis_cents - abs(payout_cents)

        # Add to cash
        self.cash_cents += payout_cents

        # Update position realized P&L
        pos.realized_pnl_cents += settlement_pnl

        # Record settlement
        settlement = Settlement(
            timestamp=timestamp,
            market_ticker=market_ticker,
            result=result,
            contracts=pos.contracts,
            avg_entry_price=pos.avg_entry_price,
            payout_cents=payout_cents,
            pnl_cents=settlement_pnl,
        )
        self.settlements.append(settlement)

        logger.info(
            f"Settlement: {market_ticker} → {result}, "
            f"position: {pos.contracts} @ {pos.avg_entry_price:.1f}¢, "
            f"payout: {payout_cents}¢, P&L: {settlement_pnl}¢"
        )

        # Close position
        pos.contracts = 0
        pos.avg_entry_price = 0.0

        return settlement

    def get_equity_cents(self, mark_prices: Optional[Dict[str, int]] = None) -> int:
        """
        Calculate total equity (cash + unrealized P&L).

        Args:
            mark_prices: Dict of market_ticker → current price in cents
                        If None, uses avg_entry_price (conservative)

        Returns:
            Total equity in cents
        """
        unrealized_pnl = 0

        for ticker, pos in self.positions.items():
            if pos.is_flat():
                continue

            # Get mark price
            if mark_prices and ticker in mark_prices:
                mark_price = mark_prices[ticker]
            else:
                mark_price = pos.avg_entry_price  # Conservative: no P&L

            # Calculate unrealized P&L
            if pos.contracts > 0:
                # Long: (mark - entry) * contracts
                unrealized_pnl += (mark_price - pos.avg_entry_price) * pos.contracts
            else:
                # Short: (entry - mark) * |contracts|
                unrealized_pnl += (pos.avg_entry_price - mark_price) * abs(pos.contracts)

        return int(self.cash_cents + unrealized_pnl)

    def get_total_pnl_cents(self) -> int:
        """
        Calculate total realized P&L (all settled markets).

        Returns:
            Total P&L in cents
        """
        total_pnl = 0

        for pos in self.positions.values():
            total_pnl += pos.realized_pnl_cents

        return total_pnl

    def get_total_fees_cents(self) -> int:
        """
        Calculate total fees paid across all markets.

        Returns:
            Total fees in cents
        """
        return sum(pos.fees_paid_cents for pos in self.positions.values())

    def get_summary(self) -> Dict:
        """
        Get portfolio summary statistics.

        Returns:
            Dict with equity, P&L, fees, returns, etc.
        """
        equity = self.get_equity_cents()
        total_pnl = self.get_total_pnl_cents()
        total_fees = self.get_total_fees_cents()
        total_return = (equity - self.initial_cash_cents) / self.initial_cash_cents

        return {
            "initial_cash_cents": self.initial_cash_cents,
            "cash_cents": self.cash_cents,
            "equity_cents": equity,
            "total_pnl_cents": total_pnl,
            "total_fees_cents": total_fees,
            "gross_pnl_cents": total_pnl + total_fees,
            "total_return_pct": total_return * 100,
            "num_trades": len(self.trades),
            "num_settlements": len(self.settlements),
            "num_open_positions": sum(1 for p in self.positions.values() if not p.is_flat()),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert trade history to DataFrame for analysis.

        Returns:
            DataFrame with columns: timestamp, market_ticker, side, contracts,
                                   price_cents, fee_type, fee_cents, pnl_cents
        """
        if not self.trades:
            return pd.DataFrame()

        records = []
        for trade in self.trades:
            records.append({
                "timestamp": trade.timestamp,
                "market_ticker": trade.market_ticker,
                "side": trade.side,
                "contracts": trade.contracts,
                "price_cents": trade.price_cents,
                "fee_type": trade.fee_type,
                "fee_cents": trade.fee_cents,
                "pnl_cents": trade.pnl_cents,
            })

        return pd.DataFrame(records)


def main():
    """Demo: Simple portfolio simulation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("\n" + "="*60)
    print("Portfolio Simulator Demo")
    print("="*60 + "\n")

    # Initialize portfolio with $1,000
    portfolio = Portfolio(initial_cash_cents=1000_00)

    now = datetime.now()

    # Trade 1: Buy 10 contracts at 50¢ (taker)
    print("Trade 1: Buy 10 YES @ 50¢")
    portfolio.execute_trade(
        timestamp=now,
        market_ticker="KXHIGHCHI-25NOV10-B70-71",
        side="buy",
        contracts=10,
        price_cents=50,
        fee_type="taker",
    )
    print(f"  Cash: ${portfolio.cash_cents/100:.2f}\n")

    # Trade 2: Sell 5 contracts at 60¢ (take profit)
    print("Trade 2: Sell 5 YES @ 60¢")
    portfolio.execute_trade(
        timestamp=now,
        market_ticker="KXHIGHCHI-25NOV10-B70-71",
        side="sell",
        contracts=5,
        price_cents=60,
        fee_type="taker",
    )
    pos = portfolio.get_position("KXHIGHCHI-25NOV10-B70-71")
    print(f"  Cash: ${portfolio.cash_cents/100:.2f}")
    print(f"  Position: {pos.contracts} contracts @ {pos.avg_entry_price:.1f}¢")
    print(f"  Realized P&L: {pos.realized_pnl_cents}¢\n")

    # Settlement: Market resolves YES
    print("Settlement: Market resolves YES")
    settlement = portfolio.settle_market(
        timestamp=now,
        market_ticker="KXHIGHCHI-25NOV10-B70-71",
        result="YES",
    )

    if settlement:
        print(f"  Payout: {settlement.payout_cents}¢")
        print(f"  Settlement P&L: {settlement.pnl_cents}¢")

    # Summary
    print("\n" + "="*60)
    print("Portfolio Summary")
    print("="*60)

    summary = portfolio.get_summary()
    print(f"Initial cash:    ${summary['initial_cash_cents']/100:>10,.2f}")
    print(f"Final equity:    ${summary['equity_cents']/100:>10,.2f}")
    print(f"Total P&L:       ${summary['total_pnl_cents']/100:>10,.2f}")
    print(f"Total fees:      ${summary['total_fees_cents']/100:>10,.2f}")
    print(f"Gross P&L:       ${summary['gross_pnl_cents']/100:>10,.2f}")
    print(f"Total return:    {summary['total_return_pct']:>10.2f}%")
    print(f"Num trades:      {summary['num_trades']:>10}")
    print(f"Num settlements: {summary['num_settlements']:>10}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
