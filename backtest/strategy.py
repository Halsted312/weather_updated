#!/usr/bin/env python3
"""
Strategy interface for Kalshi weather backtesting.

Defines abstract base class for trading strategies with entry/exit logic,
position sizing, and signal generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
from datetime import datetime
import pandas as pd


@dataclass
class Signal:
    """Trading signal for a single market at a specific time."""
    timestamp: datetime
    market_ticker: str
    action: Literal["buy", "sell", "hold", "close"]
    edge: float  # Expected edge in cents (positive = favorable)
    confidence: float  # Probability estimate [0, 1]
    size_fraction: float  # Fraction of bankroll to risk [0, 1]
    reason: str  # Human-readable explanation
    # Diagnostic fields for trade attribution
    spread_cents: int = 0  # Bid-ask spread at signal time
    p_market: float = 0.0  # Market probability (mid / 100)
    time_to_close_minutes: float = 0.0  # Minutes until market close
    price_cents: int = 50  # Execution price assumption (updated by strategy if known)


@dataclass
class Position:
    """Current position in a market."""
    market_ticker: str
    contracts: int  # Positive = long, negative = short, 0 = flat
    avg_entry_price: float  # Average entry price in cents
    entry_time: datetime
    unrealized_pnl_cents: float = 0.0


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Subclasses must implement:
    - generate_signals(): Produce trading signals for current market state
    - get_name(): Return strategy name for logging/reports
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize strategy.

        Args:
            config: Strategy-specific configuration dict
        """
        self.config = config or {}
        self.positions: Dict[str, Position] = {}

    @abstractmethod
    def generate_signals(
        self,
        timestamp: datetime,
        market_data: pd.DataFrame,
        positions: Dict[str, Position],
        bankroll_cents: int,
    ) -> List[Signal]:
        """
        Generate trading signals for current market state.

        Args:
            timestamp: Current timestamp in backtest
            market_data: DataFrame with columns:
                - market_ticker
                - open, high, low, close (prices in cents)
                - volume, num_trades
                - strike_type, floor_strike, cap_strike
                - close_time, expiration_time
                - temp_f (current weather observation)
                - ... (strategy-specific features)
            positions: Dict of current positions by market_ticker
            bankroll_cents: Available cash in cents

        Returns:
            List of Signal objects (empty list = no action)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name for logging and reports."""
        pass

    def update_positions(self, positions: Dict[str, Position]):
        """Update internal position tracking (called by backtester)."""
        self.positions = positions.copy()

    def should_enter(
        self,
        market_ticker: str,
        edge: float,
        confidence: float,
        spread_cents: int,
        max_spread_cents: int = 3,
        min_edge_cents: float = 3.0,
    ) -> bool:
        """
        Check entry conditions (default implementation).

        Args:
            market_ticker: Market identifier
            edge: Expected edge in cents
            confidence: Probability estimate [0, 1]
            spread_cents: Current bid-ask spread
            max_spread_cents: Maximum allowed spread (default: 3¢)
            min_edge_cents: Minimum edge to enter (default: 3¢)

        Returns:
            True if entry conditions met
        """
        # Don't enter if already in position
        if market_ticker in self.positions and self.positions[market_ticker].contracts != 0:
            return False

        # Check spread and edge
        if spread_cents > max_spread_cents:
            return False

        if edge < min_edge_cents:
            return False

        return True

    def should_exit(
        self,
        market_ticker: str,
        edge: float,
        unrealized_pnl_cents: float,
        min_edge_cents: float = 1.0,
        stop_loss_cents: Optional[float] = None,
    ) -> bool:
        """
        Check exit conditions (default implementation).

        Args:
            market_ticker: Market identifier
            edge: Current expected edge in cents
            unrealized_pnl_cents: Current unrealized P&L
            min_edge_cents: Minimum edge to hold (default: 1¢)
            stop_loss_cents: Stop loss threshold (negative, optional)

        Returns:
            True if exit conditions met
        """
        # Not in position
        if market_ticker not in self.positions or self.positions[market_ticker].contracts == 0:
            return False

        # Edge below threshold
        if edge < min_edge_cents:
            return True

        # Stop loss triggered
        if stop_loss_cents and unrealized_pnl_cents < stop_loss_cents:
            return True

        return False


class DummyStrategy(Strategy):
    """
    Dummy strategy for testing (always holds).

    Useful for testing backtest infrastructure without real trading logic.
    """

    def generate_signals(
        self,
        timestamp: datetime,
        market_data: pd.DataFrame,
        positions: Dict[str, Position],
        bankroll_cents: int,
    ) -> List[Signal]:
        """Return empty signal list (no trades)."""
        return []

    def get_name(self) -> str:
        """Return strategy name."""
        return "DummyStrategy"


class SimpleThresholdStrategy(Strategy):
    """
    Simple threshold-based strategy for testing.

    Buys when price < threshold, sells when price > threshold.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.buy_threshold = self.config.get("buy_threshold", 40)  # Buy below 40¢
        self.sell_threshold = self.config.get("sell_threshold", 60)  # Sell above 60¢
        self.size_fraction = self.config.get("size_fraction", 0.1)  # 10% of bankroll

    def generate_signals(
        self,
        timestamp: datetime,
        market_data: pd.DataFrame,
        positions: Dict[str, Position],
        bankroll_cents: int,
    ) -> List[Signal]:
        """Generate signals based on price thresholds."""
        signals = []

        for _, row in market_data.iterrows():
            # Support both "ticker" and "market_ticker" column names
            market_ticker = row.get("market_ticker", row.get("ticker"))
            close_price = row.get("close", 50)  # Default to 50¢ if no close price

            # Calculate edge (simplified)
            if close_price < self.buy_threshold:
                edge = self.buy_threshold - close_price
                action = "buy"
            elif close_price > self.sell_threshold:
                edge = close_price - self.sell_threshold
                action = "sell"
            else:
                continue

            # Check entry conditions
            high = row.get("high", close_price)
            low = row.get("low", close_price)
            spread_cents = int(high - low) if high and low else 0
            if not self.should_enter(market_ticker, edge, 0.5, spread_cents):
                continue

            signals.append(Signal(
                timestamp=timestamp,
                market_ticker=market_ticker,
                action=action,
                edge=edge,
                confidence=0.5,  # Dummy confidence
                size_fraction=self.size_fraction,
                reason=f"Price {close_price}¢ vs threshold {self.buy_threshold if action == 'buy' else self.sell_threshold}¢"
            ))

        return signals

    def get_name(self) -> str:
        """Return strategy name."""
        return f"SimpleThreshold(buy<{self.buy_threshold}, sell>{self.sell_threshold})"


def main():
    """Demo: Test strategy interface."""
    print("\n" + "="*60)
    print("Strategy Interface Demo")
    print("="*60 + "\n")

    # Create dummy strategy
    dummy = DummyStrategy()
    print(f"Strategy: {dummy.get_name()}")

    # Create simple threshold strategy
    threshold = SimpleThresholdStrategy(config={
        "buy_threshold": 35,
        "sell_threshold": 65,
        "size_fraction": 0.15
    })
    print(f"Strategy: {threshold.get_name()}")

    # Test signal generation
    test_data = pd.DataFrame([
        {
            "market_ticker": "KXHIGHCHI-25AUG10-B80",
            "open": 30,
            "high": 35,
            "low": 28,
            "close": 32,
            "volume": 1000,
            "num_trades": 50,
        }
    ])

    signals = threshold.generate_signals(
        timestamp=datetime.now(),
        market_data=test_data,
        positions={},
        bankroll_cents=10000_00
    )

    print(f"\nGenerated {len(signals)} signals:")
    for sig in signals:
        print(f"  {sig.action.upper()} {sig.market_ticker}")
        print(f"    Edge: {sig.edge:.1f}¢, Size: {sig.size_fraction:.1%}")
        print(f"    Reason: {sig.reason}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
