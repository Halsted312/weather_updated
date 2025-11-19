#!/usr/bin/env python3
"""
Risk management for Kalshi weather trading.

Enforces position limits and diversification constraints:
- Max exposure per (city, day, side): 10% of bankroll
- Max concurrent bins per city: 3
- Total exposure cap: 50% of bankroll
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
from datetime import datetime, date
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk management configuration."""
    max_pct_per_city_day_side: float = 0.10  # 10% per (city, day, side)
    max_bins_per_city: int = 3  # Max 3 concurrent bins per city
    max_total_exposure_pct: float = 0.50  # 50% total portfolio exposure
    max_position_size_contracts: Optional[int] = None  # Optional absolute limit


@dataclass
class Exposure:
    """Current exposure metrics."""
    city: str
    event_date: date
    side: Literal["long", "short"]
    market_tickers: List[str] = field(default_factory=list)
    total_capital_cents: int = 0
    num_bins: int = 0


class RiskManager:
    """
    Risk manager enforcing position limits and diversification.

    Tracks exposures by (city, day, side) and bins per city.
    Rejects trades that violate risk limits.
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        """
        Initialize risk manager.

        Args:
            limits: Risk limits configuration (uses defaults if None)
        """
        self.limits = limits or RiskLimits()
        self.exposures: Dict[Tuple[str, date, str], Exposure] = {}
        self.bins_per_city: Dict[str, int] = defaultdict(int)

    def check_trade(
        self,
        market_ticker: str,
        city: str,
        event_date: date,
        side: Literal["long", "short"],
        contracts: int,
        price_cents: int,
        bankroll_cents: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if trade violates risk limits.

        Args:
            market_ticker: Market identifier
            city: City (e.g., "chicago")
            event_date: Event date (settlement date)
            side: "long" (buy YES) or "short" (sell YES)
            contracts: Number of contracts
            price_cents: Price per contract in cents
            bankroll_cents: Current bankroll

        Returns:
            (allowed, reason): True if trade allowed, else (False, rejection reason)
        """
        # Calculate capital to deploy
        capital_cents = contracts * price_cents

        # Check total exposure limit
        total_exposure = self._calculate_total_exposure()
        new_total_exposure = total_exposure + capital_cents
        if new_total_exposure > self.limits.max_total_exposure_pct * bankroll_cents:
            return False, (
                f"Total exposure limit: {new_total_exposure/100:.0f} > "
                f"{self.limits.max_total_exposure_pct * bankroll_cents/100:.0f}"
            )

        # Check per (city, day, side) limit
        key = (city, event_date, side)
        current_exposure = self.exposures.get(key)
        current_capital = current_exposure.total_capital_cents if current_exposure else 0
        new_capital = current_capital + capital_cents
        max_capital = self.limits.max_pct_per_city_day_side * bankroll_cents

        if new_capital > max_capital:
            return False, (
                f"City/day/side limit: {new_capital/100:.0f} > {max_capital/100:.0f} "
                f"({city}, {event_date}, {side})"
            )

        # Check bins per city limit (only if opening new position in new market)
        if current_exposure is None or market_ticker not in current_exposure.market_tickers:
            current_bins = self.bins_per_city.get(city, 0)
            if current_bins >= self.limits.max_bins_per_city:
                return False, (
                    f"Max bins per city: {current_bins} >= {self.limits.max_bins_per_city} "
                    f"({city})"
                )

        # Check absolute position size (if configured)
        if self.limits.max_position_size_contracts:
            if contracts > self.limits.max_position_size_contracts:
                return False, (
                    f"Max position size: {contracts} > {self.limits.max_position_size_contracts}"
                )

        return True, None

    def record_trade(
        self,
        market_ticker: str,
        city: str,
        event_date: date,
        side: Literal["long", "short"],
        contracts: int,
        price_cents: int,
    ):
        """
        Record trade and update exposure tracking.

        Args:
            market_ticker: Market identifier
            city: City
            event_date: Event date
            side: "long" or "short"
            contracts: Number of contracts
            price_cents: Price per contract
        """
        capital_cents = contracts * price_cents
        key = (city, event_date, side)

        if key in self.exposures:
            # Update existing exposure
            exp = self.exposures[key]
            exp.total_capital_cents += capital_cents
            if market_ticker not in exp.market_tickers:
                exp.market_tickers.append(market_ticker)
                exp.num_bins += 1
                self.bins_per_city[city] += 1
        else:
            # Create new exposure
            self.exposures[key] = Exposure(
                city=city,
                event_date=event_date,
                side=side,
                market_tickers=[market_ticker],
                total_capital_cents=capital_cents,
                num_bins=1
            )
            self.bins_per_city[city] += 1

        logger.debug(
            f"Recorded trade: {market_ticker} ({city}, {event_date}, {side}), "
            f"capital: ${capital_cents/100:.0f}, bins: {self.bins_per_city[city]}"
        )

    def close_position(
        self,
        market_ticker: str,
        city: str,
        event_date: date,
        side: Literal["long", "short"],
        capital_cents: int,
    ):
        """
        Close position and update exposure tracking.

        Args:
            market_ticker: Market identifier
            city: City
            event_date: Event date
            side: "long" or "short"
            capital_cents: Capital to release
        """
        key = (city, event_date, side)

        if key not in self.exposures:
            logger.warning(f"Tried to close non-existent exposure: {key}")
            return

        exp = self.exposures[key]
        exp.total_capital_cents -= capital_cents

        if market_ticker in exp.market_tickers:
            exp.market_tickers.remove(market_ticker)
            exp.num_bins -= 1
            self.bins_per_city[city] -= 1

        # Remove exposure if depleted
        if exp.total_capital_cents <= 0 or exp.num_bins == 0:
            del self.exposures[key]

        logger.debug(
            f"Closed position: {market_ticker} ({city}, {event_date}, {side}), "
            f"released: ${capital_cents/100:.0f}, bins: {self.bins_per_city[city]}"
        )

    def settle_event(self, city: str, event_date: date):
        """
        Settle all positions for a given event (removes exposures).

        Args:
            city: City
            event_date: Event date that settled
        """
        keys_to_remove = [
            key for key in self.exposures.keys()
            if key[0] == city and key[1] == event_date
        ]

        for key in keys_to_remove:
            exp = self.exposures[key]
            self.bins_per_city[city] -= exp.num_bins
            del self.exposures[key]

        logger.debug(
            f"Settled event: {city} {event_date}, removed {len(keys_to_remove)} exposures"
        )

    def get_exposure_summary(self, bankroll_cents: int) -> Dict:
        """
        Get current exposure summary.

        Args:
            bankroll_cents: Current bankroll

        Returns:
            Dict with exposure metrics
        """
        total_exposure = self._calculate_total_exposure()

        by_city = defaultdict(int)
        by_city_day = defaultdict(int)

        for (city, event_date, side), exp in self.exposures.items():
            by_city[city] += exp.total_capital_cents
            by_city_day[(city, event_date)] += exp.total_capital_cents

        return {
            "total_exposure_cents": total_exposure,
            "total_exposure_pct": total_exposure / bankroll_cents if bankroll_cents > 0 else 0,
            "num_exposures": len(self.exposures),
            "bins_per_city": dict(self.bins_per_city),
            "exposure_by_city": {k: v for k, v in by_city.items()},
            "max_city_day_exposure": max(by_city_day.values()) if by_city_day else 0,
        }

    def _calculate_total_exposure(self) -> int:
        """Calculate total capital deployed across all exposures."""
        return sum(exp.total_capital_cents for exp in self.exposures.values())


def main():
    """Demo: Risk management checks."""
    print("\n" + "="*60)
    print("Risk Manager Demo")
    print("="*60 + "\n")

    # Initialize risk manager
    limits = RiskLimits(
        max_pct_per_city_day_side=0.10,
        max_bins_per_city=3,
        max_total_exposure_pct=0.50
    )
    rm = RiskManager(limits)

    bankroll = 10000_00  # $10,000

    print(f"Bankroll: ${bankroll/100:,.0f}")
    print(f"Risk Limits:")
    print(f"  Max per (city, day, side): {limits.max_pct_per_city_day_side:.0%}")
    print(f"  Max bins per city: {limits.max_bins_per_city}")
    print(f"  Max total exposure: {limits.max_total_exposure_pct:.0%}")
    print("")

    # Test trades
    trades = [
        # (market_ticker, city, date, side, contracts, price)
        ("CHI-AUG10-B80", "chicago", date(2025, 8, 10), "long", 100, 50),
        ("CHI-AUG10-B81", "chicago", date(2025, 8, 10), "long", 100, 50),
        ("CHI-AUG10-B82", "chicago", date(2025, 8, 10), "long", 100, 50),
        ("CHI-AUG10-B83", "chicago", date(2025, 8, 10), "long", 100, 50),  # Should fail (4th bin)
        ("CHI-AUG11-B80", "chicago", date(2025, 8, 11), "long", 100, 50),
        ("CHI-AUG10-L70", "chicago", date(2025, 8, 10), "long", 1000, 50),  # Should fail (capital limit)
    ]

    print("Testing trades:")
    print("-" * 60)

    for market_ticker, city, event_date, side, contracts, price in trades:
        allowed, reason = rm.check_trade(
            market_ticker, city, event_date, side, contracts, price, bankroll
        )

        capital = contracts * price
        status = "✓ ALLOWED" if allowed else "✗ REJECTED"

        print(f"{status} {market_ticker} ({city}, {event_date}, {side})")
        print(f"  Contracts: {contracts} @ {price}¢ = ${capital/100:,.0f}")

        if allowed:
            rm.record_trade(market_ticker, city, event_date, side, contracts, price)
            summary = rm.get_exposure_summary(bankroll)
            print(f"  Total exposure: ${summary['total_exposure_cents']/100:,.0f} ({summary['total_exposure_pct']:.1%})")
            print(f"  Bins: {summary['bins_per_city']}")
        else:
            print(f"  Reason: {reason}")

        print("")

    # Final summary
    print("="*60)
    print("Final Exposure Summary")
    print("="*60)

    summary = rm.get_exposure_summary(bankroll)
    print(f"Total exposure: ${summary['total_exposure_cents']/100:,.0f} ({summary['total_exposure_pct']:.1%})")
    print(f"Num exposures: {summary['num_exposures']}")
    print(f"Bins per city: {summary['bins_per_city']}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
