"""
Position Sizing and Risk Management

Implements Kelly-like position sizing based on:
- Expected value per contract
- Model uncertainty (settlement std deviation)
- Position limits and risk constraints
"""

import logging
from datetime import date
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result from position sizing calculation"""
    num_contracts: int
    notional_usd: float
    reason: str
    kelly_fraction: float = 0.0
    capped_by: Optional[str] = None


class PositionSizer:
    """
    Kelly-like position sizing with safety constraints.

    Core logic:
    1. Calculate Kelly fraction: f = edge / variance
    2. Apply conservative Kelly fraction (e.g., 0.25 = quarter Kelly)
    3. Cap by max bet size, max position size, available capital
    4. Scale down for high model uncertainty
    """

    def __init__(
        self,
        bankroll_usd: float = 10000.0,
        kelly_fraction: float = 0.25,  # Quarter Kelly (conservative)
        max_bet_usd: float = 50.0,
        max_position_contracts: int = 100,
        uncertainty_penalty: bool = True
    ):
        self.bankroll = bankroll_usd
        self.kelly_fraction = kelly_fraction
        self.max_bet_usd = max_bet_usd
        self.max_position_contracts = max_position_contracts
        self.uncertainty_penalty = uncertainty_penalty

    def calculate(
        self,
        ev_per_contract_cents: float,
        price_cents: int,
        model_prob: float,
        settlement_std_degf: float,
        current_position: int = 0
    ) -> PositionSizeResult:
        """
        Calculate optimal position size using Kelly criterion.

        Args:
            ev_per_contract_cents: Expected value per contract in cents
            price_cents: Trade price in cents
            model_prob: Model's probability the contract wins [0, 1]
            settlement_std_degf: Model uncertainty (std of settlement temp in °F)
            current_position: Current contracts held (to calculate available capacity)

        Returns:
            PositionSizeResult with num_contracts and reasoning
        """
        if ev_per_contract_cents <= 0:
            return PositionSizeResult(
                num_contracts=0,
                notional_usd=0.0,
                reason="Negative EV",
            )

        # Convert to dollars
        ev_per_contract_dollars = ev_per_contract_cents / 100.0
        price_dollars = price_cents / 100.0

        # Calculate edge per dollar risked
        edge_per_dollar = ev_per_contract_dollars / price_dollars

        # Variance for binary outcome: p(1-p) where p is win probability
        # This is the CORRECT variance, not a price proxy
        variance = model_prob * (1 - model_prob)

        # Kelly fraction: f = edge / variance
        if variance > 0:
            f_kelly = edge_per_dollar / variance
        else:
            f_kelly = 0.0

        # Apply conservative fraction
        f = f_kelly * self.kelly_fraction

        # Cap at reasonable maximum (don't bet >10% of bankroll on one trade)
        f = min(f, 0.10)

        # Uncertainty penalty: scale down for high std
        if self.uncertainty_penalty:
            # If std > 3°F, scale down linearly
            # std=3 → multiplier=1.0, std=4 → 0.75, std=5 → 0.5, std=6 → 0.25
            if settlement_std_degf > 3.0:
                penalty = max(0.25, 1.0 - (settlement_std_degf - 3.0) * 0.25)
                f = f * penalty

        # Calculate position in contracts
        stake_usd = f * self.bankroll
        contracts_from_kelly = int(stake_usd / price_dollars)

        # Apply caps
        contracts = contracts_from_kelly
        capped_by = None

        # Cap 1: Max bet size
        max_contracts_from_bet = int(self.max_bet_usd / price_dollars)
        if contracts > max_contracts_from_bet:
            contracts = max_contracts_from_bet
            capped_by = "max_bet_size"

        # Cap 2: Max position size
        available_capacity = self.max_position_contracts - current_position
        if contracts > available_capacity:
            contracts = available_capacity
            capped_by = "max_position"

        # Minimum: at least 1 contract if any edge
        if contracts == 0 and ev_per_contract_cents > 0:
            contracts = 1
            capped_by = "minimum"

        notional = contracts * price_dollars

        return PositionSizeResult(
            num_contracts=contracts,
            notional_usd=notional,
            reason=f"Kelly={f_kelly:.3f}, scaled={f:.3f}, EV={ev_per_contract_cents:.2f}¢",
            kelly_fraction=f,
            capped_by=capped_by
        )


@dataclass
class DailyPnLTracker:
    """Track daily P&L and enforce loss limits"""
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trade_count: int = 0
    start_date: date = field(default_factory=date.today)

    def add_trade(self, pnl: float):
        """Add realized P&L from a closed trade"""
        self.total_pnl += pnl
        self.realized_pnl += pnl
        self.trade_count += 1

    def update_unrealized(self, pnl: float):
        """Update unrealized P&L for open positions"""
        self.unrealized_pnl = pnl
        self.total_pnl = self.realized_pnl + self.unrealized_pnl

    def is_loss_limit_hit(self, max_loss: float) -> bool:
        """Check if daily loss limit is exceeded"""
        return self.total_pnl <= -abs(max_loss)

    def check_daily_reset(self):
        """Check if new day, reset if so"""
        today = date.today()
        if today > self.start_date:
            logger.info(
                f"Daily P&L reset: {self.trade_count} trades, "
                f"realized=${self.realized_pnl:.2f}, "
                f"unrealized=${self.unrealized_pnl:.2f}, "
                f"total=${self.total_pnl:.2f}"
            )
            self.total_pnl = 0.0
            self.realized_pnl = 0.0
            self.unrealized_pnl = 0.0
            self.trade_count = 0
            self.start_date = today


# ===== TESTS =====

if __name__ == "__main__":
    print("Position Sizing Examples:")
    print()

    sizer = PositionSizer(
        bankroll_usd=10000,
        kelly_fraction=0.25,
        max_bet_usd=50,
        max_position_contracts=100,
    )

    # Case 1: Good edge, low uncertainty
    result = sizer.calculate(
        ev_per_contract_cents=14.0,  # 14¢ EV
        price_cents=46,
        settlement_std_degf=2.5,  # Low uncertainty
        current_position=0
    )
    print(f"Good edge, low uncertainty:")
    print(f"  EV: 14¢, price: 46¢, std: 2.5°F")
    print(f"  → {result.num_contracts} contracts (${result.notional_usd:.2f})")
    print(f"  → {result.reason}")
    print(f"  → Capped by: {result.capped_by}")
    print()

    # Case 2: Good edge, high uncertainty
    result = sizer.calculate(
        ev_per_contract_cents=14.0,
        price_cents=46,
        settlement_std_degf=5.0,  # High uncertainty
        current_position=0
    )
    print(f"Good edge, high uncertainty:")
    print(f"  EV: 14¢, price: 46¢, std: 5.0°F")
    print(f"  → {result.num_contracts} contracts (${result.notional_usd:.2f})")
    print(f"  → {result.reason}")
    print(f"  → Penalty applied for high std")
    print()

    # Case 3: Small edge
    result = sizer.calculate(
        ev_per_contract_cents=5.0,
        price_cents=46,
        settlement_std_degf=2.5,
        current_position=0
    )
    print(f"Small edge:")
    print(f"  EV: 5¢, price: 46¢, std: 2.5°F")
    print(f"  → {result.num_contracts} contracts (${result.notional_usd:.2f})")
    print()

    # Case 4: Near position limit
    result = sizer.calculate(
        ev_per_contract_cents=14.0,
        price_cents=46,
        settlement_std_degf=2.5,
        current_position=95  # Already have 95 contracts
    )
    print(f"Near position limit (95/100):")
    print(f"  → {result.num_contracts} contracts")
    print(f"  → Capped by: {result.capped_by}")
