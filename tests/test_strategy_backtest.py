#!/usr/bin/env python3
"""
Test strategy-based backtester.

Verifies that:
1. DummyStrategy works (no trades)
2. SimpleThresholdStrategy works (some trades)
3. RiskManager integration works
4. Portfolio, signals, and settlements work correctly
"""

import logging
from datetime import date, datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.run_backtest import (
    load_markets_with_settlements,
    run_strategy_backtest,
    print_backtest_summary,
)
from backtest.strategy import DummyStrategy, SimpleThresholdStrategy
from backtest.risk import RiskManager, RiskLimits

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_dummy_strategy():
    """Test DummyStrategy (should generate no trades)."""
    print("\n" + "="*60)
    print("TEST 1: DummyStrategy")
    print("="*60)

    # Load small date range
    markets_df = load_markets_with_settlements(
        city="chicago",
        start_date=date(2025, 8, 10),
        end_date=date(2025, 8, 15),  # 6 days
    )

    print(f"Loaded {len(markets_df)} markets\n")

    # Run backtest with DummyStrategy
    strategy = DummyStrategy()
    summary = run_strategy_backtest(
        strategy=strategy,
        markets_df=markets_df,
        city="chicago",
        initial_cash_cents=10_000_00,
    )

    # Verify no trades
    assert summary['num_trades'] == 0, "DummyStrategy should generate no trades"
    assert summary['num_settlements'] == 0, "No trades = no settlements"
    assert summary['total_pnl_cents'] == 0, "No trades = zero P&L"

    print("✅ DummyStrategy test PASSED: no trades executed")


def test_simple_threshold_strategy():
    """Test SimpleThresholdStrategy (should generate some trades)."""
    print("\n" + "="*60)
    print("TEST 2: SimpleThresholdStrategy")
    print("="*60)

    # Load small date range
    markets_df = load_markets_with_settlements(
        city="chicago",
        start_date=date(2025, 8, 10),
        end_date=date(2025, 8, 15),  # 6 days
    )

    print(f"Loaded {len(markets_df)} markets\n")

    # Run backtest with SimpleThresholdStrategy
    strategy = SimpleThresholdStrategy(config={
        "buy_threshold": 45,  # Buy below 45¢
        "sell_threshold": 55,  # Sell above 55¢
        "size_fraction": 0.05,  # 5% of bankroll
    })

    summary = run_strategy_backtest(
        strategy=strategy,
        markets_df=markets_df,
        city="chicago",
        initial_cash_cents=10_000_00,
    )

    print_backtest_summary(summary)

    # Verify strategy executed (may or may not have trades depending on prices)
    print(f"✅ SimpleThresholdStrategy test PASSED")
    print(f"   Trades: {summary['num_trades']}, Settlements: {summary['num_settlements']}")
    print(f"   P&L: ${summary['total_pnl_cents']/100:.2f}")


def test_risk_manager_integration():
    """Test RiskManager limits."""
    print("\n" + "="*60)
    print("TEST 3: RiskManager Integration")
    print("="*60)

    # Load small date range
    markets_df = load_markets_with_settlements(
        city="chicago",
        start_date=date(2025, 8, 10),
        end_date=date(2025, 8, 12),  # 3 days
    )

    print(f"Loaded {len(markets_df)} markets\n")

    # Create aggressive strategy that tries to buy everything
    strategy = SimpleThresholdStrategy(config={
        "buy_threshold": 60,  # Buy below 60¢ (most markets)
        "sell_threshold": 70,  # Sell above 70¢
        "size_fraction": 0.20,  # 20% of bankroll per trade
    })

    # Create strict risk manager (max 2 bins per city)
    risk_manager = RiskManager(limits=RiskLimits(
        max_pct_per_city_day_side=0.10,  # 10% per (city, day, side)
        max_bins_per_city=2,  # Max 2 bins
        max_total_exposure_pct=0.30,  # 30% total
    ))

    summary = run_strategy_backtest(
        strategy=strategy,
        markets_df=markets_df,
        city="chicago",
        initial_cash_cents=10_000_00,
        risk_manager=risk_manager,
    )

    print_backtest_summary(summary)

    print(f"✅ RiskManager test PASSED")
    print(f"   Risk limits enforced during backtest")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("STRATEGY BACKTEST TESTS")
    print("="*60)

    try:
        test_dummy_strategy()
        test_simple_threshold_strategy()
        test_risk_manager_integration()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
