#!/usr/bin/env python3
"""Quick verification test for live trading platform components."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all components import correctly."""
    print("Testing imports...")

    from live_trading.config import TradingConfig
    print("✓ TradingConfig")

    from live_trading.market_data import get_market_snapshot
    print("✓ market_data")

    from live_trading.scanner import CityScannerEngine, CityOpportunity
    print("✓ scanner")

    from live_trading.incremental_executor import IncrementalOrderExecutor, OrderChunk
    print("✓ incremental_executor")

    from live_trading.daily_loss_tracker import DailyLossTracker
    print("✓ daily_loss_tracker")

    from live_trading.order_manager import OrderManager
    print("✓ order_manager")

    from live_trading.ui.display import format_opportunities_table
    print("✓ ui.display")

    from live_trading.ui.manual_cli import ManualTradingCLI
    print("✓ ui.manual_cli")

    print("\n✅ All imports successful!\n")
    return True


def test_config():
    """Test config creation and validation."""
    print("Testing TradingConfig...")

    from live_trading.config import TradingConfig

    # Default config
    config = TradingConfig()
    errors = config.validate()

    if errors:
        print(f"✗ Validation errors: {errors}")
        return False

    print(f"✓ Default config valid")
    print(f"  Max bet: ${config.max_bet_per_trade_usd}")
    print(f"  Max daily loss: ${config.max_daily_loss_usd}")
    print(f"  Inference mode: {config.inference_mode}")
    print(f"  Classifier cities: {config.edge_classifier_cities}")

    # Test from_json (with missing file = use defaults)
    config2 = TradingConfig.from_json(Path("config/auto_trader.json"))
    print(f"✓ Config from JSON")

    print("\n✅ Config tests passed!\n")
    return True


def test_components():
    """Test component instantiation."""
    print("Testing component instantiation...")

    from live_trading.config import TradingConfig
    from live_trading.inference import InferenceWrapper
    from live_trading.scanner import CityScannerEngine
    from live_trading.daily_loss_tracker import DailyLossTracker

    config = TradingConfig()
    print("✓ TradingConfig created")

    inference = InferenceWrapper()
    print(f"✓ InferenceWrapper created (ordinal models: {list(inference.live_engine.models.keys())})")

    loss_tracker = DailyLossTracker(config)
    print("✓ DailyLossTracker created")

    scanner = CityScannerEngine(
        config=config,
        inference=inference,
        ws_handler=None,  # Optional
        order_book_mgr=None,  # Optional
        market_state_tracker=None,  # Optional
    )
    print(f"✓ CityScannerEngine created")
    print(f"  Detected classifiers: {scanner.cities_with_classifiers}")

    # Test adaptive logic
    for city in ['chicago', 'austin', 'denver', 'los_angeles']:
        uses_classifier = scanner.should_use_classifier(city)
        print(f"  {city}: {'ML classifier' if uses_classifier else 'Threshold-only'}")

    print("\n✅ Component tests passed!\n")
    return True


def test_circuit_breaker():
    """Test circuit breaker logic."""
    print("Testing circuit breaker...")

    from live_trading.config import TradingConfig
    from live_trading.daily_loss_tracker import DailyLossTracker
    from datetime import date

    config = TradingConfig(max_daily_loss_usd=100.0)
    tracker = DailyLossTracker(config)

    # Simulate a loss
    tracker.daily_pnl_cents[date.today()] = -15000  # -$150 loss

    within_limit, msg = tracker.check_daily_loss_limit()
    print(f"  Daily loss check: within_limit={within_limit}, msg='{msg}'")

    if within_limit:
        print("  ✗ Circuit breaker should have triggered!")
        return False

    print("✓ Circuit breaker triggered correctly at $150 loss (limit: $100)")

    # Test can_trade
    can_trade, reason = tracker.can_trade("chicago", date.today())
    if can_trade:
        print("  ✗ Should not be able to trade!")
        return False

    print(f"✓ can_trade blocked: {reason}")

    print("\n✅ Circuit breaker tests passed!\n")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Live Trading Platform Verification Tests")
    print("=" * 60)
    print()

    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Components", test_components),
        ("Circuit Breaker", test_circuit_breaker),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"❌ {name} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ {name} FAILED with exception: {e}\n")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {failed} tests failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
