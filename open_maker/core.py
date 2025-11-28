#!/usr/bin/env python3
"""
Open-Maker Backtest Module

Implements a simple "maker at open" trading strategy:
1. At market open (10am ET on event_date - 1), use forecast to pick bracket
2. Post maker limit order at fixed price P (e.g., 40c or 50c)
3. Assume fill, hold to settlement
4. Maker fees = $0 for weather markets

Usage:
    python -m open_maker.core --city chicago --days 90 --price 0.50
    python -m open_maker.core --all-cities --days 365 --price 0.45 --bias 0.5
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from typing import List

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import CITIES

# =============================================================================
# Re-exports for backward compatibility
# =============================================================================

# Import from new modules and re-export
from .data_loading import (
    load_tuned_params,
    load_forecast_data,
    load_settlement_data,
    load_market_data,
    load_candle_data,
    get_forecast_at_open,
)

from .core_runner import (
    run_strategy,
    OpenMakerTrade,
    OpenMakerResult,
)

from .reporting import (
    print_results,
    print_debug_trades,
    print_comparison_table,
    save_results_to_db,
)

# Import OpenMakerParams from strategies.base (canonical location)
from .strategies.base import OpenMakerParams, StrategyParamsBase
from .strategies.next_over import NextOverParams
from .strategies.curve_gap import CurveGapParams

# Re-export all public API items
__all__ = [
    # Data loading
    "load_tuned_params",
    "load_forecast_data",
    "load_settlement_data",
    "load_market_data",
    "load_candle_data",
    "get_forecast_at_open",
    # Core runner
    "run_strategy",
    "OpenMakerTrade",
    "OpenMakerResult",
    # Reporting
    "print_results",
    "print_debug_trades",
    "print_comparison_table",
    "save_results_to_db",
    # Params classes
    "OpenMakerParams",
    "NextOverParams",
    "CurveGapParams",
    # Legacy wrappers
    "run_backtest",
    "run_backtest_next_over",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Legacy Backtest Functions (for backward compatibility)
# These functions delegate to the unified run_strategy() for simplicity.
# =============================================================================

def run_backtest(
    cities: List[str],
    start_date: date,
    end_date: date,
    params: OpenMakerParams,
    strategy_name: str = "open_maker_base",
) -> OpenMakerResult:
    """
    Run the open-maker base backtest.

    This is a legacy wrapper that delegates to run_strategy().
    Kept for backward compatibility with existing code.

    Args:
        cities: List of city IDs to backtest
        start_date: Start date for backtest
        end_date: End date for backtest
        params: Strategy parameters
        strategy_name: Name for this run (ignored, uses "open_maker_base")

    Returns:
        OpenMakerResult with all trades
    """
    return run_strategy(
        strategy_id="open_maker_base",
        cities=cities,
        start_date=start_date,
        end_date=end_date,
        params=params,
    )


def run_backtest_next_over(
    cities: List[str],
    start_date: date,
    end_date: date,
    params: NextOverParams,
    strategy_name: str = "open_maker_next_over",
) -> OpenMakerResult:
    """
    Run the next_over strategy backtest with exit logic.

    This is a legacy wrapper that delegates to run_strategy().
    Kept for backward compatibility with existing code.

    Args:
        cities: List of city IDs to backtest
        start_date: Start date for backtest
        end_date: End date for backtest
        params: NextOverParams with exit thresholds
        strategy_name: Name for this run (ignored, uses "open_maker_next_over")

    Returns:
        OpenMakerResult with all trades
    """
    return run_strategy(
        strategy_id="open_maker_next_over",
        cities=cities,
        start_date=start_date,
        end_date=end_date,
        params=params,
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Open-maker backtest for Kalshi weather markets"
    )
    parser.add_argument(
        "--city", action="append",
        help="City to backtest (can specify multiple)"
    )
    parser.add_argument(
        "--all-cities", action="store_true",
        help="Backtest all cities"
    )
    parser.add_argument(
        "--start-date", type=str,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Number of days to backtest (if start/end not specified)"
    )
    parser.add_argument(
        "--price", type=float, default=50.0,
        help="Entry price in cents (default: 50)"
    )
    parser.add_argument(
        "--bias", type=float, default=0.0,
        help="Temperature bias in degrees F (default: 0.0)"
    )
    parser.add_argument(
        "--offset", type=int, default=1,
        help="Basis offset days (default: 1 = previous day forecast)"
    )
    parser.add_argument(
        "--bet-amount", type=float, default=200.0,
        help="Bet amount in USD (default: 200)"
    )
    parser.add_argument(
        "--strategy", action="append",
        choices=["open_maker_base", "open_maker_next_over", "open_maker_curve_gap"],
        help="Strategy to run (can specify multiple for comparison)"
    )
    # NextOver-specific params
    parser.add_argument(
        "--decision-offset", type=int, default=-180,
        help="Minutes before predicted high for decision (default: -180)"
    )
    parser.add_argument(
        "--neighbor-min", type=int, default=50,
        help="[next_over] Neighbor price threshold to trigger exit in cents (default: 50)"
    )
    parser.add_argument(
        "--our-max", type=int, default=30,
        help="[next_over] Our price threshold to allow exit in cents (default: 30)"
    )
    # CurveGap-specific params
    parser.add_argument(
        "--delta-obs-fcst-min", type=float, default=1.5,
        help="[curve_gap] Min T_obs - T_fcst to trigger shift in degrees F (default: 1.5)"
    )
    parser.add_argument(
        "--slope-min", type=float, default=0.5,
        help="[curve_gap] Min slope to trigger shift in degrees F/hour (default: 0.5)"
    )
    parser.add_argument(
        "--max-shift-bins", type=int, default=1,
        help="[curve_gap] Max bins to shift up (default: 1)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save results to database"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print detailed info for 20 random trades"
    )
    parser.add_argument(
        "--use-tuned", action="store_true",
        help="Use tuned parameters from config/*.json (from optuna_tuner)"
    )
    parser.add_argument(
        "--fill-check", action="store_true",
        help="Enable fill realism filter - skip trades where entry price wasn't achievable"
    )

    args = parser.parse_args()

    # Determine cities
    if args.all_cities:
        cities = list(CITIES.keys())
    elif args.city:
        cities = args.city
    else:
        cities = ["chicago"]

    # Determine date range
    today = date.today()
    if args.start_date and args.end_date:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date)
    else:
        end_date = today - timedelta(days=1)
        start_date = end_date - timedelta(days=args.days)

    # Determine strategies (default to base if not specified)
    strategies = args.strategy if args.strategy else ["open_maker_base"]

    results = []

    for strategy_id in strategies:
        # Build params based on strategy type
        params = None

        # Try to load tuned params if requested
        if args.use_tuned:
            params = load_tuned_params(strategy_id, bet_amount_usd=args.bet_amount)
            if params:
                logger.info(f"Using tuned params for {strategy_id}: {params}")
            else:
                logger.warning(f"No tuned params found for {strategy_id}, using defaults")

        # Fall back to CLI args if no tuned params
        if params is None:
            if strategy_id == "open_maker_base":
                params = OpenMakerParams(
                    entry_price_cents=args.price,
                    temp_bias_deg=args.bias,
                    basis_offset_days=args.offset,
                    bet_amount_usd=args.bet_amount,
                )
            elif strategy_id == "open_maker_next_over":
                params = NextOverParams(
                    entry_price_cents=args.price,
                    temp_bias_deg=args.bias,
                    basis_offset_days=args.offset,
                    bet_amount_usd=args.bet_amount,
                    decision_offset_min=args.decision_offset,
                    neighbor_price_min_c=args.neighbor_min,
                    our_price_max_c=args.our_max,
                )
            elif strategy_id == "open_maker_curve_gap":
                params = CurveGapParams(
                    entry_price_cents=args.price,
                    temp_bias_deg=args.bias,
                    basis_offset_days=args.offset,
                    bet_amount_usd=args.bet_amount,
                    decision_offset_min=args.decision_offset,
                    delta_obs_fcst_min_deg=args.delta_obs_fcst_min,
                    slope_min_deg_per_hour=args.slope_min,
                    max_shift_bins=args.max_shift_bins,
                )
            else:
                logger.error(f"Unknown strategy: {strategy_id}")
                continue

        # Use unified runner for all strategies
        result = run_strategy(
            strategy_id=strategy_id,
            cities=cities,
            start_date=start_date,
            end_date=end_date,
            params=params,
            fill_check=args.fill_check,
        )

        results.append(result)

    # Print results
    if len(results) == 1:
        print_results(results[0])
        if args.debug:
            print_debug_trades(results[0], n=20)
    else:
        # Multiple strategies - print comparison table
        for result in results:
            print_results(result)
        print_comparison_table(results)

    # Save if requested
    if args.save:
        for result in results:
            save_results_to_db(result)


if __name__ == "__main__":
    main()
