"""
Reporting and output functions for open-maker backtests.

This module contains:
- print_results: Console output of backtest summary
- print_debug_trades: Detailed trade info for debugging
- print_comparison_table: Multi-strategy comparison
- save_results_to_db: Persist results to sim schema
"""

import logging
import random
from typing import List

from sqlalchemy.dialects.postgresql import insert

from src.db import get_db_session
from src.db.models import SimRun, SimTrade

from .core_runner import OpenMakerResult, OpenMakerTrade

logger = logging.getLogger(__name__)


def print_results(result: OpenMakerResult) -> None:
    """Print backtest results to console."""
    print("\n" + "=" * 60)
    print("OPEN-MAKER BACKTEST RESULTS")
    print("=" * 60)

    summary = result.summary()
    print(f"\nStrategy: {summary['strategy']}")
    print(f"Run ID: {summary['run_id']}")
    print(f"Period: {summary['start_date']} to {summary['end_date']}")
    print(f"Cities: {', '.join(summary['cities'])}")

    print(f"\nParameters:")
    print(f"  Entry Price: {summary['params']['entry_price_cents']}c")
    print(f"  Temp Bias: {summary['params']['temp_bias_deg']:+.1f}F")
    print(f"  Basis Offset: {summary['params']['basis_offset_days']} day(s)")
    print(f"  Bet Amount: ${summary['params']['bet_amount_usd']:.2f}")

    print(f"\nResults:")
    print(f"  Total Trades: {summary['num_trades']} ({summary['num_days']} days)")
    print(f"  Win Rate: {summary['win_rate']:.1%}")
    print(f"  Total P&L: ${summary['total_pnl_usd']:+,.2f}")
    print(f"  Avg P&L/Trade: ${summary['avg_pnl_per_trade']:+.2f}")
    print(f"  Total Wagered: ${summary['total_wagered']:,.2f}")
    print(f"  ROI: {summary['roi']:+.2f}%")

    print(f"\nRisk Metrics:")
    print(f"  P&L Std (per-trade): ${summary['pnl_std_per_trade']:.2f}")
    print(f"  Sharpe (per-trade): {summary['sharpe_per_trade']:.3f}")
    print(f"  P&L Std (daily): ${summary['pnl_std_daily']:.2f}")
    print(f"  Sharpe (daily): {summary['sharpe_daily']:.3f}")

    # By city breakdown
    by_city = result.by_city()
    if len(by_city) > 1:
        print(f"\nBy City:")
        for city, stats in sorted(by_city.items()):
            print(f"  {city:15s}: {stats['num_trades']:4d} trades, "
                  f"{stats['win_rate']:.1%} win, ${stats['total_pnl']:+8.2f} P&L")

    print("=" * 60)


def print_debug_trades(result: OpenMakerResult, n: int = 20) -> None:
    """Print detailed info for N random trades for sanity checking."""
    if not result.trades:
        print("No trades to debug")
        return

    sample = random.sample(result.trades, min(n, len(result.trades)))

    print("\n" + "=" * 80)
    print(f"DEBUG: Sample of {len(sample)} trades")
    print("=" * 80)

    for trade in sorted(sample, key=lambda t: (t.city, t.event_date)):
        bracket_str = ""
        if trade.floor_strike is not None and trade.cap_strike is not None:
            bracket_str = f"[{trade.floor_strike}, {trade.cap_strike})"
        elif trade.floor_strike is not None:
            bracket_str = f">= {trade.floor_strike}"
        elif trade.cap_strike is not None:
            bracket_str = f"< {trade.cap_strike}"

        print(f"\n{trade.city} | {trade.event_date} | {trade.ticker}")
        print(f"  Forecast: {trade.temp_fcst_open:.1f}F (basis: {trade.forecast_basis_date})")
        print(f"  Adjusted: {trade.temp_adjusted:.1f}F -> Bracket: {bracket_str}")
        print(f"  Entry: {trade.entry_price_cents:.1f}c x {trade.num_contracts} = ${trade.amount_usd:.2f}")
        print(f"  Actual TMAX: {trade.tmax_final:.1f}F")
        print(f"  Won: {trade.bin_won} | P&L: ${trade.pnl_net:+.2f}")

    print("\n" + "=" * 80)


def print_comparison_table(results: List[OpenMakerResult]) -> None:
    """Print a comparison table for multiple strategy results."""
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    print(f"{'Strategy':<25} {'Trades':>7} {'Win%':>7} {'ROI':>8} {'Sharpe_daily':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r.strategy_name:<25} {r.num_trades:>7} {r.win_rate:>6.1%} "
              f"{(r.total_pnl / r.total_wagered * 100 if r.total_wagered > 0 else 0):>7.1f}% "
              f"{r.sharpe_daily:>12.3f}")
    print("=" * 70)


def save_results_to_db(result: OpenMakerResult) -> None:
    """Save backtest results to the sim schema."""
    with get_db_session() as session:
        # Save run summary
        run_record = {
            "run_id": result.run_id,
            "strategy_id": result.strategy_name,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "params": result.summary()["params"],
            "metrics": {
                "num_trades": result.num_trades,
                "total_pnl": result.total_pnl,
                "win_rate": result.win_rate,
                "total_wagered": result.total_wagered,
            },
        }

        stmt = insert(SimRun).values(**run_record)
        stmt = stmt.on_conflict_do_update(
            index_elements=["run_id"],
            set_={"metrics": run_record["metrics"]},
        )
        session.execute(stmt)

        # Save individual trades
        for trade in result.trades:
            trade_record = {
                "run_id": result.run_id,
                "strategy_id": result.strategy_name,
                "city": trade.city,
                "event_date": trade.event_date,
                "ticker": trade.ticker,
                "decision_type": "open_maker",
                "side": "buy",
                "price_cents": trade.entry_price_cents,
                "num_contracts": trade.num_contracts,
                "amount_usd": trade.amount_usd,
                "fee_usd": trade.fee_usd,
                "pnl_usd": trade.pnl_net,
                "bin_won": trade.bin_won,
            }

            stmt = insert(SimTrade).values(**trade_record)
            stmt = stmt.on_conflict_do_nothing()
            session.execute(stmt)

        session.commit()
        logger.info(f"Saved {len(result.trades)} trades to database")
