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
import math
import statistics
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import CITIES
from src.db import get_db_session
from src.db.models import (
    WxSettlement,
    WxForecastSnapshot,
    KalshiMarket,
    SimRun,
    SimTrade,
)

from .utils import (
    kalshi_maker_fee,
    find_bracket_for_temp,
    calculate_position_size,
    calculate_pnl,
    get_city_timezone,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Strategy Parameters
# =============================================================================

@dataclass
class OpenMakerParams:
    """Tunable parameters for the open-maker strategy."""

    # Fixed entry price in cents (0-100)
    entry_price_cents: float = 50.0

    # Temperature bias: adjust forecast before bracket selection
    temp_bias_deg: float = 0.0

    # Basis offset: 1 = use previous day's forecast, 0 = same day (if available)
    basis_offset_days: int = 1

    # Position sizing
    bet_amount_usd: float = 200.0


# =============================================================================
# Trade and Result Dataclasses
# =============================================================================

@dataclass
class OpenMakerTrade:
    """Represents a single trade in the open-maker backtest."""

    city: str
    event_date: date
    ticker: str
    forecast_basis_date: date
    temp_fcst_open: float  # Raw forecast temperature
    temp_adjusted: float  # After bias adjustment
    floor_strike: Optional[float]
    cap_strike: Optional[float]
    entry_price_cents: float
    num_contracts: int
    amount_usd: float  # Total position cost
    tmax_final: float  # Actual settlement temperature
    bin_won: bool
    pnl_gross: float
    fee_usd: float
    pnl_net: float


@dataclass
class OpenMakerResult:
    """Results from an open-maker backtest run."""

    run_id: str
    strategy_name: str
    params: OpenMakerParams
    start_date: date
    end_date: date
    cities: List[str]
    trades: List[OpenMakerTrade] = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_net for t in self.trades)

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.bin_won) / len(self.trades)

    @property
    def total_wagered(self) -> float:
        return sum(t.amount_usd for t in self.trades)

    @property
    def pnl_std_per_trade(self) -> float:
        """Standard deviation of per-trade P&L."""
        if len(self.trades) < 2:
            return 0.0
        pnls = [t.pnl_net for t in self.trades]
        return statistics.stdev(pnls)

    @property
    def sharpe_per_trade(self) -> float:
        """
        Per-trade Sharpe ratio: mean(pnl) / stdev(pnl).

        Good for comparing parameter sets in optimization.
        Returns 0.0 if stdev is 0 or insufficient trades.
        """
        if len(self.trades) < 2:
            return 0.0
        pnls = [t.pnl_net for t in self.trades]
        mean_pnl = statistics.mean(pnls)
        std_pnl = statistics.stdev(pnls)
        if std_pnl == 0:
            return float('inf') if mean_pnl > 0 else 0.0
        return mean_pnl / std_pnl

    @property
    def daily_pnl_series(self) -> Dict[date, float]:
        """
        Aggregate P&L by event_date across all cities.

        Returns dict mapping event_date -> total P&L for that day.
        """
        daily_pnl: Dict[date, float] = defaultdict(float)
        for trade in self.trades:
            daily_pnl[trade.event_date] += trade.pnl_net
        return dict(daily_pnl)

    @property
    def sharpe_daily(self) -> float:
        """
        Daily aggregate Sharpe: mean(daily_pnl) / stdev(daily_pnl).

        Closer to "what does my P&L time-series look like as a daily strategy?"
        Returns 0.0 if stdev is 0 or insufficient days.
        """
        daily = list(self.daily_pnl_series.values())
        if len(daily) < 2:
            return 0.0
        mean_pnl = statistics.mean(daily)
        std_pnl = statistics.stdev(daily)
        if std_pnl == 0:
            return float('inf') if mean_pnl > 0 else 0.0
        return mean_pnl / std_pnl

    @property
    def pnl_std_daily(self) -> float:
        """Standard deviation of daily aggregate P&L."""
        daily = list(self.daily_pnl_series.values())
        if len(daily) < 2:
            return 0.0
        return statistics.stdev(daily)

    def summary(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "strategy": self.strategy_name,
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "cities": self.cities,
            "num_trades": self.num_trades,
            "num_days": len(self.daily_pnl_series),
            "total_pnl_usd": round(self.total_pnl, 2),
            "win_rate": round(self.win_rate, 4),
            "avg_pnl_per_trade": round(self.total_pnl / self.num_trades, 2) if self.trades else 0,
            "total_wagered": round(self.total_wagered, 2),
            "roi": round(self.total_pnl / self.total_wagered * 100, 2) if self.total_wagered > 0 else 0,
            "pnl_std_per_trade": round(self.pnl_std_per_trade, 2),
            "sharpe_per_trade": round(self.sharpe_per_trade, 3),
            "pnl_std_daily": round(self.pnl_std_daily, 2),
            "sharpe_daily": round(self.sharpe_daily, 3),
            "params": {
                "entry_price_cents": self.params.entry_price_cents,
                "temp_bias_deg": self.params.temp_bias_deg,
                "basis_offset_days": self.params.basis_offset_days,
                "bet_amount_usd": self.params.bet_amount_usd,
            },
        }

    def by_city(self) -> Dict[str, Dict[str, Any]]:
        """Get results broken down by city."""
        results = {}
        for city in self.cities:
            city_trades = [t for t in self.trades if t.city == city]
            if city_trades:
                results[city] = {
                    "num_trades": len(city_trades),
                    "win_rate": sum(1 for t in city_trades if t.bin_won) / len(city_trades),
                    "total_pnl": sum(t.pnl_net for t in city_trades),
                    "avg_pnl": sum(t.pnl_net for t in city_trades) / len(city_trades),
                }
        return results


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_forecast_data(
    session,
    city: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Load forecast data for a city and date range.

    For the open-maker strategy, we need the forecast available at market open,
    which is typically the previous day's forecast (lead_days=1).

    Returns DataFrame with columns:
        target_date, basis_date, tempmax_fcst_f, lead_days
    """
    query = select(
        WxForecastSnapshot.target_date,
        WxForecastSnapshot.basis_date,
        WxForecastSnapshot.tempmax_fcst_f,
        WxForecastSnapshot.lead_days,
    ).where(
        WxForecastSnapshot.city == city,
        WxForecastSnapshot.target_date.between(start_date, end_date),
        WxForecastSnapshot.lead_days.in_([0, 1, 2]),  # Get recent forecasts
    ).order_by(WxForecastSnapshot.target_date, WxForecastSnapshot.lead_days)

    result = session.execute(query)
    df = pd.DataFrame(result.fetchall(), columns=[
        "target_date", "basis_date", "tempmax_fcst_f", "lead_days"
    ])
    return df


def load_settlement_data(
    session,
    city: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Load settlement data (actual high temps)."""
    query = select(
        WxSettlement.date_local.label("event_date"),
        WxSettlement.tmax_final,
    ).where(
        WxSettlement.city == city,
        WxSettlement.date_local.between(start_date, end_date),
    ).order_by(WxSettlement.date_local)

    result = session.execute(query)
    df = pd.DataFrame(result.fetchall(), columns=["event_date", "tmax_final"])
    return df


def load_market_data(
    session,
    city: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Load market metadata (brackets) including listed_at timestamp."""
    query = select(
        KalshiMarket.ticker,
        KalshiMarket.event_date,
        KalshiMarket.strike_type,
        KalshiMarket.floor_strike,
        KalshiMarket.cap_strike,
        KalshiMarket.result,
        KalshiMarket.listed_at,
    ).where(
        KalshiMarket.city == city,
        KalshiMarket.event_date.between(start_date, end_date),
    ).order_by(KalshiMarket.event_date, KalshiMarket.floor_strike)

    result = session.execute(query)
    df = pd.DataFrame(result.fetchall(), columns=[
        "ticker", "event_date", "strike_type", "floor_strike", "cap_strike", "result", "listed_at"
    ])
    return df


def get_forecast_at_open(
    forecast_df: pd.DataFrame,
    event_date: date,
    basis_offset_days: int = 1,
) -> Optional[Tuple[date, float]]:
    """
    Get the forecast for event_date using the specified basis offset.

    Args:
        forecast_df: DataFrame with forecast data
        event_date: The event date (target_date)
        basis_offset_days: How many days before event_date to get forecast from
                          1 = previous day's forecast (lead_days=1)
                          0 = same day forecast (lead_days=0)

    Returns:
        (basis_date, tempmax_fcst_f) or None if not found
    """
    # Filter for this event_date
    day_forecasts = forecast_df[forecast_df["target_date"] == event_date]

    if day_forecasts.empty:
        return None

    # Look for forecast with the right lead_days
    target_lead_days = basis_offset_days
    matching = day_forecasts[day_forecasts["lead_days"] == target_lead_days]

    if not matching.empty:
        row = matching.iloc[0]
        return row["basis_date"], float(row["tempmax_fcst_f"])

    # Fallback: use whatever forecast is available
    row = day_forecasts.iloc[0]
    return row["basis_date"], float(row["tempmax_fcst_f"])


# =============================================================================
# Backtest Logic
# =============================================================================

def run_backtest(
    cities: List[str],
    start_date: date,
    end_date: date,
    params: OpenMakerParams,
    strategy_name: str = "open_maker_v1",
) -> OpenMakerResult:
    """
    Run the open-maker backtest.

    For each (city, event_date):
    1. Get forecast at market open time (with basis_offset_days)
    2. Apply temp_bias_deg adjustment
    3. Find bracket for adjusted temp
    4. Calculate position size at entry_price
    5. Determine P&L based on settlement

    Args:
        cities: List of city IDs to backtest
        start_date: Start date for backtest
        end_date: End date for backtest
        params: Strategy parameters
        strategy_name: Name for this run

    Returns:
        OpenMakerResult with all trades
    """
    run_id = str(uuid.uuid4())[:8]
    result = OpenMakerResult(
        run_id=run_id,
        strategy_name=strategy_name,
        params=params,
        start_date=start_date,
        end_date=end_date,
        cities=cities,
    )

    logger.info(f"Starting open-maker backtest: {strategy_name}")
    logger.info(f"  Date range: {start_date} to {end_date}")
    logger.info(f"  Cities: {cities}")
    logger.info(f"  Params: entry={params.entry_price_cents}c, bias={params.temp_bias_deg}F, "
                f"offset={params.basis_offset_days}d, bet=${params.bet_amount_usd}")

    with get_db_session() as session:
        for city in cities:
            logger.info(f"Processing {city}...")

            # Load data for this city
            forecast_df = load_forecast_data(session, city, start_date, end_date)
            settlement_df = load_settlement_data(session, city, start_date, end_date)
            market_df = load_market_data(session, city, start_date, end_date)

            if forecast_df.empty:
                logger.warning(f"  No forecast data for {city}")
                continue
            if settlement_df.empty:
                logger.warning(f"  No settlement data for {city}")
                continue
            if market_df.empty:
                logger.warning(f"  No market data for {city}")
                continue

            # Process each event_date
            event_dates = sorted(settlement_df["event_date"].unique())

            for event_date in event_dates:
                # Get forecast at open
                forecast_result = get_forecast_at_open(
                    forecast_df,
                    event_date,
                    params.basis_offset_days,
                )

                if forecast_result is None:
                    logger.debug(f"  {event_date}: No forecast available")
                    continue

                basis_date, temp_fcst = forecast_result

                # Apply bias adjustment
                temp_adjusted = temp_fcst + params.temp_bias_deg

                # Find bracket for adjusted temperature
                bracket = find_bracket_for_temp(market_df, event_date, temp_adjusted)

                if bracket is None:
                    logger.debug(f"  {event_date}: No bracket for temp {temp_adjusted:.1f}F")
                    continue

                ticker, floor_strike, cap_strike = bracket

                # Get settlement result
                settlement_row = settlement_df[settlement_df["event_date"] == event_date]
                if settlement_row.empty:
                    logger.debug(f"  {event_date}: No settlement data")
                    continue

                tmax_final = float(settlement_row.iloc[0]["tmax_final"])

                # Determine if we won
                winning_bracket = find_bracket_for_temp(market_df, event_date, tmax_final)
                bin_won = winning_bracket is not None and winning_bracket[0] == ticker

                # Calculate position size
                num_contracts, amount_usd = calculate_position_size(
                    params.entry_price_cents,
                    params.bet_amount_usd,
                )

                # Calculate fees (maker = $0 for weather)
                fee_usd = kalshi_maker_fee(params.entry_price_cents, num_contracts)

                # Calculate P&L
                pnl_net = calculate_pnl(
                    params.entry_price_cents,
                    num_contracts,
                    bin_won,
                    fee_usd,
                )

                # Gross P&L (before fees)
                entry_price_dollars = params.entry_price_cents / 100
                if bin_won:
                    pnl_gross = num_contracts * (1.0 - entry_price_dollars)
                else:
                    pnl_gross = -num_contracts * entry_price_dollars

                # Create trade record
                trade = OpenMakerTrade(
                    city=city,
                    event_date=event_date,
                    ticker=ticker,
                    forecast_basis_date=basis_date,
                    temp_fcst_open=temp_fcst,
                    temp_adjusted=temp_adjusted,
                    floor_strike=floor_strike,
                    cap_strike=cap_strike,
                    entry_price_cents=params.entry_price_cents,
                    num_contracts=num_contracts,
                    amount_usd=amount_usd,
                    tmax_final=tmax_final,
                    bin_won=bin_won,
                    pnl_gross=pnl_gross,
                    fee_usd=fee_usd,
                    pnl_net=pnl_net,
                )

                result.trades.append(trade)

            logger.info(f"  {city}: {len([t for t in result.trades if t.city == city])} trades")

    logger.info(f"Backtest complete: {result.num_trades} total trades")
    return result


# =============================================================================
# Result Persistence
# =============================================================================

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


# =============================================================================
# Output Functions
# =============================================================================

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
    import random

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
        "--save", action="store_true",
        help="Save results to database"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print detailed info for 20 random trades"
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

    # Create params
    params = OpenMakerParams(
        entry_price_cents=args.price,
        temp_bias_deg=args.bias,
        basis_offset_days=args.offset,
        bet_amount_usd=args.bet_amount,
    )

    # Run backtest
    result = run_backtest(
        cities=cities,
        start_date=start_date,
        end_date=end_date,
        params=params,
    )

    # Print results
    print_results(result)

    # Debug output if requested
    if args.debug:
        print_debug_trades(result, n=20)

    # Save if requested
    if args.save:
        save_results_to_db(result)


if __name__ == "__main__":
    main()
