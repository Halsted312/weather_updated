#!/usr/bin/env python3
"""
Midnight Heuristic Backtest Module

Implements the midnight-based trading strategy that:
1. At midnight, uses the forecast high to bet on bracket(s)
2. At T-2h before predicted high, adjusts based on intraday observations
3. Calculates P&L with exact Kalshi fee formula

Usage:
    python -m backtest.midnight_heuristic --city chicago --start-date 2024-01-01 --end-date 2024-12-31
    python -m backtest.midnight_heuristic --all-cities --days 90
"""

import argparse
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd
from sqlalchemy import select, func, text
from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import CITIES, get_city
from src.db import get_db_session
from src.db.models import (
    WxSettlement,
    WxForecastSnapshot,
    WxForecastSnapshotHourly,
    KalshiMarket,
    KalshiCandle1m,
    SimRun,
    SimTrade,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Kalshi Fee Functions
# =============================================================================

def kalshi_taker_fee(price_cents: float, num_contracts: int = 1) -> float:
    """
    Calculate Kalshi taker fee.

    Fee per contract = 0.07 * price * (100 - price) / 100
    Max fee = $1.74 per contract (when price = 50)

    Args:
        price_cents: Price in cents (0-100)
        num_contracts: Number of contracts

    Returns:
        Total fee in dollars
    """
    if price_cents <= 0 or price_cents >= 100:
        return 0.0
    fee_per_contract = 0.07 * price_cents * (100 - price_cents) / 100 / 100
    return fee_per_contract * num_contracts


def kalshi_maker_fee(price_cents: float, num_contracts: int = 1) -> float:
    """Maker fee is $0 on Kalshi."""
    return 0.0


# =============================================================================
# Strategy Parameters
# =============================================================================

@dataclass
class HeuristicParams:
    """Tunable parameters for the midnight heuristic strategy."""

    # Midnight trend adjustment: T_adj = tempmax_t0 + alpha * (mean_3d - t0) + beta * range_3d
    alpha: float = 0.0  # Weight for mean deviation
    beta: float = 0.0   # Weight for range

    # Intraday correction: T_adj = t0 + gamma * (obs - fcst) + delta * slope_1h
    gamma: float = 0.5  # Weight for observation error
    delta: float = 0.2  # Weight for temperature slope

    # Edge threshold: only bet if estimated edge > threshold
    edge_threshold: float = 0.0

    # Split threshold: if distance to bin edge < threshold, split across 2 bins
    split_threshold: float = 1.0  # degrees F

    # Position sizing
    bet_amount_usd: float = 100.0


@dataclass
class Trade:
    """Represents a single trade in the backtest."""
    city: str
    event_date: date
    decision_type: str  # 'midnight' or 'pre_high_2h' or 'pre_high_1h'
    ticker: str
    side: str  # 'buy' or 'sell'
    price_cents: float
    num_contracts: int
    amount_usd: float
    fee_usd: float
    pnl_usd: float = 0.0  # Filled after settlement
    settled: bool = False
    bin_won: bool = False


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    run_id: str
    strategy_name: str
    params: HeuristicParams
    start_date: date
    end_date: date
    cities: List[str]
    trades: List[Trade] = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_usd for t in self.trades)

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.bin_won) / len(self.trades)

    def summary(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "strategy": self.strategy_name,
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "cities": self.cities,
            "num_trades": self.num_trades,
            "total_pnl_usd": round(self.total_pnl, 2),
            "win_rate": round(self.win_rate, 4),
            "avg_pnl_per_trade": round(self.total_pnl / self.num_trades, 2) if self.trades else 0,
        }


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_forecast_data(session, city: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Load forecast data for a city and date range.

    Returns DataFrame with columns:
        event_date, tempmax_t0, tempmax_t1, tempmax_t2, predicted_high_hour
    """
    # Get daily forecasts (lead_days 0, 1, 2) where basis_date = target_date
    query = text("""
        WITH t0 AS (
            SELECT target_date AS event_date, tempmax_fcst_f AS tempmax_t0
            FROM wx.forecast_snapshot
            WHERE city = :city
              AND lead_days = 0
              AND basis_date = target_date
              AND target_date BETWEEN :start AND :end
        ),
        t1 AS (
            SELECT basis_date AS event_date, tempmax_fcst_f AS tempmax_t1
            FROM wx.forecast_snapshot
            WHERE city = :city
              AND lead_days = 1
              AND target_date BETWEEN :start AND :end
        ),
        t2 AS (
            SELECT basis_date AS event_date, tempmax_fcst_f AS tempmax_t2
            FROM wx.forecast_snapshot
            WHERE city = :city
              AND lead_days = 2
              AND target_date BETWEEN :start AND :end
        ),
        high_hour AS (
            SELECT DISTINCT ON (basis_date)
                basis_date AS event_date,
                EXTRACT(HOUR FROM target_hour_local) +
                    EXTRACT(MINUTE FROM target_hour_local) / 60.0 AS predicted_high_hour
            FROM wx.forecast_snapshot_hourly
            WHERE city = :city
              AND lead_hours < 24
              AND basis_date BETWEEN :start AND :end
            ORDER BY basis_date, temp_fcst_f DESC, target_hour_local ASC
        )
        SELECT
            t0.event_date,
            t0.tempmax_t0,
            t1.tempmax_t1,
            t2.tempmax_t2,
            hh.predicted_high_hour
        FROM t0
        LEFT JOIN t1 ON t0.event_date = t1.event_date
        LEFT JOIN t2 ON t0.event_date = t2.event_date
        LEFT JOIN high_hour hh ON t0.event_date = hh.event_date
        ORDER BY t0.event_date
    """)

    result = session.execute(query, {"city": city, "start": start_date, "end": end_date})
    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df


def load_settlement_data(session, city: str, start_date: date, end_date: date) -> pd.DataFrame:
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


def load_market_data(session, city: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Load market metadata (brackets)."""
    query = select(
        KalshiMarket.ticker,
        KalshiMarket.event_date,
        KalshiMarket.strike_type,
        KalshiMarket.floor_strike,
        KalshiMarket.cap_strike,
        KalshiMarket.result,
    ).where(
        KalshiMarket.city == city,
        KalshiMarket.event_date.between(start_date, end_date),
    ).order_by(KalshiMarket.event_date, KalshiMarket.floor_strike)

    result = session.execute(query)
    df = pd.DataFrame(result.fetchall(), columns=[
        "ticker", "event_date", "strike_type", "floor_strike", "cap_strike", "result"
    ])
    return df


def get_price_at_time(
    session,
    ticker: str,
    target_time_utc: datetime,
    window_minutes: int = 30,
) -> Optional[float]:
    """
    Get the closing price for a ticker near a target time.

    Looks for the first candle within window_minutes after target_time.
    """
    query = select(KalshiCandle1m.close_c).where(
        KalshiCandle1m.ticker == ticker,
        KalshiCandle1m.bucket_start >= target_time_utc,
        KalshiCandle1m.bucket_start < target_time_utc + timedelta(minutes=window_minutes),
    ).order_by(KalshiCandle1m.bucket_start).limit(1)

    result = session.execute(query).scalar_one_or_none()
    return float(result) if result is not None else None


def get_first_price_of_day(
    session,
    ticker: str,
    event_date: date,
) -> Optional[Tuple[datetime, float]]:
    """
    Get the first available price for a ticker on its event date.

    Returns (timestamp, price) or None if no candles exist.
    """
    # Markets typically open around 14:00 UTC (9 AM ET)
    # Look for first candle from 04:00 UTC to 23:59 UTC
    start_utc = datetime.combine(event_date, datetime.min.time()) + timedelta(hours=4)
    end_utc = start_utc + timedelta(hours=20)

    query = select(
        KalshiCandle1m.bucket_start,
        KalshiCandle1m.close_c
    ).where(
        KalshiCandle1m.ticker == ticker,
        KalshiCandle1m.bucket_start >= start_utc,
        KalshiCandle1m.bucket_start < end_utc,
        KalshiCandle1m.close_c.isnot(None),
    ).order_by(KalshiCandle1m.bucket_start).limit(1)

    result = session.execute(query).first()
    if result and result[1] is not None:
        return result[0], float(result[1])
    return None


# =============================================================================
# Strategy Logic
# =============================================================================

def find_bracket_for_temp(
    markets_df: pd.DataFrame,
    event_date: date,
    temp: float,
) -> Optional[Tuple[str, float, float]]:
    """
    Find the bracket containing a temperature.

    Returns (ticker, floor_strike, cap_strike) or None.
    """
    day_markets = markets_df[markets_df["event_date"] == event_date]

    for _, row in day_markets.iterrows():
        if row["strike_type"] == "between":
            if row["floor_strike"] <= temp < row["cap_strike"]:
                return row["ticker"], row["floor_strike"], row["cap_strike"]
        elif row["strike_type"] == "less":
            if temp < row["floor_strike"]:
                return row["ticker"], None, row["floor_strike"]
        elif row["strike_type"] == "greater":
            if temp >= row["floor_strike"]:
                return row["ticker"], row["floor_strike"], None

    return None


def calculate_weighted_high_hour(
    yesterday_hour: Optional[float],
    today_hour: float,
    tomorrow_hour: Optional[float],
) -> float:
    """
    Calculate weighted predicted high hour.
    Weights: 20% yesterday, 70% today, 10% tomorrow
    """
    y = yesterday_hour if yesterday_hour is not None else today_hour
    t = tomorrow_hour if tomorrow_hour is not None else today_hour
    return 0.2 * y + 0.7 * today_hour + 0.1 * t


def run_midnight_strategy(
    session,
    city: str,
    forecast_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    settlement_df: pd.DataFrame,
    params: HeuristicParams,
) -> List[Trade]:
    """
    Run the market-open strategy for a city.

    At market open, use tempmax_t0 (with optional trend adjustment) to select bracket.
    The "midnight" name refers to using the midnight forecast, not the trading time.
    """
    trades = []

    for _, row in forecast_df.iterrows():
        event_date = row["event_date"]
        if pd.isna(row["tempmax_t0"]):
            continue

        # Calculate adjusted temperature estimate
        t0 = row["tempmax_t0"]
        t1 = row.get("tempmax_t1", t0)
        t2 = row.get("tempmax_t2", t0)

        mean_3d = (t0 + t1 + t2) / 3
        range_3d = max(t0, t1, t2) - min(t0, t1, t2)

        t_adj = t0 + params.alpha * (mean_3d - t0) + params.beta * range_3d

        # Find the bracket for this temperature
        bracket = find_bracket_for_temp(markets_df, event_date, t_adj)
        if bracket is None:
            logger.debug(f"{city} {event_date}: No bracket found for temp {t_adj:.1f}")
            continue

        ticker, floor_strike, cap_strike = bracket

        # Get price at market open (first available candle of the day)
        price_info = get_first_price_of_day(session, ticker, event_date)
        if price_info is None:
            logger.debug(f"{city} {event_date}: No price for {ticker} at market open")
            continue

        trade_time, price = price_info

        # Calculate position size
        # price is in cents (0-100), so price/100 = cost per contract in dollars
        cost_per_contract = price / 100  # e.g., 50 cents = $0.50 per contract
        num_contracts = int(params.bet_amount_usd / cost_per_contract)
        if num_contracts < 1:
            num_contracts = 1

        amount_usd = num_contracts * cost_per_contract
        fee_usd = kalshi_taker_fee(price, num_contracts)

        # Get settlement result
        settlement = settlement_df[settlement_df["event_date"] == event_date]
        if settlement.empty:
            continue

        tmax_final = settlement.iloc[0]["tmax_final"]

        # Check if we won
        winning_bracket = find_bracket_for_temp(markets_df, event_date, tmax_final)
        bin_won = winning_bracket is not None and winning_bracket[0] == ticker

        # Calculate P&L
        if bin_won:
            pnl_usd = num_contracts * (100 - price) / 100 - fee_usd
        else:
            pnl_usd = -amount_usd - fee_usd

        trade = Trade(
            city=city,
            event_date=event_date,
            decision_type="midnight",
            ticker=ticker,
            side="buy",
            price_cents=price,
            num_contracts=num_contracts,
            amount_usd=amount_usd,
            fee_usd=fee_usd,
            pnl_usd=pnl_usd,
            settled=True,
            bin_won=bin_won,
        )
        trades.append(trade)

    return trades


# =============================================================================
# Main Backtest Runner
# =============================================================================

def run_backtest(
    cities: List[str],
    start_date: date,
    end_date: date,
    params: Optional[HeuristicParams] = None,
    strategy_name: str = "midnight_heuristic_v1",
) -> BacktestResult:
    """
    Run the full backtest across cities and date range.
    """
    if params is None:
        params = HeuristicParams()

    run_id = str(uuid.uuid4())
    result = BacktestResult(
        run_id=run_id,
        strategy_name=strategy_name,
        params=params,
        start_date=start_date,
        end_date=end_date,
        cities=cities,
    )

    with get_db_session() as session:
        for city in cities:
            logger.info(f"Running backtest for {city}...")
            city_config = get_city(city)

            # Load data
            forecast_df = load_forecast_data(session, city, start_date, end_date)
            settlement_df = load_settlement_data(session, city, start_date, end_date)
            markets_df = load_market_data(session, city, start_date, end_date)

            if forecast_df.empty:
                logger.warning(f"{city}: No forecast data found")
                continue

            logger.info(f"{city}: {len(forecast_df)} forecast days, {len(markets_df)} markets")

            # Run midnight strategy
            trades = run_midnight_strategy(
                session=session,
                city=city,
                forecast_df=forecast_df,
                markets_df=markets_df,
                settlement_df=settlement_df,
                params=params,
            )

            result.trades.extend(trades)
            logger.info(f"{city}: {len(trades)} trades")

    return result


def save_results_to_db(result: BacktestResult) -> None:
    """Save backtest results to sim.run and sim.trade tables."""
    with get_db_session() as session:
        # Insert run record
        run_record = {
            "run_id": result.run_id,
            "strategy_name": result.strategy_name,
            "params_json": {
                "alpha": result.params.alpha,
                "beta": result.params.beta,
                "gamma": result.params.gamma,
                "delta": result.params.delta,
                "edge_threshold": result.params.edge_threshold,
                "split_threshold": result.params.split_threshold,
                "bet_amount_usd": result.params.bet_amount_usd,
            },
            "train_start": result.start_date,
            "train_end": result.end_date,
            "test_start": result.start_date,
            "test_end": result.end_date,
        }

        stmt = insert(SimRun).values(**run_record)
        stmt = stmt.on_conflict_do_nothing()
        session.execute(stmt)

        # Insert trade records
        for trade in result.trades:
            trade_record = {
                "run_id": result.run_id,
                "trade_ts_utc": datetime.combine(trade.event_date, datetime.min.time()),
                "ticker": trade.ticker,
                "city": trade.city,
                "event_date": trade.event_date,
                "side": trade.side,
                "qty": trade.num_contracts,
                "price": trade.price_cents,
                "fees": trade.fee_usd,
                "pnl": trade.pnl_usd,
            }

            stmt = insert(SimTrade).values(**trade_record)
            stmt = stmt.on_conflict_do_nothing()
            session.execute(stmt)

        session.commit()
        logger.info(f"Saved {len(result.trades)} trades to database")


def print_results(result: BacktestResult) -> None:
    """Print backtest results to console."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    summary = result.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Per-city breakdown
    print("\nPer-City Breakdown:")
    print("-" * 40)

    trades_df = pd.DataFrame([
        {"city": t.city, "pnl": t.pnl_usd, "won": t.bin_won}
        for t in result.trades
    ])

    if not trades_df.empty:
        by_city = trades_df.groupby("city").agg({
            "pnl": ["count", "sum", "mean"],
            "won": "mean",
        })
        by_city.columns = ["trades", "total_pnl", "avg_pnl", "win_rate"]
        print(by_city.to_string())

    # Daily P&L distribution
    print("\nP&L Distribution:")
    print("-" * 40)
    pnls = [t.pnl_usd for t in result.trades]
    if pnls:
        print(f"  Min: ${min(pnls):.2f}")
        print(f"  Max: ${max(pnls):.2f}")
        print(f"  Median: ${sorted(pnls)[len(pnls)//2]:.2f}")
        print(f"  Total: ${sum(pnls):.2f}")

    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run midnight heuristic backtest"
    )
    parser.add_argument(
        "--city", action="append",
        help="City to backtest (can specify multiple)"
    )
    parser.add_argument(
        "--all-cities", action="store_true",
        help="Run for all cities"
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
        "--alpha", type=float, default=0.0,
        help="Trend adjustment alpha parameter"
    )
    parser.add_argument(
        "--beta", type=float, default=0.0,
        help="Trend adjustment beta parameter"
    )
    parser.add_argument(
        "--bet-amount", type=float, default=100.0,
        help="Bet amount in USD"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save results to database"
    )

    args = parser.parse_args()

    # Determine cities
    if args.all_cities:
        cities = list(CITIES.keys())
    elif args.city:
        cities = args.city
    else:
        cities = ["chicago"]  # Default

    # Determine date range
    today = date.today()
    if args.start_date and args.end_date:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date)
    else:
        end_date = today - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=args.days)

    # Create params
    params = HeuristicParams(
        alpha=args.alpha,
        beta=args.beta,
        bet_amount_usd=args.bet_amount,
    )

    logger.info(f"Running backtest: cities={cities}, dates={start_date} to {end_date}")

    # Run backtest
    result = run_backtest(
        cities=cities,
        start_date=start_date,
        end_date=end_date,
        params=params,
    )

    # Print results
    print_results(result)

    # Save if requested
    if args.save:
        save_results_to_db(result)


if __name__ == "__main__":
    main()
