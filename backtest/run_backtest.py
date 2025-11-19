#!/usr/bin/env python3
"""
Backtest harness for Kalshi weather markets.

Loads market + settlement data, runs strategy, calculates performance metrics.
"""

import logging
import argparse
import json
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import text

from db.connection import get_session
from backtest.portfolio import Portfolio
from backtest.outcome import resolve_bin
from backtest.fees import breakeven_price_cents
from backtest.strategy import Strategy, Signal
from backtest.risk import RiskManager, RiskLimits
from backtest.diagnostics import TradeDiagnostics
from backtest.model_strategy import ExecParams

logger = logging.getLogger(__name__)


def _minutes_from_hhmm(value: str) -> int:
    hour, minute = map(int, value.split(":"))
    return hour * 60 + minute


def parse_time_window_values(values: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    windows: Dict[str, List[Tuple[int, int]]] = {}
    for raw in values:
        if not raw:
            continue
        if "@" in raw:
            strike_key, window_str = raw.split("@", 1)
        else:
            strike_key, window_str = "all", raw
        strike_key = strike_key.strip().lower()
        start_str, end_str = window_str.split("-")
        window = (_minutes_from_hhmm(start_str.strip()), _minutes_from_hhmm(end_str.strip()))
        windows.setdefault(strike_key, []).append(window)
    return windows


def parse_time_overrides(values: List[str]) -> List[Dict[str, object]]:
    overrides: List[Dict[str, object]] = []
    for raw in values:
        if not raw:
            continue
        base, _, params_str = raw.partition(":")
        if "@" in base:
            strike_key, window_str = base.split("@", 1)
        else:
            strike_key, window_str = "all", base
        start_str, end_str = window_str.split("-")
        override: Dict[str, object] = {
            "window": (
                _minutes_from_hhmm(start_str.strip()),
                _minutes_from_hhmm(end_str.strip()),
            )
        }
        strike_key = strike_key.strip().lower()
        if strike_key != "all":
            override["strike_type"] = strike_key
        for token in filter(None, params_str.split(",")):
            key, _, value = token.partition("=")
            key = key.strip()
            value = value.strip()
            if key == "max_spread":
                override["max_spread_cents"] = int(value)
            elif key == "tau_open":
                override["tau_open_cents"] = int(value)
        overrides.append(override)
    return overrides


def parse_bracket_spreads(values: List[str]) -> Dict[str, int]:
    spreads: Dict[str, int] = {}
    for raw in values:
        if not raw or "=" not in raw:
            continue
        strike, value = raw.split("=", 1)
        spreads[strike.strip().lower()] = int(value)
    return spreads


def load_markets_with_settlements(
    city: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Load markets and their settlements for backtesting.

    Args:
        city: City name (chicago, new_york, etc.)
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        DataFrame with columns:
            ticker, series_ticker, event_ticker, title,
            open_time, close_time, expiration_time,
            floor_strike, cap_strike, strike_type,
            result, settlement_value,
            date_local, tmax_final, source_final
    """
    with get_session() as session:
        query = text("""
            SELECT
                m.ticker,
                m.series_ticker,
                m.event_ticker,
                m.title,
                m.subtitle,
                m.open_time,
                m.close_time,
                m.expiration_time,
                m.floor_strike,
                m.cap_strike,
                m.strike_type,
                m.result,
                m.settlement_value,
                m.status,
                (DATE(m.close_time AT TIME ZONE 'America/Chicago') - INTERVAL '1 day')::date as date_local,
                s.tmax_final,
                s.source_final
            FROM markets m
            LEFT JOIN wx.settlement s ON (
                s.city = :city
                AND s.date_local = (DATE(m.close_time AT TIME ZONE 'America/Chicago') - INTERVAL '1 day')::date
            )
            WHERE m.series_ticker LIKE :series_pattern
              AND m.status IN ('closed', 'settled', 'finalized')
              AND (DATE(m.close_time AT TIME ZONE 'America/Chicago') - INTERVAL '1 day')::date >= :start_date
              AND (DATE(m.close_time AT TIME ZONE 'America/Chicago') - INTERVAL '1 day')::date <= :end_date
            ORDER BY (DATE(m.close_time AT TIME ZONE 'America/Chicago') - INTERVAL '1 day')::date, m.ticker
        """)

        # Map city to series ticker pattern
        series_map = {
            "chicago": "KXHIGHCHI%",
            "austin": "KXHIGHAUS%",
            "miami": "KXHIGHMIA%",
            "los_angeles": "KXHIGHLAX%",
            "la": "KXHIGHLAX%",
            "denver": "KXHIGHDEN%",
            "philadelphia": "KXHIGHPHIL%",
        }

        if city not in series_map:
            raise ValueError(f"Unknown city: {city}. Available: {list(series_map.keys())}")

        result = session.execute(query, {
            "city": city,
            "series_pattern": series_map[city],
            "start_date": start_date,
            "end_date": end_date,
        }).fetchall()

        # Convert to DataFrame
        df = pd.DataFrame(result, columns=[
            "ticker", "series_ticker", "event_ticker", "title", "subtitle",
            "open_time", "close_time", "expiration_time",
            "floor_strike", "cap_strike", "strike_type",
            "result", "settlement_value", "status",
            "date_local", "tmax_final", "source_final",
        ])

        logger.info(f"Loaded {len(df)} markets for {city} ({start_date} to {end_date})")

        return df


def calculate_outcome_from_settlement(row: pd.Series) -> Optional[str]:
    """
    Calculate market outcome from settlement temperature.

    Args:
        row: DataFrame row with floor_strike, cap_strike, strike_type, tmax_final

    Returns:
        "YES" or "NO", or None if missing settlement data
    """
    if pd.isna(row["tmax_final"]):
        return None

    try:
        outcome = resolve_bin(
            tmax_f=float(row["tmax_final"]),
            floor_strike=int(row["floor_strike"]) if pd.notna(row["floor_strike"]) else None,
            cap_strike=int(row["cap_strike"]) if pd.notna(row["cap_strike"]) else None,
            strike_type=row["strike_type"],
        )
        return outcome
    except Exception as e:
        logger.error(f"Error resolving {row['ticker']}: {e}")
        return None


def run_buyhold_backtest(
    markets_df: pd.DataFrame,
    initial_cash_cents: int = 10_000_00,
    entry_price_cents: int = 50,
    contracts_per_market: int = 1,
) -> Dict:
    """
    Run a simple buy-and-hold backtest.

    Strategy: Buy N contracts of each market at fixed price, hold to settlement.

    Args:
        markets_df: DataFrame from load_markets_with_settlements()
        initial_cash_cents: Starting cash in cents
        entry_price_cents: Fixed entry price in cents (e.g., 50¢)
        contracts_per_market: Number of contracts per market

    Returns:
        Dict with portfolio summary + equity curve
    """
    portfolio = Portfolio(initial_cash_cents=initial_cash_cents)

    # Add calculated outcome column
    markets_df["calc_outcome"] = markets_df.apply(calculate_outcome_from_settlement, axis=1)

    # Track equity over time
    equity_curve = []

    # Group by event (date)
    for event_date, event_markets in markets_df.groupby("date_local"):
        if pd.isna(event_date):
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Event: {event_date}")
        logger.info(f"{'='*60}")

        # Entry: Buy all markets for this event
        for _, market in event_markets.iterrows():
            if pd.isna(market["calc_outcome"]):
                logger.warning(f"Skipping {market['ticker']}: no settlement data")
                continue

            # Execute buy
            success = portfolio.execute_trade(
                timestamp=market["open_time"],
                market_ticker=market["ticker"],
                side="buy",
                contracts=contracts_per_market,
                price_cents=entry_price_cents,
                fee_type="taker",
            )

            if not success:
                logger.warning(f"Insufficient cash for {market['ticker']}")
                break

        # Settlement: Settle all markets for this event
        for _, market in event_markets.iterrows():
            if pd.isna(market["calc_outcome"]):
                continue

            portfolio.settle_market(
                timestamp=market["expiration_time"],
                market_ticker=market["ticker"],
                result=market["calc_outcome"],
            )

        # Record equity
        equity = portfolio.get_equity_cents()
        equity_curve.append({
            "date": event_date,
            "equity_cents": equity,
            "equity_dollars": equity / 100,
        })

        logger.info(f"Equity: ${equity/100:,.2f}")

    # Calculate metrics
    summary = portfolio.get_summary()

    # Add equity curve
    equity_df = pd.DataFrame(equity_curve)

    if len(equity_df) > 1:
        # Calculate returns
        equity_df["returns"] = equity_df["equity_dollars"].pct_change()

        # Sharpe ratio (annualized, assuming ~250 trading days/year)
        mean_return = equity_df["returns"].mean()
        std_return = equity_df["returns"].std()

        if std_return > 0:
            sharpe = (mean_return / std_return) * np.sqrt(250)
        else:
            sharpe = 0.0

        # Max drawdown
        equity_df["cum_max"] = equity_df["equity_dollars"].cummax()
        equity_df["drawdown"] = (equity_df["equity_dollars"] - equity_df["cum_max"]) / equity_df["cum_max"]
        max_drawdown = equity_df["drawdown"].min()

        summary["sharpe_ratio"] = sharpe
        summary["max_drawdown_pct"] = max_drawdown * 100
        summary["equity_curve"] = equity_df

    else:
        summary["sharpe_ratio"] = 0.0
        summary["max_drawdown_pct"] = 0.0
        summary["equity_curve"] = equity_df

    return summary


def run_strategy_backtest(
    strategy: Strategy,
    markets_df: pd.DataFrame,
    city: str,
    initial_cash_cents: int = 10_000_00,
    risk_manager: Optional[RiskManager] = None,
) -> Dict:
    """
    Run backtest with pluggable strategy.

    Args:
        strategy: Strategy instance (must implement Strategy interface)
        markets_df: DataFrame from load_markets_with_settlements()
        city: City name (for risk management)
        initial_cash_cents: Starting cash in cents
        risk_manager: Optional RiskManager (creates default if None)

    Returns:
        Dict with portfolio summary + equity curve
    """
    portfolio = Portfolio(initial_cash_cents=initial_cash_cents)
    diagnostics = TradeDiagnostics()

    if risk_manager is None:
        risk_manager = RiskManager()

    # Add calculated outcome column
    markets_df["calc_outcome"] = markets_df.apply(calculate_outcome_from_settlement, axis=1)

    # Track equity over time
    equity_curve = []

    # Convert positions to dict format expected by strategy
    def get_positions_dict():
        from backtest.strategy import Position as StrategyPosition
        positions_dict = {}
        for ticker, pos in portfolio.positions.items():
            if pos.contracts != 0:
                positions_dict[ticker] = StrategyPosition(
                    market_ticker=ticker,
                    contracts=pos.contracts,
                    avg_entry_price=pos.avg_entry_price,
                    entry_time=datetime.now(),  # Not tracked in Portfolio
                    unrealized_pnl_cents=0.0,  # Calculate if needed
                )
        return positions_dict

    # Group by event (date)
    for event_date, event_markets in markets_df.groupby("date_local"):
        if pd.isna(event_date):
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Event: {event_date}")
        logger.info(f"{'='*60}")

        # Get current positions
        positions = get_positions_dict()

        # Use market close time as the decision timestamp for historical backtest
        # This represents when the strategy would make its final decision before market closes
        decision_timestamp = pd.to_datetime(event_markets.iloc[0]["close_time"])

        # Generate signals from strategy
        signals = strategy.generate_signals(
            timestamp=decision_timestamp,
            market_data=event_markets,
            positions=positions,
            bankroll_cents=portfolio.cash_cents,
        )

        logger.info(f"Strategy generated {len(signals)} signals")

        # Execute trades based on signals
        for signal in signals:
            if signal.action not in ["buy", "sell"]:
                continue

            # Find market row
            market_row = event_markets[event_markets["ticker"] == signal.market_ticker]
            if market_row.empty:
                logger.warning(f"Market {signal.market_ticker} not found")
                continue

            market = market_row.iloc[0]

            # Skip if no settlement data
            if pd.isna(market["calc_outcome"]):
                logger.warning(f"Skipping {signal.market_ticker}: no settlement data")
                continue

            # Calculate contracts from size_fraction
            # For simplicity, use actual price if provided
            # (Strategy should use Kelly sizing internally)
            signal_price = getattr(signal, "price_cents", None)
            if signal_price is None or signal_price <= 0:
                price_cents = 50
            else:
                price_cents = int(signal_price)
            max_capital = int(signal.size_fraction * portfolio.cash_cents)
            contracts = max(1, max_capital // price_cents)

            # Check risk limits
            allowed, reason = risk_manager.check_trade(
                market_ticker=signal.market_ticker,
                city=city,
                event_date=event_date,
                side="long" if signal.action == "buy" else "short",
                contracts=contracts,
                price_cents=price_cents,
                bankroll_cents=portfolio.cash_cents,
            )

            if not allowed:
                logger.warning(f"Trade rejected by risk manager: {reason}")
                continue

            # Determine fee type: Maker-first simulation
            # Estimate maker fill probability based on time and spread
            fee_type = "taker"  # Default to taker

            if signal.time_to_close_minutes is not None and signal.spread_cents is not None:
                # Probabilistic maker fill model
                import random

                # Base fill rate: 65%
                fill_prob = 0.65

                # More time → higher fill probability
                if signal.time_to_close_minutes > 120:
                    fill_prob += 0.15  # +15% if >2 hours
                elif signal.time_to_close_minutes > 60:
                    fill_prob += 0.10  # +10% if >1 hour

                # Tighter spread → higher fill probability
                if signal.spread_cents <= 2:
                    fill_prob += 0.10  # +10% for tight spread
                elif signal.spread_cents > 4:
                    fill_prob -= 0.15  # -15% for wide spread

                # Simulate fill
                if random.random() < fill_prob:
                    fee_type = "maker"

            # Execute trade
            success = portfolio.execute_trade(
                timestamp=market["open_time"],
                market_ticker=signal.market_ticker,
                side=signal.action,
                contracts=contracts,
                price_cents=price_cents,
                fee_type=fee_type,
                # Diagnostic fields from signal
                p_model=signal.confidence,
                p_market=signal.p_market,
                edge_cents=signal.edge,
                spread_cents=signal.spread_cents,
                time_to_close_minutes=signal.time_to_close_minutes,
            )

            if success:
                # Record with risk manager
                risk_manager.record_trade(
                    market_ticker=signal.market_ticker,
                    city=city,
                    event_date=event_date,
                    side="long" if signal.action == "buy" else "short",
                    contracts=contracts,
                    price_cents=price_cents,
                )
                logger.info(
                    f"Executed: {signal.action.upper()} {contracts} {signal.market_ticker} @ {price_cents}¢"
                    f" (edge: {signal.edge:.1f}¢, reason: {signal.reason})"
                )
            else:
                logger.warning(f"Insufficient cash for {signal.market_ticker}")

        # Settlement: Settle all markets for this event
        for _, market in event_markets.iterrows():
            if pd.isna(market["calc_outcome"]):
                continue

            portfolio.settle_market(
                timestamp=market["expiration_time"],
                market_ticker=market["ticker"],
                result=market["calc_outcome"],
            )

        # Settle event in risk manager
        risk_manager.settle_event(city, event_date)

        # Record equity
        equity = portfolio.get_equity_cents()
        equity_curve.append({
            "date": event_date,
            "equity_cents": equity,
            "equity_dollars": equity / 100,
        })

        logger.info(f"Equity: ${equity/100:,.2f}")

    # Calculate metrics
    summary = portfolio.get_summary()
    summary["strategy_name"] = strategy.get_name()

    # Populate diagnostics from trades
    # Create outcome lookup
    outcome_map = markets_df.set_index("ticker")["calc_outcome"].to_dict()

    for trade in portfolio.trades:
        # Get settlement outcome for this market
        outcome = outcome_map.get(trade.market_ticker)

        diagnostics.add_trade(
            timestamp=trade.timestamp,
            market_ticker=trade.market_ticker,
            side=trade.side,
            contracts=trade.contracts,
            price_cents=trade.price_cents,
            fee_type=trade.fee_type,
            fee_cents=trade.fee_cents,
            p_model=trade.p_model,
            p_market=trade.p_market,
            edge_cents=trade.edge_cents,
            spread_cents=trade.spread_cents,
            time_to_close_minutes=trade.time_to_close_minutes,
            outcome=outcome,
            pnl_cents=trade.pnl_cents,
        )

    # Compute diagnostic metrics
    diag_metrics = diagnostics.compute_metrics()
    summary["diagnostics"] = diag_metrics
    summary["diagnostics_obj"] = diagnostics  # Store for saving later

    # Add equity curve
    equity_df = pd.DataFrame(equity_curve)

    if len(equity_df) > 1:
        # Calculate returns
        equity_df["returns"] = equity_df["equity_dollars"].pct_change()

        # Sharpe ratio (annualized, assuming ~250 trading days/year)
        mean_return = equity_df["returns"].mean()
        std_return = equity_df["returns"].std()

        if std_return > 0:
            sharpe = (mean_return / std_return) * np.sqrt(250)
        else:
            sharpe = 0.0

        # Max drawdown
        equity_df["cum_max"] = equity_df["equity_dollars"].cummax()
        equity_df["drawdown"] = (equity_df["equity_dollars"] - equity_df["cum_max"]) / equity_df["cum_max"]
        max_drawdown = equity_df["drawdown"].min()

        summary["sharpe_ratio"] = sharpe
        summary["max_drawdown_pct"] = max_drawdown * 100
        summary["equity_curve"] = equity_df

    else:
        summary["sharpe_ratio"] = 0.0
        summary["max_drawdown_pct"] = 0.0
        summary["equity_curve"] = equity_df

    return summary


def print_backtest_summary(summary: Dict):
    """Print backtest results in a formatted table."""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60 + "\n")

    if "strategy_name" in summary:
        print(f"Strategy: {summary['strategy_name']}\n")

    print("Portfolio Summary:")
    print("-" * 60)
    print(f"Initial capital:     ${summary['initial_cash_cents']/100:>12,.2f}")
    print(f"Final equity:        ${summary['equity_cents']/100:>12,.2f}")
    print(f"Total P&L:           ${summary['total_pnl_cents']/100:>12,.2f}")
    print(f"Total fees:          ${summary['total_fees_cents']/100:>12,.2f}")
    print(f"Gross P&L:           ${summary['gross_pnl_cents']/100:>12,.2f}")
    print(f"Total return:        {summary['total_return_pct']:>12.2f}%")
    print()

    print("Trade Statistics:")
    print("-" * 60)
    print(f"Num trades:          {summary['num_trades']:>12,}")
    print(f"Num settlements:     {summary['num_settlements']:>12,}")
    print(f"Open positions:      {summary['num_open_positions']:>12}")
    print()

    print("Risk Metrics:")
    print("-" * 60)
    print(f"Sharpe ratio:        {summary['sharpe_ratio']:>12.2f}")
    print(f"Max drawdown:        {summary['max_drawdown_pct']:>12.2f}%")
    print()

    # Show equity curve summary
    if "equity_curve" in summary and len(summary["equity_curve"]) > 0:
        df = summary["equity_curve"]
        print("Equity Curve (first 10 days):")
        print("-" * 60)
        print(df.head(10).to_string(index=False))
        print()

        if len(df) > 10:
            print("Equity Curve (last 10 days):")
            print("-" * 60)
            print(df.tail(10).to_string(index=False))
            print()

    print("="*60 + "\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run backtest on Kalshi weather markets"
    )
    parser.add_argument(
        "--city",
        type=str,
        default="chicago",
        help="City name (chicago, new_york, etc.)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=False,  # Optional for model_kelly (can auto-detect)
        help="Start date (YYYY-MM-DD, optional for model_kelly - will auto-detect from predictions)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=False,  # Optional for model_kelly (can auto-detect)
        help="End date (YYYY-MM-DD, optional for model_kelly - will auto-detect from predictions)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["buyhold", "model_kelly"],
        default="buyhold",
        help="Strategy to run (default: buyhold)",
    )
    parser.add_argument(
        "--bracket",
        type=str,
        choices=["between", "greater", "less"],
        default="between",
        help="Bracket type for model_kelly strategy (default: between)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/trained",
        help="Directory with trained models (for model_kelly, default: models/trained)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=10_000.0,
        help="Initial cash in dollars (default: 10000)",
    )
    parser.add_argument(
        "--entry-price",
        type=int,
        default=50,
        help="Fixed entry price in cents for buyhold (default: 50)",
    )
    parser.add_argument(
        "--contracts",
        type=int,
        default=1,
        help="Contracts per market for buyhold (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save backtest summary JSON",
    )
    parser.add_argument(
        "--unified-head",
        action="store_true",
        help="Use unified coupling over 6 brackets per timestamp",
    )
    parser.add_argument(
        "--unified-tau",
        type=float,
        default=1.0,
        help="Temperature for unified head softmax coupling (default: 1.0)",
    )
    parser.add_argument(
        "--model-type",
        choices=["elasticnet", "catboost", "ev_catboost", "tmax_reg"],
        default="elasticnet",
        help="Model type to use for predictions (default: elasticnet)",
    )
    parser.add_argument(
        "--ev-models-dir",
        type=str,
        default=None,
        help="Directory with EV predictions for gating/blending (optional)",
    )
    parser.add_argument(
        "--ev-min-delta",
        type=float,
        default=0.0,
        help="Minimum EV delta (cents) required in trade direction when using EV gating",
    )
    parser.add_argument(
        "--ev-blend-weight",
        type=float,
        default=0.0,
        help="Blend weight (0-1) for EV probability when --ev-models-dir is set",
    )
    parser.add_argument(
        "--ev-max-staleness",
        type=float,
        default=90.0,
        help="Skip EV signals older than this many minutes (default: 90)",
    )
    parser.add_argument(
        "--ev-allow-missing",
        action="store_true",
        help="Allow trades when EV prediction is missing (default: skip)",
    )
    parser.add_argument(
        "--tmax-preds-csv",
        type=str,
        default=None,
        help="CSV with per-minute Tmax ensemble predictions (for model_type=tmax_reg)",
    )
    parser.add_argument(
        "--tmax-min-prob",
        type=float,
        default=0.0,
        help="Minimum confidence (max(p,1-p)) required for Tmax trades",
    )
    parser.add_argument(
        "--tmax-sigma-multiplier",
        type=float,
        default=0.0,
        help="Minimum sigma multiples away from nearest boundary for Tmax trades",
    )
    parser.add_argument(
        "--hybrid-model-type",
        choices=["elasticnet", "catboost", "ev_catboost"],
        default=None,
        help="Optional settlement model type to require agreement with",
    )
    parser.add_argument(
        "--hybrid-models-dir",
        type=str,
        default=None,
        help="Directory containing hybrid model predictions (defaults to --models-dir)",
    )
    parser.add_argument(
        "--hybrid-min-prob",
        type=float,
        default=0.0,
        help="Minimum confidence (max(p,1-p)) required from hybrid model",
    )
    parser.add_argument("--exec-max-spread", type=int, default=3)
    parser.add_argument("--exec-slippage", type=int, default=1)
    parser.add_argument("--exec-tau-open", type=int, default=5)
    parser.add_argument("--exec-tau-close", type=float, default=0.5)
    parser.add_argument("--exec-alpha-kelly", type=float, default=0.25)
    parser.add_argument("--exec-max-trade", type=float, default=0.02)
    parser.add_argument("--exec-max-city-day", type=float, default=0.10)
    parser.add_argument("--exec-bracket-spread", action="append", default=[], help="Override spreads per strike_type (e.g., between=2)")
    parser.add_argument("--exec-time-window", action="append", default=[], help="Allowed windows strike@HH:MM-HH:MM (strike optional)")
    parser.add_argument("--exec-time-override", action="append", default=[], help="Override thresholds strike@HH:MM-HH:MM:max_spread=2,tau_open=7")
    parser.add_argument("--exec-sigma-gate", type=float, help="Skip trades when sigma_est exceeds this value")
    parser.add_argument("--exec-humidity-gate", type=float, help="Skip trades when humidity metric exceeds this value")
    parser.add_argument("--market-odds-weight", type=float, default=0.0, help="Log-odds blend weight for market odds (0-1)")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # A5: Auto-detect window from prediction files if dates not provided
    if args.strategy == "model_kelly" and args.models_dir:
        if not args.start_date or not args.end_date:
            from pathlib import Path
            import pandas as pd

            print(f"[DEBUG A5] Auto-detecting date window from prediction files...")
            preds_timestamps = []
            if args.model_type == "catboost":
                base_path = Path(args.models_dir) / args.city / f"{args.bracket}_catboost"
            elif args.model_type == "ev_catboost":
                base_path = Path(args.models_dir) / args.city / f"{args.bracket}_ev_catboost"
            else:
                base_path = Path(args.models_dir) / args.city / args.bracket

            for pred_file in base_path.rglob("preds_*.csv"):
                print(f"  Reading {pred_file.name}")
                df = pd.read_csv(pred_file, usecols=['timestamp'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                preds_timestamps.extend(df['timestamp'].tolist())

            if preds_timestamps:
                preds_timestamps = pd.DatetimeIndex(preds_timestamps)
                args.start_date = preds_timestamps.min().date().isoformat()
                args.end_date = preds_timestamps.max().date().isoformat()
                print(f"[DEBUG A5] Auto-detected window from predictions: {args.start_date} to {args.end_date}")
            else:
                logger.error("No prediction files found, cannot auto-detect date range")
                return

    # Parse dates
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    num_days = (end_dt - start_dt).days + 1

    # Debug: Print exact arguments
    print(f"[DEBUG A0] args.city={args.city} args.bracket={args.bracket} "
          f"args.start={args.start_date} args.end={args.end_date} "
          f"args.models_dir={args.models_dir}")

    logger.info(f"\n{'='*60}")
    logger.info("BACKTEST CONFIGURATION")
    logger.info(f"{'='*60}\n")
    logger.info(f"City: {args.city}")
    logger.info(f"Strategy: {args.strategy}")
    if args.strategy == "model_kelly":
        logger.info(f"Bracket: {args.bracket}")
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Models dir: {args.models_dir}")
    logger.info(f"Date range: {args.start_date} to {args.end_date} ({num_days} days)")
    logger.info(f"Initial cash: ${args.initial_cash:,.2f}")
    if args.strategy == "buyhold":
        logger.info(f"Entry price: {args.entry_price}¢")
        logger.info(f"Contracts per market: {args.contracts}")

    # Load data
    logger.info(f"\nLoading markets and settlements...")
    markets_df = load_markets_with_settlements(args.city, start_dt, end_dt)

    if len(markets_df) == 0:
        logger.error("No markets found for specified date range")
        return

    # Run backtest based on strategy
    if args.strategy == "buyhold":
        logger.info(f"\nRunning buy-and-hold backtest...")
        summary = run_buyhold_backtest(
            markets_df=markets_df,
            initial_cash_cents=int(args.initial_cash * 100),
            entry_price_cents=args.entry_price,
            contracts_per_market=args.contracts,
        )
    elif args.strategy == "model_kelly":
        logger.info(f"\nInitializing model-driven strategy...")
        from backtest.model_kelly_adapter import ModelKellyBacktestStrategy

        allowed_windows = parse_time_window_values(args.exec_time_window)
        time_overrides = parse_time_overrides(args.exec_time_override)
        bracket_spreads = parse_bracket_spreads(args.exec_bracket_spread)
        exec_params = ExecParams(
            max_spread_cents=args.exec_max_spread,
            slippage_cents=args.exec_slippage,
            tau_open_cents=args.exec_tau_open,
            tau_close_cents=args.exec_tau_close,
            alpha_kelly=args.exec_alpha_kelly,
            max_bankroll_pct_city_day_side=args.exec_max_city_day,
            max_trade_notional_pct=args.exec_max_trade,
            allowed_time_windows=allowed_windows,
            time_of_day_overrides=time_overrides,
            bracket_spread_overrides=bracket_spreads,
            sigma_gate=args.exec_sigma_gate,
            humidity_gate=args.exec_humidity_gate,
        )

        strategy = ModelKellyBacktestStrategy(
            city=args.city,
            bracket=args.bracket,
            models_dir=args.models_dir,
            exec_params=exec_params,
            unified_head=args.unified_head,
            unified_tau=args.unified_tau,
            model_type=args.model_type,
            ev_models_dir=args.ev_models_dir,
            ev_min_delta_cents=args.ev_min_delta,
            ev_blend_weight=args.ev_blend_weight,
            ev_max_staleness_minutes=args.ev_max_staleness,
            ev_allow_missing=args.ev_allow_missing,
            tmax_preds_path=args.tmax_preds_csv,
            tmax_min_prob=args.tmax_min_prob,
            tmax_sigma_multiplier=args.tmax_sigma_multiplier,
            hybrid_model_type=args.hybrid_model_type,
            hybrid_models_dir=args.hybrid_models_dir,
            hybrid_min_prob=args.hybrid_min_prob,
            market_odds_weight=args.market_odds_weight,
        )

        logger.info(f"\nRunning model-driven backtest...")
        summary = run_strategy_backtest(
            strategy=strategy,
            markets_df=markets_df,
            city=args.city,
            initial_cash_cents=int(args.initial_cash * 100),
        )

        # Save diagnostics for model_kelly strategy
        if "diagnostics_obj" in summary:
            diag_filepath = f"results/diagnostics_{args.city}_{args.bracket}_{args.start_date}_{args.end_date}.csv"
            summary["diagnostics_obj"].save(diag_filepath)

            # Print diagnostic metrics
            summary["diagnostics_obj"].print_summary()
    else:
        logger.error(f"Unknown strategy: {args.strategy}")
        return

    # Print results
    print_backtest_summary(summary)

    # Save JSON summary if requested
    if args.output_json:
        json_summary = {
            "sharpe": summary["sharpe_ratio"],
            "max_drawdown": summary["max_drawdown_pct"] / 100.0,  # Convert to decimal
            "total_pnl_cents": summary["total_pnl_cents"],
            "total_fees_cents": summary["total_fees_cents"],
            "gross_pnl_cents": summary["gross_pnl_cents"],
            # ECE metrics: prefer trades if available, else fall back to all-minutes
            "ece_trades": summary.get("diagnostics", {}).get("brier_score", None),
            "ece_all_minutes": summary.get("diagnostics", {}).get("brier_all_minutes", None),
            "n_trades": summary["num_trades"]
        }
        with open(args.output_json, "w") as f:
            json.dump(json_summary, f, indent=2)
        logger.info(f"\n✓ Saved summary JSON → {args.output_json}")


if __name__ == "__main__":
    main()
