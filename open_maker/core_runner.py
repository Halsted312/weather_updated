"""
Core strategy runner for open-maker backtests.

This module contains:
- OpenMakerTrade: Single trade record dataclass
- OpenMakerResult: Backtest results container with metrics
- run_strategy: Unified backtest runner for all strategies
"""

import logging
import statistics
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from src.db import get_db_session

from .data_loading import (
    load_forecast_data,
    load_settlement_data,
    load_market_data,
    load_candle_data,
    get_forecast_at_open,
)
from .utils import (
    kalshi_maker_fee,
    kalshi_taker_fee,
    find_bracket_for_temp,
    calculate_position_size,
    calculate_pnl,
    calculate_exit_pnl,
    get_predicted_high_hour,
    compute_decision_time_utc,
    get_bracket_index,
    get_neighbor_ticker,
    load_minute_obs,
    get_forecast_temp_at_time,
    compute_obs_stats,
    check_fill_achievable,
)
from .strategies import get_strategy
from .strategies.base import StrategyParamsBase, OpenMakerParams, TradeContext, TradeDecision
from .strategies.curve_gap import CurveGapDecision

logger = logging.getLogger(__name__)


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

    # Exit info (for strategies with early exit)
    exited_early: bool = False
    exit_price_cents: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None


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
# Unified Strategy Runner
# =============================================================================

def run_strategy(
    strategy_id: str,
    cities: List[str],
    start_date: date,
    end_date: date,
    params: StrategyParamsBase,
    fill_check: bool = False,
) -> OpenMakerResult:
    """
    Unified backtest runner for all strategies.

    Uses the strategy registry to get the appropriate strategy class,
    then runs uniform entry setup and calls strategy.decide() for
    strategy-specific logic.

    Args:
        strategy_id: Strategy identifier (e.g., "open_maker_base", "open_maker_next_over")
        cities: List of city IDs to backtest
        start_date: Start date for backtest
        end_date: End date for backtest
        params: Strategy parameters (must match strategy's expected params class)
        fill_check: If True, skip trades where entry price wasn't achievable in first 2h

    Returns:
        OpenMakerResult with all trades
    """
    run_id = str(uuid.uuid4())[:8]

    # Get strategy class from registry
    StrategyCls, _ = get_strategy(strategy_id)
    strategy = StrategyCls(params)

    # Build result container with base params for compatibility
    base_params = OpenMakerParams(
        entry_price_cents=params.entry_price_cents,
        temp_bias_deg=params.temp_bias_deg,
        basis_offset_days=params.basis_offset_days,
        bet_amount_usd=params.bet_amount_usd,
    )
    result = OpenMakerResult(
        run_id=run_id,
        strategy_name=strategy_id,
        params=base_params,
        start_date=start_date,
        end_date=end_date,
        cities=cities,
    )

    logger.info(f"Starting backtest: {strategy_id}")
    logger.info(f"  Date range: {start_date} to {end_date}")
    logger.info(f"  Cities: {cities}")

    # Log params based on strategy type
    if strategy_id == "open_maker_next_over":
        logger.info(f"  Params: entry={params.entry_price_cents}c, bias={params.temp_bias_deg}F, "
                    f"offset={params.decision_offset_min}min, neighbor_min={params.neighbor_price_min_c}c, "
                    f"our_max={params.our_price_max_c}c")
    elif strategy_id == "open_maker_curve_gap":
        logger.info(f"  Params: entry={params.entry_price_cents}c, bias={params.temp_bias_deg}F, "
                    f"offset={params.decision_offset_min}min, delta_min={params.delta_obs_fcst_min_deg}F, "
                    f"slope_min={params.slope_min_deg_per_hour}F/h, max_shift={params.max_shift_bins}")
    else:
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

            # Track exits for next_over
            exits_triggered = 0
            # Track trades skipped due to fill check
            fill_skipped = 0

            # Process each event_date
            event_dates = sorted(settlement_df["event_date"].unique())

            for event_date in event_dates:
                # ==========================================
                # UNIFORM ENTRY SETUP (same for all strategies)
                # ==========================================

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

                # ==============================================
                # FILL REALISM CHECK (optional)
                # ==============================================
                if fill_check:
                    # Get listed_at for this ticker
                    ticker_row = market_df[
                        (market_df["event_date"] == event_date) &
                        (market_df["ticker"] == ticker)
                    ]
                    if not ticker_row.empty and ticker_row.iloc[0]["listed_at"] is not None:
                        listed_at = ticker_row.iloc[0]["listed_at"]
                        # Make timezone-aware if needed
                        if listed_at.tzinfo is None:
                            import pytz
                            listed_at = pytz.UTC.localize(listed_at)

                        if not check_fill_achievable(
                            session, ticker, listed_at, params.entry_price_cents
                        ):
                            fill_skipped += 1
                            logger.debug(
                                f"  {event_date}: No fill at {params.entry_price_cents}c for {ticker}"
                            )
                            continue

                # Get bracket index and sorted brackets
                bin_idx, total_bins, sorted_brackets = get_bracket_index(
                    market_df, event_date, ticker
                )

                if bin_idx < 0:
                    continue

                # Get settlement result
                settlement_row = settlement_df[settlement_df["event_date"] == event_date]
                if settlement_row.empty:
                    logger.debug(f"  {event_date}: No settlement data")
                    continue

                tmax_final = float(settlement_row.iloc[0]["tmax_final"])

                # Calculate position size
                num_contracts, amount_usd = calculate_position_size(
                    params.entry_price_cents,
                    params.bet_amount_usd,
                )

                # ==========================================
                # BUILD TRADE CONTEXT
                # ==========================================

                context = TradeContext(
                    city=city,
                    event_date=event_date,
                    forecast_basis_date=basis_date,
                    temp_fcst_open=temp_fcst,
                    temp_adjusted=temp_adjusted,
                    ticker=ticker,
                    floor_strike=floor_strike,
                    cap_strike=cap_strike,
                    bin_index=bin_idx,
                    total_bins=total_bins,
                    all_brackets=sorted_brackets,
                    markets_df=market_df,
                    entry_price_cents=params.entry_price_cents,
                    num_contracts=num_contracts,
                    amount_usd=amount_usd,
                    tmax_final=tmax_final,
                )

                # ==========================================
                # STRATEGY-SPECIFIC DECISION
                # ==========================================

                candles_df = None  # Default: no candles needed
                obs_stats = None  # For curve_gap
                T_fcst_at_decision = None  # For curve_gap

                # For next_over: compute decision time and load candles
                if strategy_id == "open_maker_next_over":
                    # Get predicted high hour from yesterday's forecast
                    predicted_high_hour = get_predicted_high_hour(
                        session, city, event_date
                    )

                    if predicted_high_hour is not None:
                        # Compute decision time
                        decision_time = compute_decision_time_utc(
                            city=city,
                            event_date=event_date,
                            predicted_high_hour=predicted_high_hour,
                            offset_minutes=params.decision_offset_min,
                        )

                        # Get neighbor ticker for candle loading
                        neighbor_ticker = get_neighbor_ticker(sorted_brackets, bin_idx, "up")
                        tickers_to_load = [ticker]
                        if neighbor_ticker:
                            tickers_to_load.append(neighbor_ticker)

                        # Load candles around decision time
                        window_start = decision_time - timedelta(minutes=10)
                        window_end = decision_time + timedelta(minutes=5)
                        candles_df = load_candle_data(
                            session, tickers_to_load, window_start, window_end
                        )

                # For curve_gap: compute decision time, load observations, get forecast
                elif strategy_id == "open_maker_curve_gap":
                    # Get predicted high hour from yesterday's forecast
                    predicted_high_hour = get_predicted_high_hour(
                        session, city, event_date
                    )

                    if predicted_high_hour is not None:
                        # Compute decision time
                        decision_time = compute_decision_time_utc(
                            city=city,
                            event_date=event_date,
                            predicted_high_hour=predicted_high_hour,
                            offset_minutes=params.decision_offset_min,
                        )

                        # Load minute observations (1h before decision time)
                        obs_start = decision_time - timedelta(minutes=75)
                        obs_end = decision_time
                        obs_df = load_minute_obs(session, city, obs_start, obs_end)

                        # Compute observation stats
                        if not obs_df.empty:
                            obs_stats = compute_obs_stats(obs_df, decision_time)

                        # Get interpolated forecast temp at decision time
                        T_fcst_at_decision = get_forecast_temp_at_time(
                            session, city, event_date, decision_time,
                            basis_date=event_date - timedelta(days=params.basis_offset_days)
                        )

                # Call strategy decision with appropriate arguments
                if strategy_id == "open_maker_curve_gap":
                    decision = strategy.decide(
                        context, candles_df,
                        obs_stats=obs_stats,
                        T_fcst_at_decision=T_fcst_at_decision
                    )
                else:
                    decision = strategy.decide(context, candles_df)

                # ==========================================
                # P&L CALCULATION
                # ==========================================

                exited_early = decision.action == "exit"
                exit_price_cents = decision.exit_price_cents
                exit_time = decision.exit_time
                exit_reason = decision.exit_reason

                # For curve_gap: check for override_bin_index
                override_bin_index = None
                if isinstance(decision, CurveGapDecision):
                    override_bin_index = decision.override_bin_index

                if exited_early:
                    exits_triggered += 1
                    # Taker fee for exit
                    fee_usd = kalshi_taker_fee(exit_price_cents, num_contracts)
                    # Exited early at exit_price
                    pnl_net = calculate_exit_pnl(
                        params.entry_price_cents,
                        exit_price_cents,
                        num_contracts,
                        fee_usd,
                    )
                    pnl_gross = pnl_net + fee_usd
                    # For exited trades, bin_won is based on exit P&L
                    bin_won = pnl_net > 0
                else:
                    # Held to settlement - maker fee ($0 for weather)
                    fee_usd = kalshi_maker_fee(params.entry_price_cents, num_contracts)

                    # Determine which bracket to use for P&L calculation
                    # For curve_gap with override: use shifted bracket
                    ticker_for_pnl = ticker
                    if override_bin_index is not None and override_bin_index != bin_idx:
                        # Get ticker for the shifted bracket
                        ticker_for_pnl = sorted_brackets.iloc[override_bin_index]["ticker"]

                    # Determine if we won based on the bracket used for P&L
                    winning_bracket = find_bracket_for_temp(market_df, event_date, tmax_final)
                    bin_won = winning_bracket is not None and winning_bracket[0] == ticker_for_pnl
                    pnl_net = calculate_pnl(
                        params.entry_price_cents,
                        num_contracts,
                        bin_won,
                        fee_usd,
                    )
                    # Gross P&L
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
                    exited_early=exited_early,
                    exit_price_cents=exit_price_cents,
                    exit_time=exit_time,
                    exit_reason=exit_reason,
                )

                result.trades.append(trade)

            # Log city summary
            city_trades = [t for t in result.trades if t.city == city]
            fill_info = f", {fill_skipped} skipped (no fill)" if fill_check and fill_skipped > 0 else ""
            if strategy_id == "open_maker_next_over":
                city_exits = sum(1 for t in city_trades if t.exited_early)
                logger.info(f"  {city}: {len(city_trades)} trades, {city_exits} exits{fill_info}")
            elif strategy_id == "open_maker_curve_gap":
                # Count trades where shift was applied (exit_reason contains "shift")
                city_shifts = sum(1 for t in city_trades if t.exit_reason and "shift" in t.exit_reason)
                logger.info(f"  {city}: {len(city_trades)} trades, {city_shifts} shifts{fill_info}")
            else:
                logger.info(f"  {city}: {len(city_trades)} trades{fill_info}")

    # Final summary
    if strategy_id == "open_maker_next_over":
        total_exits = sum(1 for t in result.trades if t.exited_early)
        logger.info(f"Backtest complete: {result.num_trades} trades, {total_exits} exits")
    elif strategy_id == "open_maker_curve_gap":
        total_shifts = sum(1 for t in result.trades if t.exit_reason and "shift" in t.exit_reason)
        logger.info(f"Backtest complete: {result.num_trades} trades, {total_shifts} shifts")
    else:
        logger.info(f"Backtest complete: {result.num_trades} total trades")

    return result
