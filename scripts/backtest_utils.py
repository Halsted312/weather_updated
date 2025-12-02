#!/usr/bin/env python3
"""
Backtest Utility Functions

Helper functions for hybrid model backtest:
- Load historical candles and settlements
- Simulate maker fills
- Calculate metrics
- Delta probabilities → bracket probabilities
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from config import live_trader_config as config
from src.trading.fees import maker_fee_total, taker_fee_total
from src.trading.risk import PositionSizer

logger = logging.getLogger(__name__)

# ==============================================================================
# Data Loading
# ==============================================================================

def query_candle_at_time(
    session,
    ticker: str,
    timestamp: datetime,
    window_minutes: int = 5,
    use_dense: bool = True
) -> Optional[Dict[str, int]]:
    """
    Get bid/ask from kalshi candles at specific timestamp.

    Args:
        session: Database session
        ticker: Kalshi market ticker
        timestamp: Target timestamp (timezone-aware)
        window_minutes: Search window around timestamp (default 5 min)
        use_dense: If True, use candles_1m_dense (forward-filled); else use sparse candles_1m

    Returns:
        Dict with 'yes_bid', 'yes_ask' in cents, or None if no data
    """
    # Use dense table (forward-filled) for live trading reliability
    # Use sparse table for backtesting if dense not available
    table = "kalshi.candles_1m_dense" if use_dense else "kalshi.candles_1m"

    # Query candle within window
    query = text(f"""
        SELECT yes_bid_close, yes_ask_close
        FROM {table}
        WHERE ticker = :ticker
          AND bucket_start >= :start_time
          AND bucket_start <= :end_time
        ORDER BY ABS(EXTRACT(EPOCH FROM (bucket_start - :target_time)))
        LIMIT 1
    """)

    start_time = timestamp - timedelta(minutes=window_minutes)
    end_time = timestamp + timedelta(minutes=window_minutes)

    result = session.execute(
        query,
        {
            'ticker': ticker,
            'target_time': timestamp,
            'start_time': start_time,
            'end_time': end_time,
        }
    ).fetchone()

    if result is None:
        return None

    return {
        'yes_bid': result[0],
        'yes_ask': result[1],
    }


def load_settlement(session, city: str, event_date) -> Optional[int]:
    """
    Get actual settled temperature from wx.settlement.

    Args:
        session: Database session
        city: City identifier ('chicago', 'austin', etc.)
        event_date: Event date

    Returns:
        Settled temperature in °F (integer), or None if not found
    """
    query = text("""
        SELECT tmax_final
        FROM wx.settlement
        WHERE city = :city AND date_local = :event_date
    """)

    result = session.execute(
        query,
        {'city': city, 'event_date': event_date}
    ).fetchone()

    if result is None or result[0] is None:
        return None

    return int(round(result[0]))


def load_brackets_for_event(session, city: str, event_date) -> pd.DataFrame:
    """
    Load Kalshi bracket structure for a given event.

    Args:
        session: Database session
        city: City identifier
        event_date: Event date

    Returns:
        DataFrame with columns: ticker, strike_type, floor_strike, cap_strike
    """
    query = text("""
        SELECT ticker, strike_type, floor_strike, cap_strike
        FROM kalshi.markets
        WHERE city = :city AND event_date = :event_date
        ORDER BY floor_strike NULLS FIRST
    """)

    result = session.execute(
        query,
        {'city': city, 'event_date': event_date}
    )

    rows = []
    for row in result:
        rows.append({
            'ticker': row[0],
            'strike_type': row[1],
            'floor_strike': row[2],
            'cap_strike': row[3],
        })

    return pd.DataFrame(rows)


# ==============================================================================
# Fill Simulation
# ==============================================================================

def simulate_maker_fill(
    yes_bid: int,
    yes_ask: int,
    action: str
) -> Tuple[bool, int]:
    """
    Simulate maker order fill using MAKER_FILL_PROBABILITY from config.

    Maker order logic:
    - BUY: Post at bid + 1¢ (improve bid)
    - SELL: Post at ask - 1¢ (improve ask)
    - Fill probability: MAKER_FILL_PROBABILITY (default 0.4)

    Args:
        yes_bid: Current YES bid in cents
        yes_ask: Current YES ask in cents
        action: 'buy' or 'sell'

    Returns:
        (filled: bool, entry_price: int in cents)
    """
    spread = yes_ask - yes_bid

    # Determine maker order price
    if action == 'buy':
        # Post at bid + 1¢ if spread > 1, otherwise at bid
        entry_price = yes_bid + 1 if spread > 1 else yes_bid
    else:  # sell
        # Post at ask - 1¢ if spread > 1, otherwise at ask
        entry_price = yes_ask - 1 if spread > 1 else yes_ask

    # Simulate probabilistic fill
    filled = random.random() < config.MAKER_FILL_PROBABILITY

    return filled, entry_price


# ==============================================================================
# Delta to Bracket Probability Conversion
# ==============================================================================

def delta_probs_to_bracket_probs(
    delta_probs: np.ndarray,
    t_base: int,
    brackets: pd.DataFrame
) -> Dict[str, float]:
    """
    Convert delta probabilities to bracket win probabilities.

    Args:
        delta_probs: Array of delta probabilities (13 classes: -2 to +10)
        t_base: Current max observed temp (baseline)
        brackets: DataFrame with bracket structure

    Returns:
        Dict mapping ticker → P(bracket wins)
    """
    # Delta classes: [-2, -1, 0, 1, ..., 10]
    DELTA_CLASSES = list(range(-2, 11))

    bracket_probs = {}

    for _, bracket in brackets.iterrows():
        ticker = bracket['ticker']
        strike_type = bracket['strike_type']
        floor_strike = bracket.get('floor_strike')
        cap_strike = bracket.get('cap_strike')

        prob = 0.0

        for i, delta in enumerate(DELTA_CLASSES):
            if i >= len(delta_probs):
                break

            settled_temp = t_base + delta

            # Check if this settled temp wins this bracket
            if strike_type == 'less':
                if cap_strike is not None and settled_temp <= cap_strike:
                    prob += delta_probs[i]

            elif strike_type == 'greater':
                if floor_strike is not None and settled_temp >= floor_strike + 1:
                    prob += delta_probs[i]

            elif strike_type == 'between':
                if (floor_strike is not None and cap_strike is not None and
                    floor_strike <= settled_temp <= cap_strike):
                    prob += delta_probs[i]

        bracket_probs[ticker] = float(prob)

    return bracket_probs


def find_best_bracket(
    bracket_probs: Dict[str, float],
    candles: Dict[str, Dict[str, int]],
    min_ev_cents: float,
    kelly_fraction: float,
    settlement_std: float
) -> Optional[Dict]:
    """
    Find the best bracket to trade (highest EV after fees and sizing).

    Args:
        bracket_probs: Dict mapping ticker → model probability
        candles: Dict mapping ticker → {'yes_bid', 'yes_ask'}
        min_ev_cents: Minimum EV threshold
        kelly_fraction: Kelly fraction for sizing
        settlement_std: Settlement std deviation (for uncertainty penalty)

    Returns:
        Dict with trade details, or None if no trade meets criteria
    """
    from src.trading.risk import PositionSizer

    candidates = []

    sizer = PositionSizer(
        bankroll_usd=config.BANKROLL_USD,
        kelly_fraction=kelly_fraction,
        max_bet_usd=config.MAX_BET_SIZE_USD,
        max_position_contracts=100,
        uncertainty_penalty=True
    )

    for ticker, model_prob in bracket_probs.items():
        if ticker not in candles:
            continue

        candle = candles[ticker]
        yes_bid = candle['yes_bid']
        yes_ask = candle['yes_ask']

        # Skip invalid spreads
        if yes_bid <= 0 or yes_ask >= 100 or yes_ask <= yes_bid:
            continue

        # Evaluate BUY (maker order at bid + 1)
        filled_buy, entry_price_buy = simulate_maker_fill(yes_bid, yes_ask, 'buy')
        if filled_buy:
            market_prob_buy = entry_price_buy / 100.0
            ev_cents_buy = (model_prob - market_prob_buy) * 100
            # Fee for maker = 0
            fee_cents_buy = maker_fee_total(entry_price_buy, 1)
            ev_after_fee_buy = ev_cents_buy - fee_cents_buy

            if ev_after_fee_buy >= min_ev_cents:
                # Size position
                size_result = sizer.calculate(
                    ev_per_contract_cents=ev_after_fee_buy,
                    price_cents=entry_price_buy,
                    model_prob=model_prob,
                    settlement_std_degf=settlement_std,
                    current_position=0
                )

                if size_result.num_contracts > 0:
                    candidates.append({
                        'ticker': ticker,
                        'action': 'buy',
                        'entry_price': entry_price_buy,
                        'model_prob': model_prob,
                        'market_prob': market_prob_buy,
                        'edge': model_prob - market_prob_buy,
                        'ev_cents': ev_after_fee_buy,
                        'num_contracts': size_result.num_contracts,
                        'kelly_fraction': size_result.kelly_fraction,
                        'capped_by': size_result.capped_by,
                    })

    # Return best EV trade
    if candidates:
        best = max(candidates, key=lambda x: x['ev_cents'])
        return best
    else:
        return None


# ==============================================================================
# Metrics Calculation
# ==============================================================================

def calculate_realized_edge(
    entry_price: int,
    settlement_outcome: bool,
    fee_cents: int,
    num_contracts: int
) -> float:
    """
    Calculate realized edge per contract.

    Args:
        entry_price: Entry price in cents
        settlement_outcome: True if bracket won, False if lost
        fee_cents: Total fee paid in cents
        num_contracts: Number of contracts traded

    Returns:
        Realized edge per contract in cents
    """
    if settlement_outcome:
        # Won: Payout = $1 per contract = 100¢
        payout_cents = 100 * num_contracts
    else:
        # Lost: Payout = $0
        payout_cents = 0

    entry_cost_cents = entry_price * num_contracts
    net_pnl_cents = payout_cents - entry_cost_cents - fee_cents

    realized_edge_cents = net_pnl_cents / num_contracts

    return realized_edge_cents


def calculate_trade_pnl(
    entry_price: int,
    settlement_outcome: bool,
    num_contracts: int,
    role: str = 'maker'
) -> Tuple[float, float]:
    """
    Calculate P&L for a trade.

    Args:
        entry_price: Entry price in cents
        settlement_outcome: True if bracket won
        num_contracts: Number of contracts
        role: 'maker' or 'taker'

    Returns:
        (gross_pnl_cents, net_pnl_cents)
    """
    if settlement_outcome:
        settlement_price = 100  # $1
    else:
        settlement_price = 0

    gross_pnl_cents = (settlement_price - entry_price) * num_contracts

    # Calculate fee
    if role == 'maker':
        fee_cents = maker_fee_total(entry_price, num_contracts)
    else:
        fee_cents = taker_fee_total(entry_price, num_contracts)

    net_pnl_cents = gross_pnl_cents - fee_cents

    return gross_pnl_cents, net_pnl_cents


def check_bracket_win(
    ticker: str,
    settled_temp: int,
    brackets: pd.DataFrame
) -> bool:
    """
    Check if a bracket wins given settled temperature.

    Args:
        ticker: Bracket ticker
        settled_temp: Actual settled temperature
        brackets: DataFrame with bracket structure

    Returns:
        True if bracket wins, False otherwise
    """
    bracket = brackets[brackets['ticker'] == ticker]

    if bracket.empty:
        return False

    bracket = bracket.iloc[0]
    strike_type = bracket['strike_type']
    floor_strike = bracket.get('floor_strike')
    cap_strike = bracket.get('cap_strike')

    if strike_type == 'less':
        return settled_temp <= cap_strike if cap_strike is not None else False
    elif strike_type == 'greater':
        return settled_temp >= floor_strike + 1 if floor_strike is not None else False
    elif strike_type == 'between':
        if floor_strike is not None and cap_strike is not None:
            return floor_strike <= settled_temp <= cap_strike
        return False

    return False
