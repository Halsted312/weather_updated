#!/usr/bin/env python3
"""
Hybrid vs TOD v1 Backtest

Compares two ordinal model strategies over 60-day test period:
1. TOD v1-only: Event day trading only (D 10:00)
2. Hybrid: D-1 trading (market-clock) + event day trading (TOD v1)

Uses realistic trading simulation:
- Historical Kalshi candles for bid/ask prices
- Actual fee model (non-linear Kalshi formula)
- Kelly position sizing with constraints
- Probabilistic maker fills (40% default)
- Full P&L accounting with risk metrics

Output:
- Side-by-side comparison: TOD v1 vs Hybrid
- Per-city breakdown
- Per-time-window breakdown (D-1 vs D)
- Trading metrics (P&L, Sharpe, drawdown)
- Prediction metrics (MAE, Within-1, Within-2)

Usage:
    python scripts/backtest_hybrid_vs_tod_v1.py [--debug] [--days N]
"""

import argparse
import logging
import pickle
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import Pool
from sqlalchemy import create_engine
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import live_trader_config as config
from src.db import get_session_factory
from src.trading.fees import maker_fee_total
from scripts.backtest_utils import (
    query_candle_at_time,
    load_settlement,
    load_brackets_for_event,
    simulate_maker_fill,
    delta_probs_to_bracket_probs,
    find_best_bracket,
    calculate_trade_pnl,
    check_bracket_win,
    calculate_realized_edge,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Backtest Configuration
# ==============================================================================

CITIES = ['chicago', 'austin', 'denver', 'los_angeles', 'miami', 'philadelphia']

# Test period (full 60 days)
TEST_START_DATE = date(2025, 9, 29)
TEST_END_DATE = date(2025, 11, 27)

# Trading times
D_MINUS_1_HOUR = 10  # D-1 10:00 local
D_HOUR = 10  # D 10:00 local

# ==============================================================================
# Trade Data Structure
# ==============================================================================

class Trade:
    """Single trade record"""
    def __init__(
        self,
        city: str,
        event_date: date,
        time_window: str,  # 'D-1' or 'D'
        model_type: str,  # 'market_clock' or 'tod_v1'
        ticker: str,
        action: str,
        entry_price: int,
        num_contracts: int,
        model_prob: float,
        market_prob: float,
        kelly_fraction: float,
        capped_by: Optional[str],
        filled: bool,
        settled_temp: Optional[int] = None,
        won: Optional[bool] = None,
        gross_pnl: Optional[float] = None,
        net_pnl: Optional[float] = None,
        realized_edge: Optional[float] = None,
    ):
        self.city = city
        self.event_date = event_date
        self.time_window = time_window
        self.model_type = model_type
        self.ticker = ticker
        self.action = action
        self.entry_price = entry_price
        self.num_contracts = num_contracts
        self.model_prob = model_prob
        self.market_prob = market_prob
        self.edge = model_prob - market_prob
        self.kelly_fraction = kelly_fraction
        self.capped_by = capped_by
        self.filled = filled
        self.settled_temp = settled_temp
        self.won = won
        self.gross_pnl = gross_pnl
        self.net_pnl = net_pnl
        self.realized_edge = realized_edge

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame"""
        return {
            'city': self.city,
            'event_date': self.event_date,
            'time_window': self.time_window,
            'model_type': self.model_type,
            'ticker': self.ticker,
            'action': self.action,
            'entry_price': self.entry_price,
            'num_contracts': self.num_contracts,
            'model_prob': self.model_prob,
            'market_prob': self.market_prob,
            'edge': self.edge,
            'kelly_fraction': self.kelly_fraction,
            'capped_by': self.capped_by,
            'filled': self.filled,
            'settled_temp': self.settled_temp,
            'won': self.won,
            'gross_pnl': self.gross_pnl,
            'net_pnl': self.net_pnl,
            'realized_edge': self.realized_edge,
        }


# ==============================================================================
# Hybrid Backtester
# ==============================================================================

class HybridBacktester:
    """
    Backtest hybrid ordinal model strategy vs TOD v1-only baseline.

    Scenario 1 (TOD v1-only):
    - Event day trading only (D 10:00)
    - Uses per-city TOD v1 models

    Scenario 2 (Hybrid):
    - D-1 trading (D-1 10:00) using market-clock global model
    - Event day trading (D 10:00) using TOD v1 per-city models
    """

    def __init__(self, start_date: date, end_date: date, cities: List[str], debug: bool = False):
        self.start_date = start_date
        self.end_date = end_date
        self.cities = cities
        self.debug = debug

        SessionLocal = get_session_factory()
        self.session = SessionLocal()

        # Load test data
        logger.info("Loading test data...")
        self.mc_test = self._load_market_clock_test()
        self.tod_tests = {city: self._load_tod_v1_test(city) for city in cities}

        # Load models
        logger.info("Loading models...")
        self.mc_model = self._load_market_clock_model()
        self.tod_models = {city: self._load_tod_v1_model(city) for city in cities}

        # Get unique events
        self.events = self._get_unique_events()
        logger.info(f"Found {len(self.events)} unique events to backtest")

    def _load_market_clock_test(self) -> pd.DataFrame:
        """Load market-clock test data"""
        path = Path('models/saved/market_clock_tod_v1/test_data.parquet')
        df = pd.read_parquet(path)
        logger.info(f"Loaded market-clock test: {len(df):,} rows")
        return df

    def _load_tod_v1_test(self, city: str) -> pd.DataFrame:
        """Load TOD v1 test data for a city"""
        path = Path(f'models/saved/{city}_tod_v1/test_data.parquet')
        df = pd.read_parquet(path)
        # TOD v1 uses 'day' column instead of 'event_date'
        if 'day' in df.columns and 'event_date' not in df.columns:
            df['event_date'] = df['day']
        logger.info(f"Loaded {city} TOD v1 test: {len(df):,} rows")
        return df

    def _load_market_clock_model(self) -> Dict:
        """Load market-clock global model"""
        path = Path('models/saved/market_clock_tod_v1/ordinal_catboost_market_clock_tod_v1.pkl')
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        logger.info(f"Loaded market-clock model: {len(model_data['classifiers'])} classifiers")
        return model_data

    def _load_tod_v1_model(self, city: str) -> Dict:
        """Load TOD v1 per-city model"""
        path = Path(f'models/saved/{city}_tod_v1/ordinal_catboost_tod_v1.pkl')
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data

    def _get_unique_events(self) -> List[tuple]:
        """Get unique (city, event_date) pairs from test data"""
        events = set()

        # From market-clock data
        for _, row in self.mc_test.iterrows():
            events.add((row['city'], row['event_date'].date() if isinstance(row['event_date'], pd.Timestamp) else row['event_date']))

        # Filter to date range AND requested cities
        events = [(city, event_date) for city, event_date in events
                  if self.start_date <= event_date <= self.end_date
                  and city in self.cities]

        return sorted(events)

    def _predict_ordinal(self, model_data: Dict, features_df: pd.DataFrame) -> np.ndarray:
        """
        Run ordinal prediction through a model.

        Returns:
            Array of delta class probabilities (13 classes)
        """
        classifiers = model_data['classifiers']
        thresholds = model_data['thresholds']

        # Get delta classes (different structure for market-clock vs TOD v1)
        if 'delta_classes' in model_data:
            delta_classes = np.array(model_data['delta_classes'])
        elif 'metadata' in model_data and 'delta_range' in model_data['metadata']:
            min_delta, max_delta = model_data['metadata']['delta_range']
            delta_classes = np.array(list(range(min_delta, max_delta + 1)))
        else:
            # Default: [-2, 10] for 13 classes
            delta_classes = np.array(list(range(-2, 11)))

        feature_cols = model_data['feature_cols']

        # Fill missing features with NaN (model handles this gracefully)
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = np.nan

        # Select and order features
        features_df = features_df[feature_cols]

        # Get categorical features if available
        cat_features = model_data.get('cat_features', [])

        n_samples = len(features_df)
        cum_proba = np.ones((n_samples, len(thresholds) + 1))

        for i, k in enumerate(thresholds):
            clf = classifiers[k]
            pool = Pool(features_df, cat_features=cat_features)
            p_ge_k = clf.predict_proba(pool)[:, 1]
            cum_proba[:, i + 1] = p_ge_k

        # Convert cumulative to class probabilities
        class_proba = np.zeros((n_samples, len(delta_classes)))
        for i in range(len(delta_classes)):
            if i == 0:
                class_proba[:, i] = 1 - cum_proba[:, 1]
            elif i == len(delta_classes) - 1:
                class_proba[:, i] = cum_proba[:, i]
            else:
                class_proba[:, i] = cum_proba[:, i] - cum_proba[:, i + 1]

        return class_proba

    def _simulate_d1_trade(self, city: str, event_date: date) -> Optional[Trade]:
        """
        Simulate D-1 trade using market-clock model.

        Trades at D-1 10:00 local time.
        """
        d_minus_1 = event_date - timedelta(days=1)

        # Get D-1 snapshot from market-clock test data
        # Note: event_date in DataFrame is datetime64, so convert for comparison
        snapshot = self.mc_test[
            (self.mc_test['city'] == city) &
            (self.mc_test['event_date'] == pd.Timestamp(event_date)) &
            (self.mc_test['is_d_minus_1'] == 1)
        ]

        if snapshot.empty:
            return None

        # Use 10:00 snapshot (or closest)
        snapshot = snapshot[snapshot['snapshot_datetime'].dt.hour == D_MINUS_1_HOUR]
        if snapshot.empty:
            return None

        snapshot = snapshot.iloc[0]

        # Get model prediction (build features_df with all columns from snapshot)
        features_df = pd.DataFrame([snapshot])
        delta_probs = self._predict_ordinal(self.mc_model, features_df)[0]

        t_base = int(snapshot['t_base'])

        # Load brackets
        brackets = load_brackets_for_event(self.session, city, event_date)
        if brackets.empty:
            return None

        # Convert delta probs to bracket probs
        bracket_probs = delta_probs_to_bracket_probs(delta_probs, t_base, brackets)

        # Query candles at D-1 10:00
        city_tz = ZoneInfo(config.CITY_TIMEZONES[city])
        timestamp = datetime(d_minus_1.year, d_minus_1.month, d_minus_1.day, D_MINUS_1_HOUR, 0, tzinfo=city_tz)

        candles = {}
        for ticker in bracket_probs.keys():
            candle = query_candle_at_time(self.session, ticker, timestamp)
            if candle:
                candles[ticker] = candle

        if not candles:
            return None

        # Find best bracket
        settlement_std = float(snapshot.get('settlement_std', 2.5))  # Fallback if missing
        best_trade = find_best_bracket(
            bracket_probs,
            candles,
            min_ev_cents=config.D_MINUS_1_MIN_EV_PER_CONTRACT_CENTS,
            kelly_fraction=config.D_MINUS_1_KELLY_FRACTION,
            settlement_std=settlement_std
        )

        if not best_trade:
            return None

        # Create trade record
        trade = Trade(
            city=city,
            event_date=event_date,
            time_window='D-1',
            model_type='market_clock',
            ticker=best_trade['ticker'],
            action=best_trade['action'],
            entry_price=best_trade['entry_price'],
            num_contracts=best_trade['num_contracts'],
            model_prob=best_trade['model_prob'],
            market_prob=best_trade['market_prob'],
            kelly_fraction=best_trade['kelly_fraction'],
            capped_by=best_trade['capped_by'],
            filled=True  # Already incorporated in find_best_bracket
        )

        return trade

    def _simulate_d_trade(self, city: str, event_date: date) -> Optional[Trade]:
        """
        Simulate event day trade using TOD v1 model.

        Trades at D 10:00 local time.
        """
        # Get D snapshot from TOD v1 test data
        tod_test = self.tod_tests[city]
        snapshot = tod_test[tod_test['event_date'] == event_date]

        if snapshot.empty:
            return None

        # Use 10:00 snapshot (or closest)
        if 'snapshot_datetime' in snapshot.columns:
            snapshot = snapshot[snapshot['snapshot_datetime'].dt.hour == D_HOUR]
        elif 'snapshot_hour' in snapshot.columns:
            snapshot = snapshot[snapshot['snapshot_hour'] == D_HOUR]

        if snapshot.empty:
            return None

        snapshot = snapshot.iloc[0]

        # Get model prediction (build features_df with all columns from snapshot)
        tod_model = self.tod_models[city]
        features_df = pd.DataFrame([snapshot])
        delta_probs = self._predict_ordinal(tod_model, features_df)[0]

        t_base = int(snapshot['t_base'])

        # Load brackets
        brackets = load_brackets_for_event(self.session, city, event_date)
        if brackets.empty:
            return None

        # Convert delta probs to bracket probs
        bracket_probs = delta_probs_to_bracket_probs(delta_probs, t_base, brackets)

        # Query candles at D 10:00
        city_tz = ZoneInfo(config.CITY_TIMEZONES[city])
        timestamp = datetime(event_date.year, event_date.month, event_date.day, D_HOUR, 0, tzinfo=city_tz)

        candles = {}
        for ticker in bracket_probs.keys():
            candle = query_candle_at_time(self.session, ticker, timestamp)
            if candle:
                candles[ticker] = candle

        if not candles:
            return None

        # Find best bracket
        settlement_std = float(snapshot.get('settlement_std', 2.5))
        best_trade = find_best_bracket(
            bracket_probs,
            candles,
            min_ev_cents=config.MIN_EV_PER_CONTRACT_CENTS,
            kelly_fraction=config.KELLY_FRACTION,
            settlement_std=settlement_std
        )

        if not best_trade:
            return None

        # Create trade record
        trade = Trade(
            city=city,
            event_date=event_date,
            time_window='D',
            model_type='tod_v1',
            ticker=best_trade['ticker'],
            action=best_trade['action'],
            entry_price=best_trade['entry_price'],
            num_contracts=best_trade['num_contracts'],
            model_prob=best_trade['model_prob'],
            market_prob=best_trade['market_prob'],
            kelly_fraction=best_trade['kelly_fraction'],
            capped_by=best_trade['capped_by'],
            filled=True
        )

        return trade

    def _settle_trade(self, trade: Trade, brackets: pd.DataFrame):
        """
        Settle a trade with actual outcome.

        Updates trade with:
        - settled_temp
        - won (True/False)
        - gross_pnl, net_pnl
        - realized_edge
        """
        # Get settlement
        settled_temp = load_settlement(self.session, trade.city, trade.event_date)
        if settled_temp is None:
            logger.warning(f"No settlement for {trade.city} {trade.event_date}")
            return

        trade.settled_temp = settled_temp

        # Check if won
        trade.won = check_bracket_win(trade.ticker, settled_temp, brackets)

        # Calculate P&L
        gross_pnl, net_pnl = calculate_trade_pnl(
            trade.entry_price,
            trade.won,
            trade.num_contracts,
            role='maker'
        )

        trade.gross_pnl = gross_pnl
        trade.net_pnl = net_pnl

        # Calculate realized edge
        fee_cents = maker_fee_total(trade.entry_price, trade.num_contracts)
        trade.realized_edge = calculate_realized_edge(
            trade.entry_price,
            trade.won,
            fee_cents,
            trade.num_contracts
        )

    def run_tod_only_scenario(self) -> List[Trade]:
        """
        Run TOD v1-only scenario (event day trading only).

        Returns:
            List of Trade objects
        """
        logger.info("\n" + "="*70)
        logger.info("SCENARIO 1: TOD v1-only (Event Day Trading Only)")
        logger.info("="*70)

        trades = []

        for city, event_date in self.events:
            # Only trade on event day
            trade = self._simulate_d_trade(city, event_date)

            if trade:
                # Load brackets for settlement
                brackets = load_brackets_for_event(self.session, city, event_date)
                self._settle_trade(trade, brackets)
                trades.append(trade)

                if self.debug:
                    logger.info(
                        f"[TOD-ONLY] {city} {event_date} | {trade.ticker} @ {trade.entry_price}¢ "
                        f"| Edge: {trade.edge*100:.1f}pp | Settled: {trade.settled_temp}°F "
                        f"| {'WIN' if trade.won else 'LOSS'} | P&L: ${trade.net_pnl/100:.2f}"
                    )

        logger.info(f"TOD v1-only: {len(trades)} trades executed")
        return trades

    def run_hybrid_scenario(self) -> List[Trade]:
        """
        Run hybrid scenario (D-1 + event day trading).

        Returns:
            List of Trade objects
        """
        logger.info("\n" + "="*70)
        logger.info("SCENARIO 2: Hybrid (D-1 + Event Day Trading)")
        logger.info("="*70)

        trades = []

        for city, event_date in self.events:
            # D-1 trade
            d1_trade = self._simulate_d1_trade(city, event_date)
            if d1_trade:
                brackets = load_brackets_for_event(self.session, city, event_date)
                self._settle_trade(d1_trade, brackets)
                trades.append(d1_trade)

                if self.debug:
                    logger.info(
                        f"[HYBRID D-1] {city} {event_date} | {d1_trade.ticker} @ {d1_trade.entry_price}¢ "
                        f"| Edge: {d1_trade.edge*100:.1f}pp | Kelly: {d1_trade.kelly_fraction:.3f} "
                        f"| Settled: {d1_trade.settled_temp}°F | {'WIN' if d1_trade.won else 'LOSS'} "
                        f"| P&L: ${d1_trade.net_pnl/100:.2f}"
                    )

            # Event day trade
            d_trade = self._simulate_d_trade(city, event_date)
            if d_trade:
                brackets = load_brackets_for_event(self.session, city, event_date)
                self._settle_trade(d_trade, brackets)
                trades.append(d_trade)

                if self.debug:
                    logger.info(
                        f"[HYBRID D] {city} {event_date} | {d_trade.ticker} @ {d_trade.entry_price}¢ "
                        f"| Edge: {d_trade.edge*100:.1f}pp | Kelly: {d_trade.kelly_fraction:.3f} "
                        f"| Settled: {d_trade.settled_temp}°F | {'WIN' if d_trade.won else 'LOSS'} "
                        f"| P&L: ${d_trade.net_pnl/100:.2f}"
                    )

        logger.info(f"Hybrid: {len(trades)} trades executed")
        return trades

    def calculate_metrics(self, trades: List[Trade]) -> Dict:
        """
        Calculate comprehensive metrics from trades.

        Returns:
            Dict with trading metrics, prediction metrics, risk metrics
        """
        if not trades:
            return {
                'num_trades': 0,
                'num_wins': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_realized_edge': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'avg_kelly_fraction': 0.0,
            }

        df = pd.DataFrame([t.to_dict() for t in trades])

        # Trading metrics
        num_trades = len(df)
        total_pnl = df['net_pnl'].sum() / 100  # Convert to dollars
        num_wins = (df['won'] == True).sum()
        win_rate = num_wins / num_trades if num_trades > 0 else 0.0
        avg_realized_edge = df['realized_edge'].mean()

        # Risk metrics (daily returns)
        daily_pnl = df.groupby('event_date')['net_pnl'].sum() / 100
        sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if len(daily_pnl) > 1 else 0.0
        max_drawdown = (daily_pnl.cumsum() - daily_pnl.cumsum().cummax()).min()

        # Prediction metrics (by time window)
        metrics = {
            'num_trades': num_trades,
            'num_wins': int(num_wins),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_realized_edge': avg_realized_edge,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_kelly_fraction': df['kelly_fraction'].mean(),
        }

        # Breakdown by time window
        for window in ['D-1', 'D']:
            window_df = df[df['time_window'] == window]
            if not window_df.empty:
                metrics[f'{window}_num_trades'] = len(window_df)
                metrics[f'{window}_win_rate'] = (window_df['won'] == True).mean()
                metrics[f'{window}_total_pnl'] = window_df['net_pnl'].sum() / 100
                metrics[f'{window}_avg_edge'] = window_df['edge'].mean() * 100  # pp

        return metrics


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Backtest hybrid ordinal model vs TOD v1-only'
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--days', type=int, help='Limit to first N days (for testing)')
    parser.add_argument('--cities', nargs='+', default=CITIES, help='Cities to backtest')

    args = parser.parse_args()

    # Adjust date range if limited
    end_date = TEST_END_DATE
    if args.days:
        end_date = TEST_START_DATE + timedelta(days=args.days - 1)
        logger.info(f"Limited to {args.days} days: {TEST_START_DATE} to {end_date}")

    # Initialize backtester
    backtester = HybridBacktester(
        start_date=TEST_START_DATE,
        end_date=end_date,
        cities=args.cities,
        debug=args.debug
    )

    # Run scenarios
    tod_trades = backtester.run_tod_only_scenario()
    hybrid_trades = backtester.run_hybrid_scenario()

    # Calculate metrics
    logger.info("\n" + "="*70)
    logger.info("RESULTS")
    logger.info("="*70)

    tod_metrics = backtester.calculate_metrics(tod_trades)
    hybrid_metrics = backtester.calculate_metrics(hybrid_trades)

    # Print comparison
    print("\n" + "="*70)
    print("BACKTEST COMPARISON: TOD v1-only vs Hybrid")
    print("="*70)
    print(f"Period: {TEST_START_DATE} to {end_date} ({(end_date - TEST_START_DATE).days + 1} days)")
    print()

    print(f"{'Metric':<30} | {'TOD v1-only':>15} | {'Hybrid':>15} | {'Delta':>15}")
    print("-" * 90)

    # Trading metrics
    print(f"{'Number of trades':<30} | {tod_metrics['num_trades']:>15} | {hybrid_metrics['num_trades']:>15} | {hybrid_metrics['num_trades'] - tod_metrics['num_trades']:>+15}")
    print(f"{'Total P&L ($)':<30} | {tod_metrics['total_pnl']:>15.2f} | {hybrid_metrics['total_pnl']:>15.2f} | {hybrid_metrics['total_pnl'] - tod_metrics['total_pnl']:>+15.2f}")
    print(f"{'Win rate':<30} | {tod_metrics['win_rate']:>14.1%} | {hybrid_metrics['win_rate']:>14.1%} | {(hybrid_metrics['win_rate'] - tod_metrics['win_rate'])*100:>+14.1f}pp")
    print(f"{'Avg realized edge (¢)':<30} | {tod_metrics['avg_realized_edge']:>15.2f} | {hybrid_metrics['avg_realized_edge']:>15.2f} | {hybrid_metrics['avg_realized_edge'] - tod_metrics['avg_realized_edge']:>+15.2f}")
    print(f"{'Sharpe ratio':<30} | {tod_metrics['sharpe_ratio']:>15.2f} | {hybrid_metrics['sharpe_ratio']:>15.2f} | {hybrid_metrics['sharpe_ratio'] - tod_metrics['sharpe_ratio']:>+15.2f}")
    print(f"{'Max drawdown ($)':<30} | {tod_metrics['max_drawdown']:>15.2f} | {hybrid_metrics['max_drawdown']:>15.2f} | {hybrid_metrics['max_drawdown'] - tod_metrics['max_drawdown']:>+15.2f}")

    # Breakdown
    print()
    print("BREAKDOWN BY TIME WINDOW (Hybrid only):")
    print("-" * 90)
    if 'D-1_num_trades' in hybrid_metrics:
        print(f"  D-1:  {hybrid_metrics['D-1_num_trades']} trades, "
              f"Win rate: {hybrid_metrics.get('D-1_win_rate', 0):.1%}, "
              f"P&L: ${hybrid_metrics.get('D-1_total_pnl', 0):.2f}, "
              f"Avg edge: {hybrid_metrics.get('D-1_avg_edge', 0):.1f}pp")
    if 'D_num_trades' in hybrid_metrics:
        print(f"  D:    {hybrid_metrics['D_num_trades']} trades, "
              f"Win rate: {hybrid_metrics.get('D_win_rate', 0):.1%}, "
              f"P&L: ${hybrid_metrics.get('D_total_pnl', 0):.2f}, "
              f"Avg edge: {hybrid_metrics.get('D_avg_edge', 0):.1f}pp")

    print()
    print("="*70)
    print("NOTE: This backtest uses 10:00-only snapshots (D-1 10:00, D 10:00)")
    print("Fees modeled using Kalshi's actual formula via src/trading/fees.py")
    print(f"Fill rate used: {config.MAKER_FILL_PROBABILITY:.0%} (MAKER_FILL_PROBABILITY)")
    print("="*70)

    # Save results
    output_dir = Path('models/reports/backtest')
    output_dir.mkdir(parents=True, exist_ok=True)

    tod_df = pd.DataFrame([t.to_dict() for t in tod_trades])
    hybrid_df = pd.DataFrame([t.to_dict() for t in hybrid_trades])

    tod_df.to_csv(output_dir / 'tod_v1_only_trades.csv', index=False)
    hybrid_df.to_csv(output_dir / 'hybrid_trades.csv', index=False)

    logger.info(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
