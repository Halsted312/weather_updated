#!/usr/bin/env python3
"""
ML Hybrid Backtest with Optuna Trading Parameter Optimization

Backtests the hybrid ML model strategy using:
- Market-Clock model for D-1 predictions
- TOD v1 model for event day predictions
- prob_to_orders module for trading recommendations
- Optuna for trading parameter optimization

The strategy:
1. At D-1 H_switch (market open), get Market-Clock model prediction
2. At D H_switch (event day open), switch to TOD v1 model
3. For each snapshot, compute P[delta] and convert to bracket recommendations
4. Execute trades when edge exceeds threshold

Usage:
    # Standard backtest
    .venv/bin/python scripts/backtest_ml_hybrid.py --city chicago --days 30

    # With custom model directory
    .venv/bin/python scripts/backtest_ml_hybrid.py --city chicago --days 60 \
        --model-dir models/saved/market_clock_validation/

    # With Optuna trading parameter sweep
    .venv/bin/python scripts/backtest_ml_hybrid.py --city chicago --days 60 \
        --use-optuna --trials 30 --model-dir models/saved/market_clock_validation/
"""

import argparse
import logging
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import Pool

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from open_maker.prob_to_orders import DeltaProbToOrders, BracketRecommendation, HorizonRiskConfig
from open_maker.utils import kalshi_taker_fee, kalshi_maker_fee, CITY_TIMEZONES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CITIES = ['chicago', 'austin', 'denver', 'los_angeles', 'miami', 'philadelphia']
H_SWITCH = 10  # Hour when D-1 switches to D (event day)

# Default delta classes for ordinal model (updated to [-10, +10])
DEFAULT_DELTA_CLASSES = list(range(-10, 11))  # [-10, -9, ..., 0, ..., +10]

# Default HorizonRiskConfig base multipliers
BASE_SIZE_MULTS = [1.0, 0.5, 0.25, 0.15, 0.08]
BASE_EDGE_MULTS = [1.0, 1.2, 1.5, 2.0, 2.5]


@dataclass
class BacktestTrade:
    """Record of a single backtest trade."""
    city: str
    event_date: date
    snapshot_time: datetime
    model_type: str  # 'market_clock' or 'tod_v1'
    time_window: str  # 'D-1' or 'D'
    ticker: str
    action: str  # 'buy_yes' or 'buy_no'
    entry_price: float
    num_contracts: int
    model_prob: float
    edge_pct: float
    expected_value: float
    settle_temp: Optional[int] = None
    won: Optional[bool] = None
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None


class HybridBacktester:
    """
    Backtest hybrid model strategy over test data.

    Uses saved test data parquet files to avoid re-running model inference.
    Supports horizon-aware position sizing via HorizonRiskConfig scaling.
    """

    def __init__(
        self,
        cities: list[str],
        test_days: int = 60,
        min_edge_pct: float = 5.0,
        min_prob: float = 0.10,
        bet_amount_usd: float = 100.0,
        maker_fill_prob: float = 0.4,
        edge_mult_scale: float = 1.0,
        size_mult_scale: float = 1.0,
        max_positions: int = 1,
        model_dir: str = 'models/saved/market_clock_tod_v1',
        quiet: bool = False,
    ):
        self.cities = cities
        self.test_days = test_days
        self.min_edge_pct = min_edge_pct
        self.min_prob = min_prob
        self.bet_amount_usd = bet_amount_usd
        self.maker_fill_prob = maker_fill_prob
        self.edge_mult_scale = edge_mult_scale
        self.size_mult_scale = size_mult_scale
        self.max_positions = max_positions
        self.model_dir = model_dir
        self.quiet = quiet

        self.trades: list[BacktestTrade] = []

        # Build horizon config with scaled multipliers
        scaled_edge_mults = [m * edge_mult_scale for m in BASE_EDGE_MULTS]
        scaled_size_mults = [m * size_mult_scale for m in BASE_SIZE_MULTS]
        self.horizon_config = HorizonRiskConfig(
            edge_multipliers=scaled_edge_mults,
            size_multipliers=scaled_size_mults,
        )

        # Load models from specified directory
        mc_model_path = f'{model_dir}/ordinal_catboost_market_clock_tod_v1.pkl'
        self.mc_model = self._load_model(mc_model_path)

        # Try to load TOD v1 models (optional - fall back to market-clock only)
        self.tod_models = {}
        for city in cities:
            tod_path = f'models/saved/{city}_tod_v1/ordinal_catboost_tod_v1.pkl'
            try:
                self.tod_models[city] = self._load_model(tod_path)
            except FileNotFoundError:
                if not quiet:
                    logger.warning(f"TOD v1 model not found for {city}, using Market-Clock only")

        # Get delta classes from model
        self.delta_classes = self.mc_model.get('delta_classes', DEFAULT_DELTA_CLASSES)

        # Initialize prob_to_orders bridge with horizon config
        self.bridge = DeltaProbToOrders(
            delta_classes=self.delta_classes,
            min_edge_pct=min_edge_pct,
            min_prob=min_prob,
            horizon_config=self.horizon_config,
        )

        # Load test data from model directory
        mc_test_path = f'{model_dir}/test_data.parquet'
        self.mc_test = pd.read_parquet(mc_test_path)

        # Try to load TOD v1 test data (optional)
        self.tod_tests = {}
        for city in cities:
            tod_test_path = f'models/saved/{city}_tod_v1/test_data.parquet'
            try:
                self.tod_tests[city] = pd.read_parquet(tod_test_path)
            except FileNotFoundError:
                pass  # Already warned above

        if not quiet:
            logger.info(f"Loaded Market-Clock test: {len(self.mc_test):,} rows")
            for city in cities:
                if city in self.tod_tests:
                    logger.info(f"Loaded {city} TOD v1 test: {len(self.tod_tests[city]):,} rows")

    def _load_model(self, path: str) -> dict:
        """Load a pickled model."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _predict_ordinal(self, model_data: dict, features_row: pd.Series) -> np.ndarray:
        """
        Run ordinal prediction for a single row.

        Returns:
            Array of delta class probabilities, shape (n_classes,)
        """
        classifiers = model_data['classifiers']
        thresholds = model_data['thresholds']
        feature_cols = model_data['feature_cols']
        delta_classes = model_data.get('delta_classes', DEFAULT_DELTA_CLASSES)

        # Create features DataFrame
        features_df = pd.DataFrame([features_row])

        # Fill missing features
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = np.nan

        X = features_df[feature_cols].copy()

        # Fill NaN with median (or 0 for simplicity)
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(0)

        n_classes = len(delta_classes)
        cum_proba = np.ones(len(thresholds) + 1)

        for i, k in enumerate(thresholds):
            clf_data = classifiers.get(k)
            if clf_data is None:
                continue

            if isinstance(clf_data, dict) and clf_data.get('type') == 'constant':
                cum_proba[i + 1] = clf_data['prob']
            else:
                pool = Pool(X)
                p_ge_k = clf_data.predict_proba(pool)[0, 1]
                cum_proba[i + 1] = p_ge_k

        # Convert cumulative to class probabilities
        class_proba = np.zeros(n_classes)
        for i in range(n_classes):
            if i == 0:
                class_proba[i] = 1 - cum_proba[1]
            elif i == n_classes - 1:
                class_proba[i] = cum_proba[i]
            else:
                class_proba[i] = cum_proba[i] - cum_proba[i + 1]

        # Normalize to ensure valid probabilities
        class_proba = np.clip(class_proba, 0, 1)
        if class_proba.sum() > 0:
            class_proba /= class_proba.sum()

        return class_proba

    def _simulate_trade(
        self,
        city: str,
        event_date: date,
        snapshot_time: datetime,
        model_type: str,
        time_window: str,
        t_base: float,
        delta_proba: np.ndarray,
        settle_f: int,
    ) -> Optional[BacktestTrade]:
        """
        Simulate a trade decision for a single snapshot.

        Returns:
            BacktestTrade if a trade was recommended, None otherwise
        """
        # Create synthetic brackets DataFrame (simplified - in practice, load from DB)
        # For backtesting purposes, we simulate bracket prices based on model probabilities
        # In production, these would come from actual Kalshi candles

        # Get model's expected settlement
        expected_settle = t_base + sum(d * p for d, p in zip(self.delta_classes, delta_proba))

        # Create mock brackets around expected settlement
        # This is a simplified simulation - real backtest should use actual candle data
        brackets = []
        for strike in range(int(expected_settle) - 5, int(expected_settle) + 6):
            if strike < 0:
                continue
            brackets.append({
                'ticker': f'TEMP-{city.upper()}-{event_date.isoformat()}-{strike}',
                'floor_strike': float(strike),
                'cap_strike': float(strike + 2),
                'yes_bid': 45.0,  # Simplified - would be from candles
                'yes_ask': 55.0,
            })

        if not brackets:
            return None

        brackets_df = pd.DataFrame(brackets)

        # Get recommendations using prob_to_orders
        recs = self.bridge.get_recommendations(delta_proba, t_base, brackets_df)

        # Find best recommendation
        best_rec = None
        best_ev = 0.0
        for rec in recs:
            if rec.action in ('buy_yes', 'buy_no') and rec.expected_value > best_ev:
                best_rec = rec
                best_ev = rec.expected_value

        if best_rec is None:
            return None

        # Check if trade would win
        won = None
        gross_pnl = None
        net_pnl = None

        if settle_f is not None:
            # Determine if bracket won
            floor_s = best_rec.floor_strike
            cap_s = best_rec.cap_strike
            bracket_wins = False
            if floor_s is not None and cap_s is not None:
                bracket_wins = floor_s <= settle_f <= cap_s
            elif floor_s is None and cap_s is not None:
                bracket_wins = settle_f <= cap_s
            elif floor_s is not None and cap_s is None:
                bracket_wins = settle_f >= floor_s

            # Determine if our position won
            if best_rec.action == 'buy_yes':
                won = bracket_wins
            else:  # buy_no
                won = not bracket_wins

            # Calculate P&L
            entry_price = best_rec.entry_price
            num_contracts = max(1, int(self.bet_amount_usd / entry_price))

            if won:
                gross_pnl = (100 - entry_price) * num_contracts / 100  # $ per contract
            else:
                gross_pnl = -entry_price * num_contracts / 100

            # Subtract fees (assume taker for simplicity)
            fee = kalshi_taker_fee(entry_price, num_contracts)
            net_pnl = gross_pnl - fee

        return BacktestTrade(
            city=city,
            event_date=event_date,
            snapshot_time=snapshot_time,
            model_type=model_type,
            time_window=time_window,
            ticker=best_rec.ticker,
            action=best_rec.action,
            entry_price=best_rec.entry_price,
            num_contracts=max(1, int(self.bet_amount_usd / best_rec.entry_price)),
            model_prob=best_rec.model_prob_yes if best_rec.action == 'buy_yes' else best_rec.model_prob_no,
            edge_pct=best_rec.edge_pct,
            expected_value=best_rec.expected_value,
            settle_temp=settle_f,
            won=won,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
        )

    def run_backtest(self) -> dict:
        """
        Run the hybrid backtest.

        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info("="*80)
        logger.info("HYBRID ML BACKTEST")
        logger.info("="*80)

        # Get unique events
        events = []
        for city in self.cities:
            city_events = self.mc_test[self.mc_test['city'] == city]['event_date'].unique()
            for ed in city_events:
                if isinstance(ed, pd.Timestamp):
                    ed = ed.date()
                events.append((city, ed))

        events = sorted(events, key=lambda x: (x[1], x[0]))

        # Limit to test_days
        if self.test_days > 0:
            events = events[-self.test_days * len(self.cities):]

        logger.info(f"Backtesting {len(events)} events")

        # Process each event
        for city, event_date in events:
            self._process_event(city, event_date)

        # Calculate metrics
        return self._calculate_metrics()

    def _process_event(self, city: str, event_date: date):
        """Process a single event (city, date)."""
        # Get D-1 snapshots from Market-Clock model
        d_minus_1_mask = (
            (self.mc_test['city'] == city) &
            (self.mc_test['event_date'] == pd.Timestamp(event_date)) &
            (self.mc_test['is_d_minus_1'] == 1)
        )
        d_minus_1_snapshots = self.mc_test[d_minus_1_mask]

        # Get D snapshots from TOD v1 model
        tod_test = self.tod_tests[city]
        if 'day' in tod_test.columns:
            d_mask = tod_test['day'] == pd.Timestamp(event_date)
        else:
            d_mask = tod_test['event_date'] == pd.Timestamp(event_date)
        d_snapshots = tod_test[d_mask]

        # Get settlement
        if not d_minus_1_snapshots.empty:
            settle_f = int(d_minus_1_snapshots.iloc[0]['settle_f'])
        elif not d_snapshots.empty:
            settle_f = int(d_snapshots.iloc[0]['settle_f'])
        else:
            return

        # Process D-1 snapshot (use 10:00 snapshot)
        if not d_minus_1_snapshots.empty:
            hour_10_mask = d_minus_1_snapshots['snapshot_datetime'].dt.hour == H_SWITCH
            d1_10 = d_minus_1_snapshots[hour_10_mask]
            if not d1_10.empty:
                row = d1_10.iloc[0]
                delta_proba = self._predict_ordinal(self.mc_model, row)
                trade = self._simulate_trade(
                    city=city,
                    event_date=event_date,
                    snapshot_time=row['snapshot_datetime'],
                    model_type='market_clock',
                    time_window='D-1',
                    t_base=row['t_base'],
                    delta_proba=delta_proba,
                    settle_f=settle_f,
                )
                if trade:
                    self.trades.append(trade)

        # Process D snapshot (use 10:00 snapshot)
        if not d_snapshots.empty:
            if 'snapshot_datetime' in d_snapshots.columns:
                hour_10_mask = d_snapshots['snapshot_datetime'].dt.hour == H_SWITCH
            else:
                hour_10_mask = d_snapshots['snapshot_hour'] == H_SWITCH
            d_10 = d_snapshots[hour_10_mask]
            if not d_10.empty:
                row = d_10.iloc[0]
                delta_proba = self._predict_ordinal(self.tod_models[city], row)
                trade = self._simulate_trade(
                    city=city,
                    event_date=event_date,
                    snapshot_time=row.get('snapshot_datetime', datetime.combine(event_date, datetime.min.time())),
                    model_type='tod_v1',
                    time_window='D',
                    t_base=row['t_base'],
                    delta_proba=delta_proba,
                    settle_f=settle_f,
                )
                if trade:
                    self.trades.append(trade)

    def _calculate_metrics(self) -> dict:
        """Calculate backtest metrics including Sharpe ratio."""
        if not self.trades:
            return {'error': 'No trades executed', 'total_trades': 0, 'sharpe': 0.0, 'total_net_pnl': 0.0}

        trades_df = pd.DataFrame([{
            'city': t.city,
            'event_date': t.event_date,
            'model_type': t.model_type,
            'time_window': t.time_window,
            'action': t.action,
            'entry_price': t.entry_price,
            'model_prob': t.model_prob,
            'edge_pct': t.edge_pct,
            'expected_value': t.expected_value,
            'won': t.won,
            'gross_pnl': t.gross_pnl,
            'net_pnl': t.net_pnl,
        } for t in self.trades])

        # Overall metrics
        total_trades = len(trades_df)
        won_trades = trades_df['won'].sum() if trades_df['won'].notna().any() else 0
        win_rate = won_trades / total_trades if total_trades > 0 else 0

        total_net_pnl = trades_df['net_pnl'].sum() if trades_df['net_pnl'].notna().any() else 0
        avg_pnl_per_trade = total_net_pnl / total_trades if total_trades > 0 else 0

        # Calculate Sharpe ratio (annualized, assuming daily returns)
        # Group by event_date to get daily P&L
        daily_pnl = trades_df.groupby('event_date')['net_pnl'].sum()
        if len(daily_pnl) > 1:
            mean_daily = daily_pnl.mean()
            std_daily = daily_pnl.std()
            sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0.0
        else:
            sharpe = 0.0

        # By time window
        d1_trades = trades_df[trades_df['time_window'] == 'D-1']
        d_trades = trades_df[trades_df['time_window'] == 'D']

        d1_pnl = d1_trades['net_pnl'].sum() if len(d1_trades) > 0 else 0
        d_pnl = d_trades['net_pnl'].sum() if len(d_trades) > 0 else 0

        results = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_net_pnl': total_net_pnl,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'sharpe': sharpe,
            'd1_trades': len(d1_trades),
            'd1_net_pnl': d1_pnl,
            'd_trades': len(d_trades),
            'd_net_pnl': d_pnl,
            'trades_df': trades_df,
        }

        return results


def print_results(results: dict, params: dict = None):
    """Print backtest results in a nice format."""
    print("\n" + "="*80)
    print("HYBRID ML BACKTEST RESULTS")
    print("="*80)

    if 'error' in results and results.get('total_trades', 0) == 0:
        print(f"ERROR: {results['error']}")
        return

    if params:
        print(f"\n## Parameters")
        for k, v in params.items():
            print(f"  {k}: {v}")

    print(f"\n## Overall Metrics")
    print(f"  Total trades: {results['total_trades']}")
    print(f"  Win rate: {results['win_rate']:.1%}")
    print(f"  Total net P&L: ${results['total_net_pnl']:.2f}")
    print(f"  Avg P&L per trade: ${results['avg_pnl_per_trade']:.2f}")
    print(f"  Sharpe ratio: {results['sharpe']:.2f}")

    print(f"\n## By Time Window")
    print(f"  D-1 (Market-Clock): {results['d1_trades']} trades, ${results['d1_net_pnl']:.2f} P&L")
    print(f"  D (TOD v1): {results['d_trades']} trades, ${results['d_net_pnl']:.2f} P&L")

    # Show sample trades
    if 'trades_df' in results and not results['trades_df'].empty:
        print(f"\n## Sample Trades (first 10)")
        sample = results['trades_df'].head(10)
        print(sample[['city', 'event_date', 'time_window', 'action', 'entry_price', 'edge_pct', 'won', 'net_pnl']].to_string())

    print("\n" + "="*80)


def run_optuna_optimization(
    cities: list[str],
    test_days: int,
    model_dir: str,
    n_trials: int,
    bet_amount_usd: float,
) -> dict:
    """Run Optuna optimization for trading parameters."""
    import optuna
    from optuna.samplers import TPESampler

    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        # Suggest trading parameters
        min_edge_pct = trial.suggest_float('min_edge_pct', 2.0, 16.0)
        min_prob = trial.suggest_float('min_prob', 0.05, 0.25)
        edge_mult_scale = trial.suggest_float('edge_mult_scale', 0.5, 2.0)
        size_mult_scale = trial.suggest_float('size_mult_scale', 0.5, 2.0)
        max_positions = trial.suggest_categorical('max_positions', [1, 2, 3])

        try:
            backtester = HybridBacktester(
                cities=cities,
                test_days=test_days,
                min_edge_pct=min_edge_pct,
                min_prob=min_prob,
                bet_amount_usd=bet_amount_usd,
                edge_mult_scale=edge_mult_scale,
                size_mult_scale=size_mult_scale,
                max_positions=max_positions,
                model_dir=model_dir,
                quiet=True,
            )
            results = backtester.run_backtest()

            # Prefer Sharpe, fall back to P&L
            sharpe = results.get('sharpe', 0.0)
            pnl = results.get('total_net_pnl', 0.0)

            # Use Sharpe if we have enough trades, otherwise P&L
            if results.get('total_trades', 0) >= 10 and sharpe != 0:
                return sharpe
            else:
                return pnl

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('-inf')

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
    )

    # Run optimization
    logger.info(f"Starting Optuna optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Get best params
    best_params = study.best_params
    best_value = study.best_value

    logger.info(f"\nBest parameters found:")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"Best objective value: {best_value:.4f}")

    return {
        'best_params': best_params,
        'best_value': best_value,
        'study': study,
    }


def main():
    parser = argparse.ArgumentParser(description='Run hybrid ML backtest with optional Optuna optimization')
    parser.add_argument('--city', type=str, default=None, help='Single city to backtest')
    parser.add_argument('--all-cities', action='store_true', help='Backtest all 6 cities')
    parser.add_argument('--days', type=int, default=30, help='Number of test days (default: 30)')
    parser.add_argument('--min-edge', type=float, default=5.0, help='Minimum edge %% (default: 5)')
    parser.add_argument('--min-prob', type=float, default=0.10, help='Minimum probability (default: 0.10)')
    parser.add_argument('--bet-amount', type=float, default=100.0, help='Bet amount in USD (default: 100)')
    parser.add_argument('--model-dir', type=str, default='models/saved/market_clock_tod_v1',
                       help='Model directory (default: models/saved/market_clock_tod_v1)')
    parser.add_argument('--edge-mult-scale', type=float, default=1.0,
                       help='Scale factor for edge multipliers (default: 1.0)')
    parser.add_argument('--size-mult-scale', type=float, default=1.0,
                       help='Scale factor for size multipliers (default: 1.0)')
    parser.add_argument('--max-positions', type=int, default=1,
                       help='Max positions per event (default: 1)')
    parser.add_argument('--use-optuna', action='store_true',
                       help='Enable Optuna trading parameter optimization')
    parser.add_argument('--trials', type=int, default=30,
                       help='Number of Optuna trials (default: 30)')
    args = parser.parse_args()

    if args.city:
        cities = [args.city]
    elif args.all_cities:
        cities = CITIES
    else:
        cities = ['chicago']  # Default

    if args.use_optuna:
        # Run Optuna optimization
        logger.info(f"Running Optuna optimization for: {', '.join(cities)}")
        logger.info(f"Test days: {args.days}, Trials: {args.trials}")

        optuna_results = run_optuna_optimization(
            cities=cities,
            test_days=args.days,
            model_dir=args.model_dir,
            n_trials=args.trials,
            bet_amount_usd=args.bet_amount,
        )

        # Run final backtest with best params
        best_params = optuna_results['best_params']
        logger.info("\nRunning final backtest with best parameters...")

        backtester = HybridBacktester(
            cities=cities,
            test_days=args.days,
            min_edge_pct=best_params['min_edge_pct'],
            min_prob=best_params['min_prob'],
            bet_amount_usd=args.bet_amount,
            edge_mult_scale=best_params['edge_mult_scale'],
            size_mult_scale=best_params['size_mult_scale'],
            max_positions=best_params['max_positions'],
            model_dir=args.model_dir,
        )

        results = backtester.run_backtest()
        print_results(results, params=best_params)

        # Print optimization summary
        print("\n" + "="*80)
        print("OPTUNA OPTIMIZATION SUMMARY")
        print("="*80)
        print(f"\nBest parameters:")
        for k, v in best_params.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print(f"\nOptimization objective value: {optuna_results['best_value']:.4f}")
        print("="*80)

    else:
        # Standard backtest
        logger.info(f"Running hybrid ML backtest for: {', '.join(cities)}")
        logger.info(f"Test days: {args.days}, Min edge: {args.min_edge}%, Bet amount: ${args.bet_amount}")

        backtester = HybridBacktester(
            cities=cities,
            test_days=args.days,
            min_edge_pct=args.min_edge,
            min_prob=args.min_prob,
            bet_amount_usd=args.bet_amount,
            edge_mult_scale=args.edge_mult_scale,
            size_mult_scale=args.size_mult_scale,
            max_positions=args.max_positions,
            model_dir=args.model_dir,
        )

        results = backtester.run_backtest()
        print_results(results)

    return 0


if __name__ == '__main__':
    sys.exit(main())
