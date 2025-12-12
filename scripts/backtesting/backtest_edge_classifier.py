#!/usr/bin/env python3
"""
Backtest simulation for Edge Classifier.

Runs the trained edge classifier on test data and generates:
- Equity curve
- ROI
- Max drawdown
- Profit factor
- Monthly breakdown
- Win/loss statistics
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.edge.classifier import EdgeClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_edge_data(city: str, pnl_mode: str = "realistic") -> pd.DataFrame:
    """Load cached edge training data.

    Args:
        city: City name
        pnl_mode: "realistic" or "simplified"

    Returns:
        DataFrame with edge data
    """
    cache_path = Path(f"models/saved/{city}/edge_training_data_{pnl_mode}.parquet")
    if not cache_path.exists():
        raise FileNotFoundError(f"No cached edge data at {cache_path}")

    df = pd.read_parquet(cache_path)
    logger.info(f"Loaded {len(df):,} rows from {cache_path}")
    return df


def run_backtest(
    df: pd.DataFrame,
    classifier: EdgeClassifier,
    starting_bankroll: float = 1000.0,
    bet_per_trade: float = 10.0,
    max_bet: float = 75.0,
    use_kelly: bool = False,
    kelly_fraction: float = 0.25,
) -> dict:
    """Run backtest simulation.

    Args:
        df: Edge data DataFrame (should be test set only)
        classifier: Trained EdgeClassifier
        starting_bankroll: Starting capital
        bet_per_trade: Fixed bet size (if not using Kelly)
        max_bet: Maximum bet size
        use_kelly: Whether to use Kelly sizing
        kelly_fraction: Fraction of Kelly to use

    Returns:
        Dict with backtest results
    """
    # Filter to valid trades only (has P&L)
    df_valid = df[df["pnl"].notna()].copy()

    if df_valid.empty:
        logger.error("No valid trades in data")
        return {}

    # Sort by day and snapshot_time for chronological order
    sort_cols = ["day"]
    if "snapshot_time" in df_valid.columns:
        sort_cols.append("snapshot_time")
    df_valid = df_valid.sort_values(sort_cols).reset_index(drop=True)

    # Get classifier predictions
    logger.info("Running classifier predictions...")
    proba = classifier.predict(df_valid)
    predictions = (proba >= classifier.decision_threshold).astype(int)

    # Initialize tracking
    equity = [starting_bankroll]
    trades = []
    daily_pnl = {}
    monthly_pnl = {}

    current_equity = starting_bankroll

    for i, (idx, row) in enumerate(df_valid.iterrows()):
        pred = predictions[i]

        if pred == 0:
            # Model says don't trade
            continue

        # Model says trade - calculate bet size
        if use_kelly:
            # Kelly sizing: f = (p*b - q) / b where b=1 for binary
            # Simplified: f = 2p - 1 for 1:1 payoff
            p = proba[i]
            kelly_bet = kelly_fraction * (2 * p - 1) * current_equity
            bet_size = min(max(kelly_bet, 0), max_bet)
        else:
            bet_size = min(bet_per_trade, max_bet)

        if bet_size <= 0:
            continue

        # Calculate P&L for this trade
        # The 'pnl' column is per-contract, scale by bet size
        pnl_per_contract = row["pnl"]
        entry_price = row.get("entry_price_cents", 50) / 100.0

        # Number of contracts we can buy with bet_size
        num_contracts = bet_size / entry_price
        trade_pnl = pnl_per_contract * num_contracts

        # Update equity
        current_equity += trade_pnl
        equity.append(current_equity)

        # Track trade
        trade_day = row["day"]
        if isinstance(trade_day, str):
            trade_day = pd.to_datetime(trade_day).date()

        trade_info = {
            "day": trade_day,
            "snapshot_time": row.get("snapshot_time"),
            "bet_size": bet_size,
            "pnl": trade_pnl,
            "pnl_pct": trade_pnl / bet_size if bet_size > 0 else 0,
            "equity_after": current_equity,
            "proba": proba[i],
            "entry_price": entry_price,
            "won": trade_pnl > 0,
        }
        trades.append(trade_info)

        # Track daily P&L
        day_str = str(trade_day)
        if day_str not in daily_pnl:
            daily_pnl[day_str] = 0
        daily_pnl[day_str] += trade_pnl

        # Track monthly P&L
        month_str = str(trade_day)[:7]  # YYYY-MM
        if month_str not in monthly_pnl:
            monthly_pnl[month_str] = 0
        monthly_pnl[month_str] += trade_pnl

    if not trades:
        logger.warning("No trades executed")
        return {"n_trades": 0}

    trades_df = pd.DataFrame(trades)
    equity_series = pd.Series(equity)

    # Calculate metrics
    total_pnl = current_equity - starting_bankroll
    roi_pct = (total_pnl / starting_bankroll) * 100

    # Win/loss stats
    n_trades = len(trades_df)
    n_wins = trades_df["won"].sum()
    n_losses = n_trades - n_wins
    win_rate = n_wins / n_trades if n_trades > 0 else 0

    # Average win/loss
    wins = trades_df[trades_df["pnl"] > 0]["pnl"]
    losses = trades_df[trades_df["pnl"] < 0]["pnl"]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0

    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    running_max = equity_series.cummax()
    drawdown = equity_series - running_max
    max_drawdown = drawdown.min()
    max_drawdown_pct = (max_drawdown / running_max[drawdown.idxmin()]) * 100 if max_drawdown < 0 else 0

    # Sharpe ratio (annualized, assuming ~250 trading days)
    daily_returns = trades_df.groupby("day")["pnl"].sum()
    if len(daily_returns) > 1:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(250)
    else:
        sharpe = 0

    # Time period
    first_day = trades_df["day"].min()
    last_day = trades_df["day"].max()
    n_days = (last_day - first_day).days if hasattr(last_day - first_day, 'days') else 0

    results = {
        # Summary
        "starting_bankroll": starting_bankroll,
        "ending_equity": current_equity,
        "total_pnl": total_pnl,
        "roi_pct": roi_pct,

        # Trade stats
        "n_trades": n_trades,
        "n_wins": int(n_wins),
        "n_losses": int(n_losses),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_trade": trades_df["pnl"].mean(),

        # Risk metrics
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_annual": sharpe,

        # Period
        "first_day": str(first_day),
        "last_day": str(last_day),
        "n_days": n_days,
        "trades_per_day": n_trades / max(n_days, 1),

        # Detailed data
        "equity_curve": equity,
        "monthly_pnl": monthly_pnl,
        "trades": trades,
    }

    return results


def print_results(results: dict, city: str):
    """Print formatted backtest results."""
    if not results or results.get("n_trades", 0) == 0:
        print("No trades to report")
        return

    print()
    print("=" * 70)
    print(f"BACKTEST RESULTS - {city.upper()}")
    print("=" * 70)
    print()

    print(f"Period: {results['first_day']} to {results['last_day']} ({results['n_days']} days)")
    print()

    print("--- CAPITAL ---")
    print(f"Starting bankroll:  ${results['starting_bankroll']:,.2f}")
    print(f"Ending equity:      ${results['ending_equity']:,.2f}")
    print(f"Total P&L:          ${results['total_pnl']:,.2f}")
    print(f"ROI:                {results['roi_pct']:.1f}%")
    print()

    print("--- TRADE STATS ---")
    print(f"Total trades:       {results['n_trades']}")
    print(f"Trades per day:     {results['trades_per_day']:.2f}")
    print(f"Wins:               {results['n_wins']} ({results['win_rate']*100:.1f}%)")
    print(f"Losses:             {results['n_losses']}")
    print(f"Avg trade P&L:      ${results['avg_trade']:.2f}")
    print(f"Avg win:            ${results['avg_win']:.2f}")
    print(f"Avg loss:           ${results['avg_loss']:.2f}")
    print()

    print("--- RISK METRICS ---")
    print(f"Profit factor:      {results['profit_factor']:.2f}")
    print(f"Max drawdown:       ${results['max_drawdown']:.2f} ({results['max_drawdown_pct']:.1f}%)")
    print(f"Sharpe (annual):    {results['sharpe_annual']:.2f}")
    print()

    # Monthly breakdown
    monthly = results.get("monthly_pnl", {})
    if monthly:
        print("--- MONTHLY P&L ---")
        for month, pnl in sorted(monthly.items()):
            print(f"  {month}: ${pnl:+,.2f}")
        print()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Backtest Edge Classifier")
    parser.add_argument(
        "--city",
        type=str,
        default="austin",
        help="City to backtest",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Starting bankroll (default: $1000)",
    )
    parser.add_argument(
        "--bet-size",
        type=float,
        default=10.0,
        help="Fixed bet size per trade (default: $10)",
    )
    parser.add_argument(
        "--max-bet",
        type=float,
        default=75.0,
        help="Maximum bet size (default: $75)",
    )
    parser.add_argument(
        "--kelly",
        action="store_true",
        help="Use Kelly sizing instead of fixed bet",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Fraction of Kelly to use (default: 0.25 = quarter Kelly)",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=True,
        help="Only use test set (last 20%% of data)",
    )
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="Use all data (train + test) - for analysis only",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("EDGE CLASSIFIER BACKTEST")
    print("=" * 70)
    print(f"City: {args.city}")
    print(f"Starting bankroll: ${args.bankroll:,.2f}")
    print(f"Bet size: ${args.bet_size:.2f} (max: ${args.max_bet:.2f})")
    print(f"Kelly sizing: {args.kelly} (fraction: {args.kelly_fraction})")
    print()

    # Load classifier
    model_path = Path(f"models/saved/{args.city}/edge_classifier")
    classifier = EdgeClassifier()
    classifier.load(model_path)
    logger.info(f"Loaded classifier: threshold={classifier.decision_threshold:.3f}")

    # Load edge data
    df = load_edge_data(args.city, pnl_mode="realistic")

    # Filter to signals only
    df = df[df["signal"] != "no_trade"].copy()
    logger.info(f"Edge signals: {len(df):,}")

    # Split to test set if requested
    if not args.all_data:
        # Use same split as training (last 20%)
        df = df.sort_values(["day"]).reset_index(drop=True)
        test_start = int(len(df) * 0.8)
        df = df.iloc[test_start:].copy()
        logger.info(f"Test set: {len(df):,} samples")

    # Run backtest
    results = run_backtest(
        df=df,
        classifier=classifier,
        starting_bankroll=args.bankroll,
        bet_per_trade=args.bet_size,
        max_bet=args.max_bet,
        use_kelly=args.kelly,
        kelly_fraction=args.kelly_fraction,
    )

    # Print results
    print_results(results, args.city)

    # Save to JSON if requested
    if args.output:
        # Remove non-serializable items for JSON
        output_results = {k: v for k, v in results.items()
                        if k not in ["equity_curve", "trades"]}
        output_results["n_equity_points"] = len(results.get("equity_curve", []))

        with open(args.output, "w") as f:
            json.dump(output_results, f, indent=2, default=str)
        print(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
