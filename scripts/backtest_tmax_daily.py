#!/usr/bin/env python3
"""Daily baseline backtest using Tmax ensemble snapshots."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sqlalchemy import text

from backtest.run_backtest import load_markets_with_settlements, calculate_outcome_from_settlement
from backtest.model_kelly_adapter import probability_from_tmax
from db.connection import get_session
from ml.dataset import CITY_CONFIG
from models.costs import effective_yes_entry_cents, effective_no_entry_cents
from weather.time_utils import get_timezone

logger = logging.getLogger(__name__)


def parse_time(value: str) -> time:
    hour, minute = map(int, value.split(":"))
    return time(hour=hour, minute=minute)


def load_candles(city: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    series_code = CITY_CONFIG[city]["series_code"]
    series_ticker = f"KXHIGH{series_code}"
    with get_session() as session:
        query = text(
            """
            SELECT market_ticker, timestamp, close
            FROM candles
            WHERE market_ticker LIKE :series_pattern
              AND period_minutes = 1
              AND timestamp >= :start_dt
              AND timestamp <= :end_dt
            ORDER BY market_ticker, timestamp
            """
        )
        rows = session.execute(
            query,
            {
                "series_pattern": f"{series_ticker}%",
                "start_dt": start_dt,
                "end_dt": end_dt,
            },
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["market_ticker", "timestamp", "close", "yes_bid", "yes_ask"])

    df = pd.DataFrame(rows, columns=["market_ticker", "timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["yes_bid"] = (df["close"] - 1).clip(lower=1)
    df["yes_ask"] = (df["close"] + 1).clip(upper=99)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily Tmax baseline backtest")
    parser.add_argument("--city", required=True, choices=list(CITY_CONFIG.keys()))
    parser.add_argument("--bracket", required=True, choices=["between", "greater", "less"])
    parser.add_argument("--tmax-preds-csv", required=True, help="CSV with per-minute Tmax predictions")
    parser.add_argument("--start-date", required=True, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--cutoff", default="16:00", help="Local cutoff HH:MM for daily trade (default: 16:00)")
    parser.add_argument("--contracts", type=int, default=1, help="Contracts per market (default: 1)")
    parser.add_argument("--initial-cash", type=float, default=10_000.0, help="Initial cash in dollars")
    parser.add_argument("--output-json", help="Optional path to save summary JSON")
    parser.add_argument("--min-edge", type=float, default=0.0, help="Minimum |p-0.5| edge to trade")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    start_date = datetime.fromisoformat(args.start_date).date()
    end_date = datetime.fromisoformat(args.end_date).date()
    cutoff_time = parse_time(args.cutoff)
    timezone_local = get_timezone(args.city)

    markets_df = load_markets_with_settlements(args.city, start_date, end_date)
    markets_df = markets_df[markets_df["strike_type"] == args.bracket].copy()
    markets_df["calc_outcome"] = markets_df.apply(calculate_outcome_from_settlement, axis=1)

    if markets_df.empty:
        logger.error("No markets available for %s between %s and %s", args.city, start_date, end_date)
        return 1

    preds_df = pd.read_csv(args.tmax_preds_csv)
    if preds_df.empty:
        logger.error("Prediction CSV %s is empty", args.tmax_preds_csv)
        return 1
    preds_df["timestamp"] = pd.to_datetime(preds_df["timestamp"], utc=True)
    preds_df["timestamp_local"] = preds_df["timestamp"].dt.tz_convert(timezone_local)
    preds_df["date_local"] = preds_df["timestamp_local"].dt.date
    preds_df["minute_of_day"] = preds_df["timestamp_local"].dt.hour * 60 + preds_df["timestamp_local"].dt.minute

    preds_by_date: Dict[datetime.date, pd.DataFrame] = {}
    for date_value, group in preds_df.groupby("date_local"):
        preds_by_date[date_value] = group.sort_values("timestamp")

    start_dt = datetime.combine(start_date, time.min, tzinfo=timezone_local).astimezone(timezone.utc)
    end_dt = datetime.combine(end_date, time.max, tzinfo=timezone_local).astimezone(timezone.utc)
    candles_df = load_candles(args.city, start_dt, end_dt)
    candles_by_ticker = {
        ticker: group.sort_values("timestamp").reset_index(drop=True)
        for ticker, group in candles_df.groupby("market_ticker")
    }

    trades: List[dict] = []
    equity = args.initial_cash * 100
    equity_curve = []

    for event_date, day_markets in markets_df.groupby("date_local"):
        snapshot_df = preds_by_date.get(event_date)
        if snapshot_df is None:
            logger.debug("No Tmax snapshot for %s", event_date)
            continue
        cutoff_minutes = cutoff_time.hour * 60 + cutoff_time.minute
        snapshot = snapshot_df[snapshot_df["minute_of_day"] <= cutoff_minutes].tail(1)
        if snapshot.empty:
            logger.debug("No Tmax data before cutoff for %s", event_date)
            continue
        snapshot_row = snapshot.iloc[0]
        mu = float(snapshot_row["pred"])
        sigma = float(snapshot_row.get("sigma_est", 4.0))

        cutoff_local = datetime.combine(event_date, cutoff_time, tzinfo=timezone_local)
        cutoff_utc = cutoff_local.astimezone(timezone.utc)

        for _, market in day_markets.iterrows():
            outcome = market["calc_outcome"]
            if outcome is None:
                continue
            ticker = market["ticker"]
            candle_df = candles_by_ticker.get(ticker)
            if candle_df is None:
                continue
            candle = candle_df[candle_df["timestamp"] <= cutoff_utc]
            if candle.empty:
                continue
            candle_row = candle.iloc[-1]
            bid = int(candle_row["yes_bid"])
            ask = int(candle_row["yes_ask"])
            if bid >= ask:
                continue

            prob_yes = probability_from_tmax(
                mu,
                max(1.0, sigma),
                market["strike_type"],
                market.get("floor_strike"),
                market.get("cap_strike"),
            )
            edge = abs(prob_yes - 0.5)
            if edge < args.min_edge:
                continue

            side = "YES" if prob_yes >= 0.5 else "NO"
            contracts = args.contracts
            if side == "YES":
                entry_cost = effective_yes_entry_cents(bid, ask) * contracts
                payout_per_contract = 100 if outcome == "YES" else 0
            else:
                entry_cost = effective_no_entry_cents(bid, ask) * contracts
                payout_per_contract = 100 if outcome == "NO" else 0

            pnl = contracts * payout_per_contract - entry_cost
            equity += pnl

            trades.append(
                {
                    "date": event_date.isoformat(),
                    "ticker": ticker,
                    "side": side,
                    "prob_yes": prob_yes,
                    "edge": edge,
                    "bid": bid,
                    "ask": ask,
                    "pnl_cents": pnl,
                }
            )

        equity_curve.append({"date": event_date, "equity_cents": equity})

    if not trades:
        logger.error("No trades generated for baseline run")
        return 1

    total_pnl = sum(t["pnl_cents"] for t in trades)
    avg_edge = float(np.mean([t["edge"] for t in trades]))
    wins = sum(1 for t in trades if t["pnl_cents"] > 0)
    win_rate = wins / len(trades)

    equity_df = pd.DataFrame(equity_curve).sort_values("date")
    equity_df["equity_dollars"] = equity_df["equity_cents"] / 100.0
    equity_df["returns"] = equity_df["equity_dollars"].pct_change().fillna(0.0)
    sharpe = 0.0
    if equity_df["returns"].std() > 0:
        sharpe = (equity_df["returns"].mean() / equity_df["returns"].std()) * np.sqrt(252)

    print("\n=== Tmax Daily Baseline Summary ===")
    print(f"City: {args.city}")
    print(f"Bracket: {args.bracket}")
    print(f"Trades: {len(trades)}")
    print(f"Total P&L: ${total_pnl / 100:,.2f}")
    print(f"Win rate: {win_rate:.1%}")
    print(f"Avg |p-0.5| edge: {avg_edge:.3f}")
    print(f"Sharpe (daily returns): {sharpe:.2f}")

    if args.output_json:
        summary = {
            "trades": len(trades),
            "total_pnl_cents": total_pnl,
            "win_rate": win_rate,
            "avg_edge": avg_edge,
            "sharpe": sharpe,
        }
        Path(args.output_json).write_text(json.dumps(summary, indent=2))
        print(f"Saved summary JSON → {args.output_json}")

    trades_path = Path("results") / f"tmax_daily_trades_{args.city}_{args.bracket}_{args.start_date}_{args.end_date}.csv"
    Path(trades_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(trades_path, index=False)
    print(f"Saved trade log → {trades_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
