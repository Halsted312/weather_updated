#!/usr/bin/env python3
"""
Peak temperature probability-vs-price strategy.

At each minute, compare model-estimated Tmax bracket probabilities (p_wx) to
market-implied probabilities (mid prices). Enter only when the edge exceeds a
threshold; maker-first fill model; hold for a fixed horizon.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.fees import net_payout_cents, total_trade_cost_cents
from db.connection import engine

LOGGER = logging.getLogger("peak_temp_strategy")


def configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


@dataclass
class StrategyConfig:
    epsilon: float = 0.03  # additional safety margin in probability units
    mkt_prob_col: str = "mid_prob"  # mid_prob|close_c|p_mkt
    hazard_min: float = 0.0
    hazard_mode: str = "any"  # any|high|low
    hazard_high_min: float = 0.0
    hazard_low_max: float = 1.0
    tod_start: int = 0
    tod_end: int = 23
    hold_minutes: int = 30
    order_size: int = 1
    maker_slippage_cents: int = 0
    taker_slippage_cents: int = 1
    allow_taker: bool = False
    taker_threshold_cents: int = 5  # only take if edge implies this many cents of cushion
    exit_edge_epsilon: float = 0.01  # optional early-exit trigger if edge collapses


def parse_args() -> argparse.Namespace:
    configure_logging()
    parser = argparse.ArgumentParser(description="Peak temperature probability-vs-price backtest")
    parser.add_argument("--city", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--epsilon", type=float, default=0.03, help="Min net edge (prob units) to trade")
    parser.add_argument(
        "--mkt-prob-col",
        type=str,
        default="mid_prob",
        choices=["mid_prob", "close_c", "p_mkt"],
        help="Column to use for market-implied probability",
    )
    parser.add_argument("--hazard-min", type=float, default=0.0)
    parser.add_argument("--hazard-mode", type=str, default="any", choices=["any", "high", "low"])
    parser.add_argument("--hazard-high-min", type=float, default=0.3, help="Min hazard for 'high' mode")
    parser.add_argument("--hazard-low-max", type=float, default=0.2, help="Max hazard for 'low' mode")
    parser.add_argument("--tod-start", type=int, default=0)
    parser.add_argument("--tod-end", type=int, default=23)
    parser.add_argument("--hold-minutes", type=int, default=30)
    parser.add_argument("--order-size", type=int, default=1)
    parser.add_argument("--maker-slippage-cents", type=int, default=0)
    parser.add_argument("--taker-slippage-cents", type=int, default=1)
    parser.add_argument("--allow-taker", action="store_true")
    parser.add_argument("--taker-threshold-cents", type=int, default=5)
    parser.add_argument(
        "--exit-edge-epsilon", type=float, default=0.01, help="Early-exit trigger when |edge| falls below this"
    )
    return parser.parse_args()


def load_panel(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load per-minute brackets with p_wx and candle data.
    """
    query = text(
        """
        SELECT
            f.city,
            f.series_ticker,
            f.event_ticker,
            f.market_ticker,
            f.ts_utc,
            f.ts_local,
            f.local_date,
            f.p_wx,
            f.p_fused_norm,
            f.p_mkt,
            f.mid_prob,
            f.hazard_next_5m,
            f.hazard_next_60m,
            b.close_c,
            b.high_c,
            b.low_c
        FROM feat.minute_panel_full f
        JOIN feat.minute_panel_base b
          ON b.market_ticker = f.market_ticker
         AND b.ts_utc = f.ts_utc
        WHERE f.city = :city
          AND f.local_date BETWEEN :start AND :end
        ORDER BY f.market_ticker, f.ts_utc
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"city": city, "start": start_date, "end": end_date})
    if df.empty:
        raise ValueError(f"No rows for {city} between {start_date} and {end_date}")
    return df


def prepare_panel(df: pd.DataFrame, hold_minutes: int) -> pd.DataFrame:
    df = df.sort_values(["market_ticker", "ts_utc"]).copy()
    grouped = df.groupby("market_ticker", sort=False)
    df["next_high_c"] = grouped["high_c"].shift(-1)
    df["next_low_c"] = grouped["low_c"].shift(-1)
    df["exit_close_c"] = grouped["close_c"].shift(-hold_minutes)
    df["exit_ts_utc"] = grouped["ts_utc"].shift(-hold_minutes)
    # Spacing check (assumes 60s cadence); warn if off
    deltas = grouped["ts_utc"].diff().dropna().dt.total_seconds()
    if not deltas.empty:
        med = deltas.median()
        mx = deltas.max()
        if abs(med - 60.0) > 10 or mx > 90:
            LOGGER.warning("Time spacing irregular: median=%.1fs max=%.1fs", med, mx)
    return df


def _maker_filled(row: pd.Series, price_cents: int) -> bool:
    return (
        pd.notna(row["next_low_c"])
        and pd.notna(row["next_high_c"])
        and row["next_low_c"] <= price_cents <= row["next_high_c"]
    )


def short_round_trip(
    contracts: int,
    entry_price_cents: int,
    exit_price_cents: int,
    entry_fee_type: str,
    exit_fee_type: str,
) -> int:
    sell_proceeds = total_trade_cost_cents(contracts, entry_price_cents, "sell", entry_fee_type)
    buy_cost = total_trade_cost_cents(contracts, exit_price_cents, "buy", exit_fee_type)
    return sell_proceeds - buy_cost


@dataclass
class TradeResult:
    ts_utc: pd.Timestamp
    exit_ts_utc: pd.Timestamp
    event_ticker: str
    market_ticker: str
    side: str
    pnl_cents: int
    entry_type: str
    exit_type: str
    edge_at_entry: float
    edge_at_exit: Optional[float]


def simulate_trades(df: pd.DataFrame, cfg: StrategyConfig) -> List[TradeResult]:
    trades: List[TradeResult] = []
    lookup = df.set_index(["ts_utc", "market_ticker"]).sort_index()
    open_until: Dict[str, pd.Timestamp] = {}
    for (ts, event), g in df.groupby(["ts_utc", "event_ticker"]):
        for _, row in g.iterrows():
            p_model = row.get("p_wx", np.nan)
            if pd.isna(p_model):
                p_model = row.get("p_fused_norm", np.nan)
            if pd.isna(p_model):
                p_model = row.get("mid_prob", np.nan)
            if pd.isna(p_model):
                continue
            mt = row["market_ticker"]
            if mt in open_until and pd.notna(open_until[mt]) and ts < open_until[mt]:
                continue
            # Market-implied probability choice
            if cfg.mkt_prob_col == "close_c":
                q = row["close_c"] / 100.0
            else:
                q = row.get(cfg.mkt_prob_col, np.nan)
                if pd.isna(q):
                    q = row["close_c"] / 100.0
            edge = p_model - q
            hazard = row.get("hazard_next_60m", row.get("hazard_next_5m", 0.0))
            hour_local = pd.to_datetime(row["ts_local"]).hour
            if hazard < cfg.hazard_min or not (cfg.tod_start <= hour_local <= cfg.tod_end):
                continue
            if cfg.hazard_mode == "high" and hazard < cfg.hazard_high_min:
                continue
            if cfg.hazard_mode == "low" and hazard > cfg.hazard_low_max:
                continue

            # Approximate round-trip cost in prob units
            entry_guess = row["close_c"] + cfg.maker_slippage_cents
            exit_guess_val = row.get("exit_close_c", row["close_c"])
            if pd.isna(exit_guess_val):
                exit_guess_val = row["close_c"]
            exit_guess = exit_guess_val + cfg.taker_slippage_cents
            cost_cents = (
                total_trade_cost_cents(cfg.order_size, entry_guess, "buy", "maker")
                + total_trade_cost_cents(cfg.order_size, exit_guess, "sell", "taker")
            )
            cost_prob = cost_cents / 100.0

            # Entry decision with cost margin
            side = None
            if edge - (cost_prob + cfg.epsilon) >= 0:
                side = "buy_yes"
            elif edge + (cost_prob + cfg.epsilon) <= 0:
                side = "sell_yes"
            else:
                continue

            key = (ts, row["market_ticker"])
            cur = lookup.loc[key]
            if pd.isna(cur.get("exit_close_c")) or pd.isna(cur.get("exit_ts_utc")):
                continue

            entry_price = int(round(cur["close_c"])) + cfg.maker_slippage_cents
            exit_price = int(round(cur["exit_close_c"])) + cfg.taker_slippage_cents

            entry_type = "maker" if _maker_filled(cur, entry_price) else None
            if entry_type is None and cfg.allow_taker:
                # Require extra cushion to take
                edge_cents = edge * 100.0
                if abs(edge_cents) >= cfg.taker_threshold_cents:
                    entry_type = "taker"
                    entry_price = entry_price + cfg.taker_slippage_cents
            if entry_type is None:
                continue

            # Optional early exit based on edge collapse at exit time if available
            try:
                exit_row = lookup.loc[(cur["exit_ts_utc"], row["market_ticker"])]
            except KeyError:
                exit_row = None
            edge_exit = None
            if isinstance(exit_row, pd.Series):
                p_exit = exit_row.get("p_wx", np.nan)
                if pd.isna(p_exit):
                    p_exit = exit_row.get("p_fused_norm", np.nan)
                if pd.isna(p_exit):
                    p_exit = exit_row.get("mid_prob", np.nan)
                if pd.notna(p_exit):
                    edge_exit = float(p_exit - exit_row["close_c"] / 100.0)
                    if abs(edge_exit) < cfg.exit_edge_epsilon:
                        exit_price = int(round(exit_row["close_c"])) + cfg.taker_slippage_cents

            if side == "buy_yes":
                pnl = net_payout_cents(cfg.order_size, entry_price, exit_price, entry_type, "taker")
            else:
                pnl = short_round_trip(cfg.order_size, entry_price, exit_price, entry_type, "taker")

            trades.append(
                TradeResult(
                    ts_utc=ts,
                    exit_ts_utc=cur["exit_ts_utc"],
                    event_ticker=event,
                    market_ticker=row["market_ticker"],
                    side=side,
                    pnl_cents=int(pnl),
                    entry_type=entry_type,
                    exit_type="taker",
                    edge_at_entry=float(edge),
                    edge_at_exit=edge_exit,
                )
            )
            open_until[mt] = cur["exit_ts_utc"]
    return trades


def summarize(trades: Sequence[TradeResult]) -> Dict[str, float]:
    ordered = sorted(trades, key=lambda t: t.exit_ts_utc)
    pnls = [t.pnl_cents for t in ordered]
    total = sum(pnls)
    sharpe = 0.0
    max_dd = 0
    if pnls:
        arr = np.array(pnls, dtype=float)
        mean = arr.mean()
        std = arr.std(ddof=0)
        sharpe = float(mean / std) if std > 0 else 0.0
        equity = np.cumsum(arr)
        peaks = np.maximum.accumulate(equity)
        dd = peaks - equity
        max_dd = int(dd.max()) if len(dd) else 0
    return {
        "trades": len(pnls),
        "pnl_cents": total,
        "pnl_dollars": total / 100.0,
        "sharpe": sharpe,
        "max_drawdown_cents": max_dd,
        "max_drawdown_dollars": max_dd / 100.0,
    }


def run_backtest(args: argparse.Namespace) -> Tuple[List[TradeResult], Dict[str, float]]:
    cfg = StrategyConfig(
        epsilon=args.epsilon,
        mkt_prob_col=args.mkt_prob_col,
        hazard_min=args.hazard_min,
        hazard_mode=args.hazard_mode,
        hazard_high_min=args.hazard_high_min,
        hazard_low_max=args.hazard_low_max,
        tod_start=args.tod_start,
        tod_end=args.tod_end,
        hold_minutes=args.hold_minutes,
        order_size=args.order_size,
        maker_slippage_cents=args.maker_slippage_cents,
        taker_slippage_cents=args.taker_slippage_cents,
        allow_taker=args.allow_taker,
        taker_threshold_cents=args.taker_threshold_cents,
        exit_edge_epsilon=args.exit_edge_epsilon,
    )
    df = load_panel(args.city, args.start_date, args.end_date)
    df = prepare_panel(df, cfg.hold_minutes)
    trades = simulate_trades(df, cfg)
    summary = summarize(trades)
    LOGGER.info(
        "Backtest %s %s→%s | trades=%d pnl=$%.2f sharpe=%.3f max_dd=$%.2f",
        args.city,
        args.start_date,
        args.end_date,
        summary["trades"],
        summary["pnl_dollars"],
        summary["sharpe"],
        summary["max_drawdown_dollars"],
    )
    return trades, summary


def main() -> None:
    args = parse_args()
    trades, summary = run_backtest(args)
    LOGGER.info("First 10 trades:")
    for t in trades[:10]:
        LOGGER.info(
            "%s %s %s side=%s pnl=%dc entry=%s exit=%s",
            t.ts_utc,
            t.event_ticker,
            t.market_ticker,
            t.side,
            t.pnl_cents,
            t.entry_type,
            t.exit_type,
        )


if __name__ == "__main__":
    main()
