#!/usr/bin/env python3
"""
Fee-aware backtest for the triad momentum signal.

This reuses the triad scoring logic, fills synthetic triad positions with a
maker-first rule, and reports P&L over a window.
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

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.fees import net_payout_cents, total_trade_cost_cents
from scripts.triad_momentum import TriadConfig, compute_scores, configure_logging, load_triad_panel, select_triads

LOGGER = logging.getLogger("backtest_triad")


@dataclass
class BacktestConfig:
    hold_minutes: int = 5
    order_size: int = 1
    hedge_multiplier: float = 0.5
    maker_slippage_cents: int = 0
    taker_slippage_cents: int = 1
    allow_taker: bool = False
    taker_threshold_cents: int = 0  # require min EV before using taker; 0 disables EV gate
    exit_fee_type: str = "taker"  # "maker" or "taker"
    implied_prob_col: Optional[str] = None  # optional column on panel rows
    calibrated_prob_col: Optional[str] = None  # optional column on panel rows


@dataclass
class TradeResult:
    ts_utc: pd.Timestamp
    exit_ts_utc: pd.Timestamp
    event_ticker: str
    market_center: str
    market_left: str
    market_right: str
    pnl_cents: int
    entry_types: Dict[str, str]
    exit_type: str
    filled: bool


def parse_args() -> argparse.Namespace:
    configure_logging()
    parser = argparse.ArgumentParser(description="Backtest triad momentum trades")
    parser.add_argument("--city", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)

    parser.add_argument("--min-volume", type=float, default=5.0)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--max-spread", type=float, default=5.0)
    parser.add_argument("--hazard-min", type=float, default=0.0)
    parser.add_argument("--triad-mass-min", type=float, default=0.0)
    parser.add_argument("--tod-start", type=int, default=0)
    parser.add_argument("--tod-end", type=int, default=23)
    parser.add_argument("--alpha-ras", type=float, default=1.0)
    parser.add_argument("--alpha-accel", type=float, default=0.5)
    parser.add_argument("--alpha-volume", type=float, default=0.2)
    parser.add_argument("--alpha-hazard", type=float, default=0.2)
    parser.add_argument("--alpha-misprice", type=float, default=0.0)
    parser.add_argument("--edge-wx-min", type=float, default=0.0)

    parser.add_argument("--hold-minutes", type=int, default=5, help="Minutes to hold the triad before exit")
    parser.add_argument("--order-size", type=int, default=1, help="Contracts on the center leg")
    parser.add_argument("--hedge-multiplier", type=float, default=0.5, help="Hedge size fraction vs. center (0 disables)")
    parser.add_argument("--maker-slippage-cents", type=int, default=0)
    parser.add_argument("--taker-slippage-cents", type=int, default=1)
    parser.add_argument("--allow-taker", action="store_true", help="Fallback to taker fills when maker doesn’t touch")
    parser.add_argument(
        "--taker-threshold-cents",
        type=int,
        default=0,
        help="Minimum expected edge (cents) required to use taker; 0 disables EV gating",
    )
    parser.add_argument(
        "--implied-prob-col",
        type=str,
        default=None,
        help="Column name for implied prob (default uses close_c/100 if absent)",
    )
    parser.add_argument(
        "--calibrated-prob-col",
        type=str,
        default="p_up_calibrated_5m",
        help="Column name for calibrated model prob; taker EV gating requires this when threshold>0",
    )

    return parser.parse_args()


def clamp_price(price_cents: int) -> int:
    return int(max(0, min(100, price_cents)))


def prepare_panel(df: pd.DataFrame, hold_minutes: int) -> pd.DataFrame:
    """Add forward-looking columns used by the fill model."""
    df = df.sort_values(["market_ticker", "ts_utc"]).copy()
    grouped = df.groupby("market_ticker", sort=False)
    df["next_high_c"] = grouped["high_c"].shift(-1)
    df["next_low_c"] = grouped["low_c"].shift(-1)
    df["exit_close_c"] = grouped["close_c"].shift(-hold_minutes)
    df["exit_ts_utc"] = grouped["ts_utc"].shift(-hold_minutes)
    return df


def short_round_trip(
    contracts: int,
    entry_price_cents: int,
    exit_price_cents: int,
    entry_fee_type: str,
    exit_fee_type: str,
) -> int:
    """Sell YES then buy back later."""
    sell_proceeds = total_trade_cost_cents(contracts, entry_price_cents, "sell", entry_fee_type)
    buy_cost = total_trade_cost_cents(contracts, exit_price_cents, "buy", exit_fee_type)
    return sell_proceeds - buy_cost


def _maker_filled(row: pd.Series, price_cents: int) -> bool:
    # Ensure we return a plain Python bool (not a pandas Series[bool]) by
    # extracting scalar values and handling missing data explicitly.
    if pd.isna(row["next_low_c"]) or pd.isna(row["next_high_c"]):
        return False
    low = float(row["next_low_c"])
    high = float(row["next_high_c"])
    return low <= price_cents <= high


def _edge_cents(row: pd.Series, cfg: BacktestConfig) -> Optional[float]:
    if cfg.taker_threshold_cents <= 0:
        return None
    implied_val = row.get(cfg.implied_prob_col) if cfg.implied_prob_col else None
    calibrated = row.get(cfg.calibrated_prob_col) if cfg.calibrated_prob_col else None

    # Normalize implied to a probability; if it looks like cents (>1), divide by 100.
    if implied_val is None or pd.isna(implied_val):
        implied = row.get("close_c", np.nan) / 100.0
    else:
        implied = float(implied_val)
        if implied > 1.0:
            implied = implied / 100.0

    if calibrated is None or pd.isna(calibrated):
        return None

    calibrated = float(calibrated)
    if calibrated > 1.0:
        calibrated = calibrated / 100.0

    # Simple expected edge in cents (center leg only). Fees are approximated by slippage here;
    # fee math is handled in the trade legs themselves.
    return (calibrated - implied) * 100.0 - cfg.taker_slippage_cents


def simulate_triad(
    intent: Dict[str, object],
    panel_lookup: pd.DataFrame,
    cfg: BacktestConfig,
) -> Optional[TradeResult]:
    ts = intent["ts_utc"]
    event = intent["event_ticker"]
    markets = {
        "center": intent["market_center"],
        "left": intent["market_left"],
        "right": intent["market_right"],
    }

    try:
        rows = {leg: panel_lookup.loc[(ts, m)] for leg, m in markets.items()}
    except KeyError:
        return None

    if any(pd.isna(rows[leg].get("exit_close_c")) for leg in rows):
        return None
    exit_ts = rows["center"].get("exit_ts_utc")
    if pd.isna(exit_ts):
        return None

    hedge_contracts = int(np.ceil(cfg.order_size * cfg.hedge_multiplier)) if cfg.hedge_multiplier > 0 else 0
    entry_types: Dict[str, str] = {}
    pnl = 0

    for leg, row in rows.items():
        is_center = leg == "center"
        qty = cfg.order_size if is_center else hedge_contracts
        if qty <= 0 and not is_center:
            continue

        entry_price = clamp_price(int(round(row["close_c"])) + cfg.maker_slippage_cents)
        exit_price = clamp_price(int(round(row["exit_close_c"])) + cfg.taker_slippage_cents)

        if _maker_filled(row, entry_price):
            entry_types[leg] = "maker"
        elif cfg.allow_taker:
            edge_c = _edge_cents(row, cfg) if is_center else None
            if cfg.taker_threshold_cents > 0 and (edge_c is None or edge_c < cfg.taker_threshold_cents):
                return None
            entry_types[leg] = "taker"
            entry_price = clamp_price(entry_price + cfg.taker_slippage_cents)
        else:
            return None

        if is_center:
            pnl += net_payout_cents(qty, entry_price, exit_price, entry_types[leg], cfg.exit_fee_type)
        else:
            pnl += short_round_trip(qty, entry_price, exit_price, entry_types[leg], cfg.exit_fee_type)

    return TradeResult(
        ts_utc=ts,
        exit_ts_utc=exit_ts,
        event_ticker=event,
        market_center=markets["center"],
        market_left=markets["left"],
        market_right=markets["right"],
        pnl_cents=int(pnl),
        entry_types=entry_types,
        exit_type=cfg.exit_fee_type,
        filled=True,
    )


def summarize(trades: Sequence[TradeResult]) -> Dict[str, float]:
    ordered = sorted(trades, key=lambda t: t.exit_ts_utc)
    pnls = [t.pnl_cents for t in ordered]
    total_pnl = sum(pnls)
    count = len(trades)
    sharpe = 0.0
    max_dd = 0
    if pnls:
        arr = np.array(pnls, dtype=float)
        mean = arr.mean()
        std = arr.std(ddof=0)
        sharpe = float(mean / std) if std > 0 else 0.0
        # max drawdown on cumulative equity in trade order
        equity = np.cumsum(arr)
        peaks = np.maximum.accumulate(equity)
        dd = peaks - equity
        max_dd = int(dd.max()) if len(dd) else 0
    return {
        "trades": count,
        "pnl_cents": total_pnl,
        "pnl_dollars": total_pnl / 100.0,
        "sharpe": sharpe,
        "max_drawdown_cents": max_dd,
        "max_drawdown_dollars": max_dd / 100.0,
    }


def run_backtest(args: argparse.Namespace, triad_cfg: Optional[TriadConfig] = None) -> Tuple[List[TradeResult], Dict[str, float]]:
    triad_cfg = triad_cfg or TriadConfig(
        min_volume=args.min_volume,
        min_score=args.min_score,
        max_spread_cents=args.max_spread,
        alpha_ras=getattr(args, "alpha_ras", 1.0),
        alpha_accel=getattr(args, "alpha_accel", 0.5),
        alpha_volume=getattr(args, "alpha_volume", 0.2),
        alpha_hazard=getattr(args, "alpha_hazard", 0.2),
        alpha_misprice=getattr(args, "alpha_misprice", 0.0),
        hazard_min=getattr(args, "hazard_min", 0.0),
        triad_mass_min=getattr(args, "triad_mass_min", 0.0),
        tod_start=getattr(args, "tod_start", 0),
        tod_end=getattr(args, "tod_end", 23),
        edge_wx_min=getattr(args, "edge_wx_min", 0.0),
    )
    df = load_triad_panel(args.city, args.start_date, args.end_date)
    df = prepare_panel(df, hold_minutes=args.hold_minutes)
    df = compute_scores(df, triad_cfg)

    intents = select_triads(df, triad_cfg.min_score)
    LOGGER.info("Scored %d triads; %d intents above threshold", len(df), len(intents))

    panel_lookup = df.set_index(["ts_utc", "market_ticker"]).sort_index()
    bt_cfg = BacktestConfig(
        hold_minutes=args.hold_minutes,
        order_size=args.order_size,
        hedge_multiplier=args.hedge_multiplier,
        maker_slippage_cents=args.maker_slippage_cents,
        taker_slippage_cents=args.taker_slippage_cents,
        allow_taker=args.allow_taker,
        taker_threshold_cents=args.taker_threshold_cents,
        implied_prob_col=args.implied_prob_col,
        calibrated_prob_col=args.calibrated_prob_col,
    )

    trades: List[TradeResult] = []
    for intent in intents:
        trade = simulate_triad(intent, panel_lookup, bt_cfg)
        if trade:
            trades.append(trade)

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
            "%s %s center=%s left=%s right=%s pnl=%dc entry=%s exit=%s",
            t.ts_utc,
            t.event_ticker,
            t.market_center,
            t.market_left,
            t.market_right,
            t.pnl_cents,
            t.entry_types,
            t.exit_type,
        )


if __name__ == "__main__":
    main()
