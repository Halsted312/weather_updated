#!/usr/bin/env python3
"""
Evaluate triad backtests over rolling windows to see stability across time.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import pathlib
import sys
from typing import List, Tuple

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_triad import run_backtest

LOGGER = logging.getLogger("eval_triad_windows")


def configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


def parse_args() -> argparse.Namespace:
    configure_logging()
    parser = argparse.ArgumentParser(description="Evaluate triad backtests over rolling windows")
    parser.add_argument("--city", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--stride-days", type=int, default=7, help="Step between windows; use <window-days for overlap")
    parser.add_argument("--hold-minutes", type=int, default=5)
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
    parser.add_argument("--allow-taker", action="store_true")
    parser.add_argument("--taker-threshold-cents", type=int, default=0)
    parser.add_argument("--order-size", type=int, default=1)
    parser.add_argument("--hedge-multiplier", type=float, default=0.5)
    parser.add_argument("--maker-slippage-cents", type=int, default=0)
    parser.add_argument("--taker-slippage-cents", type=int, default=1)
    parser.add_argument("--implied-prob-col", type=str, default=None)
    parser.add_argument("--calibrated-prob-col", type=str, default="p_up_calibrated_5m")
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


def date_range_windows(start: dt.date, end: dt.date, window_days: int, stride_days: int) -> List[Tuple[str, str]]:
    windows: List[Tuple[str, str]] = []
    cur = start
    while cur <= end:
        w_end = cur + dt.timedelta(days=window_days - 1)
        if w_end > end:
            break
        windows.append((cur.isoformat(), w_end.isoformat()))
        cur = cur + dt.timedelta(days=stride_days)
    return windows


def main() -> None:
    args = parse_args()
    start = dt.date.fromisoformat(args.start_date)
    end = dt.date.fromisoformat(args.end_date)
    wins = date_range_windows(start, end, args.window_days, args.stride_days)
    if not wins:
        LOGGER.error("No windows generated for %s→%s", args.start_date, args.end_date)
        sys.exit(1)

    records = []
    for w_start, w_end in wins:
        LOGGER.info("Window %s→%s", w_start, w_end)
        ns = argparse.Namespace(
            city=args.city,
            start_date=w_start,
            end_date=w_end,
            min_volume=args.min_volume,
            min_score=args.min_score,
            max_spread=args.max_spread,
            hold_minutes=args.hold_minutes,
            order_size=args.order_size,
            hedge_multiplier=args.hedge_multiplier,
            maker_slippage_cents=args.maker_slippage_cents,
            taker_slippage_cents=args.taker_slippage_cents,
            allow_taker=args.allow_taker,
            taker_threshold_cents=args.taker_threshold_cents,
            implied_prob_col=args.implied_prob_col,
            calibrated_prob_col=args.calibrated_prob_col,
            alpha_ras=args.alpha_ras,
            alpha_accel=args.alpha_accel,
            alpha_volume=args.alpha_volume,
            alpha_hazard=args.alpha_hazard,
            alpha_misprice=args.alpha_misprice,
            hazard_min=args.hazard_min,
            triad_mass_min=args.triad_mass_min,
            edge_wx_min=args.edge_wx_min,
            tod_start=args.tod_start,
            tod_end=args.tod_end,
        )
        _, summary = run_backtest(ns)
        records.append(
            {
                "window_start": w_start,
                "window_end": w_end,
                "trades": summary["trades"],
                "pnl_dollars": summary["pnl_dollars"],
                "sharpe": summary["sharpe"],
                "max_drawdown_dollars": summary["max_drawdown_dollars"],
            }
        )

    df = pd.DataFrame.from_records(records)
    LOGGER.info(
        "Windows=%d pnl_mean=$%.2f pnl_median=$%.2f sharpe_mean=%.3f profitable_frac=%.2f",
        len(df),
        df["pnl_dollars"].mean(),
        df["pnl_dollars"].median(),
        df["sharpe"].mean(),
        (df["pnl_dollars"] > 0).mean(),
    )
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        LOGGER.info("Wrote %s", args.output_csv)
    else:
        LOGGER.info("\n%s", df)


if __name__ == "__main__":
    main()
