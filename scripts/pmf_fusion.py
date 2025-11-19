#!/usr/bin/env python3
"""
PMF fusion layer: combine market-implied probabilities with weather-implied p_wx.

Usage:
    python scripts/pmf_fusion.py run-day --city chicago --date 2025-11-19
    python scripts/pmf_fusion.py backfill --city chicago --city miami --start-date 2025-11-01 --end-date 2025-11-19
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
from typing import List, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import engine

LOGGER = logging.getLogger("pmf_fusion")


def configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


def parse_args() -> argparse.Namespace:
    configure_logging()
    parser = argparse.ArgumentParser(description="PMF fusion layer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_day = sub.add_parser("run-day", help="Compute fused PMF for one city/day")
    run_day.add_argument("--city", required=True)
    run_day.add_argument("--date", required=True, help="Local date YYYY-MM-DD")

    backfill = sub.add_parser("backfill", help="Backfill fused PMF for a range")
    backfill.add_argument("--city", action="append", required=True)
    backfill.add_argument("--start-date", required=True)
    backfill.add_argument("--end-date", required=True)

    return parser.parse_args()


def fetch_panel_full(city: str, local_date: dt.date) -> pd.DataFrame:
    query = text(
        """
        SELECT *
        FROM feat.minute_panel_full
        WHERE city = :city
          AND local_date = :local_date
        ORDER BY ts_utc
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"city": city, "local_date": local_date})
    if df.empty:
        raise ValueError(f"No rows for {city} {local_date} in feat.minute_panel_full")
    return df


def compute_fused_probabilities(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    # market-implied probability from close price
    df["p_mkt_raw"] = np.clip(df["close_c"] / 100.0, 1e-4, 1 - 1e-4)
    df["sum_raw"] = df.groupby(["ts_utc", "event_ticker"])["p_mkt_raw"].transform("sum")
    df["num_brackets"] = df.groupby(["ts_utc", "event_ticker"])["market_ticker"].transform("count")

    def normalize(row: pd.Series) -> float:
        if row["sum_raw"] <= 1e-6:
            return 1.0 / row["num_brackets"]
        return row["p_mkt_raw"] / row["sum_raw"]

    df["p_mkt"] = df.apply(normalize, axis=1)

    # weather probability
    df["p_wx_clamped"] = np.clip(df["p_wx"].fillna(df["p_mkt"]), 1e-4, 1 - 1e-4)

    # weights: market weight increases with hazard + volume, weather weight when hazard low
    hazard = df["hazard_next_60m"].fillna(0.0).clip(0.0, 1.0)
    volume = df["volume"].fillna(0)
    volume_norm = volume / (volume.groupby(df["ts_utc"]).transform("max").replace(0, np.nan))
    volume_norm = volume_norm.fillna(0.0).clip(0.0, 1.0)

    weight_market = 0.5 + 0.4 * hazard + 0.1 * volume_norm
    weight_market = weight_market.clip(0.05, 0.95)
    weight_weather = 1.0 - weight_market

    df["p_fused_raw"] = fuse_probs(df["p_mkt"], df["p_wx_clamped"], weight_market, weight_weather)

    df["sum_fused"] = df.groupby(["ts_utc", "event_ticker"])["p_fused_raw"].transform("sum")

    def normalize_fused(row: pd.Series) -> float:
        if row["sum_fused"] <= 1e-6:
            return 1.0 / row["num_brackets"]
        return row["p_fused_raw"] / row["sum_fused"]

    df["p_fused_norm"] = df.apply(normalize_fused, axis=1)
    df["p_fused"] = df["p_fused_raw"]
    return df


def fuse_probs(p_mkt: pd.Series, p_wx: pd.Series, w_mkt: pd.Series, w_wx: pd.Series) -> pd.Series:
    eps = 1e-6
    logit = lambda p: np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    inv_logit = lambda z: 1 / (1 + np.exp(-z))
    combined_logit = w_mkt * logit(p_mkt) + w_wx * logit(p_wx)
    fused = inv_logit(combined_logit)
    return fused.clip(0.0, 1.0)


def persist_fused_rows(df: pd.DataFrame) -> None:
    update_sql = text(
        """
        UPDATE pmf.minute
        SET p_mkt = :p_mkt,
            p_fused = :p_fused,
            p_fused_norm = :p_fused_norm
        WHERE market_ticker = :market_ticker
          AND ts_utc = :ts_utc
        """
    )
    records = df[["market_ticker", "ts_utc", "p_mkt", "p_fused", "p_fused_norm"]].to_dict(orient="records")
    with engine.connect() as conn:
        conn.execute(update_sql, records)


def run_day(city: str, date_str: str) -> None:
    local_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    LOGGER.info("PMF fusion for %s on %s", city, date_str)
    panel = fetch_panel_full(city, local_date)
    fused = compute_fused_probabilities(panel)
    persist_fused_rows(fused)
    LOGGER.info("Updated %d rows", len(fused))


def backfill(cities: Sequence[str], start_date: str, end_date: str) -> None:
    start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    current = start
    while current <= end:
        for city in cities:
            try:
                run_day(city, current.isoformat())
            except Exception as exc:  # pylint:disable=broad-except
                LOGGER.warning("Failed fusion for %s %s: %s", city, current.isoformat(), exc)
        current += dt.timedelta(days=1)


def main() -> None:
    args = parse_args()
    if args.cmd == "run-day":
        run_day(args.city, args.date)
    elif args.cmd == "backfill":
        backfill(args.city, args.start_date, args.end_date)
    else:
        raise ValueError(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    main()
