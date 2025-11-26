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
import json
import math
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import bindparam, text

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
    add_alpha_args(run_day)

    backfill = sub.add_parser("backfill", help="Backfill fused PMF for a range")
    backfill.add_argument("--city", action="append", required=True)
    backfill.add_argument("--start-date", required=True)
    backfill.add_argument("--end-date", required=True)
    add_alpha_args(backfill)

    metrics = sub.add_parser("metrics", help="Evaluate fusion weights on a window")
    metrics.add_argument("--city", required=True)
    metrics.add_argument("--start-date", required=True)
    metrics.add_argument("--end-date", required=True)
    metrics.add_argument("--apply", action="store_true", help="Persist fused probabilities with this alpha")
    add_alpha_args(metrics)

    return parser.parse_args()


DEFAULT_ALPHA: Dict[str, float] = {
    "bias": 0.0,
    "log_volume": 0.15,
    "hazard": 2.0,
    "volatility": 0.0,
    "min_weight": 0.05,
    "max_weight": 0.95,
}


def add_alpha_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--alpha-config", type=str, help="YAML file with alpha parameters")
    parser.add_argument("--alpha-bias", type=float)
    parser.add_argument("--alpha-log-volume", type=float)
    parser.add_argument("--alpha-hazard", type=float)
    parser.add_argument("--alpha-volatility", type=float)
    parser.add_argument("--alpha-min-weight", type=float)
    parser.add_argument("--alpha-max-weight", type=float)


def load_alpha(args: argparse.Namespace) -> Dict[str, float]:
    alpha = DEFAULT_ALPHA.copy()
    if getattr(args, "alpha_config", None):
        with open(args.alpha_config, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        alpha.update({k.replace("-", "_"): v for k, v in data.items() if v is not None})
    mapping = {
        "alpha_bias": "bias",
        "alpha_log_volume": "log_volume",
        "alpha_hazard": "hazard",
        "alpha_volatility": "volatility",
        "alpha_min_weight": "min_weight",
        "alpha_max_weight": "max_weight",
    }
    for arg_name, key in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            alpha[key] = value
    alpha["min_weight"] = max(0.0, min(1.0, alpha["min_weight"]))
    alpha["max_weight"] = max(alpha["min_weight"], min(1.0, alpha["max_weight"]))
    return alpha


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


def compute_fusion_weights(row: pd.Series, alpha: Dict[str, float]) -> tuple[float, float]:
    hazard = float(row.get("hazard_next_60m") or 0.0)
    volume = float(row.get("volume") or 0.0)
    log_volume = math.log1p(max(volume, 0.0))
    volatility = abs(float(row.get("mid_velocity") or 0.0))
    score = (
        alpha["bias"]
        + alpha["hazard"] * hazard
        + alpha["log_volume"] * log_volume
        + alpha["volatility"] * volatility
    )
    w_mkt = 1.0 / (1.0 + math.exp(-score))
    w_mkt = max(alpha["min_weight"], min(alpha["max_weight"], w_mkt))
    return w_mkt, 1.0 - w_mkt


def compute_fused_probabilities(panel: pd.DataFrame, alpha: Dict[str, float]) -> pd.DataFrame:
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
    df["sum_wx"] = df.groupby(["ts_utc", "event_ticker"])["p_wx_clamped"].transform("sum")
    df["p_wx_norm"] = df.apply(
        lambda row: (row["p_wx_clamped"] / row["sum_wx"]) if row["sum_wx"] > 1e-6 else 1.0 / row["num_brackets"],
        axis=1,
    )

    weights = df.apply(lambda row: compute_fusion_weights(row, alpha), axis=1, result_type="expand")
    df["weight_market"] = weights[0]
    df["weight_weather"] = weights[1]

    df["p_fused_raw"] = fuse_probs(df["p_mkt"], df["p_wx_clamped"], df["weight_market"], df["weight_weather"])

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
    with engine.begin() as conn:
        conn.execute(update_sql, records)


def run_day(city: str, date_str: str, alpha: Dict[str, float], persist: bool = True, collect_metrics: bool = False):
    local_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    LOGGER.info("PMF fusion for %s on %s", city, date_str)
    panel = fetch_panel_full(city, local_date)
    fused = compute_fused_probabilities(panel, alpha)
    metrics = evaluate_fusion_metrics(fused)
    if persist:
        persist_fused_rows(fused)
        LOGGER.info("Updated %d rows", len(fused))
    if collect_metrics and metrics:
        LOGGER.info(
            "Metrics %s %s | rows=%d brier_fused=%.4f logloss_fused=%.4f entropy=%.4f",
            city,
            date_str,
            metrics["rows"],
            metrics["brier_fused"],
            metrics["logloss_fused"],
            metrics["entropy"],
        )
    return metrics


def backfill(cities: Sequence[str], start_date: str, end_date: str, alpha: Dict[str, float]) -> None:
    start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    current = start
    while current <= end:
        for city in cities:
            try:
                run_day(city, current.isoformat(), alpha, persist=True, collect_metrics=False)
            except Exception as exc:  # pylint:disable=broad-except
                LOGGER.warning("Failed fusion for %s %s: %s", city, current.isoformat(), exc)
        current += dt.timedelta(days=1)


def load_market_outcomes(market_tickers: Sequence[str]) -> pd.DataFrame:
    if len(market_tickers) == 0:
        return pd.DataFrame(columns=["market_ticker", "is_winner"])
    query = (
        text(
            """
        SELECT ticker, settlement_value
        FROM markets
        WHERE ticker IN :tickers
        """
        ).bindparams(bindparam("tickers", expanding=True))
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"tickers": tuple(market_tickers)}).fetchall()
    if not rows:
        return pd.DataFrame(columns=["market_ticker", "is_winner"])
    records = []
    for ticker, settlement_value in rows:
        if settlement_value is None:
            continue
        records.append(
            {
                "market_ticker": ticker,
                "is_winner": 1 if settlement_value >= 50 else 0,
            }
        )
    return pd.DataFrame.from_records(records)


def evaluate_fusion_metrics(fused: pd.DataFrame) -> dict | None:
    outcomes = load_market_outcomes(fused["market_ticker"].unique())
    if outcomes.empty:
        return None
    df = fused.merge(outcomes, on="market_ticker", how="left")
    df = df.dropna(subset=["is_winner"])
    if df.empty:
        return None
    df["is_winner"] = df["is_winner"].astype(int)
    brier_fused = float(np.mean((df["p_fused_norm"] - df["is_winner"]) ** 2))
    brier_mkt = float(np.mean((df["p_mkt"] - df["is_winner"]) ** 2))
    brier_wx = float(np.mean((df["p_wx_norm"] - df["is_winner"]) ** 2))

    winner_df = df[df["is_winner"] == 1]
    if winner_df.empty:
        return None
    eps = 1e-9
    logloss_fused = float(-np.mean(np.log(np.clip(winner_df["p_fused_norm"], eps, 1.0))))
    logloss_mkt = float(-np.mean(np.log(np.clip(winner_df["p_mkt"], eps, 1.0))))
    logloss_wx = float(-np.mean(np.log(np.clip(winner_df["p_wx_norm"], eps, 1.0))))

    entropy_groups = df.groupby(["ts_utc", "event_ticker"])["p_fused_norm"].apply(
        lambda probs: float(-(probs * np.log(np.clip(probs, eps, 1.0))).sum())
    )
    entropy = float(entropy_groups.mean()) if not entropy_groups.empty else float("nan")

    return {
        "rows": len(df),
        "winners": len(winner_df),
        "brier_fused": brier_fused,
        "brier_mkt": brier_mkt,
        "brier_wx": brier_wx,
        "logloss_fused": logloss_fused,
        "logloss_mkt": logloss_mkt,
        "logloss_wx": logloss_wx,
        "entropy": entropy,
    }


def summarize_metrics(metrics_list: List[dict]) -> dict | None:
    if not metrics_list:
        return None
    total_rows = sum(m["rows"] for m in metrics_list)
    total_winners = sum(m["winners"] for m in metrics_list)
    entropy_count = len(metrics_list)

    def weighted_avg(key: str, weight: str) -> float:
        if weight == "rows":
            denom = total_rows
        elif weight == "winners":
            denom = total_winners
        else:
            denom = entropy_count
        if denom == 0:
            return float("nan")
        return float(sum(m[key] * (m[weight] if weight != "entropy" else 1.0) for m in metrics_list) / denom)

    return {
        "rows": total_rows,
        "brier_fused": weighted_avg("brier_fused", "rows"),
        "brier_mkt": weighted_avg("brier_mkt", "rows"),
        "brier_wx": weighted_avg("brier_wx", "rows"),
        "logloss_fused": weighted_avg("logloss_fused", "winners"),
        "logloss_mkt": weighted_avg("logloss_mkt", "winners"),
        "logloss_wx": weighted_avg("logloss_wx", "winners"),
        "entropy": weighted_avg("entropy", "entropy"),
    }


def run_metrics(city: str, start_date: str, end_date: str, alpha: Dict[str, float], apply: bool) -> None:
    start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    metrics_list: List[dict] = []
    current = start
    while current <= end:
        try:
            metrics = run_day(city, current.isoformat(), alpha, persist=apply, collect_metrics=True)
            if metrics:
                metrics_list.append(metrics)
        except Exception as exc:  # pylint:disable=broad-except
            LOGGER.warning("Failed fusion metrics for %s %s: %s", city, current.isoformat(), exc)
        current += dt.timedelta(days=1)
    summary = summarize_metrics(metrics_list)
    if summary:
        LOGGER.info(
            "Fusion summary %s %s→%s | rows=%d brier_fused=%.4f (mkt=%.4f wx=%.4f) logloss_fused=%.4f (mkt=%.4f wx=%.4f) entropy=%.4f",
            city,
            start_date,
            end_date,
            summary["rows"],
            summary["brier_fused"],
            summary["brier_mkt"],
            summary["brier_wx"],
            summary["logloss_fused"],
            summary["logloss_mkt"],
            summary["logloss_wx"],
            summary["entropy"],
        )
        print(json.dumps(summary, indent=2))
    else:
        LOGGER.info("No metrics computed for %s %s→%s", city, start_date, end_date)


def main() -> None:
    args = parse_args()
    alpha = load_alpha(args)
    if args.cmd == "run-day":
        run_day(args.city, args.date, alpha, persist=True, collect_metrics=True)
    elif args.cmd == "backfill":
        backfill(args.city, args.start_date, args.end_date, alpha)
    elif args.cmd == "metrics":
        run_metrics(args.city, args.start_date, args.end_date, alpha, apply=args.apply)
    else:
        raise ValueError(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    main()
