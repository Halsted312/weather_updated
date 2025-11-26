#!/usr/bin/env python3
"""
Triad momentum signal scaffold.

Loads triad-enhanced features from feat.minute_panel_triads, computes
relative-acceleration-based scores, and prints candidate triad trade intents.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import engine

LOGGER = logging.getLogger("triad_momentum")


def configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


@dataclass
class TriadConfig:
    min_volume: float = 1.0
    min_score: float = 0.3
    max_spread_cents: float = 5.0  # placeholder until real spreads exist
    alpha_ras: float = 1.0
    alpha_accel: float = 0.5
    alpha_volume: float = 0.2
    alpha_hazard: float = 0.2
    alpha_misprice: float = 0.0
    hazard_min: float = 0.0
    triad_mass_min: float = 0.0
    tod_start: int = 0  # inclusive, local hour
    tod_end: int = 23   # inclusive, local hour
    edge_wx_min: float = 0.0


def parse_args() -> argparse.Namespace:
    configure_logging()
    parser = argparse.ArgumentParser(description="Triad momentum signal tool")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sig = sub.add_parser("signals", help="Compute triad scores for a window")
    sig.add_argument("--city", required=True)
    sig.add_argument("--start-date", required=True)
    sig.add_argument("--end-date", required=True)
    sig.add_argument("--min-volume", type=float, default=10.0)
    sig.add_argument("--min-score", type=float, default=1.0)
    sig.add_argument("--max-spread", type=float, default=5.0)
    sig.add_argument("--hazard-min", type=float, default=0.0)
    sig.add_argument("--triad-mass-min", type=float, default=0.0)
    sig.add_argument("--tod-start", type=int, default=0)
    sig.add_argument("--tod-end", type=int, default=23)
    sig.add_argument("--edge-wx-min", type=float, default=0.0)

    diag = sub.add_parser("diagnostics", help="Print top triad scores without gates")
    diag.add_argument("--city", required=True)
    diag.add_argument("--start-date", required=True)
    diag.add_argument("--end-date", required=True)
    diag.add_argument("--top-n", type=int, default=10)

    return parser.parse_args()


def load_triad_panel(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    query = text(
        """
        SELECT
            t.*,
            b.open_c,
            b.high_c,
            b.low_c,
            b.num_trades,
            f.p_wx,
            f.p_mkt,
            f.p_fused_norm,
            f.hazard_next_5m,
            f.hazard_next_60m
        FROM feat.minute_panel_triads t
        JOIN feat.minute_panel_base b
          ON b.market_ticker = t.market_ticker
         AND b.ts_utc = t.ts_utc
        LEFT JOIN feat.minute_panel_full f
          ON f.market_ticker = t.market_ticker
         AND f.ts_utc = t.ts_utc
        WHERE t.city = :city
          AND t.local_date BETWEEN :start AND :end
        ORDER BY t.ts_utc, t.event_ticker, t.bracket_idx
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"city": city, "start": start_date, "end": end_date})
    if df.empty:
        raise ValueError(f"No triad rows for {city} between {start_date} and {end_date}")
    df = _maybe_join_calibrated_probs(df, city)
    return df


def _maybe_join_calibrated_probs(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Attach calibrated up-probabilities for the 5m horizon if calibration + prediction
    artifacts exist on disk. The default column name is p_up_calibrated_5m.
    """

    calib_path = f"results/calibration_{city}_gbdt_5m.json"
    val_path = f"results/{city}_gbdt_5m_val.csv"
    test_path = f"results/{city}_gbdt_5m_test.csv"
    try:
        with open(calib_path, "r") as fh:
            calib = json.load(fh)
    except FileNotFoundError:
        LOGGER.debug("Calibration file not found: %s", calib_path)
        return df

    best_method = calib.get("best_method", "")
    params = calib.get("params", {}).get(best_method, {})
    if not params:
        LOGGER.debug("No params for best_method=%s in %s", best_method, calib_path)
        return df

    preds = []
    for path in (val_path, test_path):
        try:
            preds.append(pd.read_csv(path, parse_dates=["ts_utc"]))
        except FileNotFoundError:
            LOGGER.debug("Prediction file not found: %s", path)
    if not preds:
        return df

    pred_df = pd.concat(preds, ignore_index=True)
    pred_df.rename(columns={"y_prob": "p_raw"}, inplace=True)

    def apply_isotonic(x: pd.Series, x_thresh: List[float], y_thresh: List[float]) -> pd.Series:
        return pd.Series(np.interp(x, x_thresh, y_thresh), index=x.index)

    if best_method == "isotonic":
        x_thr = params.get("x_thresholds", [])
        y_thr = params.get("y_thresholds", [])
        pred_df["p_calibrated"] = apply_isotonic(pred_df["p_raw"].astype(float), x_thr, y_thr)
    elif best_method == "platt":
        coef = params.get("coef", [0.0])[0]
        intercept = params.get("intercept", [0.0])[0]
        logits = coef * pred_df["p_raw"].astype(float) + intercept
        pred_df["p_calibrated"] = 1 / (1 + np.exp(-logits))
    else:
        LOGGER.debug("Unsupported calibration method %s", best_method)
        return df

    pred_df = pred_df[["ts_utc", "market_ticker", "p_calibrated"]].dropna()
    pred_df.rename(columns={"p_calibrated": "p_up_calibrated_5m"}, inplace=True)

    merged = df.merge(pred_df, how="left", on=["ts_utc", "market_ticker"])
    return merged


def _group_zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma <= 0 or np.isnan(sigma):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / sigma


def compute_scores(df: pd.DataFrame, cfg: TriadConfig, apply_gates: bool = True) -> pd.DataFrame:
    df = df.copy()

    # Gate on liquidity (volume)
    df["liq_ok"] = df["volume"].fillna(0) >= cfg.min_volume

    # Spread proxy placeholder (until quotes available)
    df["spread_proxy"] = 0.0  # maker-friendly markets default
    df["spread_ok"] = df["spread_proxy"] <= cfg.max_spread_cents

    grouped = df.groupby(["event_ticker", "local_date"])
    df["ras_accel_z"] = grouped["ras_accel"].transform(_group_zscore)

    df["accel_diff"] = df[["mid_accel_left_diff", "mid_accel_right_diff"]].fillna(0.0).sum(axis=1)
    df["accel_diff_z"] = grouped["accel_diff"].transform(_group_zscore)

    df["vol_z"] = grouped["volume"].transform(_group_zscore)

    hazard_cols = [c for c in ["hazard_next_5m", "hazard_next_60m"] if c in df.columns]
    if hazard_cols:
        df["hazard_gate"] = df[hazard_cols[0]].fillna(0.0)
    else:
        df["hazard_gate"] = 0.0

    # Mispricing term: p_wx - implied (falls back to mid_prob)
    implied_cols = [c for c in ["p_fused_norm", "p_mkt", "mid_prob"] if c in df.columns]
    implied = df[implied_cols[0]] if implied_cols else df["mid_prob"]
    wx_col = df["p_wx"] if "p_wx" in df.columns else None
    if wx_col is not None:
        df["edge_wx"] = wx_col - implied
        df["edge_wx_z"] = grouped["edge_wx"].transform(_group_zscore)
    else:
        df["edge_wx_z"] = 0.0

    df["score_raw"] = (
        cfg.alpha_ras * df["ras_accel_z"].fillna(0.0)
        + cfg.alpha_accel * df["accel_diff_z"].fillna(0.0)
        + cfg.alpha_volume * df["vol_z"].fillna(0.0)
        + cfg.alpha_hazard * df["hazard_gate"].fillna(0.0)
        + cfg.alpha_misprice * df["edge_wx_z"].fillna(0.0)
    )

    if apply_gates:
        df["is_edge"] = (df["bracket_idx"] == 1) | (df["bracket_idx"] == df["num_brackets"])
        triad_mass_ok = df["triad_mass"] >= cfg.triad_mass_min if "triad_mass" in df.columns else True
        hazard_ok = df["hazard_gate"] >= cfg.hazard_min
        hour_local = pd.to_datetime(df["ts_local"]).dt.hour
        tod_ok = (hour_local >= cfg.tod_start) & (hour_local <= cfg.tod_end)
        edge_wx_ok = (df["edge_wx"].abs() >= cfg.edge_wx_min) if "edge_wx" in df.columns else True
        gate = df["liq_ok"] & df["spread_ok"] & (~df["is_edge"]) & triad_mass_ok & hazard_ok & tod_ok & edge_wx_ok
        df["score"] = np.where(gate, df["score_raw"], -np.inf)
    else:
        df["is_edge"] = False
        df["score"] = df["score_raw"]
    return df


def select_triads(df: pd.DataFrame, min_score: float) -> List[Dict[str, object]]:
    intents: List[Dict[str, object]] = []
    for (ts_utc, event), g in df.groupby(["ts_utc", "event_ticker"]):
        idx = g["score"].idxmax()
        if pd.isna(idx):
            continue
        row = g.loc[idx]
        if not np.isfinite(row["score"]) or row["score"] <= min_score:
            continue

        j = int(row["bracket_idx"])
        left = g[g["bracket_idx"] == j - 1]["market_ticker"]
        right = g[g["bracket_idx"] == j + 1]["market_ticker"]
        if left.empty or right.empty:
            continue

        intents.append(
            {
                "ts_utc": ts_utc,
                "event_ticker": event,
                "city": row["city"],
                "market_center": row["market_ticker"],
                "market_left": left.iloc[0],
                "market_right": right.iloc[0],
                "score": float(row["score"]),
                "side_center": "BUY_YES",
                "side_left": "SELL_YES",
                "side_right": "SELL_YES",
            }
        )
    return intents


def run_diagnostics(city: str, start_date: str, end_date: str, cfg: TriadConfig, top_n: int) -> None:
    df = load_triad_panel(city, start_date, end_date)
    df = compute_scores(df, cfg, apply_gates=False)
    grouped = df.groupby(["event_ticker", "local_date"])
    for (event, local_date), g in grouped:
        top = g.nlargest(top_n, "score_raw")
        if top.empty:
            continue
        LOGGER.info("Diagnostics for %s on %s (top %d rows)", event, local_date, len(top))
        for _, row in top.iterrows():
            LOGGER.info(
                "ts=%s idx=%d market=%s mid_accel=%.4f diffL=%.4f diffR=%.4f ras_z=%.4f diff_z=%.4f vol_z=%.4f hazard=%.4f score=%.4f",
                row["ts_utc"],
                int(row["bracket_idx"]),
                row["market_ticker"],
                float(row["mid_acceleration"] or 0.0),
                float(row["mid_accel_left_diff"] or 0.0),
                float(row["mid_accel_right_diff"] or 0.0),
                float(row["ras_accel_z"] or 0.0),
                float(row["accel_diff_z"] or 0.0),
                float(row["vol_z"] or 0.0),
                float(row["hazard_gate"] or 0.0),
                float(row["score_raw"] or 0.0),
            )


def run_signals(city: str, start_date: str, end_date: str, cfg: TriadConfig) -> None:
    df = load_triad_panel(city, start_date, end_date)
    df = compute_scores(df, cfg)
    intents = select_triads(df, cfg.min_score)
    LOGGER.info("Generated %d triad intents for %s %s→%s", len(intents), city, start_date, end_date)
    for intent in intents[:20]:
        LOGGER.info("Intent: %s", intent)


def main() -> None:
    args = parse_args()
    if args.cmd == "signals":
        cfg = TriadConfig(
            min_volume=args.min_volume,
            min_score=args.min_score,
            max_spread_cents=args.max_spread,
        )
        run_signals(args.city, args.start_date, args.end_date, cfg)
    elif args.cmd == "diagnostics":
        cfg = TriadConfig()
        run_diagnostics(args.city, args.start_date, args.end_date, cfg, top_n=args.top_n)
    else:
        raise ValueError(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    main()
