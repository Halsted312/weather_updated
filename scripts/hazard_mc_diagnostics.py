#!/usr/bin/env python3
"""
Diagnostics for hazard Monte Carlo outputs.

Provides commands to:
  - check PMF normalization across timestamps
  - evaluate morning probabilities vs realized Tmax
  - export hazard traces for plotting
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List

from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import text

from db.connection import engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hazard MC diagnostics")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pmf = sub.add_parser("check-pmf-sum", help="Check PMF normalization across timestamps")
    pmf.add_argument("--city", required=True)
    pmf.add_argument("--start-date", required=True)
    pmf.add_argument("--end-date", required=True)
    pmf.add_argument("--tolerance", type=float, default=0.01)

    eval_prob = sub.add_parser("evaluate-morning", help="Compare morning p_wx vs realized Tmax")
    eval_prob.add_argument("--city", required=True)
    eval_prob.add_argument("--start-date", required=True)
    eval_prob.add_argument("--end-date", required=True)
    eval_prob.add_argument("--window-start", default="09:00", help="Local time HH:MM")
    eval_prob.add_argument("--window-end", default="10:00", help="Local time HH:MM")
    eval_prob.add_argument("--out", type=Path, default=Path("morning_eval.csv"))

    hazard = sub.add_parser("hazard-trace", help="Export hazard trace for a day")
    hazard.add_argument("--city", required=True)
    hazard.add_argument("--date", required=True, help="Local date YYYY-MM-DD")
    hazard.add_argument("--out", type=Path, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_city_meta(city: str) -> Dict[str, str]:
    query = text(
        """
        SELECT dc.tz, wx.loc_id
        FROM dim_city dc
        JOIN wx.location wx ON wx.city = dc.city
        WHERE dc.city = :city
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query, {"city": city}).mappings().first()
    if not row:
        raise ValueError(f"City '{city}' not found in dim_city/wx.location")
    return {"tz": row["tz"], "loc_id": row["loc_id"]}


def to_utc(dt_local: dt.datetime, tz: ZoneInfo) -> dt.datetime:
    if dt_local.tzinfo is None:
        dt_local = dt_local.replace(tzinfo=tz)
    return dt_local.astimezone(dt.timezone.utc)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def check_pmf_sum(city: str, start_date: str, end_date: str, tolerance: float) -> None:
    query = text(
        """
        SELECT ts_utc, MAX(ts_local) AS ts_local, SUM(p_wx) AS sum_pwx
        FROM pmf.minute
        WHERE city = :city
          AND local_date BETWEEN :start AND :end
        GROUP BY ts_utc
        ORDER BY ts_utc
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params={"city": city, "start": start_date, "end": end_date},
        )

    if df.empty:
        print("No pmf.minute rows for given range")
        return

    df["deviation"] = df["sum_pwx"] - 1.0
    violations = df[df["deviation"].abs() > tolerance]
    print(f"Total timestamps: {len(df)}")
    print(f"Violations (> {tolerance:.4f}): {len(violations)}")
    if not violations.empty:
        print("Worst offenders:")
        print(violations.nlargest(10, "deviation").to_string(index=False))


def evaluate_morning(city: str, start_date: str, end_date: str, window_start: str, window_end: str, out: Path) -> None:
    meta = get_city_meta(city)
    tz = ZoneInfo(meta["tz"])

    pmf_query = text(
        """
        SELECT local_date, market_ticker, floor_strike, cap_strike, strike_type, AVG(p_wx) AS avg_p_wx
        FROM pmf.minute
        WHERE city = :city
          AND local_date BETWEEN :start AND :end
          AND to_char(ts_local, 'HH24:MI') >= :window_start
          AND to_char(ts_local, 'HH24:MI') < :window_end
        GROUP BY local_date, market_ticker, floor_strike, cap_strike, strike_type
        """
    )
    with engine.connect() as conn:
        pmf_df = pd.read_sql_query(
            pmf_query,
            conn,
            params={
                "city": city,
                "start": start_date,
                "end": end_date,
                "window_start": window_start,
                "window_end": window_end,
            },
        )

    tmax_df = fetch_daily_tmax(city, tz, meta["loc_id"], start_date, end_date)

    rows = []
    grouped = pmf_df.groupby("local_date")
    for _, tmax_row in tmax_df.iterrows():
        local_date = tmax_row["local_date"]
        actual_temp = tmax_row["tmax_f"]
        pmf_rows = grouped.get_group(local_date) if local_date in grouped.groups else None
        if pmf_rows is None:
            continue
        match_row = match_bracket(actual_temp, pmf_rows)
        prob = match_row["avg_p_wx"] if match_row is not None else None
        rows.append(
            {
                "local_date": local_date,
                "actual_tmax_f": actual_temp,
                "actual_market": match_row["market_ticker"] if match_row is not None else None,
                "morning_prob": prob,
            }
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        print("No evaluation rows produced.")
        return

    result_df.to_csv(out, index=False)
    print(f"Wrote morning evaluation to {out} ({len(result_df)} rows)")


def fetch_daily_tmax(city: str, tz: ZoneInfo, loc_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    start_utc = to_utc(dt.datetime.combine(start_dt, dt.time.min), tz)
    end_utc = to_utc(dt.datetime.combine(end_dt + dt.timedelta(days=1), dt.time.min), tz)

    query = text(
        """
        WITH city_obs AS (
            SELECT ts_utc,
                   temp_f,
                   (ts_utc AT TIME ZONE :tz)::date AS local_date
            FROM wx.minute_obs
            WHERE loc_id = :loc_id
              AND ts_utc >= :start_utc
              AND ts_utc < :end_utc
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY local_date ORDER BY temp_f DESC, ts_utc) AS rn
            FROM city_obs
        )
        SELECT local_date, temp_f AS tmax_f, ts_utc
        FROM ranked
        WHERE rn = 1
        ORDER BY local_date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params={"tz": tz.key, "loc_id": loc_id, "start_utc": start_utc, "end_utc": end_utc},
        )
    return df


def match_bracket(temp: float, pmf_rows: pd.DataFrame) -> pd.Series | None:
    for _, row in pmf_rows.iterrows():
        strike_type = (row["strike_type"] or "between").lower()
        floor = row["floor_strike"]
        cap = row["cap_strike"]
        if strike_type == "greater":
            if pd.isna(floor) or temp >= floor:
                return row
        elif strike_type == "less":
            if pd.isna(cap) or temp <= cap:
                return row
        else:
            floor_val = -float("inf") if pd.isna(floor) else floor
            cap_val = float("inf") if pd.isna(cap) else cap
            if floor_val <= temp <= cap_val:
                return row
    return None


def hazard_trace(city: str, date_str: str, out: Path | None) -> None:
    meta = get_city_meta(city)
    tz = ZoneInfo(meta["tz"])
    local_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()

    hazard_query = text(
        """
        SELECT ts_local, MAX(hazard_next_5m) AS hazard_5m, MAX(hazard_next_60m) AS hazard_60m,
               MAX(m_run_temp_f) AS m_run_temp
        FROM pmf.minute
        WHERE city = :city
          AND local_date = :local_date
        GROUP BY ts_local
        ORDER BY ts_local
        """
    )
    with engine.connect() as conn:
        hazard_df = pd.read_sql_query(
            hazard_query,
            conn,
            params={"city": city, "local_date": local_date},
        )

    if hazard_df.empty:
        print("No hazard rows for selected day.")
        return

    tmax_df = fetch_daily_tmax(city, tz, meta["loc_id"], date_str, date_str)
    if not tmax_df.empty:
        actual_time = tmax_df["ts_utc"].iloc[0]
        actual_temp = tmax_df["tmax_f"].iloc[0]
        hazard_df["actual_tmax_f"] = actual_temp
        hazard_df["actual_tmax_ts"] = actual_time

    if out is None:
        out = Path(f"hazard_trace_{city}_{date_str}.csv")
    hazard_df.to_csv(out, index=False)
    print(f"Wrote hazard trace to {out} ({len(hazard_df)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.cmd == "check-pmf-sum":
        check_pmf_sum(args.city, args.start_date, args.end_date, args.tolerance)
    elif args.cmd == "evaluate-morning":
        evaluate_morning(
            args.city,
            args.start_date,
            args.end_date,
            args.window_start,
            args.window_end,
            args.out,
        )
    elif args.cmd == "hazard-trace":
        hazard_trace(args.city, args.date, args.out)
    else:
        raise ValueError(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    main()
