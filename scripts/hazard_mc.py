#!/usr/bin/env python3
"""
Hazard Monte Carlo scaffolding.

This module defines the entrypoints and helper stubs required to:

1. Fetch minute-level features from `feat.minute_panel_with_weather`
2. Build a 5-minute baseline temperature path (VC Timeline forecast or diurnal template)
3. Simulate AR(1) residual paths to estimate per-bracket p_wx and hazard scalars
4. Persist results into `pmf.minute`
5. Manage residual parameters in `wx.mc_params`

Implementation TODOs are left as explicit stubs so we can fill them in during Phase 3.
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import logging
import math
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import engine


GRID_MINUTES = 5
DEFAULT_PATHS = 4000
BUCKET_MINUTES = [60, 180, 360, 720]  # minute-of-day boundaries for sigma buckets
MC_VERSION = f"v1_grid{GRID_MINUTES}"
LOGGER = logging.getLogger("hazard_mc")


def configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        )


@dataclass
class MCParams:
    city: str
    rho: float
    sigma_buckets: Dict[str, float]
    baseline: Dict[str, float] | None = None


def parse_args() -> argparse.Namespace:
    configure_logging()
    parser = argparse.ArgumentParser(description="Weather hazard Monte Carlo")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_day = sub.add_parser("run-day", help="Compute hazard MC for a single city/day")
    run_day.add_argument("--city", required=True)
    run_day.add_argument("--date", required=True, help="Local date YYYY-MM-DD")
    run_day.add_argument("--paths", type=int, default=DEFAULT_PATHS)

    backfill = sub.add_parser("backfill", help="Backfill hazard MC over a date range")
    backfill.add_argument("--city", action="append", required=True)
    backfill.add_argument("--start-date", required=True)
    backfill.add_argument("--end-date", required=True)
    backfill.add_argument("--paths", type=int, default=DEFAULT_PATHS)

    fit = sub.add_parser("fit-params", help="Fit residual params for a city")
    fit.add_argument("--city", required=True)
    fit.add_argument("--start-date", required=True)
    fit.add_argument("--end-date", required=True)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def fetch_panel_for_day(city: str, local_date: dt.date) -> pd.DataFrame:
    """
    Fetch minute-level rows (with neighbor & weather features) for the given city/date.

    Source view: feat.minute_panel_with_weather
    """
    query = text(
        """
        SELECT *
        FROM feat.minute_panel_with_weather
        WHERE city = :city
          AND local_date = :local_date
        ORDER BY ts_utc
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"city": city, "local_date": local_date})
    if df.empty:
        raise ValueError(f"No panel rows for {city} on {local_date}")
    return df


def load_mc_params(city: str) -> MCParams | None:
    """Fetch residual parameters from wx.mc_params."""
    query = text("SELECT city, rho, sigma_buckets, baseline FROM wx.mc_params WHERE city = :city")
    with engine.connect() as conn:
        row = conn.execute(query, {"city": city}).mappings().first()
    if not row:
        return None
    return MCParams(
        city=row["city"],
        rho=row["rho"],
        sigma_buckets=row["sigma_buckets"],
        baseline=row["baseline"],
    )


def save_mc_params(params: MCParams) -> None:
    """Upsert residual params into wx.mc_params."""
    query = text(
        """
        INSERT INTO wx.mc_params (city, rho, sigma_buckets, baseline, updated_at)
        VALUES (:city, :rho, :sigma_buckets, :baseline, now())
        ON CONFLICT (city)
        DO UPDATE SET
            rho = EXCLUDED.rho,
            sigma_buckets = EXCLUDED.sigma_buckets,
            baseline = EXCLUDED.baseline,
            updated_at = now()
        """
    )
    with engine.connect() as conn:
        conn.execute(
            query,
            {
                "city": params.city,
                "rho": params.rho,
                "sigma_buckets": params.sigma_buckets,
                "baseline": params.baseline,
            },
        )


# ---------------------------------------------------------------------------
# Baseline builders (stubs)
# ---------------------------------------------------------------------------

def build_baseline_from_timeline(
    city: str,
    local_date: dt.date,
    panel: pd.DataFrame,
    grid_minutes: int = GRID_MINUTES,
) -> pd.Series:
    """
    Preferred baseline: use Visual Crossing Timeline sub-hourly forecast.

    Returns a Series indexed by UTC timestamps on the forward 5-minute grid.
    """
    raise NotImplementedError("Timeline baseline builder not implemented yet")


def build_baseline_from_template(
    city: str,
    local_date: dt.date,
    panel: pd.DataFrame,
    grid_minutes: int = GRID_MINUTES,
) -> pd.Series:
    """
    Fallback baseline: persistence/diurnal template combo.

    Current implementation:
        - Use observed wx_temp_1m (fallback to wx_temp_5m) for minutes up to the latest available.
        - Forward-fill missing values.
        - For future minutes (beyond final observation), hold temperature at the last observed value.
    """
    if "ts_local" not in panel or panel["ts_local"].isna().all():
        raise ValueError("panel must include ts_local with timezone information")

    tzinfo = panel["ts_local"].iloc[0].tzinfo
    if tzinfo is None:
        # derive offset from ts_local - ts_utc if tzinfo missing
        offset = panel["ts_local"].iloc[0] - panel["ts_utc"].iloc[0]
        tzinfo = dt.timezone(offset)

    local_midnight = dt.datetime.combine(local_date, dt.time.min).replace(tzinfo=tzinfo)
    local_end = local_midnight + dt.timedelta(days=1)
    grid_local = pd.date_range(
        local_midnight,
        local_end,
        freq=f"{grid_minutes}min",
        inclusive="left",
    )
    grid_utc = grid_local.tz_convert(dt.timezone.utc)

    temp_series = panel[["ts_utc", "wx_temp_1m", "wx_temp_5m"]].copy()
    temp_series["wx_temp"] = temp_series["wx_temp_1m"].fillna(temp_series["wx_temp_5m"])
    temp_series = temp_series.dropna(subset=["wx_temp"])

    obs = temp_series.set_index("ts_utc")["wx_temp"].sort_index()
    obs = obs.reindex(grid_utc, method="ffill")

    if obs.isna().all():
        # fall back to panel close values scaled to Fahrenheit
        panel_close = panel.set_index("ts_utc")["close_c"] / 100.0 * 100
        obs = panel_close.reindex(grid_utc, method="ffill")

    obs = obs.ffill()
    obs = obs.fillna(method="bfill")

    return obs


def choose_baseline(
    city: str,
    local_date: dt.date,
    panel: pd.DataFrame,
    grid_minutes: int = GRID_MINUTES,
) -> pd.Series:
    """
    Decide which baseline to use (Timeline forecast preferred, template fallback).
    """
    try:
        return build_baseline_from_timeline(city, local_date, panel, grid_minutes)
    except NotImplementedError:
        # For scaffolding we simply delegate to template builder
        return build_baseline_from_template(city, local_date, panel, grid_minutes)


# ---------------------------------------------------------------------------
# Residual fitting / simulation (stubs)
# ---------------------------------------------------------------------------

def fit_residual_params(
    city: str,
    start_date: dt.date,
    end_date: dt.date,
    grid_minutes: int = GRID_MINUTES,
) -> MCParams:
    """
    Fit AR(1) coefficient and variance buckets from historical residuals.

    TODO: implement actual estimation.
    """
    buckets = _init_sigma_bucket_dict()
    all_eps: List[float] = []

    current = start_date
    while current <= end_date:
        try:
            panel = fetch_panel_for_day(city, current)
        except ValueError:
            current += dt.timedelta(days=1)
            continue

        temps = _extract_resampled_temps(panel, grid_minutes)
        if temps is None or len(temps) < 3:
            current += dt.timedelta(days=1)
            continue

        diffs = temps.diff().dropna()
        if diffs.empty:
            current += dt.timedelta(days=1)
            continue

        all_eps.extend(diffs.values.tolist())
        for ts, delta in diffs.items():
            minute = (ts - ts.normalize()).total_seconds() / 60
            label = _bucket_label(minute)
            buckets[label].append(delta ** 2)

        current += dt.timedelta(days=1)

    if not all_eps:
        rho = 0.0
        global_sigma = 1.0
    else:
        eps = np.asarray(all_eps)
        if len(eps) < 2 or np.var(eps[:-1]) == 0:
            rho = 0.0
        else:
            rho = float(np.corrcoef(eps[1:], eps[:-1])[0, 1])
        global_sigma = float(np.sqrt(np.mean(eps ** 2)))

    sigma_buckets: Dict[str, float] = {}
    for label, values in buckets.items():
        if values:
            sigma_buckets[label] = float(np.sqrt(np.mean(values)))
        else:
            sigma_buckets[label] = global_sigma

    return MCParams(
        city=city,
        rho=max(min(rho, 0.99), -0.99),
        sigma_buckets=sigma_buckets,
        baseline={"grid_minutes": grid_minutes, "bucket_labels": list(sigma_buckets.keys())},
    )


def simulate_paths(
    F_grid: np.ndarray,
    rho: float,
    sigma_seq: np.ndarray,
    paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate additive residual paths (shape: paths x len(F_grid)).

    eps_k = rho * eps_{k-1} + sigma_k * z_k,  z_k ~ N(0, 1)
    """
    if F_grid.size == 0:
        return np.empty((paths, 0))

    rng = np.random.default_rng(seed)
    temps = np.empty((paths, F_grid.size), dtype=float)
    residuals = np.empty((paths, F_grid.size), dtype=float)
    noise = rng.standard_normal(size=(paths, F_grid.size))

    residuals[:, 0] = sigma_seq[0] * noise[:, 0]
    temps[:, 0] = F_grid[0] + residuals[:, 0]
    for k in range(1, F_grid.size):
        residuals[:, k] = rho * residuals[:, k - 1] + sigma_seq[k] * noise[:, k]
        temps[:, k] = F_grid[k] + residuals[:, k]
    return temps


# ---------------------------------------------------------------------------
# PMF + hazard calculations (stubs)
# ---------------------------------------------------------------------------

def compute_p_wx_and_hazards(
    city: str,
    panel: pd.DataFrame,
    baseline: pd.Series,
    params: MCParams,
    paths: int = DEFAULT_PATHS,
) -> pd.DataFrame:
    """
    Core MC routine: returns long-form dataframe with columns:
        ['city','ts_utc','market_ticker','p_wx','hazard_next_5m','hazard_next_60m','m_run_temp_f']

    """
    if baseline.empty:
        raise ValueError("baseline series is empty")

    tzinfo = panel["ts_local"].iloc[0].tzinfo
    mc_version = f"{MC_VERSION}_paths{paths}"
    hazard_rows: List[Dict[str, object]] = []

    grouped = panel.groupby("ts_utc")
    for ts_utc, rows in grouped:
        rows = rows.copy()
        m_run_series = rows.get("wx_running_max")
        if m_run_series is not None and not m_run_series.isna().all():
            m_run_temp = float(m_run_series.iloc[0])
        else:
            temp_cols = ["wx_temp_1m", "wx_temp_5m"]
            temp_vals = rows[temp_cols].bfill(axis=1).iloc[:, 0]
            if temp_vals.isna().all():
                m_run_temp = float(rows["mid_prob"].iloc[0] * 100.0)
            else:
                m_run_temp = float(temp_vals.max())

        future = baseline.loc[baseline.index > ts_utc]
        if future.empty:
            # no future grid points; hazard zero, probability mass at m_run
            hazard_next_5m = 0.0
            hazard_next_60m = 0.0
            # deterministic Tmax = m_run_temp
            for _, row in rows.iterrows():
                p_wx = _bracket_probability(np.array([m_run_temp]), row)
                hazard_rows.append(
                    _build_result_row(row, m_run_temp, p_wx, hazard_next_5m, hazard_next_60m, mc_version)
                )
            continue

        sigma_seq = _sigma_sequence_for_index(future.index, params.sigma_buckets, tzinfo)
        sims = simulate_paths(future.to_numpy(), params.rho, sigma_seq, paths)
        future_max = sims.max(axis=1)
        M_total = np.maximum(future_max, m_run_temp)

        horizon_steps_5m = min(max(1, math.ceil(5 / GRID_MINUTES)), sims.shape[1])
        horizon_steps_60m = min(max(1, math.ceil(60 / GRID_MINUTES)), sims.shape[1])
        hazard_next_5m = float(np.mean(sims[:, :horizon_steps_5m].max(axis=1) > m_run_temp))
        hazard_next_60m = float(np.mean(sims[:, :horizon_steps_60m].max(axis=1) > m_run_temp))

        for _, row in rows.iterrows():
            p_wx = _bracket_probability(M_total, row)
            hazard_rows.append(
                _build_result_row(row, m_run_temp, p_wx, hazard_next_5m, hazard_next_60m, mc_version)
            )

    result_df = pd.DataFrame(hazard_rows)
    return result_df


def persist_pmf_rows(df: pd.DataFrame) -> None:
    """
    Write MC outputs into pmf.minute (upsert on (market_ticker, ts_utc)).
    """
    records = df.to_dict(orient="records")
    if not records:
        return
    insert_sql = text(
        """
        INSERT INTO pmf.minute (
            city, series_ticker, event_ticker, market_ticker,
            ts_utc, ts_local, local_date,
            floor_strike, cap_strike, strike_type,
            m_run_temp_f, p_wx,
            hazard_next_5m, hazard_next_60m, mc_version
        ) VALUES (
            :city, :series_ticker, :event_ticker, :market_ticker,
            :ts_utc, :ts_local, :local_date,
            :floor_strike, :cap_strike, :strike_type,
            :m_run_temp_f, :p_wx,
            :hazard_next_5m, :hazard_next_60m, :mc_version
        )
        ON CONFLICT (market_ticker, ts_utc)
        DO UPDATE SET
            m_run_temp_f = EXCLUDED.m_run_temp_f,
            p_wx = EXCLUDED.p_wx,
            hazard_next_5m = EXCLUDED.hazard_next_5m,
            hazard_next_60m = EXCLUDED.hazard_next_60m,
            mc_version = EXCLUDED.mc_version
        """
    )
    with engine.connect() as conn:
        conn.execute(insert_sql, records)


# ---------------------------------------------------------------------------
# CLI workflows
# ---------------------------------------------------------------------------

def run_day(city: str, date_str: str, paths: int) -> None:
    local_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    LOGGER.info("Running hazard MC for %s on %s (paths=%d)", city, date_str, paths)
    start_time = time.perf_counter()
    panel = fetch_panel_for_day(city, local_date)
    baseline = choose_baseline(city, local_date, panel)
    params = load_mc_params(city)
    if params is None:
        raise RuntimeError(f"No MC params found for {city}. Run fit-params first.")
    mc_df = compute_p_wx_and_hazards(city, panel, baseline, params, paths=paths)
    persist_pmf_rows(mc_df)
    duration = time.perf_counter() - start_time
    LOGGER.info("Completed %s %s (rows=%d) in %.2fs", city, date_str, len(mc_df), duration)


def backfill(cities: Sequence[str], start_date: str, end_date: str, paths: int) -> None:
    start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    current = start
    while current <= end:
        for city in cities:
            LOGGER.info("Backfill run-day %s %s", city, current.isoformat())
            try:
                run_day(city, current.isoformat(), paths)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Failed MC for %s %s: %s", city, current.isoformat(), exc)
        current += dt.timedelta(days=1)


def main() -> None:
    args = parse_args()
    if args.cmd == "run-day":
        run_day(args.city, args.date, args.paths)
    elif args.cmd == "backfill":
        backfill(args.city, args.start_date, args.end_date, args.paths)
    elif args.cmd == "fit-params":
        params = fit_residual_params(
            city=args.city,
            start_date=dt.datetime.strptime(args.start_date, "%Y-%m-%d").date(),
            end_date=dt.datetime.strptime(args.end_date, "%Y-%m-%d").date(),
        )
        save_mc_params(params)
    else:
        raise ValueError(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_sigma_bucket_dict() -> Dict[str, List[float]]:
    labels = []
    prev = 0
    for edge in BUCKET_MINUTES:
        labels.append(f"{prev}-{edge}")
        prev = edge
    labels.append(f"{prev}+")
    return {label: [] for label in labels}


def _bucket_label(minute: float) -> str:
    prev = 0
    for edge in BUCKET_MINUTES:
        if minute < edge:
            return f"{prev}-{edge}"
        prev = edge
    return f"{prev}+"


def _extract_resampled_temps(panel: pd.DataFrame, grid_minutes: int) -> pd.Series | None:
    if "ts_local" not in panel or "wx_temp_1m" not in panel:
        return None
    temps = panel[["ts_local", "wx_temp_1m", "wx_temp_5m"]].copy()
    temps["wx_temp"] = temps["wx_temp_1m"].fillna(temps["wx_temp_5m"])
    temps = temps.dropna(subset=["wx_temp"]).set_index("ts_local").sort_index()
    if temps.empty:
        return None
    resampled = temps["wx_temp"].resample(f"{grid_minutes}min").mean().interpolate("time").ffill().bfill()
    return resampled


def _sigma_sequence_for_index(
    index: pd.Index, sigma_buckets: Dict[str, float], tzinfo: ZoneInfo | None
) -> np.ndarray:
    seq = []
    for ts in index:
        if isinstance(ts, pd.Timestamp):
            local_ts = ts.tz_convert(tzinfo) if tzinfo else ts
            minute = local_ts.hour * 60 + local_ts.minute
        else:
            minute = 0
        label = _bucket_label(minute)
        seq.append(float(sigma_buckets.get(label, sigma_buckets[next(iter(sigma_buckets))])))
    return np.asarray(seq, dtype=float)


def _bracket_probability(samples: np.ndarray, row: pd.Series) -> float:
    strike_type = (row.get("strike_type") or "between").lower()
    floor = row.get("floor_strike")
    cap = row.get("cap_strike")

    if strike_type == "greater":
        floor_val = -np.inf if pd.isna(floor) else float(floor)
        mask = samples >= floor_val
    elif strike_type == "less":
        cap_val = np.inf if pd.isna(cap) else float(cap)
        mask = samples <= cap_val
    else:  # between
        floor_val = -np.inf if pd.isna(floor) else float(floor)
        cap_val = np.inf if pd.isna(cap) else float(cap)
        mask = (samples >= floor_val) & (samples <= cap_val)

    return float(np.mean(mask))


def _build_result_row(
    row: pd.Series,
    m_run_temp: float,
    p_wx: float,
    hazard_5m: float,
    hazard_60m: float,
    mc_version: str,
) -> Dict[str, object]:
    return {
        "city": row["city"],
        "series_ticker": row["series_ticker"],
        "event_ticker": row["event_ticker"],
        "market_ticker": row["market_ticker"],
        "ts_utc": row["ts_utc"],
        "ts_local": row["ts_local"],
        "local_date": row["local_date"],
        "floor_strike": row.get("floor_strike"),
        "cap_strike": row.get("cap_strike"),
        "strike_type": row.get("strike_type"),
        "m_run_temp_f": m_run_temp,
        "p_wx": p_wx,
        "hazard_next_5m": hazard_5m,
        "hazard_next_60m": hazard_60m,
        "mc_version": mc_version,
    }
