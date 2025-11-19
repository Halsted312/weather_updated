#!/usr/bin/env python3
"""Batch trainer/backtester for the Tmax ensemble across multiple cities."""

from __future__ import annotations

import argparse
import logging
import os
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from ml.city_config import CITY_CONFIG

DEFAULT_CUTOFFS = ["12:00", "14:00", "16:00", "18:00"]


def parse_cities(city_args: List[str]) -> List[str]:
    if not city_args or "all" in {c.lower() for c in city_args}:
        return list(CITY_CONFIG.keys())
    cities = []
    for city in city_args:
        city_lower = city.lower()
        if city_lower not in CITY_CONFIG:
            raise ValueError(f"Unknown city '{city}'. Known: {', '.join(CITY_CONFIG.keys())}")
        cities.append(city_lower)
    return cities


def run_cmd(cmd: List[str], env: dict | None = None) -> None:
    logging.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        logging.error("Command failed (%s): %s", result.returncode, result.stderr.strip())
        raise RuntimeError(result.stderr.strip())
    if result.stdout:
        logging.debug(result.stdout.strip())


def build_artifact_paths(args, city: str) -> Tuple[Path, Path, Path, Path]:
    preds = Path(args.results_dir) / f"tmax_preds_{city}_{args.tag}.csv"
    metadata = Path(args.results_dir) / f"tmax_model_{city}_{args.tag}.json"
    seq = Path(args.models_dir) / f"tmax_seq_{city}_{args.tag}.pt"
    backtest_json = Path(args.results_dir) / f"backtest_{city}_tmax_{args.tag}.json"
    return preds, metadata, seq, backtest_json


def train_city(args, city: str, export_path: Path, metadata_path: Path, seq_path: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/train_tmax_regressor.py",
        "--city",
        city,
        "--start",
        args.start,
        "--end",
        args.end,
    ]
    for cutoff in args.cutoffs:
        cmd.extend(["--cutoffs", cutoff])
    cmd.extend(["--export-csv", str(export_path)])
    cmd.extend(["--export-metadata", str(metadata_path)])
    cmd.extend(["--optuna-trials", str(args.optuna_trials)])
    cmd.extend(["--seq-optuna-trials", str(args.seq_optuna_trials)])
    if not args.skip_seq_export:
        cmd.extend(["--export-seq-model", str(seq_path)])
    if args.enable_catboost:
        cmd.append("--enable-catboost")
        cmd.extend(["--catboost-trials", str(args.catboost_trials)])
    run_cmd(cmd)


def backtest_city(args, city: str, preds_path: Path, output_path: Path) -> None:
    cmd = [
        sys.executable,
        "backtest/run_backtest.py",
        "--strategy",
        "model_kelly",
        "--city",
        city,
        "--bracket",
        args.bracket,
        "--start-date",
        args.start,
        "--end-date",
        args.end,
        "--model-type",
        "tmax_reg",
        "--tmax-preds-csv",
        str(preds_path),
        "--tmax-min-prob",
        str(args.tmax_min_prob),
        "--tmax-sigma-multiplier",
        str(args.tmax_sigma_multiplier),
        "--initial-cash",
        str(args.initial_cash),
        "--output-json",
        str(output_path),
    ]
    if args.hybrid_model_type:
        cmd.extend([
            "--hybrid-model-type",
            args.hybrid_model_type,
            "--hybrid-min-prob",
            str(args.hybrid_min_prob),
        ])
    run_cmd(cmd)


def run_daily_baseline(args, city: str, preds_path: Path) -> None:
    output_json = Path(args.results_dir) / f"tmax_daily_{city}_{args.start}_{args.end}.json"
    cmd = [
        sys.executable,
        "scripts/backtest_tmax_daily.py",
        "--city",
        city,
        "--bracket",
        args.bracket,
        "--tmax-preds-csv",
        str(preds_path),
        "--start-date",
        args.start,
        "--end-date",
        args.end,
        "--cutoff",
        args.daily_cutoff,
        "--min-edge",
        str(args.daily_min_edge),
        "--output-json",
        str(output_json),
    ]
    run_cmd(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/backtest Tmax ensemble across cities")
    parser.add_argument("--cities", nargs="*", default=["all"], help="Cities to process (default: all)")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--cutoffs",
        nargs="*",
        default=DEFAULT_CUTOFFS,
        help="Local time cutoffs passed to train_tmax_regressor",
    )
    parser.add_argument("--bracket", default="between", choices=["between", "greater", "less"])
    parser.add_argument("--tmax-min-prob", type=float, default=0.6)
    parser.add_argument("--tmax-sigma-multiplier", type=float, default=0.75)
    parser.add_argument("--hybrid-model-type", choices=["elasticnet", "catboost", "ev_catboost"], default=None)
    parser.add_argument("--hybrid-min-prob", type=float, default=0.5)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--run-daily-baseline", action="store_true")
    parser.add_argument("--daily-cutoff", default="16:00")
    parser.add_argument("--daily-min-edge", type=float, default=0.05)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--tag", default="latest", help="Suffix for artifact names (default: latest)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models-dir", default="models/trained")
    parser.add_argument("--optuna-trials", type=int, default=25)
    parser.add_argument("--seq-optuna-trials", type=int, default=5)
    parser.add_argument("--skip-seq-export", action="store_true")
    parser.add_argument("--enable-catboost", action="store_true", help="Add CatBoost regressor component")
    parser.add_argument("--catboost-trials", type=int, default=25)

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    try:
        cities = parse_cities(args.cities)
    except ValueError as exc:
        logging.error(exc)
        return 1

    failures: list[str] = []

    for city in cities:
        logging.info("=== Processing %s ===", city)
        preds_path, metadata_path, seq_path, output_path = build_artifact_paths(args, city)
        try:
            train_city(args, city, preds_path, metadata_path, seq_path)
            if not args.skip_backtest:
                backtest_city(args, city, preds_path, output_path)
            if args.run_daily_baseline:
                run_daily_baseline(args, city, preds_path)
        except Exception as exc:
            logging.error("City %s failed: %s", city, exc)
            failures.append(f"{city}: {exc}")
            continue

    if failures:
        logging.error("Completed with failures: %s", "; ".join(failures))
        return 1

    logging.info("All cities completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
