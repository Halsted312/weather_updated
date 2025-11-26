#!/usr/bin/env python3
"""Summarize CLI/CF6 settlement inserts for monitoring the poller."""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import List

from sqlalchemy import text

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


QUERY = text(
    """
    SELECT city,
           COUNT(*) FILTER (WHERE tmax_cli IS NOT NULL) AS cli_rows,
           COUNT(*) FILTER (WHERE tmax_cf6 IS NOT NULL) AS cf6_rows,
           COUNT(*) FILTER (WHERE tmax_iem_cf6 IS NOT NULL) AS iem_cf6_rows,
           COUNT(*) FILTER (WHERE tmax_vc IS NOT NULL) AS vc_rows,
           COUNT(*) FILTER (WHERE tmax_ghcnd IS NOT NULL) AS ghcnd_rows
    FROM wx.settlement
    WHERE date_local BETWEEN :start AND :end
    GROUP BY city
    ORDER BY city
    """
)


def run_rollup(start: date, end: date) -> List[dict]:
    with get_session() as session:
        rows = session.execute(QUERY, {"start": start, "end": end}).mappings().all()
    return [dict(row) for row in rows]


def save_csv(rows: List[dict], output_path: Path) -> None:
    if not rows:
        logger.warning("No rows to save for %s", output_path)
        return
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["city", "cli_rows", "cf6_rows", "iem_cf6_rows", "vc_rows", "ghcnd_rows"],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved rollup to %s", output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate settlement rollup for monitoring")
    parser.add_argument("--days", type=int, default=1, help="Number of trailing days to summarise (default: 1)")
    parser.add_argument("--output", help="Optional CSV output path")
    args = parser.parse_args()

    end = date.today()
    start = end - timedelta(days=max(0, args.days - 1))
    rows = run_rollup(start, end)

    if not rows:
        logger.info("No settlement rows between %s and %s", start, end)
        return 0

    logger.info("Settlement rollup (%s → %s)", start, end)
    for row in rows:
        logger.info(
            "%-13s CLI=%3d CF6=%3d IEM=%3d VC=%3d GHCND=%3d",
            row["city"],
            row["cli_rows"],
            row["cf6_rows"],
            row["iem_cf6_rows"],
            row["vc_rows"],
            row["ghcnd_rows"],
        )

    if args.output:
        save_csv(rows, Path(args.output))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
