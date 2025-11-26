#!/usr/bin/env python3
"""Backfill missing strike metadata directly in PostgreSQL."""

from __future__ import annotations

import argparse
import logging
from typing import Dict, List

from sqlalchemy import text

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import get_session
from kalshi.strike_parser import ensure_strike_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_candidates(city: str | None) -> List[Dict]:
    where_clause = "WHERE strike_type IS NULL"
    if city:
        where_clause += " AND series_ticker = :series"

    stmt = text(
        f"""
        SELECT ticker, series_ticker, event_ticker, title, subtitle,
               strike_type, floor_strike, cap_strike,
               rules_primary, rules_secondary
        FROM markets
        {where_clause}
        ORDER BY close_time
        """
    )

    params = {"series": city.upper()} if city else {}

    with get_session() as session:
        rows = session.execute(stmt, params).mappings().all()
    return [dict(row) for row in rows]


def apply_updates(rows: List[Dict]) -> int:
    if not rows:
        return 0

    updated = 0
    with get_session() as session:
        for row in rows:
            enriched = ensure_strike_metadata(row.copy())
            if enriched.get("strike_type") == row.get("strike_type"):
                continue
            stmt = text(
                """
                UPDATE markets
                SET strike_type = :strike_type,
                    floor_strike = COALESCE(:floor_strike, floor_strike),
                    cap_strike = COALESCE(:cap_strike, cap_strike)
                WHERE ticker = :ticker
                """
            )
            session.execute(
                stmt,
                {
                    "strike_type": enriched.get("strike_type"),
                    "floor_strike": enriched.get("floor_strike"),
                    "cap_strike": enriched.get("cap_strike"),
                    "ticker": enriched["ticker"],
                },
            )
            updated += 1
        session.commit()
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer missing strike metadata inside Postgres")
    parser.add_argument("--series", help="Optional series ticker (e.g., KXHIGHCHI)")
    args = parser.parse_args()

    rows = fetch_candidates(args.series)
    logger.info("Found %s candidate markets", len(rows))
    updated = apply_updates(rows)
    logger.info("Updated %s markets with inferred strike metadata", updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
