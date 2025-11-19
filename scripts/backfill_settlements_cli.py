#!/usr/bin/env python3
"""
Backfill settlement data using IEM CLI API (official Kalshi source).

This pulls historical Daily Climate Report values directly from Iowa
Environmental Mesonet, which archives the CLI bulletins with structured JSON.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List

import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from weather.iem_cli import IEMCliClient, SETTLEMENT_STATIONS
from db.connection import get_session
from db.loaders import upsert_settlement

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill settlement data using IEM CLI JSON API"
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--cities",
        default="all",
        help="Comma-separated city list or 'all' (default)",
    )
    parser.add_argument(
        "--fetch-raw-text",
        action="store_true",
        help="Fetch raw CLI AFOS text for audit payloads",
    )
    return parser.parse_args()


def backfill_city(
    client: IEMCliClient,
    city: str,
    start_date: date,
    end_date: date,
    fetch_raw_text: bool,
) -> Dict[str, int]:
    logger.info(f"\n{'='*60}")
    logger.info(f"BACKFILLING CLI: {city.upper()}")
    logger.info(f"{'='*60}\n")

    stats = {"records": 0, "errors": 0}

    settlements = client.get_settlements_for_city(
        city, start_date, end_date, fetch_raw_text
    )

    if not settlements:
        logger.warning(f"No CLI records found for {city}")
        return stats

    with get_session() as session:
        for record in settlements:
            try:
                upsert_settlement(session, record)
                stats["records"] += 1
            except Exception as exc:
                logger.error(
                    f"Failed upserting {city} {record['date_local']}: {exc}",
                    exc_info=True,
                )
                stats["errors"] += 1
        session.commit()

    logger.info(
        f"✓ {city.upper()} CLI backfill complete ({stats['records']} records)"
    )
    return stats


def main():
    args = parse_args()
    load_dotenv()

    if args.cities == "all":
        cities = list(SETTLEMENT_STATIONS.keys())
    else:
        cities = [c.strip() for c in args.cities.split(",")]

    for city in cities:
        if city not in SETTLEMENT_STATIONS:
            logger.error(f"Unknown city: {city}")
            return 1

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    client = IEMCliClient()
    total = {"records": 0, "errors": 0}

    for city in cities:
        stats = backfill_city(
            client,
            city,
            start_date,
            end_date,
            fetch_raw_text=args.fetch_raw_text,
        )
        total["records"] += stats["records"]
        total["errors"] += stats["errors"]

    logger.info("\nSUMMARY")
    logger.info("=======")
    logger.info(f"Cities processed: {len(cities)}")
    logger.info(f"Records inserted: {total['records']}")
    logger.info(f"Errors: {total['errors']}")

    return 0 if total["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
