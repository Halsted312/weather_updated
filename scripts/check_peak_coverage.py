#!/usr/bin/env python3
"""
Quick coverage checker for peak-temp strategy inputs.

Reports min/max dates with non-null p_wx, p_fused_norm, and mid_prob in
feat.minute_panel_full for a given city and date range.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import List, Tuple

import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.connection import engine

LOGGER = logging.getLogger("check_peak_coverage")


def configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


def parse_args() -> argparse.Namespace:
    configure_logging()
    parser = argparse.ArgumentParser(description="Check p_wx/p_fused coverage for peak strategy.")
    parser.add_argument("--city", required=True)
    parser.add_argument("--start-date", required=False, default="1900-01-01")
    parser.add_argument("--end-date", required=False, default="2100-01-01")
    return parser.parse_args()


def fetch_coverage(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    query = text(
        """
        SELECT
          MIN(CASE WHEN p_wx IS NOT NULL THEN local_date END) AS min_p_wx,
          MAX(CASE WHEN p_wx IS NOT NULL THEN local_date END) AS max_p_wx,
          MIN(CASE WHEN p_fused_norm IS NOT NULL THEN local_date END) AS min_p_fused,
          MAX(CASE WHEN p_fused_norm IS NOT NULL THEN local_date END) AS max_p_fused,
          MIN(CASE WHEN mid_prob IS NOT NULL THEN local_date END) AS min_mid_prob,
          MAX(CASE WHEN mid_prob IS NOT NULL THEN local_date END) AS max_mid_prob,
          COUNT(*) AS rows_total,
          SUM(CASE WHEN p_wx IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_p_wx
        FROM feat.minute_panel_full
        WHERE city = :city
          AND local_date BETWEEN :start AND :end
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"city": city, "start": start_date, "end": end_date})
    return df


def main() -> None:
    args = parse_args()
    df = fetch_coverage(args.city, args.start_date, args.end_date)
    LOGGER.info("Coverage for %s %s→%s:\n%s", args.city, args.start_date, args.end_date, df.to_string(index=False))


if __name__ == "__main__":
    main()
