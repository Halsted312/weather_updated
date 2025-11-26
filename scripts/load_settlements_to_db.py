#!/usr/bin/env python3
"""
Load reconciled settlement data to database.

Maps CSV columns to database schema:
- tmax_cf6_f → tmax_cf6 (source='CF6')
- tmax_ads_f → tmax_iem_cf6 (source='IEM_CF6') when CF6 is missing
- source_final determines which column to populate

Database precedence (via COALESCE): CLI > CF6 > IEM_CF6 > GHCND > VC
"""

import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import get_session
from db.loaders import upsert_settlement

logger = logging.getLogger(__name__)


def load_settlements_from_csv(csv_path: str, delete_range: bool = True) -> dict:
    """
    Load settlement data from CSV to database.

    Args:
        csv_path: Path to settlements_reconciled.csv
        delete_range: If True, delete existing records in date range before loading

    Returns:
        Dict with loading statistics
    """
    # Read CSV
    logger.info(f"Reading settlement data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Convert date_local to date
    df["date_local"] = pd.to_datetime(df["date_local"]).dt.date

    # Get date range
    min_date = df["date_local"].min()
    max_date = df["date_local"].max()
    logger.info(f"Date range: {min_date} to {max_date}")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Cities: {df['city'].unique().tolist()}")

    # Delete existing records in range if requested
    if delete_range:
        logger.info(f"Deleting existing records in range {min_date} to {max_date}")
        with get_session() as session:
            from sqlalchemy import text
            result = session.execute(
                text("DELETE FROM wx.settlement WHERE date_local BETWEEN :start AND :end"),
                {"start": min_date, "end": max_date}
            )
            session.commit()
            deleted = result.rowcount
            logger.info(f"Deleted {deleted} existing records")

    # Map source_final to database source names
    # "cf6" → 'CF6' (official NWS product)
    # "ads" → 'IEM_CF6' (IEM fallback when CF6 missing)
    # "vc" → 'VC' (Visual Crossing)
    source_map = {
        "cf6": "CF6",
        "ads": "IEM_CF6",  # ADS goes into IEM_CF6 column as IEM fallback
        "vc": "VC",
    }

    # Track statistics
    stats = {
        "total": len(df),
        "by_source": {},
        "errors": 0,
    }

    # Load each record
    logger.info("Loading settlement records to database...")
    with get_session() as session:
        for idx, row in df.iterrows():
            try:
                # Determine which source to use
                source_key = row["source_final"]
                if source_key not in source_map:
                    logger.warning(f"Unknown source: {source_key} for {row['city']} on {row['date_local']}")
                    stats["errors"] += 1
                    continue

                db_source = source_map[source_key]

                # Get temperature value from appropriate column
                if source_key == "cf6":
                    tmax_f = row["tmax_cf6_f"]
                elif source_key == "ads":
                    tmax_f = row["tmax_ads_f"]
                elif source_key == "vc":
                    tmax_f = row.get("tmax_vc_f")
                else:
                    tmax_f = row["tmax_final_f"]

                # Skip if temperature is missing
                if pd.isna(tmax_f):
                    logger.warning(f"Missing temperature for {row['city']} on {row['date_local']}")
                    stats["errors"] += 1
                    continue

                # Build settlement data dict
                # Convert pandas NaN to None for JSON compatibility
                def sanitize_value(val):
                    if pd.isna(val):
                        return None
                    return float(val) if isinstance(val, (int, float)) else val

                settlement_data = {
                    "city": row["city"],
                    "date_local": row["date_local"],
                    "tmax_f": int(tmax_f),
                    "source": db_source,
                    "is_preliminary": True,  # CF6 and ADS are preliminary (not CLI/GHCND)
                    "raw_payload": {
                        "tmax_cf6_f": sanitize_value(row.get("tmax_cf6_f")),
                        "tmax_ads_f": sanitize_value(row.get("tmax_ads_f")),
                        "delta_ads_minus_cf6": sanitize_value(row.get("delta_ads_minus_cf6")),
                    },
                }

                # Upsert to database
                upsert_settlement(session, settlement_data)

                # Track by source
                stats["by_source"][db_source] = stats["by_source"].get(db_source, 0) + 1

                # Progress logging and commit
                if (idx + 1) % 500 == 0:
                    session.commit()
                    logger.info(f"Processed {idx + 1}/{len(df)} records...")

            except Exception as e:
                session.rollback()  # Rollback the failed transaction
                logger.error(f"Error loading {row['city']} on {row['date_local']}: {e}")
                stats["errors"] += 1
                continue

        # Final commit
        session.commit()
        logger.info("All records committed to database")

    return stats


def print_loading_report(stats: dict) -> None:
    """
    Print human-readable loading report.

    Args:
        stats: Statistics dict from load_settlements_from_csv()
    """
    print("\n" + "="*60)
    print("SETTLEMENT LOADING REPORT")
    print("="*60)

    print(f"\nTotal records loaded: {stats['total']}")
    print(f"Errors: {stats['errors']}")
    print(f"Successful: {stats['total'] - stats['errors']}")

    print(f"\nBy database source:")
    for source, count in sorted(stats['by_source'].items()):
        print(f"  {source}: {count}")

    print("="*60 + "\n")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Load reconciled settlement data to database"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to settlements_reconciled.csv",
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Skip deleting existing records in range (append mode)",
    )

    args = parser.parse_args()

    print(f"\nLoading settlements from: {args.csv}")
    print()

    # Load data
    stats = load_settlements_from_csv(
        args.csv,
        delete_range=not args.no_delete,
    )

    # Print report
    print_loading_report(stats)
