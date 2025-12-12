#!/usr/bin/env python3
"""
Migration: Add source column to kalshi.candles_1m table.

This migration adds support for dual storage of candlesticks from:
- 'api_event': Kalshi Event Candlesticks API (primary)
- 'trades': Aggregated from individual trades (fallback/audit)

The source column becomes part of the composite primary key:
PRIMARY KEY (ticker, bucket_start, source)

Usage:
    python scripts/migrate_candles_add_source.py
    python scripts/migrate_candles_add_source.py --dry-run
"""

import argparse
import logging
import sys

from sqlalchemy import text

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.db import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_column_exists(conn, schema: str, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    result = conn.execute(
        text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = :schema
                AND table_name = :table
                AND column_name = :column
            )
        """),
        {"schema": schema, "table": table, "column": column}
    )
    return result.scalar()


def get_row_count(conn, schema: str, table: str) -> int:
    """Get row count for a table."""
    result = conn.execute(
        text(f"SELECT COUNT(*) FROM {schema}.{table}")
    )
    return result.scalar()


def run_migration(dry_run: bool = False):
    """Run the migration to add source column."""
    engine = get_engine()

    with engine.connect() as conn:
        # Check if column already exists
        if check_column_exists(conn, "kalshi", "candles_1m", "source"):
            logger.info("Column 'source' already exists in kalshi.candles_1m - skipping migration")
            return

        # Get current row count
        row_count = get_row_count(conn, "kalshi", "candles_1m")
        logger.info(f"Current row count in kalshi.candles_1m: {row_count}")

        if dry_run:
            logger.info("DRY RUN - would execute the following SQL:")
            logger.info("  1. Add 'source' column with default 'trades'")
            logger.info("  2. Drop existing primary key constraint")
            logger.info("  3. Add new composite primary key (ticker, bucket_start, source)")
            logger.info("  4. Create index on (ticker, source)")
            return

        # Execute migration
        logger.info("Starting migration...")

        # Step 1: Add source column with default value
        logger.info("Step 1: Adding source column...")
        conn.execute(text("""
            ALTER TABLE kalshi.candles_1m
            ADD COLUMN source VARCHAR(20) NOT NULL DEFAULT 'trades'
        """))
        conn.commit()

        # Step 2: Drop existing primary key (we need to know the constraint name)
        logger.info("Step 2: Dropping existing primary key...")
        # PostgreSQL automatically names PK constraints as {table}_pkey
        conn.execute(text("""
            ALTER TABLE kalshi.candles_1m
            DROP CONSTRAINT candles_1m_pkey
        """))
        conn.commit()

        # Step 3: Add new composite primary key
        logger.info("Step 3: Adding new composite primary key...")
        conn.execute(text("""
            ALTER TABLE kalshi.candles_1m
            ADD PRIMARY KEY (ticker, bucket_start, source)
        """))
        conn.commit()

        # Step 4: Create helpful index
        logger.info("Step 4: Creating index on (ticker, source)...")
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_candles_1m_ticker_source
            ON kalshi.candles_1m (ticker, source)
        """))
        conn.commit()

        # Verify
        final_count = get_row_count(conn, "kalshi", "candles_1m")
        logger.info(f"Migration complete! Final row count: {final_count}")

        if row_count > 0:
            logger.info(f"All {row_count} existing rows have been assigned source='trades'")


def main():
    parser = argparse.ArgumentParser(description="Add source column to kalshi.candles_1m")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    run_migration(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
