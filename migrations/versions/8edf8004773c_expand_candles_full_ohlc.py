"""expand_candles_full_ohlc

Revision ID: 8edf8004773c
Revises: 061c683440ed
Create Date: 2025-11-30 17:19:02.639714

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8edf8004773c'
down_revision: Union[str, None] = '061c683440ed'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Expand kalshi.candles_1m to capture full bid/ask OHLC from Kalshi API.

    Changes:
    1. Rename existing columns for semantic clarity
    2. Add full YES bid/ask OHLC (open/high/low)
    3. Add trade price statistics (mean/previous/min/max)
    4. Add period_minutes for future hourly/daily aggregations

    Note: Existing data will have NULL for new columns.
    Recommend TRUNCATE before re-backfill for clean data.
    """
    # Step 1: Rename existing columns for clarity
    op.execute("ALTER TABLE kalshi.candles_1m RENAME COLUMN open_c TO trade_open")
    op.execute("ALTER TABLE kalshi.candles_1m RENAME COLUMN high_c TO trade_high")
    op.execute("ALTER TABLE kalshi.candles_1m RENAME COLUMN low_c TO trade_low")
    op.execute("ALTER TABLE kalshi.candles_1m RENAME COLUMN close_c TO trade_close")
    op.execute("ALTER TABLE kalshi.candles_1m RENAME COLUMN yes_bid_c TO yes_bid_close")
    op.execute("ALTER TABLE kalshi.candles_1m RENAME COLUMN yes_ask_c TO yes_ask_close")

    # Step 2: Add YES bid open/high/low
    op.execute("ALTER TABLE kalshi.candles_1m ADD COLUMN yes_bid_open SMALLINT")
    op.execute("ALTER TABLE kalshi.candles_1m ADD COLUMN yes_bid_high SMALLINT")
    op.execute("ALTER TABLE kalshi.candles_1m ADD COLUMN yes_bid_low SMALLINT")

    # Step 3: Add YES ask open/high/low
    op.execute("ALTER TABLE kalshi.candles_1m ADD COLUMN yes_ask_open SMALLINT")
    op.execute("ALTER TABLE kalshi.candles_1m ADD COLUMN yes_ask_high SMALLINT")
    op.execute("ALTER TABLE kalshi.candles_1m ADD COLUMN yes_ask_low SMALLINT")

    # Step 4: Add trade price statistics
    op.execute("ALTER TABLE kalshi.candles_1m ADD COLUMN trade_mean SMALLINT")
    op.execute("ALTER TABLE kalshi.candles_1m ADD COLUMN trade_previous SMALLINT")
    op.execute("ALTER TABLE kalshi.candles_1m ADD COLUMN trade_min SMALLINT")
    op.execute("ALTER TABLE kalshi.candles_1m ADD COLUMN trade_max SMALLINT")

    # Step 5: Add period_minutes for future-proofing
    op.execute("ALTER TABLE kalshi.candles_1m ADD COLUMN period_minutes SMALLINT NOT NULL DEFAULT 1")


def downgrade() -> None:
    """Revert candle schema expansion."""
    # Drop added columns
    op.execute("ALTER TABLE kalshi.candles_1m DROP COLUMN period_minutes")
    op.execute("ALTER TABLE kalshi.candles_1m DROP COLUMN trade_max")
    op.execute("ALTER TABLE kalshi.candles_1m DROP COLUMN trade_min")
    op.execute("ALTER TABLE kalshi.candles_1m DROP COLUMN trade_previous")
    op.execute("ALTER TABLE kalshi.candles_1m DROP COLUMN trade_mean")
    op.execute("ALTER TABLE kalshi.candles_1m DROP COLUMN yes_ask_low")
    op.execute("ALTER TABLE kalshi.candles_1m DROP COLUMN yes_ask_high")
    op.execute("ALTER TABLE kalshi.candles_1m DROP COLUMN yes_ask_open")
    op.execute("ALTER TABLE kalshi.candles_1m DROP COLUMN yes_bid_low")
    op.execute("ALTER TABLE kalshi.candles_1m DROP COLUMN yes_bid_high")
    op.execute("ALTER TABLE kalshi.candles_1m DROP COLUMN yes_bid_open")

    # Restore original column names
    op.execute("ALTER TABLE kalshi.candles_1m RENAME COLUMN yes_ask_close TO yes_ask_c")
    op.execute("ALTER TABLE kalshi.candles_1m RENAME COLUMN yes_bid_close TO yes_bid_c")
    op.execute("ALTER TABLE kalshi.candles_1m RENAME COLUMN trade_close TO close_c")
    op.execute("ALTER TABLE kalshi.candles_1m RENAME COLUMN trade_low TO low_c")
    op.execute("ALTER TABLE kalshi.candles_1m RENAME COLUMN trade_high TO high_c")
    op.execute("ALTER TABLE kalshi.candles_1m RENAME COLUMN trade_open TO open_c")
