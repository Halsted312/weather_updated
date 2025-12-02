"""add_source_column_to_candles

Revision ID: 061c683440ed
Revises: 009
Create Date: 2025-11-30 16:22:12.797345

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '061c683440ed'
down_revision: Union[str, None] = '009'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add 'source' column to kalshi.candles_1m to track data provenance.

    The source column distinguishes between:
    - 'api_event': From Kalshi Event Candlesticks API (efficient, one call per event)
    - 'trades': Aggregated from individual trades (fallback/audit)

    Also updates primary key to (ticker, bucket_start, source) to allow
    both sources for the same ticker/time.
    """
    # Step 1: Add source column with default value
    op.execute("""
        ALTER TABLE kalshi.candles_1m
        ADD COLUMN source TEXT NOT NULL DEFAULT 'api_event'
    """)

    # Step 2: Drop existing primary key constraint
    op.execute("""
        ALTER TABLE kalshi.candles_1m
        DROP CONSTRAINT candles_1m_pkey
    """)

    # Step 3: Add new composite primary key
    op.execute("""
        ALTER TABLE kalshi.candles_1m
        ADD PRIMARY KEY (ticker, bucket_start, source)
    """)


def downgrade() -> None:
    """Revert source column changes."""
    # Drop composite PK
    op.execute("""
        ALTER TABLE kalshi.candles_1m
        DROP CONSTRAINT candles_1m_pkey
    """)

    # Restore original PK
    op.execute("""
        ALTER TABLE kalshi.candles_1m
        ADD PRIMARY KEY (ticker, bucket_start)
    """)

    # Drop source column
    op.execute("""
        ALTER TABLE kalshi.candles_1m
        DROP COLUMN source
    """)
