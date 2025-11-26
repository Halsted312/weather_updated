"""add_settlement_tables_nws_cli_cf6

Revision ID: dab74fb952df
Revises: d8fedec351a7
Create Date: 2025-11-12 14:31:27.872607

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dab74fb952df'
down_revision: Union[str, Sequence[str], None] = 'd8fedec351a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create wx.settlement table for NWS CLI/CF6/GHCND settlement data
    op.execute("""
        CREATE TABLE IF NOT EXISTS wx.settlement (
            city            text NOT NULL,
            icao            text NOT NULL,
            issuedby        text NOT NULL,
            date_local      date NOT NULL,
            tmax_f          double precision NOT NULL,
            source          text NOT NULL,
            is_preliminary  boolean NOT NULL DEFAULT false,
            retrieved_at    timestamptz NOT NULL DEFAULT now(),
            raw_payload     text NOT NULL,
            PRIMARY KEY (city, date_local, source)
        )
    """)

    # Add settlement columns to markets table
    op.execute("""
        ALTER TABLE markets
            ADD COLUMN IF NOT EXISTS settlement_tmax_f double precision,
            ADD COLUMN IF NOT EXISTS settlement_source text,
            ADD COLUMN IF NOT EXISTS settlement_verified_at timestamptz
    """)

    # Add index for faster lookups by city and date
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_settlement_city_date
        ON wx.settlement (city, date_local)
    """)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop settlement columns from markets
    op.execute("""
        ALTER TABLE markets
            DROP COLUMN IF EXISTS settlement_tmax_f,
            DROP COLUMN IF EXISTS settlement_source,
            DROP COLUMN IF EXISTS settlement_verified_at
    """)

    # Drop wx.settlement table
    op.execute("DROP TABLE IF EXISTS wx.settlement")
