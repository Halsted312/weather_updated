"""Expand wx.settlement for multi-source TMAX and Kalshi bucket tracking.

Revision ID: 003
Revises: 002
Create Date: 2025-11-26

Adds:
- tmax_iem_f, tmax_ncei_f for additional temperature sources
- Per-source raw payloads (raw_payload_cli, raw_payload_cf6, raw_payload_iem, raw_payload_ncei)
- Kalshi settlement bucket tracking (settled_ticker, settled_bucket_type, etc.)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add additional TMAX source columns
    op.add_column(
        "settlement",
        sa.Column("tmax_iem_f", sa.SmallInteger(), nullable=True),
        schema="wx",
    )
    op.add_column(
        "settlement",
        sa.Column("tmax_ncei_f", sa.SmallInteger(), nullable=True),
        schema="wx",
    )

    # Add per-source raw payload columns
    op.add_column(
        "settlement",
        sa.Column("raw_payload_cli", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        schema="wx",
    )
    op.add_column(
        "settlement",
        sa.Column("raw_payload_cf6", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        schema="wx",
    )
    op.add_column(
        "settlement",
        sa.Column("raw_payload_iem", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        schema="wx",
    )
    op.add_column(
        "settlement",
        sa.Column("raw_payload_ncei", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        schema="wx",
    )

    # Add Kalshi settlement bucket tracking columns
    op.add_column(
        "settlement",
        sa.Column("settled_ticker", sa.Text(), nullable=True),
        schema="wx",
    )
    op.add_column(
        "settlement",
        sa.Column("settled_bucket_type", sa.Text(), nullable=True),
        schema="wx",
    )
    op.add_column(
        "settlement",
        sa.Column("settled_floor_strike", sa.SmallInteger(), nullable=True),
        schema="wx",
    )
    op.add_column(
        "settlement",
        sa.Column("settled_cap_strike", sa.SmallInteger(), nullable=True),
        schema="wx",
    )
    op.add_column(
        "settlement",
        sa.Column("settled_bucket_label", sa.Text(), nullable=True),
        schema="wx",
    )

    # Create index on settled_ticker for lookups
    op.create_index(
        "ix_settlement_settled_ticker",
        "settlement",
        ["settled_ticker"],
        schema="wx",
    )


def downgrade() -> None:
    # Drop index
    op.drop_index("ix_settlement_settled_ticker", table_name="settlement", schema="wx")

    # Drop Kalshi settlement bucket columns
    op.drop_column("settlement", "settled_bucket_label", schema="wx")
    op.drop_column("settlement", "settled_cap_strike", schema="wx")
    op.drop_column("settlement", "settled_floor_strike", schema="wx")
    op.drop_column("settlement", "settled_bucket_type", schema="wx")
    op.drop_column("settlement", "settled_ticker", schema="wx")

    # Drop per-source raw payload columns
    op.drop_column("settlement", "raw_payload_ncei", schema="wx")
    op.drop_column("settlement", "raw_payload_iem", schema="wx")
    op.drop_column("settlement", "raw_payload_cf6", schema="wx")
    op.drop_column("settlement", "raw_payload_cli", schema="wx")

    # Drop additional TMAX source columns
    op.drop_column("settlement", "tmax_ncei_f", schema="wx")
    op.drop_column("settlement", "tmax_iem_f", schema="wx")
