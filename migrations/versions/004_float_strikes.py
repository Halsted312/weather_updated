"""Change strike columns from SMALLINT to DOUBLE PRECISION for float buckets.

Revision ID: 004
Revises: 003
Create Date: 2025-11-26

Kalshi now uses float bucket formats (B55.5, B57.5) with 2F width instead of
integer buckets (B50, B55) with 5F width. This migration updates strike columns
to support fractional values.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Convert strike columns in kalshi.markets to DOUBLE PRECISION
    op.alter_column(
        "markets",
        "floor_strike",
        type_=sa.Float(),
        existing_type=sa.SmallInteger(),
        schema="kalshi",
    )
    op.alter_column(
        "markets",
        "cap_strike",
        type_=sa.Float(),
        existing_type=sa.SmallInteger(),
        schema="kalshi",
    )

    # Convert strike columns in wx.settlement to DOUBLE PRECISION
    op.alter_column(
        "settlement",
        "settled_floor_strike",
        type_=sa.Float(),
        existing_type=sa.SmallInteger(),
        schema="wx",
    )
    op.alter_column(
        "settlement",
        "settled_cap_strike",
        type_=sa.Float(),
        existing_type=sa.SmallInteger(),
        schema="wx",
    )


def downgrade() -> None:
    # Revert wx.settlement strike columns to SMALLINT
    op.alter_column(
        "settlement",
        "settled_cap_strike",
        type_=sa.SmallInteger(),
        existing_type=sa.Float(),
        schema="wx",
    )
    op.alter_column(
        "settlement",
        "settled_floor_strike",
        type_=sa.SmallInteger(),
        existing_type=sa.Float(),
        schema="wx",
    )

    # Revert kalshi.markets strike columns to SMALLINT
    op.alter_column(
        "markets",
        "cap_strike",
        type_=sa.SmallInteger(),
        existing_type=sa.Float(),
        schema="kalshi",
    )
    op.alter_column(
        "markets",
        "floor_strike",
        type_=sa.SmallInteger(),
        existing_type=sa.Float(),
        schema="kalshi",
    )
