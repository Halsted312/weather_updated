"""Add forward-fill tracking to vc_minute_weather.

Revision ID: 008
Revises: 007
Create Date: 2025-11-28

Adds is_forward_filled boolean column to track temperature values that were
forward-filled due to VC API errors (e.g., -77.9F sentinel values).
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add forward-fill tracking column
    op.add_column(
        "vc_minute_weather",
        sa.Column("is_forward_filled", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")),
        schema="wx",
    )

    # Add index for querying forward-filled records
    op.create_index(
        "ix_vc_minute_weather_forward_filled",
        "vc_minute_weather",
        ["is_forward_filled"],
        schema="wx",
        postgresql_where=sa.text("is_forward_filled = TRUE"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_vc_minute_weather_forward_filled",
        table_name="vc_minute_weather",
        schema="wx",
    )
    op.drop_column("vc_minute_weather", "is_forward_filled", schema="wx")
