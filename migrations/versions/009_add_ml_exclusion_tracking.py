"""Add ML exclusion tracking to vc_minute_weather.

Revision ID: 009
Revises: 008
Create Date: 2025-11-28

Adds exclude_from_ml boolean column to flag days with significant data quality
issues (≥5% NULL temps) that should be excluded from ML training.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add ML exclusion flag
    op.add_column(
        "vc_minute_weather",
        sa.Column("exclude_from_ml", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")),
        schema="wx",
    )

    # Add index for ML queries (WHERE exclude_from_ml = FALSE)
    op.create_index(
        "ix_vc_minute_weather_ml_usable",
        "vc_minute_weather",
        ["vc_location_id", "datetime_utc"],
        schema="wx",
        postgresql_where=sa.text("exclude_from_ml = FALSE"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_vc_minute_weather_ml_usable",
        table_name="vc_minute_weather",
        schema="wx",
    )
    op.drop_column("vc_minute_weather", "exclude_from_ml", schema="wx")
