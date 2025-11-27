"""Add hourly forecast snapshot table for 72-hour forecast curves.

Revision ID: 005
Revises: 004
Create Date: 2025-11-26

Stores hourly forecast data (72 hours / 3 days) from Visual Crossing
for ML analysis of temperature curves and trend detection.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create hourly forecast snapshot table
    op.create_table(
        "forecast_snapshot_hourly",
        sa.Column("city", sa.Text(), nullable=False),
        sa.Column("target_hour_local", sa.DateTime(), nullable=False),  # Local time
        sa.Column("target_hour_epoch", sa.BigInteger(), nullable=False),  # UTC epoch
        sa.Column("basis_date", sa.Date(), nullable=False),  # Local date forecast made
        sa.Column("lead_hours", sa.Integer(), nullable=False),  # 0-71
        sa.Column("provider", sa.Text(), server_default="visualcrossing"),
        sa.Column("tz_name", sa.Text(), nullable=False),  # IANA timezone
        # Forecast values
        sa.Column("temp_fcst_f", sa.Float(), nullable=True),
        sa.Column("feelslike_fcst_f", sa.Float(), nullable=True),
        sa.Column("humidity_fcst", sa.Float(), nullable=True),
        sa.Column("precip_fcst_in", sa.Float(), nullable=True),
        sa.Column("precip_prob_fcst", sa.Float(), nullable=True),
        sa.Column("windspeed_fcst_mph", sa.Float(), nullable=True),
        sa.Column("conditions_fcst", sa.Text(), nullable=True),
        # Metadata
        sa.Column("raw_json", JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
        ),
        # Primary key
        sa.PrimaryKeyConstraint("city", "target_hour_local", "basis_date"),
        schema="wx",
    )

    # Create index on basis_date for efficient daily queries
    op.create_index(
        "ix_forecast_snapshot_hourly_basis_date",
        "forecast_snapshot_hourly",
        ["basis_date"],
        schema="wx",
    )

    # Create index on lead_hours for filtering by forecast horizon
    op.create_index(
        "ix_forecast_snapshot_hourly_lead_hours",
        "forecast_snapshot_hourly",
        ["lead_hours"],
        schema="wx",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_forecast_snapshot_hourly_lead_hours",
        table_name="forecast_snapshot_hourly",
        schema="wx",
    )
    op.drop_index(
        "ix_forecast_snapshot_hourly_basis_date",
        table_name="forecast_snapshot_hourly",
        schema="wx",
    )
    op.drop_table("forecast_snapshot_hourly", schema="wx")
