"""Add meta schema for ingestion checkpoints.

Revision ID: 002
Revises: 001
Create Date: 2025-11-26
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "002"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create meta schema
    op.execute("CREATE SCHEMA IF NOT EXISTS meta")

    # Create ingestion_checkpoint table
    op.create_table(
        "ingestion_checkpoint",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("pipeline_name", sa.Text(), nullable=False),
        sa.Column("city", sa.Text(), nullable=True),
        sa.Column("last_processed_date", sa.Date(), nullable=True),
        sa.Column("last_processed_cursor", sa.Text(), nullable=True),
        sa.Column("last_processed_ticker", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), server_default="running", nullable=False),
        sa.Column("total_processed", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        schema="meta",
    )

    # Create partial unique index for active (running) checkpoints
    op.create_index(
        "ix_checkpoint_active",
        "ingestion_checkpoint",
        ["pipeline_name", "city"],
        unique=True,
        schema="meta",
        postgresql_where=sa.text("status = 'running'"),
    )

    # Create index on status for queries
    op.create_index(
        "ix_checkpoint_status",
        "ingestion_checkpoint",
        ["status"],
        schema="meta",
    )


def downgrade() -> None:
    op.drop_index("ix_checkpoint_status", table_name="ingestion_checkpoint", schema="meta")
    op.drop_index("ix_checkpoint_active", table_name="ingestion_checkpoint", schema="meta")
    op.drop_table("ingestion_checkpoint", schema="meta")
    op.execute("DROP SCHEMA IF EXISTS meta")
