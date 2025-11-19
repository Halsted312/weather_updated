"""add_pmf_minute_table

Revision ID: a1cd1f3d8e4d
Revises: e19ccb99e1df
Create Date: 2025-11-19 19:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "a1cd1f3d8e4d"
down_revision: Union[str, Sequence[str], None] = "e19ccb99e1df"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS pmf")
    op.create_table(
        'minute',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('city', sa.Text(), nullable=False),
        sa.Column('series_ticker', sa.Text(), nullable=False),
        sa.Column('event_ticker', sa.Text(), nullable=False),
        sa.Column('market_ticker', sa.Text(), nullable=False),
        sa.Column('ts_utc', sa.DateTime(timezone=True), nullable=False),
        sa.Column('ts_local', sa.DateTime(timezone=True), nullable=False),
        sa.Column('local_date', sa.Date(), nullable=False),
        sa.Column('floor_strike', sa.Numeric(), nullable=True),
        sa.Column('cap_strike', sa.Numeric(), nullable=True),
        sa.Column('strike_type', sa.Text(), nullable=True),
        sa.Column('m_run_temp_f', sa.Numeric(), nullable=True),
        sa.Column('p_wx', sa.Numeric(), nullable=False),
        sa.Column('hazard_next_5m', sa.Numeric(), nullable=True),
        sa.Column('hazard_next_60m', sa.Numeric(), nullable=True),
        sa.Column('mc_version', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        schema='pmf'
    )
    op.create_index('idx_pmf_minute_city_ts', 'minute', ['city', 'ts_utc'], schema='pmf')
    op.create_index('idx_pmf_minute_market_ts', 'minute', ['market_ticker', 'ts_utc'], unique=True, schema='pmf')

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS wx.mc_params (
            city text PRIMARY KEY,
            rho double precision NOT NULL,
            sigma_buckets jsonb NOT NULL,
            baseline jsonb,
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS wx.mc_params")
    op.drop_index('idx_pmf_minute_market_ts', table_name='minute', schema='pmf')
    op.drop_index('idx_pmf_minute_city_ts', table_name='minute', schema='pmf')
    op.drop_table('minute', schema='pmf')
