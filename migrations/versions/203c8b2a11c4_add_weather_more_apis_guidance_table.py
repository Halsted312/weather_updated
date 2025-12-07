"""add_weather_more_apis_guidance_table

Revision ID: 203c8b2a11c4
Revises: 010
Create Date: 2025-12-06 02:01:17.498789

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '203c8b2a11c4'
down_revision: Union[str, None] = '010'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Create wx.weather_more_apis_guidance table for NOAA model guidance (NBM, HRRR, NDFD).

    This table stores scalar summaries (peak window max temps) per model run,
    avoiding minute-level explosion.
    """
    op.create_table(
        'weather_more_apis_guidance',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('city_id', sa.String(20), nullable=False),
        sa.Column('target_date', sa.Date(), nullable=False),
        sa.Column('model', sa.String(10), nullable=False),
        sa.Column('run_datetime_utc', sa.DateTime(timezone=True), nullable=False),
        sa.Column('peak_window_max_f', sa.Numeric(5, 2), nullable=True),
        sa.Column('timezone', sa.String(50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.CheckConstraint("model IN ('nbm', 'hrrr', 'ndfd')", name='ck_weather_more_apis_guidance_model'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('city_id', 'target_date', 'model', 'run_datetime_utc',
                           name='uq_weather_more_apis_guidance_key'),
        schema='wx'
    )

    # Index for feature loading queries (city, target_date, model)
    op.create_index(
        'ix_weather_more_apis_guidance_city_target_model',
        'weather_more_apis_guidance',
        ['city_id', 'target_date', 'model'],
        schema='wx'
    )

    # Index for temporal queries (finding latest/previous runs)
    op.create_index(
        'ix_weather_more_apis_guidance_run_time',
        'weather_more_apis_guidance',
        ['run_datetime_utc'],
        schema='wx'
    )


def downgrade() -> None:
    """Drop wx.weather_more_apis_guidance table and indexes."""
    op.drop_index('ix_weather_more_apis_guidance_run_time',
                  table_name='weather_more_apis_guidance', schema='wx')
    op.drop_index('ix_weather_more_apis_guidance_city_target_model',
                  table_name='weather_more_apis_guidance', schema='wx')
    op.drop_table('weather_more_apis_guidance', schema='wx')
