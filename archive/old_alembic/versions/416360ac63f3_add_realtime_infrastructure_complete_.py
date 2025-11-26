"""add_realtime_infrastructure_complete_flag_and_rt_signals

Revision ID: 416360ac63f3
Revises: 73be298978ae
Create Date: 2025-11-16 07:46:23.614538

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '416360ac63f3'
down_revision: Union[str, Sequence[str], None] = '73be298978ae'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Upgrade schema: Add realtime infrastructure.

    1. Add 'complete' boolean to candles table (marks candles with all data available)
    2. Create rt_signals table for real-time trading signals
    """
    # Add 'complete' flag to candles table
    # This marks whether a candle has all required data (price + weather) for model inference
    op.add_column(
        'candles',
        sa.Column('complete', sa.Boolean(), nullable=False, server_default='false')
    )

    # Index on complete flag for fast filtering in rt_loop
    op.create_index(
        'idx_candles_complete',
        'candles',
        ['complete', 'timestamp'],
        unique=False
    )

    # Create rt_signals table for real-time trading signals
    op.create_table(
        'rt_signals',
        sa.Column('ts_utc', sa.DateTime(), nullable=False, comment='Signal generation timestamp (UTC)'),
        sa.Column('market_ticker', sa.String(100), nullable=False, comment='Market ticker'),
        sa.Column('city', sa.String(50), nullable=False, comment='City name'),
        sa.Column('bracket', sa.String(20), nullable=False, comment='Bracket type: between, greater, less'),

        # Probabilities (0-100 scale for consistency with prices in cents)
        sa.Column('p_model', sa.Float(), nullable=False, comment='Model probability (0-1)'),
        sa.Column('p_market', sa.Float(), nullable=False, comment='Market-implied probability (0-1)'),
        sa.Column('p_blend', sa.Float(), nullable=False, comment='Blended probability via opinion pooling (0-1)'),

        # Trading edge and sizing
        sa.Column('edge_cents', sa.Float(), nullable=False, comment='Expected edge in cents after fees'),
        sa.Column('kelly_fraction', sa.Float(), nullable=True, comment='Kelly-optimal fraction'),
        sa.Column('size_fraction', sa.Float(), nullable=True, comment='Actual position size (fractional Kelly)'),

        # Market conditions
        sa.Column('spread_cents', sa.Integer(), nullable=True, comment='Bid-ask spread in cents'),
        sa.Column('minutes_to_close', sa.Integer(), nullable=False, comment='Minutes until market close'),

        # Model provenance
        sa.Column('model_id', sa.String(200), nullable=False, comment='Model identifier (city_bracket_window)'),
        sa.Column('wf_window', sa.String(100), nullable=False, comment='Walk-forward window (e.g., win_20250802_20250919)'),

        # Audit
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),

        # Primary key: one signal per market per timestamp
        sa.PrimaryKeyConstraint('ts_utc', 'market_ticker'),

        # Foreign key to markets
        sa.ForeignKeyConstraint(['market_ticker'], ['markets.ticker'])
    )

    # Indexes for rt_signals queries
    op.create_index('idx_rt_signals_city_bracket', 'rt_signals', ['city', 'bracket'], unique=False)
    op.create_index('idx_rt_signals_ts', 'rt_signals', ['ts_utc'], unique=False)
    op.create_index('idx_rt_signals_edge', 'rt_signals', ['edge_cents'], unique=False)


def downgrade() -> None:
    """
    Downgrade schema: Remove realtime infrastructure.
    """
    # Drop rt_signals table
    op.drop_index('idx_rt_signals_edge', table_name='rt_signals')
    op.drop_index('idx_rt_signals_ts', table_name='rt_signals')
    op.drop_index('idx_rt_signals_city_bracket', table_name='rt_signals')
    op.drop_table('rt_signals')

    # Remove complete flag from candles
    op.drop_index('idx_candles_complete', table_name='candles')
    op.drop_column('candles', 'complete')
