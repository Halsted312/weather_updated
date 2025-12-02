"""add_kalshi_market_events

Revision ID: 8144dca12469
Revises: 8edf8004773c
Create Date: 2025-12-01 00:23:08.790650

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8144dca12469'
down_revision: Union[str, None] = '8edf8004773c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create market_events table
    op.create_table(
        'market_events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('ticker', sa.Text(), nullable=False),
        sa.Column('event_type', sa.Text(), nullable=False),
        sa.Column('detected_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('details', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint(
            "event_type IN ('market_open', 'status_change', 'trading_started', 'market_close', 'settled')",
            name='valid_event_type'
        ),
        schema='kalshi'
    )
    # Create indexes
    op.create_index('idx_market_events_ticker', 'market_events', ['ticker'], schema='kalshi')
    op.create_index('idx_market_events_detected_at', 'market_events', ['detected_at'], schema='kalshi')


def downgrade() -> None:
    op.drop_index('idx_market_events_detected_at', table_name='market_events', schema='kalshi')
    op.drop_index('idx_market_events_ticker', table_name='market_events', schema='kalshi')
    op.drop_table('market_events', schema='kalshi')
