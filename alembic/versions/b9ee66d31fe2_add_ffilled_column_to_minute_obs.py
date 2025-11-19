"""add_ffilled_column_to_minute_obs

Revision ID: b9ee66d31fe2
Revises: ca1e230aad1c
Create Date: 2025-11-14 20:23:03.109401

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b9ee66d31fe2'
down_revision: Union[str, Sequence[str], None] = 'ca1e230aad1c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Add ffilled column to wx.minute_obs."""
    # Add ffilled column with default FALSE (real observations, not forward-filled)
    op.add_column(
        'minute_obs',
        sa.Column('ffilled', sa.Boolean(), nullable=False, server_default='false'),
        schema='wx'
    )


def downgrade() -> None:
    """Downgrade schema: Remove ffilled column."""
    op.drop_column('minute_obs', 'ffilled', schema='wx')
