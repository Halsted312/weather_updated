"""add_stations_column_to_minute_obs

Revision ID: 73be298978ae
Revises: b9ee66d31fe2
Create Date: 2025-11-15 02:08:07.292990

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '73be298978ae'
down_revision: Union[str, Sequence[str], None] = 'b9ee66d31fe2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: Add stations column to wx.minute_obs."""
    # Add stations column for tracking which station VC used for each minute
    op.add_column(
        'minute_obs',
        sa.Column('stations', sa.String(50), nullable=True),
        schema='wx'
    )


def downgrade() -> None:
    """Downgrade schema: Remove stations column."""
    op.drop_column('minute_obs', 'stations', schema='wx')
