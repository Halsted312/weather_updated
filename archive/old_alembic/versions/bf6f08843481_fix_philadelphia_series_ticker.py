"""fix_philadelphia_series_ticker

Revision ID: bf6f08843481
Revises: 416360ac63f3
Create Date: 2025-11-19 21:38:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'bf6f08843481'
down_revision: Union[str, Sequence[str], None] = '416360ac63f3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Correct the Philadelphia series ticker to match the Kalshi payload."""
    op.execute("""
        UPDATE dim_city
        SET series_ticker = 'KXHIGHPHIL'
        WHERE city = 'philadelphia'
          AND series_ticker = 'KXHIGHPHL'
    """)


def downgrade() -> None:
    """Revert the Philadelphia series ticker change."""
    op.execute("""
        UPDATE dim_city
        SET series_ticker = 'KXHIGHPHL'
        WHERE city = 'philadelphia'
          AND series_ticker = 'KXHIGHPHIL'
    """)
