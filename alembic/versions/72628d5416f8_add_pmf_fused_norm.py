"""add_pmf_fused_norm

Revision ID: 72628d5416f8
Revises: cd4499b7bfcc
Create Date: 2025-11-19 21:15:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "72628d5416f8"
down_revision: Union[str, Sequence[str], None] = "cd4499b7bfcc"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('minute', sa.Column('p_fused_norm', sa.Numeric(), nullable=True), schema='pmf')


def downgrade() -> None:
    op.drop_column('minute', 'p_fused_norm', schema='pmf')
