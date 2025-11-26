"""add_pmf_fusion_columns

Revision ID: cd4499b7bfcc
Revises: e48082827515
Create Date: 2025-11-19 20:45:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "cd4499b7bfcc"
down_revision: Union[str, Sequence[str], None] = "e48082827515"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('minute', sa.Column('p_mkt', sa.Numeric(), nullable=True), schema='pmf')
    op.add_column('minute', sa.Column('p_fused', sa.Numeric(), nullable=True), schema='pmf')


def downgrade() -> None:
    op.drop_column('minute', 'p_fused', schema='pmf')
    op.drop_column('minute', 'p_mkt', schema='pmf')
