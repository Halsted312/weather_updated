"""add_feat_panel_with_pmf_view

Revision ID: e48082827515
Revises: a1cd1f3d8e4d
Create Date: 2025-11-19 20:20:00

"""
from typing import Sequence, Union

from alembic import op


revision: str = "e48082827515"
down_revision: Union[str, Sequence[str], None] = "a1cd1f3d8e4d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


VIEW_SQL = """
CREATE OR REPLACE VIEW feat.minute_panel_full AS
SELECT
    w.*, 
    pm.p_wx,
    pm.hazard_next_5m,
    pm.hazard_next_60m,
    pm.mc_version
FROM feat.minute_panel_with_weather w
LEFT JOIN pmf.minute pm
  ON pm.market_ticker = w.market_ticker
 AND pm.ts_utc = w.ts_utc
"""


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS feat")
    op.execute(VIEW_SQL)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS feat.minute_panel_full")
