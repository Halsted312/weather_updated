"""add_cross_bracket_features_view

Revision ID: c8d5b249be61
Revises: bf1f3921ab2f
Create Date: 2025-11-19 18:02:00

"""
from typing import Sequence, Union

from alembic import op


revision: str = "c8d5b249be61"
down_revision: Union[str, Sequence[str], None] = "bf1f3921ab2f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


VIEW_SQL = """
CREATE MATERIALIZED VIEW feat.minute_panel_neighbors AS
WITH ordered AS (
    SELECT
        b.*,
        ROW_NUMBER() OVER (
            PARTITION BY b.series_ticker, b.local_date
            ORDER BY COALESCE(b.floor_strike, -1), COALESCE(b.cap_strike, 1e9)
        ) AS bracket_rank
    FROM feat.minute_panel_base b
)
SELECT
    cur.city,
    cur.series_ticker,
    cur.event_ticker,
    cur.market_ticker,
    cur.ts_utc,
    cur.ts_local,
    cur.local_date,
    cur.close_c,
    cur.mid_prob,
    cur.mid_velocity,
    cur.mid_acceleration,
    cur.clv,
    cur.volume,
    cur.volume_delta,
    cur.strike_type,
    cur.floor_strike,
    cur.cap_strike,
    left_neighbor.mid_prob AS left_mid_prob,
    right_neighbor.mid_prob AS right_mid_prob,
    left_neighbor.mid_velocity AS left_mid_velocity,
    right_neighbor.mid_velocity AS right_mid_velocity,
    left_neighbor.mid_acceleration AS left_mid_acceleration,
    right_neighbor.mid_acceleration AS right_mid_acceleration,
    (cur.mid_prob - left_neighbor.mid_prob) AS mid_prob_left_diff,
    (cur.mid_prob - right_neighbor.mid_prob) AS mid_prob_right_diff,
    (cur.mid_velocity - left_neighbor.mid_velocity) AS velocity_left_diff,
    (cur.mid_velocity - right_neighbor.mid_velocity) AS velocity_right_diff,
    (cur.mid_acceleration - left_neighbor.mid_acceleration) AS acceleration_left_diff,
    (cur.mid_acceleration - right_neighbor.mid_acceleration) AS acceleration_right_diff,
    (COALESCE(left_neighbor.mid_prob, 0) + cur.mid_prob + COALESCE(right_neighbor.mid_prob, 0)) AS triad_mass
FROM ordered cur
LEFT JOIN ordered left_neighbor
       ON left_neighbor.series_ticker = cur.series_ticker
      AND left_neighbor.ts_utc = cur.ts_utc
      AND left_neighbor.bracket_rank = cur.bracket_rank - 1
LEFT JOIN ordered right_neighbor
       ON right_neighbor.series_ticker = cur.series_ticker
      AND right_neighbor.ts_utc = cur.ts_utc
      AND right_neighbor.bracket_rank = cur.bracket_rank + 1;
"""

DROP_SQL = "DROP MATERIALIZED VIEW IF EXISTS feat.minute_panel_neighbors"


def upgrade() -> None:
    op.execute(VIEW_SQL)
    op.execute("CREATE UNIQUE INDEX idx_feat_neighbors_market_ts ON feat.minute_panel_neighbors (market_ticker, ts_utc)")
    op.execute("CREATE INDEX idx_feat_neighbors_city_ts ON feat.minute_panel_neighbors (city, ts_utc)")
    op.execute("CREATE INDEX idx_feat_neighbors_series_rank_ts ON feat.minute_panel_neighbors (series_ticker, local_date, ts_utc)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_feat_neighbors_series_rank_ts")
    op.execute("DROP INDEX IF EXISTS idx_feat_neighbors_city_ts")
    op.execute("DROP INDEX IF EXISTS idx_feat_neighbors_market_ts")
    op.execute(DROP_SQL)
