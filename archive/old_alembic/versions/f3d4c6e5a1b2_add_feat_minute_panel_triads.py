"""add_feat_minute_panel_triads

Revision ID: f3d4c6e5a1b2
Revises: e48082827515
Create Date: 2025-11-19 17:30:00

"""
from typing import Sequence, Union

from alembic import op


revision: str = "f3d4c6e5a1b2"
down_revision: Union[str, Sequence[str], None] = "e48082827515"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


VIEW_SQL = """
CREATE OR REPLACE VIEW feat.minute_panel_triads AS
WITH ordered AS (
    SELECT
        w.*,
        ROW_NUMBER() OVER (
            PARTITION BY w.event_ticker, w.local_date, w.ts_utc
            ORDER BY w.floor_strike, w.cap_strike, w.market_ticker
        ) AS bracket_idx,
        COUNT(*) OVER (
            PARTITION BY w.event_ticker, w.local_date, w.ts_utc
        ) AS num_brackets
    FROM feat.minute_panel_with_weather w
),
neighbors AS (
    SELECT
        o.*,
        LAG(o.mid_prob) OVER (
            PARTITION BY o.event_ticker, o.local_date, o.ts_utc
            ORDER BY o.bracket_idx
        ) AS mid_prob_left,
        LEAD(o.mid_prob) OVER (
            PARTITION BY o.event_ticker, o.local_date, o.ts_utc
            ORDER BY o.bracket_idx
        ) AS mid_prob_right,
        LAG(o.mid_velocity) OVER (
            PARTITION BY o.event_ticker, o.local_date, o.ts_utc
            ORDER BY o.bracket_idx
        ) AS mid_velocity_left,
        LEAD(o.mid_velocity) OVER (
            PARTITION BY o.event_ticker, o.local_date, o.ts_utc
            ORDER BY o.bracket_idx
        ) AS mid_velocity_right,
        LAG(o.mid_acceleration) OVER (
            PARTITION BY o.event_ticker, o.local_date, o.ts_utc
            ORDER BY o.bracket_idx
        ) AS mid_acceleration_left,
        LEAD(o.mid_acceleration) OVER (
            PARTITION BY o.event_ticker, o.local_date, o.ts_utc
            ORDER BY o.bracket_idx
        ) AS mid_acceleration_right
    FROM ordered o
)
SELECT
    n.city,
    n.series_ticker,
    n.event_ticker,
    n.market_ticker,
    n.ts_utc,
    n.ts_local,
    n.local_date,
    n.floor_strike,
    n.cap_strike,
    n.strike_type,
    n.bracket_idx,
    n.num_brackets,
    n.close_c,
    n.mid_prob,
    n.mid_velocity,
    n.mid_acceleration,
    n.clv,
    n.volume,
    n.volume_delta,
    n.wx_running_max,
    n.mid_prob_left,
    n.mid_prob_right,
    n.mid_velocity_left,
    n.mid_velocity_right,
    n.mid_acceleration_left,
    n.mid_acceleration_right,
    (n.mid_velocity - n.mid_velocity_left)  AS mid_velocity_left_diff,
    (n.mid_velocity - n.mid_velocity_right) AS mid_velocity_right_diff,
    (n.mid_acceleration - n.mid_acceleration_left)  AS mid_accel_left_diff,
    (n.mid_acceleration - n.mid_acceleration_right) AS mid_accel_right_diff,
    (COALESCE(n.mid_prob_left, 0.0) + n.mid_prob + COALESCE(n.mid_prob_right, 0.0)) AS triad_mass,
    CASE
        WHEN n.mid_acceleration_left IS NOT NULL AND n.mid_acceleration_right IS NOT NULL
            THEN n.mid_acceleration - 0.5 * (n.mid_acceleration_left + n.mid_acceleration_right)
        ELSE 0.0
    END AS ras_accel,
    CASE
        WHEN n.mid_velocity_left IS NOT NULL AND n.mid_velocity_right IS NOT NULL
            THEN n.mid_velocity - 0.5 * (n.mid_velocity_left + n.mid_velocity_right)
        ELSE 0.0
    END AS ras_vel
FROM neighbors n;
"""


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS feat")
    op.execute(VIEW_SQL)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS feat.minute_panel_triads")
