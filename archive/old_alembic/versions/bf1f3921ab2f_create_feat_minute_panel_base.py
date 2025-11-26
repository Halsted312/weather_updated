"""create_feat_minute_panel_base

Revision ID: bf1f3921ab2f
Revises: 1c6cf6795c1b
Create Date: 2025-11-19 17:35:00

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "bf1f3921ab2f"
down_revision: Union[str, Sequence[str], None] = "1c6cf6795c1b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


VIEW_SQL = """
CREATE MATERIALIZED VIEW feat.minute_panel_base AS
WITH base AS (
    SELECT
        c.timestamp AS ts_utc,
        c.market_ticker,
        c.open AS open_c,
        c.high AS high_c,
        c.low AS low_c,
        c.close AS close_c,
        c.volume,
        c.num_trades,
        m.series_ticker,
        m.event_ticker,
        m.floor_strike,
        m.cap_strike,
        m.strike_type
    FROM candles c
    JOIN markets m ON m.ticker = c.market_ticker
    WHERE c.period_minutes = 1
)
SELECT
    dc.city,
    b.series_ticker,
    b.event_ticker,
    b.market_ticker,
    b.ts_utc,
    (b.ts_utc AT TIME ZONE dc.tz)::date AS local_date,
    (b.ts_utc AT TIME ZONE dc.tz) AS ts_local,
    b.open_c,
    b.high_c,
    b.low_c,
    b.close_c,
    b.volume,
    b.num_trades,
    b.floor_strike,
    b.cap_strike,
    b.strike_type,
    (b.close_c / 100.0) AS mid_prob,
    CASE
        WHEN b.high_c > b.low_c THEN (b.close_c - b.low_c) / NULLIF(b.high_c - b.low_c, 0)
        ELSE 0.5
    END AS clv,
    (b.close_c - LAG(b.close_c) OVER w) / 100.0 AS mid_velocity,
    (b.close_c - 2 * LAG(b.close_c) OVER w + LAG(b.close_c, 2) OVER w) / 100.0 AS mid_acceleration,
    b.volume - LAG(b.volume) OVER w AS volume_delta,
    LAG(b.close_c) OVER w AS close_prev
FROM base b
JOIN dim_city dc ON dc.series_ticker = b.series_ticker
WINDOW w AS (PARTITION BY b.market_ticker ORDER BY b.ts_utc)
"""

DROP_SQL = "DROP MATERIALIZED VIEW IF EXISTS feat.minute_panel_base"


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS feat")
    op.execute(VIEW_SQL)
    op.execute("CREATE UNIQUE INDEX idx_feat_minute_panel_market_ts ON feat.minute_panel_base (market_ticker, ts_utc)")
    op.execute("CREATE INDEX idx_feat_minute_panel_city_ts ON feat.minute_panel_base (city, ts_utc)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_feat_minute_panel_city_ts")
    op.execute("DROP INDEX IF EXISTS idx_feat_minute_panel_market_ts")
    op.execute(DROP_SQL)
