"""add_weather_join_view

Revision ID: e19ccb99e1df
Revises: d5981c5d7a2c
Create Date: 2025-11-19 18:25:00

"""
from typing import Sequence, Union

from alembic import op


revision: str = "e19ccb99e1df"
down_revision: Union[str, Sequence[str], None] = "d5981c5d7a2c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


VIEW_SQL = """
CREATE MATERIALIZED VIEW feat.minute_panel_with_weather AS
WITH loc_map AS (
    SELECT dc.city, wx.loc_id
    FROM dim_city dc
    JOIN wx.location wx ON wx.city = dc.city
)
SELECT
    nb.*,
    wx1.temp_f AS wx_temp_1m,
    wx1.humidity AS wx_humidity_1m,
    wx1.dew_f AS wx_dew_1m,
    wx1.windspeed_mph AS wx_windspeed_1m,
    MAX(wx1.temp_f) OVER (PARTITION BY nb.city ORDER BY nb.ts_utc ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS wx_running_max
FROM feat.minute_panel_neighbors nb
JOIN loc_map lm ON lm.city = nb.city
LEFT JOIN wx.minute_obs_1m wx1 ON wx1.loc_id = lm.loc_id AND wx1.ts_utc = nb.ts_utc;
"""


def upgrade() -> None:
    op.execute(VIEW_SQL)
    op.execute("CREATE UNIQUE INDEX idx_feat_panel_weather_market_ts ON feat.minute_panel_with_weather (market_ticker, ts_utc)")
    op.execute("CREATE INDEX idx_feat_panel_weather_city_ts ON feat.minute_panel_with_weather (city, ts_utc)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_feat_panel_weather_city_ts")
    op.execute("DROP INDEX IF EXISTS idx_feat_panel_weather_market_ts")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS feat.minute_panel_with_weather")
