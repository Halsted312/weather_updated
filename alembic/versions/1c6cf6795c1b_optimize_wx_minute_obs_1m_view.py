"""optimize_wx_minute_obs_1m_view

Revision ID: 1c6cf6795c1b
Revises: bf6f08843481
Create Date: 2025-11-19 11:38:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "1c6cf6795c1b"
down_revision: Union[str, Sequence[str], None] = "bf6f08843481"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Replace the expensive correlated-subquery MV with a window-function version."""
    op.execute("DROP MATERIALIZED VIEW IF EXISTS wx.minute_obs_1m CASCADE")

    op.execute(
        """
        CREATE MATERIALIZED VIEW wx.minute_obs_1m AS
        WITH bounds AS (
            SELECT loc_id,
                   date_trunc('day', MIN(ts_utc))     AS start_utc,
                   date_trunc('day', MAX(ts_utc)) + INTERVAL '1 day' AS end_utc
            FROM wx.minute_obs
            GROUP BY loc_id
        ),
        grid AS (
            SELECT b.loc_id,
                   gs AS ts_utc
            FROM bounds b
            CROSS JOIN LATERAL generate_series(
                b.start_utc,
                b.end_utc - INTERVAL '1 minute',
                INTERVAL '1 minute'
            ) AS gs
        ),
        joined AS (
            SELECT g.loc_id,
                   g.ts_utc,
                   m.ts_utc AS obs_ts
            FROM grid g
            LEFT JOIN wx.minute_obs m
                ON m.loc_id = g.loc_id
               AND m.ts_utc = g.ts_utc
        ),
        locator AS (
            SELECT loc_id,
                   ts_utc,
                   MAX(obs_ts) OVER (
                       PARTITION BY loc_id
                       ORDER BY ts_utc
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS last_obs_ts
            FROM joined
        )
        SELECT l.loc_id,
               l.ts_utc,
               o.temp_f,
               o.humidity,
               o.dew_f,
               o.windspeed_mph,
               o.windgust_mph,
               o.pressure_mb,
               o.precip_in,
               o.preciptype,
               o.source,
               o.raw_json
        FROM locator l
        LEFT JOIN wx.minute_obs o
               ON o.loc_id = l.loc_id
              AND o.ts_utc = l.last_obs_ts
        """
    )

    op.execute("CREATE UNIQUE INDEX idx_wx_minute_obs_1m_loc_ts ON wx.minute_obs_1m(loc_id, ts_utc)")
    op.execute("CREATE INDEX idx_wx_minute_obs_1m_ts ON wx.minute_obs_1m(ts_utc)")


def downgrade() -> None:
    """Recreate the previous (correlated-subquery) MV."""
    op.execute("DROP MATERIALIZED VIEW IF EXISTS wx.minute_obs_1m CASCADE")

    op.execute(
        """
        CREATE MATERIALIZED VIEW wx.minute_obs_1m AS
        WITH base AS (
            SELECT loc_id, ts_utc, temp_f, humidity, dew_f,
                   windspeed_mph, windgust_mph, pressure_mb, precip_in, preciptype,
                   source, raw_json
            FROM wx.minute_obs
        ),
        bounds AS (
            SELECT loc_id,
                   date_trunc('day', MIN(ts_utc)) AS start_utc,
                   date_trunc('day', MAX(ts_utc)) + INTERVAL '1 day' AS end_utc
            FROM base
            GROUP BY loc_id
        ),
        grid AS (
            SELECT b.loc_id, g.ts AS ts_utc
            FROM bounds b
            CROSS JOIN LATERAL generate_series(
                b.start_utc,
                b.end_utc - INTERVAL '1 minute',
                INTERVAL '1 minute'
            ) AS g(ts)
        ),
        asof_join AS (
            SELECT g.loc_id, g.ts_utc,
                   (SELECT temp_f FROM base b
                    WHERE b.loc_id = g.loc_id AND b.ts_utc <= g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS temp_f,
                   (SELECT humidity FROM base b
                    WHERE b.loc_id = g.loc_id AND b.ts_utc <= g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS humidity,
                   (SELECT dew_f FROM base b
                    WHERE b.loc_id = g.loc_id AND b.ts_utc <= g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS dew_f,
                   (SELECT windspeed_mph FROM base b
                    WHERE b.loc_id = g.loc_id AND b.ts_utc <= g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS windspeed_mph,
                   (SELECT windgust_mph FROM base b
                    WHERE b.loc_id = g.loc_id AND b.ts_utc <= g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS windgust_mph,
                   (SELECT pressure_mb FROM base b
                    WHERE b.loc_id = g.loc_id AND b.ts_utc <= g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS pressure_mb,
                   (SELECT precip_in FROM base b
                    WHERE b.loc_id = g.loc_id AND b.ts_utc <= g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS precip_in,
                   (SELECT preciptype FROM base b
                    WHERE b.loc_id = g.loc_id AND b.ts_utc <= g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS preciptype,
                   (SELECT source FROM base b
                    WHERE b.loc_id = g.loc_id AND b.ts_utc <= g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS source,
                   (SELECT raw_json FROM base b
                    WHERE b.loc_id = g.loc_id AND b.ts_utc <= g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS raw_json
            FROM grid g
        )
        SELECT * FROM asof_join
        """
    )

    op.execute("CREATE INDEX idx_wx_minute_obs_1m_loc_ts ON wx.minute_obs_1m(loc_id, ts_utc)")
    op.execute("CREATE INDEX idx_wx_minute_obs_1m_ts ON wx.minute_obs_1m(ts_utc)")
