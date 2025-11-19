"""add wx schema for visual crossing weather data

Revision ID: d8fedec351a7
Revises: 2654f635dce0
Create Date: 2025-11-11 22:46:39.240766

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd8fedec351a7'
down_revision: Union[str, Sequence[str], None] = '2654f635dce0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create wx schema
    op.execute("CREATE SCHEMA IF NOT EXISTS wx")

    # Create wx.location table
    op.execute("""
        CREATE TABLE wx.location (
            loc_id text PRIMARY KEY,
            vc_key text NOT NULL,
            city text NOT NULL,
            created_at timestamptz DEFAULT now()
        )
    """)

    # Create wx.minute_obs table (5-minute raw observations)
    op.execute("""
        CREATE TABLE wx.minute_obs (
            loc_id text NOT NULL REFERENCES wx.location(loc_id) ON DELETE CASCADE,
            ts_utc timestamptz NOT NULL,
            temp_f double precision,
            humidity double precision,
            dew_f double precision,
            windspeed_mph double precision,
            windgust_mph double precision,
            pressure_mb double precision,
            precip_in double precision,
            preciptype text,
            source text DEFAULT 'visualcrossing',
            raw_json jsonb,
            PRIMARY KEY (loc_id, ts_utc)
        )
    """)

    # Create indexes for performance
    op.execute("CREATE INDEX idx_wx_minute_obs_ts ON wx.minute_obs(ts_utc)")
    op.execute("CREATE INDEX idx_wx_minute_obs_loc_ts ON wx.minute_obs(loc_id, ts_utc)")

    # Create materialized view for 1-minute upsampled grid (LOCF)
    op.execute("""
        CREATE MATERIALIZED VIEW wx.minute_obs_1m AS
        WITH base AS (
            SELECT loc_id, ts_utc, temp_f, humidity, dew_f,
                   windspeed_mph, windgust_mph, pressure_mb, precip_in, preciptype
            FROM wx.minute_obs
        ),
        bounds AS (
            SELECT loc_id,
                   date_trunc('day', min(ts_utc)) AS start_utc,
                   date_trunc('day', max(ts_utc)) + interval '1 day' AS end_utc
            FROM base
            GROUP BY 1
        ),
        grid AS (
            SELECT b.loc_id, g.ts AS ts_utc
            FROM bounds b
            CROSS JOIN LATERAL generate_series(
                b.start_utc,
                b.end_utc - interval '1 minute',
                interval '1 minute'
            ) AS g(ts)
        ),
        asof_join AS (
            SELECT g.loc_id, g.ts_utc,
                   (SELECT temp_f FROM base b
                    WHERE b.loc_id=g.loc_id AND b.ts_utc<=g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS temp_f,
                   (SELECT humidity FROM base b
                    WHERE b.loc_id=g.loc_id AND b.ts_utc<=g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS humidity,
                   (SELECT dew_f FROM base b
                    WHERE b.loc_id=g.loc_id AND b.ts_utc<=g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS dew_f,
                   (SELECT windspeed_mph FROM base b
                    WHERE b.loc_id=g.loc_id AND b.ts_utc<=g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS windspeed_mph,
                   (SELECT windgust_mph FROM base b
                    WHERE b.loc_id=g.loc_id AND b.ts_utc<=g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS windgust_mph,
                   (SELECT pressure_mb FROM base b
                    WHERE b.loc_id=g.loc_id AND b.ts_utc<=g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS pressure_mb,
                   (SELECT precip_in FROM base b
                    WHERE b.loc_id=g.loc_id AND b.ts_utc<=g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS precip_in,
                   (SELECT preciptype FROM base b
                    WHERE b.loc_id=g.loc_id AND b.ts_utc<=g.ts_utc
                    ORDER BY b.ts_utc DESC LIMIT 1) AS preciptype
            FROM grid g
        )
        SELECT * FROM asof_join
    """)

    # Create index on materialized view
    op.execute("CREATE INDEX idx_wx_minute_obs_1m_loc_ts ON wx.minute_obs_1m(loc_id, ts_utc)")
    op.execute("CREATE INDEX idx_wx_minute_obs_1m_ts ON wx.minute_obs_1m(ts_utc)")


def downgrade() -> None:
    """Downgrade schema."""
    # Drop materialized view
    op.execute("DROP MATERIALIZED VIEW IF EXISTS wx.minute_obs_1m")

    # Drop tables
    op.execute("DROP TABLE IF EXISTS wx.minute_obs CASCADE")
    op.execute("DROP TABLE IF EXISTS wx.location CASCADE")

    # Drop schema
    op.execute("DROP SCHEMA IF EXISTS wx CASCADE")
