"""refactor_settlement_columnar_schema

Revision ID: ca1e230aad1c
Revises: dab74fb952df
Create Date: 2025-11-12 17:22:50.217783

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ca1e230aad1c'
down_revision: Union[str, Sequence[str], None] = 'dab74fb952df'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade to columnar settlement schema with dim_city table."""

    # Step 1: Create dim_city dimension table
    op.execute("""
        CREATE TABLE IF NOT EXISTS dim_city (
            city            text PRIMARY KEY,
            series_ticker   text NOT NULL UNIQUE,
            icao            text NOT NULL,
            issuedby        text NOT NULL,
            ghcnd_station   text NOT NULL,
            tz              text NOT NULL  -- IANA timezone
        )
    """)

    # Populate dim_city with 7 cities
    op.execute("""
        INSERT INTO dim_city (city, series_ticker, icao, issuedby, ghcnd_station, tz) VALUES
            ('chicago',      'KXHIGHCHI', 'KMDW', 'MDW', 'GHCND:USW00014819', 'America/Chicago'),
            ('new_york',     'KXHIGHNY',  'KNYC', 'NYC', 'GHCND:USW00094728', 'America/New_York'),
            ('austin',       'KXHIGHAUS', 'KAUS', 'AUS', 'GHCND:USW00013958', 'America/Chicago'),
            ('miami',        'KXHIGHMIA', 'KMIA', 'MIA', 'GHCND:USW00012839', 'America/New_York'),
            ('los_angeles',  'KXHIGHLAX', 'KLAX', 'LAX', 'GHCND:USW00023174', 'America/Los_Angeles'),
            ('denver',       'KXHIGHDEN', 'KDEN', 'DEN', 'GHCND:USW00003017', 'America/Denver'),
            ('philadelphia', 'KXHIGHPHL', 'KPHL', 'PHL', 'GHCND:USW00013739', 'America/New_York')
        ON CONFLICT (city) DO NOTHING
    """)

    # Step 2: Create settlement source precedence function
    op.execute("""
        CREATE OR REPLACE FUNCTION wx.choose_settlement_source(
            has_cli      boolean,
            has_cf6      boolean,
            has_iem      boolean,
            has_ghcnd    boolean,
            has_vc       boolean,
            prelim_cli   boolean,
            prelim_cf6   boolean
        ) RETURNS text IMMUTABLE AS $$
            SELECT CASE
                -- CLI is always final (precedence 1)
                WHEN has_cli THEN 'CLI'

                -- CF6 from weather.gov scrape (precedence 2)
                WHEN has_cf6 THEN 'CF6'

                -- IEM CF6 JSON (precedence 3, current primary)
                WHEN has_iem THEN 'IEM_CF6'

                -- GHCND archive (precedence 4, audit only)
                WHEN has_ghcnd THEN 'GHCND'

                -- VC proxy from minute_obs (precedence 5, diagnostic only)
                WHEN has_vc THEN 'VC'

                ELSE NULL
            END;
        $$ LANGUAGE SQL
    """)

    # Step 3: Backup existing wx.settlement table
    op.execute("ALTER TABLE IF EXISTS wx.settlement RENAME TO settlement_old")

    # Step 4: Create new columnar wx.settlement table
    op.execute("""
        CREATE TABLE wx.settlement (
            city                text NOT NULL,
            date_local          date NOT NULL,

            -- Source-specific TMAX values (smallint = integer °F)
            tmax_cli            smallint,
            tmax_cf6            smallint,
            tmax_iem_cf6        smallint,
            tmax_ghcnd          smallint,
            tmax_vc             smallint,

            -- Preliminary flags per source
            is_prelim_cli       boolean DEFAULT false,
            is_prelim_cf6       boolean DEFAULT true,
            is_prelim_iem       boolean DEFAULT true,
            is_prelim_ghcnd     boolean DEFAULT false,
            is_prelim_vc        boolean DEFAULT true,

            -- Metadata per source (when last fetched)
            retrieved_at_cli    timestamptz,
            retrieved_at_cf6    timestamptz,
            retrieved_at_iem    timestamptz,
            retrieved_at_ghcnd  timestamptz,
            retrieved_at_vc     timestamptz,

            -- Audit payloads (JSONB for efficiency)
            raw_payloads        jsonb DEFAULT '{}'::jsonb,

            -- Computed final settlement (GENERATED ALWAYS)
            source_final        text GENERATED ALWAYS AS (
                wx.choose_settlement_source(
                    tmax_cli IS NOT NULL,
                    tmax_cf6 IS NOT NULL,
                    tmax_iem_cf6 IS NOT NULL,
                    tmax_ghcnd IS NOT NULL,
                    tmax_vc IS NOT NULL,
                    is_prelim_cli,
                    is_prelim_cf6
                )
            ) STORED,

            tmax_final          smallint GENERATED ALWAYS AS (
                COALESCE(tmax_cli, tmax_cf6, tmax_iem_cf6, tmax_ghcnd, tmax_vc)
            ) STORED,

            PRIMARY KEY (city, date_local),
            FOREIGN KEY (city) REFERENCES dim_city(city) ON DELETE CASCADE
        )
    """)

    # Create indexes
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_settlement_city_date ON wx.settlement (city, date_local)
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_settlement_source_final ON wx.settlement (source_final)
    """)

    # Step 5: Migrate existing data from settlement_old to new table
    op.execute("""
        INSERT INTO wx.settlement (
            city, date_local,
            tmax_iem_cf6, retrieved_at_iem, is_prelim_iem,
            raw_payloads
        )
        SELECT
            city,
            date_local,
            tmax_f::smallint,
            retrieved_at,
            is_preliminary,
            jsonb_build_object('IEM_CF6', raw_payload)
        FROM wx.settlement_old
        WHERE source = 'IEM_CF6'
        ON CONFLICT (city, date_local) DO UPDATE SET
            tmax_iem_cf6 = EXCLUDED.tmax_iem_cf6,
            retrieved_at_iem = EXCLUDED.retrieved_at_iem,
            is_prelim_iem = EXCLUDED.is_prelim_iem,
            raw_payloads = wx.settlement.raw_payloads || EXCLUDED.raw_payloads
    """)


def downgrade() -> None:
    """Downgrade to multi-row settlement schema."""

    # Restore old table
    op.execute("DROP TABLE IF EXISTS wx.settlement")
    op.execute("ALTER TABLE IF EXISTS wx.settlement_old RENAME TO settlement")

    # Drop function
    op.execute("DROP FUNCTION IF EXISTS wx.choose_settlement_source")

    # Drop dim_city (keep data in case needed)
    # op.execute("DROP TABLE IF EXISTS dim_city")
