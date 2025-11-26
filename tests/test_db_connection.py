"""Tests for database connection and schema setup."""

import pytest
from sqlalchemy import text


class TestDatabaseConnection:
    """Test database connection and basic operations."""

    def test_connection_works(self, db_engine):
        """Test that we can connect to the database."""
        with db_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            row = result.fetchone()
            assert row is not None
            assert row[0] == 1

    def test_timescaledb_extension(self, db_engine):
        """Test that TimescaleDB extension is enabled."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == "timescaledb"

    def test_schemas_exist(self, db_engine):
        """Test that wx, kalshi, and sim schemas exist."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT schema_name
                    FROM information_schema.schemata
                    WHERE schema_name IN ('wx', 'kalshi', 'sim')
                    ORDER BY schema_name
                """)
            )
            schemas = [row[0] for row in result.fetchall()]
            assert "kalshi" in schemas
            assert "sim" in schemas
            assert "wx" in schemas


class TestWxSchema:
    """Test wx schema tables."""

    def test_settlement_table_exists(self, db_engine):
        """Test that wx.settlement table exists."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'wx' AND table_name = 'settlement'
                """)
            )
            row = result.fetchone()
            assert row is not None

    def test_minute_obs_table_exists(self, db_engine):
        """Test that wx.minute_obs table exists."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'wx' AND table_name = 'minute_obs'
                """)
            )
            row = result.fetchone()
            assert row is not None

    def test_minute_obs_is_hypertable(self, db_engine):
        """Test that wx.minute_obs is a TimescaleDB hypertable."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT hypertable_name
                    FROM timescaledb_information.hypertables
                    WHERE hypertable_schema = 'wx' AND hypertable_name = 'minute_obs'
                """)
            )
            row = result.fetchone()
            assert row is not None

    def test_forecast_snapshot_table_exists(self, db_engine):
        """Test that wx.forecast_snapshot table exists."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'wx' AND table_name = 'forecast_snapshot'
                """)
            )
            row = result.fetchone()
            assert row is not None


class TestKalshiSchema:
    """Test kalshi schema tables."""

    def test_markets_table_exists(self, db_engine):
        """Test that kalshi.markets table exists."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'kalshi' AND table_name = 'markets'
                """)
            )
            row = result.fetchone()
            assert row is not None

    def test_candles_1m_table_exists(self, db_engine):
        """Test that kalshi.candles_1m table exists."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'kalshi' AND table_name = 'candles_1m'
                """)
            )
            row = result.fetchone()
            assert row is not None

    def test_candles_1m_is_hypertable(self, db_engine):
        """Test that kalshi.candles_1m is a TimescaleDB hypertable."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT hypertable_name
                    FROM timescaledb_information.hypertables
                    WHERE hypertable_schema = 'kalshi' AND hypertable_name = 'candles_1m'
                """)
            )
            row = result.fetchone()
            assert row is not None

    def test_ws_raw_table_exists(self, db_engine):
        """Test that kalshi.ws_raw table exists."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'kalshi' AND table_name = 'ws_raw'
                """)
            )
            row = result.fetchone()
            assert row is not None

    def test_ws_raw_is_hypertable(self, db_engine):
        """Test that kalshi.ws_raw is a TimescaleDB hypertable."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT hypertable_name
                    FROM timescaledb_information.hypertables
                    WHERE hypertable_schema = 'kalshi' AND hypertable_name = 'ws_raw'
                """)
            )
            row = result.fetchone()
            assert row is not None


class TestSimSchema:
    """Test sim schema tables."""

    def test_run_table_exists(self, db_engine):
        """Test that sim.run table exists."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'sim' AND table_name = 'run'
                """)
            )
            row = result.fetchone()
            assert row is not None

    def test_trade_table_exists(self, db_engine):
        """Test that sim.trade table exists."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'sim' AND table_name = 'trade'
                """)
            )
            row = result.fetchone()
            assert row is not None

    def test_trade_is_hypertable(self, db_engine):
        """Test that sim.trade is a TimescaleDB hypertable."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT hypertable_name
                    FROM timescaledb_information.hypertables
                    WHERE hypertable_schema = 'sim' AND hypertable_name = 'trade'
                """)
            )
            row = result.fetchone()
            assert row is not None
