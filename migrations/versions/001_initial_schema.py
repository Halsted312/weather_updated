"""Initial schema with TimescaleDB hypertables.

Revision ID: 001_initial
Revises:
Create Date: 2024-11-26

Creates three schemas (wx, kalshi, sim) with all tables and TimescaleDB hypertables.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create schemas
    op.execute("CREATE SCHEMA IF NOT EXISTS wx")
    op.execute("CREATE SCHEMA IF NOT EXISTS kalshi")
    op.execute("CREATE SCHEMA IF NOT EXISTS sim")

    # ==========================================================================
    # Schema: wx (Weather/Labels)
    # ==========================================================================

    # wx.settlement - Official NWS daily max (ground truth)
    op.create_table(
        'settlement',
        sa.Column('city', sa.Text(), nullable=False),
        sa.Column('date_local', sa.Date(), nullable=False),
        sa.Column('tmax_cli_f', sa.SmallInteger(), nullable=True),
        sa.Column('tmax_cf6_f', sa.SmallInteger(), nullable=True),
        sa.Column('tmax_ads_f', sa.SmallInteger(), nullable=True),
        sa.Column('tmax_final', sa.SmallInteger(), nullable=False),
        sa.Column('source_final', sa.Text(), nullable=False),
        sa.Column('raw_payload', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.PrimaryKeyConstraint('city', 'date_local'),
        schema='wx'
    )

    # wx.minute_obs - Visual Crossing 5-min observations (HYPERTABLE)
    op.create_table(
        'minute_obs',
        sa.Column('loc_id', sa.String(10), nullable=False),
        sa.Column('ts_utc', sa.DateTime(timezone=True), nullable=False),
        sa.Column('temp_f', sa.Float(), nullable=True),
        sa.Column('feelslike_f', sa.Float(), nullable=True),
        sa.Column('humidity', sa.Float(), nullable=True),
        sa.Column('dew_f', sa.Float(), nullable=True),
        sa.Column('precip_in', sa.Float(), nullable=True),
        sa.Column('precip_prob', sa.Float(), nullable=True),
        sa.Column('snow_in', sa.Float(), nullable=True),
        sa.Column('snow_depth_in', sa.Float(), nullable=True),
        sa.Column('windspeed_mph', sa.Float(), nullable=True),
        sa.Column('winddir', sa.Float(), nullable=True),
        sa.Column('windgust_mph', sa.Float(), nullable=True),
        sa.Column('pressure_mb', sa.Float(), nullable=True),
        sa.Column('visibility_mi', sa.Float(), nullable=True),
        sa.Column('cloud_cover', sa.Float(), nullable=True),
        sa.Column('solar_radiation', sa.Float(), nullable=True),
        sa.Column('solar_energy', sa.Float(), nullable=True),
        sa.Column('uv_index', sa.Float(), nullable=True),
        sa.Column('conditions', sa.Text(), nullable=True),
        sa.Column('icon', sa.Text(), nullable=True),
        sa.Column('source', sa.String(20), server_default='visualcrossing', nullable=True),
        sa.Column('stations', sa.String(100), nullable=True),
        sa.Column('ffilled', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('raw_json', postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint('loc_id', 'ts_utc'),
        schema='wx'
    )

    # Convert to hypertable
    op.execute("""
        SELECT create_hypertable(
            'wx.minute_obs',
            'ts_utc',
            chunk_time_interval => INTERVAL '7 days',
            if_not_exists => TRUE
        )
    """)

    # wx.forecast_snapshot - VC historical forecasts
    op.create_table(
        'forecast_snapshot',
        sa.Column('city', sa.Text(), nullable=False),
        sa.Column('target_date', sa.Date(), nullable=False),
        sa.Column('basis_date', sa.Date(), nullable=False),
        sa.Column('lead_days', sa.Integer(), nullable=False),
        sa.Column('provider', sa.Text(), server_default='visualcrossing', nullable=True),
        sa.Column('tempmax_fcst_f', sa.Float(), nullable=True),
        sa.Column('tempmin_fcst_f', sa.Float(), nullable=True),
        sa.Column('precip_fcst_in', sa.Float(), nullable=True),
        sa.Column('precip_prob_fcst', sa.Float(), nullable=True),
        sa.Column('humidity_fcst', sa.Float(), nullable=True),
        sa.Column('windspeed_fcst_mph', sa.Float(), nullable=True),
        sa.Column('conditions_fcst', sa.Text(), nullable=True),
        sa.Column('raw_json', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.PrimaryKeyConstraint('city', 'target_date', 'basis_date'),
        schema='wx'
    )

    # ==========================================================================
    # Schema: kalshi (Market Data)
    # ==========================================================================

    # kalshi.markets - Contract metadata
    op.create_table(
        'markets',
        sa.Column('ticker', sa.Text(), nullable=False),
        sa.Column('city', sa.Text(), nullable=True),
        sa.Column('event_date', sa.Date(), nullable=True),
        sa.Column('exchange_market_id', sa.Text(), nullable=True),
        sa.Column('strike_type', sa.Text(), nullable=True),
        sa.Column('floor_strike', sa.SmallInteger(), nullable=True),
        sa.Column('cap_strike', sa.SmallInteger(), nullable=True),
        sa.Column('listed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('close_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expiration_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.Text(), nullable=True),
        sa.Column('result', sa.Text(), nullable=True),
        sa.Column('settlement_value', sa.SmallInteger(), nullable=True),
        sa.Column('raw_json', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.PrimaryKeyConstraint('ticker'),
        schema='kalshi'
    )
    op.create_index('idx_kalshi_markets_city_date', 'markets', ['city', 'event_date'], schema='kalshi')

    # kalshi.candles_1m - 1-minute OHLCV (HYPERTABLE)
    op.create_table(
        'candles_1m',
        sa.Column('ticker', sa.Text(), nullable=False),
        sa.Column('bucket_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('open_c', sa.SmallInteger(), nullable=True),
        sa.Column('high_c', sa.SmallInteger(), nullable=True),
        sa.Column('low_c', sa.SmallInteger(), nullable=True),
        sa.Column('close_c', sa.SmallInteger(), nullable=True),
        sa.Column('yes_bid_c', sa.SmallInteger(), nullable=True),
        sa.Column('yes_ask_c', sa.SmallInteger(), nullable=True),
        sa.Column('volume', sa.Integer(), nullable=True),
        sa.Column('open_interest', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('ticker', 'bucket_start'),
        sa.ForeignKeyConstraint(['ticker'], ['kalshi.markets.ticker']),
        schema='kalshi'
    )

    # Convert to hypertable
    op.execute("""
        SELECT create_hypertable(
            'kalshi.candles_1m',
            'bucket_start',
            chunk_time_interval => INTERVAL '7 days',
            if_not_exists => TRUE
        )
    """)

    # kalshi.ws_raw - WebSocket raw log (HYPERTABLE)
    op.create_table(
        'ws_raw',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('ts_utc', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('source', sa.Text(), server_default='kalshi', nullable=False),
        sa.Column('stream', sa.Text(), nullable=True),
        sa.Column('topic', sa.Text(), nullable=True),
        sa.Column('payload', postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint('id', 'ts_utc'),
        schema='kalshi'
    )

    # Convert to hypertable
    op.execute("""
        SELECT create_hypertable(
            'kalshi.ws_raw',
            'ts_utc',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        )
    """)

    # kalshi.orders - My orders
    op.create_table(
        'orders',
        sa.Column('order_id', sa.Text(), nullable=False),
        sa.Column('ticker', sa.Text(), nullable=False),
        sa.Column('side', sa.Text(), nullable=False),
        sa.Column('qty', sa.Integer(), nullable=False),
        sa.Column('price_c', sa.SmallInteger(), nullable=False),
        sa.Column('order_type', sa.Text(), nullable=True),
        sa.Column('status', sa.Text(), nullable=True),
        sa.Column('raw_json', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.PrimaryKeyConstraint('order_id'),
        schema='kalshi'
    )

    # kalshi.fills - My fills
    op.create_table(
        'fills',
        sa.Column('fill_id', sa.Text(), nullable=False),
        sa.Column('order_id', sa.Text(), nullable=False),
        sa.Column('ticker', sa.Text(), nullable=False),
        sa.Column('side', sa.Text(), nullable=False),
        sa.Column('qty', sa.Integer(), nullable=False),
        sa.Column('price_c', sa.SmallInteger(), nullable=False),
        sa.Column('fee_c', sa.SmallInteger(), nullable=True),
        sa.Column('ts_utc', sa.DateTime(timezone=True), nullable=False),
        sa.Column('raw_json', postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint('fill_id'),
        sa.ForeignKeyConstraint(['order_id'], ['kalshi.orders.order_id']),
        schema='kalshi'
    )

    # ==========================================================================
    # Schema: sim (Backtest/Simulation)
    # ==========================================================================

    # sim.run - Backtest run metadata
    op.create_table(
        'run',
        sa.Column('run_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('strategy_name', sa.Text(), nullable=False),
        sa.Column('params_json', postgresql.JSONB(), nullable=True),
        sa.Column('train_start', sa.Date(), nullable=True),
        sa.Column('train_end', sa.Date(), nullable=True),
        sa.Column('test_start', sa.Date(), nullable=True),
        sa.Column('test_end', sa.Date(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.PrimaryKeyConstraint('run_id'),
        schema='sim'
    )

    # sim.trade - Simulated trades (HYPERTABLE)
    op.create_table(
        'trade',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('trade_ts_utc', sa.DateTime(timezone=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('ticker', sa.Text(), nullable=False),
        sa.Column('city', sa.Text(), nullable=True),
        sa.Column('event_date', sa.Date(), nullable=True),
        sa.Column('side', sa.Text(), nullable=False),
        sa.Column('qty', sa.Integer(), nullable=False),
        sa.Column('price_c', sa.SmallInteger(), nullable=False),
        sa.Column('fee_c', sa.SmallInteger(), server_default='0', nullable=True),
        sa.Column('pnl_c', sa.Integer(), nullable=True),
        sa.Column('position_after', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id', 'trade_ts_utc'),
        sa.ForeignKeyConstraint(['run_id'], ['sim.run.run_id']),
        schema='sim'
    )

    # Convert to hypertable
    op.execute("""
        SELECT create_hypertable(
            'sim.trade',
            'trade_ts_utc',
            chunk_time_interval => INTERVAL '30 days',
            if_not_exists => TRUE
        )
    """)


def downgrade() -> None:
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('trade', schema='sim')
    op.drop_table('run', schema='sim')
    op.drop_table('fills', schema='kalshi')
    op.drop_table('orders', schema='kalshi')
    op.drop_table('ws_raw', schema='kalshi')
    op.drop_table('candles_1m', schema='kalshi')
    op.drop_table('markets', schema='kalshi')
    op.drop_table('forecast_snapshot', schema='wx')
    op.drop_table('minute_obs', schema='wx')
    op.drop_table('settlement', schema='wx')

    # Drop schemas
    op.execute("DROP SCHEMA IF EXISTS sim CASCADE")
    op.execute("DROP SCHEMA IF EXISTS kalshi CASCADE")
    op.execute("DROP SCHEMA IF EXISTS wx CASCADE")
