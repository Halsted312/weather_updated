"""
SQLAlchemy ORM models for all three schemas: wx, kalshi, sim.
"""

from datetime import date, datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# =============================================================================
# Schema: wx (Weather/Labels)
# =============================================================================


class WxSettlement(Base):
    """Official NWS daily max temperature (ground truth for Kalshi settlement).

    Multi-source TMAX tracking:
    - tmax_cli_f: NWS CLI (via IEM parsed JSON)
    - tmax_cf6_f: NWS CF6 (preliminary monthly)
    - tmax_iem_f: IEM CLI JSON API (historical)
    - tmax_ncei_f: NCEI daily-summaries (canonical fallback)
    - tmax_ads_f: Legacy ADS data

    Kalshi bucket tracking:
    - settled_ticker: The winning market ticker
    - settled_bucket_type: 'between' | 'less' | 'greater'
    - settled_floor_strike, settled_cap_strike: Temperature bounds
    - settled_bucket_label: Human-readable label
    """

    __tablename__ = "settlement"
    __table_args__ = {"schema": "wx"}

    city: Mapped[str] = mapped_column(Text, primary_key=True)
    date_local: Mapped[date] = mapped_column(Date, primary_key=True)

    # Source-specific temperatures (integer °F)
    tmax_cli_f: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    tmax_cf6_f: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    tmax_iem_f: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    tmax_ncei_f: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    tmax_ads_f: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    # Chosen settlement value
    tmax_final: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    source_final: Mapped[str] = mapped_column(Text, nullable=False)  # 'cli' | 'cf6' | 'iem' | 'ncei' | 'ads'

    # Legacy raw payload (kept for backwards compatibility)
    raw_payload: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Per-source raw payloads
    raw_payload_cli: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    raw_payload_cf6: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    raw_payload_iem: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    raw_payload_ncei: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Kalshi settlement bucket tracking
    settled_ticker: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    settled_bucket_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 'between' | 'less' | 'greater'
    settled_floor_strike: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    settled_cap_strike: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    settled_bucket_label: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), onupdate=text("NOW()")
    )


class WxMinuteObs(Base):
    """Visual Crossing 5-minute weather observations (features only, never settlement)."""

    __tablename__ = "minute_obs"
    __table_args__ = {"schema": "wx"}

    loc_id: Mapped[str] = mapped_column(String(10), primary_key=True)  # KMDW, KDEN, etc.
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)

    # Core weather data
    temp_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feelslike_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dew_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Precipitation
    precip_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precip_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    snow_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    snow_depth_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Wind
    windspeed_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winddir: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windgust_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Atmospheric
    pressure_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    visibility_mi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cloud_cover: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Solar
    solar_radiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    solar_energy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    uv_index: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Conditions
    conditions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    icon: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    source: Mapped[str] = mapped_column(String(20), default="visualcrossing")
    stations: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    ffilled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)


class WxForecastSnapshot(Base):
    """Visual Crossing historical forecast snapshots for Option-1 backtest."""

    __tablename__ = "forecast_snapshot"
    __table_args__ = {"schema": "wx"}

    city: Mapped[str] = mapped_column(Text, primary_key=True)
    target_date: Mapped[date] = mapped_column(Date, primary_key=True)
    basis_date: Mapped[date] = mapped_column(Date, primary_key=True)

    lead_days: Mapped[int] = mapped_column(Integer, nullable=False)
    provider: Mapped[str] = mapped_column(Text, default="visualcrossing")

    # Forecast values
    tempmax_fcst_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tempmin_fcst_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precip_fcst_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precip_prob_fcst: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity_fcst: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeed_fcst_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    conditions_fcst: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )


# =============================================================================
# Schema: kalshi (Market Data)
# =============================================================================


class KalshiMarket(Base):
    """Kalshi market/contract metadata."""

    __tablename__ = "markets"
    __table_args__ = {"schema": "kalshi"}

    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    city: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    event_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    exchange_market_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    strike_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 'between' | 'less' | 'greater'
    floor_strike: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    cap_strike: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    listed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    close_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    expiration_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    status: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 'open' | 'closed' | 'settled'
    result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 'yes' | 'no'
    settlement_value: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), onupdate=text("NOW()")
    )

    # Relationships
    candles: Mapped[list["KalshiCandle1m"]] = relationship(back_populates="market")


class KalshiCandle1m(Base):
    """1-minute OHLCV candlesticks for Kalshi markets.

    Supports dual storage with `source` column:
    - 'api_event': From Kalshi Event Candlesticks API (primary)
    - 'trades': Aggregated from individual trades (fallback/audit)
    """

    __tablename__ = "candles_1m"
    __table_args__ = {"schema": "kalshi"}

    ticker: Mapped[str] = mapped_column(
        Text, ForeignKey("kalshi.markets.ticker"), primary_key=True
    )
    bucket_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    # Source of candle data: 'api_event' | 'trades'
    source: Mapped[str] = mapped_column(
        String(20), primary_key=True, default="trades"
    )

    # OHLC prices (0-100 cents)
    open_c: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    high_c: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    low_c: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    close_c: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    # Bid/Ask
    yes_bid_c: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    yes_ask_c: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    # Volume
    volume: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    open_interest: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    market: Mapped["KalshiMarket"] = relationship(back_populates="candles")


class KalshiWsRaw(Base):
    """Raw WebSocket message log - store EVERYTHING."""

    __tablename__ = "ws_raw"
    __table_args__ = {"schema": "kalshi"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), primary_key=True
    )

    source: Mapped[str] = mapped_column(Text, default="kalshi", nullable=False)
    stream: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 'market-data', 'trades', etc.
    topic: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # ticker or channel
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)


class KalshiOrder(Base):
    """My orders on Kalshi."""

    __tablename__ = "orders"
    __table_args__ = {"schema": "kalshi"}

    order_id: Mapped[str] = mapped_column(Text, primary_key=True)
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    side: Mapped[str] = mapped_column(Text, nullable=False)  # 'buy' | 'sell'
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    price_c: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    order_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 'limit' | 'market'
    status: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 'pending' | 'filled' | 'cancelled'

    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), onupdate=text("NOW()")
    )

    # Relationships
    fills: Mapped[list["KalshiFill"]] = relationship(back_populates="order")


class KalshiFill(Base):
    """My fills on Kalshi."""

    __tablename__ = "fills"
    __table_args__ = {"schema": "kalshi"}

    fill_id: Mapped[str] = mapped_column(Text, primary_key=True)
    order_id: Mapped[str] = mapped_column(
        Text, ForeignKey("kalshi.orders.order_id"), nullable=False
    )
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    side: Mapped[str] = mapped_column(Text, nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    price_c: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    fee_c: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Relationships
    order: Mapped["KalshiOrder"] = relationship(back_populates="fills")


# =============================================================================
# Schema: sim (Backtest/Simulation)
# =============================================================================


class SimRun(Base):
    """Backtest run metadata."""

    __tablename__ = "run"
    __table_args__ = {"schema": "sim"}

    run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    strategy_name: Mapped[str] = mapped_column(Text, nullable=False)
    params_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    train_start: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    train_end: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    test_start: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    test_end: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )

    # Relationships
    trades: Mapped[list["SimTrade"]] = relationship(back_populates="run")


class SimTrade(Base):
    """Simulated trades from backtests."""

    __tablename__ = "trade"
    __table_args__ = {"schema": "sim"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_ts_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )

    run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("sim.run.run_id"), nullable=False
    )
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    city: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    event_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    side: Mapped[str] = mapped_column(Text, nullable=False)  # 'buy' | 'sell'
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    price_c: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    fee_c: Mapped[int] = mapped_column(SmallInteger, default=0)
    pnl_c: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    position_after: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    run: Mapped["SimRun"] = relationship(back_populates="trades")


# =============================================================================
# Schema: meta (Operations/Infrastructure)
# =============================================================================


class IngestCheckpoint(Base):
    """Track ingestion progress for resumable backfills.

    Each (pipeline_name, city) combo tracks where we left off,
    enabling resume on crash/restart.
    """

    __tablename__ = "ingestion_checkpoint"
    __table_args__ = {"schema": "meta"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # What are we tracking?
    pipeline_name: Mapped[str] = mapped_column(Text, nullable=False)
    city: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Where did we get to?
    last_processed_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    last_processed_cursor: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_processed_ticker: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Status tracking
    status: Mapped[str] = mapped_column(Text, default="running")  # running, completed, failed
    total_processed: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), onupdate=text("NOW()")
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
