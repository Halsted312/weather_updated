"""
SQLAlchemy database models for Kalshi weather markets.

All timestamps stored in UTC. Prices stored in cents (integers).
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    Text,
    ForeignKey,
    Index,
    UniqueConstraint,
    JSON,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Series(Base):
    """Kalshi series (e.g., KXHIGHCHI for Chicago)."""

    __tablename__ = "series"

    series_ticker = Column(String(50), primary_key=True)
    title = Column(String(200))
    category = Column(String(100))
    frequency = Column(String(50))
    settlement_source_json = Column(JSON)
    raw_json = Column(JSON)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    markets = relationship("Market", back_populates="series")

    def __repr__(self):
        return f"<Series(ticker={self.series_ticker}, title={self.title})>"


class Market(Base):
    """Individual market contract."""

    __tablename__ = "markets"

    ticker = Column(String(100), primary_key=True)
    series_ticker = Column(String(50), ForeignKey("series.series_ticker"), nullable=False)
    event_ticker = Column(String(100))
    title = Column(String(200))
    subtitle = Column(String(200))

    # Timestamps (stored as UTC)
    open_time = Column(DateTime, nullable=False)
    close_time = Column(DateTime, nullable=False)
    expiration_time = Column(DateTime)

    # Status and settlement
    status = Column(String(50), nullable=False, index=True)  # open, closed, settled, finalized
    result = Column(String(10))  # yes, no
    settlement_value = Column(Float)

    # Strike info (for ranged markets)
    floor_strike = Column(Float)
    cap_strike = Column(Float)
    strike_type = Column(String(50))

    # Market data (prices in cents)
    last_price = Column(Integer)  # in cents
    yes_bid = Column(Integer)
    yes_ask = Column(Integer)
    no_bid = Column(Integer)
    no_ask = Column(Integer)

    # Volume and liquidity
    volume = Column(Integer)
    volume_24h = Column(Integer)
    open_interest = Column(Integer)
    liquidity = Column(Integer)

    # Rules
    rules_primary = Column(Text)
    rules_secondary = Column(Text)

    # Raw JSON from API
    raw_json = Column(JSON)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    series = relationship("Series", back_populates="markets")
    candles = relationship("Candle", back_populates="market", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="market", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_market_series_close", "series_ticker", "close_time"),
        Index("idx_market_status", "status"),
    )

    def __repr__(self):
        return f"<Market(ticker={self.ticker}, status={self.status})>"


class Candle(Base):
    """OHLCV candlestick data aggregated from trades."""

    __tablename__ = "candles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_ticker = Column(String(100), ForeignKey("markets.ticker"), nullable=False)

    # Time period
    timestamp = Column(DateTime, nullable=False)  # UTC, start of period
    period_minutes = Column(Integer, nullable=False)  # 1 or 5

    # OHLC (prices in cents)
    open = Column(Integer, nullable=False)
    high = Column(Integer, nullable=False)
    low = Column(Integer, nullable=False)
    close = Column(Integer, nullable=False)

    # Volume
    volume = Column(Integer, nullable=False)  # total contracts traded
    num_trades = Column(Integer)  # number of individual trades

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    market = relationship("Market", back_populates="candles")

    # Unique constraint and indexes
    __table_args__ = (
        UniqueConstraint("market_ticker", "timestamp", "period_minutes", name="uq_candle"),
        Index("idx_candle_market_time", "market_ticker", "timestamp"),
        Index("idx_candle_period", "period_minutes"),
    )

    def __repr__(self):
        return f"<Candle(market={self.market_ticker}, time={self.timestamp}, period={self.period_minutes}m)>"


class Trade(Base):
    """Individual trade from Kalshi."""

    __tablename__ = "trades"

    trade_id = Column(String(100), primary_key=True)
    market_ticker = Column(String(100), ForeignKey("markets.ticker"), nullable=False)

    # Trade details (prices in cents)
    yes_price = Column(Integer, nullable=False)
    no_price = Column(Integer)
    price = Column(Float)  # normalized 0-1
    count = Column(Integer, nullable=False)  # number of contracts
    taker_side = Column(String(10), nullable=False)  # yes or no

    # Timing
    created_time = Column(DateTime, nullable=False)  # UTC

    # Raw JSON
    raw_json = Column(JSON)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    market = relationship("Market", back_populates="trades")

    # Indexes
    __table_args__ = (
        Index("idx_trade_market_time", "market_ticker", "created_time"),
    )

    def __repr__(self):
        return f"<Trade(id={self.trade_id}, market={self.market_ticker}, price={self.yes_price})>"


class WeatherObserved(Base):
    """Observed weather data from NOAA."""

    __tablename__ = "weather_observed"

    id = Column(Integer, primary_key=True, autoincrement=True)
    station_id = Column(String(50), nullable=False)
    date = Column(DateTime, nullable=False)  # Date in UTC (local midnight)

    # Temperature (both units for convenience)
    tmax_f = Column(Float)  # Fahrenheit
    tmax_c = Column(Float)  # Celsius

    # Data source
    source = Column(String(100))  # e.g., "NCEI Daily Summaries"
    raw_json = Column(JSON)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Unique constraint and indexes
    __table_args__ = (
        UniqueConstraint("station_id", "date", name="uq_weather"),
        Index("idx_weather_station_date", "station_id", "date"),
    )

    def __repr__(self):
        return f"<Weather(station={self.station_id}, date={self.date}, tmax={self.tmax_f}°F)>"


class IngestionLog(Base):
    """Track ingestion runs for incremental updates."""

    __tablename__ = "ingestion_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    series_ticker = Column(String(50), nullable=False)

    # What was fetched
    markets_fetched = Column(Integer, default=0)
    trades_fetched = Column(Integer, default=0)
    candles_1m_generated = Column(Integer, default=0)
    candles_5m_generated = Column(Integer, default=0)

    # Time range processed
    min_close_date = Column(DateTime)
    max_close_date = Column(DateTime)

    # Status
    status = Column(String(50), nullable=False)  # success, failed, partial
    error_message = Column(Text)

    # Duration
    duration_seconds = Column(Float)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Indexes
    __table_args__ = (
        Index("idx_ingestion_series_date", "series_ticker", "run_date"),
    )

    def __repr__(self):
        return f"<IngestionLog(series={self.series_ticker}, date={self.run_date}, status={self.status})>"


class WxLocation(Base):
    """Weather station location metadata for Visual Crossing."""

    __tablename__ = "location"
    __table_args__ = {"schema": "wx"}

    loc_id = Column(String(10), primary_key=True)  # e.g., "KMDW"
    vc_key = Column(String(20), nullable=False)  # e.g., "stn:KMDW"
    city = Column(String(50), nullable=False)  # e.g., "chicago"

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    observations = relationship("WxMinuteObs", back_populates="location", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<WxLocation(loc_id={self.loc_id}, city={self.city})>"


class WxMinuteObs(Base):
    """5-minute weather observations from Visual Crossing."""

    __tablename__ = "minute_obs"
    __table_args__ = (
        Index("idx_wx_minute_obs_ts", "ts_utc"),
        Index("idx_wx_minute_obs_loc_ts", "loc_id", "ts_utc"),
        {"schema": "wx"},
    )

    # Composite primary key
    loc_id = Column(String(10), ForeignKey("wx.location.loc_id", ondelete="CASCADE"), primary_key=True, nullable=False)
    ts_utc = Column(DateTime, primary_key=True, nullable=False)

    # Weather variables
    temp_f = Column(Float)  # Temperature in Fahrenheit
    humidity = Column(Float)  # Humidity percentage (0-100)
    dew_f = Column(Float)  # Dew point in Fahrenheit
    windspeed_mph = Column(Float)  # Wind speed in mph
    windgust_mph = Column(Float)  # Wind gust in mph
    pressure_mb = Column(Float)  # Pressure in millibars
    precip_in = Column(Float)  # Precipitation in inches
    preciptype = Column(String(50))  # Comma-separated: rain, snow, etc.

    # Metadata
    source = Column(String(20), default="visualcrossing")
    stations = Column(String(50))  # Station ID used by VC for this minute (e.g., "KMDW") for diagnostics
    ffilled = Column(Boolean, default=False, nullable=False)  # TRUE if forward-filled (synthetic), FALSE if real observation
    raw_json = Column(JSON)

    # Relationship
    location = relationship("WxLocation", back_populates="observations")

    def __repr__(self):
        return f"<WxMinuteObs(loc={self.loc_id}, time={self.ts_utc}, temp={self.temp_f}°F)>"


class FeatureSnapshot(Base):
    """Persisted feature vectors for ML training/debugging."""

    __tablename__ = "feature_snapshot"
    __table_args__ = (
        UniqueConstraint("market_ticker", "timestamp", "feature_set", name="uq_feature_snapshot"),
        Index("idx_feature_snapshot_city_ts", "city", "timestamp"),
        {"schema": "ml"},
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(String(50), nullable=False)
    market_ticker = Column(String(100), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    feature_set = Column(String(50), nullable=False)
    features = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<FeatureSnapshot(city={self.city}, ticker={self.market_ticker}, ts={self.timestamp})>"
