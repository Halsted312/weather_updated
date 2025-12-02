"""
SQLAlchemy models for trading.* schema.

These models map to the tables created in migration 010_create_trading_schema.py
"""

from datetime import datetime, date
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Date, ForeignKey,
    TIMESTAMP, CheckConstraint, Index, ARRAY, Text
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class TradingSession(Base):
    """
    Trading session tracking.

    One session per daemon run. Captures full config snapshot and tracks
    aggregate metrics across the session.
    """
    __tablename__ = "sessions"
    __table_args__ = {"schema": "trading"}

    session_id: UUID = Column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default="gen_random_uuid()"
    )
    started_at: datetime = Column(TIMESTAMP(timezone=True), nullable=False)
    ended_at: Optional[datetime] = Column(TIMESTAMP(timezone=True), nullable=True)
    config_json: Dict[str, Any] = Column(JSONB, nullable=False)
    status: str = Column(
        String(20),
        nullable=False,
        server_default="running",
        doc="running | stopped | error"
    )
    total_trades: int = Column(Integer, nullable=False, default=0, server_default="0")
    total_pnl_cents: int = Column(Integer, nullable=False, default=0, server_default="0")
    cities_enabled: Optional[List[str]] = Column(ARRAY(Text), nullable=True)
    dry_run: bool = Column(Boolean, nullable=False, default=True, server_default="true")

    # Relationships
    decisions = relationship("TradingDecision", back_populates="session", cascade="all, delete-orphan")
    orders = relationship("TradingOrder", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("status IN ('running', 'stopped', 'error')", name="ck_sessions_status"),
        {"schema": "trading"},
    )


class TradingDecision(Base):
    """
    Edge evaluation decision log.

    Every evaluation (trade or no-trade) is logged with full context.
    Enables analysis of edge classifier performance and decision patterns.
    """
    __tablename__ = "decisions"
    __table_args__ = (
        CheckConstraint("signal IN ('buy_high', 'buy_low', 'no_trade')", name="ck_decisions_signal"),
        Index("idx_decisions_session", "session_id", "created_at"),
        Index("idx_decisions_city_event", "city", "event_date"),
        {"schema": "trading"},
    )

    decision_id: UUID = Column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default="gen_random_uuid()"
    )
    session_id: UUID = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("trading.sessions.session_id", ondelete="CASCADE"),
        nullable=False
    )
    created_at: datetime = Column(TIMESTAMP(timezone=True), nullable=False)

    # Context
    city: str = Column(String(20), nullable=False)
    event_date: date = Column(Date, nullable=False)
    ticker: Optional[str] = Column(String(50), nullable=True)

    # Edge analysis
    forecast_implied_temp: Optional[float] = Column(Float, nullable=True)
    market_implied_temp: Optional[float] = Column(Float, nullable=True)
    edge_degf: Optional[float] = Column(Float, nullable=True)
    signal: Optional[str] = Column(String(20), nullable=True)
    edge_classifier_prob: Optional[float] = Column(Float, nullable=True)

    # Decision
    should_trade: bool = Column(Boolean, nullable=False)
    reason: Optional[str] = Column(Text, nullable=True)

    # Snapshots
    market_snapshot: Optional[Dict[str, Any]] = Column(JSONB, nullable=True)
    features_snapshot: Optional[Dict[str, Any]] = Column(JSONB, nullable=True)

    # Outcome
    order_placed: bool = Column(Boolean, nullable=False, default=False, server_default="false")
    order_id: Optional[UUID] = Column(PG_UUID(as_uuid=True), nullable=True)

    # Relationships
    session = relationship("TradingSession", back_populates="decisions")


class TradingOrder(Base):
    """
    Order lifecycle tracking.

    Tracks orders from placement → fill/cancel, including maker→taker conversions.
    """
    __tablename__ = "orders"
    __table_args__ = (
        CheckConstraint("side IN ('yes', 'no')", name="ck_orders_side"),
        CheckConstraint("action IN ('buy', 'sell')", name="ck_orders_action"),
        CheckConstraint("num_contracts > 0", name="ck_orders_num_contracts_positive"),
        CheckConstraint(
            "maker_price_cents IS NULL OR (maker_price_cents BETWEEN 1 AND 99)",
            name="ck_orders_maker_price_range"
        ),
        CheckConstraint(
            "status IN ('pending', 'filled', 'cancelled', 'converted_to_taker', 'partial_fill')",
            name="ck_orders_status"
        ),
        Index("idx_orders_session", "session_id", "created_at"),
        Index("idx_orders_ticker", "ticker", "created_at"),
        Index("idx_orders_status", "status", postgresql_where="status = 'pending'"),
        Index("idx_orders_conversion", "taker_conversion_at", postgresql_where="taker_conversion_at IS NOT NULL"),
        {"schema": "trading"},
    )

    order_id: UUID = Column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default="gen_random_uuid()"
    )
    session_id: UUID = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("trading.sessions.session_id", ondelete="CASCADE"),
        nullable=False
    )
    decision_id: Optional[UUID] = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("trading.decisions.decision_id"),
        nullable=True
    )
    created_at: datetime = Column(TIMESTAMP(timezone=True), nullable=False)

    # Market identification
    city: str = Column(String(20), nullable=False)
    event_date: date = Column(Date, nullable=False)
    ticker: str = Column(String(50), nullable=False)
    bracket_label: Optional[str] = Column(String(20), nullable=True)

    # Order details
    side: str = Column(String(10), nullable=False, doc="yes | no")
    action: str = Column(String(10), nullable=False, doc="buy | sell")
    num_contracts: int = Column(Integer, nullable=False)
    notional_usd: Optional[float] = Column(Float, nullable=True)

    # Pricing
    maker_price_cents: Optional[int] = Column(Integer, nullable=True)
    taker_conversion_at: Optional[datetime] = Column(TIMESTAMP(timezone=True), nullable=True)
    taker_price_cents: Optional[int] = Column(Integer, nullable=True)
    final_fill_price_cents: Optional[int] = Column(Integer, nullable=True)
    is_taker_fill: bool = Column(Boolean, nullable=False, default=False, server_default="false")

    # Status tracking
    status: str = Column(
        String(20),
        nullable=False,
        default="pending",
        server_default="pending",
        doc="pending | filled | cancelled | converted_to_taker | partial_fill"
    )
    status_history: List[Dict[str, Any]] = Column(JSONB, nullable=False, default=list, server_default="[]")

    # Maker→taker conversion metrics
    volume_at_order: Optional[int] = Column(Integer, nullable=True)
    maker_timeout_used_sec: Optional[int] = Column(Integer, nullable=True)

    # Settlement
    settlement_temp: Optional[float] = Column(Float, nullable=True)
    pnl_cents: Optional[int] = Column(Integer, nullable=True)

    # Relationships
    session = relationship("TradingSession", back_populates="orders")


class HealthMetric(Base):
    """
    Performance and health metrics.

    Optional table for tracking inference latency, WebSocket lag, etc.
    """
    __tablename__ = "health_metrics"
    __table_args__ = {"schema": "trading"}

    metric_id: UUID = Column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default="gen_random_uuid()"
    )
    session_id: Optional[UUID] = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("trading.sessions.session_id"),
        nullable=True
    )
    timestamp: datetime = Column(TIMESTAMP(timezone=True), nullable=False)
    metric_name: str = Column(String(50), nullable=False)
    metric_value: Optional[float] = Column(Float, nullable=True)
    metric_metadata: Optional[Dict[str, Any]] = Column("metadata", JSONB, nullable=True)
