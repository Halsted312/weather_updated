"""Create trading schema for edge-based live trading.

Revision ID: 010
Revises: 009
Create Date: 2025-12-01

Creates new trading.* schema with:
- trading.sessions: tracks daemon runs with config snapshots
- trading.decisions: logs every edge evaluation (trade or no-trade)
- trading.orders: tracks order lifecycle with maker→taker conversion
- trading.health_metrics: optional performance metrics
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic.
revision = "010"
down_revision = "8144dca12469"  # Based on current head: add_kalshi_market_events
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create trading schema
    op.execute("CREATE SCHEMA IF NOT EXISTS trading")

    # Session tracking table
    op.create_table(
        "sessions",
        sa.Column("session_id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("ended_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("config_json", JSONB, nullable=False),
        sa.Column("status", sa.VARCHAR(20), nullable=False, server_default="running"),
        sa.Column("total_trades", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_pnl_cents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cities_enabled", sa.ARRAY(sa.Text()), nullable=True),
        sa.Column("dry_run", sa.Boolean(), nullable=False, server_default="true"),
        sa.CheckConstraint("status IN ('running', 'stopped', 'error')", name="ck_sessions_status"),
        schema="trading",
    )

    # Decision logging table (every evaluation)
    op.create_table(
        "decisions",
        sa.Column("decision_id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("trading.sessions.session_id", ondelete="CASCADE")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),

        # Context
        sa.Column("city", sa.VARCHAR(20), nullable=False),
        sa.Column("event_date", sa.Date(), nullable=False),
        sa.Column("ticker", sa.VARCHAR(50), nullable=True),

        # Edge analysis
        sa.Column("forecast_implied_temp", sa.Float(), nullable=True),
        sa.Column("market_implied_temp", sa.Float(), nullable=True),
        sa.Column("edge_degf", sa.Float(), nullable=True),
        sa.Column("signal", sa.VARCHAR(20), nullable=True),
        sa.Column("edge_classifier_prob", sa.Float(), nullable=True),

        # Decision
        sa.Column("should_trade", sa.Boolean(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),

        # Snapshots
        sa.Column("market_snapshot", JSONB, nullable=True),
        sa.Column("features_snapshot", JSONB, nullable=True),

        # Outcome
        sa.Column("order_placed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("order_id", UUID(as_uuid=True), nullable=True),

        sa.CheckConstraint("signal IN ('buy_high', 'buy_low', 'no_trade')", name="ck_decisions_signal"),
        schema="trading",
    )

    # Create indexes for decisions
    op.create_index(
        "idx_decisions_session",
        "decisions",
        ["session_id", "created_at"],
        schema="trading",
    )
    op.create_index(
        "idx_decisions_city_event",
        "decisions",
        ["city", "event_date"],
        schema="trading",
    )

    # Order lifecycle tracking table
    op.create_table(
        "orders",
        sa.Column("order_id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("trading.sessions.session_id", ondelete="CASCADE")),
        sa.Column("decision_id", UUID(as_uuid=True), sa.ForeignKey("trading.decisions.decision_id")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),

        # Market identification
        sa.Column("city", sa.VARCHAR(20), nullable=False),
        sa.Column("event_date", sa.Date(), nullable=False),
        sa.Column("ticker", sa.VARCHAR(50), nullable=False),
        sa.Column("bracket_label", sa.VARCHAR(20), nullable=True),

        # Order details
        sa.Column("side", sa.VARCHAR(10), nullable=False),
        sa.Column("action", sa.VARCHAR(10), nullable=False),
        sa.Column("num_contracts", sa.Integer(), nullable=False),
        sa.Column("notional_usd", sa.Float(), nullable=True),

        # Pricing
        sa.Column("maker_price_cents", sa.Integer(), nullable=True),
        sa.Column("taker_conversion_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("taker_price_cents", sa.Integer(), nullable=True),
        sa.Column("final_fill_price_cents", sa.Integer(), nullable=True),
        sa.Column("is_taker_fill", sa.Boolean(), nullable=False, server_default="false"),

        # Status tracking
        sa.Column("status", sa.VARCHAR(20), nullable=False, server_default="pending"),
        sa.Column("status_history", JSONB, nullable=False, server_default="[]"),

        # Maker→taker conversion metrics
        sa.Column("volume_at_order", sa.Integer(), nullable=True),
        sa.Column("maker_timeout_used_sec", sa.Integer(), nullable=True),

        # Settlement
        sa.Column("settlement_temp", sa.Float(), nullable=True),
        sa.Column("pnl_cents", sa.Integer(), nullable=True),

        sa.CheckConstraint("side IN ('yes', 'no')", name="ck_orders_side"),
        sa.CheckConstraint("action IN ('buy', 'sell')", name="ck_orders_action"),
        sa.CheckConstraint("num_contracts > 0", name="ck_orders_num_contracts_positive"),
        sa.CheckConstraint("maker_price_cents IS NULL OR (maker_price_cents BETWEEN 1 AND 99)", name="ck_orders_maker_price_range"),
        sa.CheckConstraint("status IN ('pending', 'filled', 'cancelled', 'converted_to_taker', 'partial_fill')", name="ck_orders_status"),
        schema="trading",
    )

    # Create indexes for orders
    op.create_index(
        "idx_orders_session",
        "orders",
        ["session_id", "created_at"],
        schema="trading",
    )
    op.create_index(
        "idx_orders_ticker",
        "orders",
        ["ticker", "created_at"],
        schema="trading",
    )
    op.create_index(
        "idx_orders_status",
        "orders",
        ["status"],
        schema="trading",
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index(
        "idx_orders_conversion",
        "orders",
        ["taker_conversion_at"],
        schema="trading",
        postgresql_where=sa.text("taker_conversion_at IS NOT NULL"),
    )

    # Optional: health metrics table
    op.create_table(
        "health_metrics",
        sa.Column("metric_id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("trading.sessions.session_id")),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("metric_name", sa.VARCHAR(50), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=True),
        sa.Column("metadata", JSONB, nullable=True),
        schema="trading",
    )


def downgrade() -> None:
    # Drop tables in reverse order (respects foreign keys)
    op.drop_table("health_metrics", schema="trading")
    op.drop_table("orders", schema="trading")
    op.drop_table("decisions", schema="trading")
    op.drop_table("sessions", schema="trading")

    # Drop schema
    op.execute("DROP SCHEMA IF EXISTS trading CASCADE")
