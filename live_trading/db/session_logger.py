"""
Session logger for trading decisions and orders.

Provides methods to log:
- Session start/end
- Every edge evaluation decision (trade or no-trade)
- Order placement and lifecycle
- Health metrics
"""

import logging
from datetime import datetime, date
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from live_trading.db.models import TradingSession, TradingDecision, TradingOrder, HealthMetric
from live_trading.config import TradingConfig
from src.db.connection import get_db_session

logger = logging.getLogger(__name__)


class SessionLogger:
    """Logs trading session data to trading.* schema."""

    def __init__(self):
        """Initialize session logger."""
        self.current_session_id: Optional[UUID] = None

    def start_session(
        self,
        config: TradingConfig,
        dry_run: bool
    ) -> UUID:
        """
        Start a new trading session.

        Args:
            config: Trading configuration snapshot
            dry_run: Whether this is a dry-run session

        Returns:
            session_id for this run
        """
        session_id = uuid4()

        with get_db_session() as db:
            session = TradingSession(
                session_id=session_id,
                started_at=datetime.now(),
                config_json=config.to_json(),
                status="running",
                cities_enabled=config.enabled_cities,
                dry_run=dry_run
            )
            db.add(session)

        self.current_session_id = session_id
        logger.info(
            f"Started session {session_id} "
            f"(dry_run={dry_run}, cities={config.enabled_cities})"
        )

        return session_id

    def end_session(
        self,
        session_id: UUID,
        status: str = "stopped"
    ) -> None:
        """
        Mark session as ended.

        Args:
            session_id: Session to end
            status: Final status (stopped | error)
        """
        with get_db_session() as db:
            session = db.query(TradingSession).filter_by(session_id=session_id).first()
            if session:
                session.ended_at = datetime.now()
                session.status = status

                logger.info(
                    f"Ended session {session_id} "
                    f"(status={status}, trades={session.total_trades}, "
                    f"pnl=${session.total_pnl_cents/100:.2f})"
                )

    def log_decision(
        self,
        session_id: UUID,
        city: str,
        event_date: date,
        edge_decision: Any,  # EdgeDecision from inference.py
        market_snapshot: Dict[str, Any],
        features_snapshot: Optional[Dict[str, Any]] = None,
        order_id: Optional[UUID] = None
    ) -> UUID:
        """
        Log an edge evaluation decision (trade or no-trade).

        Args:
            session_id: Current session
            city: City being evaluated
            event_date: Event date being evaluated
            edge_decision: EdgeDecision object from inference
            market_snapshot: Market state at decision time
            features_snapshot: Features used in edge classifier
            order_id: If order was placed, the order UUID

        Returns:
            decision_id
        """
        decision_id = uuid4()

        with get_db_session() as db:
            decision = TradingDecision(
                decision_id=decision_id,
                session_id=session_id,
                created_at=datetime.now(),
                city=city,
                event_date=event_date,
                ticker=market_snapshot.get('ticker'),
                forecast_implied_temp=edge_decision.forecast_implied_temp,
                market_implied_temp=edge_decision.market_implied_temp,
                edge_degf=edge_decision.edge_degf,
                signal=edge_decision.signal,
                edge_classifier_prob=edge_decision.edge_classifier_prob,
                should_trade=edge_decision.should_trade,
                reason=edge_decision.reason,
                market_snapshot=market_snapshot,
                features_snapshot=features_snapshot,
                order_placed=(order_id is not None),
                order_id=order_id
            )
            db.add(decision)

        logger.debug(
            f"Logged decision {decision_id}: {city} {event_date} "
            f"edge={edge_decision.edge_degf:.2f}°F "
            f"should_trade={edge_decision.should_trade}"
        )

        return decision_id

    def update_decision_with_order(
        self,
        decision_id: UUID,
        order_id: UUID
    ) -> None:
        """
        Update decision to link it with the order that was placed.

        Args:
            decision_id: Decision to update
            order_id: Order that was placed
        """
        with get_db_session() as db:
            decision = db.query(TradingDecision).filter_by(decision_id=decision_id).first()
            if decision:
                decision.order_placed = True
                decision.order_id = order_id
                logger.debug(f"Linked decision {decision_id} to order {order_id}")

    def log_order(
        self,
        order_id: UUID,
        session_id: UUID,
        decision_id: Optional[UUID],
        city: str,
        event_date: date,
        ticker: str,
        bracket_label: str,
        side: str,
        action: str,
        num_contracts: int,
        maker_price_cents: int,
        notional_usd: float,
        volume_at_order: Optional[int] = None,
        maker_timeout_used_sec: Optional[int] = None
    ) -> UUID:
        """
        Log an order placement.

        Args:
            order_id: Real Kalshi order ID (from API response)
            session_id: Current session
            decision_id: Decision that triggered this order
            city: City
            event_date: Event date
            ticker: Kalshi market ticker
            bracket_label: Bracket label (e.g., "82-83", ">90")
            side: "yes" or "no"
            action: "buy" or "sell"
            num_contracts: Number of contracts
            maker_price_cents: Limit price in cents
            notional_usd: Notional value in USD
            volume_at_order: Recent volume when order placed
            maker_timeout_used_sec: Timeout before taker conversion

        Returns:
            order_id (same as input, for convenience)
        """

        with get_db_session() as db:
            order = TradingOrder(
                order_id=order_id,
                session_id=session_id,
                decision_id=decision_id,
                created_at=datetime.now(),
                city=city,
                event_date=event_date,
                ticker=ticker,
                bracket_label=bracket_label,
                side=side,
                action=action,
                num_contracts=num_contracts,
                notional_usd=notional_usd,
                maker_price_cents=maker_price_cents,
                status="pending",
                status_history=[{
                    "status": "pending",
                    "timestamp": datetime.now().isoformat(),
                    "note": "Order placed as maker"
                }],
                volume_at_order=volume_at_order,
                maker_timeout_used_sec=maker_timeout_used_sec
            )
            db.add(order)

        logger.info(
            f"Logged order {order_id}: {action} {side} {num_contracts}x {ticker} @ {maker_price_cents}¢"
        )

        return order_id

    def update_order_status(
        self,
        order_id: UUID,
        new_status: str,
        note: str,
        taker_conversion_at: Optional[datetime] = None,
        taker_price_cents: Optional[int] = None,
        final_fill_price_cents: Optional[int] = None,
        is_taker_fill: Optional[bool] = None
    ) -> None:
        """
        Update order status and append to status history.

        Args:
            order_id: Order to update
            new_status: New status value
            note: Explanation for status change
            taker_conversion_at: Timestamp if converted to taker
            taker_price_cents: Price if converted to taker
            final_fill_price_cents: Actual fill price
            is_taker_fill: Whether filled as taker
        """
        with get_db_session() as db:
            order = db.query(TradingOrder).filter_by(order_id=order_id).first()
            if not order:
                logger.warning(f"Order {order_id} not found for status update")
                return

            # Append to status history
            status_entry = {
                "status": new_status,
                "timestamp": datetime.now().isoformat(),
                "note": note
            }
            history = order.status_history or []
            history.append(status_entry)

            # Update fields
            order.status = new_status
            order.status_history = history

            if taker_conversion_at:
                order.taker_conversion_at = taker_conversion_at
            if taker_price_cents:
                order.taker_price_cents = taker_price_cents
            if final_fill_price_cents:
                order.final_fill_price_cents = final_fill_price_cents
            if is_taker_fill is not None:
                order.is_taker_fill = is_taker_fill

            logger.info(f"Updated order {order_id}: {new_status} - {note}")

    def record_settlement(
        self,
        order_id: UUID,
        settlement_temp: float,
        pnl_cents: int
    ) -> None:
        """
        Record settlement outcome for an order.

        Args:
            order_id: Order that settled
            settlement_temp: Actual settlement temperature
            pnl_cents: P&L in cents (positive = profit)
        """
        with get_db_session() as db:
            order = db.query(TradingOrder).filter_by(order_id=order_id).first()
            if not order:
                logger.warning(f"Order {order_id} not found for settlement")
                return

            order.settlement_temp = settlement_temp
            order.pnl_cents = pnl_cents

            # Update session total P&L
            session = db.query(TradingSession).filter_by(session_id=order.session_id).first()
            if session:
                session.total_pnl_cents = (session.total_pnl_cents or 0) + pnl_cents

            logger.info(
                f"Recorded settlement for {order_id}: "
                f"temp={settlement_temp:.1f}°F pnl=${pnl_cents/100:.2f}"
            )

    def log_health_metric(
        self,
        metric_name: str,
        metric_value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a health/performance metric.

        Args:
            metric_name: Name of metric (e.g., "inference_latency_ms")
            metric_value: Numeric value
            metadata: Additional context
        """
        with get_db_session() as db:
            metric = HealthMetric(
                session_id=self.current_session_id,
                timestamp=datetime.now(),
                metric_name=metric_name,
                metric_value=metric_value,
                metric_metadata=metadata
            )
            db.add(metric)

        logger.debug(f"Health metric: {metric_name}={metric_value}")
