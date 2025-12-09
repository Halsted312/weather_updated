"""
Daily loss tracker with circuit breaker.

Extends PositionTracker with explicit circuit breaker functionality
for automatic trading daemons.
"""

import logging
from datetime import date, datetime
from typing import Tuple

from live_trading.config import TradingConfig
from live_trading.position_tracker import PositionTracker

logger = logging.getLogger(__name__)


class DailyLossTracker(PositionTracker):
    """
    Position tracker with enhanced daily loss circuit breaker.

    Adds explicit circuit breaker state tracking and convenience
    methods for automatic trading systems.
    """

    def __init__(self, config: TradingConfig):
        """
        Initialize daily loss tracker.

        Args:
            config: Trading configuration with max_daily_loss_usd
        """
        super().__init__(config)

        # Circuit breaker state
        self.daily_loss_limit_hit = False
        self.circuit_breaker_triggered_at: datetime | None = None
        self.session_start_time = datetime.now()

    def check_daily_loss_limit(self, day: date | None = None) -> Tuple[bool, str]:
        """
        Check if daily loss limit has been reached.

        This is a convenience method that explicitly checks the loss limit
        and returns a clear boolean + message for circuit breaker logic.

        Args:
            day: Date to check (default: today)

        Returns:
            (within_limit, message)
            - within_limit: True if trading can continue, False if limit hit
            - message: Human-readable status message
        """
        if day is None:
            day = date.today()

        daily_pnl_usd = self.get_daily_pnl_usd(day)

        if daily_pnl_usd < 0:
            # We have a loss
            loss_usd = abs(daily_pnl_usd)

            if loss_usd >= self.config.max_daily_loss_usd:
                # Circuit breaker triggered
                if not self.daily_loss_limit_hit:
                    self.daily_loss_limit_hit = True
                    self.circuit_breaker_triggered_at = datetime.now()
                    logger.critical(
                        f"🔴 CIRCUIT BREAKER TRIGGERED: "
                        f"Daily loss ${loss_usd:.2f} >= ${self.config.max_daily_loss_usd:.2f}"
                    )

                return False, (
                    f"Circuit breaker: Daily loss ${loss_usd:.2f} >= "
                    f"${self.config.max_daily_loss_usd:.2f} limit"
                )

            # Loss exists but below limit
            return True, f"Daily P&L: -${loss_usd:.2f} / ${self.config.max_daily_loss_usd:.2f} limit"

        elif daily_pnl_usd > 0:
            # We're profitable
            return True, f"Daily P&L: +${daily_pnl_usd:.2f} (profitable day)"

        else:
            # Break-even
            return True, "Daily P&L: $0.00 (break-even)"

    def can_trade(
        self,
        city: str,
        event_date: date,
        ignore_circuit_breaker: bool = False
    ) -> Tuple[bool, str]:
        """
        Combined check: position limits AND daily loss limit.

        This is the primary method for automatic traders to check
        if a new trade can be opened.

        Args:
            city: City for new position
            event_date: Event date for new position
            ignore_circuit_breaker: If True, skip circuit breaker check

        Returns:
            (can_trade, reason)
        """
        # First check position limits (parent class)
        can_open, reason = self.can_open_position(
            city=city,
            event_date=event_date,
            check_daily_loss=False  # We'll check explicitly below
        )

        if not can_open:
            return False, reason

        # Check daily loss circuit breaker
        if not ignore_circuit_breaker:
            within_limit, loss_msg = self.check_daily_loss_limit()
            if not within_limit:
                return False, loss_msg

        return True, "OK - all checks passed"

    def reset_circuit_breaker(self) -> None:
        """
        Manually reset circuit breaker.

        Use with caution! Typically called:
        - At start of new trading day
        - After manual intervention/review
        - For testing purposes
        """
        self.daily_loss_limit_hit = False
        self.circuit_breaker_triggered_at = None
        logger.warning("Circuit breaker manually reset")

    def get_circuit_breaker_status(self) -> dict:
        """
        Get current circuit breaker status.

        Returns:
            Dict with status information
        """
        today = date.today()
        daily_pnl_usd = self.get_daily_pnl_usd(today)

        status = {
            'triggered': self.daily_loss_limit_hit,
            'triggered_at': self.circuit_breaker_triggered_at,
            'daily_pnl_usd': daily_pnl_usd,
            'daily_loss_limit_usd': self.config.max_daily_loss_usd,
        }

        if daily_pnl_usd < 0:
            status['loss_pct'] = abs(daily_pnl_usd) / self.config.max_daily_loss_usd * 100
        else:
            status['loss_pct'] = 0.0

        return status

    def log_circuit_breaker_status(self) -> None:
        """Log current circuit breaker status."""
        status = self.get_circuit_breaker_status()

        if status['triggered']:
            logger.warning(
                f"🔴 Circuit breaker ACTIVE: "
                f"Daily P&L ${status['daily_pnl_usd']:.2f}, "
                f"Limit ${status['daily_loss_limit_usd']:.2f}, "
                f"Triggered at {status['triggered_at']}"
            )
        else:
            loss_pct = status['loss_pct']
            if loss_pct > 75:
                level = "🟠 WARNING"
            elif loss_pct > 50:
                level = "🟡 CAUTION"
            else:
                level = "🟢 OK"

            logger.info(
                f"{level}: Daily P&L ${status['daily_pnl_usd']:.2f}, "
                f"Limit ${status['daily_loss_limit_usd']:.2f} "
                f"({loss_pct:.1f}% of limit)"
            )

    def should_pause_trading(self, cooldown_minutes: int = 5) -> bool:
        """
        Check if trading should be paused due to circuit breaker.

        Args:
            cooldown_minutes: Minutes to wait after circuit breaker before checking

        Returns:
            True if trading should pause
        """
        if not self.daily_loss_limit_hit:
            return False

        if self.circuit_breaker_triggered_at is None:
            return True  # Circuit breaker active, no cooldown info

        # Check if we're still in cooldown period
        elapsed = datetime.now() - self.circuit_breaker_triggered_at
        elapsed_minutes = elapsed.total_seconds() / 60

        if elapsed_minutes < cooldown_minutes:
            return True  # Still in cooldown

        # Cooldown expired, but loss limit still hit
        # Continue pausing until day rolls over or manual reset
        return True

    def start_new_day(self) -> None:
        """
        Start a new trading day.

        Resets circuit breaker and daily P&L tracking.
        Call this at the beginning of each trading day.
        """
        today = date.today()
        logger.info(f"Starting new trading day: {today}")

        # Reset circuit breaker
        self.reset_circuit_breaker()

        # Reset daily P&L (keeps last 7 days of history)
        self.reset_daily_pnl()

        # Update session start time
        self.session_start_time = datetime.now()

        logger.info("New trading day initialized")
