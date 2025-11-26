"""
Adaptive rate limiter with automatic backoff on errors.

For Kalshi Advanced tier: 30 reads/sec, 30 writes/sec.
Default: 28 req/sec with automatic reduction on 429/rate limit errors.
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class AdaptiveRateLimiter:
    """
    Token bucket rate limiter with adaptive backoff.

    Automatically reduces rate when errors occur and gradually
    recovers back to max rate when requests succeed.
    """

    def __init__(
        self,
        max_rate: float = 28.0,
        min_rate: float = 5.0,
        burst: int = 5,
        backoff_factor: float = 0.5,
        recovery_factor: float = 1.1,
        recovery_interval: float = 10.0,
    ):
        """
        Initialize adaptive rate limiter.

        Args:
            max_rate: Maximum requests per second (default 28 for Kalshi)
            min_rate: Minimum rate to back off to (floor)
            burst: Maximum burst size (tokens that can accumulate)
            backoff_factor: Multiply rate by this on error (e.g., 0.5 = halve)
            recovery_factor: Multiply rate by this on recovery (e.g., 1.1 = +10%)
            recovery_interval: Seconds of success before attempting recovery
        """
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.current_rate = max_rate
        self.burst = burst
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor
        self.recovery_interval = recovery_interval

        self.tokens = float(burst)
        self.last_update = time.monotonic()
        self.last_error_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.consecutive_successes = 0
        self.lock = threading.Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token, blocking if necessary.

        Args:
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            True if token acquired, False if timeout
        """
        deadline = time.monotonic() + timeout if timeout else float("inf")

        while True:
            with self.lock:
                now = time.monotonic()

                # Check if we should try to recover rate
                self._maybe_recover(now)

                # Refill tokens based on time elapsed
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.current_rate)
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return True

            if time.monotonic() >= deadline:
                return False

            # Wait for next token (avoid busy waiting)
            wait_time = (1.0 - self.tokens) / self.current_rate if self.current_rate > 0 else 0.1
            time.sleep(min(wait_time, 0.1))

    def wait(self) -> None:
        """Block until a token is available."""
        self.acquire(timeout=None)

    def report_success(self) -> None:
        """Report a successful request (for rate recovery)."""
        with self.lock:
            self.last_success_time = time.monotonic()
            self.consecutive_successes += 1

    def report_error(self, is_rate_limit: bool = False) -> None:
        """
        Report an error. If it's a rate limit error, back off immediately.

        Args:
            is_rate_limit: True if this was a 429 or rate limit error
        """
        with self.lock:
            now = time.monotonic()
            self.last_error_time = now
            self.consecutive_successes = 0

            if is_rate_limit:
                old_rate = self.current_rate
                self.current_rate = max(
                    self.min_rate,
                    self.current_rate * self.backoff_factor
                )
                if self.current_rate != old_rate:
                    logger.warning(
                        f"Rate limit hit! Reducing rate: {old_rate:.1f} -> {self.current_rate:.1f} req/sec"
                    )

    def _maybe_recover(self, now: float) -> None:
        """Try to recover rate if we've had sustained success."""
        if self.current_rate >= self.max_rate:
            return

        if self.last_success_time is None:
            return

        # Only recover if we've had success for recovery_interval seconds
        time_since_error = now - (self.last_error_time or 0)
        if time_since_error < self.recovery_interval:
            return

        # Gradually increase rate
        old_rate = self.current_rate
        self.current_rate = min(
            self.max_rate,
            self.current_rate * self.recovery_factor
        )

        if self.current_rate != old_rate:
            logger.info(
                f"Rate recovering: {old_rate:.1f} -> {self.current_rate:.1f} req/sec"
            )

    @property
    def rate(self) -> float:
        """Current rate (for compatibility)."""
        return self.current_rate

    def get_status(self) -> dict:
        """Get current limiter status for debugging."""
        with self.lock:
            return {
                "current_rate": self.current_rate,
                "max_rate": self.max_rate,
                "min_rate": self.min_rate,
                "tokens": self.tokens,
                "consecutive_successes": self.consecutive_successes,
            }


def get_kalshi_rate_limiter(num_parallel_workers: int = 1) -> AdaptiveRateLimiter:
    """
    Get a rate limiter configured for Kalshi API.

    Kalshi Advanced tier: 30 reads/sec, 30 writes/sec
    Default: 28 req/sec with adaptive backoff.

    Args:
        num_parallel_workers: Number of parallel processes sharing the limit.
            Each worker gets rate = 28 / num_parallel_workers

    Returns:
        AdaptiveRateLimiter configured for this worker
    """
    global_rate = 28.0  # Kalshi Advanced tier, with small buffer
    per_worker_rate = global_rate / num_parallel_workers
    min_rate = max(1.0, 5.0 / num_parallel_workers)
    burst = max(2, int(5 / num_parallel_workers))

    logger.info(
        f"Kalshi rate limiter: {per_worker_rate:.1f} req/sec "
        f"(1 of {num_parallel_workers} workers, min={min_rate:.1f}, burst={burst})"
    )

    return AdaptiveRateLimiter(
        max_rate=per_worker_rate,
        min_rate=min_rate,
        burst=burst,
    )


# Default singleton for single-process usage (28 req/sec)
# For multi-process, create your own with get_kalshi_rate_limiter(num_workers)
KALSHI_LIMITER = AdaptiveRateLimiter(max_rate=28.0, min_rate=5.0, burst=5)
