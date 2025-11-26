"""
Retry decorators with exponential backoff using tenacity.

Provides pre-configured retry decorators for different API providers.
"""

import logging
from functools import wraps
from typing import Callable, Tuple, Type, TypeVar

import requests
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# Type variable for generic function decoration
F = TypeVar("F", bound=Callable)


def is_rate_limit_error(exception: BaseException) -> bool:
    """Check if an exception is a rate limit error (429)."""
    if isinstance(exception, requests.HTTPError):
        return exception.response is not None and exception.response.status_code == 429
    return False


def create_retry_decorator(
    max_attempts: int = 5,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (
        requests.exceptions.RequestException,
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    ),
) -> Callable[[F], F]:
    """
    Create a retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exceptions: Tuple of exception types to retry on

    Returns:
        Decorator function
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# Pre-configured decorators for different APIs

# Kalshi API - 5 attempts, 1-30s exponential backoff
kalshi_retry = create_retry_decorator(
    max_attempts=5,
    min_wait=1.0,
    max_wait=30.0,
)

# Visual Crossing API - 3 attempts, faster retry (unlimited quota)
vc_retry = create_retry_decorator(
    max_attempts=3,
    min_wait=0.5,
    max_wait=10.0,
)

# NWS/NOAA API - 5 attempts, polite retry (be nice to gov servers)
nws_retry = create_retry_decorator(
    max_attempts=5,
    min_wait=2.0,
    max_wait=60.0,
)


def with_rate_limiter(limiter, retry_decorator=None):
    """
    Decorator that combines rate limiting with optional retry.

    Usage:
        @with_rate_limiter(KALSHI_LIMITER, kalshi_retry)
        def make_api_call():
            ...

    Args:
        limiter: AdaptiveRateLimiter instance
        retry_decorator: Optional tenacity retry decorator

    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Wait for rate limit token
            limiter.wait()

            try:
                result = func(*args, **kwargs)
                limiter.report_success()
                return result
            except requests.HTTPError as e:
                # Check for rate limit error
                is_429 = e.response is not None and e.response.status_code == 429
                limiter.report_error(is_rate_limit=is_429)
                raise
            except Exception as e:
                limiter.report_error(is_rate_limit=False)
                raise

        # Apply retry decorator if provided
        if retry_decorator:
            wrapper = retry_decorator(wrapper)

        return wrapper
    return decorator
