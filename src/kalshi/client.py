"""
Kalshi API Client

Handles authentication with API key + private key (PEM) and provides
methods to fetch series, markets, candlesticks, and trades.

Features:
- Adaptive rate limiting (28 req/sec, backs off on 429 errors)
- Automatic retry with exponential backoff
- JWT authentication
"""

import base64
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import requests  # Note: jwt no longer needed with RSA-PSS signature auth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.kalshi.schemas import (
    Candle,
    CandlestickResponse,
    EventCandlestickResponse,
    Market,
    MarketsResponse,
    SeriesResponse,
    Trade,
    TradesResponse,
)
from src.utils.rate_limiter import AdaptiveRateLimiter, get_kalshi_rate_limiter

logger = logging.getLogger(__name__)


class KalshiClient:
    """Client for Kalshi API with authentication and adaptive rate limiting."""

    def __init__(
        self,
        api_key: str,
        private_key_path: str,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
        rate_limiter: Optional[AdaptiveRateLimiter] = None,
        num_parallel_workers: int = 1,
    ):
        """
        Initialize Kalshi API client.

        Args:
            api_key: Kalshi API key ID
            private_key_path: Path to PEM private key file
            base_url: Base URL for Kalshi API
            rate_limiter: Optional custom rate limiter. If None, creates one
                based on num_parallel_workers.
            num_parallel_workers: Number of parallel workers sharing Kalshi quota.
                Used to create rate limiter if none provided. Default 1 = 28 req/sec.
                With 6 parallel workers, each gets ~4.7 req/sec.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

        # Setup adaptive rate limiter
        if rate_limiter:
            self.rate_limiter = rate_limiter
        else:
            self.rate_limiter = get_kalshi_rate_limiter(num_parallel_workers)

        # Load private key
        private_key_file = Path(private_key_path)
        if not private_key_file.exists():
            raise FileNotFoundError(f"Private key not found: {private_key_path}")

        with open(private_key_file, "rb") as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
            )

        # Setup session with HTTP-level retries for transient errors
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],  # NOT 429 - we handle that
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        logger.info(
            f"Kalshi client initialized: {self.base_url} "
            f"(rate: {self.rate_limiter.current_rate:.1f} req/sec)"
        )

    def _generate_signature(self, method: str, path: str, timestamp_ms: int) -> str:
        """
        Generate RSA-PSS signature for Kalshi API authentication.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path without query parameters
            timestamp_ms: Current timestamp in milliseconds

        Returns:
            Base64-encoded signature string
        """
        # Message to sign: timestamp + method + path
        message = f"{timestamp_ms}{method}{path}".encode("utf-8")

        # Sign with RSA-PSS padding and SHA256
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        return base64.b64encode(signature).decode("utf-8")

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
    ) -> Dict[str, Any]:
        """
        Make authenticated request to Kalshi API with rate limiting and retry.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            params: Query parameters
            json_data: JSON body for POST requests
            max_retries: Maximum retry attempts for rate limit errors

        Returns:
            Response JSON as dict

        Raises:
            requests.HTTPError: On HTTP errors after retries exhausted
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # Extract path without query params for signature
        parsed = urlparse(url)
        path = parsed.path

        last_exception = None

        for attempt in range(max_retries):
            # Wait for rate limiter token
            self.rate_limiter.wait()

            # Generate fresh timestamp and signature for each attempt
            timestamp_ms = int(time.time() * 1000)
            signature = self._generate_signature(method, path, timestamp_ms)

            headers = {
                "KALSHI-ACCESS-KEY": self.api_key,
                "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
                "KALSHI-ACCESS-SIGNATURE": signature,
                "Content-Type": "application/json",
            }

            logger.debug(f"{method} {url} params={params}")

            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_data,
                    timeout=30,
                )

                # Check for rate limit error
                if response.status_code == 429:
                    self.rate_limiter.report_error(is_rate_limit=True)
                    retry_after = int(response.headers.get("Retry-After", 2))
                    logger.warning(
                        f"Rate limited (429). Backing off {retry_after}s. "
                        f"Attempt {attempt + 1}/{max_retries}"
                    )
                    time.sleep(retry_after)
                    last_exception = requests.HTTPError(
                        f"429 Rate Limited", response=response
                    )
                    continue

                response.raise_for_status()
                self.rate_limiter.report_success()
                return response.json()

            except requests.exceptions.RequestException as e:
                self.rate_limiter.report_error(is_rate_limit=False)
                last_exception = e
                logger.warning(
                    f"Request error: {e}. Attempt {attempt + 1}/{max_retries}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue

        # All retries exhausted
        if last_exception:
            raise last_exception
        raise requests.HTTPError(f"Request failed after {max_retries} attempts")

    def get_series(self, series_ticker: str) -> SeriesResponse:
        """
        Get series metadata.

        Args:
            series_ticker: Series ticker (e.g., "KXHIGHCHI")

        Returns:
            SeriesResponse with series metadata
        """
        logger.info(f"Fetching series: {series_ticker}")
        data = self._request("GET", f"/series/{series_ticker}")
        return SeriesResponse(**data)

    def get_markets(
        self,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> MarketsResponse:
        """
        Get markets with pagination support.

        Args:
            series_ticker: Filter by series ticker
            status: Filter by status (e.g., "closed,settled")
            min_close_ts: Minimum close timestamp (Unix seconds)
            max_close_ts: Maximum close timestamp (Unix seconds)
            limit: Page size (max 100)
            cursor: Pagination cursor

        Returns:
            MarketsResponse with markets list and cursor
        """
        params: Dict[str, Any] = {"limit": limit}

        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if min_close_ts:
            params["min_close_ts"] = min_close_ts
        if max_close_ts:
            params["max_close_ts"] = max_close_ts
        if cursor:
            params["cursor"] = cursor

        logger.info(f"Fetching markets with params: {params}")
        data = self._request("GET", "/markets", params=params)
        return MarketsResponse(**data)

    def get_all_markets(
        self,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
    ) -> List[Market]:
        """
        Get all markets with automatic pagination.

        Args:
            series_ticker: Filter by series ticker
            status: Filter by status
            min_close_ts: Minimum close timestamp
            max_close_ts: Maximum close timestamp

        Returns:
            List of all Market objects
        """
        all_markets: List[Market] = []
        cursor: Optional[str] = None

        while True:
            response = self.get_markets(
                series_ticker=series_ticker,
                status=status,
                min_close_ts=min_close_ts,
                max_close_ts=max_close_ts,
                cursor=cursor,
            )

            all_markets.extend(response.markets)
            cursor = response.cursor

            if not cursor:
                break

            logger.info(f"Fetched {len(all_markets)} markets so far...")

        logger.info(f"Total markets fetched: {len(all_markets)}")
        return all_markets

    def get_market(self, ticker: str) -> Market:
        """
        Get single market details.

        Args:
            ticker: Market ticker

        Returns:
            Market details
        """
        logger.info(f"Fetching market: {ticker}")
        data = self._request("GET", f"/markets/{ticker}")
        return Market(**data.get("market", data))

    def get_market_candlesticks(
        self,
        series_ticker: str,
        market_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> CandlestickResponse:
        """
        Get market candlesticks (OHLC data).

        Args:
            series_ticker: Series ticker
            market_ticker: Market ticker
            start_ts: Start timestamp (Unix seconds)
            end_ts: End timestamp (Unix seconds)
            period_interval: Period in minutes (1, 60, or 1440)

        Returns:
            CandlestickResponse with candles list
        """
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }

        logger.debug(
            f"Fetching candlesticks for {market_ticker} "
            f"from {start_ts} to {end_ts} (interval={period_interval}m)"
        )

        data = self._request(
            "GET",
            f"/series/{series_ticker}/markets/{market_ticker}/candlesticks",
            params=params,
        )
        return CandlestickResponse(**data)

    def get_all_candlesticks(
        self,
        series_ticker: str,
        market_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> List[Candle]:
        """
        Get all candlesticks with automatic pagination using adjusted_end_ts.

        Args:
            series_ticker: Series ticker
            market_ticker: Market ticker
            start_ts: Start timestamp (Unix seconds)
            end_ts: End timestamp (Unix seconds)
            period_interval: Period in minutes (1, 60, or 1440)

        Returns:
            List of all Candle objects
        """
        all_candles: List[Candle] = []
        current_end_ts = end_ts

        while True:
            response = self.get_market_candlesticks(
                series_ticker=series_ticker,
                market_ticker=market_ticker,
                start_ts=start_ts,
                end_ts=current_end_ts,
                period_interval=period_interval,
            )

            if not response.candles:
                break

            all_candles.extend(response.candles)

            # Use adjusted_end_ts for pagination if available
            if response.adjusted_end_ts and response.adjusted_end_ts > start_ts:
                current_end_ts = response.adjusted_end_ts
            else:
                break

            logger.info(f"Fetched {len(all_candles)} candles so far...")

        # Sort by timestamp ascending
        all_candles.sort(key=lambda c: c.end_period_ts)
        logger.info(f"Total candles fetched for {market_ticker}: {len(all_candles)}")
        return all_candles

    def get_event_candlesticks(
        self,
        series_ticker: str,
        event_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> EventCandlestickResponse:
        """
        Get event candlesticks (all markets in one call).

        Event candlesticks return data for all markets within an event:
        - More efficient than per-market calls (one API call for all brackets)
        - Returns market_tickers list and market_candlesticks dict
        - Works for weather markets (unlike per-market endpoint)

        Args:
            series_ticker: Series ticker (e.g., "KXHIGHCHI")
            event_ticker: Event ticker (e.g., "KXHIGHCHI-25NOV24")
            start_ts: Start timestamp (Unix seconds)
            end_ts: End timestamp (Unix seconds)
            period_interval: Period in minutes (1, 60, or 1440)

        Returns:
            EventCandlestickResponse with candles for all markets in the event
        """
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }

        logger.debug(
            f"Fetching event candlesticks for {event_ticker} "
            f"from {start_ts} to {end_ts}"
        )

        data = self._request(
            "GET",
            f"/series/{series_ticker}/events/{event_ticker}/candlesticks",
            params=params,
        )
        return EventCandlestickResponse(**data)

    def get_all_event_candlesticks(
        self,
        series_ticker: str,
        event_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> Dict[str, List[Candle]]:
        """
        Get all event candlesticks with pagination support.

        NOTE: Pagination works backwards from end_ts using adjusted_end_ts.
        The API returns data up to adjusted_end_ts, so we track seen timestamps
        to avoid infinite loops.

        Args:
            series_ticker: Series ticker
            event_ticker: Event ticker
            start_ts: Start timestamp (Unix seconds)
            end_ts: End timestamp (Unix seconds)
            period_interval: Period in minutes (1, 60, or 1440)

        Returns:
            Dict mapping market ticker to list of Candle objects
        """
        all_candles: Dict[str, List[Candle]] = {}
        current_end_ts = end_ts
        seen_end_ts = set()  # Track to avoid infinite loops
        max_iterations = 100  # Safety limit

        for iteration in range(max_iterations):
            # Avoid infinite loop
            if current_end_ts in seen_end_ts:
                logger.debug("Breaking: already fetched this end_ts")
                break
            seen_end_ts.add(current_end_ts)

            response = self.get_event_candlesticks(
                series_ticker=series_ticker,
                event_ticker=event_ticker,
                start_ts=start_ts,
                end_ts=current_end_ts,
                period_interval=period_interval,
            )

            # Check if we got any data
            if not response.has_data():
                logger.debug("Breaking: no candle data in response")
                break

            # Merge candles for each market
            new_candles_count = 0
            for ticker in response.market_tickers:
                candles = response.get_candles_for_market(ticker)
                if ticker not in all_candles:
                    all_candles[ticker] = []
                # Only add candles we haven't seen
                existing_ts = {c.end_period_ts for c in all_candles[ticker]}
                for candle in candles:
                    if candle.end_period_ts not in existing_ts:
                        all_candles[ticker].append(candle)
                        new_candles_count += 1

            # If no new candles were added, stop
            if new_candles_count == 0:
                logger.debug("Breaking: no new candles added")
                break

            # Use adjusted_end_ts for next page if available and valid
            if (
                response.adjusted_end_ts
                and response.adjusted_end_ts > start_ts
                and response.adjusted_end_ts < current_end_ts
            ):
                current_end_ts = response.adjusted_end_ts
                logger.debug(
                    f"Pagination: fetched {new_candles_count} new candles, "
                    f"next end_ts={current_end_ts}"
                )
            else:
                break

        # Sort candles by timestamp for each market
        for ticker in all_candles:
            all_candles[ticker].sort(key=lambda c: c.end_period_ts or 0)

        total_candles = sum(len(c) for c in all_candles.values())
        logger.info(
            f"Total candles fetched for event {event_ticker}: "
            f"{total_candles} across {len(all_candles)} markets"
        )
        return all_candles

    def get_trades(
        self,
        ticker: str,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        limit: int = 1000,
        cursor: Optional[str] = None,
    ) -> TradesResponse:
        """
        Get trades for a market (fallback for minute bars).

        Args:
            ticker: Market ticker
            min_ts: Minimum timestamp
            max_ts: Maximum timestamp
            limit: Page size (max 1000)
            cursor: Pagination cursor

        Returns:
            TradesResponse with trades list and cursor
        """
        params: Dict[str, Any] = {
            "ticker": ticker,
            "limit": limit,
        }

        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor

        logger.debug(f"Fetching trades for {ticker}")
        data = self._request("GET", "/markets/trades", params=params)
        return TradesResponse(**data)

    def get_all_trades(
        self,
        ticker: str,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> List[Trade]:
        """
        Get all trades with automatic pagination.

        Args:
            ticker: Market ticker
            min_ts: Minimum timestamp
            max_ts: Maximum timestamp

        Returns:
            List of all Trade objects
        """
        all_trades: List[Trade] = []
        cursor: Optional[str] = None

        while True:
            response = self.get_trades(
                ticker=ticker,
                min_ts=min_ts,
                max_ts=max_ts,
                cursor=cursor,
            )

            all_trades.extend(response.trades)
            cursor = response.cursor

            if not cursor:
                break

        logger.info(f"Total trades fetched for {ticker}: {len(all_trades)}")
        return all_trades

    def get_orderbook(self, ticker: str, depth: int = 10) -> Dict[str, Any]:
        """
        Get orderbook for a market.

        Args:
            ticker: Market ticker
            depth: Number of price levels (default 10)

        Returns:
            Orderbook with yes/no bids and asks
        """
        params = {"depth": depth}
        logger.debug(f"Fetching orderbook for {ticker}")
        data = self._request("GET", f"/markets/{ticker}/orderbook", params=params)
        return data.get("orderbook", data)

    def get_balance(self) -> Dict[str, Any]:
        """
        Get account balance.

        Returns:
            Balance info including available_balance (in cents)
        """
        logger.info("Fetching account balance")
        data = self._request("GET", "/portfolio/balance")
        return data

    def create_order(
        self,
        ticker: str,
        side: str,  # "yes" or "no"
        action: str,  # "buy" or "sell"
        count: int,
        order_type: str = "limit",
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        client_order_id: Optional[str] = None,
        expiration_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create an order on Kalshi.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            order_type: "limit" or "market"
            yes_price: Price in cents for YES side (1-99)
            no_price: Price in cents for NO side (1-99)
            client_order_id: Optional client-provided order ID
            expiration_ts: Optional expiration timestamp

        Returns:
            Order response with order_id
        """
        body: Dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": order_type,
        }

        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if client_order_id:
            body["client_order_id"] = client_order_id
        if expiration_ts:
            body["expiration_ts"] = expiration_ts

        logger.info(
            f"Creating order: {action.upper()} {count} {side.upper()} @ "
            f"{yes_price if side == 'yes' else no_price}c on {ticker}"
        )
        data = self._request("POST", "/portfolio/orders", json_data=body)
        return data
