"""
Kalshi API Client

Handles authentication with API key + private key (PEM) and provides
methods to fetch series, markets, candlesticks, and trades.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import jwt
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.kalshi.schemas import (
    Candle,
    CandlestickResponse,
    Market,
    MarketsResponse,
    SeriesResponse,
    Trade,
    TradesResponse,
)

logger = logging.getLogger(__name__)


class KalshiClient:
    """Client for Kalshi API with authentication and rate limiting."""

    def __init__(
        self,
        api_key: str,
        private_key_path: str,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
        rate_limit_delay: float = 0.2,  # 5 requests per second
    ):
        """
        Initialize Kalshi API client.

        Args:
            api_key: Kalshi API key ID
            private_key_path: Path to PEM private key file
            base_url: Base URL for Kalshi API
            rate_limit_delay: Delay between requests (seconds)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0

        # Load private key
        private_key_file = Path(private_key_path)
        if not private_key_file.exists():
            raise FileNotFoundError(f"Private key not found: {private_key_path}")

        with open(private_key_file, "r") as f:
            self.private_key = f.read()

        # Setup session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        logger.info(f"Kalshi client initialized with base URL: {self.base_url}")

    def _generate_token(self, expiry_minutes: int = 60) -> str:
        """
        Generate JWT token for authentication.

        Args:
            expiry_minutes: Token expiry time in minutes

        Returns:
            JWT token string
        """
        now = int(time.time())
        payload = {
            "iss": self.api_key,
            "iat": now,
            "exp": now + (expiry_minutes * 60),
        }

        token = jwt.encode(payload, self.private_key, algorithm="RS256")
        return token

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make authenticated request to Kalshi API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            params: Query parameters
            json_data: JSON body for POST requests

        Returns:
            Response JSON as dict

        Raises:
            requests.HTTPError: On HTTP errors
        """
        self._rate_limit()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self._generate_token()}",
            "Content-Type": "application/json",
        }

        logger.debug(f"{method} {url} params={params}")

        response = self.session.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_data,
            timeout=30,
        )

        response.raise_for_status()
        return response.json()

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
    ) -> CandlestickResponse:
        """
        Get event candlesticks (aggregated across all markets in event).

        Args:
            series_ticker: Series ticker
            event_ticker: Event ticker
            start_ts: Start timestamp (Unix seconds)
            end_ts: End timestamp (Unix seconds)
            period_interval: Period in minutes (1, 60, or 1440)

        Returns:
            CandlestickResponse with event candles
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
        return CandlestickResponse(**data)

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
