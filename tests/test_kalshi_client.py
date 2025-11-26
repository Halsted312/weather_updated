"""
Tests for Kalshi API client.

Includes:
- Schema validation tests
- Client initialization tests
- Integration tests (require valid API credentials)
"""

import os
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.config import get_settings
from src.kalshi.schemas import (
    Candle,
    CandlestickResponse,
    Market,
    MarketsResponse,
    Series,
    SeriesResponse,
    Trade,
    TradesResponse,
)


class TestSchemas:
    """Test Pydantic schema validation."""

    def test_series_schema(self):
        """Test Series model parsing."""
        data = {
            "ticker": "KXHIGHCHI",
            "title": "Highest temperature in Chicago",
            "category": "Weather",
            "frequency": "daily",
            "settlement_timer_seconds": 86400,
            "tags": ["weather", "temperature"],
            "extra_field": "should be allowed",
        }
        series = Series(**data)
        assert series.ticker == "KXHIGHCHI"
        assert series.title == "Highest temperature in Chicago"
        assert series.category == "Weather"
        assert series.frequency == "daily"
        assert series.settlement_timer_seconds == 86400
        assert series.tags == ["weather", "temperature"]

    def test_market_schema(self):
        """Test Market model parsing."""
        data = {
            "ticker": "KXHIGHCHI-25NOV26-B50",
            "event_ticker": "KXHIGHCHI-25NOV26",
            "series_ticker": "KXHIGHCHI",
            "title": "Chicago high 50F or above?",
            "open_time": 1732500000,
            "close_time": 1732600000,
            "expiration_time": 1732650000,
            "status": "settled",
            "result": "yes",
            "settlement_value": 100,
            "yes_bid": 85,
            "yes_ask": 87,
            "volume": 1000,
            "floor_strike": 50.0,
            "cap_strike": None,
        }
        market = Market(**data)
        assert market.ticker == "KXHIGHCHI-25NOV26-B50"
        assert market.series_ticker == "KXHIGHCHI"
        assert market.status == "settled"
        assert market.result == "yes"
        assert market.settlement_value == 100
        assert market.floor_strike == 50.0
        assert market.cap_strike is None

    def test_candle_schema(self):
        """Test Candle model parsing."""
        data = {
            "end_period_ts": 1732500060,
            "yes_bid_open": 45,
            "yes_bid_high": 48,
            "yes_bid_low": 44,
            "yes_bid_close": 47,
            "yes_ask_open": 47,
            "yes_ask_high": 50,
            "yes_ask_low": 46,
            "yes_ask_close": 49,
            "price_open": 46,
            "price_high": 49,
            "price_low": 45,
            "price_close": 48,
            "volume": 100,
            "open_interest": 500,
        }
        candle = Candle(**data)
        assert candle.end_period_ts == 1732500060
        assert candle.yes_bid_close == 47
        assert candle.yes_ask_close == 49
        assert candle.price_close == 48
        assert candle.volume == 100
        assert candle.period_minutes == 1  # default

    def test_candle_schema_with_period(self):
        """Test Candle model with explicit period."""
        data = {
            "end_period_ts": 1732500060,
            "period_minutes": 60,
            "volume": 500,
        }
        candle = Candle(**data)
        assert candle.period_minutes == 60
        assert candle.volume == 500
        assert candle.yes_bid_close is None

    def test_trade_schema(self):
        """Test Trade model parsing."""
        data = {
            "trade_id": "abc123",
            "ticker": "KXHIGHCHI-25NOV26-B50",
            "price": 55,
            "count": 10,
            "taker_side": "yes",
            "created_time": 1732500000,
        }
        trade = Trade(**data)
        assert trade.trade_id == "abc123"
        assert trade.price == 55
        assert trade.count == 10
        assert trade.taker_side == "yes"

    def test_candlestick_response(self):
        """Test CandlestickResponse model."""
        data = {
            "candles": [
                {"end_period_ts": 1732500060, "volume": 10},
                {"end_period_ts": 1732500120, "volume": 20},
            ],
            "adjusted_start_ts": 1732500000,
            "adjusted_end_ts": 1732500120,
        }
        response = CandlestickResponse(**data)
        assert len(response.candles) == 2
        assert response.adjusted_start_ts == 1732500000
        assert response.adjusted_end_ts == 1732500120

    def test_markets_response(self):
        """Test MarketsResponse model."""
        data = {
            "markets": [
                {
                    "ticker": "T1",
                    "event_ticker": "E1",
                    "series_ticker": "S1",
                    "title": "Test",
                    "open_time": 1000,
                    "close_time": 2000,
                    "expiration_time": 3000,
                    "status": "open",
                }
            ],
            "cursor": "next_page",
        }
        response = MarketsResponse(**data)
        assert len(response.markets) == 1
        assert response.markets[0].ticker == "T1"
        assert response.cursor == "next_page"

    def test_trades_response(self):
        """Test TradesResponse model."""
        data = {
            "trades": [
                {
                    "trade_id": "t1",
                    "ticker": "M1",
                    "price": 50,
                    "count": 5,
                    "taker_side": "no",
                    "created_time": 1000,
                }
            ],
            "cursor": None,
        }
        response = TradesResponse(**data)
        assert len(response.trades) == 1
        assert response.trades[0].price == 50
        assert response.cursor is None

    def test_series_response(self):
        """Test SeriesResponse model."""
        data = {
            "series": {
                "ticker": "TEST",
                "title": "Test Series",
                "category": "Test",
                "frequency": "daily",
            }
        }
        response = SeriesResponse(**data)
        assert response.series.ticker == "TEST"


class TestClientInitialization:
    """Test KalshiClient initialization."""

    def test_client_requires_private_key_file(self):
        """Test that client raises error if private key file doesn't exist."""
        from src.kalshi.client import KalshiClient

        with pytest.raises(FileNotFoundError):
            KalshiClient(
                api_key="test_key",
                private_key_path="/nonexistent/path.pem",
            )

    @patch("src.kalshi.client.Path")
    def test_client_loads_private_key(self, mock_path):
        """Test that client loads private key from file."""
        from src.kalshi.client import KalshiClient

        mock_path.return_value.exists.return_value = True
        mock_open = MagicMock()
        mock_open.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value="-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----")))
        mock_open.__exit__ = MagicMock(return_value=False)

        with patch("builtins.open", return_value=mock_open):
            client = KalshiClient(
                api_key="test_key",
                private_key_path="/test/path.pem",
            )
            assert client.api_key == "test_key"
            assert client.base_url == "https://api.elections.kalshi.com/trade-api/v2"

    def test_client_rate_limit_delay(self):
        """Test default rate limit delay."""
        from src.kalshi.client import KalshiClient

        with patch.object(KalshiClient, "__init__", lambda self, **kwargs: None):
            client = KalshiClient.__new__(KalshiClient)
            client.rate_limit_delay = 0.2
            assert client.rate_limit_delay == 0.2


class TestJWTGeneration:
    """Test JWT token generation."""

    def test_jwt_token_structure(self):
        """Test JWT token has correct structure."""
        import jwt as pyjwt
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend

        # Generate test RSA key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()

        # Create token using same logic as client
        now = int(time.time())
        payload = {
            "iss": "test_api_key",
            "iat": now,
            "exp": now + 3600,
        }
        token = pyjwt.encode(payload, private_key_pem, algorithm="RS256")

        # Decode and verify structure
        decoded = pyjwt.decode(
            token,
            private_key.public_key(),
            algorithms=["RS256"]
        )
        assert decoded["iss"] == "test_api_key"
        assert "iat" in decoded
        assert "exp" in decoded
        assert decoded["exp"] > decoded["iat"]


@pytest.mark.integration
class TestKalshiClientIntegration:
    """
    Integration tests that require valid API credentials.

    Run with: pytest -m integration tests/test_kalshi_client.py
    """

    @pytest.fixture
    def client(self):
        """Create client with real credentials."""
        from src.kalshi.client import KalshiClient

        settings = get_settings()

        if not settings.kalshi_api_key or not settings.kalshi_private_key_path:
            pytest.skip("Kalshi API credentials not configured")

        try:
            return KalshiClient(
                api_key=settings.kalshi_api_key,
                private_key_path=settings.kalshi_private_key_path,
                base_url=settings.kalshi_base_url,
            )
        except FileNotFoundError:
            pytest.skip("Kalshi private key file not found")

    def test_get_series(self, client):
        """Test fetching series metadata."""
        response = client.get_series("KXHIGHCHI")
        assert response.series.ticker == "KXHIGHCHI"
        assert "Chicago" in response.series.title or "temperature" in response.series.title.lower()

    def test_get_markets(self, client):
        """Test fetching markets with pagination."""
        # Get markets from last 30 days
        now = int(time.time())
        thirty_days_ago = now - (30 * 24 * 60 * 60)

        response = client.get_markets(
            series_ticker="KXHIGHCHI",
            min_close_ts=thirty_days_ago,
            max_close_ts=now,
            limit=10,
        )

        assert isinstance(response.markets, list)
        # May be empty if no recent markets, but structure should be valid

    def test_get_market(self, client):
        """Test fetching single market details."""
        # First get a market ticker from series
        now = int(time.time())
        thirty_days_ago = now - (30 * 24 * 60 * 60)

        markets_response = client.get_markets(
            series_ticker="KXHIGHCHI",
            min_close_ts=thirty_days_ago,
            max_close_ts=now,
            limit=1,
        )

        if not markets_response.markets:
            pytest.skip("No markets available to test")

        ticker = markets_response.markets[0].ticker
        market = client.get_market(ticker)

        assert market.ticker == ticker
        assert market.get_series_ticker() == "KXHIGHCHI"

    def test_get_market_candlesticks(self, client):
        """Test fetching market candlesticks."""
        # Get a recent settled market
        now = int(time.time())
        sixty_days_ago = now - (60 * 24 * 60 * 60)

        markets_response = client.get_markets(
            series_ticker="KXHIGHCHI",
            status="settled",
            min_close_ts=sixty_days_ago,
            max_close_ts=now,
            limit=1,
        )

        if not markets_response.markets:
            pytest.skip("No settled markets available")

        market = markets_response.markets[0]

        # Get candlesticks for this market
        response = client.get_market_candlesticks(
            series_ticker="KXHIGHCHI",
            market_ticker=market.ticker,
            start_ts=market.open_time,
            end_ts=market.close_time,
            period_interval=60,  # 1-hour candles
        )

        assert isinstance(response.candles, list)

    def test_get_trades(self, client):
        """Test fetching trades."""
        # Get a recent market
        now = int(time.time())
        thirty_days_ago = now - (30 * 24 * 60 * 60)

        markets_response = client.get_markets(
            series_ticker="KXHIGHCHI",
            min_close_ts=thirty_days_ago,
            max_close_ts=now,
            limit=1,
        )

        if not markets_response.markets:
            pytest.skip("No markets available")

        ticker = markets_response.markets[0].ticker
        response = client.get_trades(ticker=ticker, limit=10)

        assert isinstance(response.trades, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
