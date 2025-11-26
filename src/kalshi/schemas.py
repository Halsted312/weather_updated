"""
Pydantic schemas for Kalshi API responses.

Models match the Kalshi API v2 response structures.
Handles both ISO 8601 timestamps and Unix timestamps.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


def parse_timestamp(v: Union[str, int, None]) -> Optional[int]:
    """Convert ISO 8601 string or Unix timestamp to Unix timestamp."""
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        # Parse ISO 8601 format: "2025-11-24T15:00:00Z"
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except ValueError:
            return None
    return None


class Series(BaseModel):
    """Series metadata from Kalshi."""

    ticker: str
    title: str
    category: str
    frequency: str
    settlement_timer_seconds: Optional[int] = None
    settlement_sources: Optional[List[Dict[str, Any]]] = None
    tags: Optional[List[str]] = None
    description: Optional[str] = None

    model_config = {"extra": "allow"}


class Market(BaseModel):
    """Market contract from Kalshi."""

    ticker: str
    event_ticker: str
    series_ticker: Optional[str] = None  # May not be in response, can derive from ticker
    title: str
    subtitle: Optional[str] = None

    # Timestamps (stored as Unix seconds, accepts ISO 8601 strings)
    open_time: Optional[int] = None
    close_time: Optional[int] = None
    expiration_time: Optional[int] = None

    # Status and settlement
    status: str  # e.g., "open", "closed", "settled"
    result: Optional[str] = None  # "yes" or "no"
    settlement_value: Optional[float] = None

    # Market details (prices in cents)
    yes_bid: Optional[int] = None
    yes_ask: Optional[int] = None
    no_bid: Optional[int] = None
    no_ask: Optional[int] = None
    last_price: Optional[int] = None

    # Volume and liquidity
    volume: Optional[int] = None
    volume_24h: Optional[int] = None
    open_interest: Optional[int] = None
    liquidity: Optional[int] = None

    # Strike/bracket info (for ranged markets)
    floor_strike: Optional[float] = None
    cap_strike: Optional[float] = None
    strike_type: Optional[str] = None

    # Rules and description
    rules_primary: Optional[str] = None
    rules_secondary: Optional[str] = None
    can_close_early: Optional[bool] = None

    model_config = {"extra": "allow"}

    @field_validator("open_time", "close_time", "expiration_time", mode="before")
    @classmethod
    def parse_timestamps(cls, v: Union[str, int, None]) -> Optional[int]:
        """Convert ISO 8601 strings to Unix timestamps."""
        return parse_timestamp(v)

    def get_series_ticker(self) -> str:
        """Get series ticker from market ticker if not in response."""
        if self.series_ticker:
            return self.series_ticker
        # Derive from ticker: KXHIGHCHI-25NOV26-B50 -> KXHIGHCHI
        parts = self.ticker.split("-")
        if parts:
            return parts[0]
        return ""


class Candle(BaseModel):
    """Candlestick (OHLC) data for a market."""

    # Timestamp for this candle (accepts ISO 8601 or Unix)
    end_period_ts: Optional[int] = None
    period_minutes: int = 1  # 1, 60, or 1440

    # Yes bid prices (in cents)
    yes_bid_open: Optional[int] = None
    yes_bid_high: Optional[int] = None
    yes_bid_low: Optional[int] = None
    yes_bid_close: Optional[int] = None

    # Yes ask prices (in cents)
    yes_ask_open: Optional[int] = None
    yes_ask_high: Optional[int] = None
    yes_ask_low: Optional[int] = None
    yes_ask_close: Optional[int] = None

    # Last trade price (in cents)
    price_open: Optional[int] = None
    price_high: Optional[int] = None
    price_low: Optional[int] = None
    price_close: Optional[int] = None

    # Volume and open interest
    volume: Optional[int] = None
    open_interest: Optional[int] = None

    model_config = {"extra": "allow"}

    @field_validator("end_period_ts", mode="before")
    @classmethod
    def parse_timestamp_field(cls, v: Union[str, int, None]) -> Optional[int]:
        """Convert ISO 8601 strings to Unix timestamps."""
        return parse_timestamp(v)


def parse_price_to_cents(v: Union[float, int, None]) -> Optional[int]:
    """Convert price (cents or dollars) to cents as integer."""
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        # API now returns dollars as float, convert to cents
        # 0.01 -> 1 cent, 0.55 -> 55 cents
        return int(round(v * 100))
    return None


class Trade(BaseModel):
    """Individual trade from Kalshi."""

    trade_id: str
    ticker: str

    # Trade details (stored in cents)
    price: Optional[int] = None  # in cents
    count: int  # number of contracts
    taker_side: str  # "yes" or "no"

    # Timing (accepts ISO 8601 or Unix)
    created_time: Optional[int] = None

    # Optional fields
    yes_price: Optional[int] = None
    no_price: Optional[int] = None

    model_config = {"extra": "allow"}

    @field_validator("price", "yes_price", "no_price", mode="before")
    @classmethod
    def parse_price(cls, v: Union[float, int, None]) -> Optional[int]:
        """Convert price to cents."""
        return parse_price_to_cents(v)

    @field_validator("created_time", mode="before")
    @classmethod
    def parse_created_time(cls, v: Union[str, int, None]) -> Optional[int]:
        """Convert ISO 8601 strings to Unix timestamps."""
        return parse_timestamp(v)


class CandlestickResponse(BaseModel):
    """Response from per-market candlesticks endpoint."""

    candles: List[Candle] = Field(default_factory=list)
    adjusted_start_ts: Optional[int] = None
    adjusted_end_ts: Optional[int] = None

    @field_validator("adjusted_start_ts", "adjusted_end_ts", mode="before")
    @classmethod
    def parse_adjusted_timestamps(cls, v: Union[str, int, None]) -> Optional[int]:
        """Convert ISO 8601 strings to Unix timestamps."""
        return parse_timestamp(v)


class EventCandlestickResponse(BaseModel):
    """Response from event candlesticks endpoint (multiple markets).

    Event candlesticks return all markets for an event in a single call.
    The API returns market_candlesticks as a LIST of lists (parallel to market_tickers),
    NOT a dict. Each index in market_candlesticks corresponds to the same index in market_tickers.

    Example response:
    {
        "market_tickers": ["TICKER1", "TICKER2", ...],
        "market_candlesticks": [
            [candles for TICKER1...],
            [candles for TICKER2...],
        ],
        "adjusted_start_ts": ...,
        "adjusted_end_ts": ...
    }
    """

    market_tickers: List[str] = Field(default_factory=list)
    # API returns list of lists, parallel to market_tickers
    market_candlesticks: List[List[Dict[str, Any]]] = Field(default_factory=list)
    adjusted_start_ts: Optional[int] = None
    adjusted_end_ts: Optional[int] = None

    @field_validator("adjusted_start_ts", "adjusted_end_ts", mode="before")
    @classmethod
    def parse_adjusted_timestamps(cls, v: Union[str, int, None]) -> Optional[int]:
        """Convert ISO 8601 strings to Unix timestamps."""
        return parse_timestamp(v)

    def get_candles_for_market(self, ticker: str) -> List[Candle]:
        """Get parsed Candle objects for a specific market."""
        try:
            idx = self.market_tickers.index(ticker)
            if idx < len(self.market_candlesticks):
                raw_candles = self.market_candlesticks[idx]
                return [Candle(**c) for c in raw_candles]
        except ValueError:
            pass
        return []

    def get_all_candles(self) -> Dict[str, List[Candle]]:
        """Get all candles parsed as Candle objects by ticker."""
        result = {}
        for i, ticker in enumerate(self.market_tickers):
            if i < len(self.market_candlesticks):
                candles = self.market_candlesticks[i]
                result[ticker] = [Candle(**c) for c in candles]
            else:
                result[ticker] = []
        return result

    def has_data(self) -> bool:
        """Check if any market has candlestick data."""
        return any(len(candles) > 0 for candles in self.market_candlesticks)


class MarketsResponse(BaseModel):
    """Response from markets endpoint with pagination."""

    markets: List[Market] = Field(default_factory=list)
    cursor: Optional[str] = None


class TradesResponse(BaseModel):
    """Response from trades endpoint with pagination."""

    trades: List[Trade] = Field(default_factory=list)
    cursor: Optional[str] = None


class SeriesResponse(BaseModel):
    """Response from series endpoint."""

    series: Series
