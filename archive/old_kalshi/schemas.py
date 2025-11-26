"""
Pydantic schemas for Kalshi API responses.

Models match the Kalshi API v2 response structures.
"""

from typing import Optional, List, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field


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

    class Config:
        extra = "allow"  # Store additional fields as raw_json


class Market(BaseModel):
    """Market contract from Kalshi."""

    ticker: str
    event_ticker: str
    series_ticker: str
    title: str
    subtitle: Optional[str] = None

    # Timestamps (Unix seconds)
    open_time: int
    close_time: int
    expiration_time: int

    # Status and settlement
    status: str  # e.g., "open", "closed", "settled"
    result: Optional[str] = None  # "yes" or "no"
    settlement_value: Optional[float] = None

    # Market details
    yes_bid: Optional[int] = None  # in cents
    yes_ask: Optional[int] = None  # in cents
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

    class Config:
        extra = "allow"


class Candle(BaseModel):
    """Candlestick (OHLC) data for a market."""

    # Timestamp for this candle
    end_period_ts: int  # Unix seconds
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

    class Config:
        extra = "allow"


class Trade(BaseModel):
    """Individual trade from Kalshi."""

    trade_id: str
    ticker: str

    # Trade details
    price: int  # in cents
    count: int  # number of contracts
    taker_side: str  # "yes" or "no"

    # Timing
    created_time: int  # Unix seconds

    # Optional fields
    yes_price: Optional[int] = None
    no_price: Optional[int] = None

    class Config:
        extra = "allow"


class CandlestickResponse(BaseModel):
    """Response from candlesticks endpoint."""

    candles: List[Candle] = Field(default_factory=list)
    adjusted_start_ts: Optional[int] = None
    adjusted_end_ts: Optional[int] = None


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
