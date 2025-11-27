#!/usr/bin/env python3
"""
WebSocket listener for Kalshi market open events.

Connects to Kalshi's WebSocket API and monitors for market_lifecycle events.
When a market transitions to "open" state, logs it to sim.market_open_log.

This is a read-only listener - it does NOT place orders.
The intent is to verify we can reliably detect market open events.

Usage:
    python -m open_maker.market_open_listener
    python -m open_maker.market_open_listener --verbose

Deployment:
    Can be run as a background daemon alongside kalshi_ws_recorder
"""

import argparse
import asyncio
import base64
import json
import logging
import re
import signal
import sys
import time
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, InvalidStatusCode

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import CITIES, get_settings
from src.db import get_db_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Our weather series tickers
WEATHER_SERIES = [city.series_ticker for city in CITIES.values()]

# Map series ticker back to city
SERIES_TO_CITY = {city.series_ticker: city.city_id for city in CITIES.values()}

# WebSocket path for signing
WS_PATH = "/trade-api/ws/v2"

# Shutdown flag for graceful termination
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, requesting graceful shutdown...")
    shutdown_requested = True


def get_ws_url(base_url: str) -> str:
    """
    Derive WebSocket URL from REST base URL.

    https://api.elections.kalshi.com/trade-api/v2
    -> wss://api.elections.kalshi.com/trade-api/ws/v2
    """
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = ws_url.replace("/trade-api/v2", "/trade-api/ws/v2")
    return ws_url


def sign_pss_text(private_key: Any, message: str) -> str:
    """
    Sign a message using RSA-PSS with SHA256.

    Args:
        private_key: Loaded private key object
        message: Message to sign

    Returns:
        Base64-encoded signature
    """
    signature = private_key.sign(
        message.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode()


def create_ws_auth_headers(api_key: str, private_key: Any) -> Dict[str, str]:
    """
    Create authentication headers for WebSocket connection.

    Per Kalshi docs, WebSocket auth uses:
    - KALSHI-ACCESS-KEY: API key ID
    - KALSHI-ACCESS-SIGNATURE: RSA-PSS signed message
    - KALSHI-ACCESS-TIMESTAMP: Unix timestamp in milliseconds

    The message to sign is: timestamp + "GET" + "/trade-api/ws/v2"
    """
    timestamp = str(int(time.time() * 1000))
    msg_string = timestamp + "GET" + WS_PATH
    signature = sign_pss_text(private_key, msg_string)

    return {
        "KALSHI-ACCESS-KEY": api_key,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }


def parse_event_ticker(event_ticker: str) -> Optional[date]:
    """
    Parse event date from event ticker.

    Example: KXHIGHCHI-25NOV28 -> 2025-11-28
    """
    # Match pattern like -25NOV28 at the end
    match = re.search(r'-(\d{2})([A-Z]{3})(\d{2})$', event_ticker)
    if not match:
        return None

    year_suffix = int(match.group(1))
    month_str = match.group(2)
    day = int(match.group(3))

    # Map month abbreviations
    months = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }
    month = months.get(month_str)
    if not month:
        return None

    # Assume 2000s for 2-digit year
    year = 2000 + year_suffix

    try:
        return date(year, month, day)
    except ValueError:
        return None


class MarketOpenListener:
    """Listens for market open events via WebSocket."""

    def __init__(
        self,
        api_key: str,
        private_key_path: str,
        base_url: str,
        verbose: bool = False,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.ws_url = get_ws_url(base_url)
        self.verbose = verbose

        # Load private key
        private_key_file = Path(private_key_path)
        if not private_key_file.exists():
            raise FileNotFoundError(f"Private key not found: {private_key_path}")

        with open(private_key_file, "rb") as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
            )

        # Track seen events to avoid duplicate logging
        self.seen_events: set = set()

        logger.info(f"MarketOpenListener initialized")
        logger.info(f"  WebSocket URL: {self.ws_url}")
        logger.info(f"  Watching series: {WEATHER_SERIES}")

    def _build_subscription(self) -> Dict[str, Any]:
        """Build subscription message for market lifecycle events."""
        return {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["market_lifecycle"],
                "series_tickers": WEATHER_SERIES,
            },
        }

    async def _log_market_open(self, data: Dict[str, Any]) -> None:
        """Log a market open event to the database."""
        series_ticker = data.get("series_ticker", "")
        event_ticker = data.get("event_ticker", "")
        market_ticker = data.get("market_ticker", "")

        # Dedupe by market_ticker
        if market_ticker in self.seen_events:
            return
        self.seen_events.add(market_ticker)

        # Extract city from series ticker
        city = SERIES_TO_CITY.get(series_ticker)

        # Parse event date from event ticker
        event_date = parse_event_ticker(event_ticker)

        record = {
            "series_ticker": series_ticker,
            "event_ticker": event_ticker,
            "market_ticker": market_ticker,
            "city": city,
            "event_date": event_date,
            "opened_at": datetime.now(timezone.utc),
            "raw_data": data,
        }

        try:
            from sqlalchemy import text
            with get_db_session() as session:
                session.execute(
                    text("""
                        INSERT INTO sim.market_open_log
                        (series_ticker, event_ticker, market_ticker, city, event_date, opened_at, raw_data)
                        VALUES (:series_ticker, :event_ticker, :market_ticker, :city, :event_date, :opened_at, :raw_data)
                    """),
                    {
                        **record,
                        "raw_data": json.dumps(data),
                    }
                )
                session.commit()

            logger.info(
                f"[MARKET OPEN] {market_ticker} | "
                f"city={city} | event_date={event_date} | "
                f"opened_at={record['opened_at'].strftime('%H:%M:%S')} UTC"
            )

        except Exception as e:
            logger.error(f"Failed to log market open: {e}")

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Process incoming WebSocket message."""
        msg_type = message.get("type", "")
        channel = message.get("channel", "")

        if self.verbose:
            logger.debug(f"Received: type={msg_type}, channel={channel}")

        # Handle subscription confirmation
        if msg_type == "subscribed":
            channels = message.get("msg", {}).get("channels", [])
            logger.info(f"Subscribed to channels: {channels}")
            return

        # Handle market lifecycle events
        if channel == "market_lifecycle":
            msg_data = message.get("msg", {})
            state = msg_data.get("state", "")

            if state == "open":
                await self._log_market_open(msg_data)
            elif self.verbose:
                logger.debug(
                    f"Lifecycle event: state={state}, "
                    f"market={msg_data.get('market_ticker', 'unknown')}"
                )

    async def connect_and_listen(self) -> None:
        """Main WebSocket connection loop with reconnection logic."""
        global shutdown_requested
        reconnect_delay = 1  # Start with 1 second

        while not shutdown_requested:
            try:
                headers = create_ws_auth_headers(self.api_key, self.private_key)

                logger.info(f"Connecting to {self.ws_url}...")

                async with websockets.connect(
                    self.ws_url,
                    extra_headers=headers,
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    logger.info("Connected! Subscribing to market_lifecycle...")

                    # Send subscription
                    sub_msg = self._build_subscription()
                    await ws.send(json.dumps(sub_msg))

                    # Reset reconnect delay on successful connection
                    reconnect_delay = 1

                    # Listen for messages
                    async for raw in ws:
                        if shutdown_requested:
                            break

                        try:
                            message = json.loads(raw)
                            await self._handle_message(message)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON: {raw[:100]}...")

            except ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")
            except ConnectionClosedError as e:
                logger.warning(f"Connection closed with error: {e}")
            except InvalidStatusCode as e:
                logger.error(f"Invalid status code: {e}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

            if not shutdown_requested:
                logger.info(f"Reconnecting in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)
                # Exponential backoff up to 60 seconds
                reconnect_delay = min(reconnect_delay * 2, 60)

        logger.info("Listener shutting down")


async def main_async(args):
    """Async main entry point."""
    settings = get_settings()

    listener = MarketOpenListener(
        api_key=settings.kalshi_api_key,
        private_key_path=settings.kalshi_private_key_path,
        base_url=settings.kalshi_base_url,
        verbose=args.verbose,
    )

    await listener.connect_and_listen()


def main():
    parser = argparse.ArgumentParser(
        description="WebSocket listener for Kalshi market open events"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
