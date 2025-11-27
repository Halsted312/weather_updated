#!/usr/bin/env python3
"""
Kalshi WebSocket recorder daemon.

Connects to Kalshi's WebSocket API and records all incoming messages
to the kalshi.ws_raw table for later analysis.

Subscribes to:
- Market data (orderbook updates, ticker updates)
- Trades (public trade feed)
- Fills (authenticated - your own fills)

For our 6 weather series:
- KXHIGHCHI (Chicago)
- KXHIGHAUS (Austin)
- KXHIGHDEN (Denver)
- KXHIGHLAX (Los Angeles)
- KXHIGHMIA (Miami)
- KXHIGHPHIL (Philadelphia)

Usage:
    python scripts/kalshi_ws_recorder.py
    python scripts/kalshi_ws_recorder.py --no-fills  # Skip authenticated fills
    python scripts/kalshi_ws_recorder.py --series KXHIGHCHI KXHIGHAUS

Deployment:
    systemctl start kalshi-ws-recorder
    docker-compose up -d kalshi-ws-recorder
"""

import argparse
import asyncio
import base64
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, InvalidStatusCode

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import CITIES, get_settings
from src.db import get_db_session, KalshiWsRaw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Our weather series tickers
WEATHER_SERIES = [city.series_ticker for city in CITIES.values()]

# WebSocket path for signing
WS_PATH = "/trade-api/ws/v2"

# Shutdown flag for graceful termination
shutdown_requested = False


def get_ws_url(base_url: str) -> str:
    """
    Derive WebSocket URL from REST base URL.

    https://api.elections.kalshi.com/trade-api/v2
    -> wss://api.elections.kalshi.com/trade-api/ws/v2
    """
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
    # Replace /v2 with /ws/v2
    ws_url = ws_url.replace("/trade-api/v2", "/trade-api/ws/v2")
    return ws_url


def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, requesting graceful shutdown...")
    shutdown_requested = True


def sign_pss_text(private_key_pem: str, message: str) -> str:
    """
    Sign a message using RSA-PSS with SHA256.

    This is the signature method required by Kalshi API.

    Args:
        private_key_pem: PEM-encoded private key string
        message: Message to sign

    Returns:
        Base64-encoded signature
    """
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode(),
        password=None,
    )
    signature = private_key.sign(
        message.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode()


def create_ws_auth_headers(api_key: str, private_key_pem: str) -> Dict[str, str]:
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
    signature = sign_pss_text(private_key_pem, msg_string)

    return {
        "KALSHI-ACCESS-KEY": api_key,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }


def build_subscription_message(
    channels: List[str],
    series_tickers: List[str],
    include_fills: bool = True,
) -> Dict[str, Any]:
    """
    Build WebSocket subscription message.

    Kalshi WS subscription format:
    {
        "id": 1,
        "cmd": "subscribe",
        "params": {
            "channels": ["orderbook_delta", "ticker", "trade", "fill"],
            "market_tickers": [...],  # or series_tickers
        }
    }
    """
    channels_list = channels.copy()
    if include_fills:
        channels_list.append("fill")

    return {
        "id": 1,
        "cmd": "subscribe",
        "params": {
            "channels": channels_list,
            "series_tickers": series_tickers,
        },
    }


def extract_message_metadata(message: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract metadata from a WebSocket message for categorization.

    Returns:
        Dict with 'stream' (channel type) and 'topic' (ticker or identifier)
    """
    msg_type = message.get("type", "")
    channel = message.get("channel", "")

    # Determine stream (channel type)
    stream = channel or msg_type or "unknown"

    # Determine topic (usually ticker or market)
    topic = None

    # Check various places where ticker might be
    if "ticker" in message:
        topic = message["ticker"]
    elif "market_ticker" in message:
        topic = message["market_ticker"]
    elif "msg" in message and isinstance(message["msg"], dict):
        msg = message["msg"]
        topic = msg.get("ticker") or msg.get("market_ticker")

    # For subscription confirmations
    if msg_type == "subscribed":
        topic = ",".join(message.get("msg", {}).get("channels", []))

    return {
        "stream": stream,
        "topic": topic,
    }


async def store_message(
    message: Dict[str, Any],
    batch: List[Dict[str, Any]],
    batch_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Add message to batch and flush if batch is full.

    Returns updated batch (empty if flushed, with new message otherwise)
    """
    metadata = extract_message_metadata(message)

    record = {
        "ts_utc": datetime.now(timezone.utc),
        "source": "kalshi",
        "stream": metadata["stream"],
        "topic": metadata["topic"],
        "payload": message,
    }
    batch.append(record)

    if len(batch) >= batch_size:
        await flush_batch(batch)
        return []

    return batch


async def flush_batch(batch: List[Dict[str, Any]]) -> int:
    """
    Flush batch of messages to database.

    Returns number of rows inserted.
    """
    if not batch:
        return 0

    try:
        with get_db_session() as session:
            for record in batch:
                ws_raw = KalshiWsRaw(
                    ts_utc=record["ts_utc"],
                    source=record["source"],
                    stream=record["stream"],
                    topic=record["topic"],
                    payload=record["payload"],
                )
                session.add(ws_raw)
            session.commit()
            logger.debug(f"Flushed {len(batch)} messages to kalshi.ws_raw")
            return len(batch)
    except Exception as e:
        logger.error(f"Error flushing batch to database: {e}")
        return 0


async def websocket_handler(
    ws_url: str,
    api_key: str,
    private_key: str,
    series_tickers: List[str],
    include_fills: bool = True,
    batch_size: int = 100,
    reconnect_delay: int = 5,
):
    """
    Main WebSocket handler with automatic reconnection.

    Args:
        ws_url: WebSocket URL to connect to
        api_key: Kalshi API key
        private_key: PEM private key string
        series_tickers: List of series to subscribe to
        include_fills: Whether to subscribe to fills channel (requires auth)
        batch_size: Number of messages to batch before DB write
        reconnect_delay: Seconds to wait before reconnecting
    """
    global shutdown_requested

    total_messages = 0
    connection_count = 0
    batch: List[Dict[str, Any]] = []

    while not shutdown_requested:
        connection_count += 1
        logger.info(f"Connection attempt #{connection_count} to {ws_url}")

        try:
            # Generate fresh auth headers for connection
            headers = create_ws_auth_headers(api_key, private_key)

            async with websockets.connect(
                ws_url,
                extra_headers=headers,
                ping_interval=30,
                ping_timeout=10,
            ) as websocket:
                logger.info(f"Connected to Kalshi WebSocket: {ws_url}")

                # Subscribe to channels
                channels = ["orderbook_delta", "ticker", "trade"]
                sub_msg = build_subscription_message(
                    channels=channels,
                    series_tickers=series_tickers,
                    include_fills=include_fills,
                )

                await websocket.send(json.dumps(sub_msg))
                logger.info(
                    f"Subscribed to channels={channels}, series={series_tickers}, "
                    f"fills={include_fills}"
                )

                # Message processing loop
                async for raw_message in websocket:
                    if shutdown_requested:
                        break

                    try:
                        message = json.loads(raw_message)
                        total_messages += 1

                        # Store message
                        batch = await store_message(message, batch, batch_size)

                        # Periodic logging
                        if total_messages % 1000 == 0:
                            logger.info(
                                f"Total messages received: {total_messages:,}"
                            )

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse message: {e}")
                        # Store raw string as payload
                        batch.append({
                            "ts_utc": datetime.now(timezone.utc),
                            "source": "kalshi",
                            "stream": "raw_error",
                            "topic": None,
                            "payload": {"raw": raw_message, "error": str(e)},
                        })

        except InvalidStatusCode as e:
            logger.error(f"WebSocket handshake failed: HTTP {e.status_code}")
            if e.status_code in (401, 403):
                logger.error("Authentication error - check API key and private key")
        except ConnectionClosedError as e:
            logger.warning(f"WebSocket connection closed: code={e.code}, reason={e.reason}")
        except ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: code={e.code}, reason={e.reason}")
        except Exception as e:
            logger.error(f"WebSocket error: {type(e).__name__}: {e}")

        # Flush remaining batch before reconnect
        if batch:
            await flush_batch(batch)
            batch = []

        if shutdown_requested:
            break

        # Wait before reconnecting
        logger.info(f"Reconnecting in {reconnect_delay} seconds...")
        await asyncio.sleep(reconnect_delay)

    # Final flush
    if batch:
        await flush_batch(batch)

    logger.info(f"WebSocket handler stopped. Total messages: {total_messages:,}")


def run_recorder(
    series_tickers: Optional[List[str]] = None,
    include_fills: bool = True,
    batch_size: int = 100,
    reconnect_delay: int = 5,
):
    """
    Main entry point for the WebSocket recorder.

    Args:
        series_tickers: List of series to subscribe to (default: all weather series)
        include_fills: Whether to subscribe to fills channel
        batch_size: Number of messages to batch before DB write
        reconnect_delay: Seconds to wait before reconnecting
    """
    global shutdown_requested

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Get settings
    settings = get_settings()

    # Derive WebSocket URL from REST base URL
    ws_url = get_ws_url(settings.kalshi_base_url)

    # Load private key
    private_key_path = Path(settings.kalshi_private_key_path)
    if not private_key_path.exists():
        logger.error(f"Private key not found: {private_key_path}")
        sys.exit(1)

    with open(private_key_path, "r") as f:
        private_key = f.read()

    # Default to all weather series
    if series_tickers is None:
        series_tickers = WEATHER_SERIES

    logger.info("=" * 60)
    logger.info("Kalshi WebSocket Recorder Starting")
    logger.info(f"  WebSocket URL: {ws_url}")
    logger.info(f"  Series: {series_tickers}")
    logger.info(f"  Include fills: {include_fills}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Reconnect delay: {reconnect_delay}s")
    logger.info("=" * 60)

    # Run the async handler
    try:
        asyncio.run(
            websocket_handler(
                ws_url=ws_url,
                api_key=settings.kalshi_api_key,
                private_key=private_key,
                series_tickers=series_tickers,
                include_fills=include_fills,
                batch_size=batch_size,
                reconnect_delay=reconnect_delay,
            )
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

    logger.info("=" * 60)
    logger.info("Kalshi WebSocket Recorder Stopped")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Kalshi WebSocket recorder daemon"
    )
    parser.add_argument(
        "--series", nargs="+",
        help=f"Series tickers to subscribe to (default: {WEATHER_SERIES})"
    )
    parser.add_argument(
        "--no-fills", action="store_true",
        help="Skip subscribing to fills channel (no auth required for public data)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Number of messages to batch before DB write (default: 100)"
    )
    parser.add_argument(
        "--reconnect-delay", type=int, default=5,
        help="Seconds to wait before reconnecting (default: 5)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test connection only - connect, receive a few messages, then exit"
    )

    args = parser.parse_args()

    if args.test:
        # Test mode - just verify connection works
        logger.info("Running in test mode...")
        settings = get_settings()
        ws_url = get_ws_url(settings.kalshi_base_url)
        logger.info(f"WebSocket URL: {ws_url}")

        private_key_path = Path(settings.kalshi_private_key_path)
        if not private_key_path.exists():
            logger.error(f"Private key not found: {private_key_path}")
            sys.exit(1)

        with open(private_key_path, "r") as f:
            private_key = f.read()

        async def test_connection():
            headers = create_ws_auth_headers(settings.kalshi_api_key, private_key)
            logger.info(f"Auth headers: KALSHI-ACCESS-KEY={settings.kalshi_api_key[:8]}..., TIMESTAMP={headers['KALSHI-ACCESS-TIMESTAMP']}")

            try:
                logger.info(f"Connecting to {ws_url}...")
                async with websockets.connect(
                    ws_url,
                    extra_headers=headers,
                    ping_interval=30,
                ) as ws:
                    logger.info("Handshake successful! Connection established.")

                    # Subscribe to one series
                    sub_msg = {
                        "id": 1,
                        "cmd": "subscribe",
                        "params": {
                            "channels": ["ticker"],
                            "series_tickers": ["KXHIGHCHI"],
                        },
                    }
                    await ws.send(json.dumps(sub_msg))
                    logger.info("Subscription sent, waiting for messages...")

                    # Receive a few messages
                    for i in range(5):
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=10)
                            data = json.loads(msg)
                            logger.info(f"Received message {i+1}: {json.dumps(data, indent=2)[:300]}...")
                        except asyncio.TimeoutError:
                            logger.info("No more messages (timeout after 10s)")
                            break

                    logger.info("Test complete!")

            except InvalidStatusCode as e:
                logger.error(f"Handshake failed: HTTP {e.status_code}")
                if e.status_code in (401, 403):
                    logger.error("Authentication error - check API key and private key")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Test failed: {type(e).__name__}: {e}")
                sys.exit(1)

        asyncio.run(test_connection())
        return

    # Normal mode
    run_recorder(
        series_tickers=args.series,
        include_fills=not args.no_fills,
        batch_size=args.batch_size,
        reconnect_delay=args.reconnect_delay,
    )


if __name__ == "__main__":
    main()
