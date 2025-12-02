"""
WebSocket handler for Kalshi real-time market data.

Provides:
- RSA-PSS authentication
- Auto-reconnection with exponential backoff
- Message routing to registered handlers
- Subscription management and resubscription on reconnect
"""

import asyncio
import base64
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Awaitable

import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)

# WebSocket path for signing (must match connection URL)
WS_PATH = "/trade-api/ws/v2"

MessageHandler = Callable[[dict], Awaitable[None]]


class KalshiAuth:
    """Handles RSA-PSS authentication for Kalshi WebSocket."""

    def __init__(self, api_key: str, private_key_path: str):
        """
        Initialize authentication.

        Args:
            api_key: Kalshi API key ID
            private_key_path: Path to PEM private key file
        """
        self.api_key = api_key

        with open(private_key_path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(),
                password=None
            )

    def _sign_message(self, method: str, path: str) -> tuple[str, str]:
        """
        Sign message for Kalshi authentication.

        Args:
            method: HTTP method ("GET" for WebSocket)
            path: Path without query string

        Returns:
            (timestamp_ms, base64_signature)
        """
        timestamp_ms = str(int(time.time() * 1000))
        message = timestamp_ms + method + path.split("?", 1)[0]

        signature = self.private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,  # Fixed: was MAX_LENGTH
            ),
            hashes.SHA256(),
        )

        signature_b64 = base64.b64encode(signature).decode("utf-8")
        return timestamp_ms, signature_b64

    def ws_headers(self) -> Dict[str, str]:
        """
        Build authentication headers for WebSocket handshake.

        Returns:
            Dict with KALSHI-ACCESS-* headers
        """
        timestamp, signature = self._sign_message("GET", WS_PATH)

        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }


class WebSocketHandler:
    """
    Kalshi WebSocket client with auto-reconnection.

    Features:
    - Persistent connection with exponential backoff reconnection
    - Message routing by channel type
    - Subscription management
    - Automatic resubscription on reconnect
    """

    def __init__(
        self,
        ws_url: str,
        auth: KalshiAuth,
        reconnect_min_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
    ):
        """
        Initialize WebSocket handler.

        Args:
            ws_url: Full WebSocket URL (e.g., wss://api.elections.kalshi.com/trade-api/ws/v2)
            auth: Authentication handler
            reconnect_min_delay: Minimum delay between reconnection attempts
            reconnect_max_delay: Maximum delay between reconnection attempts
        """
        self.ws_url = ws_url
        self.auth = auth
        self.reconnect_min_delay = reconnect_min_delay
        self.reconnect_max_delay = reconnect_max_delay

        self.ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._next_id = 1

        # Message handlers by channel/type
        self.handlers: Dict[str, List[MessageHandler]] = {}

        # Subscription tracking for resubscribe
        self.subscriptions: List[dict] = []

    def register_handler(self, channel: str, handler: MessageHandler) -> None:
        """
        Register async handler for a channel.

        Args:
            channel: Channel name (e.g., "ticker", "orderbook_delta", "fill")
            handler: Async function to handle messages from this channel
        """
        if channel not in self.handlers:
            self.handlers[channel] = []

        self.handlers[channel].append(handler)
        logger.debug(f"Registered handler for channel '{channel}'")

    async def start(self) -> None:
        """
        Start persistent WebSocket loop with reconnection.

        Runs until stop() is called.
        """
        self._running = True
        backoff = self.reconnect_min_delay

        while self._running:
            try:
                await self._connect_and_run()
                backoff = self.reconnect_min_delay  # Reset on clean exit

            except InvalidStatusCode as e:
                logger.error(f"WebSocket authentication failed: {e}")
                break  # Don't retry auth failures

            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}, reconnecting...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.reconnect_max_delay)

            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.reconnect_max_delay)

    async def stop(self) -> None:
        """
        Request graceful shutdown.

        Stops the reconnection loop and closes active connection.
        """
        self._running = False

        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")

    async def subscribe(
        self,
        channels: List[str],
        market_tickers: Optional[List[str]] = None,
        series_tickers: Optional[List[str]] = None,
    ) -> None:
        """
        Subscribe to channels.

        Args:
            channels: List of channel names (e.g., ["ticker", "fill"])
            market_tickers: Optional list of specific market tickers
            series_tickers: Optional list of series tickers (for filtering)
        """
        params = {"channels": channels}

        if market_tickers:
            params["market_tickers"] = market_tickers
        if series_tickers:
            params["series_tickers"] = series_tickers

        # Store for resubscribe
        self.subscriptions.append(params)

        # Send subscribe command
        await self._send_command("subscribe", params)

    async def unsubscribe(self, sids: List[int]) -> None:
        """
        Unsubscribe from subscriptions.

        Args:
            sids: List of subscription IDs to unsubscribe
        """
        await self._send_command("unsubscribe", {"sids": sids})

    async def _send_command(self, cmd: str, params: Optional[dict] = None) -> None:
        """
        Send command to Kalshi WebSocket.

        Args:
            cmd: Command name ("subscribe", "unsubscribe", etc.)
            params: Command parameters
        """
        if not self.ws:
            logger.warning(f"Cannot send command '{cmd}': WebSocket not connected")
            return

        msg_id = self._next_id
        self._next_id += 1

        message = {"id": msg_id, "cmd": cmd}
        if params:
            message["params"] = params

        try:
            await self.ws.send(json.dumps(message))
            logger.debug(f"Sent command: {cmd} (id={msg_id})")
        except Exception as e:
            logger.error(f"Failed to send command {cmd}: {e}")

    async def _connect_and_run(self) -> None:
        """
        Connect to WebSocket, subscribe to channels, and process messages.

        This method blocks until connection closes or error occurs.
        """
        headers = self.auth.ws_headers()
        logger.info(f"Connecting to {self.ws_url}")

        async with websockets.connect(
            self.ws_url,
            extra_headers=headers,  # Fixed: was additional_headers
            ping_interval=20,  # Send ping every 20 seconds
            ping_timeout=20,   # Wait max 20 seconds for pong
        ) as ws:
            self.ws = ws
            logger.info("WebSocket connected successfully")

            # Resubscribe to all saved subscriptions
            await self._resubscribe()

            # Message loop
            async for raw_message in ws:
                if not self._running:
                    break

                await self._handle_message(raw_message)

    async def _resubscribe(self) -> None:
        """
        Resubscribe to all saved subscriptions.

        Called after reconnection to restore subscriptions.
        """
        for sub_params in self.subscriptions:
            await self._send_command("subscribe", sub_params)

        if self.subscriptions:
            logger.info(f"Resubscribed to {len(self.subscriptions)} channels")

    async def _handle_message(self, raw: str) -> None:
        """
        Parse and route incoming message to appropriate handlers.

        Args:
            raw: Raw JSON message string
        """
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")
            return

        msg_type = message.get("type")
        channel = message.get("channel") or msg_type

        # Handle control messages
        if msg_type == "subscribed":
            sid = message.get("msg", {}).get("sid")
            ch = message.get("msg", {}).get("channel")
            logger.info(f"Subscribed: sid={sid} channel={ch}")
            return

        elif msg_type == "unsubscribed":
            sid = message.get("sid")
            logger.info(f"Unsubscribed: sid={sid}")
            return

        elif msg_type == "ok":
            logger.debug(f"OK response: id={message.get('id')}")
            return

        elif msg_type == "error":
            error_msg = message.get("msg", {})
            code = error_msg.get("code")
            msg_text = error_msg.get("msg")
            logger.error(f"WebSocket error {code}: {msg_text}")
            return

        # Route to registered handlers
        if channel in self.handlers:
            for handler in self.handlers[channel]:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(
                        f"Handler error for channel '{channel}': {e}",
                        exc_info=True
                    )
        else:
            logger.debug(f"No handler for channel '{channel}' (type={msg_type})")
