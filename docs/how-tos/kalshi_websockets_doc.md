# Kalshi WebSockets – Full Client Design & Code (Python)

> This document is a self-contained reference for building a robust Kalshi WebSocket client (v2) in Python. It’s meant for offline coding agents and assumes no internet access.

---

## 0. Goals & Mental Model

Kalshi’s WebSocket API gives you a **single authenticated connection** that can be subscribed to multiple channels:

* Orderbook updates (snapshots + incremental deltas)([Kalshi API Documentation][1])
* Ticker updates (price, volume, open interest)([Kalshi API Documentation][2])
* Public trades (tape)([Kalshi API Documentation][3])
* Your fills (private)([Kalshi API Documentation][4])
* Your market positions (private)([Kalshi API Documentation][5])
* Market & event lifecycle updates (creation, open/close, settlement)([Kalshi API Documentation][6])
* Multivariate lookups (combo collections)([Kalshi API Documentation][7])
* Communications (RFQs & quotes)([Kalshi API Documentation][8])

**Core idea for the design:**

* **Layer 1 – Transport**: WebSocket connection + auth + reconnection
* **Layer 2 – Protocol**: subscribe/unsubscribe, `sid` → channel mapping, error handling([Kalshi API Documentation][9])
* **Layer 3 – Data models**: typed messages (ticker, orderbook, fills, etc.)
* **Layer 4 – Domain logic**: orderbook builder, P&L tracking, strategy hooks

Your trading / RL system should live on **Layer 4** and never have to re-learn WS quirks.

---

## 1. Config & Environment

Kalshi has **two environments** for WS:([Kalshi API Documentation][10])

* **Prod**: `wss://api.elections.kalshi.com/trade-api/ws/v2`
* **Demo**: `wss://demo-api.kalshi.co/trade-api/ws/v2`

We’ll centralize this in a config object.

```python
# kalshi_config.py

from dataclasses import dataclass

@dataclass
class KalshiConfig:
    key_id: str                 # KALSHI-ACCESS-KEY
    private_key_path: str       # path to PEM file
    is_demo: bool = True        # demo vs prod

    @property
    def ws_url(self) -> str:
        host = (
            "wss://demo-api.kalshi.co"
            if self.is_demo
            else "wss://api.elections.kalshi.com"
        )
        path = "/trade-api/ws/v2"
        return host + path

    @property
    def ws_path_for_signing(self) -> str:
        # MUST match the path used in the WS URL (no host, no query string)
        return "/trade-api/ws/v2"
```

### Why this design?

* Keeps **environment switching** a one-line change (`is_demo=True/False`).
* Guarantees the **signed path** is consistent across the codebase (Kalshi WS signing uses `TIMESTAMP + "GET" + PATH` with no query string).([Kalshi API Documentation][10])

---

## 2. Authentication & Signing

WebSocket connections use the **same RSA-PSS signing** scheme as Kalshi’s REST trading endpoints:([Kalshi API Documentation][10])

* Generate an API key pair in the UI (Key ID + private key PEM)
* For each handshake:

  * Compute `timestamp_ms`
  * Build message: `timestamp_ms + "GET" + "/trade-api/ws/v2"`
  * Sign with RSA-PSS SHA-256
  * Base64-encode signature
  * Provide three headers:

    * `KALSHI-ACCESS-KEY`
    * `KALSHI-ACCESS-TIMESTAMP`
    * `KALSHI-ACCESS-SIGNATURE`

### 2.1 Auth class

```python
# kalshi_auth.py

import base64
import time
from typing import Dict

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from kalshi_config import KalshiConfig


class KalshiAuth:
    """
    Handles RSA signing for Kalshi API (REST + WebSocket).

    For WebSockets we sign:
        TIMESTAMP_MS + "GET" + "/trade-api/ws/v2"
    and send it as KALSHI-ACCESS-SIGNATURE.
    """

    def __init__(self, cfg: KalshiConfig):
        self.cfg = cfg
        with open(cfg.private_key_path, "rb") as fh:
            self._private_key = serialization.load_pem_private_key(
                fh.read(), password=None
            )

    def _sign(self, method: str, path: str) -> (str, str):
        """
        Return (timestamp_ms, base64_signature).
        """
        ts_ms = str(int(time.time() * 1000))
        payload = ts_ms + method + path.split("?", 1)[0]
        signature_bytes = self._private_key.sign(
            payload.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        signature_b64 = base64.b64encode(signature_bytes).decode("utf-8")
        return ts_ms, signature_b64

    def ws_headers(self) -> Dict[str, str]:
        """
        Build headers for WebSocket handshake.
        """
        ts_ms, sig = self._sign("GET", self.cfg.ws_path_for_signing)
        return {
            "KALSHI-ACCESS-KEY": self.cfg.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": sig,
        }
```

#### Why this design?

* **Single point** of auth logic for both WS & REST (you can reuse `_sign` in your REST client).
* Testable: you can unit-test `_sign` with known vectors.
* The **path to sign** is always taken from `config`, so you can’t accidentally sign `/` instead of `/trade-api/ws/v2`.

---

## 3. Message Envelopes & Typed Models

All WS messages follow a common structure like:([Kalshi API Documentation][9])

* `type`: message type (`"ticker"`, `"orderbook_delta"`, `"fill"`, etc.)
* `sid`: subscription id (optional)
* `seq`: sequence number (for ordered application of deltas, etc.)
* `id`: echoes your command id for some responses
* `msg`: payload object (channel-specific)
* sometimes extras: `subscriptions`, etc.

We’ll model this with **Pydantic** for safety, and build typed payloads for the most important messages.

> Requirement: `pip install pydantic` (v2+ or v1 is fine; code is v1-ish but trivial to adapt).

```python
# kalshi_messages.py

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel


# ---------- Base envelope ----------

class WsEnvelope(BaseModel):
    """
    Generic wrapper for messages from Kalshi's WebSocket API.
    """
    type: str
    sid: Optional[int] = None
    seq: Optional[int] = None
    id: Optional[int] = None
    msg: Optional[Dict] = None
    subscriptions: Optional[List[Dict]] = None  # for list_subscriptions responses


# ---------- Ticker ----------

class TickerMsg(BaseModel):
    market_ticker: str
    price: int                # cents
    yes_bid: Optional[int] = None
    yes_ask: Optional[int] = None
    price_dollars: Optional[str] = None
    yes_bid_dollars: Optional[str] = None
    no_bid_dollars: Optional[str] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    dollar_volume: Optional[int] = None
    dollar_open_interest: Optional[int] = None
    ts: Optional[int] = None  # epoch seconds


# ---------- Orderbook (snapshot + delta) ----------

class OrderbookSnapshotMsg(BaseModel):
    market_ticker: str
    yes: List[List[int]]      # [[price, size], ...]
    yes_dollars: Optional[List[List[str]]] = None
    no: List[List[int]]
    no_dollars: Optional[List[List[str]]] = None


class OrderbookDeltaMsg(BaseModel):
    market_ticker: str
    price: int
    price_dollars: Optional[str] = None
    delta: int                # change in contracts at this price level
    side: Literal["yes", "no"]


# ---------- Public trades ----------

class TradeMsg(BaseModel):
    market_ticker: str
    yes_price: Optional[int] = None
    yes_price_dollars: Optional[str] = None
    no_price: Optional[int] = None
    no_price_dollars: Optional[str] = None
    count: int                # contracts traded
    taker_side: Optional[Literal["yes", "no"]] = None
    ts: Optional[int] = None


# ---------- Fills ----------

class FillMsg(BaseModel):
    trade_id: str
    order_id: str
    market_ticker: str
    is_taker: bool
    side: Literal["yes", "no"]
    yes_price: Optional[int] = None
    yes_price_dollars: Optional[str] = None
    count: int
    action: Literal["buy", "sell"]
    ts: Optional[int] = None
    post_position: Optional[int] = None


# ---------- Market positions ----------

class MarketPositionMsg(BaseModel):
    user_id: str
    market_ticker: str
    position: int                        # net contracts
    position_cost: int                   # in 1/10,000 USD (centi-cents)
    realized_pnl: int                    # centi-cents
    fees_paid: int                       # centi-cents
    volume: int


# ---------- Lifecycle ----------

class MarketLifecycleMsg(BaseModel):
    market_ticker: str
    event_type: str                      # created, updated, settled, etc.
    open_ts: Optional[int] = None
    close_ts: Optional[int] = None
    additional_metadata: Optional[Dict] = None


class EventLifecycleMsg(BaseModel):
    event_ticker: str
    title: Optional[str] = None
    sub_title: Optional[str] = None
    collateral_return_type: Optional[str] = None
    series_ticker: Optional[str] = None
    strike_date: Optional[int] = None


# ---------- Multivariate ----------

class MultivariateLookupMsg(BaseModel):
    collection_ticker: str
    event_ticker: str
    market_ticker: str
    selected_markets: List[Dict]


# ---------- Communications (RFQ + quotes) ----------

class RfqCreatedMsg(BaseModel):
    id: str
    creator_id: str
    market_ticker: str
    event_ticker: str
    contracts: int
    target_cost: int
    target_cost_dollars: Optional[str] = None
    created_ts: Optional[str] = None


class RfqDeletedMsg(BaseModel):
    id: str
    creator_id: str
    market_ticker: str
    event_ticker: str
    contracts: int
    target_cost: int
    target_cost_dollars: Optional[str] = None
    deleted_ts: Optional[str] = None


class QuoteCreatedMsg(BaseModel):
    quote_id: str
    rfq_id: str
    quote_creator_id: str
    rfq_creator_id: str
    market_ticker: str
    event_ticker: str
    yes_bid: Optional[int] = None
    no_bid: Optional[int] = None
    yes_bid_dollars: Optional[str] = None
    no_bid_dollars: Optional[str] = None
    yes_contracts_offered: Optional[int] = None
    no_contracts_offered: Optional[int] = None
    rfq_target_cost: Optional[int] = None
    rfq_target_cost_dollars: Optional[str] = None
    created_ts: Optional[str] = None


class QuoteAcceptedMsg(BaseModel):
    quote_id: str
    rfq_id: str
    quote_creator_id: str
    rfq_creator_id: str
    market_ticker: str
    event_ticker: str
    yes_bid: Optional[int] = None
    no_bid: Optional[int] = None
    yes_bid_dollars: Optional[str] = None
    no_bid_dollars: Optional[str] = None
    accepted_side: Optional[Literal["yes", "no"]] = None
    yes_contracts_offered: Optional[int] = None
    no_contracts_offered: Optional[int] = None
    rfq_target_cost: Optional[int] = None
    rfq_target_cost_dollars: Optional[str] = None
```

### Why this design?

* Downstream logic gets **typed objects**, not raw dicts.
* If Kalshi changes a field name or type, you’ll fail-fast at parse time.
* Your trading / RL layer can be written with **type hints** and IDE autocompletion.

---

## 4. Connection & Subscription Manager

Now we build a single class that:

1. Connects to Kalshi WS with proper headers
2. Sends commands: `subscribe`, `unsubscribe`, `list_subscriptions`, `update_subscription`([Kalshi API Documentation][9])
3. Receives messages and routes them to handlers
4. Handles **reconnection + re-subscription**
5. Integrates ping/pong (with `websockets` library handling control frames automatically)([Kalshi API Documentation][11])

```python
# kalshi_ws_client.py

import asyncio
import json
from typing import Dict, List, Optional, Callable, Awaitable

import websockets
from websockets import WebSocketClientProtocol

from kalshi_config import KalshiConfig
from kalshi_auth import KalshiAuth
from kalshi_messages import WsEnvelope, TickerMsg, OrderbookSnapshotMsg, \
    OrderbookDeltaMsg, TradeMsg, FillMsg, MarketPositionMsg, \
    MarketLifecycleMsg, EventLifecycleMsg, MultivariateLookupMsg, \
    RfqCreatedMsg, RfqDeletedMsg, QuoteCreatedMsg, QuoteAcceptedMsg


MessageHandler = Callable[[WsEnvelope], Awaitable[None]]


class KalshiWSClient:
    """
    High-level WebSocket client for Kalshi.

    Responsibilities:
      - Connect with proper headers
      - Subscribe/unsubscribe
      - Decode envelopes and dispatch to callbacks
      - Handle reconnection and re-subscription
    """

    def __init__(
        self,
        cfg: KalshiConfig,
        reconnect_max_delay: float = 30.0,
    ):
        self.cfg = cfg
        self.auth = KalshiAuth(cfg)
        self.ws: Optional[WebSocketClientProtocol] = None

        self._next_id = 1
        self._sid_to_channel: Dict[int, str] = {}
        self._desired_subscriptions: List[Dict] = []  # for re-subscribe logic

        # Simple callback registry by message type
        self._handlers: Dict[str, List[MessageHandler]] = {}

        # reconnection
        self._reconnect_max_delay = reconnect_max_delay
        self._running = False

    # --------------------- Public API ---------------------

    def on(self, msg_type: str, handler: MessageHandler) -> None:
        """
        Register an async handler for a given message type.
        Example: client.on("ticker", handle_ticker)
        """
        self._handlers.setdefault(msg_type, []).append(handler)

    async def start(self):
        """
        Start the persistent WS loop with automatic reconnection.
        """
        self._running = True
        backoff = 1.0

        while self._running:
            try:
                await self._connect_and_run()
                backoff = 1.0  # reset on clean close
            except Exception as exc:
                print(f"[WS] Connection error: {exc!r}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._reconnect_max_delay)

    async def stop(self):
        """
        Request a graceful shutdown.
        """
        self._running = False
        if self.ws:
            await self.ws.close()

    # ------------------ Internal methods ------------------

    async def _connect_and_run(self):
        headers = self.auth.ws_headers()
        print(f"[WS] Connecting to {self.cfg.ws_url}")
        async with websockets.connect(
            self.cfg.ws_url,
            additional_headers=headers,
            ping_interval=20,   # library will send ping frames
            ping_timeout=20,
        ) as ws:
            self.ws = ws
            print("[WS] Connected, sending initial subscriptions...")
            await self._restore_desired_subscriptions()

            async for raw in ws:
                await self._handle_raw_message(raw)

    async def _restore_desired_subscriptions(self):
        """
        Re-send subscriptions after reconnect.
        """
        for sub in self._desired_subscriptions:
            await self._send_command("subscribe", sub)

    def _next_msg_id(self) -> int:
        mid = self._next_id
        self._next_id += 1
        return mid

    async def _send_command(self, cmd: str, params: Optional[Dict] = None) -> int:
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")
        msg_id = self._next_msg_id()
        payload = {"id": msg_id, "cmd": cmd}
        if params is not None:
            payload["params"] = params
        await self.ws.send(json.dumps(payload))
        return msg_id

    # ---------- Subscription helpers ----------

    async def subscribe(
        self,
        channels: List[str],
        market_tickers: Optional[List[str]] = None,
    ) -> None:
        """
        Subscribe to one or more channels. For market-scoped channels,
        pass the desired tickers.
        """
        params: Dict[str, object] = {"channels": channels}
        if market_tickers:
            if len(market_tickers) == 1:
                params["market_ticker"] = market_tickers[0]
            else:
                params["market_tickers"] = market_tickers

        # Remember this subscription so we can recreate it on reconnect
        self._desired_subscriptions.append(params.copy())
        await self._send_command("subscribe", params)

    async def unsubscribe_sids(self, sids: List[int]) -> None:
        await self._send_command("unsubscribe", {"sids": sids})

    async def list_subscriptions(self) -> None:
        await self._send_command("list_subscriptions")

    async def update_subscription_add_markets(
        self, sid: int, market_tickers: List[str]
    ) -> None:
        params = {
            "sid": sid,
            "market_tickers": market_tickers,
            "action": "add_markets",
        }
        await self._send_command("update_subscription", params)

    # ---------- Incoming message handling ----------

    async def _handle_raw_message(self, raw: str):
        try:
            data = json.loads(raw)
            env = WsEnvelope(**data)
        except Exception as err:
            print(f"[WS] Failed to parse message: {err!r} raw={raw!r}")
            return

        # subscription responses
        if env.type == "subscribed" and env.msg:
            sid = env.msg.get("sid")
            ch = env.msg.get("channel")
            if sid is not None and ch:
                self._sid_to_channel[sid] = ch
                print(f"[WS] Subscribed: sid={sid} channel={ch}")

        elif env.type == "unsubscribed":
            sid = env.sid
            if sid in self._sid_to_channel:
                ch = self._sid_to_channel.pop(sid)
                print(f"[WS] Unsubscribed: sid={sid} channel={ch}")
            else:
                print(f"[WS] Unsubscribed unknown sid={sid}")

        elif env.type == "ok":
            print(f"[WS] OK response (id={env.id})")

        elif env.type == "error" and env.msg:
            code = env.msg.get("code")
            msg = env.msg.get("msg")
            print(f"[WS] ERROR {code}: {msg} (id={env.id})")

        # dispatch to type-specific handlers
        await self._dispatch(env)

    async def _dispatch(self, env: WsEnvelope):
        handlers = self._handlers.get(env.type, [])
        if not handlers:
            # fall back: no explicit handler registered
            return

        for h in handlers:
            try:
                await h(env)
            except Exception as exc:
                print(f"[WS] Handler error for type={env.type}: {exc!r}")
```

### Why this design?

* The **client doesn’t know about orderbooks or strategies** – it just routes messages.
* Reconnection is handled in `start()` and `_connect_and_run()` with **backoff**.
* We keep a list of “desired subscriptions”; on reconnect we re-issue them.
* The `on(msg_type, handler)` pattern lets you plug in many independent consumers (e.g., one for orderbooks, one for logging, one for RL buffer).

---

## 5. Type-Specific Adapters (turn envelopes into typed models)

Your handlers should convert `env.msg` into the typed model appropriate for `env.type`.

Here is a helper module that does that conversion and calls more domain-specific callbacks.

```python
# kalshi_adapters.py

from typing import Awaitable, Callable, Dict

from kalshi_messages import (
    WsEnvelope,
    TickerMsg,
    OrderbookSnapshotMsg,
    OrderbookDeltaMsg,
    TradeMsg,
    FillMsg,
    MarketPositionMsg,
    MarketLifecycleMsg,
    EventLifecycleMsg,
    MultivariateLookupMsg,
    RfqCreatedMsg,
    RfqDeletedMsg,
    QuoteCreatedMsg,
    QuoteAcceptedMsg,
)

TickerHandler = Callable[[TickerMsg], Awaitable[None]]
OrderbookSnapshotHandler = Callable[[OrderbookSnapshotMsg], Awaitable[None]]
OrderbookDeltaHandler = Callable[[OrderbookDeltaMsg], Awaitable[None]]
TradeHandler = Callable[[TradeMsg], Awaitable[None]]
FillHandler = Callable[[FillMsg], Awaitable[None]]
PositionHandler = Callable[[MarketPositionMsg], Awaitable[None]]


async def handle_ticker(env: WsEnvelope, fn: TickerHandler):
    if not env.msg:
        return
    msg = TickerMsg(**env.msg)
    await fn(msg)


async def handle_orderbook_snapshot(env: WsEnvelope, fn: OrderbookSnapshotHandler):
    if not env.msg:
        return
    msg = OrderbookSnapshotMsg(**env.msg)
    await fn(msg)


async def handle_orderbook_delta(env: WsEnvelope, fn: OrderbookDeltaHandler):
    if not env.msg:
        return
    msg = OrderbookDeltaMsg(**env.msg)
    await fn(msg)


async def handle_trade(env: WsEnvelope, fn: TradeHandler):
    if not env.msg:
        return
    msg = TradeMsg(**env.msg)
    await fn(msg)


async def handle_fill(env: WsEnvelope, fn: FillHandler):
    if not env.msg:
        return
    msg = FillMsg(**env.msg)
    await fn(msg)


async def handle_market_position(env: WsEnvelope, fn: PositionHandler):
    if not env.msg:
        return
    msg = MarketPositionMsg(**env.msg)
    await fn(msg)
```

You can add similar tiny wrappers for lifecycle, communications, etc., as needed.

### Why this design?

* Keeps **parsing logic in one place**; your strategy code never has to touch `WsEnvelope`.
* Makes it trivial for a coding agent to extend when new message types appear.

---

## 6. Orderbook Builder (Snapshot + Delta)

The **orderbook channel** sends:([Kalshi API Documentation][1])

* One **snapshot** (`type = "orderbook_snapshot"`) with full depth for a market
* Followed by **deltas** (`type = "orderbook_delta"`) describing updates at a single price level (`price`, `delta`, `side`).

We want a:

* Per-market data structure:

  * `yes_book: Dict[int, int]` – price (cents) → size
  * `no_book: Dict[int, int]` – price (cents) → size
* Methods:

  * `apply_snapshot(snapshot)`
  * `apply_delta(delta)`
  * `best_bid_ask()` or mid, etc.

```python
# kalshi_orderbook.py

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from kalshi_messages import OrderbookSnapshotMsg, OrderbookDeltaMsg


@dataclass
class OrderbookSide:
    """
    One side of the book: price (int cents) -> size (int contracts).
    """
    levels: Dict[int, int] = field(default_factory=dict)

    def apply_levels(self, levels: list[list[int]]) -> None:
        """
        Replace entire side from [[price, size], ...].
        """
        self.levels.clear()
        for price, size in levels:
            if size > 0:
                self.levels[price] = size

    def apply_delta(self, price: int, delta: int) -> None:
        """
        Adjust size at a given price by delta. Remove if size becomes <= 0.
        """
        new_size = self.levels.get(price, 0) + delta
        if new_size > 0:
            self.levels[price] = new_size
        elif price in self.levels:
            del self.levels[price]

    def best_bid(self) -> Optional[Tuple[int, int]]:
        """
        Highest price with positive size.
        """
        if not self.levels:
            return None
        price = max(self.levels.keys())
        return price, self.levels[price]

    def best_ask(self) -> Optional[Tuple[int, int]]:
        """
        Lowest price with positive size.
        """
        if not self.levels:
            return None
        price = min(self.levels.keys())
        return price, self.levels[price]


@dataclass
class MarketOrderbook:
    market_ticker: str
    yes: OrderbookSide = field(default_factory=OrderbookSide)
    no: OrderbookSide = field(default_factory=OrderbookSide)
    last_seq: Optional[int] = None

    def apply_snapshot(self, msg: OrderbookSnapshotMsg, seq: Optional[int] = None):
        self.market_ticker = msg.market_ticker
        self.yes.apply_levels(msg.yes)
        self.no.apply_levels(msg.no)
        self.last_seq = seq

    def apply_delta(self, msg: OrderbookDeltaMsg, seq: Optional[int] = None):
        # Optional sequence handling if you want to detect gaps
        if self.last_seq is not None and seq is not None and seq != self.last_seq + 1:
            print(
                f"[BOOK {self.market_ticker}] Sequence gap: "
                f"expected {self.last_seq + 1}, got {seq}"
            )
            # In a real system you'd trigger a resubscribe or snapshot refresh here.

        side = self.yes if msg.side == "yes" else self.no
        side.apply_delta(msg.price, msg.delta)

        self.last_seq = seq

    def best_yes_bid(self) -> Optional[Tuple[int, int]]:
        return self.yes.best_bid()

    def best_yes_ask(self) -> Optional[Tuple[int, int]]:
        return self.yes.best_ask()

    def best_no_bid(self) -> Optional[Tuple[int, int]]:
        return self.no.best_bid()

    def best_no_ask(self) -> Optional[Tuple[int, int]]:
        return self.no.best_ask()
```

### Why this design?

* This is **pure logic**: no network, no JSON. Easy to unit test with fabricated snapshots + deltas.
* The `last_seq` tracking lets you detect WS gaps and trigger recovery. Kalshi’s WS spec recommends treating sequence as monotonic per subscription.([Kalshi API Documentation][1])

---

## 7. Wiring It All Together (example main)

This “main” script shows how all the pieces collaborate:

* Create config + client
* Subscribe to:

  * `ticker` for all markets
  * `orderbook_delta` for a chosen set of markets
  * `trade`, `fill`, `market_positions` for your account
* Maintain an orderbook per market
* Print a few signals / events as they come in

```python
# run_kalshi_ws_example.py

import asyncio
from typing import Dict

from kalshi_config import KalshiConfig
from kalshi_ws_client import KalshiWSClient
from kalshi_messages import WsEnvelope
from kalshi_messages import (
    TickerMsg,
    OrderbookSnapshotMsg,
    OrderbookDeltaMsg,
    TradeMsg,
    FillMsg,
    MarketPositionMsg,
)
from kalshi_adapters import (
    handle_ticker,
    handle_orderbook_snapshot,
    handle_orderbook_delta,
    handle_trade,
    handle_fill,
    handle_market_position,
)
from kalshi_orderbook import MarketOrderbook


async def main():
    cfg = KalshiConfig(
        key_id="YOUR_KEY_ID",
        private_key_path="/path/to/private_key.pem",
        is_demo=True,
    )

    client = KalshiWSClient(cfg)

    # In-memory orderbooks by market
    books: Dict[str, MarketOrderbook] = {}

    # ------- Handlers that parse envelope -> typed model -> logic -------

    async def on_ticker(env: WsEnvelope):
        async def _logic(msg: TickerMsg):
            # example: just log a simple line
            print(f"[TICKER] {msg.market_ticker} price={msg.price} vol={msg.volume}")
        await handle_ticker(env, _logic)

    async def on_orderbook_snapshot(env: WsEnvelope):
        async def _logic(msg: OrderbookSnapshotMsg):
            ob = books.setdefault(
                msg.market_ticker,
                MarketOrderbook(market_ticker=msg.market_ticker),
            )
            ob.apply_snapshot(msg, seq=env.seq)
            bb = ob.best_yes_bid()
            ba = ob.best_yes_ask()
            print(f"[BOOK] snapshot {msg.market_ticker} best_yes_bid={bb} best_yes_ask={ba}")
        await handle_orderbook_snapshot(env, _logic)

    async def on_orderbook_delta(env: WsEnvelope):
        async def _logic(msg: OrderbookDeltaMsg):
            ob = books.setdefault(
                msg.market_ticker,
                MarketOrderbook(market_ticker=msg.market_ticker),
            )
            ob.apply_delta(msg, seq=env.seq)
        await handle_orderbook_delta(env, _logic)

    async def on_trade(env: WsEnvelope):
        async def _logic(msg: TradeMsg):
            print(
                f"[TRADE] {msg.market_ticker} "
                f"count={msg.count} taker_side={msg.taker_side}"
            )
        await handle_trade(env, _logic)

    async def on_fill(env: WsEnvelope):
        async def _logic(msg: FillMsg):
            print(
                f"[FILL] market={msg.market_ticker} "
                f"side={msg.side} count={msg.count} taker={msg.is_taker}"
            )
        await handle_fill(env, _logic)

    async def on_position(env: WsEnvelope):
        async def _logic(msg: MarketPositionMsg):
            dollars_pos_cost = msg.position_cost / 10_000
            dollars_realized = msg.realized_pnl / 10_000
            print(
                f"[POS] {msg.market_ticker} pos={msg.position} "
                f"cost=${dollars_pos_cost:.4f} realized=${dollars_realized:.4f}"
            )
        await handle_market_position(env, _logic)

    # ------- Register handlers with client -------

    client.on("ticker", on_ticker)
    client.on("orderbook_snapshot", on_orderbook_snapshot)
    client.on("orderbook_delta", on_orderbook_delta)
    client.on("trade", on_trade)
    client.on("fill", on_fill)
    client.on("market_position", on_position)

    # ------- Start client and send initial subscriptions -------

    async def startup_subscriptions():
        # 1) Ticker for all markets
        await client.subscribe(["ticker"])

        # 2) Orderbook deltas for specific markets (replace with your tickers)
        watch_markets = [
            "KXHIGHDEN-25JAN06-T38",  # example
            # add your weather brackets here
        ]
        await client.subscribe(["orderbook_delta"], watch_markets)

        # 3) Public trades for same markets
        await client.subscribe(["trades"], watch_markets)

        # 4) Private channels (fills, positions) – no market filters needed
        await client.subscribe(["fill"])
        await client.subscribe(["market_positions"])

    # Slight trick: start client loop, then call subscriptions once connected
    async def runner():
        await asyncio.sleep(1.0)  # small delay to let connection establish
        await startup_subscriptions()

    # Kick off client + startup in parallel
    await asyncio.gather(client.start(), runner())


if __name__ == "__main__":
    asyncio.run(main())
```

### Why this design?

* `run_kalshi_ws_example.py` is the **minimal working integration** that an agent can adapt.
* Your **strategy code** only sees objects like `TickerMsg`, `MarketOrderbook`, etc.
* You can plug this into your **RL loop**: every incoming event becomes an observation, and you can record to Parquet or feed actions via a queue.

---

## 8. How This Maps to Your Weather / Kalshi Trading Bot

For your specific use case (high-temp bracket markets + RL):

1. **Orderbook feed**

   * Use `MarketOrderbook` per bracket market.
   * Compute **mid**, **best bid/ask**, implied **probabilities**, etc. from book/`ticker`.

2. **State representation**

   * For each RL timestep (e.g., every minute):

     * Read your **local book snapshot** (from `MarketOrderbook`)
     * Join with your **Visual Crossing forecasts / deltas**
     * Add your **position state** from `MarketPositionMsg`

3. **Action execution**

   * Your RL agent decides:

     * Buy/sell YES/NO at specific prices/size.
   * You send those orders via REST (`/orders` endpoints) using the **same KalshiAuth** (reuse `_sign`).([Kalshi API Documentation][12])

4. **Reward stream**

   * Use `MarketPositionMsg` (realized PnL, position) and `FillMsg` to compute incremental rewards and track slippage/fees.([Kalshi API Documentation][5])

5. **Offline training**

   * Record WS traffic to log files:

     * Save raw `WsEnvelope` JSON lines
     * Replay them later into `MarketOrderbook` / RL environment for offline training.

---

## 9. Testing & Hardening Checklist

A coding agent implementing this should also:

1. **Unit-test orderbook logic**

   * Synthetic snapshot `yes=[[10, 5], [20, 3]]` etc.
   * Deltas that add/remove levels; confirm `best_bid/ask` behaviour.

2. **Unit-test parsing**

   * Use sample JSON snippets from the docs:

     * `ticker` message from Market Ticker docs([Kalshi API Documentation][2])
     * `market_position` from Market Positions docs([Kalshi API Documentation][5])
     * RFQ / quotes from Communications docs([Kalshi API Documentation][8])
   * Ensure Pydantic models parse those without error.

3. **Test reconnection**

   * Simulate connection drop by closing `ws` locally, verify:

     * Backoff kicks in
     * Client reconnects
     * Subscriptions are re-sent
     * `MarketOrderbook`s get new snapshots (avoid stale seq gaps forever).

4. **Error code handling**

   * The Quick Start lists WS error codes (missing params, unknown channel, etc.) – ensure you log them and don’t silently swallow.([Kalshi API Documentation][10])

5. **Backpressure management**

   * If your RL logic is heavy, do *not* run it directly in WS handlers.
   * Instead, push parsed messages onto an `asyncio.Queue` and have separate worker tasks consume from that queue.

---
