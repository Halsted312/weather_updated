# Kalshi WebSockets – Offline Reference (v2)

This document summarizes the **Kalshi WebSocket API** based on:

* **Quick Start: WebSockets**([docs.kalshi.com][1])
* **WebSockets Reference pages** (Connection, Keep-Alive, Orderbook, Ticker, Public Trades, User Fills, Market Positions, Market & Event Lifecycle, Multivariate Lookups, Communications)([docs.kalshi.com][2])
* **API Keys** (signing & headers)([docs.kalshi.com][3])

Use this as an offline “mini-spec” so agents can connect, authenticate, subscribe, and process all channels without hitting the docs.

---

## 0. Big Picture

Kalshi’s WebSocket API gives **real-time streaming** of:

* Order book changes
* Trade executions
* Market ticker (price/volume/open interest)
* Your fills
* Your positions / P&L
* Market & event lifecycle
* Multivariate collection updates
* Communications (RFQs / quotes)([docs.kalshi.com][1])

All of this happens over **one authenticated WebSocket connection**. You then **subscribe** to channels over that connection.

---

## 1. Environments & Endpoints

### 1.1 WebSocket URLs

**Production**

```text
wss://api.elections.kalshi.com/trade-api/ws/v2
```

**Demo (sandbox)**

```text
wss://demo-api.kalshi.co/trade-api/ws/v2
```

([docs.kalshi.com][1])

All WebSocket communication goes over this path. The “Websockets” pages also show the base WSS host:

```text
wss://api.elections.kalshi.com
```

([docs.kalshi.com][2])

…but when you actually connect, you use the full path `/trade-api/ws/v2` as above.

---

## 2. Authentication & Signing

All WebSocket connections use the **same RSA key & signing scheme** as REST trading endpoints.([docs.kalshi.com][1])

### 2.1 API Keys

From the web UI, you:

1. Go to **Account / Profile Settings → API Keys**
2. Click **Create New API Key**
3. You receive:

   * **Private key** in PEM (RSA) format
   * **Key ID** (string)

The private key is only shown once; you must store it securely.([docs.kalshi.com][3])

### 2.2 Required Headers

When opening the WebSocket, you **must** include these HTTP headers:([docs.kalshi.com][1])

* `KALSHI-ACCESS-KEY` – your **Key ID**
* `KALSHI-ACCESS-TIMESTAMP` – request timestamp in **milliseconds** since epoch
* `KALSHI-ACCESS-SIGNATURE` – **base64 RSA-PSS** signature over:

```text
timestamp + HTTP_METHOD + PATH_WITHOUT_QUERY
```

For WebSockets, the signed text is:

```text
TIMESTAMP + "GET" + "/trade-api/ws/v2"
```

([docs.kalshi.com][1])

### 2.3 Signature Details (RSA-PSS)

* Algorithm: **RSA-PSS with SHA-256**, MGF1(SHA-256), salt length = digest length.([docs.kalshi.com][3])
* Sign the **UTF-8** encoded message string.
* Base64-encode the resulting signature string and put it in `KALSHI-ACCESS-SIGNATURE`.([docs.kalshi.com][1])

### 2.4 Pseudocode for Signing (Python)

> This is functionally equivalent to the docs, but rewritten so it’s not a verbatim copy.

```python
import base64
import time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

WS_PATH = "/trade-api/ws/v2"

def load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def sign_for_kalshi(private_key, method: str, path: str) -> (str, str):
    ts_ms = str(int(time.time() * 1000))
    to_sign = ts_ms + method + path.split("?", 1)[0]
    signature = private_key.sign(
        to_sign.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return ts_ms, base64.b64encode(signature).decode("utf-8")

def build_ws_headers(private_key, key_id: str) -> dict:
    ts_ms, sig = sign_for_kalshi(private_key, "GET", WS_PATH)
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig,
    }
```

---

## 3. WebSocket Protocol Basics

### 3.1 Connection Lifecycle

The typical lifecycle is:([docs.kalshi.com][1])

1. Generate authentication headers.
2. Open WebSocket connection to `…/trade-api/ws/v2` including those headers.
3. Send **commands** (subscribe, unsubscribe, etc.) as JSON.
4. Receive **streaming messages** from subscribed channels.
5. Handle disconnects with reconnection + resubscribe logic.

### 3.2 Client → Server Commands

All commands share a basic shape:

```jsonc
{
  "id": 123,           // client-chosen request id
  "cmd": "subscribe",  // or "unsubscribe", "list_subscriptions", "update_subscription"
  "params": { ... }    // command-specific parameters
}
```

From the docs:([docs.kalshi.com][2])

#### 3.2.1 Subscribe

Subscribe to one or more channels:

```jsonc
{
  "id": 1,
  "cmd": "subscribe",
  "params": {
    "channels": ["orderbook_delta"],  // required
    // For market-specific channels:
    "market_ticker": "CPI-22DEC-TN0.1"
    // or
    // "market_tickers": ["CPI-22DEC-TN0.1", "CPI-22DEC-TN0.2"]
  }
}
```

* `channels`: array of channel names (e.g. `"ticker"`, `"orderbook_delta"`, `"trades"`, `"fill"`, etc.).
* Some channels ignore market filters (e.g. fills, communications).([docs.kalshi.com][2])

#### 3.2.2 Unsubscribe

Unsubscribe by subscription ID(s):

```jsonc
{
  "id": 124,
  "cmd": "unsubscribe",
  "params": {
    "sids": [1, 2]
  }
}
```

* `sids`: one or more subscription IDs previously returned by the server.([docs.kalshi.com][2])

#### 3.2.3 List Subscriptions

List active subscriptions:

```jsonc
{
  "id": 3,
  "cmd": "list_subscriptions"
}
```

([docs.kalshi.com][2])

#### 3.2.4 Update Subscription

Add or remove markets in an existing subscription:

```jsonc
{
  "id": 124,
  "cmd": "update_subscription",
  "params": {
    "sids": [456],
    "market_tickers": ["NEW-MARKET-1", "NEW-MARKET-2"],
    "action": "add_markets"   // or "delete_markets"
  }
}
```

Single-sid variant: use `sid` instead of `sids`.([docs.kalshi.com][2])

---

### 3.3 Server → Client Envelope

Server messages generally look like:

```jsonc
{
  "type": "ticker" | "orderbook_snapshot" | "orderbook_delta" | "trade" | "fill" | "error" | "subscribed" | "unsubscribed" | "ok" | ...,
  "sid": 11,          // subscription id (if tied to a subscription)
  "seq": 42,          // monotonically increasing per subscription (for ordered processing)
  "id": 1,            // echoes client command id (for some responses)
  "msg": { ... }      // channel-specific payload
}
```

Specific examples from docs:([docs.kalshi.com][4])

* `type = "subscribed"` → server acknowledges a new subscription and returns a `sid`.
* `type = "unsubscribed"` → subscription removed.
* `type = "ok"` → generic success response.
* `type = "error"` → error; see section 3.4.

> ⚠️ **Doc inconsistency:** The Quick Start example uses `data` as the payload property (`data["data"]["market_ticker"]`), while the channel reference pages use `msg`. You should treat **`msg` as canonical**, and write your client to look at `msg`, not `data`.([docs.kalshi.com][1])

---

### 3.4 Error Handling & Error Codes

Error messages look like:([docs.kalshi.com][1])

```jsonc
{
  "id": 123,
  "type": "error",
  "msg": {
    "code": 6,
    "msg": "Already subscribed"
  }
}
```

From the Quick Start, these are the documented **WebSocket error codes**:([docs.kalshi.com][1])

| Code | Meaning (paraphrased)                                |
| ---- | ---------------------------------------------------- |
| 1    | Generic processing failure                           |
| 2    | Missing `params` object in a command                 |
| 3    | Missing `channels` field in `subscribe`              |
| 4    | Missing `sids` in `unsubscribe`                      |
| 5    | Unknown command name                                 |
| 7    | Unknown subscription ID                              |
| 8    | Unknown/invalid channel name                         |
| 9    | Missing auth when accessing private channel          |
| 10   | Channel-specific error                               |
| 11   | Invalid parameter value                              |
| 12   | Exactly one subscription ID required (update action) |
| 13   | Unsupported action in `update_subscription`          |
| 14   | Missing market ticker where required                 |
| 15   | Missing `action` in `update_subscription`            |
| 16   | Market ticker not found                              |
| 17   | Internal server error                                |

Your client should:

* Log `code` and `msg`.
* Decide whether to **retry**, **fix the command**, or **abort** based on the code.

---

## 4. Connection Keep-Alive (Ping / Pong)

Kalshi uses WebSocket **control frames** for connection health:([docs.kalshi.com][5])

* Server sends **Ping (0x9)** frames roughly every 10 seconds.
* Body is the literal string `"heartbeat"`.
* Client must answer with **Pong (0xA)**.
* Client is allowed to send pings as well, Kalshi will respond with pong.

If you use the Python **`websockets`** library, ping/pong handling is automatic; you don’t need to manually respond. Other libraries may require explicit ping/pong handling.([docs.kalshi.com][5])

---

## 5. Data Channels (Reference)

Below are the main channels, their purpose, requirements, and example payload shapes.

### 5.1 Orderbook Updates – `orderbook_delta`

**Type(s)**

* `orderbook_snapshot`
* `orderbook_delta`([docs.kalshi.com][4])

**Use case** – Maintain a full real-time order book.

**Requirements**

* Must specify a market (or markets) in `params`:

  * `market_ticker: "..."` (single)
  * or `market_tickers: ["...", "..."]` (multiple)([docs.kalshi.com][4])
* Server sends:

  * One **full snapshot** (`orderbook_snapshot`) per subscription
  * Then **incremental deltas** (`orderbook_delta`)

**Snapshot shape (example)**

```jsonc
{
  "type": "orderbook_snapshot",
  "sid": 2,
  "seq": 2,
  "msg": {
    "market_ticker": "FED-23DEC-T3.00",
    "yes": [[8, 300], [22, 333]],          // price in cents, size in contracts
    "yes_dollars": [["0.08", 300], ["0.22", 333]],
    "no": [[54, 20], [56, 146]],
    "no_dollars": [["0.54", 20], ["0.56", 146]]
  }
}
```

([docs.kalshi.com][4])

**Delta shape (example)**

```jsonc
{
  "type": "orderbook_delta",
  "sid": 2,
  "seq": 3,
  "msg": {
    "market_ticker": "FED-23DEC-T3.00",
    "price": 96,
    "price_dollars": "0.96",
    "delta": -54,        // change in contracts at this price level
    "side": "yes"        // "yes" or "no"
  }
}
```

([docs.kalshi.com][4])

**Client responsibilities**

* Cache the **latest snapshot** per `sid`.
* Apply each `orderbook_delta` in **sequence order** (use `seq`).
* Rebuild local book if you detect gaps or disconnects.

> Note: Quick Start uses `"orderbook_delta"` as the channel name, but one example shows `["orderbook"]`. Prefer `["orderbook_delta"]` (the channel used in the v2 example).([docs.kalshi.com][1])

---

### 5.2 Market Ticker – `ticker`

**Type**

* `ticker`([docs.kalshi.com][6])

**Use case**

* Current **price**, **volume**, **open interest**, etc. for a market.

**Requirements**

* Market filter is **optional**:

  * With no `market_ticker(s)` param: you get ticks for **all markets**.
  * You may also filter to specific markets (Quick Start suggests `market_tickers` filter).([docs.kalshi.com][1])

**Example message**

```jsonc
{
  "type": "ticker",
  "sid": 11,
  "seq": 100,
  "msg": {
    "market_ticker": "FED-23DEC-T3.00",
    "price": 48,                 // mid or last, in cents
    "yes_bid": 45,
    "yes_ask": 53,
    "price_dollars": "0.480",
    "yes_bid_dollars": "0.450",
    "no_bid_dollars": "0.550",
    "volume": 33896,
    "open_interest": 20422,
    "dollar_volume": 16948,
    "dollar_open_interest": 10211,
    "ts": 1669149841          // epoch seconds
  }
}
```

([docs.kalshi.com][6])

---

### 5.3 Public Trades – `trades` channel (messages of type `trade`)

**Type**

* `trade`([docs.kalshi.com][7])

**Use case**

* Public trade feed; track prints & flow for any (or all) markets.

**Requirements**

* Market filter optional:

  * Omit market filters to receive **all public trades**.
  * Provide market filters to limit feed.([docs.kalshi.com][7])

**Example message**

```jsonc
{
  "type": "trade",
  "sid": 11,
  "seq": 500,
  "msg": {
    "market_ticker": "HIGHNY-22DEC23-B53.5",
    "yes_price": 36,
    "yes_price_dollars": "0.360",
    "no_price": 64,
    "no_price_dollars": "0.640",
    "count": 136,          // contracts filled
    "taker_side": "no",    // "yes" or "no"
    "ts": 1669149841
  }
}
```

([docs.kalshi.com][7])

---

### 5.4 User Fills – `fill`

**Type**

* `fill`([docs.kalshi.com][8])

**Use case**

* Your **individual order fills** (private channel).

**Requirements**

* **Authentication required.**
* Market filters are **ignored**; you always receive all of **your** fills.([docs.kalshi.com][8])

**Example message**

```jsonc
{
  "type": "fill",
  "sid": 13,
  "seq": 42,
  "msg": {
    "trade_id": "d91bc706-ee49-470d-82d8-11418bda6fed",
    "order_id": "ee587a1c-8b87-4dcf-b721-9f6f790619fa",
    "market_ticker": "HIGHNY-22DEC23-B53.5",
    "is_taker": true,
    "side": "yes",                  // "yes" or "no"
    "yes_price": 75,                // in cents
    "yes_price_dollars": "0.750",
    "count": 278,                   // contracts filled
    "action": "buy",                // "buy" or "sell"
    "ts": 1671899397,
    "post_position": 500            // resulting position after this fill
  }
}
```

([docs.kalshi.com][8])

---

### 5.5 Market Positions – `market_positions`

**Type**

* `market_position`([docs.kalshi.com][9])

**Use case**

* Track **your positions, cost basis, realized P&L, fees, and volume** in real time.

**Requirements**

* **Authentication required.**
* Market filter optional:

  * Omit market filters to get updates for **all your positions**.([docs.kalshi.com][9])

**Monetary units**

* Monetary fields (`position_cost`, `realized_pnl`, `fees_paid`) are in **centi-cents** (1/10,000 of a dollar). Divide by 10,000 to get dollars.([docs.kalshi.com][9])

**Example message**

```jsonc
{
  "type": "market_position",
  "sid": 14,
  "seq": 10,
  "msg": {
    "user_id": "user123",
    "market_ticker": "FED-23DEC-T3.00",
    "position": 100,
    "position_cost": 500000,     // = 50.0000 USD
    "realized_pnl": 100000,      // = 10.0000 USD
    "fees_paid": 10000,          // = 1.0000 USD
    "volume": 15
  }
}
```

([docs.kalshi.com][9])

---

### 5.6 Market & Event Lifecycle – `market_lifecycle_v2` & `event_lifecycle`

**Types**

* `market_lifecycle_v2`
* `event_lifecycle`([docs.kalshi.com][10])

**Use case**

* Track **market creation, activation/deactivation, close time changes, determination, settlement** and event creation/changes.

**Requirements**

* Market filter optional. With no filter you see lifecycle events for all markets/events.([docs.kalshi.com][10])

**Market lifecycle example**

```jsonc
{
  "type": "market_lifecycle_v2",
  "sid": 13,
  "seq": 1,
  "msg": {
    "market_ticker": "INXD-23SEP14-B4487",
    "event_type": "created",           // e.g. created, updated, settled, etc.
    "open_ts": 1694635200,
    "close_ts": 1694721600,
    "additional_metadata": {
      "name": "S&P 500 daily return on Sep 14",
      "title": "S&P 500 closes up by 0.02% or more",
      "yes_sub_title": "S&P 500 closes up 0.02%+",
      "no_sub_title": "S&P 500 closes up <0.02%",
      "rules_primary": "...",          // condensed rules text
      "can_close_early": true,
      "expected_expiration_ts": 1694721600,
      "strike_type": "greater",
      "floor_strike": "4487"
    }
  }
}
```

([docs.kalshi.com][10])

**Event lifecycle example**

```jsonc
{
  "type": "event_lifecycle",
  "sid": 5,
  "seq": 2,
  "msg": {
    "event_ticker": "INXD-23SEP14",
    "title": "INX title",
    "sub_title": "INX subtitle",
    "collateral_return_type": "DIRECNET",
    "series_ticker": "INXD",
    "strike_date": 1694721600
  }
}
```

([docs.kalshi.com][10])

---

### 5.7 Multivariate Lookups – `multivariate`

**Type**

* `multivariate_lookup`([docs.kalshi.com][11])

**Use case**

* Notifications related to **multivariate collections** (e.g., combos or correlated legs in a multivariate product).

**Requirements**

* Market filter optional.([docs.kalshi.com][11])

**Example message**

```jsonc
{
  "type": "multivariate_lookup",
  "sid": 13,
  "seq": 3,
  "msg": {
    "collection_ticker": "KXOSCARWINNERS-25",
    "event_ticker": "KXOSCARWINNERS-25C0CE5",
    "market_ticker": "KXOSCARWINNERS-25C0CE5-36353",
    "selected_markets": [
      {
        "event_ticker": "KXOSCARACTO-25",
        "market_ticker": "KXOSCARACTO-25-AB",
        "side": "yes"
      },
      {
        "event_ticker": "KXOSCARACTR-25",
        "market_ticker": "KXOSCARACTR-25-DM",
        "side": "yes"
      }
    ]
  }
}
```

([docs.kalshi.com][11])

---

### 5.8 Communications – `communications`

**Types**

* `rfq_created`
* `rfq_deleted`
* `quote_created`
* `quote_accepted`([docs.kalshi.com][12])

**Use case**

* Real-time **RFQ and quote** notifications for the Kalshi “communications” / RFQ system.

**Requirements**

* **Authentication required.**
* Market filters are ignored.
* You receive:

  * All RFQ events for RFQs **you create**.
  * Quote events when:

    * You created the RFQ **or**
    * You created the quote.([docs.kalshi.com][12])

**Example: RFQ created**

```jsonc
{
  "type": "rfq_created",
  "sid": 15,
  "seq": 1,
  "msg": {
    "id": "rfq_123",
    "creator_id": "comm_abc123",
    "market_ticker": "FED-23DEC-T3.00",
    "event_ticker": "FED-23DEC",
    "contracts": 100,
    "target_cost": 3500,
    "target_cost_dollars": "0.35",
    "created_ts": "2024-12-01T10:00:00Z"
  }
}
```

([docs.kalshi.com][12])

**Example: Quote created**

```jsonc
{
  "type": "quote_created",
  "sid": 15,
  "seq": 2,
  "msg": {
    "quote_id": "quote_456",
    "rfq_id": "rfq_123",
    "quote_creator_id": "comm_def456",
    "rfq_creator_id": "comm_abc123",
    "market_ticker": "FED-23DEC-T3.00",
    "event_ticker": "FED-23DEC",
    "yes_bid": 35,
    "no_bid": 65,
    "yes_bid_dollars": "0.35",
    "no_bid_dollars": "0.65",
    "yes_contracts_offered": 100,
    "no_contracts_offered": 200,
    "rfq_target_cost": 3500,
    "rfq_target_cost_dollars": "0.35",
    "created_ts": "2024-12-01T10:02:00Z"
  }
}
```

([docs.kalshi.com][12])

**Example: Quote accepted**

```jsonc
{
  "type": "quote_accepted",
  "sid": 15,
  "seq": 3,
  "msg": {
    "quote_id": "quote_456",
    "rfq_id": "rfq_123",
    "quote_creator_id": "comm_def456",
    "rfq_creator_id": "comm_abc123",
    "market_ticker": "FED-23DEC-T3.00",
    "event_ticker": "FED-23DEC",
    "yes_bid": 35,
    "no_bid": 65,
    "yes_bid_dollars": "0.35",
    "no_bid_dollars": "0.65",
    "accepted_side": "yes",
    "yes_contracts_offered": 100,
    "no_contracts_offered": 200,
    "rfq_target_cost": 3500,
    "rfq_target_cost_dollars": "0.35"
  }
}
```

([docs.kalshi.com][12])

---

## 6. Putting It Together – Example Python Client Skeleton

Below is a **non-verbatim** but faithful skeleton of a Kalshi WebSocket client using `websockets`. It:

* Loads & signs with the RSA private key.
* Connects to demo or prod.
* Subscribes to ticker + orderbook.
* Dispatches messages by `type`.

This is designed so your coding agent can extend it (e.g., add more channels, persistence, reconnection logic).

```python
"""
kalshi_ws_client.py

Minimal async client for Kalshi WebSockets (v2).
"""

import asyncio
import json
import time
import base64
from typing import Dict, List, Optional

import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


WS_HOST_DEMO = "wss://demo-api.kalshi.co"
WS_PATH = "/trade-api/ws/v2"
WS_URL_DEMO = WS_HOST_DEMO + WS_PATH


class KalshiAuth:
    def __init__(self, key_id: str, private_key_path: str):
        self.key_id = key_id
        with open(private_key_path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(), password=None
            )

    def _sign(self, method: str, path: str) -> (str, str):
        ts_ms = str(int(time.time() * 1000))
        msg = ts_ms + method + path.split("?", 1)[0]
        signature = self.private_key.sign(
            msg.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return ts_ms, base64.b64encode(signature).decode("utf-8")

    def headers_for_ws(self) -> Dict[str, str]:
        ts_ms, sig = self._sign("GET", WS_PATH)
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": sig,
        }


class KalshiWSClient:
    def __init__(self, auth: KalshiAuth, ws_url: str = WS_URL_DEMO):
        self.auth = auth
        self.ws_url = ws_url
        self.ws = None  # type: Optional[websockets.WebSocketClientProtocol]
        self._next_id = 1
        self._sid_to_channel: Dict[int, str] = {}

    def _next_msg_id(self) -> int:
        mid = self._next_id
        self._next_id += 1
        return mid

    async def connect(self):
        headers = self.auth.headers_for_ws()
        self.ws = await websockets.connect(self.ws_url, additional_headers=headers)
        print(f"[WS] Connected to {self.ws_url}")

    async def send_command(self, cmd: str, params: Optional[dict] = None) -> int:
        if self.ws is None:
            raise RuntimeError("WebSocket not connected")
        msg_id = self._next_msg_id()
        payload = {"id": msg_id, "cmd": cmd}
        if params is not None:
            payload["params"] = params
        await self.ws.send(json.dumps(payload))
        return msg_id

    async def subscribe(
        self,
        channels: List[str],
        market_tickers: Optional[List[str]] = None,
    ) -> int:
        params: Dict[str, object] = {"channels": channels}
        if market_tickers:
            # For some channels use "market_tickers" (multiple) or "market_ticker" (single)
            if len(market_tickers) == 1:
                params["market_ticker"] = market_tickers[0]
            else:
                params["market_tickers"] = market_tickers
        msg_id = await self.send_command("subscribe", params)
        return msg_id

    async def unsubscribe(self, sids: List[int]):
        await self.send_command("unsubscribe", {"sids": sids})

    async def list_subscriptions(self):
        await self.send_command("list_subscriptions")

    # -------- Subscription helpers --------

    async def subscribe_ticker_all(self):
        await self.subscribe(["ticker"])

    async def subscribe_orderbook(self, markets: List[str]):
        await self.subscribe(["orderbook_delta"], markets)

    async def subscribe_trades(self, markets: Optional[List[str]] = None):
        await self.subscribe(["trades"], markets)

    # -------- Message loop --------

    async def handle_message(self, raw: str):
        msg = json.loads(raw)
        mtype = msg.get("type")
        sid = msg.get("sid")
        seq = msg.get("seq")
        body = msg.get("msg") or msg.get("data")  # support both, just in case

        if mtype == "subscribed":
            channel = body.get("channel")
            sid_val = body.get("sid")
            self._sid_to_channel[sid_val] = channel
            print(f"[WS] Subscribed sid={sid_val} channel={channel}")

        elif mtype == "unsubscribed":
            sid_val = msg.get("sid")
            if sid_val in self._sid_to_channel:
                ch = self._sid_to_channel.pop(sid_val)
                print(f"[WS] Unsubscribed sid={sid_val} channel={ch}")
            else:
                print(f"[WS] Unsubscribed unknown sid={sid_val}")

        elif mtype == "ok":
            print(f"[WS] OK response: {msg}")

        elif mtype == "error":
            err = body or {}
            print(f"[WS] ERROR code={err.get('code')} msg={err.get('msg')} full={msg}")

        elif mtype in {"orderbook_snapshot", "orderbook_delta"}:
            await self._on_orderbook(mtype, sid, seq, body)

        elif mtype == "ticker":
            await self._on_ticker(sid, seq, body)

        elif mtype == "trade":
            await self._on_trade(sid, seq, body)

        elif mtype == "fill":
            await self._on_fill(sid, seq, body)

        elif mtype == "market_position":
            await self._on_market_position(sid, seq, body)

        elif mtype in {"market_lifecycle_v2", "event_lifecycle"}:
            await self._on_lifecycle(mtype, sid, seq, body)

        elif mtype == "multivariate_lookup":
            await self._on_multivariate(sid, seq, body)

        elif mtype in {"rfq_created", "rfq_deleted", "quote_created", "quote_accepted"}:
            await self._on_communications(mtype, sid, seq, body)

        else:
            print(f"[WS] Unhandled message type={mtype}: {msg}")

    # --- Handlers (stubs for your agent to extend) ---

    async def _on_orderbook(self, kind, sid, seq, body):
        ticker = body.get("market_ticker")
        print(f"[BOOK] {kind} sid={sid} seq={seq} ticker={ticker}")

    async def _on_ticker(self, sid, seq, body):
        t = body.get("market_ticker")
        price = body.get("price")
        print(f"[TICKER] {t}: price={price} (sid={sid}, seq={seq})")

    async def _on_trade(self, sid, seq, body):
        t = body.get("market_ticker")
        side = body.get("taker_side")
        print(f"[TRADE] {t} taker_side={side} (sid={sid}, seq={seq})")

    async def _on_fill(self, sid, seq, body):
        oid = body.get("order_id")
        print(f"[FILL] order_id={oid} (sid={sid}, seq={seq})")

    async def _on_market_position(self, sid, seq, body):
        t = body.get("market_ticker")
        pos = body.get("position")
        print(f"[POS] {t}: position={pos} (sid={sid}, seq={seq})")

    async def _on_lifecycle(self, kind, sid, seq, body):
        print(f"[LIFECYCLE] {kind} {body.get('market_ticker') or body.get('event_ticker')}")

    async def _on_multivariate(self, sid, seq, body):
        print(f"[MV] collection={body.get('collection_ticker')}")

    async def _on_communications(self, kind, sid, seq, body):
        print(f"[COMM] {kind} rfq_id={body.get('rfq_id') or body.get('id')}")

    # -------- Top-level run loop --------

    async def run(self):
        async with websockets.connect(
            self.ws_url, additional_headers=self.auth.headers_for_ws()
        ) as ws:
            self.ws = ws
            print("[WS] Connected, sending initial subscriptions...")

            # Example initial subscriptions:
            await self.subscribe_ticker_all()
            await self.subscribe_orderbook(["KXHARRIS24-LSV"])  # demo ticker

            async for raw in ws:
                await self.handle_message(raw)


async def main():
    auth = KalshiAuth(
        key_id="YOUR_KEY_ID",
        private_key_path="path/to/private_key.pem",
    )
    client = KalshiWSClient(auth, WS_URL_DEMO)
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 7. Best Practices (From Docs, Condensed)

From Kalshi’s Quick Start “Best Practices” section:([docs.kalshi.com][1])

**Connection**

* Implement **automatic reconnection** with exponential backoff.
* After reconnect, **rebuild your state**:

  * Re-authenticate (new headers).
  * Re-subscribe to channels.
  * Request fresh orderbook snapshots (or let the channel send initial snapshot) before applying deltas.

**Data handling**

* Process messages **asynchronously** – do not block the event loop.
* Validate payloads; be robust to missing or malformed fields.
* For orderbooks:

  * Always apply the **snapshot first**, then deltas.
  * Use `seq` to detect out-of-order or missing messages.

**Security**

* Never ship your private key to client-side (browser) code.
* Rotate API keys periodically; store private keys securely (file permissions, key vault, etc.).([docs.kalshi.com][3])

**Performance**

* Subscribe only to the channels/markets you actually need.
* If volume is heavy, batch downstream processing (e.g. queue + worker threads) so the WebSocket handler remains fast.([docs.kalshi.com][1])

---
