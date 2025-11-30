#!/home/halsted/Python/weather_updated/.venv/bin/python
"""
Test Kalshi WebSocket connection with proper diagnostics.

Based on teacher's feedback - tests:
1. Correct URL
2. Subscription confirmation
3. Message receipt
"""

import asyncio
import json
import time
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import websockets

from src.config.settings import get_settings

async def test_websocket():
    settings = get_settings()

    # Load private key
    with open(settings.kalshi_private_key_path, 'rb') as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)

    # Generate auth
    timestamp_ms = int(time.time() * 1000)
    message = f"{timestamp_ms}GET/trade-api/ws/v2"

    signature = private_key.sign(
        message.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    headers = {
        "KALSHI-ACCESS-KEY": settings.kalshi_api_key,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
    }

    # TRY BOTH URLs
    urls = [
        "wss://trading-api.kalshi.com/trade-api/ws/v2",
        "wss://api.elections.kalshi.com/trade-api/ws/v2",
    ]

    for ws_url in urls:
        print("\n" + "=" * 80)
        print(f"Testing: {ws_url}")
        print("=" * 80)

        try:
            async with websockets.connect(ws_url, extra_headers=headers, ping_interval=30) as ws:
                print("✅ CONNECTED!")

                # Test 1: Subscribe to ticker channel for ALL markets (most active)
                print("\nTest 1: Subscribe to 'ticker' channel (ALL markets)")
                sub1 = {
                    "id": 1,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["ticker"],
                        # NO market_tickers = subscribe to ALL
                    }
                }

                await ws.send(json.dumps(sub1))
                print("  Sent subscription...")

                # Wait for confirmation + messages
                for i in range(10):
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        data = json.loads(msg)
                        msg_type = data.get('type', 'unknown')

                        if msg_type == 'subscribed':
                            print(f"  ✅ SUBSCRIPTION CONFIRMED: {data}")
                        elif 'ticker' in data.get('channel', ''):
                            ticker = data.get('msg', {}).get('market_ticker', 'unknown')
                            print(f"  📊 TICKER UPDATE: {ticker}")
                        else:
                            print(f"  📨 Message: type={msg_type}, data={str(data)[:100]}")

                    except asyncio.TimeoutError:
                        print(f"  ⏱️  Timeout after {i} messages")
                        break

                print("\nTest 2: Subscribe to orderbook_delta for weather series")
                sub2 = {
                    "id": 2,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["orderbook_delta"],
                        "series_tickers": ["KXHIGHCHI", "KXHIGHAUS"],
                    }
                }

                await ws.send(json.dumps(sub2))
                print("  Sent subscription...")

                for i in range(5):
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        data = json.loads(msg)
                        msg_type = data.get('type', 'unknown')

                        if msg_type == 'subscribed':
                            print(f"  ✅ SUBSCRIPTION CONFIRMED: {data}")
                        elif 'orderbook_delta' in data.get('channel', ''):
                            ticker = data.get('msg', {}).get('market_ticker', 'unknown')
                            print(f"  📈 ORDERBOOK: {ticker}")
                        else:
                            print(f"  📨 Message: {str(data)[:100]}")

                    except asyncio.TimeoutError:
                        print(f"  ⏱️  Timeout after {i} messages")
                        break

                print(f"\n✅ {ws_url} WORKS!")
                return  # Success!

        except Exception as e:
            print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
