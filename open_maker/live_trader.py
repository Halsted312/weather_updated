#!/usr/bin/env python3
"""
Live Open-Maker Trader - Execute trades at market open using base strategy.

Connects to Kalshi's WebSocket API and monitors for market open events.
When a market opens, it:
1. Loads forecast data from database
2. Applies base strategy logic (forecast + bias -> bracket selection)
3. Places maker limit order at configured price

This uses the same entry logic as the backtest but in real-time.

Usage:
    python -m open_maker.live_trader --dry-run          # Preview without placing orders
    python -m open_maker.live_trader --bet-amount 20    # Live trading with $20 per city
    python -m open_maker.live_trader --use-tuned        # Use tuned params from JSON

Safety:
    - Start with --dry-run to verify behavior
    - Uses maker orders (no taker fees)
    - Does NOT implement intraday exits (hold to settlement only)
"""

import argparse
import asyncio
import base64
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, InvalidStatusCode

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import CITIES, get_settings
from src.db import get_db_session
from src.kalshi.client import KalshiClient
from open_maker.core import (
    OpenMakerParams,
    load_tuned_params,
)
from open_maker.utils import (
    find_bracket_for_temp,
    calculate_position_size,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Our weather series tickers
WEATHER_SERIES = [city.series_ticker for city in CITIES.values()]

# Map series ticker back to city_id
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
    """Derive WebSocket URL from REST base URL."""
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = ws_url.replace("/trade-api/v2", "/trade-api/ws/v2")
    return ws_url


def sign_pss_text(private_key: Any, message: str) -> str:
    """Sign a message using RSA-PSS with SHA256."""
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
    """Create authentication headers for WebSocket connection."""
    timestamp = str(int(time.time() * 1000))
    msg_string = timestamp + "GET" + WS_PATH
    signature = sign_pss_text(private_key, msg_string)
    return {
        "KALSHI-ACCESS-KEY": api_key,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }


def parse_event_ticker(event_ticker: str) -> Optional[date]:
    """Parse event date from event ticker (e.g., KXHIGHCHI-25NOV28 -> 2025-11-28)."""
    import re
    match = re.search(r'-(\d{2})([A-Z]{3})(\d{2})$', event_ticker)
    if not match:
        return None

    year_suffix = int(match.group(1))
    month_str = match.group(2)
    day = int(match.group(3))

    months = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }
    month = months.get(month_str)
    if not month:
        return None

    year = 2000 + year_suffix
    try:
        return date(year, month, day)
    except ValueError:
        return None


@dataclass
class LiveOrder:
    """Track a live order placement."""
    order_id: str
    client_order_id: str
    city: str
    event_date: date
    ticker: str
    side: str
    price_cents: int
    num_contracts: int
    amount_usd: float
    placed_at: datetime
    status: str = "pending"


class LiveOpenMakerTrader:
    """Live trader using open-maker base strategy."""

    def __init__(
        self,
        api_key: str,
        private_key_path: str,
        base_url: str,
        params: OpenMakerParams,
        dry_run: bool = False,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.ws_url = get_ws_url(base_url)
        self.params = params
        self.dry_run = dry_run

        # Load private key
        private_key_file = Path(private_key_path)
        if not private_key_file.exists():
            raise FileNotFoundError(f"Private key not found: {private_key_path}")

        with open(private_key_file, "rb") as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
            )

        # Initialize REST client for placing orders
        if not dry_run:
            self.client = KalshiClient(
                api_key=api_key,
                private_key_path=private_key_path,
                base_url=base_url,
            )
        else:
            self.client = None

        # Track orders placed this session
        self.orders: List[LiveOrder] = []
        self.seen_markets: set = set()

        logger.info("LiveOpenMakerTrader initialized")
        logger.info(f"  WebSocket URL: {self.ws_url}")
        logger.info(f"  Dry run: {self.dry_run}")
        logger.info(f"  Params: entry={params.entry_price_cents}c, bias={params.temp_bias_deg}F")

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

    def _get_forecast_at_open(
        self, session, city: str, event_date: date
    ) -> Optional[float]:
        """Get forecast for a city/date at market open time."""
        from sqlalchemy import select
        from src.db.models import WxForecastSnapshot

        basis_date = event_date - timedelta(days=self.params.basis_offset_days)

        query = select(WxForecastSnapshot.tempmax_fcst_f).where(
            WxForecastSnapshot.city == city,
            WxForecastSnapshot.target_date == event_date,
            WxForecastSnapshot.basis_date == basis_date,
        )
        result = session.execute(query).scalar_one_or_none()
        return float(result) if result else None

    def _load_market_data(
        self, session, city: str, event_date: date
    ):
        """Load market data for bracket selection."""
        from sqlalchemy import select
        from src.db.models import KalshiMarket
        import pandas as pd

        query = select(
            KalshiMarket.ticker,
            KalshiMarket.event_date,
            KalshiMarket.strike_type,
            KalshiMarket.floor_strike,
            KalshiMarket.cap_strike,
        ).where(
            KalshiMarket.city == city,
            KalshiMarket.event_date == event_date,
        )
        rows = session.execute(query).fetchall()

        if not rows:
            return None

        return pd.DataFrame(
            rows,
            columns=["ticker", "event_date", "strike_type", "floor_strike", "cap_strike"]
        )

    async def _handle_market_open(self, data: Dict[str, Any]) -> None:
        """Process a market open event."""
        series_ticker = data.get("series_ticker", "")
        event_ticker = data.get("event_ticker", "")
        market_ticker = data.get("market_ticker", "")

        # Dedupe
        if market_ticker in self.seen_markets:
            return
        self.seen_markets.add(market_ticker)

        # Extract city
        city = SERIES_TO_CITY.get(series_ticker)
        if not city:
            logger.warning(f"Unknown series: {series_ticker}")
            return

        # Parse event date
        event_date = parse_event_ticker(event_ticker)
        if not event_date:
            logger.warning(f"Could not parse event date from {event_ticker}")
            return

        logger.info(f"[OPEN] {market_ticker} | city={city} | event_date={event_date}")

        # Run strategy
        await self._run_strategy(city, event_date, market_ticker)

    async def _run_strategy(
        self, city: str, event_date: date, market_ticker: str
    ) -> None:
        """Run the base strategy and place order if appropriate."""
        try:
            with get_db_session() as session:
                # Get forecast
                temp_fcst = self._get_forecast_at_open(session, city, event_date)
                if temp_fcst is None:
                    logger.warning(f"  No forecast for {city}/{event_date}")
                    return

                # Apply bias
                temp_adjusted = temp_fcst + self.params.temp_bias_deg

                # Load market data
                market_df = self._load_market_data(session, city, event_date)
                if market_df is None or market_df.empty:
                    logger.warning(f"  No market data for {city}/{event_date}")
                    return

                # Find bracket
                bracket = find_bracket_for_temp(market_df, event_date, temp_adjusted)
                if bracket is None:
                    logger.warning(f"  No bracket for temp {temp_adjusted:.1f}F")
                    return

                ticker, floor_strike, cap_strike = bracket

                # Calculate position size
                num_contracts, amount_usd = calculate_position_size(
                    self.params.entry_price_cents,
                    self.params.bet_amount_usd,
                )

                logger.info(
                    f"  TRADE: {ticker} | fcst={temp_fcst:.1f}F, adj={temp_adjusted:.1f}F | "
                    f"bracket=[{floor_strike}, {cap_strike}) | "
                    f"entry={self.params.entry_price_cents}c x {num_contracts} = ${amount_usd:.2f}"
                )

                # Place order
                if not self.dry_run:
                    order_result = await self._place_order(
                        city=city,
                        event_date=event_date,
                        ticker=ticker,
                        num_contracts=num_contracts,
                        amount_usd=amount_usd,
                    )
                    if order_result:
                        logger.info(f"  ORDER PLACED: {order_result.order_id}")
                else:
                    logger.info(f"  [DRY RUN] Would place order for {ticker}")

        except Exception as e:
            logger.error(f"  Strategy error: {e}", exc_info=True)

    async def _place_order(
        self,
        city: str,
        event_date: date,
        ticker: str,
        num_contracts: int,
        amount_usd: float,
    ) -> Optional[LiveOrder]:
        """Place a maker limit order."""
        import uuid

        client_order_id = f"om-{city[:3]}-{event_date.strftime('%m%d')}-{uuid.uuid4().hex[:6]}"

        try:
            result = self.client.create_order(
                ticker=ticker,
                side="yes",
                action="buy",
                count=num_contracts,
                order_type="limit",
                yes_price=int(self.params.entry_price_cents),
                client_order_id=client_order_id,
            )

            order = LiveOrder(
                order_id=result.get("order", {}).get("order_id", "unknown"),
                client_order_id=client_order_id,
                city=city,
                event_date=event_date,
                ticker=ticker,
                side="yes",
                price_cents=int(self.params.entry_price_cents),
                num_contracts=num_contracts,
                amount_usd=amount_usd,
                placed_at=datetime.now(timezone.utc),
                status="submitted",
            )
            self.orders.append(order)

            # Log to database
            await self._log_order(order)

            return order

        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return None

    async def _log_order(self, order: LiveOrder) -> None:
        """Log order to sim.live_orders table."""
        try:
            from sqlalchemy import text
            with get_db_session() as session:
                session.execute(
                    text("""
                        INSERT INTO sim.live_orders
                        (order_id, client_order_id, city, event_date, ticker, side,
                         price_cents, num_contracts, amount_usd, placed_at, status, strategy)
                        VALUES (:order_id, :client_order_id, :city, :event_date, :ticker, :side,
                                :price_cents, :num_contracts, :amount_usd, :placed_at, :status, :strategy)
                    """),
                    {
                        "order_id": order.order_id,
                        "client_order_id": order.client_order_id,
                        "city": order.city,
                        "event_date": order.event_date,
                        "ticker": order.ticker,
                        "side": order.side,
                        "price_cents": order.price_cents,
                        "num_contracts": order.num_contracts,
                        "amount_usd": order.amount_usd,
                        "placed_at": order.placed_at,
                        "status": order.status,
                        "strategy": "open_maker_base",
                    }
                )
                session.commit()
        except Exception as e:
            logger.warning(f"Failed to log order to DB: {e}")

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Process incoming WebSocket message."""
        msg_type = message.get("type", "")
        channel = message.get("channel", "")

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
                await self._handle_market_open(msg_data)

    async def connect_and_trade(self) -> None:
        """Main WebSocket connection loop."""
        global shutdown_requested
        reconnect_delay = 1

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

                    # Reset reconnect delay
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
                reconnect_delay = min(reconnect_delay * 2, 60)

        # Print summary
        logger.info(f"Trader shutting down. Orders placed: {len(self.orders)}")
        for order in self.orders:
            logger.info(
                f"  {order.order_id}: {order.ticker} {order.price_cents}c x {order.num_contracts}"
            )


async def main_async(args):
    """Async main entry point."""
    settings = get_settings()

    # Build params
    if args.use_tuned:
        params = load_tuned_params("open_maker_base", bet_amount_usd=args.bet_amount)
        if params:
            logger.info(f"Using tuned params: {params}")
        else:
            logger.warning("No tuned params found, using defaults")
            params = OpenMakerParams(
                entry_price_cents=30.0,
                temp_bias_deg=1.1,
                basis_offset_days=1,
                bet_amount_usd=args.bet_amount,
            )
    else:
        params = OpenMakerParams(
            entry_price_cents=args.price,
            temp_bias_deg=args.bias,
            basis_offset_days=1,
            bet_amount_usd=args.bet_amount,
        )

    trader = LiveOpenMakerTrader(
        api_key=settings.kalshi_api_key,
        private_key_path=settings.kalshi_private_key_path,
        base_url=settings.kalshi_base_url,
        params=params,
        dry_run=args.dry_run,
    )

    await trader.connect_and_trade()


def main():
    parser = argparse.ArgumentParser(
        description="Live trader for open-maker base strategy"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview trades without placing orders"
    )
    parser.add_argument(
        "--bet-amount", type=float, default=20.0,
        help="Bet amount per city in USD (default: 20)"
    )
    parser.add_argument(
        "--price", type=float, default=30.0,
        help="Entry price in cents (default: 30)"
    )
    parser.add_argument(
        "--bias", type=float, default=1.1,
        help="Temperature bias in degrees F (default: 1.1)"
    )
    parser.add_argument(
        "--use-tuned", action="store_true",
        help="Use tuned parameters from config/*.json"
    )

    args = parser.parse_args()

    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
