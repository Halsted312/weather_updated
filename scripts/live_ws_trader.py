#!/home/halsted/Python/weather_updated/.venv/bin/python
"""
Live WebSocket Trading Daemon

Listens to Kalshi order book updates and executes trades based on
ordinal model predictions vs market-implied probabilities.

Usage:
    # Dry-run (no actual orders):
    python scripts/live_ws_trader.py --dry-run

    # Live trading:
    python scripts/live_ws_trader.py --live --bet-size 50 --max-daily-loss 500
"""

import asyncio
import json
import logging
import signal
import sys
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Optional, Set
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo
import time

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import get_engine, get_session
from src.kalshi.client import KalshiClient
from src.config.settings import get_settings
from models.inference.live_engine import LiveInferenceEngine, PredictionResult
from src.trading.fees import find_best_trade, classify_liquidity_role, compute_ev_per_contract
from src.trading.risk import PositionSizer, DailyPnLTracker
from config import live_trader_config as config

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_DIR / 'trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== GLOBALS =====
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle graceful shutdown"""
    global shutdown_requested
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    shutdown_requested = True


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# ===== ORDER BOOK STATE =====
@dataclass
class OrderBookState:
    """Track order book state for a single market"""
    ticker: str
    best_bid: int = 0  # cents
    best_ask: int = 100  # cents
    last_update: datetime = field(default_factory=lambda: datetime.now())

    @property
    def mid_price(self) -> float:
        """Midpoint price"""
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def market_prob(self) -> float:
        """Market-implied probability"""
        return self.mid_price / 100.0

    @property
    def spread(self) -> int:
        """Bid-ask spread in cents"""
        return self.best_ask - self.best_bid


# ===== POSITION TRACKING =====
@dataclass
class PositionTracker:
    """Track open positions by ticker and by (city, event)"""
    positions: Dict[str, int] = field(default_factory=dict)  # ticker → contracts
    event_positions: Dict[tuple, int] = field(default_factory=dict)  # (city, event_date) → count

    def add_position(self, ticker: str, city: str, event_date: date, contracts: int):
        """Add a position"""
        self.positions[ticker] = self.positions.get(ticker, 0) + contracts

        # Track by event
        event_key = (city, event_date)
        self.event_positions[event_key] = self.event_positions.get(event_key, 0) + 1

    def has_position(self, ticker: str) -> bool:
        """Check if we have a position in this ticker"""
        return ticker in self.positions and self.positions[ticker] > 0

    def get_event_position_count(self, city: str, event_date: date) -> int:
        """Get number of positions for a (city, event)"""
        return self.event_positions.get((city, event_date), 0)

    def total_positions(self) -> int:
        """Total number of open positions"""
        return len([t for t, c in self.positions.items() if c > 0])


# ===== MAIN TRADING CLASS =====
class LiveWebSocketTrader:
    """Main trading daemon"""

    def __init__(
        self,
        dry_run: bool = True,
        bet_size: float = config.MAX_BET_SIZE_USD,
        max_daily_loss: float = config.MAX_DAILY_LOSS_USD,
        cities: Optional[list] = None
    ):
        self.dry_run = dry_run
        self.bet_size = bet_size
        self.max_daily_loss = max_daily_loss
        self.cities = cities or config.CITIES

        # Components
        self.engine = get_engine()
        self.inference = LiveInferenceEngine(inference_cooldown_sec=30.0)
        self.kalshi_client = None if dry_run else self._init_kalshi_client()

        # Risk management
        self.position_sizer = PositionSizer(
            bankroll_usd=10000.0,
            kelly_fraction=0.25,
            max_bet_usd=bet_size,
            max_position_contracts=100
        )
        self.daily_pnl = DailyPnLTracker()

        # State
        self.order_books: Dict[str, OrderBookState] = {}
        self.positions = PositionTracker()
        self.seen_tickers: Set[str] = set()

        logger.info(
            f"Initialized LiveWebSocketTrader "
            f"(dry_run={dry_run}, bet_size=${bet_size}, "
            f"max_daily_loss=${max_daily_loss}, cities={self.cities})"
        )

    def _init_kalshi_client(self) -> KalshiClient:
        """Initialize Kalshi REST client"""
        settings = get_settings()
        return KalshiClient(
            api_key=settings.kalshi_api_key,
            private_key_path=settings.kalshi_private_key_path,
            base_url=settings.kalshi_base_url,
        )

    async def run(self):
        """Main entry point"""
        reconnect_delay = config.WS_RECONNECT_DELAY_MIN

        while not shutdown_requested:
            try:
                await self._connect_and_trade()
                reconnect_delay = config.WS_RECONNECT_DELAY_MIN  # Reset on success

            except InvalidStatusCode as e:
                logger.error(f"WebSocket auth failed: {e}")
                break  # Don't retry on auth failure

            except ConnectionClosed as e:
                logger.warning(f"WebSocket closed: {e}, reconnecting in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, config.WS_RECONNECT_DELAY_MAX)

            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, config.WS_RECONNECT_DELAY_MAX)

        logger.info("Shutdown complete")

    async def _connect_and_trade(self):
        """Connect to WebSocket and process messages"""
        settings = get_settings()

        # Build WebSocket URL
        ws_url = settings.kalshi_base_url.replace("https://", "wss://").replace("/trade-api/v2", "/trade-api/ws/v2")

        # Generate auth headers
        headers = self._create_ws_auth_headers()

        logger.info(f"Connecting to {ws_url}...")

        async with websockets.connect(
            ws_url,
            extra_headers=headers,
            ping_interval=config.WS_PING_INTERVAL,
            ping_timeout=config.WS_PING_TIMEOUT,
        ) as websocket:
            logger.info("✓ WebSocket connected")

            # Subscribe to ticker channel (HAS BID/ASK AND ACTUALLY WORKS!)
            subscription = {
                "id": 1,
                "cmd": "subscribe",
                "params": {
                    "channels": ["ticker"],
                    "series_tickers": config.SERIES_TICKERS
                }
            }

            await websocket.send(json.dumps(subscription))
            logger.info(f"✓ Subscribed to ticker channel for {len(config.SERIES_TICKERS)} series")

            # Process messages
            async for raw_message in websocket:
                if shutdown_requested:
                    break

                try:
                    message = json.loads(raw_message)
                    await self._handle_message(message)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")

                except Exception as e:
                    logger.error(f"Error handling message: {e}", exc_info=True)

    def _create_ws_auth_headers(self) -> dict:
        """Create WebSocket authentication headers"""
        settings = get_settings()

        timestamp_ms = int(time.time() * 1000)
        message = f"{timestamp_ms}GET/trade-api/ws/v2"

        # Sign with private key (RSA-PSS)
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        import base64

        with open(settings.kalshi_private_key_path, 'rb') as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)

        signature = private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        return {
            "KALSHI-ACCESS-KEY": settings.kalshi_api_key,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        }

    async def _handle_message(self, message: dict):
        """Process WebSocket message"""
        msg_type = message.get("type")
        channel = message.get("channel")

        if channel == "ticker":
            # Ticker messages have bid/ask directly in msg!
            await self._handle_ticker_update(message.get("msg", {}))
        elif channel == "orderbook_delta":
            await self._handle_orderbook_delta(message.get("msg", {}))

    async def _handle_ticker_update(self, msg: dict):
        """
        Handle ticker message - HAS BID/ASK DIRECTLY!

        Format: {"market_ticker": "...", "yes_bid": 31, "yes_ask": 36, ...}
        """
        ticker = msg.get("market_ticker", "")
        if not ticker:
            return

        yes_bid = msg.get("yes_bid")
        yes_ask = msg.get("yes_ask")

        if yes_bid is None or yes_ask is None:
            return

        # Update order book
        if ticker not in self.order_books:
            self.order_books[ticker] = OrderBookState(ticker=ticker)

        self.order_books[ticker].best_bid = yes_bid
        self.order_books[ticker].best_ask = yes_ask
        self.order_books[ticker].last_update = datetime.now()

        # Try to trade!
        await self._check_trading_opportunity(ticker)

    async def _handle_orderbook_delta(self, msg: dict):
        """
        Handle order book update and potentially trade.

        Message format:
        {
            "market_ticker": "KXHIGHCHI-25NOV28-B33.5",
            "yes": {
                "bid": [[45, 100], [44, 200]],
                "ask": [[48, 50], [49, 100]]
            }
        }
        """
        ticker = msg.get("market_ticker", "")
        if not ticker:
            return

        # Update order book state
        yes_data = msg.get("yes", {})
        bids = yes_data.get("bid", [])
        asks = yes_data.get("ask", [])

        if not bids or not asks:
            return

        # Get best bid/ask
        best_bid = max([price for price, qty in bids], default=0)
        best_ask = min([price for price, qty in asks], default=100)

        # Update state
        if ticker not in self.order_books:
            self.order_books[ticker] = OrderBookState(ticker=ticker)

        self.order_books[ticker].best_bid = best_bid
        self.order_books[ticker].best_ask = best_ask
        self.order_books[ticker].last_update = datetime.now()

        # Check if we should trade
        await self._check_trading_opportunity(ticker)

    async def _check_trading_opportunity(self, ticker: str):
        """Check if we should trade this ticker"""
        # Parse ticker
        city, event_date = self._parse_ticker(ticker)
        if not city or not event_date:
            return

        # Filter by cities
        if city not in self.cities:
            return

        # Check position limits
        if self.positions.has_position(ticker):
            return  # Already have position in this exact ticker

        if self.positions.total_positions() >= config.MAX_POSITIONS:
            logger.warning("Max total positions reached, skipping trade")
            return

        # Check per-city-event limit
        event_position_count = self.positions.get_event_position_count(city, event_date)
        if event_position_count >= config.MAX_PER_CITY_EVENT:
            logger.warning(
                f"Max positions for {city} {event_date} reached "
                f"({event_position_count}/{config.MAX_PER_CITY_EVENT}), skipping trade"
            )
            return

        # Check daily loss limit
        self.daily_pnl.check_daily_reset()
        if self.daily_pnl.is_loss_limit_hit(self.max_daily_loss):
            logger.warning(
                f"Daily loss limit hit (${self.daily_pnl.total_pnl:.2f}), skipping trade"
            )
            return

        # Get order book state
        book = self.order_books.get(ticker)
        if not book:
            return

        # Run inference (with caching)
        with get_session(self.engine) as session:
            try:
                prediction = self.inference.predict(city, event_date, session)

                if prediction is None:
                    return  # Prediction failed validation

            except Exception as e:
                logger.error(f"Inference failed for {city} {event_date}: {e}")
                return

        # Get model probability for this ticker
        model_prob = prediction.bracket_probs.get(ticker)
        if model_prob is None:
            return  # Model didn't predict this bracket (prob < 35%)

        # Make trading decision using fee-aware EV
        decision = self._make_decision(model_prob, book, prediction.settlement_std)

        if decision['action'] != 'pass':
            await self._execute_trade(
                ticker, city, event_date, prediction, book, decision
            )

    def _parse_ticker(self, ticker: str) -> tuple:
        """
        Parse market ticker to extract city and event date.

        Example: KXHIGHCHI-25NOV28-B33.5 → ('chicago', date(2025, 11, 28))
        """
        try:
            parts = ticker.split('-')
            if len(parts) < 2:
                return None, None

            series = parts[0]  # KXHIGHCHI
            date_str = parts[1]  # 25NOV28

            # Map series to city
            series_to_city = {
                'KXHIGHCHI': 'chicago',
                'KXHIGHAUS': 'austin',
                'KXHIGHDEN': 'denver',
                'KXHIGHLAX': 'los_angeles',
                'KXHIGHMIA': 'miami',
                'KXHIGHPHIL': 'philadelphia',
            }

            city = series_to_city.get(series)
            if not city:
                return None, None

            # Parse date: 25NOV28 → 2025-11-28
            event_date = datetime.strptime(date_str, "%d%b%y").date()

            return city, event_date

        except Exception as e:
            logger.error(f"Failed to parse ticker {ticker}: {e}")
            return None, None

    def _make_decision(
        self,
        model_prob: float,
        book: OrderBookState,
        settlement_std: float
    ) -> dict:
        """
        Find best trade using fee-aware EV calculation.

        Evaluates all sides: buy YES, sell YES (and implicitly NO sides)
        Uses real Kalshi fee formula and maker/taker classification.

        Returns:
            {
                'action': 'buy' | 'sell' | 'pass',
                'side': 'yes' | 'no' | None,
                'price': cents (if action != 'pass'),
                'ev_cents': expected value per contract,
                'role': 'maker' | 'taker' | None,
                'reason': str
            }
        """
        # Use find_best_trade from fees module
        side, action, price, ev_cents, role = find_best_trade(
            model_prob=model_prob,
            yes_bid=book.best_bid,
            yes_ask=book.best_ask,
            min_ev_cents=config.MIN_EV_PER_CONTRACT_CENTS,
            maker_fill_prob=config.MAKER_FILL_PROBABILITY
        )

        if side is None:
            return {
                'action': 'pass',
                'side': None,
                'price': None,
                'ev_cents': 0.0,
                'role': None,
                'reason': f'No trade with EV > {config.MIN_EV_PER_CONTRACT_CENTS}¢ (model={model_prob:.2%}, market={book.market_prob:.2%})'
            }

        # Bounds check: Kalshi only accepts prices 1-99
        if price < 1 or price > 99:
            logger.warning(f"Invalid price {price}¢ (must be 1-99), rejecting trade")
            return {
                'action': 'pass',
                'side': None,
                'price': None,
                'ev_cents': 0.0,
                'role': None,
                'reason': f'Invalid price {price}¢'
            }

        return {
            'action': action,
            'side': side,
            'price': price,
            'ev_cents': ev_cents,
            'role': role,
            'reason': f'{role} {action} {side} @ {price}¢, EV={ev_cents:.2f}¢'
        }

    async def _execute_trade(
        self,
        ticker: str,
        city: str,
        event_date: date,
        prediction: PredictionResult,
        book: OrderBookState,
        decision: dict
    ):
        """Execute trade (or log if dry-run)"""
        action = decision['action']
        side = decision['side']
        price = decision['price']
        ev_cents = decision['ev_cents']
        role = decision['role']
        reason = decision['reason']
        model_prob = prediction.bracket_probs[ticker]

        # Calculate position size using Kelly-like sizer
        size_result = self.position_sizer.calculate(
            ev_per_contract_cents=ev_cents,
            price_cents=price,
            model_prob=model_prob,
            settlement_std_degf=prediction.settlement_std,
            current_position=self.positions.positions.get(ticker, 0)
        )

        num_contracts = size_result.num_contracts

        if num_contracts == 0:
            logger.debug(f"Position sizer returned 0 contracts: {size_result.reason}")
            return

        # Log decision
        notional_usd = (price / 100) * num_contracts

        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'city': city,
            'event_date': str(event_date),
            'model_prob': round(model_prob, 4),
            'market_prob': round(book.market_prob, 4),
            'edge_prob': round(model_prob - book.market_prob, 4),
            'ev_per_contract_cents': round(ev_cents, 2),
            'settlement_std_degf': round(prediction.settlement_std, 2),
            'expected_settle': round(prediction.expected_settle, 1),
            'ci_90': [prediction.ci_90_low, prediction.ci_90_high],
            't_base': prediction.t_base,
            'action': action,
            'side': side,
            'role': role,
            'price': price,
            'num_contracts': num_contracts,
            'notional_usd': round(notional_usd, 2),
            'kelly_fraction': round(size_result.kelly_fraction, 4),
            'capped_by': size_result.capped_by,
            'reason': reason,
            'dry_run': self.dry_run,
        }

        # Write to trade log
        with open(config.TRADE_LOG, 'a') as f:
            f.write(json.dumps(trade_log) + '\n')

        if self.dry_run:
            logger.info(
                f"[DRY-RUN] {action.upper()} {side.upper()} @ {price}¢ ({role}) "
                f"| {num_contracts}x @ ${notional_usd:.2f} "
                f"| {ticker} "
                f"| model={model_prob:.1%}, mkt={book.market_prob:.1%}, "
                f"EV={ev_cents:.2f}¢, std={prediction.settlement_std:.2f}°F"
            )
            return

        # Place real order
        try:
            logger.info(
                f"[LIVE] Placing {action} order: {ticker} @ {price}¢ "
                f"({num_contracts} contracts)"
            )

            order_result = self.kalshi_client.create_order(
                ticker=ticker,
                side=side,
                action=action,
                count=num_contracts,
                order_type="limit",  # Always use limit orders
                yes_price=price if side == "yes" else None,
                no_price=price if side == "no" else None,
            )

            order_id = order_result.get('order', {}).get('order_id')
            logger.info(f"✓ Order placed: {order_id}")

            # Track position
            self.positions.add_position(ticker, city, event_date, num_contracts)

            # Log to database
            self._log_order_to_db(ticker, city, event_date, prediction, book, decision, order_id, size_result)

        except Exception as e:
            logger.error(f"✗ Order placement failed: {e}")
            # Write error log
            error_log = {
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'error': str(e),
            }
            with open(config.ERROR_LOG, 'a') as f:
                f.write(json.dumps(error_log) + '\n')

    def _log_order_to_db(
        self,
        ticker: str,
        city: str,
        event_date: date,
        prediction: PredictionResult,
        book: OrderBookState,
        decision: dict,
        order_id: str,
        size_result
    ):
        """Log order to sim.live_orders with full metrics"""
        with get_session(self.engine) as session:
            from sqlalchemy import text

            model_prob = prediction.bracket_probs[ticker]

            # Try to insert with extended fields, fall back to basic if table doesn't have them
            try:
                query = text("""
                    INSERT INTO sim.live_orders (
                        order_id, ticker, city, event_date,
                        model_prob, market_prob, edge,
                        ev_per_contract_cents, settlement_std_degf,
                        price_cents, num_contracts, notional_usd,
                        side, action, role,
                        placed_at, strategy, status
                    )
                    VALUES (
                        :order_id, :ticker, :city, :event_date,
                        :model_prob, :market_prob, :edge,
                        :ev_cents, :std,
                        :price_cents, :num_contracts, :notional,
                        :side, :action, :role,
                        NOW(), 'live_ws_ev_trader', 'pending'
                    )
                """)

                notional = (decision['price'] / 100) * size_result.num_contracts

                session.execute(query, {
                    'order_id': order_id,
                    'ticker': ticker,
                    'city': city,
                    'event_date': event_date,
                    'model_prob': model_prob,
                    'market_prob': book.market_prob,
                    'edge': model_prob - book.market_prob,
                    'ev_cents': decision['ev_cents'],
                    'std': prediction.settlement_std,
                    'price_cents': decision['price'],
                    'num_contracts': size_result.num_contracts,
                    'notional': notional,
                    'side': decision['side'],
                    'action': decision['action'],
                    'role': decision['role'],
                })

                session.commit()

            except Exception as e:
                # Fall back to basic logging if extended fields don't exist
                logger.warning(f"Extended DB logging failed, using basic: {e}")

                query = text("""
                    INSERT INTO sim.live_orders (
                        order_id, ticker, city, event_date,
                        price_cents, num_contracts,
                        placed_at, strategy, status
                    )
                    VALUES (
                        :order_id, :ticker, :city, :event_date,
                        :price_cents, :num_contracts,
                        NOW(), 'live_ws_ev_trader', 'pending'
                    )
                """)

                session.execute(query, {
                    'order_id': order_id,
                    'ticker': ticker,
                    'city': city,
                    'event_date': event_date,
                    'price_cents': decision['price'],
                    'num_contracts': size_result.num_contracts,
                })

                session.commit()


# ===== MAIN =====
def main():
    parser = argparse.ArgumentParser(description='Live WebSocket Trading Daemon')

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry-run mode (no actual orders, default)'
    )

    parser.add_argument(
        '--live',
        action='store_true',
        help='LIVE TRADING MODE (places real orders)'
    )

    parser.add_argument(
        '--bet-size',
        type=float,
        default=config.MAX_BET_SIZE_USD,
        help=f'Bet size per trade in USD (default: ${config.MAX_BET_SIZE_USD})'
    )

    parser.add_argument(
        '--max-daily-loss',
        type=float,
        default=config.MAX_DAILY_LOSS_USD,
        help=f'Max daily loss in USD (default: ${config.MAX_DAILY_LOSS_USD})'
    )

    parser.add_argument(
        '--cities',
        nargs='+',
        choices=config.CITIES,
        help='Cities to trade (default: all)'
    )

    parser.add_argument(
        '--yes',
        action='store_true',
        help='Skip confirmation prompt (for automated/background runs)'
    )

    args = parser.parse_args()

    # Dry-run is default unless --live is specified
    dry_run = not args.live

    if not dry_run:
        logger.warning("=" * 80)
        logger.warning("LIVE TRADING MODE ENABLED - REAL ORDERS WILL BE PLACED")
        logger.warning(f"Bet size: ${args.bet_size}/trade, Max daily loss: ${args.max_daily_loss}")
        logger.warning("=" * 80)

        # Skip confirmation if --yes flag or not a TTY
        if not args.yes and sys.stdin.isatty():
            input("Press ENTER to confirm...")
        else:
            logger.warning("Skipping confirmation (--yes flag or non-interactive)")

    trader = LiveWebSocketTrader(
        dry_run=dry_run,
        bet_size=args.bet_size,
        max_daily_loss=args.max_daily_loss,
        cities=args.cities,
    )

    asyncio.run(trader.run())


if __name__ == '__main__':
    main()
