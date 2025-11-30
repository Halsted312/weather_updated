#!/home/halsted/Python/weather_updated/.venv/bin/python
"""
Active Polling Trader - Checks markets every minute

Instead of waiting for WebSocket orderbook_delta (which only fires on changes),
this actively polls the REST API every minute to find trading opportunities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.kalshi.client import KalshiClient
from src.config.settings import get_settings
from src.db.connection import get_db_session
from models.inference.live_engine import LiveInferenceEngine
from src.trading.fees import find_best_trade
from src.trading.risk import PositionSizer
from config import live_trader_config as config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("ACTIVE POLLING TRADER - Checks every 60 seconds")
    logger.info("=" * 80)

    # Initialize
    settings = get_settings()
    client = KalshiClient(
        api_key=settings.kalshi_api_key,
        private_key_path=settings.kalshi_private_key_path,
        base_url=settings.kalshi_base_url,
    )

    inference = LiveInferenceEngine(inference_cooldown_sec=60.0)
    logger.info(f"✓ Loaded {len(inference.models)} models")

    sizer = PositionSizer(
        bankroll_usd=10000,
        kelly_fraction=0.25,
        max_bet_usd=50,
    )

    positions = set()  # Track tickers we've traded

    logger.info("Starting polling loop (Ctrl+C to stop)...")

    while True:
        try:
            logger.info("")
            logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Checking markets...")

            # Check each city
            for city in ['chicago', 'austin']:
                series = {'chicago': 'KXHIGHCHI', 'austin': 'KXHIGHAUS'}[city]

                # Get open markets
                markets = client.get_all_markets(series_ticker=series, status='open')

                for market in markets:
                    if not market.yes_bid or not market.yes_ask:
                        continue

                    if market.ticker in positions:
                        continue  # Already have position

                    # Parse date
                    try:
                        date_str = market.ticker.split('-')[1]
                        event_date = datetime.strptime(date_str, "%d%b%y").date()
                    except:
                        continue

                    # Only trade TODAY's markets
                    if event_date != date.today():
                        continue

                    # Run inference
                    with get_db_session() as session:
                        prediction = inference.predict(city, event_date, session)

                        if prediction is None:
                            continue

                        model_prob = prediction.bracket_probs.get(market.ticker)
                        if model_prob is None:
                            continue

                    # Find best trade
                    side, action, price, ev, role = find_best_trade(
                        model_prob=model_prob,
                        yes_bid=market.yes_bid,
                        yes_ask=market.yes_ask,
                        min_ev_cents=3.0,
                        maker_fill_prob=0.4
                    )

                    if side is None:
                        continue  # No positive EV trade

                    # Size position
                    size_result = sizer.calculate(
                        ev_per_contract_cents=ev,
                        price_cents=price,
                        model_prob=model_prob,
                        settlement_std_degf=prediction.settlement_std,
                    )

                    if size_result.num_contracts == 0:
                        continue

                    # TRADE FOUND!
                    logger.info("")
                    logger.info("=" * 60)
                    logger.info(f"TRADE OPPORTUNITY: {market.ticker}")
                    logger.info(f"  {action.upper()} {side.upper()} @ {price}¢ ({role})")
                    logger.info(f"  Model: {model_prob:.1%}, Market: {((market.yes_bid + market.yes_ask)/2/100):.1%}")
                    logger.info(f"  EV: {ev:.2f}¢, Std: {prediction.settlement_std:.2f}°F")
                    logger.info(f"  Size: {size_result.num_contracts} contracts (${size_result.notional_usd:.2f})")
                    logger.info("=" * 60)

                    # PLACE ORDER
                    try:
                        result = client.create_order(
                            ticker=market.ticker,
                            side=side,
                            action=action,
                            count=size_result.num_contracts,
                            order_type="limit",
                            yes_price=price if side == "yes" else None,
                        )

                        order_id = result.get('order', {}).get('order_id')
                        logger.info(f"✅ ORDER PLACED: {order_id}")
                        positions.add(market.ticker)

                    except Exception as e:
                        logger.error(f"✗ Order failed: {e}")

            # Wait 60 seconds
            logger.info("Waiting 60 seconds...")
            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("\\nStopping...")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            time.sleep(60)

if __name__ == '__main__':
    main()
