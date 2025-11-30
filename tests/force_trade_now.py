#!/home/halsted/Python/weather_updated/.venv/bin/python
"""
Force a trade RIGHT NOW to test the complete system.

Finds an active market, runs inference, calculates EV, and trades if positive.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.kalshi.client import KalshiClient
from src.config.settings import get_settings
from src.db.connection import get_db_session
from models.inference.live_engine import LiveInferenceEngine
from src.trading.fees import find_best_trade
from src.trading.risk import PositionSizer
from config import live_trader_config as config

print("=" * 80)
print("FORCE TRADE NOW - Complete System Test")
print("=" * 80)
print()

# Initialize
settings = get_settings()
client = KalshiClient(
    api_key=settings.kalshi_api_key,
    private_key_path=settings.kalshi_private_key_path,
    base_url=settings.kalshi_base_url,
)

# Load inference engine
print("Loading models...")
inference = LiveInferenceEngine(inference_cooldown_sec=30.0)
print(f"✓ Loaded {len(inference.models)} city models")
print()

# Get all open markets
print("Fetching open markets for all 6 cities...")
all_opportunities = []

for city in ['austin', 'chicago', 'denver', 'los_angeles', 'miami', 'philadelphia']:
    series_ticker = {
        'chicago': 'KXHIGHCHI',
        'austin': 'KXHIGHAUS',
        'denver': 'KXHIGHDEN',
        'los_angeles': 'KXHIGHLAX',
        'miami': 'KXHIGHMIA',
        'philadelphia': 'KXHIGHPHIL',
    }[city]

    try:
        markets = client.get_all_markets(series_ticker=series_ticker, status="open")

        for market in markets[:3]:  # Check first 3 markets per city
            if market.yes_bid and market.yes_ask:
                all_opportunities.append({
                    'city': city,
                    'ticker': market.ticker,
                    'event_date': market.event_ticker.split('-')[1],  # Extract date
                    'yes_bid': market.yes_bid,
                    'yes_ask': market.yes_ask,
                })

    except Exception as e:
        print(f"  {city}: {e}")

print(f"✓ Found {len(all_opportunities)} potential opportunities")
print()

# Run inference and find best trade
print("Analyzing opportunities...")

best_opportunity = None
best_ev = 0

with get_db_session() as session:
    for opp in all_opportunities[:10]:  # Check first 10
        city = opp['city']

        # Parse event date
        try:
            event_date = datetime.strptime(opp['event_date'], "%d%b%y").date()
        except:
            continue

        # Run inference
        try:
            prediction = inference.predict(city, event_date, session)

            if prediction is None:
                continue

            # Get model prob for this ticker
            model_prob = prediction.bracket_probs.get(opp['ticker'])

            if model_prob is None:
                continue

            # Find best trade
            side, action, price, ev, role = find_best_trade(
                model_prob=model_prob,
                yes_bid=opp['yes_bid'],
                yes_ask=opp['yes_ask'],
                min_ev_cents=1.0,  # Lower threshold for testing
                maker_fill_prob=0.4
            )

            if side and ev > best_ev:
                best_ev = ev
                best_opportunity = {
                    **opp,
                    'model_prob': model_prob,
                    'prediction': prediction,
                    'decision': {
                        'side': side,
                        'action': action,
                        'price': price,
                        'ev': ev,
                        'role': role
                    }
                }

                print(f"  {city}: {opp['ticker']} - EV={ev:.2f}¢ ({role} {action} {side} @ {price}¢)")

        except Exception as e:
            print(f"  {city}: Error - {e}")

print()

if not best_opportunity:
    print("✗ No positive EV opportunities found")
    print("  This could mean:")
    print("  1. Markets are efficiently priced")
    print("  2. Not enough observation data yet")
    print("  3. Markets are closed/settled")
    sys.exit(0)

# Show best opportunity
opp = best_opportunity
dec = opp['decision']

print("=" * 80)
print("BEST OPPORTUNITY FOUND!")
print("=" * 80)
print(f"City: {opp['city'].upper()}")
print(f"Ticker: {opp['ticker']}")
print(f"Event: {opp['event_date']}")
print()
print(f"Model prediction:")
print(f"  Probability: {opp['model_prob']:.1%}")
print(f"  Expected settle: {opp['prediction'].expected_settle:.1f}°F")
print(f"  Uncertainty (std): {opp['prediction'].settlement_std:.2f}°F")
print(f"  90% CI: [{opp['prediction'].ci_90_low}, {opp['prediction'].ci_90_high}]°F")
print()
print(f"Market:")
print(f"  Bid: {opp['yes_bid']}¢, Ask: {opp['yes_ask']}¢")
print(f"  Mid: {(opp['yes_bid'] + opp['yes_ask']) / 2:.1f}¢")
print()
print(f"Best trade:")
print(f"  {dec['action'].upper()} {dec['side'].upper()} @ {dec['price']}¢ ({dec['role']})")
print(f"  Expected value: {dec['ev']:.2f}¢ per contract")
print()

# Position sizing
sizer = PositionSizer(
    bankroll_usd=10000.0,
    kelly_fraction=0.25,
    max_bet_usd=50.0,
)

size = sizer.calculate(
    ev_per_contract_cents=dec['ev'],
    price_cents=dec['price'],
    model_prob=opp['model_prob'],
    settlement_std_degf=opp['prediction'].settlement_std,
)

print(f"Position sizing:")
print(f"  Contracts: {size.num_contracts}")
print(f"  Notional: ${size.notional_usd:.2f}")
print(f"  Kelly fraction: {size.kelly_fraction:.4f}")
print()

# Place trade
print("=" * 80)
response = input("Place this trade? (yes/no): ")

if response.lower() == 'yes':
    print("Placing order...")

    try:
        result = client.create_order(
            ticker=opp['ticker'],
            side=dec['side'],
            action=dec['action'],
            count=size.num_contracts,
            order_type="limit",
            yes_price=dec['price'] if dec['side'] == 'yes' else None,
            no_price=dec['price'] if dec['side'] == 'no' else None,
        )

        order_id = result.get('order', {}).get('order_id')
        print(f"\n✅ ORDER PLACED!")
        print(f"   Order ID: {order_id}")
        print(f"   System: VERIFIED WORKING!")

    except Exception as e:
        print(f"\n✗ Order failed: {e}")
else:
    print("Trade cancelled")
