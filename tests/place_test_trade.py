#!/home/halsted/Python/weather_updated/.venv/bin/python
"""
Quick test script to place a single test trade.

Tests the full flow: get markets → find best price → place order
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import date, datetime
from src.kalshi.client import KalshiClient
from src.config.settings import get_settings
from src.db.connection import get_engine, get_db_session
from sqlalchemy import text


def get_austin_markets(client):
    """Get all open Austin markets from API"""
    print("Fetching Austin markets from Kalshi API...")

    try:
        # Get all markets for Austin series
        markets_response = client.get_all_markets(
            series_ticker="KXHIGHAUS",
            status="open"
        )

        # Response is a list of Market objects
        markets = []
        for market in markets_response:
            markets.append({
                'ticker': market.ticker,
                'yes_bid': market.yes_bid,
                'yes_ask': market.yes_ask,
            })

        print(f"✓ Found {len(markets)} open markets")
        return markets

    except Exception as e:
        print(f"✗ API call failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def find_highest_priced_bracket(markets):
    """Find bracket with highest mid price"""
    best = None
    best_mid = 0

    for market in markets:
        # Get mid price
        if isinstance(market, dict):
            # From API
            ticker = market.get('ticker')
            yes_bid = market.get('yes_bid')
            yes_ask = market.get('yes_ask')
        else:
            # From object
            ticker = market.ticker
            yes_bid = market.yes_bid
            yes_ask = market.yes_ask

        if yes_bid and yes_ask:
            mid = (yes_bid + yes_ask) / 2.0

            if mid > best_mid:
                best_mid = mid
                best = {
                    'ticker': ticker,
                    'yes_bid': yes_bid,
                    'yes_ask': yes_ask,
                    'mid': mid
                }

    return best


def place_test_order(client, ticker, bet_size_usd, dry_run=True):
    """Place a small test order"""
    # Get fresh order book
    print(f"\nGetting order book for {ticker}...")
    try:
        orderbook = client.get_orderbook(ticker, depth=5)

        # Orderbook format: {'yes': [[price, size], ...], 'no': [[price, size], ...]}
        yes_levels = orderbook.get('yes', [])
        no_levels = orderbook.get('no', [])

        if not yes_levels:
            print(f"✗ No YES bid/ask data available")
            return None

        # YES side: bids are descending (best first), asks are ascending (best first)
        # But the response gives us combined - need to figure out which are bids vs asks
        # Actually, looking at the data: 'yes': [[41, 13], [45, 3], [46, 4]]
        # These appear to be on the BID side (people buying YES at these prices)
        # And 'no': [[31, 481], ...] are people buying NO

        # To get YES ask, we need to convert NO bid
        # YES ask = 100 - NO bid
        # YES bid = levels in 'yes'

        yes_bids = yes_levels  # These are YES bids
        if no_levels:
            # NO bid converts to YES ask
            no_bid = max([level[0] for level in no_levels])  # Best NO bid
            yes_ask_from_no = 100 - no_bid
        else:
            yes_ask_from_no = 100

        # Best YES bid is highest price
        best_bid = max([level[0] for level in yes_bids]) if yes_bids else 0

        # For testing, let's just use the highest YES bid we see
        best_ask = yes_ask_from_no

        print(f"  YES bids: {yes_bids}")
        print(f"  NO bids: {no_levels}")
        print(f"  Derived - Best bid: {best_bid}¢, Best ask: {best_ask}¢")

    except Exception as e:
        print(f"✗ Failed to get orderbook: {e}")
        return None

    # Calculate order details
    # Post a maker order 1 cent above best bid
    order_price = best_bid + 1

    # Bounds check: Kalshi only accepts 1-99¢
    if order_price >= 100:
        print(f"  ⚠️  Cannot improve bid (would be {order_price}¢ >= 100¢)")
        print(f"  Using best_bid directly: {best_bid}¢")
        order_price = best_bid
    if order_price < 1:
        order_price = 1

    # Size: $5 bet
    num_contracts = int(bet_size_usd / (order_price / 100))
    if num_contracts == 0:
        num_contracts = 1

    actual_cost = (order_price / 100) * num_contracts

    print(f"\nOrder details:")
    print(f"  Ticker: {ticker}")
    print(f"  Side: YES")
    print(f"  Action: BUY")
    print(f"  Type: LIMIT")
    print(f"  Price: {order_price}¢")
    print(f"  Size: {num_contracts} contracts")
    print(f"  Cost: ${actual_cost:.2f}")
    print(f"  Role: MAKER (0% fee)")

    if dry_run:
        print("\n[DRY-RUN] Would place this order (use --live to execute)")
        return None

    # Place real order
    print("\n🚨 PLACING REAL ORDER...")

    try:
        result = client.create_order(
            ticker=ticker,
            side="yes",
            action="buy",
            count=num_contracts,
            order_type="limit",
            yes_price=order_price,
        )

        order_id = result.get('order', {}).get('order_id')
        print(f"\n✅ ORDER PLACED SUCCESSFULLY!")
        print(f"   Order ID: {order_id}")
        print(f"   Ticker: {ticker}")
        print(f"   Price: {order_price}¢")
        print(f"   Size: {num_contracts} contracts")
        print(f"   Cost: ${actual_cost:.2f}")

        return result

    except Exception as e:
        print(f"\n✗ ORDER FAILED: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Place a test trade')
    parser.add_argument('--live', action='store_true', help='Actually place the order')
    parser.add_argument('--bet-size', type=float, default=5.0, help='Bet size in USD')
    args = parser.parse_args()

    print("=" * 80)
    print("TEST TRADE PLACER - Austin Highest Priced Bracket")
    print("=" * 80)
    print()

    # Initialize
    settings = get_settings()
    engine = get_engine()

    if args.live:
        print("🚨 LIVE MODE - Will place real order")
        client = KalshiClient(
            api_key=settings.kalshi_api_key,
            private_key_path=settings.kalshi_private_key_path,
            base_url=settings.kalshi_base_url,
        )
    else:
        print("✓ DRY-RUN MODE - No actual orders")
        client = KalshiClient(
            api_key=settings.kalshi_api_key,
            private_key_path=settings.kalshi_private_key_path,
            base_url=settings.kalshi_base_url,
        )

    # Get Austin markets
    markets = get_austin_markets(client)

    if not markets:
        print("✗ No markets found")
        return

    # Find highest priced bracket
    best = find_highest_priced_bracket(markets)

    if not best:
        print("✗ No valid brackets found")
        return

    print(f"\nHighest priced bracket:")
    print(f"  Ticker: {best['ticker']}")
    print(f"  Bid: {best['yes_bid']}¢, Ask: {best['yes_ask']}¢")
    print(f"  Mid: {best['mid']:.1f}¢ (implied {best['mid']:.1f}% probability)")

    # Place order
    place_test_order(client, best['ticker'], args.bet_size, dry_run=not args.live)


if __name__ == '__main__':
    main()
