#!/usr/bin/env python3
"""
Ad-hoc CLI to view edge decisions for a specific city/date.

Usage:
    python scripts/edge_decision_cli.py --city chicago --event-date 2025-12-02
    python scripts/edge_decision_cli.py --city chicago --event-date 2025-12-02 --config config/edge_trader.json

Displays:
- Forecast vs market implied temperatures
- Edge magnitude and signal
- Edge classifier probability
- Recommended trade (bracket, side, price)
- All bracket probabilities vs market prices

Read-only: No orders placed, no WebSocket needed.
"""

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from live_trading.config import TradingConfig
from live_trading.inference import InferenceWrapper
from src.kalshi.client import KalshiClient
from src.config.settings import get_settings
from src.db.connection import get_db_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Series ticker mapping
SERIES_MAP = {
    'chicago': 'KXHIGHCHI',
    'austin': 'KXHIGHAUS',
    'denver': 'KXHIGHDEN',
    'los_angeles': 'KXHIGHLAX',
    'miami': 'KXHIGHMIA',
    'philadelphia': 'KXHIGHPHI',
}


def get_market_snapshot(kalshi_client: KalshiClient, city: str, event_date: date) -> dict:
    """
    Pull markets for city/date from Kalshi REST API and build snapshot.

    Args:
        kalshi_client: Kalshi client
        city: City identifier
        event_date: Event date

    Returns:
        Market snapshot dict
    """
    series_ticker = SERIES_MAP.get(city)
    if not series_ticker:
        raise ValueError(f"Unknown city: {city}")

    # Get all markets for this series
    logger.info(f"Fetching markets for {series_ticker}...")
    response = kalshi_client.get_markets(series_ticker=series_ticker)  # No status filter
    markets = response.markets  # Pydantic model attribute

    # Filter to event_date (markets are Pydantic objects)
    # Kalshi format: 25DEC02 = YY + MMM + DD
    date_str = event_date.strftime("%y%b%d").upper()  # e.g., "25DEC02"
    matching_markets = [m for m in markets if date_str in m.ticker]

    logger.info(f"Found {len(matching_markets)} markets for {city} {event_date}")

    # Build brackets
    brackets = []
    for market in matching_markets:
        brackets.append({
            'ticker': market.ticker,
            'yes_bid': market.yes_bid or 50,
            'yes_ask': market.yes_ask or 50,
            'floor_strike': market.floor_strike,
            'cap_strike': market.cap_strike,
            'volume': market.volume or 0,
            'open_interest': market.open_interest or 0,
        })

    if not brackets:
        raise ValueError(f"No markets found for {city} {event_date}")

    # Compute best bid/ask
    best_bid = max((b.get('yes_bid') or 0) for b in brackets)
    best_ask = min((b.get('yes_ask') or 100) for b in brackets)

    return {
        'city': city,
        'event_date': event_date,
        'brackets': brackets,
        'best_bid': best_bid,
        'best_ask': best_ask,
        'timestamp': datetime.now(),
        'num_brackets': len(brackets)
    }


def print_decision_summary(decision, market_snapshot, config):
    """Print human-readable decision summary."""

    print("=" * 70)
    print(f"EDGE DECISION: {market_snapshot['city'].upper()} {market_snapshot['event_date']}")
    print("=" * 70)

    print("\n📊 Temperature Analysis:")
    print(f"  Forecast implied: {decision.forecast_implied_temp:.1f}°F")
    print(f"  Market implied:   {decision.market_implied_temp:.1f}°F")
    print(f"  Edge:             {decision.edge_degf:+.2f}°F ({decision.signal})")
    print(f"  Forecast uncertainty: ±{decision.forecast_uncertainty:.2f}°F")

    print("\n🤖 Edge Classifier:")
    print(f"  Probability (edge is real): {decision.edge_classifier_prob:.3f}")
    print(f"  Confidence threshold:       {config.effective_confidence_threshold:.3f}")
    print(f"  Should trade:               {'✅ YES' if decision.should_trade else '❌ NO'}")

    if decision.should_trade:
        print("\n💰 Recommended Trade:")
        print(f"  Bracket:  {decision.recommended_bracket}")
        print(f"  Side:     {decision.recommended_side}")
        print(f"  Action:   {decision.recommended_action}")

        # Find bracket in snapshot
        for bracket in market_snapshot['brackets']:
            if bracket['ticker'] == decision.recommended_bracket:
                print(f"  Bid/Ask:  {bracket['yes_bid']}¢ / {bracket['yes_ask']}¢")
                print(f"  Volume:   {bracket.get('volume', 0)} contracts")
                break
    else:
        print(f"\n❌ No trade recommended: {decision.reason}")

    print("\n📈 All Brackets (Market Prices):")
    print(f"{'Ticker':<40} {'Strikes':<12} {'Bid':<6} {'Ask':<6} {'Mid':<6} {'Vol':<8}")
    print("-" * 70)

    for bracket in sorted(market_snapshot['brackets'], key=lambda b: b.get('floor_strike') or 0):
        ticker = bracket['ticker']
        floor = bracket.get('floor_strike')
        cap = bracket.get('cap_strike')

        if floor is None:
            strike_str = f"<{cap:.0f}"
        elif cap is None:
            strike_str = f">{floor:.0f}"
        else:
            strike_str = f"{floor:.0f}-{cap:.0f}"

        bid = bracket.get('yes_bid', 0)
        ask = bracket.get('yes_ask', 100)
        mid = (bid + ask) / 2.0
        vol = bracket.get('volume', 0)

        marker = " ← RECOMMENDED" if ticker == decision.recommended_bracket else ""

        print(f"{ticker[-20:]:<40} {strike_str:<12} {bid:<6} {ask:<6} {mid:<6.1f} {vol:<8}{marker}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="View edge decision for a specific city/date (read-only)"
    )
    parser.add_argument("--city", type=str, required=True, help="City (chicago, austin, etc.)")
    parser.add_argument("--event-date", type=str, required=True, help="Event date (YYYY-MM-DD)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/edge_trader.json"),
        help="Config file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(args.log_level)

    # Parse event date
    try:
        event_date = date.fromisoformat(args.event_date)
    except ValueError:
        print(f"Error: Invalid date format '{args.event_date}'. Use YYYY-MM-DD")
        sys.exit(1)

    # Load config
    config = TradingConfig.from_json(args.config)
    logger.info(f"Loaded config: aggressiveness={config.aggressiveness:.2f}")

    # Initialize components
    settings = get_settings()
    kalshi_client = KalshiClient(
        api_key=settings.kalshi_api_key,
        private_key_path=settings.kalshi_private_key_path,
        base_url=settings.kalshi_base_url
    )
    inference = InferenceWrapper()

    try:
        # Get market snapshot from REST API
        market_snapshot = get_market_snapshot(kalshi_client, args.city, event_date)

        # Run edge evaluation
        with get_db_session() as session:
            decision = inference.evaluate_edge(
                city=args.city,
                event_date=event_date,
                market_snapshot=market_snapshot,
                session=session,
                edge_threshold_degf=config.edge_threshold_degf,
                confidence_threshold=config.effective_confidence_threshold
            )

        # Print summary
        print_decision_summary(decision, market_snapshot, config)

        # Print model bracket probabilities if available
        if decision.prediction_result and decision.prediction_result.bracket_probs:
            print("\n🎯 Model Bracket Probabilities:")
            probs = decision.prediction_result.bracket_probs
            for ticker in sorted(probs.keys(), key=lambda t: probs[t], reverse=True)[:10]:
                prob = probs[ticker]
                print(f"  {ticker[-20:]:<40} {prob:>6.1%}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=(args.log_level == "DEBUG"))
        sys.exit(1)


if __name__ == "__main__":
    main()
