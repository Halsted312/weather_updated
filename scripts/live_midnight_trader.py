#!/usr/bin/env python3
"""
Live Midnight Trader - Execute trades at market open based on midnight forecast.

Trades up to $50 per city on the bracket containing the forecasted high temp.

Usage:
    python scripts/live_midnight_trader.py --dry-run  # Preview trades without executing
    python scripts/live_midnight_trader.py --max-bet 50  # Live trading with $50 max per city
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from typing import Optional, Tuple, List, Dict
from zoneinfo import ZoneInfo

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from sqlalchemy import select, text, func
from src.config import CITIES, get_city
from src.db import get_db_session
from src.db.models import (
    WxForecastSnapshot,
    KalshiMarket,
    KalshiCandle1m,
)
from src.config.settings import get_settings
# Note: KalshiClient import will need adjustment when implementing live trading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Strategy parameters (from Optuna tuning)
DEFAULT_ALPHA = 0.0
DEFAULT_BETA = 0.0


def get_midnight_forecast(session, city: str, event_date: date) -> Optional[float]:
    """Get the midnight forecast (tempmax_t0) for a city/date."""
    query = select(WxForecastSnapshot.tempmax_fcst_f).where(
        WxForecastSnapshot.city == city,
        WxForecastSnapshot.target_date == event_date,
        WxForecastSnapshot.lead_days == 0,
        WxForecastSnapshot.basis_date == event_date,
    )
    result = session.execute(query).scalar_one_or_none()
    return float(result) if result else None


def get_3day_forecasts(session, city: str, event_date: date) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Get t0, t1, t2 forecasts for trend calculation."""
    t0 = get_midnight_forecast(session, city, event_date)

    # t1: tomorrow's forecast from today's basis
    query_t1 = select(WxForecastSnapshot.tempmax_fcst_f).where(
        WxForecastSnapshot.city == city,
        WxForecastSnapshot.basis_date == event_date,
        WxForecastSnapshot.lead_days == 1,
    )
    t1 = session.execute(query_t1).scalar_one_or_none()

    # t2: day after tomorrow from today's basis
    query_t2 = select(WxForecastSnapshot.tempmax_fcst_f).where(
        WxForecastSnapshot.city == city,
        WxForecastSnapshot.basis_date == event_date,
        WxForecastSnapshot.lead_days == 2,
    )
    t2 = session.execute(query_t2).scalar_one_or_none()

    return (
        float(t0) if t0 else None,
        float(t1) if t1 else None,
        float(t2) if t2 else None,
    )


def calculate_adjusted_temp(t0: float, t1: float, t2: float, alpha: float, beta: float) -> float:
    """Calculate trend-adjusted temperature estimate."""
    mean_3d = (t0 + t1 + t2) / 3
    range_3d = max(t0, t1, t2) - min(t0, t1, t2)
    return t0 + alpha * (mean_3d - t0) + beta * range_3d


def find_bracket_for_temp(session, city: str, event_date: date, temp: float) -> Optional[Dict]:
    """Find the bracket containing a temperature."""
    query = select(
        KalshiMarket.ticker,
        KalshiMarket.strike_type,
        KalshiMarket.floor_strike,
        KalshiMarket.cap_strike,
    ).where(
        KalshiMarket.city == city,
        KalshiMarket.event_date == event_date,
    ).order_by(KalshiMarket.floor_strike)

    markets = session.execute(query).fetchall()

    # First: check "between" brackets
    for row in markets:
        if row.strike_type == "between":
            if row.floor_strike is not None and row.cap_strike is not None:
                if row.floor_strike <= temp < row.cap_strike:
                    return {
                        "ticker": row.ticker,
                        "strike_type": row.strike_type,
                        "floor_strike": row.floor_strike,
                        "cap_strike": row.cap_strike,
                    }

    # Second: check "less" tail brackets
    for row in markets:
        if row.strike_type in ("less", "less_or_equal"):
            if row.cap_strike is not None and temp < row.cap_strike:
                return {
                    "ticker": row.ticker,
                    "strike_type": row.strike_type,
                    "floor_strike": row.floor_strike,
                    "cap_strike": row.cap_strike,
                }

    # Third: check "greater" tail brackets (only if cap_strike is NULL - true tail)
    for row in markets:
        if row.strike_type in ("greater", "greater_or_equal"):
            if row.floor_strike is not None and row.cap_strike is None and temp >= row.floor_strike:
                return {
                    "ticker": row.ticker,
                    "strike_type": row.strike_type,
                    "floor_strike": row.floor_strike,
                    "cap_strike": row.cap_strike,
                }

    return None


def get_current_ask_price(client, ticker: str) -> Optional[float]:
    """Get current best ask price from Kalshi API."""
    if client is None:
        return None
    try:
        orderbook = client.get_orderbook(ticker)
        if orderbook and orderbook.get("yes") and orderbook["yes"].get("asks"):
            asks = orderbook["yes"]["asks"]
            if asks:
                # Best ask is the lowest price
                best_ask = min(asks, key=lambda x: x["price"])
                return best_ask["price"]  # Already in cents
    except Exception as e:
        logger.warning(f"Failed to get orderbook for {ticker}: {e}")
    return None


def get_market_status(client, ticker: str) -> Optional[str]:
    """Get market status from Kalshi API."""
    if client is None:
        return "active"  # Assume active for dry-run
    try:
        market = client.get_market(ticker)
        if market:
            return market.get("status")
    except Exception as e:
        logger.warning(f"Failed to get market status for {ticker}: {e}")
    return None


def place_order(
    client,
    ticker: str,
    side: str,  # "yes" or "no"
    price_cents: int,
    num_contracts: int,
    dry_run: bool = True,
) -> Optional[Dict]:
    """Place an order on Kalshi."""
    if dry_run:
        logger.info(f"[DRY RUN] Would place: {side.upper()} {num_contracts} @ {price_cents}c on {ticker}")
        return {"dry_run": True, "ticker": ticker, "side": side, "price": price_cents, "count": num_contracts}

    if client is None:
        logger.error("No Kalshi client available for live trading")
        return None

    try:
        response = client.create_order(
            ticker=ticker,
            side=side,
            action="buy",
            type="limit",
            count=num_contracts,
            yes_price=price_cents if side == "yes" else None,
            no_price=price_cents if side == "no" else None,
        )
        logger.info(f"Order placed: {response}")
        return response
    except Exception as e:
        logger.error(f"Failed to place order: {e}")
        return None


def get_trade_recommendation(
    session,
    city: str,
    event_date: date,
    alpha: float,
    beta: float,
) -> Optional[Dict]:
    """Get trade recommendation for a city/date."""
    # Get forecasts
    t0, t1, t2 = get_3day_forecasts(session, city, event_date)

    if t0 is None:
        logger.warning(f"{city}: No forecast data for {event_date}")
        return None

    # Use t0 if t1/t2 missing
    t1 = t1 if t1 is not None else t0
    t2 = t2 if t2 is not None else t0

    # Calculate adjusted temp
    t_adj = calculate_adjusted_temp(t0, t1, t2, alpha, beta)

    # Find bracket
    bracket = find_bracket_for_temp(session, city, event_date, t_adj)

    if bracket is None:
        logger.warning(f"{city}: No bracket found for temp {t_adj:.1f}")
        return None

    return {
        "city": city,
        "event_date": event_date,
        "forecast_t0": t0,
        "forecast_t1": t1,
        "forecast_t2": t2,
        "adjusted_temp": t_adj,
        **bracket,
    }


def execute_trades(
    recommendations: List[Dict],
    max_bet_usd: float,
    dry_run: bool = True,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
):
    """Execute trades based on recommendations."""
    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE - No actual trades will be placed")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("LIVE TRADING MODE")
        logger.info("=" * 60)

    # Create Kalshi client for live trading
    client = None
    if not dry_run:
        logger.error("Live trading not implemented yet - use Kalshi web UI")
        logger.info("Here are the recommended trades to execute manually:")
        dry_run = True  # Fall through to show recommendations

    results = []

    for rec in recommendations:
        city = rec["city"]
        ticker = rec["ticker"]

        logger.info(f"\n{city.upper()} - {rec['event_date']}")
        logger.info(f"  Forecast: {rec['forecast_t0']:.1f}F (t0), {rec['forecast_t1']:.1f}F (t1), {rec['forecast_t2']:.1f}F (t2)")
        logger.info(f"  Adjusted: {rec['adjusted_temp']:.1f}F (alpha={alpha}, beta={beta})")
        logger.info(f"  Bracket: {ticker} ({rec['strike_type']} [{rec['floor_strike']}, {rec['cap_strike']})")

        # Get current price
        ask_price = None
        if client:
            status = get_market_status(client, ticker)
            if status != "active":
                logger.warning(f"  Market not active (status={status}), skipping")
                continue

            ask_price = get_current_ask_price(client, ticker)
            if ask_price is None:
                logger.warning(f"  No ask price available, skipping")
                continue

            logger.info(f"  Current ask: {ask_price}c")
        else:
            # Dry run - estimate price
            ask_price = 50  # Assume 50c for estimation
            logger.info(f"  Estimated ask: ~{ask_price}c (dry run)")

        # Calculate position size
        cost_per_contract = ask_price / 100  # Convert cents to dollars
        num_contracts = int(max_bet_usd / cost_per_contract)
        if num_contracts < 1:
            num_contracts = 1

        actual_cost = num_contracts * cost_per_contract

        logger.info(f"  Position: {num_contracts} contracts @ {ask_price}c = ${actual_cost:.2f}")

        # Place order
        result = place_order(
            client=client,
            ticker=ticker,
            side="yes",
            price_cents=int(ask_price),
            num_contracts=num_contracts,
            dry_run=dry_run,
        )

        results.append({
            "city": city,
            "ticker": ticker,
            "contracts": num_contracts,
            "price_cents": ask_price,
            "cost_usd": actual_cost,
            "result": result,
        })

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRADE SUMMARY")
    logger.info("=" * 60)

    total_cost = sum(r["cost_usd"] for r in results)
    logger.info(f"Total positions: {len(results)}")
    logger.info(f"Total cost: ${total_cost:.2f}")

    for r in results:
        status = "OK" if r["result"] else "FAILED"
        logger.info(f"  {r['city']}: {r['contracts']} @ {r['price_cents']}c = ${r['cost_usd']:.2f} [{status}]")


def main():
    parser = argparse.ArgumentParser(
        description="Live midnight trader for Kalshi weather markets"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Preview trades without executing (default: True)"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Actually execute trades (overrides --dry-run)"
    )
    parser.add_argument(
        "--max-bet", type=float, default=50.0,
        help="Maximum bet per city in USD (default: $50)"
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"Alpha parameter (default: {DEFAULT_ALPHA})"
    )
    parser.add_argument(
        "--beta", type=float, default=DEFAULT_BETA,
        help=f"Beta parameter (default: {DEFAULT_BETA})"
    )
    parser.add_argument(
        "--city", action="append",
        help="Specific city to trade (can specify multiple)"
    )
    parser.add_argument(
        "--all-cities", action="store_true",
        help="Trade all cities"
    )
    parser.add_argument(
        "--date", type=str,
        help="Event date (YYYY-MM-DD), default: tomorrow"
    )

    args = parser.parse_args()

    # Determine dry run mode
    dry_run = not args.live

    # Determine cities
    if args.all_cities:
        cities = list(CITIES.keys())
    elif args.city:
        cities = args.city
    else:
        cities = ["chicago"]  # Default

    # Determine event date (tomorrow by default)
    if args.date:
        event_date = date.fromisoformat(args.date)
    else:
        event_date = date.today() + timedelta(days=1)

    logger.info(f"Event date: {event_date}")
    logger.info(f"Cities: {cities}")
    logger.info(f"Max bet: ${args.max_bet}")
    logger.info(f"Parameters: alpha={args.alpha}, beta={args.beta}")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")

    # Get recommendations
    recommendations = []
    with get_db_session() as session:
        for city in cities:
            rec = get_trade_recommendation(
                session=session,
                city=city,
                event_date=event_date,
                alpha=args.alpha,
                beta=args.beta,
            )
            if rec:
                recommendations.append(rec)

    if not recommendations:
        logger.warning("No trade recommendations generated!")
        return

    logger.info(f"\nGenerated {len(recommendations)} trade recommendations")

    # Execute trades
    execute_trades(
        recommendations=recommendations,
        max_bet_usd=args.max_bet,
        dry_run=dry_run,
        alpha=args.alpha,
        beta=args.beta,
    )


if __name__ == "__main__":
    main()
