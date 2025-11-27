#!/usr/bin/env python3
"""
Manual Trade Script - Place single test orders on demand.

This script allows you to manually place orders for testing purposes,
using the same logic as live_trader.py but without waiting for WebSocket events.

Use this to:
1. Test that the full order flow works (DB -> forecast -> bracket -> order)
2. Place small test trades on already-open markets
3. Verify Kalshi API connectivity and order placement

Usage:
    # Dry run for single city
    python -m open_maker.manual_trade --city chicago --event-date 2025-11-27 --bet-amount 5 --use-tuned --dry-run

    # Live $5 test trade
    python -m open_maker.manual_trade --city chicago --event-date 2025-11-27 --bet-amount 5 --use-tuned

    # Place trades for all 6 cities
    python -m open_maker.manual_trade --all-cities --event-date 2025-11-27 --bet-amount 5 --use-tuned

    # Manual params
    python -m open_maker.manual_trade --city chicago --event-date 2025-11-27 --bet-amount 5 --price 30 --bias 1.1
"""

import argparse
import json
import logging
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

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


@dataclass
class TradeResult:
    """Result of a manual trade attempt."""
    city: str
    event_date: date
    ticker: str
    temp_fcst: float
    temp_adjusted: float
    floor_strike: Optional[float]
    cap_strike: Optional[float]
    entry_price_cents: float
    num_contracts: int
    amount_usd: float
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: str = "pending"
    error: Optional[str] = None


def get_forecast_at_open(
    session,
    city: str,
    event_date: date,
    basis_offset_days: int = 1,
) -> Optional[float]:
    """
    Get forecast for a city/date at market open time.

    Args:
        session: SQLAlchemy session
        city: City ID
        event_date: Target event date
        basis_offset_days: How many days before event_date (1 = yesterday's forecast)

    Returns:
        Forecast temperature in F, or None if not found
    """
    from sqlalchemy import select
    from src.db.models import WxForecastSnapshot

    basis_date = event_date - timedelta(days=basis_offset_days)

    query = select(WxForecastSnapshot.tempmax_fcst_f).where(
        WxForecastSnapshot.city == city,
        WxForecastSnapshot.target_date == event_date,
        WxForecastSnapshot.basis_date == basis_date,
    )
    result = session.execute(query).scalar_one_or_none()
    return float(result) if result else None


def load_market_data(
    session,
    city: str,
    event_date: date,
) -> Optional[pd.DataFrame]:
    """
    Load market data for bracket selection from local database.

    Args:
        session: SQLAlchemy session
        city: City ID
        event_date: Target event date

    Returns:
        DataFrame with market data, or None if no data
    """
    from sqlalchemy import select
    from src.db.models import KalshiMarket

    query = select(
        KalshiMarket.ticker,
        KalshiMarket.event_date,
        KalshiMarket.strike_type,
        KalshiMarket.floor_strike,
        KalshiMarket.cap_strike,
        KalshiMarket.status,
    ).where(
        KalshiMarket.city == city,
        KalshiMarket.event_date == event_date,
    )
    rows = session.execute(query).fetchall()

    if not rows:
        return None

    return pd.DataFrame(
        rows,
        columns=["ticker", "event_date", "strike_type", "floor_strike", "cap_strike", "status"]
    )


def fetch_markets_from_api(
    client: KalshiClient,
    city: str,
    event_date: date,
) -> Optional[pd.DataFrame]:
    """
    Fetch market data directly from Kalshi API.

    Args:
        client: KalshiClient instance
        city: City ID
        event_date: Target event date

    Returns:
        DataFrame with market data, or None if no data
    """
    city_config = CITIES.get(city)
    if not city_config:
        logger.warning(f"Unknown city: {city}")
        return None

    series_ticker = city_config.series_ticker

    try:
        # Fetch markets from API
        response = client.get_markets(
            series_ticker=series_ticker,
            status="open",
            limit=100,
        )

        markets = response.markets
        if not markets:
            return None

        # Filter to event_date
        rows = []
        for m in markets:
            # Parse event_date from ticker or subtitle
            # Event ticker format: KXHIGHCHI-25NOV27
            if m.event_ticker:
                import re
                match = re.search(r'-(\d{2})([A-Z]{3})(\d{2})$', m.event_ticker)
                if match:
                    year = 2000 + int(match.group(1))
                    month_str = match.group(2)
                    day = int(match.group(3))
                    months = {
                        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                    }
                    month = months.get(month_str)
                    if month:
                        try:
                            m_event_date = date(year, month, day)
                            if m_event_date == event_date:
                                rows.append({
                                    "ticker": m.ticker,
                                    "event_date": m_event_date,
                                    "strike_type": m.strike_type,
                                    "floor_strike": m.floor_strike,
                                    "cap_strike": m.cap_strike,
                                    "status": m.status,
                                })
                        except ValueError:
                            pass

        if not rows:
            return None

        return pd.DataFrame(rows)

    except Exception as e:
        logger.error(f"Failed to fetch markets from API: {e}")
        return None


def place_order(
    client: KalshiClient,
    ticker: str,
    num_contracts: int,
    price_cents: int,
    city: str,
    event_date: date,
) -> Dict[str, Any]:
    """
    Place a maker limit order.

    Args:
        client: KalshiClient instance
        ticker: Contract ticker
        num_contracts: Number of contracts to buy
        price_cents: Price in cents
        city: City ID (for client_order_id)
        event_date: Event date (for client_order_id)

    Returns:
        Order result dict from Kalshi API
    """
    client_order_id = f"mt-{city[:3]}-{event_date.strftime('%m%d')}-{uuid.uuid4().hex[:6]}"

    result = client.create_order(
        ticker=ticker,
        side="yes",
        action="buy",
        count=num_contracts,
        order_type="limit",
        yes_price=price_cents,
        client_order_id=client_order_id,
    )

    return {
        "order_id": result.get("order", {}).get("order_id", "unknown"),
        "client_order_id": client_order_id,
        "result": result,
    }


def log_order_to_db(
    session,
    trade_result: TradeResult,
) -> None:
    """Log order to sim.live_orders table."""
    from sqlalchemy import text

    session.execute(
        text("""
            INSERT INTO sim.live_orders
            (order_id, client_order_id, city, event_date, ticker, side,
             price_cents, num_contracts, amount_usd, placed_at, status, strategy)
            VALUES (:order_id, :client_order_id, :city, :event_date, :ticker, :side,
                    :price_cents, :num_contracts, :amount_usd, :placed_at, :status, :strategy)
        """),
        {
            "order_id": trade_result.order_id or "dry-run",
            "client_order_id": trade_result.client_order_id or f"dry-{trade_result.city[:3]}",
            "city": trade_result.city,
            "event_date": trade_result.event_date,
            "ticker": trade_result.ticker,
            "side": "yes",
            "price_cents": int(trade_result.entry_price_cents),
            "num_contracts": trade_result.num_contracts,
            "amount_usd": trade_result.amount_usd,
            "placed_at": datetime.now(timezone.utc),
            "status": trade_result.status,
            "strategy": "open_maker_base_manual",
        }
    )
    session.commit()


def get_forecast_flexible(
    session,
    city: str,
    event_date: date,
    preferred_basis_offset: int = 1,
) -> Optional[tuple]:
    """
    Get forecast with flexible basis date fallback.

    Tries:
    1. basis_date = event_date - preferred_basis_offset
    2. basis_date = event_date - 2
    3. Most recent basis_date available

    Args:
        session: SQLAlchemy session
        city: City ID
        event_date: Target event date
        preferred_basis_offset: Preferred basis offset in days

    Returns:
        (temp_fcst, basis_date) or None
    """
    from sqlalchemy import select, desc
    from src.db.models import WxForecastSnapshot

    # Try offsets in order of preference
    offsets_to_try = [preferred_basis_offset, 2, 0, 3]

    for offset in offsets_to_try:
        basis_date = event_date - timedelta(days=offset)
        query = select(
            WxForecastSnapshot.tempmax_fcst_f,
            WxForecastSnapshot.basis_date,
        ).where(
            WxForecastSnapshot.city == city,
            WxForecastSnapshot.target_date == event_date,
            WxForecastSnapshot.basis_date == basis_date,
            WxForecastSnapshot.tempmax_fcst_f > 0,  # Filter out zeros
        )
        result = session.execute(query).first()
        if result:
            return float(result[0]), result[1]

    # Final fallback: any available forecast for this target_date
    query = select(
        WxForecastSnapshot.tempmax_fcst_f,
        WxForecastSnapshot.basis_date,
    ).where(
        WxForecastSnapshot.city == city,
        WxForecastSnapshot.target_date == event_date,
        WxForecastSnapshot.tempmax_fcst_f > 0,
    ).order_by(desc(WxForecastSnapshot.basis_date)).limit(1)

    result = session.execute(query).first()
    if result:
        return float(result[0]), result[1]

    return None


def run_manual_trade(
    city: str,
    event_date: date,
    params: OpenMakerParams,
    dry_run: bool = True,
    client: Optional[KalshiClient] = None,
    manual_fcst: Optional[float] = None,
) -> Optional[TradeResult]:
    """
    Execute manual trade logic for one city/date.

    Args:
        city: City ID
        event_date: Event date
        params: Strategy parameters
        dry_run: If True, don't place real orders
        client: KalshiClient for real orders
        manual_fcst: Manual forecast temperature to use instead of DB lookup

    Returns:
        TradeResult with outcome
    """
    with get_db_session() as session:
        # 1. Get forecast (manual override or DB lookup)
        if manual_fcst is not None:
            temp_fcst = manual_fcst
            basis_date = date.today()
            logger.info(f"  {city}: Using manual forecast: {temp_fcst:.1f}F")
        else:
            forecast_result = get_forecast_flexible(
                session, city, event_date, params.basis_offset_days
            )
            if forecast_result is None:
                logger.warning(f"  {city}: No forecast for {event_date}")
                return None
            temp_fcst, basis_date = forecast_result

        # 2. Apply bias
        temp_adjusted = temp_fcst + params.temp_bias_deg

        # 3. Load market data from DB first
        market_df = load_market_data(session, city, event_date)

        # 4. If no data in DB and we have a client, fetch from API
        if (market_df is None or market_df.empty) and client is not None:
            logger.info(f"  {city}: Fetching markets from Kalshi API...")
            market_df = fetch_markets_from_api(client, city, event_date)

        if market_df is None or market_df.empty:
            logger.warning(f"  {city}: No market data for {event_date}")
            return None

        # 4. Find bracket
        bracket = find_bracket_for_temp(market_df, event_date, temp_adjusted)
        if bracket is None:
            logger.warning(f"  {city}: No bracket for temp {temp_adjusted:.1f}F")
            return None

        ticker, floor_strike, cap_strike = bracket

        # Check market status (Kalshi uses "active" for tradeable markets)
        market_row = market_df[market_df["ticker"] == ticker]
        if not market_row.empty:
            status = market_row.iloc[0]["status"]
            if status not in ("open", "active"):
                logger.warning(f"  {city}: Market {ticker} status is '{status}' (not tradeable)")
                # Continue anyway for dry-run

        # 5. Calculate position size
        num_contracts, amount_usd = calculate_position_size(
            params.entry_price_cents,
            params.bet_amount_usd,
        )

        # Format bracket string
        if floor_strike is not None and cap_strike is not None:
            bracket_str = f"[{floor_strike}, {cap_strike})"
        elif floor_strike is not None:
            bracket_str = f">= {floor_strike}"
        elif cap_strike is not None:
            bracket_str = f"< {cap_strike}"
        else:
            bracket_str = "unknown"

        # Log decision info
        print(f"\n{'='*60}")
        print(f"TRADE: {city.upper()}")
        print(f"{'='*60}")
        print(f"  Event Date:    {event_date}")
        print(f"  Ticker:        {ticker}")
        print(f"  Bracket:       {bracket_str}")
        print(f"  Forecast:      {temp_fcst:.1f}F (basis: {basis_date})")
        print(f"  Adjusted:      {temp_adjusted:.1f}F (bias: {params.temp_bias_deg:+.1f}F)")
        print(f"  Entry:         {params.entry_price_cents:.0f}c x {num_contracts} = ${amount_usd:.2f}")

        result = TradeResult(
            city=city,
            event_date=event_date,
            ticker=ticker,
            temp_fcst=temp_fcst,
            temp_adjusted=temp_adjusted,
            floor_strike=floor_strike,
            cap_strike=cap_strike,
            entry_price_cents=params.entry_price_cents,
            num_contracts=num_contracts,
            amount_usd=amount_usd,
        )

        # 6. Place order (or log dry run)
        if dry_run:
            print(f"\n  [DRY RUN] Would place order: BUY {num_contracts} YES @ {params.entry_price_cents:.0f}c")
            result.status = "dry_run"
        else:
            if client is None:
                logger.error("Client required for live orders")
                result.status = "error"
                result.error = "No client"
                return result

            try:
                order_result = place_order(
                    client=client,
                    ticker=ticker,
                    num_contracts=num_contracts,
                    price_cents=int(params.entry_price_cents),
                    city=city,
                    event_date=event_date,
                )
                result.order_id = order_result["order_id"]
                result.client_order_id = order_result["client_order_id"]
                result.status = "submitted"

                print(f"\n  ORDER PLACED!")
                print(f"    Order ID:        {result.order_id}")
                print(f"    Client Order ID: {result.client_order_id}")

                # Log to database
                log_order_to_db(session, result)
                print(f"    Logged to DB:    sim.live_orders")

            except Exception as e:
                logger.error(f"Order placement failed: {e}")
                result.status = "error"
                result.error = str(e)
                print(f"\n  ORDER FAILED: {e}")

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Manual trade script for open-maker strategy"
    )
    parser.add_argument(
        "--city", action="append",
        help="City to trade (can specify multiple)"
    )
    parser.add_argument(
        "--all-cities", action="store_true",
        help="Trade all 6 cities"
    )
    parser.add_argument(
        "--event-date", type=str, required=True,
        help="Event date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--bet-amount", type=float, default=5.0,
        help="Bet amount per city in USD (default: 5)"
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
        help="Use tuned parameters from config/open_maker_base_best_params.json"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview trades without placing orders"
    )
    parser.add_argument(
        "--manual-fcst", type=float, default=None,
        help="Manual forecast temperature to use (bypasses DB lookup)"
    )

    args = parser.parse_args()

    # Parse event date
    event_date = date.fromisoformat(args.event_date)

    # Determine cities
    if args.all_cities:
        cities = list(CITIES.keys())
    elif args.city:
        cities = args.city
    else:
        parser.error("Must specify --city or --all-cities")

    # Build params
    if args.use_tuned:
        params = load_tuned_params("open_maker_base", bet_amount_usd=args.bet_amount)
        if params:
            logger.info(f"Using tuned params: entry={params.entry_price_cents}c, bias={params.temp_bias_deg}F")
        else:
            logger.warning("No tuned params found, using CLI defaults")
            params = OpenMakerParams(
                entry_price_cents=args.price,
                temp_bias_deg=args.bias,
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

    # Initialize Kalshi client (needed for both dry_run market lookup and live trading)
    settings = get_settings()
    client = KalshiClient(
        api_key=settings.kalshi_api_key,
        private_key_path=settings.kalshi_private_key_path,
        base_url=settings.kalshi_base_url,
    )
    logger.info("Kalshi client initialized")

    # Print header
    print("\n" + "=" * 60)
    print("MANUAL TRADE SCRIPT")
    print("=" * 60)
    print(f"Event Date: {event_date}")
    print(f"Cities:     {', '.join(cities)}")
    print(f"Params:     entry={params.entry_price_cents}c, bias={params.temp_bias_deg:+.1f}F")
    print(f"Bet Amount: ${params.bet_amount_usd:.2f} per city")
    print(f"Mode:       {'DRY RUN' if args.dry_run else 'LIVE'}")

    # Run trades
    results = []
    for city in cities:
        result = run_manual_trade(
            city=city,
            event_date=event_date,
            params=params,
            dry_run=args.dry_run,
            client=client,
            manual_fcst=args.manual_fcst,
        )
        if result:
            results.append(result)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_amount = sum(r.amount_usd for r in results)
    submitted = [r for r in results if r.status == "submitted"]
    dry_runs = [r for r in results if r.status == "dry_run"]
    errors = [r for r in results if r.status == "error"]

    print(f"Total trades:  {len(results)}")
    print(f"Total amount:  ${total_amount:.2f}")

    if dry_runs:
        print(f"Dry runs:      {len(dry_runs)}")
    if submitted:
        print(f"Submitted:     {len(submitted)}")
        for r in submitted:
            print(f"  - {r.city}: {r.ticker} @ {r.entry_price_cents}c x {r.num_contracts}")
    if errors:
        print(f"Errors:        {len(errors)}")
        for r in errors:
            print(f"  - {r.city}: {r.error}")

    if not args.dry_run and submitted:
        print("\nOrders logged to: sim.live_orders")
        print("Check Kalshi UI to verify resting bids")

    print("=" * 60)


if __name__ == "__main__":
    main()
