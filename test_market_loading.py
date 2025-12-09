#!/usr/bin/env python3
"""Test market metadata loading."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.db.connection import get_db_session
from src.db.models import KalshiMarket
from datetime import date, timedelta

def test_load_markets():
    """Test loading active markets from database."""
    enabled_cities = ['chicago', 'austin', 'denver', 'los_angeles', 'miami', 'philadelphia']

    with get_db_session() as session:
        start_date = date.today()
        end_date = start_date + timedelta(days=7)

        print(f"Querying markets: {start_date} to {end_date}")
        print(f"Cities: {enabled_cities}")

        markets = session.query(KalshiMarket).filter(
            KalshiMarket.event_date >= start_date,
            KalshiMarket.event_date <= end_date,
            KalshiMarket.status == 'active',
            KalshiMarket.city.in_(enabled_cities)
        ).all()

        print(f"\nFound {len(markets)} markets")

        # Group by city and date
        by_city_date = {}
        for m in markets:
            key = (m.city, m.event_date)
            if key not in by_city_date:
                by_city_date[key] = []
            by_city_date[key].append(m)

        print(f"\nMarkets by city/date:")
        for (city, evt_date), markets_list in sorted(by_city_date.items()):
            print(f"\n  {city.upper()} {evt_date}: {len(markets_list)} brackets")
            for m in markets_list[:3]:
                print(f"    {m.ticker} - {m.strike_type} [{m.floor_strike}, {m.cap_strike}]")

if __name__ == "__main__":
    test_load_markets()
