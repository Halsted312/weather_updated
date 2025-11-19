#!/usr/bin/env python3
"""
Load Kalshi market data from parquet files into PostgreSQL.

Reads series, markets, trades, and candles from discovery output and
loads them into the database using idempotent upsert functions.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import get_session
from db.loaders import (
    upsert_series,
    upsert_market,
    bulk_upsert_candles,
    bulk_upsert_trades,
    log_ingestion,
    refresh_1m_grid,
)
from kalshi.strike_parser import ensure_strike_metadata
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Mapping of old ticker format to new format (Kalshi changed naming convention)
SERIES_TICKER_MAPPING = {
    "HIGHAUS": "KXHIGHAUS",
    "HIGHCHI": "KXHIGHCHI",
    "HIGHDEN": "KXHIGHDEN",
    "HIGHLAX": "KXHIGHLAX",
    "HIGHNY": "KXHIGHNY",
    "HIGHPHIL": "KXHIGHPHIL",
    "HIGHMIA": "KXHIGHMIA",
}


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj):
            return None  # Convert NaN to None for JSON
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def load_city_data(city_dir: Path) -> Dict:
    """Load parquet files for a city."""
    city_name = city_dir.name
    logger.info(f"\n{'='*60}")
    logger.info(f"LOADING DATA FOR {city_name.upper()}")
    logger.info(f"Directory: {city_dir}")
    logger.info(f"{'='*60}\n")

    stats = {
        "city": city_name,
        "series": 0,
        "markets": 0,
        "trades": 0,
        "candles_1m": 0,
        "candles_5m": 0,
    }
    series_df = pd.DataFrame()

    # Load series
    series_file = city_dir / "series.parquet"
    if series_file.exists():
        series_df = pd.read_parquet(series_file)
        logger.info(f"Loaded {len(series_df)} series records")

        with get_session() as session:
            for _, row in series_df.iterrows():
                series_data = convert_numpy_types(row.to_dict())
                upsert_series(session, series_data)
                stats["series"] += 1
            session.commit()
    else:
        logger.warning(f"No series.parquet found in {city_dir}")

    # Load markets
    markets_file = city_dir / "markets.parquet"
    if markets_file.exists():
        markets_df = pd.read_parquet(markets_file)

        def _derive_series_ticker(event_ticker: Any) -> Any:
            """Infer series ticker from event ticker if missing."""
            if not isinstance(event_ticker, str) or "-" not in event_ticker:
                return None
            base = event_ticker.rsplit("-", 1)[0]
            return SERIES_TICKER_MAPPING.get(base, base)

        if "series_ticker" not in markets_df.columns:
            markets_df["series_ticker"] = markets_df["event_ticker"].apply(_derive_series_ticker)
        else:
            missing_mask = markets_df["series_ticker"].isna()
            if missing_mask.any():
                markets_df.loc[missing_mask, "series_ticker"] = markets_df.loc[
                    missing_mask, "event_ticker"
                ].apply(_derive_series_ticker)
        logger.info(f"Loaded {len(markets_df)} market records")

        with get_session() as session:
            for _, row in markets_df.iterrows():
                market_data = convert_numpy_types(row.to_dict())
                # Extract series_ticker from event_ticker (e.g., "KXHIGHCHI-25NOV13" → "KXHIGHCHI")
                if "event_ticker" in market_data and "series_ticker" not in market_data:
                    event_ticker = market_data["event_ticker"]
                    if event_ticker and "-" in event_ticker:
                        extracted_ticker = event_ticker.rsplit("-", 1)[0]
                        # Map old ticker format to new format if needed
                        market_data["series_ticker"] = SERIES_TICKER_MAPPING.get(extracted_ticker, extracted_ticker)
                market_data = ensure_strike_metadata(market_data)
                upsert_market(session, market_data)
                stats["markets"] += 1

                if stats["markets"] % 100 == 0:
                    logger.info(f"  → Upserted {stats['markets']}/{len(markets_df)} markets")
                    session.commit()  # Commit periodically

            session.commit()
    else:
        logger.warning(f"No markets.parquet found in {city_dir}")

    # Load trades
    trades_file = city_dir / "trades.parquet"
    if trades_file.exists():
        trades_df = pd.read_parquet(trades_file)
        if "market_ticker" not in trades_df.columns and "ticker" in trades_df.columns:
            trades_df["market_ticker"] = trades_df["ticker"]
        logger.info(f"Loaded {len(trades_df)} trade records")

        # Convert to list of dicts for bulk upsert
        trades_data = trades_df.to_dict("records")

        # Bulk upsert in chunks
        chunk_size = 5000
        with get_session() as session:
            for i in range(0, len(trades_data), chunk_size):
                chunk = trades_data[i : i + chunk_size]
                rows = bulk_upsert_trades(session, chunk)
                stats["trades"] += rows
                session.commit()

                if stats["trades"] % 10000 == 0:
                    logger.info(f"  → Upserted {stats['trades']}/{len(trades_data)} trades")
    else:
        logger.warning(f"No trades.parquet found in {city_dir}")

    # Load 1-minute candles
    candles_1m_file = city_dir / "candles_1m.parquet"
    if candles_1m_file.exists():
        candles_1m_df = pd.read_parquet(candles_1m_file)
        if "ticker" not in candles_1m_df.columns and "market_ticker" in candles_1m_df.columns:
            candles_1m_df["ticker"] = candles_1m_df["market_ticker"]
        logger.info(f"Loaded {len(candles_1m_df)} 1-minute candle records")

        # Convert to list of dicts
        candles_data = candles_1m_df.to_dict("records")

        # Bulk upsert in chunks
        chunk_size = 5000
        with get_session() as session:
            for i in range(0, len(candles_data), chunk_size):
                chunk = candles_data[i : i + chunk_size]
                rows = bulk_upsert_candles(session, chunk)
                stats["candles_1m"] += rows
                session.commit()

                if stats["candles_1m"] % 10000 == 0:
                    logger.info(f"  → Upserted {stats['candles_1m']}/{len(candles_data)} 1-min candles")
    else:
        logger.warning(f"No candles_1m.parquet found in {city_dir}")

    # Load 5-minute candles
    candles_5m_file = city_dir / "candles_5m.parquet"
    if candles_5m_file.exists():
        candles_5m_df = pd.read_parquet(candles_5m_file)
        if "ticker" not in candles_5m_df.columns and "market_ticker" in candles_5m_df.columns:
            candles_5m_df["ticker"] = candles_5m_df["market_ticker"]
        logger.info(f"Loaded {len(candles_5m_df)} 5-minute candle records")

        # Convert to list of dicts
        candles_data = candles_5m_df.to_dict("records")

        # Bulk upsert in chunks
        chunk_size = 5000
        with get_session() as session:
            for i in range(0, len(candles_data), chunk_size):
                chunk = candles_data[i : i + chunk_size]
                rows = bulk_upsert_candles(session, chunk)
                stats["candles_5m"] += rows
                session.commit()

                if stats["candles_5m"] % 10000 == 0:
                    logger.info(f"  → Upserted {stats['candles_5m']}/{len(candles_data)} 5-min candles")
    else:
        logger.warning(f"No candles_5m.parquet found in {city_dir}")

    # Log ingestion summary
    if "series_ticker" in markets_df.columns and not markets_df.empty:
        series_value = markets_df["series_ticker"].iloc[0]
    elif not series_df.empty and "ticker" in series_df.columns:
        series_value = series_df["ticker"].iloc[0]
    else:
        series_value = city_name

    close_times = None
    if "close_time" in markets_df.columns and not markets_df.empty:
        close_times = pd.to_datetime(markets_df["close_time"])

    with get_session() as session:
        log_ingestion(
            session,
            series_ticker=series_value,
            markets_fetched=len(markets_df),
            trades_fetched=len(trades_df) if trades_file.exists() else 0,
            candles_1m=len(candles_1m_df) if candles_1m_file.exists() else 0,
            candles_5m=len(candles_5m_df) if candles_5m_file.exists() else 0,
            min_close_date=close_times.min() if close_times is not None else None,
            max_close_date=close_times.max() if close_times is not None else None,
            status="success",
        )
        session.commit()

    logger.info(f"\n✓ {city_name.upper()} data loaded successfully!")
    logger.info(f"  Series: {stats['series']}")
    logger.info(f"  Markets: {stats['markets']}")
    logger.info(f"  Trades: {stats['trades']}")
    logger.info(f"  1-min candles: {stats['candles_1m']}")
    logger.info(f"  5-min candles: {stats['candles_5m']}")

    return stats


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Load Kalshi market data from parquet files into PostgreSQL"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing city subdirectories with parquet files",
    )
    parser.add_argument(
        "--cities",
        type=str,
        nargs="+",
        help="Cities to load (default: all subdirectories)",
    )
    parser.add_argument(
        "--refresh-grid",
        action="store_true",
        help="Refresh wx.minute_obs_1m materialized view after loading",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Find city directories
    if args.cities:
        city_dirs = [input_dir / city for city in args.cities if (input_dir / city).exists()]
    else:
        city_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if not city_dirs:
        logger.error(f"No city directories found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(city_dirs)} city directories to load")

    # Load each city
    all_stats = []
    for city_dir in sorted(city_dirs):
        try:
            stats = load_city_data(city_dir)
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"Error loading {city_dir.name}: {e}", exc_info=True)
            continue

    # Refresh materialized view if requested
    if args.refresh_grid:
        logger.info("\n" + "="*60)
        logger.info("REFRESHING wx.minute_obs_1m MATERIALIZED VIEW")
        logger.info("="*60)

        with get_session() as session:
            refresh_1m_grid(session)
            session.commit()

        logger.info("✓ Materialized view refreshed")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("ALL CITIES LOADED!")
    logger.info("="*60)

    total_stats = {
        "series": sum(s["series"] for s in all_stats),
        "markets": sum(s["markets"] for s in all_stats),
        "trades": sum(s["trades"] for s in all_stats),
        "candles_1m": sum(s["candles_1m"] for s in all_stats),
        "candles_5m": sum(s["candles_5m"] for s in all_stats),
    }

    logger.info(f"\nTotal Summary:")
    logger.info(f"  Cities: {len(all_stats)}")
    logger.info(f"  Series: {total_stats['series']}")
    logger.info(f"  Markets: {total_stats['markets']}")
    logger.info(f"  Trades: {total_stats['trades']}")
    logger.info(f"  1-min candles: {total_stats['candles_1m']}")
    logger.info(f"  5-min candles: {total_stats['candles_5m']}")


if __name__ == "__main__":
    main()
