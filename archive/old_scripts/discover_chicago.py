#!/usr/bin/env python3
"""
Fetch Kalshi weather market data for Chicago.

Downloads series metadata, markets, and 1-minute candlesticks
for the specified date range and saves to parquet files.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path to import kalshi module
sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi.client import KalshiClient
from kalshi.schemas import Market, Candle
from kalshi.strike_parser import ensure_strike_metadata

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch Kalshi Chicago weather market data"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=100,
        help="Number of days of historical data to fetch (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/raw/chicago",
        help="Output directory for parquet files (default: ./data/raw/chicago)",
    )
    parser.add_argument(
        "--series",
        type=str,
        default="KXHIGHCHI",
        help="Series ticker (default: KXHIGHCHI)",
    )
    return parser.parse_args()


def fetch_series_metadata(client: KalshiClient, series_ticker: str) -> Dict[str, Any]:
    """Fetch and save series metadata."""
    logger.info(f"Fetching series metadata for {series_ticker}...")
    response = client.get_series(series_ticker)
    series = response.get("series", {})
    logger.info(f"Series: {series.get('title', 'N/A')}")
    logger.info(f"Category: {series.get('category', 'N/A')}")
    logger.info(f"Frequency: {series.get('frequency', 'N/A')}")

    if "settlement_sources" in series:
        logger.info(f"Settlement sources: {series['settlement_sources']}")

    return series


def fetch_markets(
    client: KalshiClient,
    series_ticker: str,
    days: int,
) -> List[Dict[str, Any]]:
    """Fetch all markets for the series in the date range."""
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Convert to Unix timestamps
    min_close_ts = int(start_date.timestamp())
    max_close_ts = int(end_date.timestamp())

    logger.info(
        f"Fetching markets from {start_date.date()} to {end_date.date()} "
        f"({days} days)..."
    )

    markets = client.get_all_markets(
        series_ticker=series_ticker,
        status="closed,settled",
        min_close_ts=min_close_ts,
        max_close_ts=max_close_ts,
    )

    markets = [ensure_strike_metadata(dict(m)) for m in markets]

    logger.info(f"Found {len(markets)} markets")

    # Sort by close_time
    markets.sort(key=lambda m: m.get("close_time", 0))

    return markets


def parse_iso_timestamp(iso_str: str) -> int:
    """Convert ISO 8601 timestamp to Unix seconds."""
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(dt.timestamp())


def aggregate_trades_to_candles(
    trades: List[Dict[str, Any]],
    period_minutes: int = 1,
) -> pd.DataFrame:
    """
    Aggregate trades into OHLCV candlesticks.

    Args:
        trades: List of trade dicts with created_time, yes_price, count
        period_minutes: Candle period in minutes (1 or 5)

    Returns:
        DataFrame with OHLCV data per period
    """
    if not trades:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(trades)

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["created_time"])

    # Use yes_price as the main price (in cents)
    df["price"] = df["yes_price"]

    # Sort by timestamp
    df = df.sort_values("timestamp")

    # Create period buckets
    df["period"] = df["timestamp"].dt.floor(f"{period_minutes}min")

    # Aggregate by period
    candles = df.groupby("period").agg(
        {
            "price": ["first", "max", "min", "last"],  # OHLC
            "count": "sum",  # Volume (total contracts traded)
            "trade_id": "count",  # Number of trades
        }
    )

    # Flatten column names
    candles.columns = ["_".join(col).strip() for col in candles.columns.values]
    candles = candles.rename(
        columns={
            "price_first": "open",
            "price_max": "high",
            "price_min": "low",
            "price_last": "close",
            "count_sum": "volume",
            "trade_id_count": "num_trades",
        }
    )

    # Reset index to make period a column
    candles = candles.reset_index()
    candles = candles.rename(columns={"period": "timestamp"})

    # Add period_minutes column
    candles["period_minutes"] = period_minutes

    # Filter out incomplete candles (only keep completed periods)
    # A candle is complete when: candle_start + period_duration < current_time
    from datetime import timezone
    current_time = pd.Timestamp.now(tz=timezone.utc)
    candle_end_time = candles["timestamp"] + pd.Timedelta(minutes=period_minutes)
    candles = candles[candle_end_time < current_time].copy()

    return candles


def create_ohlcv_bars(
    trades: List[Dict[str, Any]],
    market_ticker: str,
) -> Dict[str, pd.DataFrame]:
    """
    Create both 1-minute and 5-minute OHLCV bars from trades.

    Args:
        trades: List of trade dicts
        market_ticker: Market ticker to add to candles

    Returns:
        Dict with keys '1min' and '5min' containing DataFrames
    """
    results = {}

    # Generate 1-minute bars
    candles_1m = aggregate_trades_to_candles(trades, period_minutes=1)
    if not candles_1m.empty:
        candles_1m["market_ticker"] = market_ticker
        results["1min"] = candles_1m

    # Generate 5-minute bars
    candles_5m = aggregate_trades_to_candles(trades, period_minutes=5)
    if not candles_5m.empty:
        candles_5m["market_ticker"] = market_ticker
        results["5min"] = candles_5m

    return results


def fetch_market_trades(
    client: KalshiClient,
    market: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Fetch all trades for a market."""
    ticker = market["ticker"]
    open_time_str = market["open_time"]
    close_time_str = market["close_time"]

    # Convert ISO timestamps to Unix seconds
    open_time = parse_iso_timestamp(open_time_str)
    close_time = parse_iso_timestamp(close_time_str)

    logger.info(f"Fetching trades for {ticker}...")

    try:
        trades = client.get_all_trades(
            ticker=ticker,
            min_ts=open_time,
            max_ts=close_time,
        )

        logger.info(f"  → Got {len(trades)} trades")
        return trades

    except Exception as e:
        logger.error(f"  → Error fetching trades for {ticker}: {e}")
        return []


def _infer_strike_type_from_ticker(ticker: Optional[str]) -> Optional[str]:
    """Best-effort strike_type inference from Kalshi ticker suffix."""

    if not ticker or not isinstance(ticker, str):
        return None

    suffix = ticker.rsplit("-", 1)[-1].lower()

    if suffix.startswith("b"):
        return "between"
    if suffix.startswith("g"):
        return "greater"
    if suffix.startswith("l"):
        return "less"
    return None


def save_to_parquet(
    data: List[Dict[str, Any]],
    output_dir: Path,
    filename: str,
) -> None:
    """Save data to parquet file."""
    if not data:
        logger.warning(f"No data to save for {filename}")
        return

    if filename == "markets.parquet":
        annotated: List[Dict[str, Any]] = []
        inferred_count = 0
        fallback_count = 0

        for raw_row in data:
            row = {k: (None if pd.isna(v) else v) for k, v in dict(raw_row).items()}
            original_type = row.get("strike_type")
            if "strike_type" in row:
                row.pop("strike_type")

            enriched = ensure_strike_metadata(row)
            strike_type = enriched.get("strike_type")

            if not strike_type:
                fallback_type = _infer_strike_type_from_ticker(enriched.get("ticker"))
                if fallback_type:
                    enriched["strike_type"] = fallback_type
                    strike_type = fallback_type
                    fallback_count += 1

            if not original_type and strike_type:
                inferred_count += 1

            if strike_type:
                enriched["strike_type"] = strike_type.lower()
            else:
                enriched["strike_type"] = "unknown"

            annotated.append(enriched)

        missing_unknown = sum(1 for row in annotated if row.get("strike_type") == "unknown")
        if inferred_count or fallback_count:
            logger.info(
                "Annotated strike_type for %d markets (%d via ticker fallback)",
                inferred_count,
                fallback_count,
            )
        if missing_unknown:
            logger.warning(
                "Unable to infer strike_type for %d markets (tagged as 'unknown')",
                missing_unknown,
            )

        data = annotated

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    df = pd.DataFrame(data)
    df.to_parquet(filepath, index=False, engine="pyarrow")

    logger.info(f"Saved {len(data)} rows to {filepath}")


def generate_summary_report(
    series: Dict[str, Any],
    markets: List[Dict[str, Any]],
    all_candles: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate and save summary report."""
    report = []
    report.append("=" * 60)
    report.append("CHICAGO WEATHER MARKETS DATA SUMMARY")
    report.append("=" * 60)
    report.append("")

    # Series info
    report.append(f"Series: {series.get('ticker', 'N/A')}")
    report.append(f"Title: {series.get('title', 'N/A')}")
    report.append(f"Category: {series.get('category', 'N/A')}")
    report.append(f"Frequency: {series.get('frequency', 'N/A')}")
    report.append("")

    # Markets info
    report.append(f"Total Markets: {len(markets)}")

    if markets:
        market_dates = [
            datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).date()
            for m in markets
        ]
        report.append(f"Date Range: {min(market_dates)} to {max(market_dates)}")

        # Count by status
        statuses = {}
        for m in markets:
            status = m.get("status", "unknown")
            statuses[status] = statuses.get(status, 0) + 1

        report.append("Markets by status:")
        for status, count in sorted(statuses.items()):
            report.append(f"  {status}: {count}")

        # Settlement analysis
        settled = [m for m in markets if m.get("result")]
        report.append(f"\nSettled markets: {len(settled)}")

        if settled:
            yes_count = sum(1 for m in settled if m.get("result") == "yes")
            no_count = sum(1 for m in settled if m.get("result") == "no")
            report.append(f"  YES: {yes_count}")
            report.append(f"  NO: {no_count}")

    report.append("")

    # Candlesticks info
    report.append(f"Total 1-minute candles: {len(all_candles):,}")

    if all_candles:
        df = pd.DataFrame(all_candles)
        markets_with_candles = df["market_ticker"].nunique()
        report.append(f"Markets with candlestick data: {markets_with_candles}")

        avg_candles = len(all_candles) / markets_with_candles
        report.append(f"Average candles per market: {avg_candles:.0f}")

        # Data quality checks
        report.append("\nData Quality:")
        for col in ["yes_bid_close", "yes_ask_close", "price_close"]:
            if col in df.columns:
                missing = df[col].isna().sum()
                pct_missing = (missing / len(df)) * 100
                report.append(f"  {col} missing: {missing} ({pct_missing:.1f}%)")

    report.append("")
    report.append("=" * 60)

    # Save report
    report_text = "\n".join(report)
    report_file = output_dir / "summary_report.txt"
    with open(report_file, "w") as f:
        f.write(report_text)

    logger.info(f"Summary report saved to {report_file}")
    print("\n" + report_text)


def main() -> None:
    """Main execution function."""
    args = parse_args()

    # Load environment variables
    load_dotenv()

    api_key = os.getenv("KALSHI_API_KEY")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    base_url = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")

    if not api_key or not private_key_path:
        logger.error("Missing KALSHI_API_KEY or KALSHI_PRIVATE_KEY_PATH in environment")
        sys.exit(1)

    output_dir = Path(args.output)
    logger.info(f"Output directory: {output_dir}")

    # Initialize client
    logger.info("Initializing Kalshi client...")
    client = KalshiClient(
        api_key=api_key,
        private_key_path=private_key_path,
        base_url=base_url,
    )

    # Fetch series metadata
    series = fetch_series_metadata(client, args.series)
    save_to_parquet([series], output_dir, "series.parquet")

    # Fetch markets
    markets = fetch_markets(client, args.series, args.days)
    save_to_parquet(markets, output_dir, "markets.parquet")

    # Fetch trades for each market and aggregate into candles
    all_trades = []
    all_candles_1m = []
    all_candles_5m = []

    for i, market in enumerate(markets, 1):
        ticker = market['ticker']
        logger.info(f"Processing market {i}/{len(markets)}: {ticker}")

        # Fetch trades
        trades = fetch_market_trades(client, market)
        all_trades.extend(trades)

        # Aggregate into 1-min and 5-min bars
        if trades:
            candles_dict = create_ohlcv_bars(trades, ticker)

            if "1min" in candles_dict:
                candles_1m_df = candles_dict["1min"]
                all_candles_1m.append(candles_1m_df)
                logger.info(f"  → Generated {len(candles_1m_df)} 1-minute candles")

            if "5min" in candles_dict:
                candles_5m_df = candles_dict["5min"]
                all_candles_5m.append(candles_5m_df)
                logger.info(f"  → Generated {len(candles_5m_df)} 5-minute candles")

    # Save raw trades
    save_to_parquet(all_trades, output_dir, "trades.parquet")

    # Save aggregated candles
    if all_candles_1m:
        candles_1m_combined = pd.concat(all_candles_1m, ignore_index=True)
        candles_1m_combined.to_parquet(output_dir / "candles_1m.parquet", index=False)
        logger.info(f"Saved {len(candles_1m_combined)} 1-minute candles to candles_1m.parquet")

    if all_candles_5m:
        candles_5m_combined = pd.concat(all_candles_5m, ignore_index=True)
        candles_5m_combined.to_parquet(output_dir / "candles_5m.parquet", index=False)
        logger.info(f"Saved {len(candles_5m_combined)} 5-minute candles to candles_5m.parquet")

    # Convert to list of dicts for summary report
    all_candles = candles_1m_combined.to_dict('records') if all_candles_1m else []

    # Generate summary report
    generate_summary_report(series, markets, all_candles, output_dir)

    logger.info("✓ Data fetch complete!")


if __name__ == "__main__":
    main()
