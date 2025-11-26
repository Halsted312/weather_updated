"""
Database loaders with idempotent upsert logic.

All functions can be safely re-run without creating duplicates.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import select

from db.models import Series, Market, Candle, Trade, WeatherObserved, IngestionLog, WxLocation, WxMinuteObs
from sqlalchemy import text
import pandas as pd

logger = logging.getLogger(__name__)


def upsert_series(session: Session, series_data: Dict[str, Any]) -> Series:
    """
    Insert or update series metadata.

    Args:
        session: Database session
        series_data: Series data dict from API

    Returns:
        Series object
    """
    series_ticker = series_data.get("ticker")

    # Check if exists
    stmt = select(Series).where(Series.series_ticker == series_ticker)
    existing = session.execute(stmt).scalar_one_or_none()

    if existing:
        # Update
        existing.title = series_data.get("title", existing.title)
        existing.category = series_data.get("category", existing.category)
        existing.frequency = series_data.get("frequency", existing.frequency)
        existing.settlement_source_json = series_data.get("settlement_sources")
        existing.raw_json = series_data
        existing.updated_at = datetime.utcnow()
        logger.debug(f"Updated series: {series_ticker}")
        return existing
    else:
        # Insert
        series = Series(
            series_ticker=series_ticker,
            title=series_data.get("title"),
            category=series_data.get("category"),
            frequency=series_data.get("frequency"),
            settlement_source_json=series_data.get("settlement_sources"),
            raw_json=series_data,
        )
        session.add(series)
        logger.debug(f"Inserted series: {series_ticker}")
        return series


def upsert_market(session: Session, market_data: Dict[str, Any]) -> Market:
    """
    Insert or update market.

    Args:
        session: Database session
        market_data: Market data dict from API

    Returns:
        Market object
    """
    ticker = market_data.get("ticker")

    # Parse timestamps
    def parse_time(time_str):
        if not time_str:
            return None
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))

    # Check if exists
    stmt = select(Market).where(Market.ticker == ticker)
    existing = session.execute(stmt).scalar_one_or_none()

    market_dict = {
        "ticker": ticker,
        "series_ticker": market_data.get("series_ticker"),
        "event_ticker": market_data.get("event_ticker"),
        "title": market_data.get("title"),
        "subtitle": market_data.get("subtitle"),
        "open_time": parse_time(market_data.get("open_time")),
        "close_time": parse_time(market_data.get("close_time")),
        "expiration_time": parse_time(market_data.get("expiration_time")),
        "status": market_data.get("status"),
        "result": market_data.get("result"),
        "settlement_value": market_data.get("settlement_value"),
        "floor_strike": market_data.get("floor_strike"),
        "cap_strike": market_data.get("cap_strike"),
        "strike_type": market_data.get("strike_type"),
        "last_price": market_data.get("last_price"),
        "yes_bid": market_data.get("yes_bid"),
        "yes_ask": market_data.get("yes_ask"),
        "no_bid": market_data.get("no_bid"),
        "no_ask": market_data.get("no_ask"),
        "volume": market_data.get("volume"),
        "volume_24h": market_data.get("volume_24h"),
        "open_interest": market_data.get("open_interest"),
        "liquidity": market_data.get("liquidity"),
        "rules_primary": market_data.get("rules_primary"),
        "rules_secondary": market_data.get("rules_secondary"),
        "raw_json": market_data,
    }

    if existing:
        # Update
        for key, value in market_dict.items():
            if key != "ticker":  # Don't update primary key
                setattr(existing, key, value)
        existing.updated_at = datetime.utcnow()
        logger.debug(f"Updated market: {ticker}")
        return existing
    else:
        # Insert
        market = Market(**market_dict)
        session.add(market)
        logger.debug(f"Inserted market: {ticker}")
        return market


def bulk_upsert_candles(session: Session, candles: List[Dict[str, Any]]) -> int:
    """
    Bulk insert or update candles using PostgreSQL UPSERT.

    Args:
        session: Database session
        candles: List of candle dicts with keys:
            - market_ticker
            - timestamp
            - period_minutes
            - open, high, low, close
            - volume, num_trades

    Returns:
        Number of candles upserted
    """
    if not candles:
        return 0

    # Prepare data
    candle_records = []
    for candle in candles:
        candle_records.append({
            "market_ticker": candle["market_ticker"],
            "timestamp": candle["timestamp"],
            "period_minutes": candle["period_minutes"],
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"],
            "num_trades": candle.get("num_trades"),
        })

    # PostgreSQL INSERT ... ON CONFLICT DO UPDATE
    stmt = insert(Candle).values(candle_records)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_candle",  # unique constraint name
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
            "num_trades": stmt.excluded.num_trades,
        },
    )

    session.execute(stmt)
    logger.info(f"Upserted {len(candles)} candles")
    return len(candles)


def bulk_upsert_trades(session: Session, trades: List[Dict[str, Any]]) -> int:
    """
    Bulk insert trades (skip duplicates).

    Args:
        session: Database session
        trades: List of trade dicts from API

    Returns:
        Number of trades inserted
    """
    if not trades:
        return 0

    # Parse timestamps
    def parse_time(time_str):
        if not time_str:
            return None
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))

    # Prepare trade records
    trade_records = []
    for trade in trades:
        trade_records.append({
            "trade_id": trade.get("trade_id"),
            "market_ticker": trade.get("ticker"),
            "yes_price": trade.get("yes_price"),
            "no_price": trade.get("no_price"),
            "price": trade.get("price"),
            "count": trade.get("count"),
            "taker_side": trade.get("taker_side"),
            "created_time": parse_time(trade.get("created_time")),
            "raw_json": trade,
        })

    # PostgreSQL INSERT ... ON CONFLICT DO NOTHING (trades are immutable)
    stmt = insert(Trade).values(trade_records)
    stmt = stmt.on_conflict_do_nothing(index_elements=["trade_id"])

    result = session.execute(stmt)
    inserted = result.rowcount if hasattr(result, 'rowcount') else len(trades)

    logger.info(f"Inserted {inserted} new trades (skipped {len(trades) - inserted} duplicates)")
    return inserted


def upsert_weather(session: Session, weather_data: Dict[str, Any]) -> WeatherObserved:
    """
    Insert or update weather observation.

    Args:
        session: Database session
        weather_data: Weather data dict with keys:
            - station_id
            - date
            - tmax_f, tmax_c
            - source
            - raw_json

    Returns:
        WeatherObserved object
    """
    station_id = weather_data.get("station_id")
    date = weather_data.get("date")

    # Check if exists
    stmt = select(WeatherObserved).where(
        WeatherObserved.station_id == station_id,
        WeatherObserved.date == date,
    )
    existing = session.execute(stmt).scalar_one_or_none()

    if existing:
        # Update
        existing.tmax_f = weather_data.get("tmax_f")
        existing.tmax_c = weather_data.get("tmax_c")
        existing.source = weather_data.get("source")
        existing.raw_json = weather_data.get("raw_json")
        existing.updated_at = datetime.utcnow()
        logger.debug(f"Updated weather: {station_id} on {date}")
        return existing
    else:
        # Insert
        weather = WeatherObserved(
            station_id=station_id,
            date=date,
            tmax_f=weather_data.get("tmax_f"),
            tmax_c=weather_data.get("tmax_c"),
            source=weather_data.get("source"),
            raw_json=weather_data.get("raw_json"),
        )
        session.add(weather)
        logger.debug(f"Inserted weather: {station_id} on {date}")
        return weather


def log_ingestion(
    session: Session,
    series_ticker: str,
    markets_fetched: int,
    trades_fetched: int,
    candles_1m: int,
    candles_5m: int,
    min_close_date: datetime = None,
    max_close_date: datetime = None,
    status: str = "success",
    error_message: str = None,
    duration_seconds: float = None,
) -> IngestionLog:
    """
    Log an ingestion run.

    Args:
        session: Database session
        series_ticker: Series processed
        markets_fetched: Number of markets fetched
        trades_fetched: Number of trades fetched
        candles_1m: Number of 1-min candles generated
        candles_5m: Number of 5-min candles generated
        min_close_date: Earliest market close date
        max_close_date: Latest market close date
        status: success, failed, partial
        error_message: Error message if failed
        duration_seconds: How long it took

    Returns:
        IngestionLog object
    """
    log_entry = IngestionLog(
        series_ticker=series_ticker,
        markets_fetched=markets_fetched,
        trades_fetched=trades_fetched,
        candles_1m_generated=candles_1m,
        candles_5m_generated=candles_5m,
        min_close_date=min_close_date,
        max_close_date=max_close_date,
        status=status,
        error_message=error_message,
        duration_seconds=duration_seconds,
    )
    session.add(log_entry)
    logger.info(f"Logged ingestion for {series_ticker}: {status}")
    return log_entry


def upsert_wx_location(session: Session, loc_id: str, vc_key: str, city: str) -> WxLocation:
    """
    Insert or update wx location metadata.

    Args:
        session: Database session
        loc_id: Location ID (e.g., "KMDW")
        vc_key: Visual Crossing location key (e.g., "stn:KMDW")
        city: City name (e.g., "chicago")

    Returns:
        WxLocation object
    """
    # Check if exists
    stmt = select(WxLocation).where(WxLocation.loc_id == loc_id)
    existing = session.execute(stmt).scalar_one_or_none()

    if existing:
        # Update
        existing.vc_key = vc_key
        existing.city = city
        logger.debug(f"Updated wx location: {loc_id}")
        return existing
    else:
        # Insert
        location = WxLocation(
            loc_id=loc_id,
            vc_key=vc_key,
            city=city,
        )
        session.add(location)
        logger.debug(f"Inserted wx location: {loc_id}")
        return location


def bulk_upsert_wx_minutes(session: Session, loc_id: str, df: pd.DataFrame) -> int:
    """
    Bulk insert or update Visual Crossing minute observations.

    Args:
        session: Database session
        loc_id: Location ID (e.g., "KMDW")
        df: DataFrame with columns: ts_utc, temp_f, humidity, dew_f, windspeed_mph,
                                     windgust_mph, pressure_mb, precip_in, preciptype, raw_json,
                                     ffilled (optional, defaults to False)

    Returns:
        Number of rows upserted
    """
    if df.empty:
        return 0

    # Prepare records
    records = []
    for _, row in df.iterrows():
        records.append({
            "loc_id": loc_id,
            "ts_utc": row["ts_utc"],
            "temp_f": row.get("temp_f"),
            "humidity": row.get("humidity"),
            "dew_f": row.get("dew_f"),
            "windspeed_mph": row.get("windspeed_mph"),
            "windgust_mph": row.get("windgust_mph"),
            "pressure_mb": row.get("pressure_mb"),
            "precip_in": row.get("precip_in"),
            "preciptype": row.get("preciptype"),
            "source": "visualcrossing",
            "stations": row.get("stations"),  # Station ID used by VC for this minute
            "ffilled": row.get("ffilled", False),  # Default to FALSE if not provided
            "raw_json": row.get("raw_json"),
        })

    # PostgreSQL INSERT ... ON CONFLICT DO UPDATE
    stmt = insert(WxMinuteObs).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["loc_id", "ts_utc"],  # Composite primary key
        set_={
            "temp_f": stmt.excluded.temp_f,
            "humidity": stmt.excluded.humidity,
            "dew_f": stmt.excluded.dew_f,
            "windspeed_mph": stmt.excluded.windspeed_mph,
            "windgust_mph": stmt.excluded.windgust_mph,
            "pressure_mb": stmt.excluded.pressure_mb,
            "precip_in": stmt.excluded.precip_in,
            "preciptype": stmt.excluded.preciptype,
            "stations": stmt.excluded.stations,  # Update station info on conflict
            "ffilled": stmt.excluded.ffilled,
            "raw_json": stmt.excluded.raw_json,
        },
    )

    session.execute(stmt)
    logger.info(f"Upserted {len(records)} minute observations for {loc_id}")
    return len(records)


def refresh_1m_grid(session: Session) -> None:
    """
    Refresh the wx.minute_obs_1m materialized view (LOCF upsampled grid).

    This should be called after bulk inserting new wx.minute_obs records.

    Args:
        session: Database session
    """
    logger.info("Refreshing wx.minute_obs_1m materialized view...")

    # Use REFRESH MATERIALIZED VIEW CONCURRENTLY (requires unique index)
    # Note: The migration already created the necessary indexes
    try:
        session.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY wx.minute_obs_1m"))
        logger.info("Materialized view refreshed successfully")
    except Exception as e:
        # Fall back to non-concurrent refresh if concurrent fails
        logger.warning(f"Concurrent refresh failed ({e}), trying non-concurrent...")
        session.rollback()  # Rollback failed transaction before retrying
        session.execute(text("REFRESH MATERIALIZED VIEW wx.minute_obs_1m"))
        logger.info("Materialized view refreshed (non-concurrent)")


def upsert_settlement(session: Session, settlement_data: Dict[str, Any]) -> None:
    """
    Insert or update settlement data in columnar wx.settlement table.

    Args:
        session: Database session
        settlement_data: Dict with keys:
            - city: City name (e.g., "chicago")
            - date_local: Local date (datetime.date)
            - tmax_f: Maximum temperature in Fahrenheit (will be cast to smallint)
            - source: "CLI", "CF6", "IEM_CF6", "GHCND", or "VC"
            - is_preliminary: Boolean (True for CF6/IEM_CF6/VC, False for CLI/GHCND)
            - raw_payload: Raw HTML or JSON string

            Optional (for backwards compatibility):
            - icao: ICAO code (ignored, fetched from dim_city)
            - issuedby: NWS office code (ignored, fetched from dim_city)
    """
    import json

    source = settlement_data['source']

    # Map source to column names
    # Note: IEM has inconsistent naming (tmax_iem_cf6 but is_prelim_iem/retrieved_at_iem)
    source_col_map = {
        'CLI': ('cli', 'cli', 'cli'),           # (tmax_suffix, retrieved_suffix, prelim_suffix)
        'CF6': ('cf6', 'cf6', 'cf6'),
        'IEM_CF6': ('iem_cf6', 'iem', 'iem'),   # tmax has _cf6, metadata columns don't
        'GHCND': ('ghcnd', 'ghcnd', 'ghcnd'),
        'VC': ('vc', 'vc', 'vc'),
    }

    if source not in source_col_map:
        raise ValueError(
            f"Unknown source: {source}. Must be one of {list(source_col_map.keys())}"
        )

    tmax_suffix, retrieved_suffix, prelim_suffix = source_col_map[source]
    tmax_col = f"tmax_{tmax_suffix}"
    retrieved_col = f"retrieved_at_{retrieved_suffix}"
    prelim_col = f"is_prelim_{prelim_suffix}"

    # Build JSONB fragment for raw_payloads
    raw_json = json.dumps({source: settlement_data['raw_payload']})

    # Build dynamic SQL for columnar upsert
    stmt = text(f"""
        INSERT INTO wx.settlement (
            city, date_local,
            {tmax_col}, {retrieved_col}, {prelim_col},
            raw_payloads
        )
        VALUES (
            :city, :date_local,
            CAST(:tmax_f AS smallint), now(), :is_preliminary,
            CAST(:raw_payloads AS jsonb)
        )
        ON CONFLICT (city, date_local)
        DO UPDATE SET
            {tmax_col} = EXCLUDED.{tmax_col},
            {retrieved_col} = EXCLUDED.{retrieved_col},
            {prelim_col} = EXCLUDED.{prelim_col},
            raw_payloads = COALESCE(wx.settlement.raw_payloads, '{{}}'::jsonb) || EXCLUDED.raw_payloads
    """)

    session.execute(stmt, {
        'city': settlement_data['city'],
        'date_local': settlement_data['date_local'],
        'tmax_f': settlement_data['tmax_f'],
        'is_preliminary': settlement_data['is_preliminary'],
        'raw_payloads': raw_json,
    })

    logger.debug(
        f"Upserted settlement: {settlement_data['city']} "
        f"{settlement_data['date_local']} ({source} → {tmax_col}={settlement_data['tmax_f']}°F)"
    )
