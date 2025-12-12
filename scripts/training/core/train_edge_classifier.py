#!/usr/bin/env python3
"""Train ML Edge Classifier using CatBoost with Optuna tuning.

This script:
1. Loads combined train+test parquet data (700+ days)
2. Runs the ordinal model to generate edge signals
3. Joins with settlement outcomes
4. Trains EdgeClassifier with Optuna hyperparameter tuning

Usage:
    PYTHONPATH=. .venv/bin/python scripts/train_edge_classifier.py
    PYTHONPATH=. .venv/bin/python scripts/train_edge_classifier.py --city chicago --trials 30
    PYTHONPATH=. .venv/bin/python scripts/train_edge_classifier.py --trials 150 --workers 12
"""

import argparse
import hashlib
import json
import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

# Add project root to path (scripts/training/core/ -> 3 levels up)
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# NOTE: DB import moved to functions that need it (for DB-free operation on fast computer)
# from src.db import get_db_session  # Only import when actually needed
from models.training.ordinal_trainer import OrdinalDeltaTrainer
from models.edge.implied_temp import (
    compute_market_implied_temp,
    compute_forecast_implied_temp,
)
from models.edge.detector import detect_edge, EdgeSignal
from models.edge.classifier import EdgeClassifier
from src.trading.fees import (
    taker_fee_total,
    maker_fee_total,
    compute_ev_per_contract,
    find_best_trade,
)
from src.trading.risk import PositionSizer

# Suppress warnings during training
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path, algorithm: str = "md5") -> Optional[str]:
    """Compute hash of a file's contents.

    This is git-sync proof - same content = same hash regardless of timestamps.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (default: md5, fast enough for cache validation)

    Returns:
        Hex digest string, or None if file doesn't exist
    """
    if not file_path.exists():
        return None

    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def should_regenerate_edge_data(
    cache_path: Path,
    ordinal_model_path: Path,
    candle_parquet_path: Path,
    threshold: float,
    sample_rate: int,
    pnl_mode: str,
) -> bool:
    """Check if edge data cache is still valid or needs regeneration.

    Args:
        cache_path: Path to edge_training_data_*.parquet cache file
        ordinal_model_path: Path to ordinal model
        candle_parquet_path: Path to candle parquet
        threshold: Edge detection threshold
        sample_rate: Sampling rate for training
        pnl_mode: P&L mode (realistic/simplified)

    Returns:
        True if cache should be regenerated, False if cache can be reused
    """
    if not cache_path.exists():
        logger.info("⚠️  Regenerating: cached edge data not found")
        return True

    meta_path = cache_path.with_suffix('.meta.json')
    if not meta_path.exists():
        logger.warning("⚠️  Regenerating: cache metadata missing (old cache format)")
        return True

    try:
        with open(meta_path) as f:
            cached_meta = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"⚠️  Regenerating: cannot read cache metadata ({e})")
        return True

    # Check if ordinal model changed (using content hash, not timestamp)
    if ordinal_model_path.exists():
        current_model_hash = compute_file_hash(ordinal_model_path)
        cached_hash = cached_meta.get('ordinal_model_hash') or cached_meta.get('ordinal_model_mtime')  # Fallback for old caches
        # For old caches that used mtime, we need to regenerate once to get hash
        if cached_meta.get('ordinal_model_hash') is None and cached_meta.get('ordinal_model_mtime') is not None:
            logger.info("⚠️  Regenerating: upgrading cache from mtime to hash-based validation")
            return True
        if cached_hash != current_model_hash:
            logger.info("⚠️  Regenerating: ordinal model content changed")
            logger.info(f"   Cached hash: {cached_hash[:16]}..." if cached_hash else "   Cached: None")
            logger.info(f"   Current hash: {current_model_hash[:16]}..." if current_model_hash else "   Current: None")
            return True

    # Check if candle data changed (using content hash, not timestamp)
    if candle_parquet_path and candle_parquet_path.exists():
        current_candle_hash = compute_file_hash(candle_parquet_path)
        cached_candle_hash = cached_meta.get('candles_hash') or cached_meta.get('candles_mtime')  # Fallback
        # For old caches that used mtime, we need to regenerate once to get hash
        if cached_meta.get('candles_hash') is None and cached_meta.get('candles_mtime') is not None:
            logger.info("⚠️  Regenerating: upgrading cache from mtime to hash-based validation")
            return True
        if cached_candle_hash != current_candle_hash:
            logger.info("⚠️  Regenerating: candle data content changed")
            logger.info(f"   Cached hash: {cached_candle_hash[:16]}..." if cached_candle_hash else "   Cached: None")
            logger.info(f"   Current hash: {current_candle_hash[:16]}..." if current_candle_hash else "   Current: None")
            return True

    # NOTE: threshold is NOT checked - it's just a filter, not fundamental to data
    # We can train with different thresholds on the same edge data
    # if cached_meta.get('threshold') != threshold:
    #     logger.info(f"⚠️  Regenerating: threshold changed ({cached_meta.get('threshold')} → {threshold})")
    #     return True

    if cached_meta.get('sample_rate') != sample_rate:
        logger.info(f"⚠️  Regenerating: sample_rate changed ({cached_meta.get('sample_rate')} → {sample_rate})")
        return True

    if cached_meta.get('pnl_mode') != pnl_mode:
        logger.info(f"⚠️  Regenerating: pnl_mode changed ({cached_meta.get('pnl_mode')} → {pnl_mode})")
        return True

    # Cache is valid
    logger.info(f"✅ Using cached edge data: {cache_path}")
    logger.info(f"   Generated: {cached_meta.get('generated_at', 'unknown')}")
    logger.info(f"   Generation threshold: {cached_meta.get('threshold')}°F (will filter to {threshold}°F)")
    logger.info(f"   Ordinal model: {cached_meta.get('ordinal_model_path')}")
    logger.info(f"   Sample rate: {sample_rate}, P&L mode: {pnl_mode}")
    return False


def save_edge_data_metadata(
    cache_path: Path,
    city: str,
    ordinal_model_path: Path,
    candle_parquet_path: Optional[Path],
    threshold: float,
    sample_rate: int,
    pnl_mode: str,
):
    """Save metadata alongside edge training data cache.

    This enables smart cache invalidation - we can reuse cache when inputs haven't changed.
    Uses content hashes (not timestamps) for git-sync robustness.
    """
    meta = {
        'city': city,
        'ordinal_model_path': str(ordinal_model_path),
        'ordinal_model_hash': compute_file_hash(ordinal_model_path) if ordinal_model_path.exists() else None,
        'candles_parquet_path': str(candle_parquet_path) if candle_parquet_path else None,
        'candles_hash': compute_file_hash(candle_parquet_path) if (candle_parquet_path and candle_parquet_path.exists()) else None,
        'threshold': threshold,
        'sample_rate': sample_rate,
        'pnl_mode': pnl_mode,
        'generated_at': datetime.now().isoformat(),
    }

    meta_path = cache_path.with_suffix('.meta.json')
    try:
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved cache metadata: {meta_path}")
    except Exception as e:
        logger.warning(f"Could not save cache metadata: {e}")


# City configurations
CITY_CONFIG = {
    "chicago": {"ticker_prefix": "KXHIGHCHI", "tz": "America/Chicago"},
    "austin": {"ticker_prefix": "KXHIGHAUS", "tz": "America/Chicago"},
    "denver": {"ticker_prefix": "KXHIGHDEN", "tz": "America/Denver"},
    "los_angeles": {"ticker_prefix": "KXHIGHLA", "tz": "America/Los_Angeles"},
    "miami": {"ticker_prefix": "KXHIGHMIA", "tz": "America/New_York"},
    "philadelphia": {"ticker_prefix": "KXHIGHPHI", "tz": "America/New_York"},
}


def parse_bracket_label(label: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse bracket label to get floor and cap strikes.

    Args:
        label: Bracket label like "82-83", "<80", ">90"

    Returns:
        (floor, cap) tuple. None means unbounded.
    """
    if label.startswith("<"):
        # Below bracket: <80 means floor=None, cap=80
        cap = float(label[1:])
        return (None, cap)
    elif label.startswith(">"):
        # Above bracket: >90 means floor=90, cap=None
        floor = float(label[1:])
        return (floor, None)
    elif "-" in label:
        # Range bracket: 82-83 means floor=82, cap=83
        parts = label.split("-")
        floor = float(parts[0])
        cap = float(parts[1])
        return (floor, cap)
    else:
        # Unknown format
        return (None, None)


def is_settlement_in_bracket(settlement: float, floor: Optional[float], cap: Optional[float]) -> bool:
    """Check if settlement temperature falls in bracket.

    Kalshi brackets are typically "floor <= T < cap" for middle brackets.
    Edge brackets: "<X" means T < X, ">X" means T >= X.

    Args:
        settlement: Settlement temperature
        floor: Floor strike (None = no lower bound)
        cap: Cap strike (None = no upper bound)

    Returns:
        True if settlement is in bracket
    """
    if floor is None and cap is None:
        return False

    if floor is None:
        # Below bracket: T < cap
        return settlement < cap
    elif cap is None:
        # Above bracket: T >= floor
        return settlement >= floor
    else:
        # Range bracket: floor <= T < cap
        return floor <= settlement < cap


def select_best_bracket_for_trade(
    signal: EdgeSignal,
    forecast_implied: float,
    market_implied: float,
    bracket_candles: dict,
    model_probs: Optional[dict] = None,
    maker_fill_prob: float = 0.4,
    min_ev_cents: float = 3.0,
) -> Optional[dict]:
    """Select the best bracket to trade based on edge signal and EV.

    Uses find_best_trade() to evaluate all brackets and select the one
    with highest expected value, supporting both YES and NO bets.

    Args:
        signal: Edge signal (BUY_HIGH or BUY_LOW)
        forecast_implied: Forecast-implied temperature
        market_implied: Market-implied temperature
        bracket_candles: Dict of bracket_label -> candle DataFrame
        model_probs: Optional dict of bracket_label -> model probability
        maker_fill_prob: Probability maker order fills (0-1)
        min_ev_cents: Minimum EV to consider trade

    Returns:
        Dict with best trade info, or None if no good trade found
    """
    if signal == EdgeSignal.NO_TRADE:
        return None

    candidates = []

    for label, candle_df in bracket_candles.items():
        if candle_df.empty:
            continue

        # Get latest bid/ask
        latest = candle_df.iloc[-1]
        yes_bid = int(latest.get("yes_bid_close", 0) or 0)
        yes_ask = int(latest.get("yes_ask_close", 100) or 100)

        # Skip invalid quotes
        if yes_bid <= 0 or yes_ask >= 100 or yes_ask <= yes_bid:
            continue

        # Parse bracket
        floor, cap = parse_bracket_label(label)
        if floor is None and cap is None:
            continue

        # Estimate model probability for this bracket
        # Simple heuristic: distance from forecast_implied to bracket center
        if floor is None:
            bracket_center = cap - 2  # Below bracket
        elif cap is None:
            bracket_center = floor + 2  # Above bracket
        else:
            bracket_center = (floor + cap) / 2

        # Convert distance to probability estimate
        # Closer to forecast = higher probability
        distance = abs(forecast_implied - bracket_center)

        # Rough probability: exponential decay from forecast
        # P ~ exp(-distance / scale), normalized
        scale = 3.0  # Degrees F
        raw_prob = np.exp(-distance / scale)

        # Adjust based on signal direction
        if signal == EdgeSignal.BUY_HIGH:
            # Forecast is higher than market - we expect temp to be higher
            # Increase prob for brackets above forecast_implied
            if bracket_center > forecast_implied:
                raw_prob *= 0.3  # Lower prob for brackets above forecast
            # Market is underestimating - good to buy YES on forecast's bracket
        else:  # BUY_LOW
            # Forecast is lower than market - we expect temp to be lower
            if bracket_center < forecast_implied:
                raw_prob *= 0.3

        # Normalize to reasonable range [0.1, 0.9]
        model_prob = min(0.9, max(0.1, raw_prob))

        # Use model_probs if provided
        if model_probs and label in model_probs:
            model_prob = model_probs[label]

        # Find best trade for this bracket
        side, action, price, ev, role = find_best_trade(
            model_prob=model_prob,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            min_ev_cents=min_ev_cents,
            maker_fill_prob=maker_fill_prob,
        )

        if side is not None and ev >= min_ev_cents:
            candidates.append({
                "label": label,
                "floor": floor,
                "cap": cap,
                "side": side,
                "action": action,
                "price": price,
                "ev_cents": ev,
                "role": role,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "model_prob": model_prob,
            })

    if not candidates:
        return None

    # Return bracket with highest EV
    best = max(candidates, key=lambda x: x["ev_cents"])
    return best


def calculate_realistic_pnl(
    trade: dict,
    settlement: float,
    num_contracts: int = 1,
) -> dict:
    """Calculate realistic P&L for a trade.

    Args:
        trade: Trade dict from select_best_bracket_for_trade()
        settlement: Settlement temperature
        num_contracts: Number of contracts

    Returns:
        Dict with P&L details
    """
    floor = trade["floor"]
    cap = trade["cap"]
    price = trade["price"]
    side = trade["side"]
    action = trade["action"]
    role = trade["role"]

    # Check if bracket won
    bracket_won = is_settlement_in_bracket(settlement, floor, cap)

    # Calculate fee
    if role == "maker":
        fee_cents = maker_fee_total(price, num_contracts)
    else:  # taker
        fee_cents = taker_fee_total(price, num_contracts)
    fee_usd = fee_cents / 100.0

    # Calculate P&L based on side and action
    price_usd = price / 100.0

    if side == "yes" and action == "buy":
        # Long YES: win if bracket_won
        if bracket_won:
            pnl_gross = num_contracts * (1.0 - price_usd)  # Receive $1, paid price
        else:
            pnl_gross = -num_contracts * price_usd  # Contract worthless

    elif side == "yes" and action == "sell":
        # Short YES (= Long NO): win if NOT bracket_won
        if not bracket_won:
            pnl_gross = num_contracts * price_usd  # Keep premium, no payout
        else:
            pnl_gross = -num_contracts * (1.0 - price_usd)  # Pay $1, received price

    elif side == "no" and action == "buy":
        # Long NO: win if NOT bracket_won
        if not bracket_won:
            pnl_gross = num_contracts * (1.0 - price_usd)
        else:
            pnl_gross = -num_contracts * price_usd

    else:  # side == "no" and action == "sell"
        # Short NO: win if bracket_won
        if bracket_won:
            pnl_gross = num_contracts * price_usd
        else:
            pnl_gross = -num_contracts * (1.0 - price_usd)

    pnl_net = pnl_gross - fee_usd

    # Determine if we "won" the trade
    trade_won = pnl_net > 0

    return {
        "pnl_gross": pnl_gross,
        "pnl_net": pnl_net,
        "fee_usd": fee_usd,
        "bracket_won": bracket_won,
        "trade_won": trade_won,
        "entry_price_cents": price,
        "role": role,
        "side": side,
        "action": action,
    }


def load_combined_data(city: str) -> pd.DataFrame:
    """Load and combine train + test parquet files.

    Args:
        city: City name (e.g., 'chicago')

    Returns:
        Combined DataFrame with all data
    """
    base_path = Path(f"models/saved/{city}")

    train_path = base_path / "train_data_full.parquet"
    test_path = base_path / "test_data_full.parquet"

    dfs = []

    if train_path.exists():
        df_train = pd.read_parquet(train_path)
        logger.info(f"Loaded train data: {len(df_train):,} rows")
        dfs.append(df_train)
    else:
        logger.warning(f"Train data not found: {train_path}")

    if test_path.exists():
        df_test = pd.read_parquet(test_path)
        logger.info(f"Loaded test data: {len(df_test):,} rows")
        dfs.append(df_test)
    else:
        logger.warning(f"Test data not found: {test_path}")

    if not dfs:
        raise FileNotFoundError(f"No data found for {city}")

    df_combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined data: {len(df_combined):,} rows")

    # Sort by day and time
    if "day" in df_combined.columns and "cutoff_time" in df_combined.columns:
        df_combined = df_combined.sort_values(["day", "cutoff_time"]).reset_index(
            drop=True
        )

    return df_combined


def load_bracket_candles_for_event(
    session,
    city: str,
    event_date: date,
    snapshot_time,
) -> dict[str, pd.DataFrame]:
    """Load bracket candles for all brackets of an event up to snapshot time.

    Handles both old format (HIGHAUS-) and new format (KXHIGHAUS-) tickers.
    """
    config = CITY_CONFIG.get(city)
    if not config:
        return {}

    prefix = config["ticker_prefix"]
    event_suffix = event_date.strftime("%y%b%d").upper()

    # Query for BOTH old and new ticker formats
    # Old: HIGHAUS-23AUG01-...
    # New: KXHIGHAUS-24DEC01-...
    old_prefix = prefix.replace("KXHIGH", "HIGH")  # Remove "KX"

    query = text("""
        SELECT
            ticker,
            bucket_start,
            yes_bid_close,
            yes_ask_close,
            volume,
            open_interest
        FROM kalshi.candles_1m_dense
        WHERE (ticker LIKE :new_pattern OR ticker LIKE :old_pattern)
          AND bucket_start <= :snapshot_time
        ORDER BY ticker, bucket_start
    """)

    result = session.execute(
        query,
        {
            "new_pattern": f"{prefix}-{event_suffix}%",
            "old_pattern": f"{old_prefix}-{event_suffix}%",
            "snapshot_time": snapshot_time,
        },
    )
    rows = result.fetchall()

    if not rows:
        return {}

    df = pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "bucket_start",
            "yes_bid_close",
            "yes_ask_close",
            "volume",
            "open_interest",
        ],
    )

    bracket_candles = {}
    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker].copy()
        parts = ticker.split("-")
        if len(parts) >= 3:
            bracket_part = parts[-1]
            if bracket_part.startswith("T"):
                strike = bracket_part[1:].replace("LO", "").replace("HI", "")
                try:
                    strike_val = int(strike)
                    if "LO" in bracket_part:
                        label = f"<{strike_val}"
                    elif "HI" in bracket_part:
                        label = f">{strike_val}"
                    else:
                        label = f"{strike_val}-{strike_val + 1}"
                except ValueError:
                    label = bracket_part
            elif bracket_part.startswith("B"):
                strike = bracket_part[1:]
                try:
                    strike_val = float(strike)  # Handle decimal strikes (e.g., 86.5)
                    label = f"{strike_val:g}-{strike_val + 1:g}"  # Format: "86.5-87.5" or "86-87"
                except ValueError:
                    label = bracket_part
            else:
                label = bracket_part
            bracket_candles[label] = ticker_df

    return bracket_candles


def get_settlement_temp(session, city: str, event_date: date) -> Optional[float]:
    """Get settlement temperature for an event."""
    query = text("""
        SELECT tmax_final
        FROM wx.settlement
        WHERE city = :city AND date_local = :event_date
    """)
    result = session.execute(query, {"city": city, "event_date": event_date})
    row = result.fetchone()
    return float(row[0]) if row else None


def load_all_settlements(city: str, dates: list, settlements_parquet: Optional[Path] = None) -> dict:
    """Batch load all settlements (from parquet or DB).

    Args:
        city: City name
        dates: List of dates to load
        settlements_parquet: Optional path to settlements parquet file (bypasses DB)

    Returns:
        Dictionary mapping date -> settlement temperature
    """
    if not dates:
        return {}

    # Use parquet if provided
    if settlements_parquet is not None:
        logger.info(f"Loading settlements from parquet: {settlements_parquet}")
        if not settlements_parquet.exists():
            logger.error(f"Settlements parquet not found: {settlements_parquet}")
            return {}

        df = pd.read_parquet(settlements_parquet)
        # Normalize date column to date objects
        df['date_local'] = pd.to_datetime(df['date_local']).dt.date
        date_set = set(dates)
        df_filtered = df[df['date_local'].isin(date_set)]
        # Filter out null tmax_final
        df_filtered = df_filtered[df_filtered['tmax_final'].notna()]
        return {row.date_local: float(row.tmax_final) for _, row in df_filtered.iterrows()}

    # Otherwise use DB
    from src.db import get_db_session  # Lazy import (only when DB actually needed)
    with get_db_session() as session:
        # Build date list for SQL
        date_strs = [f"'{d}'" for d in dates]
        query = text(f"""
            SELECT date_local, tmax_final
            FROM wx.settlement
            WHERE city = :city AND date_local IN ({','.join(date_strs)})
        """)
        result = session.execute(query, {"city": city})
        return {row[0]: float(row[1]) for row in result.fetchall()}


def load_all_candles_batch(city: str, dates: list) -> dict:
    """Batch load ALL bracket candles for ALL days in ONE query.

    Args:
        city: City name
        dates: List of dates to load candles for

    Returns:
        Dictionary mapping (day, label) -> DataFrame of candles
    """
    if not dates:
        return {}

    config = CITY_CONFIG.get(city)
    if not config:
        return {}

    prefix = config["ticker_prefix"]
    old_prefix = prefix.replace("KXHIGH", "HIGH")  # Remove "KX" for old format

    # Build date suffix mapping
    date_to_suffix = {}
    for d in dates:
        suffix = d.strftime("%y%b%d").upper()
        date_to_suffix[suffix] = d

    logger.info(f"Loading candles for {len(dates)} days (checking both old and new ticker formats)...")

    # CRITICAL FIX: Batch queries to avoid PostgreSQL query size limits
    # With 1062 days × 2 formats = 2124 patterns = 42KB query (too large!)
    # Process in chunks of 200 days = 400 patterns per query
    BATCH_SIZE = 200
    all_rows = []

    from src.db import get_db_session  # Lazy import (only when DB actually needed)
    with get_db_session() as session:
        for batch_start in range(0, len(dates), BATCH_SIZE):
            batch_dates = dates[batch_start:batch_start + BATCH_SIZE]
            ticker_patterns = []

            for d in batch_dates:
                suffix = d.strftime("%y%b%d").upper()
                # Add both formats
                ticker_patterns.append(f"{prefix}-{suffix}%")
                ticker_patterns.append(f"{old_prefix}-{suffix}%")

            patterns_sql = ",".join([f"'{p}'" for p in ticker_patterns])
            query = text(f"""
                SELECT
                    ticker,
                    bucket_start,
                    yes_bid_close,
                    yes_ask_close,
                    volume,
                    open_interest
                FROM kalshi.candles_1m_dense
                WHERE ticker LIKE ANY (ARRAY[{patterns_sql}])
                ORDER BY ticker, bucket_start
            """)

            result = session.execute(query)
            batch_rows = result.fetchall()
            all_rows.extend(batch_rows)

            if batch_rows:
                logger.info(f"  Batch {batch_start//BATCH_SIZE + 1}: Loaded {len(batch_rows):,} candles")

    if not all_rows:
        logger.error("No candles loaded from database!")
        return {}

    logger.info(f"Total loaded: {len(all_rows):,} candle rows")
    rows = all_rows

    df = pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "bucket_start",
            "yes_bid_close",
            "yes_ask_close",
            "volume",
            "open_interest",
        ],
    )

    # Organize by (day, label) for fast lookup
    candle_cache = {}
    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker].copy()
        parts = ticker.split("-")
        if len(parts) >= 3:
            # Extract date suffix and bracket label
            date_suffix = parts[1]  # e.g., "24NOV15"
            day = date_to_suffix.get(date_suffix)
            if day is None:
                continue

            bracket_part = parts[-1]
            if bracket_part.startswith("T"):
                strike = bracket_part[1:].replace("LO", "").replace("HI", "")
                try:
                    strike_val = int(strike)
                    if "LO" in bracket_part:
                        label = f"<{strike_val}"
                    elif "HI" in bracket_part:
                        label = f">{strike_val}"
                    else:
                        label = f"{strike_val}-{strike_val + 1}"
                except ValueError:
                    label = bracket_part
            elif bracket_part.startswith("B"):
                strike = bracket_part[1:]
                try:
                    strike_val = float(strike)  # Handle decimal strikes (e.g., 86.5)
                    label = f"{strike_val:g}-{strike_val + 1:g}"  # Format: "86.5-87.5" or "86-87"
                except ValueError:
                    label = bracket_part
            else:
                label = bracket_part

            candle_cache[(day, label)] = ticker_df

    logger.info(f"✅ Organized into {len(candle_cache):,} (day, bracket) entries from parquet")
    logger.info(f"Sample cache keys: {list(candle_cache.keys())[:5]}")
    return candle_cache


def load_all_candles_from_parquet(city: str, parquet_path: Path, dates: list) -> dict:
    """Load candle data from a parquet file instead of DB.

    This is much faster than DB queries and allows training without DB access.
    Use scripts/export_kalshi_candles.py to create the parquet file.

    Args:
        city: City name
        parquet_path: Path to parquet file with candle data
        dates: List of dates to load candles for

    Returns:
        Dictionary mapping (day, label) -> DataFrame of candles
    """
    if not dates:
        return {}

    logger.info(f"Loading candles from parquet: {parquet_path}")

    if not parquet_path.exists():
        logger.error(f"Parquet file not found: {parquet_path}")
        return {}

    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df):,} candle rows from parquet")

    config = CITY_CONFIG.get(city)
    if not config:
        return {}

    prefix = config["ticker_prefix"]
    old_prefix = prefix.replace("KXHIGH", "HIGH")

    # Build date suffix mapping
    date_to_suffix = {}
    valid_suffixes = set()
    for d in dates:
        suffix = d.strftime("%y%b%d").upper()
        date_to_suffix[suffix] = d
        valid_suffixes.add(suffix)

    # Filter to relevant tickers (matching city prefix and dates)
    def extract_suffix(ticker):
        parts = ticker.split("-")
        return parts[1] if len(parts) >= 2 else None

    df["date_suffix"] = df["ticker"].apply(extract_suffix)
    df = df[df["date_suffix"].isin(valid_suffixes)].copy()
    logger.info(f"Filtered to {len(df):,} rows for requested dates")

    # Next: Organize into (day, bracket) cache - this can take 10-20 min for large datasets
    logger.info(f"Organizing {len(df):,} candle rows by (day, bracket)...")
    logger.info(f"  (This may take 10-20 minutes for 10M+ rows - please wait)")

    # Rename columns to match DB schema expected by rest of code
    # Export uses yes_bid/yes_ask, DB query uses yes_bid_close/yes_ask_close
    column_mapping = {
        "yes_bid": "yes_bid_close",
        "yes_ask": "yes_ask_close",
    }
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    # Organize by (day, label) for fast lookup
    candle_cache = {}
    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker].copy()
        parts = ticker.split("-")
        if len(parts) >= 3:
            date_suffix = parts[1]
            day = date_to_suffix.get(date_suffix)
            if day is None:
                continue

            bracket_part = parts[-1]
            if bracket_part.startswith("T"):
                strike = bracket_part[1:].replace("LO", "").replace("HI", "")
                try:
                    strike_val = int(strike)
                    if "LO" in bracket_part:
                        label = f"<{strike_val}"
                    elif "HI" in bracket_part:
                        label = f">{strike_val}"
                    else:
                        label = f"{strike_val}-{strike_val + 1}"
                except ValueError:
                    label = bracket_part
            elif bracket_part.startswith("B"):
                strike = bracket_part[1:]
                try:
                    strike_val = float(strike)
                    label = f"{strike_val:g}-{strike_val + 1:g}"
                except ValueError:
                    label = bracket_part
            else:
                label = bracket_part

            candle_cache[(day, label)] = ticker_df

    logger.info(f"Organized into {len(candle_cache):,} (day, bracket) entries from parquet")
    return candle_cache


def get_candles_from_cache(
    candle_cache: dict, day: date, snapshot_time, city: str
) -> dict[str, pd.DataFrame]:
    """Get bracket candles for a specific day/time from pre-loaded cache.

    Args:
        candle_cache: Pre-loaded cache of (day, bracket) -> DataFrame
        day: Event date
        snapshot_time: Snapshot timestamp (naive local time)
        city: City name (for timezone lookup)

    Returns:
        Dict of bracket_label -> DataFrame of candles up to snapshot_time
    """
    bracket_candles = {}

    # CRITICAL: snapshot_time is in LOCAL time (naive), candles are in UTC
    # Need to convert snapshot to UTC for comparison
    snapshot_ts = pd.Timestamp(snapshot_time)

    # Get city-specific timezone from CITY_CONFIG
    city_tz = CITY_CONFIG.get(city, {}).get("tz", "America/Chicago")

    # If snapshot is naive, localize to city timezone and convert to UTC
    if snapshot_ts.tz is None:
        try:
            # Localize to city timezone (handle DST transitions)
            # ambiguous='NaT': Skip ambiguous times during fall-back (~2-4 hours/year)
            # nonexistent='shift_forward': For spring-forward, shift to next valid time
            snapshot_ts = snapshot_ts.tz_localize(city_tz, ambiguous='NaT', nonexistent='shift_forward')

            # Check if resulted in NaT (ambiguous time)
            if pd.isna(snapshot_ts):
                return {}

            # Convert to UTC for comparison with candles
            snapshot_ts = snapshot_ts.tz_convert("UTC")
        except Exception:
            # DST transition edge case - return empty
            return {}

    # Make naive for comparison (both are now in UTC)
    snapshot_ts = snapshot_ts.tz_convert(None) if snapshot_ts.tz else snapshot_ts

    # Find all entries for this day
    for (d, label), df in candle_cache.items():
        if d != day:
            continue
        # Filter to candles up to snapshot_time (handle timezone)
        bucket_start = df["bucket_start"]
        if bucket_start.dt.tz is not None:
            bucket_start = bucket_start.dt.tz_convert(None)
        filtered = df[bucket_start <= snapshot_ts].copy()
        if not filtered.empty:
            bracket_candles[label] = filtered

    return bracket_candles


# Multiple maker_fill_prob values for Optuna tuning
MULTI_MFP_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def _process_single_day(
    day: date,
    day_df: pd.DataFrame,
    model: OrdinalDeltaTrainer,
    candle_cache: dict,
    settlement: float,
    edge_threshold: float,
    sample_rate: int,
    city: str,
    maker_fill_prob: float = 0.4,
    use_realistic_pnl: bool = True,
    multi_mfp_mode: bool = False,
) -> list:
    """Process a single day's edge data - runs in thread with shared memory.

    Args:
        day: The date to process
        day_df: Feature DataFrame for this day
        model: Pre-loaded ordinal model (shared)
        candle_cache: Pre-loaded candle cache (shared)
        settlement: Settlement temperature
        edge_threshold: Threshold for edge detection
        sample_rate: Sample every Nth snapshot
        city: City name for timezone lookup
        maker_fill_prob: Probability maker order fills (0-1) - used if multi_mfp_mode=False
        use_realistic_pnl: Use realistic P&L with fees (True) or simplified binary (False)
        multi_mfp_mode: If True, compute P&L for multiple maker_fill_prob values
                       (columns: pnl_mfp_20, pnl_mfp_30, etc.) for Optuna tuning

    Returns:
        List of result dictionaries for this day
    """
    if day_df.empty or settlement is None:
        return []

    results = []
    unique_times = day_df["cutoff_time"].unique()
    sampled_times = unique_times[::sample_rate]

    for snapshot_time in sampled_times:
        snapshot_df = day_df[day_df["cutoff_time"] == snapshot_time]
        if snapshot_df.empty:
            continue

        # Get base temp
        base_temp = snapshot_df["t_forecast_base"].iloc[0]
        if pd.isna(base_temp):
            base_temp = snapshot_df["fcst_prev_max_f"].iloc[0]
        if pd.isna(base_temp):
            continue

        # Get model prediction
        try:
            delta_probs = model.predict_proba(snapshot_df)
            forecast_result = compute_forecast_implied_temp(
                delta_probs=delta_probs[0],
                base_temp=base_temp,
            )
        except Exception:
            continue

        # Get market-implied temp from cache (NO DB QUERY!)
        bracket_candles = get_candles_from_cache(candle_cache, day, snapshot_time, city)
        if not bracket_candles:
            # DEBUG: This is likely where it's failing
            # logger.warning(f"No bracket candles for {day} at {snapshot_time}")
            continue

        market_result = compute_market_implied_temp(
            bracket_candles=bracket_candles,
            snapshot_time=snapshot_time,
        )
        if not market_result.valid:
            # DEBUG: Market implied temp calculation failed
            # logger.warning(f"Invalid market result for {day} at {snapshot_time}")
            continue

        # Detect edge
        edge_result = detect_edge(
            forecast_implied=forecast_result.implied_temp,
            market_implied=market_result.implied_temp,
            forecast_uncertainty=forecast_result.uncertainty,
            market_uncertainty=market_result.uncertainty,
            threshold=edge_threshold,
        )

        # Compute P&L - REALISTIC with fees or simplified binary
        pnl = None
        pnl_gross = None
        fee_usd = 0.0
        entry_price_cents = None
        trade_role = None
        trade_side = None
        trade_action = None
        target_bracket_label = None
        bracket_won = None
        trade_won = None
        ev_cents = None

        # Multi-MFP mode: compute P&L for multiple maker_fill_prob values
        multi_mfp_pnl = {}  # Will store pnl_mfp_20, pnl_mfp_30, etc.

        if edge_result.signal != EdgeSignal.NO_TRADE:
            if use_realistic_pnl:
                if multi_mfp_mode:
                    # MULTI-MFP MODE: Compute P&L for each maker_fill_prob value
                    for mfp_val in MULTI_MFP_VALUES:
                        mfp_key = f"pnl_mfp_{int(mfp_val * 100):02d}"
                        best_trade_mfp = select_best_bracket_for_trade(
                            signal=edge_result.signal,
                            forecast_implied=forecast_result.implied_temp,
                            market_implied=market_result.implied_temp,
                            bracket_candles=bracket_candles,
                            model_probs=None,
                            maker_fill_prob=mfp_val,
                            min_ev_cents=2.0,
                        )
                        if best_trade_mfp is not None:
                            pnl_result_mfp = calculate_realistic_pnl(
                                trade=best_trade_mfp,
                                settlement=settlement,
                                num_contracts=1,
                            )
                            multi_mfp_pnl[mfp_key] = pnl_result_mfp["pnl_net"]
                        else:
                            multi_mfp_pnl[mfp_key] = None

                    # Use default maker_fill_prob for the main "pnl" column
                    default_mfp_key = f"pnl_mfp_{int(maker_fill_prob * 100):02d}"
                    pnl = multi_mfp_pnl.get(default_mfp_key)

                    # Also compute primary trade details using the default mfp
                    best_trade = select_best_bracket_for_trade(
                        signal=edge_result.signal,
                        forecast_implied=forecast_result.implied_temp,
                        market_implied=market_result.implied_temp,
                        bracket_candles=bracket_candles,
                        model_probs=None,
                        maker_fill_prob=maker_fill_prob,
                        min_ev_cents=2.0,
                    )
                    if best_trade is not None:
                        pnl_result = calculate_realistic_pnl(
                            trade=best_trade,
                            settlement=settlement,
                            num_contracts=1,
                        )
                        pnl_gross = pnl_result["pnl_gross"]
                        fee_usd = pnl_result["fee_usd"]
                        entry_price_cents = pnl_result["entry_price_cents"]
                        trade_role = pnl_result["role"]
                        trade_side = pnl_result["side"]
                        trade_action = pnl_result["action"]
                        target_bracket_label = best_trade["label"]
                        bracket_won = pnl_result["bracket_won"]
                        trade_won = pnl_result["trade_won"]
                        ev_cents = best_trade["ev_cents"]
                else:
                    # SINGLE-MFP MODE: Use only the specified maker_fill_prob
                    best_trade = select_best_bracket_for_trade(
                        signal=edge_result.signal,
                        forecast_implied=forecast_result.implied_temp,
                        market_implied=market_result.implied_temp,
                        bracket_candles=bracket_candles,
                        model_probs=None,
                        maker_fill_prob=maker_fill_prob,
                        min_ev_cents=2.0,
                    )

                    if best_trade is not None:
                        pnl_result = calculate_realistic_pnl(
                            trade=best_trade,
                            settlement=settlement,
                            num_contracts=1,
                        )

                        pnl = pnl_result["pnl_net"]
                        pnl_gross = pnl_result["pnl_gross"]
                        fee_usd = pnl_result["fee_usd"]
                        entry_price_cents = pnl_result["entry_price_cents"]
                        trade_role = pnl_result["role"]
                        trade_side = pnl_result["side"]
                        trade_action = pnl_result["action"]
                        target_bracket_label = best_trade["label"]
                        bracket_won = pnl_result["bracket_won"]
                        trade_won = pnl_result["trade_won"]
                        ev_cents = best_trade["ev_cents"]
                    else:
                        pnl = None
            else:
                # SIMPLIFIED: Binary P&L (+1/-1) - legacy mode
                if edge_result.signal == EdgeSignal.BUY_HIGH:
                    pnl = 1.0 if settlement > market_result.implied_temp else -1.0
                else:
                    pnl = 1.0 if settlement < market_result.implied_temp else -1.0

        # Extract features from original data
        row_data = {
            "day": day,
            "snapshot_time": snapshot_time,
            "forecast_temp": forecast_result.implied_temp,
            "market_temp": market_result.implied_temp,
            "edge": edge_result.edge,
            "signal": edge_result.signal.value,
            "confidence": edge_result.confidence,
            "forecast_uncertainty": forecast_result.uncertainty,
            "market_uncertainty": market_result.uncertainty,
            "base_temp": base_temp,
            "predicted_delta": forecast_result.predicted_delta,
            "settlement_temp": settlement,
            "pnl": pnl,
            # NEW: Realistic P&L details
            "pnl_gross": pnl_gross,
            "fee_usd": fee_usd,
            "entry_price_cents": entry_price_cents,
            "trade_role": trade_role,
            "trade_side": trade_side,
            "trade_action": trade_action,
            "target_bracket": target_bracket_label,
            "bracket_won": bracket_won,
            "trade_won": trade_won,
            "ev_cents": ev_cents,
        }

        # Add multi-MFP P&L columns if in multi-MFP mode
        if multi_mfp_mode:
            row_data.update(multi_mfp_pnl)

        # Add context features from original data
        for col in [
            "snapshot_hour",
            "hours_to_event_close",
            "minutes_since_market_open",
            "obs_fcst_max_gap",
            "fcst_remaining_potential",
            "temp_volatility_30min",
            "market_bid_ask_spread",
        ]:
            if col in snapshot_df.columns:
                row_data[col] = snapshot_df[col].iloc[0]

        results.append(row_data)

    return results


def generate_edge_data(
    city: str,
    df: pd.DataFrame,
    model_path: Path,
    edge_threshold: float = 1.5,
    sample_rate: int = 12,
    max_workers: int = 14,
    maker_fill_prob: float = 0.4,
    use_realistic_pnl: bool = True,
    multi_mfp_mode: bool = False,
    candle_parquet: Optional[Path] = None,
    settlements_parquet: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate edge detection data using parallel processing.

    Uses ThreadPoolExecutor with shared memory for model and candle cache.
    All DB queries are done upfront in batch (2 queries total).

    Args:
        city: City name
        df: Combined feature DataFrame
        model_path: Path to ordinal model
        edge_threshold: Threshold for edge detection
        sample_rate: Sample every Nth snapshot per day (1 = all)
        max_workers: Number of parallel workers
        maker_fill_prob: Probability maker order fills (0-1) for EV calculation
        multi_mfp_mode: Compute P&L for multiple maker_fill_prob values (for Optuna tuning)
        use_realistic_pnl: Use realistic P&L with fees (True) or simplified binary (False)
        candle_parquet: Optional path to parquet file with candle data (instead of DB)

    Returns:
        DataFrame with edge features and outcomes
    """
    logger.info(f"Generating edge data for {city} with {max_workers} workers...")

    unique_days = sorted(df["day"].unique())
    logger.info(f"Processing {len(unique_days)} unique days")

    # Load model ONCE (shared by all threads)
    logger.info(f"Loading model from {model_path}...")
    model = OrdinalDeltaTrainer()
    model.load(model_path)

    # Batch load all settlements (from parquet or 1 query)
    logger.info("Batch loading settlements...")
    settlements = load_all_settlements(city, unique_days, settlements_parquet=settlements_parquet)
    logger.info(f"Loaded {len(settlements)} settlements")

    # Filter to days with settlements
    days_with_settlement = [d for d in unique_days if d in settlements]
    logger.info(f"Days with settlement data: {len(days_with_settlement)}")

    # Batch load ALL candles (DB query OR parquet file)
    if candle_parquet:
        logger.info(f"Loading candles from parquet: {candle_parquet}")
        candle_cache = load_all_candles_from_parquet(city, candle_parquet, days_with_settlement)
    else:
        logger.info("Batch loading ALL candles from database (this may take a moment)...")
        candle_cache = load_all_candles_batch(city, days_with_settlement)
    logger.info(f"Candle cache built: {len(candle_cache)} (day, bracket) entries")

    if not candle_cache:
        logger.error("Candle cache is EMPTY! No candles loaded.")
        return pd.DataFrame()

    # Show sample cache keys
    sample_keys = list(candle_cache.keys())[:5]
    logger.info(f"Sample cache keys: {sample_keys}")

    # Prepare day data
    day_data = []
    for day in days_with_settlement:
        day_df = df[df["day"] == day].copy()
        if day_df.empty:
            continue
        day_data.append((day, day_df, settlements[day]))

    if not day_data:
        logger.warning("No days to process")
        return pd.DataFrame()

    logger.info(f"Processing {len(day_data)} days with {max_workers} threads...")

    # Process in parallel with ThreadPoolExecutor (shared memory!)
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_day,
                day,
                day_df,
                model,  # Shared model
                candle_cache,  # Shared candle cache
                settlement,
                edge_threshold,
                sample_rate,
                city,  # Pass city for timezone lookup
                maker_fill_prob,  # For EV calculation
                use_realistic_pnl,  # Realistic vs simplified P&L
                multi_mfp_mode,  # Compute P&L for multiple maker_fill_prob values
            ): day
            for day, day_df, settlement in day_data
        }

        with tqdm(total=len(futures), desc="Processing days") as pbar:
            for future in as_completed(futures):
                day = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Error processing day {day}: {e}")
                pbar.update(1)

    if not all_results:
        logger.warning("No edge data generated")
        return pd.DataFrame()

    df_results = pd.DataFrame(all_results)
    logger.info(f"Generated {len(df_results):,} edge samples")

    # Filter to signals only (exclude no_trade)
    df_signals = df_results[df_results["signal"] != "no_trade"]
    logger.info(f"Signals with outcomes: {len(df_signals[df_signals['pnl'].notna()]):,}")

    return df_results


def main():
    parser = argparse.ArgumentParser(description="Train ML Edge Classifier")
    parser.add_argument(
        "--city",
        type=str,
        default="chicago",
        choices=list(CITY_CONFIG.keys()),
        help="City to train on",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for Optuna",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Edge threshold in degrees F (default: from config/edge_thresholds.py)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=4,
        help="Sample every Nth snapshot per day (1 = all)",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for trade decision",
    )
    parser.add_argument(
        "--optuna-metric",
        type=str,
        default="sharpe",
        choices=["auc", "filtered_precision", "f1", "mean_pnl", "sharpe"],
        help="Optuna objective metric (default: sharpe)",
    )
    parser.add_argument(
        "--min-trades-for-metric",
        type=int,
        default=50,
        help="Minimum trades for P&L/Sharpe metrics stability (default: 50)",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of CV splits for DayGroupedTimeSeriesSplit (default: 5)",
    )
    parser.add_argument(
        "--no-threshold-tuning",
        action="store_true",
        help="Disable validation-based tuning of decision threshold",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate edge data even if cached",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Limit to first N days (for testing)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to ordinal model. Default: models/saved/{city}/ordinal_catboost_optuna.pkl",
    )
    parser.add_argument(
        "--maker-fill-prob",
        type=float,
        default=0.4,
        help="Maker order fill probability for EV calculation (default: 0.4)",
    )
    parser.add_argument(
        "--no-realistic-pnl",
        action="store_true",
        help="Use simplified binary P&L (+1/-1) instead of realistic P&L with fees",
    )
    parser.add_argument(
        "--multi-mfp",
        action="store_true",
        help="Compute P&L for multiple maker_fill_prob values (enables Optuna tuning of mfp)",
    )
    parser.add_argument(
        "--tune-mfp",
        action="store_true",
        help="Add maker_fill_prob to Optuna search space (requires --multi-mfp cached data)",
    )
    parser.add_argument(
        "--candle-parquet",
        type=str,
        default=None,
        help="Path to parquet file with candle data (use instead of database). "
             "Create with: python scripts/export_kalshi_candles.py --city {city}",
    )
    parser.add_argument(
        "--regenerate-only",
        action="store_true",
        help="Generate edge data parquet and exit (no training). Use on slow machine with DB.",
    )
    parser.add_argument(
        "--from-parquet",
        action="store_true",
        help="Load settlements and candles from parquet files (no DB required). "
             "Auto-detects: models/raw_data/{city}/settlements.parquet and "
             "models/candles/candles_{city}.parquet",
    )
    args = parser.parse_args()

    # Resolve edge threshold from config if not specified
    if args.threshold is None:
        from config.edge_thresholds import get_min_edge_threshold
        args.threshold = get_min_edge_threshold(args.city)
        logger.info(f"Using threshold from config: {args.threshold}°F")

    # Auto-detect candle parquet if not specified
    if args.candle_parquet is None:
        default_parquet = Path(f"models/candles/candles_{args.city}.parquet")
        if default_parquet.exists():
            args.candle_parquet = str(default_parquet)
            logger.info(f"Auto-detected candle parquet: {default_parquet}")

    # Auto-detect settlements parquet when --from-parquet is set
    args.settlements_parquet = None
    if args.from_parquet:
        settlements_parquet = Path(f"models/raw_data/{args.city}/settlements.parquet")
        if settlements_parquet.exists():
            args.settlements_parquet = settlements_parquet
            logger.info(f"Auto-detected settlements parquet: {settlements_parquet}")
        else:
            logger.warning(f"--from-parquet set but settlements parquet not found: {settlements_parquet}")

    use_realistic_pnl = not args.no_realistic_pnl

    print("=" * 60)
    print("ML EDGE CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"City: {args.city}")
    print(f"Optuna trials: {args.trials}")
    print(f"Optuna metric: {args.optuna_metric}")
    print(f"Workers: {args.workers}")
    print(f"Edge threshold: {args.threshold}°F")
    print(f"Sample rate: every {args.sample_rate}th snapshot")
    print(f"P&L mode: {'REALISTIC (with fees)' if use_realistic_pnl else 'SIMPLIFIED (binary)'}")
    if use_realistic_pnl:
        print(f"Maker fill probability: {args.maker_fill_prob:.1%}")
    if args.multi_mfp:
        print(f"Multi-MFP mode: ENABLED (computing P&L for {len(MULTI_MFP_VALUES)} mfp values)")
    if args.tune_mfp:
        print(f"Tune MFP: ENABLED (Optuna will search maker_fill_prob)")
    if args.model_path:
        print(f"Ordinal model: {args.model_path}")
    else:
        print(f"Ordinal model: models/saved/{args.city}/ordinal_catboost_optuna.pkl (default)")
    if args.max_days:
        print(f"Max days: {args.max_days}")
    if args.candle_parquet:
        print(f"Candle source: parquet ({args.candle_parquet})")
    else:
        print("Candle source: database")
    if args.settlements_parquet:
        print(f"Settlement source: parquet ({args.settlements_parquet})")
    else:
        print("Settlement source: database")
    if args.from_parquet:
        print("Mode: PARQUET-ONLY (no DB required)")
    print()

    # Check for cached edge data (skip cache if max_days is set - test mode)
    # Use different cache files for realistic vs simplified P&L, and for multi-MFP mode
    if args.multi_mfp:
        pnl_suffix = "realistic_multi_mfp"
    else:
        pnl_suffix = "realistic" if use_realistic_pnl else "simplified"
    cache_path = Path(f"models/saved/{args.city}/edge_training_data_{pnl_suffix}.parquet")

    # Determine ordinal model path early (needed for cache validation)
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = Path(f"models/saved/{args.city}/ordinal_catboost_optuna.pkl")

    # Determine candle path early (needed for cache validation)
    if args.candle_parquet:
        candle_path = Path(args.candle_parquet)
    else:
        candle_path = Path(f"models/candles/candles_{args.city}.parquet")

    # Smart cache invalidation with metadata tracking
    if args.regenerate or args.max_days:
        use_cache = False
        if args.regenerate:
            logger.info("⚠️  Regenerating: --regenerate flag set")
        if args.max_days:
            logger.info(f"⚠️  Regenerating: --max-days={args.max_days} (test mode)")
    else:
        use_cache = not should_regenerate_edge_data(
            cache_path=cache_path,
            ordinal_model_path=model_path,
            candle_parquet_path=candle_path,
            threshold=args.threshold,
            sample_rate=args.sample_rate,
            pnl_mode=pnl_suffix,
        )

    if use_cache:
        df_edge = pd.read_parquet(cache_path)

        # Filter to desired threshold if cache was generated with lower threshold
        meta_path = cache_path.with_suffix('.meta.json')
        cache_threshold = 0.5  # Default if no metadata
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                cache_threshold = meta.get('threshold', 0.5)
            except:
                pass

        if args.threshold > cache_threshold:
            initial_count = len(df_edge)
            df_edge = df_edge[df_edge['edge'].abs() >= args.threshold].copy()
            logger.info(f"   Filtered cache from {cache_threshold}°F to {args.threshold}°F")
            logger.info(f"   {initial_count:,} edges → {len(df_edge):,} edges")
    else:
        # Validate ordinal model exists
        if not model_path.exists():
            logger.error(f"❌ Ordinal model not found: {model_path}")
            logger.error(f"   Run ordinal training first:")
            logger.error(f"   python scripts/train_city_ordinal_optuna.py --city {args.city}")
            return 1

        logger.info(f"Using ordinal model: {model_path}")

        # Load combined data
        df_combined = load_combined_data(args.city)

        # Apply max-days limit if set (use LAST N days for candle overlap)
        if args.max_days:
            unique_days = sorted(df_combined["day"].unique())
            days_to_keep = unique_days[-args.max_days:]  # Last N days (most recent)
            df_combined = df_combined[df_combined["day"].isin(days_to_keep)]
            logger.info(f"Limited to last {len(days_to_keep)} days ({len(df_combined):,} rows)")

        # Generate edge data with parallel processing
        df_edge = generate_edge_data(
            city=args.city,
            df=df_combined,
            model_path=model_path,
            edge_threshold=args.threshold,
            sample_rate=args.sample_rate,
            max_workers=args.workers,
            maker_fill_prob=args.maker_fill_prob,
            use_realistic_pnl=use_realistic_pnl,
            multi_mfp_mode=args.multi_mfp,
            candle_parquet=Path(args.candle_parquet) if args.candle_parquet else None,
            settlements_parquet=args.settlements_parquet,
        )

        if df_edge.empty:
            logger.error("No edge data generated")
            return 1

        # Cache for future runs (only if not in test mode)
        if not args.max_days:
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Handle existing file with wrong permissions
            if cache_path.exists():
                try:
                    cache_path.unlink()  # Remove existing file
                except PermissionError:
                    # File owned by root or another user - write to alternate location
                    alt_path = cache_path.with_suffix(".new.parquet")
                    logger.warning(f"Cannot overwrite {cache_path} (permission denied)")
                    logger.warning(f"Writing to alternate path: {alt_path}")
                    df_edge.to_parquet(alt_path, index=False)
                    logger.info(f"Cached edge data to {alt_path}")
                    logger.info(f"To fix: sudo rm {cache_path} && mv {alt_path} {cache_path}")
                    cache_path = alt_path  # Update for later references
                else:
                    df_edge.to_parquet(cache_path, index=False)
                    logger.info(f"Cached edge data to {cache_path}")
                    # Save metadata for smart cache invalidation
                    save_edge_data_metadata(
                        cache_path=cache_path,
                        city=args.city,
                        ordinal_model_path=model_path,
                        candle_parquet_path=candle_path if args.candle_parquet else None,
                        threshold=args.threshold,
                        sample_rate=args.sample_rate,
                        pnl_mode=pnl_suffix,
                    )
            else:
                df_edge.to_parquet(cache_path, index=False)
                logger.info(f"Cached edge data to {cache_path}")
                # Save metadata for smart cache invalidation
                save_edge_data_metadata(
                    cache_path=cache_path,
                    city=args.city,
                    ordinal_model_path=model_path,
                    candle_parquet_path=candle_path if args.candle_parquet else None,
                    threshold=args.threshold,
                    sample_rate=args.sample_rate,
                    pnl_mode=pnl_suffix,
                )

    # Early exit if --regenerate-only (for slow machine with DB)
    if args.regenerate_only:
        print("\n" + "=" * 60)
        print("EDGE DATA GENERATION COMPLETE (--regenerate-only)")
        print("=" * 60)
        print(f"Output: {cache_path}")
        print(f"Rows: {len(df_edge):,}")
        n_signals = (df_edge["signal"] != "no_trade").sum()
        n_valid_pnl = df_edge["pnl"].notna().sum()
        print(f"Signals (non-no_trade): {n_signals:,}")
        print(f"Valid P&L rows: {n_valid_pnl:,}")
        if n_valid_pnl > 0:
            mean_pnl = df_edge["pnl"].mean()
            print(f"Mean P&L: ${mean_pnl:.4f}")
        print("\nCopy this file to fast machine for training/sweeps.")
        return 0

    # Filter to signals only (exclude no_trade for training)
    df_signals = df_edge[df_edge["signal"] != "no_trade"].copy()
    logger.info(f"Training on {len(df_signals):,} edge signals")

    # Show class balance
    n_wins = (df_signals["pnl"] > 0).sum()
    n_total = df_signals["pnl"].notna().sum()
    print(f"\nClass balance: {n_wins}/{n_total} wins ({n_wins/n_total:.1%})")

    # Enhanced statistics for realistic P&L
    if use_realistic_pnl and "pnl_gross" in df_signals.columns:
        valid_pnl = df_signals[df_signals["pnl"].notna()]
        print("\n--- REALISTIC P&L STATISTICS ---")
        print(f"Total samples with valid trades: {len(valid_pnl):,}")
        print(f"Average P&L per trade: ${valid_pnl['pnl'].mean():.4f}")
        print(f"Std P&L per trade: ${valid_pnl['pnl'].std():.4f}")
        print(f"Total gross P&L: ${valid_pnl['pnl_gross'].sum():.2f}")
        print(f"Total fees paid: ${valid_pnl['fee_usd'].sum():.2f}")
        print(f"Total net P&L: ${valid_pnl['pnl'].sum():.2f}")

        if "trade_role" in valid_pnl.columns:
            role_counts = valid_pnl["trade_role"].value_counts()
            print(f"\nTrade roles: {dict(role_counts)}")

        if "trade_side" in valid_pnl.columns:
            side_counts = valid_pnl["trade_side"].value_counts()
            print(f"Trade sides: {dict(side_counts)}")

        if "trade_action" in valid_pnl.columns:
            action_counts = valid_pnl["trade_action"].value_counts()
            print(f"Trade actions: {dict(action_counts)}")

        if "entry_price_cents" in valid_pnl.columns:
            print(f"\nEntry price range: {valid_pnl['entry_price_cents'].min():.0f}¢ - {valid_pnl['entry_price_cents'].max():.0f}¢")
            print(f"Average entry price: {valid_pnl['entry_price_cents'].mean():.1f}¢")
    print()

    # Train EdgeClassifier
    print("=" * 60)
    print(f"OPTUNA TRAINING ({args.trials} trials)")
    print("=" * 60)

    classifier = EdgeClassifier(
        n_trials=args.trials,
        n_jobs=args.workers,
        decision_threshold=args.decision_threshold,
        optimize_metric=args.optuna_metric,
        min_trades_for_metric=args.min_trades_for_metric,
        tune_mfp=args.tune_mfp,
    )

    metrics = classifier.train(
        df_signals,
        target_col="pnl",
        shuffle=False,  # CRITICAL: Must be False to prevent leakage!
        tune_threshold=not args.no_threshold_tuning,
        cv_splits=args.cv_splits,
    )

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Test AUC: {metrics['test_auc']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.1%}")
    print()
    print(f"Baseline win rate: {metrics['baseline_win_rate']:.1%}")
    print(f"Filtered win rate: {metrics['filtered_win_rate']:.1%}")
    print(f"Improvement: +{(metrics['filtered_win_rate'] - metrics['baseline_win_rate'])*100:.1f}pp")
    print()
    print(f"Trades recommended: {metrics['n_trades_recommended']}/{metrics['n_test_total']} ({metrics['n_trades_recommended']/metrics['n_test_total']:.1%})")
    print()

    # Feature importance
    print("Feature Importance:")
    importance = metrics["feature_importance"]
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
    print()

    # Save model
    save_path = Path(f"models/saved/{args.city}/edge_classifier")
    classifier.save(save_path, city=args.city)
    print(f"Model saved to: {save_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
