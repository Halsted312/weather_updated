"""
Live WebSocket Trader Configuration

User-approved settings:
- Bet size: $50/trade
- Max daily loss: $500
- Simple single-threaded architecture
- Structured logging to repository folder
- Full DB logging
"""

from pathlib import Path

# ===== TRADING THRESHOLDS (EV-based) =====
MIN_EV_PER_CONTRACT_CENTS = 3.0  # Minimum expected value per contract (in cents)
MIN_BRACKET_PROB = 0.35          # Only trade brackets with >35% win probability
MAKER_FILL_PROBABILITY = 0.4     # Assume 40% fill rate for maker orders (conservative)

# ===== INFERENCE SETTINGS =====
INFERENCE_COOLDOWN_SEC = 30.0  # Cache predictions for 30 seconds (avoid re-running on every tick)

# ===== POSITION LIMITS (USER APPROVED) =====
MAX_BET_SIZE_USD = 50.0          # $50 per trade (user specified)
MAX_DAILY_LOSS_USD = 500.0       # $500 max daily loss (user specified)
MAX_TOTAL_EXPOSURE_USD = 1000.0  # Max total open exposure ($50 x ~20 positions)
MAX_POSITIONS = 20               # Max concurrent positions (allows multiple brackets per city)
MAX_PER_CITY_EVENT = 3           # Max 3 positions per city/event (allow multiple brackets)

# ===== KELLY SIZING =====
BANKROLL_USD = 10000.0      # Effective bankroll for Kelly calculation
KELLY_FRACTION = 0.25       # Use quarter-Kelly (conservative)

# ===== DATA QUALITY CHECKS =====
MAX_FORECAST_AGE_HOURS = 24  # Reject forecasts older than 24h
MIN_OBSERVATIONS = 12        # Need at least 12 observations (1 hour at 5-min intervals)
MAX_MODEL_STD_DEGF = 4.0    # Model uncertainty threshold (std deviation in °F)
MAX_MODEL_CI_SPAN_DEGF = 10.0  # Max 90% confidence interval span

# ===== MODEL CONFIGURATION =====
MODEL_DIR = Path("models/saved")

# ===== MODEL VARIANT SELECTION =====
# Switch between different ordinal model architectures
ORDINAL_MODEL_VARIANT = "tod_v1"  # Options: "baseline", "hourly", "tod_v1"

# Model variant configurations
MODEL_VARIANTS = {
    "baseline": {
        "folder_suffix": "",
        "filename": "ordinal_catboost_optuna.pkl",
        "snapshot_hours": [10, 12, 14, 16, 18, 20, 22, 23],
        "requires_snapping": True,
        "description": "Original sparse-hour model (8 snapshots/day, 30 Optuna trials)",
    },
    "hourly": {
        "folder_suffix": "_hourly80",
        "filename": "ordinal_catboost_hourly_80trials.pkl",
        "snapshot_hours": list(range(10, 24)),
        "requires_snapping": True,
        "description": "Hourly model (14 snapshots/day, 80 Optuna trials)",
    },
    "tod_v1": {
        "folder_suffix": "_tod_v1",
        "filename": "ordinal_catboost_tod_v1.pkl",
        "snapshot_hours": None,  # Arbitrary timestamps
        "requires_snapping": False,
        "description": "Time-of-day aware model (15-min or 5-min intervals, 80 trials)",
    },
}

# Legacy settings (for backward compatibility)
MODEL_FILENAME = MODEL_VARIANTS[ORDINAL_MODEL_VARIANT]["filename"]
SNAPSHOT_HOURS = MODEL_VARIANTS[ORDINAL_MODEL_VARIANT]["snapshot_hours"] or list(range(10, 24))

# Training configuration for tod_v1 (when variant="tod_v1")
TOD_SNAPSHOT_INTERVAL_MIN = 15  # 15-minute snapshots (56 per day)
                                 # Can be changed to 5 for 5-minute snapshots (168 per day)

# Cities to trade
CITIES = ['chicago', 'austin', 'denver', 'los_angeles', 'miami', 'philadelphia']

# Series tickers for WebSocket subscription
SERIES_TICKERS = [
    "KXHIGHCHI",   # Chicago
    "KXHIGHAUS",   # Austin
    "KXHIGHDEN",   # Denver
    "KXHIGHLAX",   # Los Angeles
    "KXHIGHMIA",   # Miami
    "KXHIGHPHIL"   # Philadelphia
]

# ===== KALSHI FEE STRUCTURE =====
TAKER_FEE_RATE = 0.07  # 7% fee on taker orders (crossing the spread)
MAKER_FEE_RATE = 0.00  # 0% fee on maker orders (posting liquidity)

# ===== LOGGING CONFIGURATION =====
LOG_DIR = Path("logs/live_trader")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log files
TRADE_LOG = LOG_DIR / "trades.jsonl"           # Structured trade decisions
ERROR_LOG = LOG_DIR / "errors.jsonl"           # Error tracking
DAILY_SUMMARY_LOG = LOG_DIR / "daily_summary.jsonl"  # Daily P&L summaries
PERFORMANCE_LOG = LOG_DIR / "performance.jsonl"  # Latency/performance metrics

# ===== WEBSOCKET CONFIGURATION =====
WS_RECONNECT_DELAY_MIN = 1    # Start reconnect delay (seconds)
WS_RECONNECT_DELAY_MAX = 60   # Max reconnect delay (seconds)
WS_PING_INTERVAL = 30         # WebSocket ping interval (seconds)
WS_PING_TIMEOUT = 10          # WebSocket ping timeout (seconds)

# ===== CITY TIMEZONE MAPPING =====
CITY_TIMEZONES = {
    'chicago': 'America/Chicago',
    'austin': 'America/Chicago',
    'denver': 'America/Denver',
    'los_angeles': 'America/Los_Angeles',
    'miami': 'America/New_York',
    'philadelphia': 'America/New_York',
}

# ===== CITY CODE MAPPING (for database queries) =====
CITY_CODES = {
    'chicago': 'CHI',
    'austin': 'AUS',
    'denver': 'DEN',
    'los_angeles': 'LAX',
    'miami': 'MIA',
    'philadelphia': 'PHL',
}

# ===== VALIDATION FLAGS =====
REQUIRE_FORECAST_FRESH = True    # Enforce forecast age check
REQUIRE_MIN_OBSERVATIONS = True  # Enforce minimum observation count
REQUIRE_MODEL_CONFIDENCE = True  # Enforce model uncertainty check
ALLOW_DUPLICATE_ORDERS = False   # Prevent duplicate orders for same market
