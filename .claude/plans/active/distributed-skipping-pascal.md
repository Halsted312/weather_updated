---
plan_id: edge-trader-live-system
created: 2025-12-01
status: in_progress
priority: critical
agent: kalshi-weather-quant
---

# Edge Trader Live Trading System

## Objective
Build a production-ready WebSocket-based live trading system that uses the trained edge classifier to make high-confidence trades on Kalshi weather markets.

## Key Requirements
- Trade every minute when confident (high frequency, ML-filtered)
- Hybrid order strategy: maker first → taker after timeout
- Volume-weighted maker timeout (high volume = wait longer)
- Aggressiveness dial (0-1) controlling multiple parameters
- Full trade logging for investigation and tuning
- Multi-city ready (start with Chicago)
- Always-on Docker service

---

## Architecture

### Folder Structure (New)
```
live_trading/
├── __init__.py
├── edge_trader.py           # Main WebSocket trading daemon
├── config.py                # All tunable parameters
├── order_manager.py         # Limit order tracking, maker→taker conversion
├── inference.py             # Real-time edge detection + classifier
├── db/
│   ├── __init__.py
│   ├── models.py            # SQLAlchemy models for trading tables
│   └── session_logger.py    # Log decisions, snapshots, lifecycle
├── websocket/
│   ├── __init__.py
│   ├── handler.py           # WebSocket connection + message routing
│   └── order_book.py        # Order book state management
└── utils.py

models/                      # Reorganized per-city structure
├── chicago/
│   ├── ordinal_catboost_optuna.pkl
│   ├── ordinal_catboost_optuna.json
│   ├── edge_classifier.pkl
│   └── edge_classifier.json
├── austin/                  # (after training)
├── denver/
├── los_angeles/
├── miami/
├── philadelphia/
└── shared/                  # Cross-city models if needed
```

### Config System (`live_trading/config.py`)
```python
@dataclass
class TradingConfig:
    # === AGGRESSIVENESS (0-1 dial) ===
    aggressiveness: float = 0.5  # Master dial

    # === POSITION LIMITS ===
    max_bet_per_trade_usd: float = 50.0
    max_daily_loss_usd: float = 500.0
    max_positions_per_city: int = 4
    allow_multiple_orders_same_bracket: bool = True

    # === ORDER EXECUTION ===
    maker_timeout_base_seconds: int = 120  # 2 minutes default
    volume_timeout_multiplier: float = 1.5  # High vol → longer wait
    volume_lookback_minutes: int = 30

    # === EDGE CLASSIFIER ===
    edge_confidence_threshold_base: float = 0.5  # Adjusted by aggressiveness
    edge_threshold_degf: float = 1.5

    # === BRACKET SELECTION ===
    bracket_selection: str = "forecast_implied"  # or "market_implied", "edge_boundary"

    # === FEES ===
    maker_fee_pct: float = 0.0
    taker_fee_pct: float = 0.07

    # === CITIES ===
    enabled_cities: list = field(default_factory=lambda: ["chicago"])

    # === DERIVED (computed from aggressiveness) ===
    @property
    def effective_confidence_threshold(self) -> float:
        # aggressiveness=0 → 0.7, aggressiveness=1 → 0.3
        return 0.7 - (self.aggressiveness * 0.4)

    @property
    def effective_kelly_fraction(self) -> float:
        # aggressiveness=0 → 0.1, aggressiveness=1 → 0.5
        return 0.1 + (self.aggressiveness * 0.4)

    @property
    def effective_maker_timeout_multiplier(self) -> float:
        # aggressiveness=0 → 1.5x base, aggressiveness=1 → 0.5x base
        return 1.5 - (self.aggressiveness * 1.0)
```

### Hot-Reload Config (`live_trading/config.py`)
```python
import os
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigReloader(FileSystemEventHandler):
    """Watch config file and reload on changes."""
    def __init__(self, config_path: str, callback):
        self.config_path = config_path
        self.callback = callback
        self._last_mtime = 0

    def on_modified(self, event):
        if event.src_path == self.config_path:
            mtime = os.path.getmtime(self.config_path)
            if mtime != self._last_mtime:
                self._last_mtime = mtime
                self.callback()

def load_config(path: str) -> TradingConfig:
    """Load config from JSON file, merge with defaults."""
    with open(path) as f:
        overrides = json.load(f)
    return TradingConfig(**overrides)

# Usage in main:
# config = load_config("config/trading.json")
# watcher = ConfigReloader("config/trading.json", lambda: reload_config())
# observer = Observer()
# observer.schedule(watcher, path="config/", recursive=False)
# observer.start()
```

### Timezone & DST Handling
```python
from zoneinfo import ZoneInfo
from datetime import datetime, date, time

CITY_TIMEZONES = {
    "chicago": ZoneInfo("America/Chicago"),
    "austin": ZoneInfo("America/Chicago"),  # Central
    "denver": ZoneInfo("America/Denver"),
    "los_angeles": ZoneInfo("America/Los_Angeles"),
    "miami": ZoneInfo("America/New_York"),  # Eastern
    "philadelphia": ZoneInfo("America/New_York"),
}

def get_event_date_for_city(city: str, utc_now: datetime = None) -> date:
    """
    Get the current 'weather day' for a city.
    NWS uses LOCAL STANDARD TIME for climate days.
    Weather day runs from ~6am LST to ~6am LST next day.
    """
    if utc_now is None:
        utc_now = datetime.now(ZoneInfo("UTC"))

    tz = CITY_TIMEZONES[city]
    local_now = utc_now.astimezone(tz)

    # Weather day typically ends around 6-7am local
    # If before 6am, we're still in "yesterday's" weather day
    if local_now.hour < 6:
        return (local_now - timedelta(days=1)).date()
    return local_now.date()

def get_market_close_local(city: str, event_date: date) -> datetime:
    """
    Get market close time in local timezone.
    Weather markets typically close around settlement time.
    """
    tz = CITY_TIMEZONES[city]
    # Midnight local = end of weather day
    return datetime.combine(event_date + timedelta(days=1), time(0, 0), tzinfo=tz)
```

### Volume-Weighted Maker Timeout
```python
def compute_maker_timeout(config: TradingConfig, recent_volume: int) -> int:
    """
    High volume → wait longer (more likely to get filled as maker)
    Low volume → convert to taker faster (ensure execution)
    """
    base = config.maker_timeout_base_seconds
    aggr_multiplier = config.effective_maker_timeout_multiplier

    # Normalize volume (e.g., 0-100 contracts in 30min → 0-1)
    volume_factor = min(recent_volume / 100.0, 2.0)  # Cap at 2x

    effective_timeout = base * aggr_multiplier * (1 + volume_factor * 0.5)
    return int(effective_timeout)
```

---

## Database Schema (new tables in `trading` schema)

### `trading.sessions`
```sql
CREATE TABLE trading.sessions (
    session_id UUID PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    config_json JSONB NOT NULL,      -- Full config snapshot
    status VARCHAR(20) NOT NULL,     -- running, stopped, error
    total_trades INT DEFAULT 0,
    total_pnl_cents INT DEFAULT 0
);
```

### `trading.decisions`
```sql
CREATE TABLE trading.decisions (
    decision_id UUID PRIMARY KEY,
    session_id UUID REFERENCES trading.sessions,
    created_at TIMESTAMPTZ NOT NULL,
    city VARCHAR(20) NOT NULL,
    event_date DATE NOT NULL,

    -- Edge Detection Results
    forecast_implied_temp FLOAT,
    market_implied_temp FLOAT,
    edge_degf FLOAT,
    signal VARCHAR(20),              -- buy_high, buy_low, no_trade

    -- Edge Classifier Results
    edge_classifier_prob FLOAT,
    should_trade BOOLEAN,

    -- Context at Decision Time
    market_snapshot JSONB,           -- bracket prices, bid/ask
    features_snapshot JSONB,         -- key features used

    -- Outcome
    order_placed BOOLEAN,
    order_id UUID,
    reason TEXT                      -- why trade/no-trade
);
```

### `trading.orders`
```sql
CREATE TABLE trading.orders (
    order_id UUID PRIMARY KEY,
    session_id UUID REFERENCES trading.sessions,
    decision_id UUID REFERENCES trading.decisions,
    created_at TIMESTAMPTZ NOT NULL,

    -- Order Details
    city VARCHAR(20) NOT NULL,
    event_date DATE NOT NULL,
    ticker VARCHAR(50) NOT NULL,
    bracket_label VARCHAR(20),
    side VARCHAR(10),                -- yes, no
    action VARCHAR(10),              -- buy, sell

    -- Pricing
    maker_price_cents INT,
    taker_conversion_at TIMESTAMPTZ,
    taker_price_cents INT,
    final_fill_price_cents INT,

    -- Sizing
    num_contracts INT,
    notional_usd FLOAT,

    -- Status Tracking
    status VARCHAR(20),              -- pending, filled, cancelled, converted
    status_history JSONB,            -- [{status, timestamp}, ...]

    -- Metrics
    volume_at_order INT,
    maker_timeout_used_sec INT,

    -- Settlement
    settlement_temp FLOAT,
    pnl_cents INT
);
```

---

## Core Components

### 1. WebSocket Handler (`live_trading/websocket/handler.py`)
- Connect to `wss://api.elections.kalshi.com/trade-api/ws/v2` (prod)
- RSA-PSS authentication with SHA-256 signing
- Subscribe to channels:
  - `ticker` - bid/ask/price/volume updates
  - `orderbook_delta` - full order book depth
  - `fill` - our order fills (private)
  - `market_positions` - our positions and P&L (private)
  - `market_lifecycle_v2` - market open/close times, settlement
  - `event_lifecycle` - event creation and updates
  - `trades` - public trade tape for volume tracking
- Maintain order book state per bracket using snapshot + delta
- Auto-reconnect with exponential backoff (max 30s)
- Re-subscribe to all channels on reconnect

### 2. Order Manager (`live_trading/order_manager.py`)
- Track all pending limit orders
- Background task: check for maker timeout → convert to taker
- Handle partial fills
- Enforce position limits (max 4 per city, total positions)
- Cancel stale orders on shutdown

### 3. Inference Engine (`live_trading/inference.py`)
- Load ordinal model + edge classifier per city
- Cache predictions for 30 seconds (avoid redundant inference)
- Compute: features → ordinal → forecast_implied → edge → classifier
- Return: `TradeDecision(should_trade, confidence, bracket, signal)`

### 4. Edge Trader Main Loop (`live_trading/edge_trader.py`)
```python
async def main_loop():
    while True:
        for city in config.enabled_cities:
            # 1. Get latest market prices from WebSocket state
            market_state = order_book.get_state(city, event_date)

            # 2. Run inference
            decision = inference.evaluate_edge(city, event_date, market_state)

            # 3. Log decision (always, even if no trade)
            db.log_decision(session_id, decision)

            # 4. Check if should trade
            if decision.should_trade and not position_limits_exceeded():
                # 5. Place maker order
                order = place_maker_order(decision)
                order_manager.track(order)

        await asyncio.sleep(60)  # Check every minute
```

### 5. Session Logger (`live_trading/db/session_logger.py`)
- Create session on startup with config snapshot
- Log every decision (trade or no-trade) with full context
- Log order lifecycle events
- Store market snapshots at decision time

### 6. Market State Manager (`live_trading/websocket/market_state.py`)
- Track market metadata from `market_lifecycle_v2` channel:
  - `open_ts` - when market opens for trading
  - `close_ts` - when market closes (settlement time)
  - `event_type` - created, updated, settled
  - `additional_metadata` - name, title, rules, strike_type, floor_strike
- Track event metadata from `event_lifecycle` channel:
  - `event_ticker` - e.g., "KXHIGHCHI-25DEC01"
  - `series_ticker` - e.g., "KXHIGHCHI"
  - `strike_date` - settlement date timestamp
- Auto-discover weather market tickers by subscribing to lifecycle without filter
- Filter for relevant cities: KXHIGHCHI, KXHIGHDEN, KXHIGHLA, KXHIGHMIA, KXHIGHPHI, KXHIGHAUS

### 7. Position Tracker (`live_trading/position_tracker.py`)
- Track our positions from `market_positions` channel:
  - `position` - net contracts
  - `position_cost` - in centi-cents (÷10,000 for USD)
  - `realized_pnl` - in centi-cents
  - `fees_paid` - in centi-cents
- Track fills from `fill` channel:
  - `order_id`, `trade_id`
  - `is_taker` - for fee calculation
  - `post_position` - position after fill
- Enforce position limits per city (max 4 orders)

---

## Implementation Steps

### Phase 1: Foundation (Day 1)
1. Create `live_trading/` folder structure
2. Reorganize `models/saved/` → `models/{city}/`
3. Create `config.py` with all parameters
4. Create database migrations for `trading` schema
5. Create basic SQLAlchemy models

### Phase 2: Core Components (Day 2)
6. Implement `inference.py` - load models, run edge detection
7. Implement `websocket/handler.py` - connection management
8. Implement `websocket/order_book.py` - state tracking
9. Implement `db/session_logger.py` - logging

### Phase 3: Order Management (Day 3)
10. Implement `order_manager.py` - limit order tracking
11. Implement maker→taker conversion logic
12. Implement volume-weighted timeout
13. Implement position limit checks

### Phase 4: Main Trader (Day 4)
14. Implement `edge_trader.py` main loop
15. Add CLI arguments (--live, --config-file, --city)
16. Add graceful shutdown (SIGTERM/SIGINT)
17. Add systemd service file for always-on

### Phase 5: Docker & Testing (Day 5)
18. Create Dockerfile for live_trading service
19. Update docker-compose.yml
20. Test with low config values (small bets)
21. Monitor and tune

---

## Files to Create

| File | Purpose |
|------|---------|
| `live_trading/__init__.py` | Package init |
| `live_trading/config.py` | All tunable parameters + hot-reload |
| `live_trading/edge_trader.py` | Main daemon |
| `live_trading/inference.py` | Model inference |
| `live_trading/order_manager.py` | Order lifecycle |
| `live_trading/position_tracker.py` | Position & fill tracking |
| `live_trading/utils.py` | Shared utilities |
| `live_trading/db/__init__.py` | DB package |
| `live_trading/db/models.py` | SQLAlchemy models |
| `live_trading/db/session_logger.py` | Logging |
| `live_trading/websocket/__init__.py` | WS package |
| `live_trading/websocket/handler.py` | WS connection + auth |
| `live_trading/websocket/order_book.py` | Order book state |
| `live_trading/websocket/market_state.py` | Market metadata tracking |
| `live_trading/websocket/messages.py` | Pydantic message models |
| `config/trading.json` | Default config file |
| `alembic/versions/xxx_trading_schema.py` | Migration |
| `docker/edge_trader.Dockerfile` | Service container |

## Files to Modify

| File | Changes |
|------|---------|
| `docker-compose.yml` | Add edge_trader service |
| `models/saved/` | Reorganize to `models/{city}/` |

## Critical Reference Files

| File | Why |
|------|-----|
| `docs/how-tos/kalshi_websockets.md` | WebSocket API spec, channels, auth |
| `docs/how-tos/kalshi_websockets_doc.md` | Full client design, Pydantic models |
| `models/saved/chicago/edge_classifier.pkl` | Trained edge classifier |
| `models/saved/chicago/ordinal_catboost_optuna.pkl` | Ordinal model for forecasts |
| `src/config/city_config.py` | City→station→timezone mappings |
| `src/features/feature_pipeline.py` | Feature engineering |
| `models/edge/edge_detector.py` | Edge detection logic |

---

## Aggressiveness Scale Effects

| Aggressiveness | Confidence Threshold | Kelly Fraction | Maker Timeout Mult |
|----------------|---------------------|----------------|-------------------|
| 0.0 (conservative) | 0.70 | 0.10 (1/10 Kelly) | 1.5x |
| 0.25 | 0.60 | 0.20 | 1.25x |
| 0.50 (default) | 0.50 | 0.30 | 1.0x |
| 0.75 | 0.40 | 0.40 | 0.75x |
| 1.0 (aggressive) | 0.30 | 0.50 (1/2 Kelly) | 0.5x |

---

## Success Criteria
- [ ] WebSocket connects with RSA-PSS auth and receives real-time prices
- [ ] All channels subscribed: ticker, orderbook_delta, fill, market_positions, market_lifecycle_v2
- [ ] Edge classifier runs inference in <100ms
- [ ] Orders placed successfully via REST API
- [ ] Maker→taker conversion works with volume-weighted timeout
- [ ] All decisions logged to database with full context
- [ ] Config hot-reload works (file watcher triggers reload)
- [ ] Timezone handling correct across DST transitions
- [ ] Market metadata tracked (open/close times, settlement)
- [ ] Position limits enforced (max 4 per city)
- [ ] Docker service runs continuously with auto-restart
- [ ] Graceful shutdown: cancel pending orders, save state

## Sign-off Log

### 2025-12-01 (Plan Creation)
**Status**: Plan complete, ready for implementation

**Key additions from user feedback**:
- Hot-reload config via file watcher for future frontend integration
- Full WebSocket channel subscriptions including market_lifecycle_v2
- Timezone/DST handling with city-specific ZoneInfo mappings
- Position tracking via market_positions and fill channels
- Volume-weighted maker timeout (high volume → wait longer)

**Next steps**:
1. Create `live_trading/` folder structure
2. Reorganize models to `models/{city}/`
3. Implement config.py with hot-reload
4. Create database migrations for trading schema
5. Build WebSocket handler with all channels

**Critical files to reference during implementation**:
- `docs/how-tos/kalshi_websockets.md` - API spec
- `docs/how-tos/kalshi_websockets_doc.md` - Client design patterns
- `models/edge/edge_detector.py` - Edge detection logic to integrate
