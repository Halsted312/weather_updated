---
plan_id: live-websocket-trading-system
created: 2025-11-28
status: guide-document
priority: high
agent: kalshi-weather-quant
---

# Live WebSocket Trading System - Implementation Guide

> **For Next Agent**: This document describes how to build a real-time trading system that uses Kalshi WebSocket for live order book data, compares market prices to model-predicted probabilities, and makes optimal make/take decisions accounting for fees.

---

## 1. System Overview

### What We're Building

A WebSocket-based live trading daemon that:
1. **Receives** live order book updates for all 6 weather cities
2. **Runs** the ordinal CatBoost model to get bracket probabilities
3. **Compares** model probs vs Kalshi market implied probs
4. **Decides** whether to make (post limit order) or take (cross spread)
5. **Executes** trades via REST API

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                    kalshi_live_trader.py                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐    │
│  │  WebSocket     │     │  Inference     │     │  Decision      │    │
│  │  Handler       │     │  Engine        │     │  Engine        │    │
│  │                │     │                │     │                │    │
│  │ • orderbook_   │────▶│ • Load models  │────▶│ • Compare EV   │    │
│  │   delta        │     │ • Get weather  │     │ • Fee adjust   │    │
│  │ • ticker       │     │ • Run predict  │     │ • Make/take    │    │
│  │ • trade        │     │                │     │                │    │
│  └────────────────┘     └────────────────┘     └────────┬───────┘    │
│                                                          │            │
│                                                          ▼            │
│                                               ┌────────────────┐     │
│                                               │  Order         │     │
│                                               │  Executor      │     │
│                                               │                │     │
│                                               │ • REST API     │     │
│                                               │ • Logging      │     │
│                                               └────────────────┘     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Existing Infrastructure

### Files You'll Use

| File | Purpose |
|------|---------|
| `scripts/kalshi_ws_recorder.py` | **Reference** for WS connection, auth, channels |
| `open_maker/live_trader.py` | **Reference** for market_lifecycle handling |
| `src/kalshi/client.py` | REST client - use `create_order()`, `get_orderbook()` |
| `models/training/ordinal_trainer.py` | Load model, call `predict_proba()` |
| `scripts/test_inference_all_cities.py` | Example inference pipeline |
| `src/config/cities.py` | City configs with `series_ticker` |

### Database Tables

| Table | Purpose |
|-------|---------|
| `kalshi.ws_raw` | Raw WebSocket messages (already recording) |
| `sim.live_orders` | Track orders placed by this system |
| `wx.vc_minute_weather` | Live weather data (from `poll_vc_live_daemon.py`) |

---

## 3. Kalshi Fee Structure

**This is critical for the make/take decision.**

| Order Type | Fee Rate | Applied To |
|------------|----------|------------|
| **Taker** | 7% | Premium paid (price × 7%) |
| **Maker** | 0% | No fee |

### Fee Math Examples

**Scenario: You believe bracket has 55% win probability**

```python
# If you TAKE at ask=50c
cost = 50 + (50 * 0.07) = 53.5c   # Effective cost
EV = 0.55 * 100 - 53.5 = +1.5c   # Expected value per contract

# If you MAKE at bid=45c (and get filled)
cost = 45 + 0 = 45c              # No maker fee
EV = 0.55 * 100 - 45 = +10c      # Much better!
```

### Decision Rule

```python
def should_trade(model_prob, best_ask, best_bid):
    """
    Returns: ('take', price), ('make', price), or (None, None)
    """
    TAKER_FEE = 0.07
    MIN_EDGE = 0.03  # Require 3% edge minimum

    # Taking: compare model_prob to breakeven including fee
    take_breakeven = best_ask * (1 + TAKER_FEE) / 100
    take_edge = model_prob - take_breakeven

    # Making: compare model_prob to our limit price (no fee)
    # We'd post slightly below best_ask or at best_bid + 1
    make_price = best_bid + 1  # One cent above best bid
    make_breakeven = make_price / 100
    make_edge = model_prob - make_breakeven

    if take_edge > MIN_EDGE:
        return ('take', best_ask)
    elif make_edge > MIN_EDGE:
        return ('make', make_price)
    else:
        return (None, None)
```

---

## 4. WebSocket Channels Needed

### Subscription Message

```python
{
    "id": 1,
    "cmd": "subscribe",
    "params": {
        "channels": ["orderbook_delta", "ticker", "trade"],
        "series_tickers": [
            "KXHIGHCHI", "KXHIGHAUS", "KXHIGHDEN",
            "KXHIGHLAX", "KXHIGHMIA", "KXHIGHPHIL"
        ]
    }
}
```

### Channel Data

**`orderbook_delta`** - Order book changes
```python
{
    "type": "orderbook_delta",
    "channel": "orderbook_delta",
    "msg": {
        "market_ticker": "KXHIGHCHI-25NOV28-B33.5",
        "yes": {
            "bid": [[45, 100], [44, 200]],  # [price_cents, quantity]
            "ask": [[48, 50], [49, 100]]
        },
        "no": {...}
    }
}
```

**`ticker`** - Price summary updates
```python
{
    "type": "ticker",
    "channel": "ticker",
    "msg": {
        "market_ticker": "KXHIGHCHI-25NOV28-B33.5",
        "yes_bid": 45,
        "yes_ask": 48,
        "last_price": 46,
        "volume": 1234
    }
}
```

---

## 5. Order Book State Management

You need to maintain in-memory state of the order book:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class OrderBook:
    """In-memory order book for a single market."""
    market_ticker: str
    yes_bids: List[Tuple[int, int]] = field(default_factory=list)  # [(price, qty), ...]
    yes_asks: List[Tuple[int, int]] = field(default_factory=list)
    last_update: datetime = None

    @property
    def best_bid(self) -> int:
        """Best (highest) YES bid price in cents."""
        return max((p for p, q in self.yes_bids), default=0)

    @property
    def best_ask(self) -> int:
        """Best (lowest) YES ask price in cents."""
        return min((p for p, q in self.yes_asks), default=100)

    @property
    def mid(self) -> float:
        """Midpoint price."""
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> int:
        """Bid-ask spread in cents."""
        return self.best_ask - self.best_bid

    def update(self, delta_msg: dict):
        """Apply orderbook_delta message."""
        if "yes" in delta_msg:
            if "bid" in delta_msg["yes"]:
                self.yes_bids = [(p, q) for p, q in delta_msg["yes"]["bid"]]
            if "ask" in delta_msg["yes"]:
                self.yes_asks = [(p, q) for p, q in delta_msg["yes"]["ask"]]
        self.last_update = datetime.now(timezone.utc)


class OrderBookManager:
    """Manage order books for all active markets."""

    def __init__(self):
        self.books: Dict[str, OrderBook] = {}

    def get_or_create(self, market_ticker: str) -> OrderBook:
        if market_ticker not in self.books:
            self.books[market_ticker] = OrderBook(market_ticker=market_ticker)
        return self.books[market_ticker]

    def update_from_delta(self, msg: dict):
        ticker = msg.get("market_ticker")
        if ticker:
            book = self.get_or_create(ticker)
            book.update(msg)
```

---

## 6. Inference Pipeline

### Loading Models

```python
from models.training.ordinal_trainer import OrdinalDeltaTrainer
from pathlib import Path

class InferenceEngine:
    """Run ordinal models for all cities."""

    CITIES = ['chicago', 'austin', 'denver', 'los_angeles', 'miami', 'philadelphia']

    def __init__(self):
        self.trainers = {}
        self._load_models()

    def _load_models(self):
        for city in self.CITIES:
            model_path = Path(f"models/saved/{city}/ordinal_catboost_optuna.pkl")
            if model_path.exists():
                trainer = OrdinalDeltaTrainer()
                trainer.load(model_path)
                self.trainers[city] = trainer
                logger.info(f"Loaded model for {city}")

    def predict(self, city: str, features: pd.DataFrame) -> np.ndarray:
        """
        Get probability distribution over delta classes.

        Args:
            city: City identifier
            features: DataFrame with feature columns

        Returns:
            Array of shape (n_samples, 13) with P(delta=k) for k in [-2, ..., +10]
        """
        if city not in self.trainers:
            raise ValueError(f"No model for {city}")
        return self.trainers[city].predict_proba(features)
```

### Feature Preparation

You'll need to prepare features from live data:

```python
def prepare_live_features(city: str, event_date: date, session) -> pd.DataFrame:
    """
    Build feature vector from live weather data.

    Queries:
    - Latest VC observation (wx.vc_minute_weather)
    - Latest VC forecast (wx.vc_forecast_daily)
    - Historical data for lag features
    """
    # This mirrors what's in models/features/base.py
    # Key features:
    # - tempmax_fcst_f (forecast high)
    # - current_temp_f (latest observation)
    # - temp_trend (change over last hour)
    # - humidity, pressure, wind
    # - time-of-day features
    # - lag features (delta_lag_1d, delta_lag_2d, etc.)
    ...
```

---

## 7. Delta-to-Bracket Mapping

The model predicts P(delta=k) for each delta class. You need to map this to bracket probabilities:

```python
def delta_to_bracket_prob(
    delta_probs: np.ndarray,  # Shape: (13,) for delta in [-2, ..., +10]
    forecast_temp: float,
    market_df: pd.DataFrame,  # Brackets with floor_strike, cap_strike
) -> Dict[str, float]:
    """
    Convert delta probabilities to bracket win probabilities.

    delta = settled_temp - forecast_temp

    For bracket [floor, cap):
        P(win) = sum of P(delta=k) where floor <= forecast+k <= cap
    """
    DELTA_CLASSES = list(range(-2, 11))  # [-2, -1, 0, ..., 10]

    bracket_probs = {}
    for _, row in market_df.iterrows():
        ticker = row['ticker']
        floor_strike = row.get('floor_strike')
        cap_strike = row.get('cap_strike')
        strike_type = row['strike_type']

        prob = 0.0
        for i, delta in enumerate(DELTA_CLASSES):
            settled_temp = forecast_temp + delta

            # Check if this settled temp wins this bracket
            if strike_type == 'less':
                # Wins if temp <= cap
                if settled_temp <= cap_strike:
                    prob += delta_probs[i]
            elif strike_type == 'greater':
                # Wins if temp >= floor + 1
                if settled_temp >= floor_strike + 1:
                    prob += delta_probs[i]
            elif strike_type == 'between':
                # Wins if floor <= temp <= cap (check subtitle for exact bounds)
                if floor_strike <= settled_temp <= cap_strike:
                    prob += delta_probs[i]

        bracket_probs[ticker] = prob

    return bracket_probs
```

---

## 8. Main Trading Loop

```python
class LiveWebSocketTrader:
    """Main trading daemon."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.order_book_mgr = OrderBookManager()
        self.inference = InferenceEngine()
        self.kalshi_client = None if dry_run else KalshiClient(...)

        # Track positions to avoid doubling down
        self.positions: Dict[str, int] = {}  # ticker -> contracts held

    async def handle_orderbook_delta(self, msg: dict):
        """Process orderbook update and potentially trade."""
        market_ticker = msg.get("market_ticker", "")

        # Update order book
        self.order_book_mgr.update_from_delta(msg)
        book = self.order_book_mgr.books.get(market_ticker)
        if not book:
            return

        # Parse ticker to get city and event_date
        city, event_date = self._parse_market_ticker(market_ticker)
        if not city:
            return

        # Skip if we already have a position
        if self.positions.get(market_ticker, 0) > 0:
            return

        # Run inference
        try:
            model_prob = await self._get_model_prob(city, event_date, market_ticker)
            if model_prob is None:
                return

            # Make trading decision
            action, price = self._decide(model_prob, book.best_ask, book.best_bid)

            if action == 'take':
                await self._execute_take(market_ticker, price, model_prob)
            elif action == 'make':
                await self._execute_make(market_ticker, price, model_prob)

        except Exception as e:
            logger.error(f"Error processing {market_ticker}: {e}")

    def _decide(self, model_prob: float, best_ask: int, best_bid: int):
        """Make/take/pass decision."""
        TAKER_FEE = 0.07
        MIN_EDGE = 0.03

        # Calculate expected values
        take_cost = best_ask * (1 + TAKER_FEE)
        take_ev = model_prob * 100 - take_cost

        make_price = min(best_bid + 1, best_ask - 1)  # Improve bid by 1c
        make_ev = model_prob * 100 - make_price

        # Prefer taking if edge is good (faster fill)
        if take_ev / 100 > MIN_EDGE:
            return ('take', best_ask)
        elif make_ev / 100 > MIN_EDGE:
            return ('make', make_price)
        else:
            return (None, None)
```

---

## 9. File Structure to Create

```
scripts/
└── kalshi_live_ws_trader.py     # Main daemon (new)

src/trading/                      # New module
├── __init__.py
├── order_book.py                 # OrderBook, OrderBookManager
├── inference_engine.py           # Model loading, prediction
├── decision_engine.py            # Fee calculations, make/take logic
└── executor.py                   # Order placement, position tracking

models/
└── inference/                    # Optional: separated inference utils
    └── live_predictor.py
```

---

## 10. Configuration Parameters

```python
# Trading parameters (tune these)
TAKER_FEE_RATE = 0.07        # 7% taker fee
MAKER_FEE_RATE = 0.00        # 0% maker fee
MIN_EDGE_TAKE = 0.05         # Require 5% edge to take
MIN_EDGE_MAKE = 0.03         # Require 3% edge to make
MAX_POSITION_PER_MARKET = 50 # Max contracts per market
BET_AMOUNT_USD = 20.0        # Default bet size

# Timing
INFERENCE_COOLDOWN_SEC = 30  # Don't re-run inference too often
ORDER_BOOK_STALE_SEC = 60    # Ignore stale order books
```

---

## 11. Testing Checklist

- [ ] WebSocket connects and receives orderbook_delta messages
- [ ] Order books update correctly from delta messages
- [ ] Models load for all 6 cities
- [ ] Inference runs and produces valid probabilities (sum to 1.0)
- [ ] Delta-to-bracket mapping is correct
- [ ] Fee calculations are correct
- [ ] Dry-run mode logs decisions without placing orders
- [ ] Real orders place successfully (small size first!)
- [ ] Position tracking prevents doubling down

---

## 12. Example Dry-Run Output

```
2025-11-28 10:15:32 - INFO - Connected to WebSocket
2025-11-28 10:15:32 - INFO - Subscribed to channels: orderbook_delta, ticker, trade
2025-11-28 10:15:33 - INFO - Loaded model for chicago
2025-11-28 10:15:33 - INFO - Loaded model for austin
...
2025-11-28 10:16:01 - INFO - [CHI] KXHIGHCHI-25NOV28-B33.5
                              Model P(win)=0.42, Best ask=45c, Best bid=40c
                              Take EV: 0.42*100 - 45*1.07 = -6.15c (PASS)
                              Make EV: 0.42*100 - 41 = +1.00c (EDGE TOO SMALL)
                              Decision: PASS

2025-11-28 10:17:15 - INFO - [DEN] KXHIGHDEN-25NOV28-B45.5
                              Model P(win)=0.58, Best ask=48c, Best bid=45c
                              Take EV: 0.58*100 - 48*1.07 = +6.64c (TRADE!)
                              Decision: TAKE @ 48c
                              [DRY RUN] Would buy 41 contracts @ 48c = $20.00
```

---

## 13. Key Files to Read Before Implementing

1. `scripts/kalshi_ws_recorder.py` - WS connection patterns
2. `open_maker/live_trader.py` - Trading flow, order placement
3. `models/training/ordinal_trainer.py` - Model loading, `predict_proba()`
4. `src/config/cities.py` - Series tickers, city config
5. `src/kalshi/client.py` - `create_order()` API

---

## 14. Safety Considerations

1. **Start with dry-run** - Always test decision logic first
2. **Small bet sizes** - Start with $5-10 per market
3. **Position limits** - Cap exposure per market
4. **Stale data checks** - Don't trade on old order books
5. **Model confidence** - Consider adding probability threshold
6. **Rate limiting** - Don't spam orders
7. **Logging** - Log all decisions and trades for analysis
