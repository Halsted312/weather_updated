---
plan_id: edge-detection-model
created: 2025-12-01
status: in_progress
priority: high
agent: kalshi-weather-quant
---

# Phase 2: Edge Detection Model

## Objective

Build a model that identifies trading edges by comparing our forecast-implied temperature vs market-implied temperature. This is "should I trade?" not "what's the temperature?"

## Context

**Phase 1 Result**: Market features (bid, ask, volume, momentum) added no signal to temperature prediction - 11/12 features had zero importance. This makes sense because:
- Market prices = what traders think
- Our forecast features = what physics/weather says
- The ordinal model already uses excellent weather features

**Phase 2 Insight**: Instead of using market data to predict temperature, use it to find **mispricings**:
- If our forecast says 85°F but market implies 82°F → BUY higher brackets
- If our forecast says 82°F but market implies 85°F → BUY lower brackets

## Key Concepts

### Market-Implied Temperature

From bracket prices, we can compute what temperature the market "expects":

```
Bracket prices (yes_bid):
  80-81: 5¢   → 5% probability
  81-82: 15¢  → 15% probability
  82-83: 35¢  → 35% probability  ← Market thinks ~82.5°F most likely
  83-84: 25¢  → 25% probability
  84-85: 10¢  → 10% probability
  85+:   5¢   → 5% probability

Market-implied temp ≈ Σ(bracket_midpoint × probability) = ~82.5°F
```

### Forecast-Implied Temperature

From our ordinal model's predictions:
```
Model predicts: P(delta=0) = 30%, P(delta=+1) = 25%, P(delta=-1) = 20%...
If base temp = 82°F:
  Forecast-implied temp ≈ base + Σ(delta × probability) = ~82.3°F
```

### Edge Signal

```
edge = forecast_implied_temp - market_implied_temp

if edge > +1.5°F:  → BUY higher brackets (market is too low)
if edge < -1.5°F: → BUY lower brackets (market is too high)
else:             → NO TRADE (no edge)
```

## Implementation Plan

### Step 1: Compute Market-Implied Temperature

Create function to extract expected temp from bracket prices:

```python
def compute_market_implied_temp(
    bracket_candles: dict[str, pd.DataFrame],
    snapshot_time: datetime,
) -> dict:
    """
    Returns:
        market_implied_temp: Expected temp from bracket prices
        market_uncertainty: Std dev of distribution
        market_skew: Upside vs downside probability mass
        bracket_probs: Dict of bracket -> probability
    """
```

### Step 2: Compute Forecast-Implied Temperature

Use ordinal model predictions to compute expected temp:

```python
def compute_forecast_implied_temp(
    model: OrdinalDeltaTrainer,
    df_snapshot: pd.DataFrame,
    base_temp: float,
) -> dict:
    """
    Returns:
        forecast_implied_temp: Expected temp from model
        forecast_uncertainty: Std dev of prediction distribution
        predicted_delta: Most likely delta
        delta_probs: Array of delta probabilities
    """
```

### Step 3: Edge Detection

Compare forecast vs market to find edges:

```python
def detect_edge(
    forecast_implied: float,
    market_implied: float,
    threshold: float = 1.5,
) -> dict:
    """
    Returns:
        edge: forecast - market (positive = market too low)
        signal: 'BUY_HIGH', 'BUY_LOW', or 'NO_TRADE'
        confidence: |edge| / threshold
    """
```

### Step 4: Backtest Framework

Test edge detection on historical data:

```python
def backtest_edge_strategy(
    city: str,
    start_date: date,
    end_date: date,
    model: OrdinalDeltaTrainer,
    edge_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    For each event day:
    1. At each snapshot time, compute edge
    2. If |edge| > threshold, record trade signal
    3. Track P&L based on settlement

    Returns DataFrame with:
        day, snapshot_time, forecast_temp, market_temp, edge,
        signal, entry_price, settlement, pnl
    """
```

## Files to Create

| File | Purpose |
|------|---------|
| `models/edge/implied_temp.py` | Market and forecast implied temp computation |
| `models/edge/detector.py` | Edge detection logic |
| `scripts/backtest_edge.py` | Backtest script for edge strategy |

## Success Metrics

| Metric | Target |
|--------|--------|
| Edge detection rate | Find edges on 20-40% of snapshots |
| Edge accuracy | When edge > 1.5°F, correct direction 60%+ of time |
| Sharpe ratio | > 1.0 on backtested trades |

## Sign-off Log

### 2025-12-01 13:30 CST
**Status**: Starting implementation

**Plan**:
1. Create `models/edge/implied_temp.py` with market/forecast implied temp
2. Create `models/edge/detector.py` with edge detection logic
3. Create `scripts/backtest_edge.py` to test on Chicago
4. Run backtest and analyze results

**Next**: Implement Step 1 - market-implied temperature
