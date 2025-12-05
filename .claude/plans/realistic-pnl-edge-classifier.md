---
plan_id: realistic-pnl-edge-classifier
created: 2025-12-05
status: draft
priority: high
agent: kalshi-weather-quant
---

# Enhanced Edge Classifier with Realistic P&L

## Problem Statement

Current edge classifier training uses **simplified binary P&L**:
```python
# Current (simplified)
pnl = 1.0 if settlement > market_implied_temp else -1.0
```

This ignores:
- Entry price (bid/ask spread)
- Kalshi fees (taker ~1.75¢, maker 0)
- Position sizing (Kelly fraction)
- Actual dollar amounts
- Early exit scenarios

**Result**: Sharpe ratio and other metrics are meaningless for real trading.

---

## Existing Infrastructure (Already Built!)

| Component | Location | What It Does |
|-----------|----------|--------------|
| `taker_fee_total()` | `src/trading/fees.py` | Kalshi 7% formula: `ceil(0.07 × P × N × (1-P) × 100)` |
| `maker_fee_total()` | `src/trading/fees.py` | Returns 0 for weather markets |
| `classify_liquidity_role()` | `src/trading/fees.py` | Determines maker vs taker |
| `compute_ev_per_contract()` | `src/trading/fees.py` | EV with fees factored in |
| `find_best_trade()` | `src/trading/fees.py` | Optimal trade selection |
| `PositionSizer` | `src/trading/risk.py` | Kelly sizing with caps |
| `calculate_pnl()` | `open_maker/utils.py` | Full P&L with fees |
| `calculate_exit_pnl()` | `open_maker/utils.py` | Early exit P&L |

---

## Enhancement Plan

### Phase 1: Realistic P&L Calculation in Training

**File**: `scripts/train_edge_classifier.py`

**Change `process_day_for_edge()` function** (lines 503-508):

```python
# CURRENT (binary)
if edge_result.signal == EdgeSignal.BUY_HIGH:
    pnl = 1.0 if settlement > market_result.implied_temp else -1.0
else:
    pnl = 1.0 if settlement < market_result.implied_temp else -1.0
```

**ENHANCED**:
```python
from src.trading.fees import taker_fee_total, classify_liquidity_role
from open_maker.utils import calculate_pnl

# Get actual entry price from candles
if edge_result.signal == EdgeSignal.BUY_HIGH:
    # We'd buy YES on a bracket above market_implied
    # Use the bracket closest to forecast_implied
    target_bracket = find_target_bracket(forecast_implied, brackets)
    entry_price_cents = target_bracket['yes_ask']  # Taker: cross the ask

    # Determine if we won (settlement in our bracket)
    bin_won = is_settlement_in_bracket(settlement, target_bracket)

    # Calculate realistic P&L
    num_contracts = 1  # Normalize to 1 contract for training
    fee_usd = taker_fee_total(entry_price_cents, num_contracts) / 100

    if bin_won:
        pnl_gross = 1.0 - (entry_price_cents / 100)  # Win: receive $1, paid entry
    else:
        pnl_gross = -(entry_price_cents / 100)  # Lose: contract worthless

    pnl = pnl_gross - fee_usd  # Net P&L in dollars

elif edge_result.signal == EdgeSignal.BUY_LOW:
    # Similar logic for BUY_LOW (we think temp will be lower)
    target_bracket = find_target_bracket(forecast_implied, brackets, direction='low')
    entry_price_cents = target_bracket['yes_ask']
    bin_won = is_settlement_in_bracket(settlement, target_bracket)
    fee_usd = taker_fee_total(entry_price_cents, num_contracts) / 100
    pnl = calculate_pnl(entry_price_cents, num_contracts, bin_won, fee_usd)
```

### Phase 2: Add Position Sizing to Training Data

Store additional columns for downstream analysis:

```python
row_data = {
    # Existing fields...
    "pnl": pnl,  # Now realistic

    # NEW: Trading execution details
    "entry_price_cents": entry_price_cents,
    "target_bracket": target_bracket_label,
    "bin_won": bin_won,
    "fee_cents": fee_usd * 100,
    "spread_cents": target_bracket['yes_ask'] - target_bracket['yes_bid'],

    # NEW: Position sizing (using defaults)
    "kelly_bet_usd": calculate_kelly_bet(edge, entry_price_cents, bankroll=10000),
    "contracts_suggested": num_contracts_suggested,
}
```

### Phase 3: Enhanced Metrics in EdgeClassifier

**File**: `models/edge/classifier.py`

Add to `train()` method's test evaluation:

```python
# After: y_pred = (y_pred_proba >= self.decision_threshold).astype(int)

# --- ENHANCED METRICS ---
trade_mask = y_pred == 1

if trade_mask.sum() > 0:
    # Realistic P&L metrics
    pnl_trades = pnl_test[trade_mask]
    entry_prices = df_test['entry_price_cents'].values[trade_mask]

    # Convert to dollars per trade
    mean_pnl_usd = float(pnl_trades.mean())
    std_pnl_usd = float(pnl_trades.std())
    total_pnl_usd = float(pnl_trades.sum())

    # Sharpe per trade (annualized assuming ~250 trading days, ~4 trades/day)
    trades_per_year = 1000  # Approximate
    sharpe_annual = (mean_pnl_usd / std_pnl_usd) * np.sqrt(trades_per_year) if std_pnl_usd > 0 else 0

    # ROI
    total_wagered = (entry_prices / 100).sum()
    roi_pct = (total_pnl_usd / total_wagered) * 100 if total_wagered > 0 else 0

    # Max drawdown (cumulative)
    cumulative_pnl = pnl_trades.cumsum()
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = cumulative_pnl - running_max
    max_drawdown_usd = float(drawdown.min())

    # Win rate (already have)
    # Profit factor
    wins = pnl_trades[pnl_trades > 0].sum()
    losses = abs(pnl_trades[pnl_trades < 0].sum())
    profit_factor = wins / losses if losses > 0 else float('inf')

    # Average win / average loss
    avg_win = pnl_trades[pnl_trades > 0].mean() if (pnl_trades > 0).sum() > 0 else 0
    avg_loss = pnl_trades[pnl_trades < 0].mean() if (pnl_trades < 0).sum() > 0 else 0
```

### Phase 4: Equity Curve & Simulation Output

Add `--simulate` flag to training script:

```bash
python scripts/train_edge_classifier.py --city austin --trials 80 --simulate --bankroll 1000
```

Output:
```
=== BACKTEST SIMULATION ($1,000 starting bankroll) ===

Period: 2023-05-12 to 2025-12-03 (931 days)
Starting bankroll: $1,000.00

Trading Summary:
  Total trades:     1,936
  Win rate:         87.1%
  Avg trade P&L:    $0.74

P&L Summary:
  Gross P&L:        $1,433.20
  Total fees:       $34.87
  Net P&L:          $1,398.33
  ROI:              139.8%

Risk Metrics:
  Sharpe (annual):  1.85
  Max drawdown:     $127.45 (8.3% of peak)
  Profit factor:    4.2
  Avg win:          $0.89
  Avg loss:         -$0.42

Final equity:       $2,398.33

Monthly breakdown:
  2023-06: +$42.10 (58 trades)
  2023-07: +$67.30 (71 trades)
  ...
```

### Phase 5: Configuration Integration

Use existing config from `config/live_trader_config.py`:

```python
from config.live_trader_config import (
    MAX_BET_SIZE_USD,
    KELLY_FRACTION,
    BANKROLL_USD,
    MAKER_FILL_PROBABILITY,
)

# In training, apply same constraints as live trading
position_sizer = PositionSizer(
    bankroll_usd=BANKROLL_USD,
    kelly_fraction=KELLY_FRACTION,
    max_bet_usd=MAX_BET_SIZE_USD,
)
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `scripts/train_edge_classifier.py` | Realistic P&L in `process_day_for_edge()`, add simulation output |
| `models/edge/classifier.py` | Enhanced metrics in `train()` |
| `models/edge/classifier.py` | New `simulate()` method for equity curves |

## New Files to Create

| File | Purpose |
|------|---------|
| `models/edge/simulation.py` | Equity curve, drawdown, monthly breakdown |

---

## Data Requirements

The candle data already has bid/ask:
- `yes_bid_close` - Latest bid price
- `yes_ask_close` - Latest ask price

We need to ensure these are captured in the edge training features.

---

## Metrics Comparison: Before vs After

| Metric | Current (Binary) | Enhanced (Realistic) |
|--------|------------------|---------------------|
| P&L values | +1 or -1 | Actual dollars (-0.50 to +0.50 typical) |
| Fees | None | ~1.75¢ per taker trade |
| Sharpe meaning | Direction accuracy | Risk-adjusted returns |
| ROI | N/A | (Net P&L / Capital deployed) |
| Drawdown | N/A | Cumulative equity drawdown |
| Profit factor | N/A | Gross wins / Gross losses |

---

## Implementation Steps

### Step 1: Update `process_day_for_edge()` (2 hours)
- Add bracket bid/ask to returned data
- Calculate realistic P&L with fees
- Track entry price and whether bracket won

### Step 2: Update `EdgeClassifier.train()` metrics (1 hour)
- Add ROI, max drawdown, profit factor
- Add annualized Sharpe calculation
- Add avg win / avg loss

### Step 3: Add simulation output (2 hours)
- Equity curve calculation
- Monthly breakdown
- Starting bankroll → ending equity

### Step 4: Testing & Validation (2 hours)
- Compare old vs new metrics
- Verify fee calculations match Kalshi
- Validate Sharpe formula

### Step 5: Retrain Austin with enhanced metrics (1 hour)
- Run training with `--regenerate`
- Compare results to binary version

---

## Success Criteria

1. P&L values are in realistic dollar amounts (not +1/-1)
2. Fees reduce win magnitude by ~1.75¢ average
3. Sharpe ratio is annualized and meaningful
4. Can simulate "$1,000 over time" equity curve
5. Output includes: ROI, max drawdown, profit factor, avg win/loss
6. Metrics match what `open_maker/core_runner.py` produces

---

## Questions for User

1. **Taker vs Maker**: Should training assume:
   - All taker (conservative - guaranteed fill, pay 7% fee)?
   - Mix based on fill probability (40% maker fill)?
   - Pure maker (optimistic - 0 fee but uncertain fill)?

2. **Position sizing in training**:
   - Normalize to 1 contract (current)?
   - Apply Kelly sizing to each trade?
   - Fixed dollar amount per trade?

3. **Early exit modeling**:
   - Train on hold-to-settlement only (simpler)?
   - Add exit signals when edge flips (complex)?

4. **Bracket selection**:
   - Closest bracket to forecast implied?
   - Bracket with highest expected EV?
   - Same bracket selection as live trading uses?

---

## Dependencies

- Uses existing: `src/trading/fees.py`, `src/trading/risk.py`, `open_maker/utils.py`
- No new external packages needed
- Compatible with existing EdgeClassifier API
