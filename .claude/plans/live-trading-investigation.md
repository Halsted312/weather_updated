# Plan: Live Trading Investigation & YES/NO BUY/SELL Understanding

**Created**: 2025-12-09
**Status**: draft
**Priority**: high
**Agent**: kalshi-weather-quant

---

## Objective

Investigate the live trading infrastructure, understand the WebSocket implementation, and fully understand the YES/NO × BUY/SELL trade mechanics for edge-based trading.

---

## Context

As of 2025-12-09:
- All 6 cities have profitable edge-based strategies (threshold sweeps complete)
- Chicago has an excellent edge classifier (Sharpe 2.77, 94% win)
- Infrastructure exists: `live_trading/edge_trader.py`, `live_trading/websocket/handler.py`
- Daemons running: candle polling, forecast polling, WS recording
- **Gap**: Haven't fully explored YES vs NO and BUY vs SELL mechanics in live trading

---

## Tasks

### Task 1: Understand Kalshi Contract Mechanics
- [ ] Review how Kalshi binary contracts work (YES pays $1 if true, NO pays $1 if false)
- [ ] Understand the relationship: YES price + NO price ≈ $1 (minus spread)
- [ ] Document when to BUY YES vs SELL YES vs BUY NO vs SELL NO

### Task 2: Explore Live Trading Scripts
- [ ] Read `live_trading/edge_trader.py` - main edge-based trader
- [ ] Read `live_trading/websocket/handler.py` - WebSocket implementation (already read)
- [ ] Read `scripts/live_ws_trader.py` - WebSocket-based live trader
- [ ] Read `scripts/live_midnight_trader.py` - Midnight heuristic trader
- [ ] Identify how they currently decide YES vs NO

### Task 3: Understand Trade Selection Logic
- [ ] Read `src/trading/fees.py` - `find_best_trade()` function
- [ ] Understand how it evaluates YES BUY, YES SELL, NO BUY, NO SELL
- [ ] Review `compute_ev_per_contract()` for EV calculation
- [ ] Document maker vs taker fee differences

### Task 4: Review Current Edge Signal → Trade Mapping
- [ ] How does `EdgeSignal.BUY_HIGH` translate to actual order?
- [ ] How does `EdgeSignal.BUY_LOW` translate to actual order?
- [ ] Where is `select_best_bracket_for_trade()` used in live trading?
- [ ] Does current system support both YES and NO positions?

### Task 5: Paper Trading Test
- [ ] Run live trader in dry-run mode
- [ ] Monitor what trades it would make
- [ ] Verify edge detection is working with live data
- [ ] Check that ordinal model + threshold are applied correctly

---

## Files to Review

| File | Purpose |
|------|---------|
| `live_trading/edge_trader.py` | Main edge-based trading logic |
| `live_trading/websocket/handler.py` | WebSocket connection & auth |
| `scripts/live_ws_trader.py` | WebSocket live trader script |
| `scripts/live_midnight_trader.py` | Midnight heuristic trader |
| `src/trading/fees.py` | Fee calculations, `find_best_trade()` |
| `src/trading/risk.py` | Position sizing |
| `models/edge/detector.py` | EdgeSignal enum, `detect_edge()` |
| `open_maker/live_trader.py` | Another live trader implementation |

---

## Key Questions to Answer

1. **YES vs NO**: When forecast > market, should we:
   - BUY YES on a high bracket (betting temp will be high)?
   - SELL NO on a high bracket (same economic exposure)?
   - Which has better EV after fees?

2. **BUY vs SELL**:
   - BUY = pay premium, receive $1 if win, lose premium if lose
   - SELL = receive premium, pay $1 if lose, keep premium if win
   - When is each preferable?

3. **Maker vs Taker**:
   - Maker = post limit order, better fees, may not fill
   - Taker = hit existing order, worse fees, guaranteed fill
   - How does `maker_fill_prob` affect strategy?

4. **Multi-bracket**:
   - Can we trade multiple brackets simultaneously?
   - How do we handle position limits?

---

## Trade Direction Cheat Sheet (to verify)

```
Edge Signal      | Market Belief           | Trade Action
-----------------|-------------------------|------------------
BUY_HIGH         | Forecast > Market temp  | BUY YES on high bracket
                 |                         | OR SELL NO on high bracket
-----------------|-------------------------|------------------
BUY_LOW          | Forecast < Market temp  | BUY YES on low bracket
                 |                         | OR SELL NO on low bracket
```

The `find_best_trade()` function should evaluate all 4 options:
- YES BUY at ask price
- YES SELL at bid price
- NO BUY at (100 - yes_bid) price
- NO SELL at (100 - yes_ask) price

And return the one with highest EV after fees.

---

## Sign-off Log

### 2025-12-09 05:20 AM (Plan Created)
**Status**: Draft - ready for next session

**Completed this session**:
- ✅ Trained Austin and Chicago edge classifiers
- ✅ Ran threshold sweeps for all 6 cities
- ✅ Updated config with optimal thresholds
- ✅ Fixed cache validation (mtime → hash)
- ✅ Changed default optuna metric to sharpe

**Ready for live trading**:
| City | Threshold | Strategy |
|------|-----------|----------|
| Chicago | 10.0°F | Use classifier (Sharpe 2.77) |
| Denver | 10.0°F | Use classifier (Sharpe 0.78) |
| Austin | 9.5°F | Threshold only (85% baseline) |
| LA | 11.0°F | Threshold only (Sharpe 0.54) |
| Miami | 9.0°F | Threshold only (Sharpe 0.55) |
| Philly | 11.0°F | Threshold only (Sharpe 0.48) |

**Next session goals**:
1. Deep dive into live trading scripts
2. Understand YES/NO × BUY/SELL mechanics
3. Run paper trading test
4. Prepare for first live trades
