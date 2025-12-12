# Live Traders - Archived 2025-12-11

## Purpose
Live trading scripts that execute orders on Kalshi based on model predictions.

## Why Archived
The inference pipeline needs fixes before live trading can resume.
These scripts are preserved until a new live trader is built with correct inference.

## Files

| File | Description | Was Imported By |
|------|-------------|-----------------|
| `live_ws_trader.py` | WebSocket-based live trader (790 lines) | Nothing |
| `live_midnight_trader.py` | Midnight heuristic trader | Nothing |
| `live_active_trader.py` | Active trading daemon | Nothing |
| `auto_trader_daemon.py` | Automated trading with safety limits | Nothing |
| `manual_trader.py` | Manual trade CLI | Nothing |
| `edge_decision_cli.py` | Edge decision CLI tool | Nothing |

## Dependencies
These scripts depend on:
- `models/inference/predictor.py` - Delta prediction
- `models/inference/probability.py` - Bracket probabilities
- `src/kalshi/client.py` - Order execution
- `src/trading/fees.py` - Fee calculations

## Revival Notes
To create a new live trader:
1. First fix the inference pipeline in `models/inference/`
2. Verify feature computation matches training exactly
3. Test with dry-run mode extensively
4. Start with small position sizes

These archived scripts contain useful patterns for:
- WebSocket handling
- Order lifecycle management
- Position sizing
- Safety limits and logging
