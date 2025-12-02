---
plan_id: handoff-live-trading
created: 2025-12-01
status: in_progress
priority: high
agent: kalshi-weather-quant
---

# Live Trading Handoff Guide

## Objective
Concise handoff for the junior agent to implement the live trading system plan while reusing existing code and docs.

## Context
- Active plan: `.claude/plans/active/distributed-skipping-pascal.md` (production WebSocket trader with maker->taker, edge classifier, logging, Docker).
- Current live paths already exist: `scripts/live_ws_trader.py`, `scripts/live_active_trader.py`, `scripts/kalshi_ws_recorder.py`.
- Inference + edge stack: `models/inference/live_engine.py`, `models/edge/{implied_temp.py,detector.py,classifier.py}`, saved models in `models/saved/chicago/*` and other cities.
- Risk/fees utilities: `src/trading/{fees.py,risk.py}`.
- Config in use today: `config/live_trader_config.py` (hybrid D-1/D models, EV thresholds, city lists).
- WebSocket spec and client design: `docs/how-tos/kalshi_websockets.md`, `docs/how-tos/kalshi_websockets_doc.md`.
- File map & rules: `docs/permanent/FILE_DICTIONARY_GUIDE.md`, `CLAUDE.md`, `README.md`.

## Key References (paths)
- Project guidance: `CLAUDE.md`, `README.md`, `docs/permanent/FILE_DICTIONARY_GUIDE.md`, `docs/how-tos/DATETIME_AND_API_REFERENCE.md`.
- WebSockets: `docs/how-tos/kalshi_websockets.md`, `docs/how-tos/kalshi_websockets_doc.md`, `scripts/kalshi_ws_recorder.py`.
- Live traders today: `scripts/live_ws_trader.py`, `scripts/live_active_trader.py`.
- Inference/edge: `models/inference/live_engine.py`, `models/edge/{implied_temp.py,detector.py,classifier.py}`, `scripts/train_edge_classifier.py`, artifacts in `models/saved/{city}/`.
- Risk & fees: `src/trading/{fees.py,risk.py}`.
- Config: `config/live_trader_config.py`, `src/config/city_config.py` (timezones/stations).

## Architecture Crosswalk (reuse vs new)
- **WebSocket transport** → reuse message shapes & auth from `docs/how-tos/kalshi_websockets_doc.md`; existing listener patterns in `scripts/kalshi_ws_recorder.py` & `scripts/live_ws_trader.py`. Target: `live_trading/websocket/{handler.py,order_book.py,market_state.py}`.
- **Inference** → reuse `models/inference/live_engine.py` (model loading + 30s cache) and `models/edge/{implied_temp,detector,classifier}` for market-implied temp + edge classifier. Target: `live_trading/inference.py` wraps these.
- **Risk/fees/sizing** → reuse `src/trading/{fees.py,risk.py}`; avoid duplicating EV math. Target: `live_trading/order_manager.py` uses these helpers.
- **Config** → reconcile new `live_trading/config.py` (aggressiveness dial, maker timeout) with `config/live_trader_config.py` (EV thresholds, hybrid model selection). Decide single source and derive the rest.
- **Logging/DB** → today: JSONL in `logs/live_trader/` and DB rows in `sim.live_orders`. Plan calls for `trading.sessions/decisions/orders`. Decide whether to dual-write during transition.
- **Models layout** → current `models/saved/{city}/...`; plan suggests `models/{city}/`. Either reorganize with shims or keep existing layout and update plan wording.

## Tasks for Junior Agent
- [ ] Align configs: define canonical config module (merge aggressiveness dial, Kelly, EV thresholds, hybrid model switches) and expose JSON override path.
- [ ] Build `live_trading/websocket/handler.py` + `order_book.py` using the doc’s client pattern (auth headers, reconnect/backoff, resubscribe, seq gap detection).
- [ ] Implement `live_trading/inference.py` that calls `LiveInferenceEngine` + edge detector/classifier; preserve caching interval.
- [ ] Implement `live_trading/order_manager.py` for maker→taker conversion (volume-weighted timeout), partial fills, position limits; reuse `fees.py` and `risk.py`.
- [ ] Implement `live_trading/db/{models.py,session_logger.py}` mirroring plan tables; optionally keep writing to `sim.live_orders` during transition.
- [ ] Wire `live_trading/edge_trader.py` main loop: per-city minute loop, decision logging, risk checks, order placement; include graceful shutdown.
- [ ] Add `config/trading.json` sample with sensible defaults (small size, dry-run on by default).
- [ ] Docker/systemd: add service entry to `docker-compose.yml` and a unit file for always-on; start in demo/small-size mode.
- [ ] Tests/smoke: unit-test orderbook builder, edge inference path, and maker→taker timeout math; run a WS connect + subscribe smoke in demo.

## Decisions Needed (ask lead)
1) Replace or run alongside current `scripts/live_ws_trader.py`? If replace, migrate logging/metrics; if parallel, pick different configs/ports/log paths.
2) DB logging: dual-write to existing `sim.*` + new `trading.*` during rollout, or cut over immediately?
3) Model directory: keep `models/saved/{city}/` and update plan wording, or move files and patch loaders?
4) Dry-run defaults: should the new service start in paper-trade mode until explicitly flipped?
5) Aggressiveness dial mapping: which knobs are user-controlled (confidence threshold, Kelly fraction, maker timeout) vs fixed?
6) Rollout scope: Chicago-only first? Demo WS first? Required controls for prod (max size, position caps)?

## Risks / Watchouts
- Timezone/climate-day boundaries must match `docs/how-tos/DATETIME_AND_API_REFERENCE.md` and `src/config/city_config.py`; centralize helper instead of ad-hoc math.
- Sequence gaps on WS: ensure auto-resubscribe and snapshot refresh to avoid stale books.
- Model cache invalidation: respect existing cooldown (30–60s) to avoid heavy DB/model churn.
- Maker→taker logic: volume-weighted timeout needs a clear volume source (trades channel vs last 30m volume); document the choice.
- Fee calculations: always use `src/trading/fees.py`; avoid hard-coded 7% approximations.
- Logging schema drift: if dual-writing, keep field names consistent to enable comparisons.

## Recommended Implementation Order
1) WS client + orderbook with reconnect/resubscribe, tested in demo; record raw stream for replay.
2) Inference wrapper that returns bracket probs + edge classifier decision, using cached models.
3) Config consolidation (single source of truth + JSON override + hot-reload hook if desired).
4) Order manager (position limits, maker→taker) using existing risk/fee utils.
5) Decision logging + DB models; optionally dual-write to current logs.
6) Main loop + CLI flags (`--live/--dry-run`, `--config-file`, `--city`).
7) Docker/systemd and smoke tests.

## Validation Checklist
- [ ] Demo WS connect + subscribe to ticker/orderbook_delta/trades/fill/market_positions; seq gap recovery verified.
- [ ] Inference returns in <100ms with cache; edge classifier outputs `should_trade`.
- [ ] Maker→taker timeout honors volume factor and aggressiveness dial.
- [ ] Position limits enforced per city and globally; max daily loss respected.
- [ ] Logs appear in DB table(s) and JSONL with complete context (markets, features, decisions, orders).
- [ ] Dry-run mode places no real orders; live mode uses guarded bet size caps.

## Notes for Future Work
- Consider replay harness using `scripts/kalshi_ws_recorder.py` logs to simulate WS feed for testing order manager and inference offline.
- Add small backtest comparing new live decisions vs historical candles to ensure parity with existing `open_maker` logic where applicable.
