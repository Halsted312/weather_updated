---
plan_id: kalshi-data-ingestion-verification
created: 2025-11-30
status: in_progress
priority: critical
agent: kalshi-weather-quant
---

# Kalshi Data Refresh & Verification

## Objective
Ingest and validate complete Kalshi market + candle data (D-1 through event close) so hybrid backtests and live inference have reliable prices at market-clock timestamps.

## Context
- Supersedes prior Kalshi data tasks in `expand-kalshi-candles-schema.md` and unblocks zero-trade debugging in `debug-backtest-zero-trades.md`.
- Candle schema already expanded in code (`KalshiCandle1m`, `src/kalshi/schemas.py`, `scripts/backfill_kalshi_candles.py`); need DB confirmation and full backfill.
- Market-clock hybrid backtest (60-day window starting 2025-09-29) needs consistent 10:00 local candles and market metadata to simulate fills.
- Checkpointed ingestion helpers exist (`src/db/checkpoint.py`), rate limiting via `src/utils/rate_limiter.py`, and Kalshi client in `src/kalshi/client.py`.

## Tasks
- [ ] Baseline coverage check: quantify existing `kalshi.markets` and `kalshi.candles_1m` rows per city/event_date for 2023-01-01 → latest; flag gaps at D-1 10:00/D 10:00 local.
- [ ] Schema/state verification: confirm Alembic migration for expanded candles is applied; ensure `period_minutes`, bid/ask/trade OHLC, and trade stats columns are non-null where API supplies values.
- [ ] Market backfill run: execute `scripts/backfill_kalshi_markets.py` for all cities (at least 2023-01-01 → present) with checkpoints; capture inserted/updated counts.
- [ ] Candle backfill run: execute `scripts/backfill_kalshi_candles.py` for both sources (`api_event`, `trades` if needed) across the same window; resume-friendly with checkpoints.
- [ ] Data quality validation: sanity checks for crossed spreads, missing buckets, UTC/local alignment at open (D-1 10:00), and fillable spreads vs MAKER_FILL_PROBABILITY.
- [ ] Backtest readiness check: rerun a small hybrid slice (e.g., 5–10 events per city) to verify >0 trades and available candles at decision times.
- [ ] Documentation & sign-off: summarize coverage stats, queries used, and any remaining gaps; update this plan and related docs if schema fixes were required.

## Files to Create/Modify
| Action | Path | Notes |
|--------|------|-------|
| MODIFY | `scripts/backfill_kalshi_markets.py` | Only if coverage issues require logic tweaks (date window, checkpointing, parsing). |
| MODIFY | `scripts/backfill_kalshi_candles.py` | Only if ingest bugs found (bucket_start calc, source handling, retries). |
| MODIFY | `migrations/` (new file) | Only if DB schema is missing expanded candle columns; otherwise no-op. |
| MODIFY | `docs/KALSHI_CANDLESTICK_SCHEMA_REVIEW.md` | Add verification results and any schema clarifications. |
| MODIFY | `docs/SESSION_SUMMARY_*.md` or plan sign-off | Record ingestion run outcomes and metrics. |

## Technical Details
- Candle schema: `kalshi.candles_1m` primary key `(ticker, bucket_start, source)`, includes `period_minutes`, full yes_bid/yes_ask OHLC, trade OHLC + stats (`trade_mean/previous/min/max`), `volume`, `open_interest`.
- API mapping: `Candle.end_period_ts` → `bucket_start = end_period_ts - 60` (UTC). `period_minutes` defaults to 1; trades aggregation path also writes these fields.
- Sources: `api_event` (preferred) and optional `trades` aggregation; ensure dedupe by source on upsert.
- Time alignment: markets open ~10:00 local on D-1; backtest requires candles at D-1 10:00 and D 10:00 local → convert to UTC per city timezone in validation queries.
- Safety: use `get_settings()` for credentials, `KALSHI_LIMITER` for rate limiting, and checkpoints (`meta.ingestion_checkpoint`) for resume.
- Validation queries: counts per city/lead day, null/negative spread checks (`yes_bid_high <= yes_ask_low`), bucket continuity around opens, and parity between markets and candles per ticker/day.

## Completion Criteria
- [ ] Expanded candle schema confirmed applied; no missing columns.
- [ ] Markets backfilled through latest available date for all six cities with recorded counts.
- [ ] Candles backfilled for same window with non-null OHLC where API provides; no widespread crossed spreads or missing opens.
- [ ] Coverage at D-1 10:00 and D 10:00 local verified for hybrid backtest window (2025-09-29 to 2025-11-27) and recent dates.
- [ ] Hybrid backtest smoke slice produces expected trade counts (>0 per city when edge exists).
- [ ] Plan sign-off updated with metrics, queries, and any follow-up actions.

## Sign-off Log

### 2025-11-30 19:22 CST
**Status**: In progress – plan scaffolded  
**Last completed**:
- Read project instructions (CLAUDE.md, README.md, agent profiles).
- Reviewed active plans, Kalshi ingestion scripts, and expanded candle schema in code.
- Skimmed permanent docs for datetime/API and file structure guidance.

**Next steps**:
1. Run coverage queries for `kalshi.markets` and `kalshi.candles_1m` (per city, per event_date, D-1/D open times).
2. Confirm Alembic migration state for candle schema; decide if migration patch needed.
3. Prepare/resume market and candle backfill commands with checkpoints; capture row counts.

**Blockers**: None noted yet (assumes DB connectivity and Kalshi creds available).
