# Agent Guidance for `weather_updated`

This file is the primary, repo-wide instruction set for any AI coding assistant working here.
It complements (and does not replace) existing permanent docs.

## 1. Required Context (read first)

Before changing code, always skim:

1. `README.md` — goals, architecture, strategy overview.
2. `CLAUDE.md` — personas and safety rules (treat as canonical).
3. `docs/permanent/AGENT_INSTRUCTIONS.md` — detailed invariants and workflows.
4. `docs/permanent/DATETIME_AND_API_REFERENCE.md` — timezones, event-day rules.
5. `docs/permanent/FILE_DICTIONARY_GUIDE.md` — intended module layout and refactor targets.
6. Any task‑specific doc under `docs/` that matches the area you’re editing (strategy, ingestion, inference).

If docs and code disagree, treat the docs as intended truth and flag the mismatch to the user before changing semantics.

## 2. Choose a Persona

Use the persona model defined in `CLAUDE.md`:

- **Kalshi Weather Quant** for domain/strategy/forecast/settlement logic.
- **Refactor Planner** for structural cleanup with behavior parity.
- **Dev Assistant** for general engineering and docs.

State which hat you’re wearing when starting a new task.

## 3. Non‑Negotiable Invariants

Do not violate these without explicit user approval:

- **Local vs UTC correctness** for every city/day (see `DATETIME_AND_API_REFERENCE.md`).
- **Historical vs current Visual Crossing forecasts are never mixed** in one helper or backtest path.
- **Bracket mapping is canonical** — reuse existing helpers (`find_bracket_for_temp`, `determine_winning_bracket`, etc.) rather than re‑implementing.
- **Backtests respect information timing** (no leakage from same‑day obs/settlement).
- **Live trading remains safe-by-default** (`--dry-run` default, tiny size, full logging).

## 4. Quality Checklist for Every Change

When editing or adding code, do all of the following:

### 4.1 Avoid Duplicate Logic

- Search for existing implementations before writing a new function/class.
- Prefer central utilities over ad‑hoc copies, especially for:
  - bracket/strike parsing
  - forecast lookup and fallback rules
  - fee/P&L and fill realism
  - time conversions and event‑day alignment
- If a near‑duplicate already exists, refactor to one canonical version or explain why not.

### 4.2 Detect Dead/Unused Code

- Look for files, functions, and branches that are never referenced or only referenced by legacy code.
- Use lightweight static checks (`ruff`, `mypy`) and repo search (`rg`) to confirm usage.
- Do not delete dead code unless asked; instead:
  - report candidates
  - suggest a safe removal/refactor plan
  - note any hidden dependencies (scripts, cron, systemd).

### 4.3 Keep Files Modular (target <500 lines)

- If a file is already large, avoid adding more bulk.
- When a file exceeds ~500 lines or mixes concerns, propose a split that matches the file dictionary:
  - extract pure helpers into `utils` modules
  - keep I/O/side effects in orchestrators
  - re‑export from `__init__.py` to preserve public APIs
- Known refactor hotspots: `open_maker/core.py`, `open_maker/utils.py`, `open_maker/manual_trade.py`, `open_maker/live_trader.py`.

### 4.4 Field/Schema Name Alignment

- Treat `src/db/models.py` as authoritative for DB column names and types.
- When building DataFrames or dicts, keep column keys consistent with ORM fields or documented schemas.
- If you must rename or alias fields, do it once in a dedicated mapping layer and document it.

### 4.5 Style, Types, and Tests

- Follow repo formatting/linting:
  - Black/Ruff line length 100, Python 3.11+ (`pyproject.toml`).
- Add type hints for new public functions and dataclasses.
- Prefer small, explicit helpers over long nested logic.
- Run `pytest` for any non‑trivial behavioral change; add tests if the area already has tests.

## 5. Docs and Strategy Review Expectations

- When working in an area, scan the relevant `docs/*.md`:
  - confirm the implementation matches current strategy intent
  - highlight outdated steps (legacy script names, old schema)
  - propose concrete doc updates when confusion is likely
- If you notice a strategic improvement (edge logic, risk controls, data quality), surface it as:
  - a brief rationale
  - a low‑risk experiment plan
  - clear separation between “idea” and “implemented change.”

## 6. Output to the User

For every task, end with:

1. What changed (files + high‑level behavior).
2. Any invariants checked.
3. Any duplicates/dead‑code/refactor candidates found.
4. Next steps or optional improvements.

Keep recommendations actionable and tied to the permanent docs.

