# reverse_engineer_kalshi_temps.md

## Purpose

This document describes a **rules-based reverse-engineering harness** to map:

> **Visual Crossing 5-minute temperatures → actual Kalshi/NWS daily high temperatures (integer °F)**

The goal is to:

1. Use **real settlement highs** (NWS CLI / Kalshi) as the referee.
2. Apply a **family of deterministic rules** to the Visual Crossing (VC) 5-minute time series.
3. Measure which rule best reproduces the actual daily high.
4. Surface the **rare “edge-case” days** where all simple rules fail, so we can inspect and refine.

This is for **backtesting and live trading** on Kalshi weather markets, where the only thing that matters is:

> _“Which integer °F bracket does Kalshi settle on for this day & city?”_


---

## Background & Assumptions

### 1. What Kalshi settles on

- Weather markets settle on the **daily high temperature reported by NWS climate products** (CF6/CLI) for a specific station (e.g., KMDW for Chicago Midway, KAUS for Austin Bergstrom).
- These climate products report a **single integer °F daily maximum** per day and station.

For our purposes:

- `tmax_final_f` in `wx.settlement` (from IEM/NCEI) is a very good proxy for **Kalshi’s final settlement high**.
- We can optionally add a `kalshi.settlement` table later and confirm equivalence, but we don’t need it to build the harness.

### 2. What Visual Crossing gives us

Visual Crossing (VC):

- Ingests raw station data (NOAA ISD, etc.) which are typically in **0.1°C units**.
- Blends multiple nearby stations, interpolates in space and time, and then converts to Fahrenheit for `unitGroup=us`.
- Returns **sub-hourly (5-minute) temperatures as floats in °F**, often with one decimal place.

Key implications:

- VC 5-minute temps are a **smooth, continuous approximation** of the actual temperature at a location.
- They are **not** the same as the station’s internal running averages that NWS uses to compute the CLI high.
- There will always be a small irreducible error (station choice, interpolation, QC, rounding).

### 3. What we can realistically learn

We cannot perfectly reconstruct:

- ASOS internal 2–5 second sampling,
- 1-minute and 5-minute running averages,
- NWS QC / spike rejection.

But we **can**:

1. Define a small, interpretable **rule family** `R(series) → integer °F`.
2. Run these rules across historical days.
3. Measure how often each rule matches the real settlement.
4. Identify **outliers** to refine rules or handle manually.

Our target is to pick a simple rule that:

- Has very high exact-match rate (ideally > 99%),
- Is easy to compute live from VC 5-minute series,
- Is robust and transparent for debugging.


---

## Data Model & Inputs

### 1. Visual Crossing 5-minute temperatures

Assumed Timescale table (adapt to actual names):

```sql
CREATE TABLE wx.vc_minute_weather (
    city      TEXT            NOT NULL,
    ts_utc    TIMESTAMPTZ     NOT NULL,
    temp_f    DOUBLE PRECISION,  -- Visual Crossing 5-min temperature in °F
    -- optional: temp_c DOUBLE PRECISION
    PRIMARY KEY (city, ts_utc)
);
````

Properties:

* One row every 5 minutes per city (approximate).
* `temp_f` typically has 1 decimal place.

### 2. Settlement table (Kalshi/NWS daily high)

We’ll treat `wx.settlement` as the canonical NWS high and clone it into a Kalshi-specific alias:

```sql
CREATE TABLE IF NOT EXISTS wx.settlement_kalshi (
    city    TEXT     NOT NULL,
    day     DATE     NOT NULL,
    tmax_f  SMALLINT NOT NULL,
    PRIMARY KEY (city, day)
);
```

Backfill from existing settlement data:

```sql
INSERT INTO wx.settlement_kalshi (city, day, tmax_f)
SELECT city, date_utc AS day, tmax_final AS tmax_f
FROM wx.settlement
ON CONFLICT (city, day) DO NOTHING;
```

Later, if we ingest actual Kalshi settlements, we can:

* Either update `wx.settlement_kalshi`,
* Or introduce `kalshi.settlement` and point the harness there.

### 3. City metadata (station & timezone)

We need a mapping from `city` (our key) to:

* Station ID (for documentation),
* IANA timezone (for daily window alignment).

Example Python structure:

```python
CITY_META = {
    "chicago":      {"station": "KMDW", "timezone": "America/Chicago"},
    "austin":       {"station": "KAUS", "timezone": "America/Chicago"},
    "miami":        {"station": "KMIA", "timezone": "America/New_York"},
    "denver":       {"station": "KDEN", "timezone": "America/Denver"},
    "los_angeles":  {"station": "KLAX", "timezone": "America/Los_Angeles"},
    "philadelphia": {"station": "KPHL", "timezone": "America/New_York"},
}
```

For the initial harness, we can define the CLI day as:

> midnight → midnight **in local civil time** for that city.

Later refinements (LST vs DST) are possible but not required immediately.

---

## High-Level Architecture

We will implement a **standalone reverse-engineering harness** with:

1. A **core module** `analysis/reverse_engineering.py`:

   * Data structures (`DaySeries`, `RuleStats`),
   * A set of deterministic rule functions,
   * Utility to load data from Timescale.

2. A **CLI script** `scripts/reverse_engineer_settlement.py`:

   * Connects to Timescale,
   * Runs rules across historical dates/cities,
   * Prints rule accuracy summary,
   * Writes a CSV of mismatch days.

3. (Later) A **follow-up integration**:

   * Once we pick a winning rule, we can:

     * Persist its prediction per day,
     * Use it as a feature in the trading stack,
     * Optionally encode a small “rounding range” or confidence band.

---

## Rule Family (Deterministic, No Probabilities)

For each `(city, day)`:

* Let `temps_f[t]` be the VC 5-minute °F series in the CLI window.
* Let `T_settle` be the true daily high °F from `wx.settlement_kalshi`.

We define multiple simple rules `R_i`:

### Rule 1 – `max_round`: round(max)

> “Take the maximum VC °F and round to nearest integer.”

```python
def rule_max_round(temps_f: list[float]) -> int | None:
    if not temps_f:
        return None
    return int(round(max(temps_f)))
```

### Rule 2 – `max_floor`: floor(max)

> “Take the maximum VC °F and floor to integer.”

```python
import math

def rule_max_floor(temps_f: list[float]) -> int | None:
    if not temps_f:
        return None
    return int(math.floor(max(temps_f)))
```

### Rule 3 – `c_first`: Celsius-first rounding path

> “Convert each sample to °C, round to nearest °C, convert back to °F, then take max.”

This models a plausible C↔F pipeline where some rounding happens in Celsius, which can create 1°F edge cases.

```python
def rule_c_first(temps_f: list[float]) -> int | None:
    if not temps_f:
        return None
    candidates: list[int] = []
    for f in temps_f:
        c = (f - 32.0) * 5.0 / 9.0
        c_rounded = round(c)
        f_from_c = round(c_rounded * 9.0 / 5.0 + 32.0)
        candidates.append(int(f_from_c))
    return max(candidates)
```

### Rule 4 – `plateau_10min`: spike vs plateau (10-minute support)

> “Pick the highest integer °F for which we see at least 10 minutes of support (2 consecutive 5-min samples) above (k − ε).”

This captures your intuition:

* A **single 5-minute spike** to 93.1°F might be treated differently than
* A **2-hour plateau** around 92–93°F.

```python
def rule_plateau_10min(
    temps_f: list[float],
    min_consecutive_steps: int = 2,   # 2 * 5min = 10min
    eps_f: float = 0.2,
) -> int | None:
    if not temps_f:
        return None

    max_f = max(temps_f)
    # search k in a small band around max_f
    k_min = int(math.floor(max_f)) - 2
    k_max = int(math.ceil(max_f)) + 2

    best_k: int | None = None

    for k in range(k_min, k_max + 1):
        run = 0
        longest_run = 0
        for f in temps_f:
            if f >= k - eps_f:
                run += 1
                longest_run = max(longest_run, run)
            else:
                run = 0

        if longest_run >= min_consecutive_steps:
            if best_k is None or k > best_k:
                best_k = k

    if best_k is None:
        return int(round(max_f))
    return best_k
```

### Rule 5 – `plateau_20min`: stricter plateau (20-minute support)

Same as above but requiring 4 consecutive steps (20 minutes):

```python
def rule_plateau_20min(temps_f: list[float]) -> int | None:
    return rule_plateau_10min(temps_f, min_consecutive_steps=4, eps_f=0.2)
```

> **Note:** We don’t have to guess which rule is “right.”
> The harness will **evaluate all of them** and we choose based on empirical accuracy.

---

## Core Data Structures

Create a module: `analysis/reverse_engineering.py`.

### `DaySeries`

Represents one `(city, day)` pair:

```python
from dataclasses import dataclass
from datetime import date
from typing import List

@dataclass
class DaySeries:
    city: str
    day: date
    temps_f: List[float]   # VC 5-min °F temps in the CLI window
    settle_f: int          # observed settlement high (tmax_f) from wx.settlement_kalshi
```

### `RuleStats`

Track performance per rule:

```python
from dataclasses import dataclass

@dataclass
class RuleStats:
    name: str
    total: int = 0
    matches: int = 0
    off_by_1: int = 0
    off_by_2plus: int = 0

    def update(self, pred: int | None, actual: int):
        if pred is None:
            return
        self.total += 1
        diff = pred - actual
        if diff == 0:
            self.matches += 1
        elif abs(diff) == 1:
            self.off_by_1 += 1
        else:
            self.off_by_2plus += 1

    @property
    def accuracy(self) -> float:
        return self.matches / self.total if self.total else 0.0
```

---

## Loading Data from Timescale

We assume:

* PostgreSQL / Timescale,
* `DATABASE_URL` environment variable,
* `psycopg` (v3) for Python.

```python
import os
import psycopg
from datetime import datetime, date, time, timedelta
from typing import Optional, List, Tuple

def get_conn():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Set DATABASE_URL env var")
    return psycopg.connect(dsn)
```

### Fetch settlements

```python
def fetch_settlements(
    conn,
    cities: Optional[List[str]] = None,
    start_day: Optional[date] = None,
    end_day: Optional[date] = None,
) -> List[Tuple[str, date, int]]:
    """
    Load (city, day, tmax_f) from wx.settlement_kalshi.
    """
    where_clauses = []
    params: List[object] = []

    if cities:
        where_clauses.append("city = ANY(%s)")
        params.append(cities)
    if start_day:
        where_clauses.append("day >= %s")
        params.append(start_day)
    if end_day:
        where_clauses.append("day <= %s")
        params.append(end_day)

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    sql = f"""
        SELECT city, day, tmax_f
        FROM wx.settlement_kalshi
        {where_sql}
        ORDER BY city, day
    """

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [(r[0], r[1], r[2]) for r in rows]
```

### Fetch VC 5-minute temps for a day

For now, we treat the CLI window as UTC midnight→midnight.
Later we can refine to true local timezone (using `CITY_META` and `pytz`).

```python
def fetch_vc_temps_for_day(conn, city: str, day: date) -> List[float]:
    """
    Pull all Visual Crossing 5-min temps for a given (city, day).

    First pass: treat day as UTC midnight->midnight.
    Later we can adjust for local time windows if needed.
    """
    start_dt = datetime.combine(day, time(0, 0))
    end_dt = start_dt + timedelta(days=1)

    sql = """
        SELECT temp_f
        FROM wx.vc_minute_weather
        WHERE city = %s
          AND ts_utc >= %s
          AND ts_utc <  %s
        ORDER BY ts_utc
    """
    with conn.cursor() as cur:
        cur.execute(sql, (city, start_dt, end_dt))
        rows = cur.fetchall()

    return [r[0] for r in rows if r[0] is not None]
```

### Constructing `DaySeries`

```python
from typing import Optional

def load_day_series(
    conn,
    city: str,
    day: date,
) -> Optional[DaySeries]:
    """
    Load the settlement value and VC 5-min temps for this (city, day).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT tmax_f
            FROM wx.settlement_kalshi
            WHERE city = %s
              AND day  = %s
            """,
            (city, day),
        )
        row = cur.fetchone()

    if not row:
        return None

    settle_f = int(row[0])
    temps_f = fetch_vc_temps_for_day(conn, city, day)
    if not temps_f:
        return None

    return DaySeries(city=city, day=day, temps_f=temps_f, settle_f=settle_f)
```

---

## Evaluation Harness

We want a function to:

* Loop over all `(city, day)` settlements,
* Apply each rule,
* Update `RuleStats`,
* Record mismatches to a list for later CSV export.

```python
from typing import Dict, Callable, List

RuleFn = Callable[[list[float]], int | None]

def evaluate_rules_over_range(
    conn,
    cities: Optional[List[str]] = None,
    start_day: Optional[date] = None,
    end_day: Optional[date] = None,
):
    settlements = fetch_settlements(conn, cities, start_day, end_day)

    rules: Dict[str, RuleFn] = {
        "max_round":     rule_max_round,
        "max_floor":     rule_max_floor,
        "c_first":       rule_c_first,
        "plateau_10min": rule_plateau_10min,
        "plateau_20min": rule_plateau_20min,
    }

    stats: Dict[str, RuleStats] = {
        name: RuleStats(name=name) for name in rules.keys()
    }

    mismatches: List[dict] = []

    for city, day, settle_f in settlements:
        temps_f = fetch_vc_temps_for_day(conn, city, day)
        if not temps_f:
            continue

        for name, fn in rules.items():
            pred = fn(temps_f)
            if pred is None:
                continue

            stats[name].update(pred, settle_f)

            if pred != settle_f:
                mismatches.append(
                    {
                        "city": city,
                        "day": day.isoformat(),
                        "rule": name,
                        "settle_f": settle_f,
                        "pred_f": pred,
                        "diff": pred - settle_f,
                        "vc_max_f": max(temps_f),
                    }
                )

    return stats, mismatches
```

### Printing summary & writing mismatches

```python
def print_stats(stats: Dict[str, RuleStats]) -> None:
    print("=== Rule accuracy summary ===")
    for name, st in sorted(stats.items(), key=lambda kv: kv[1].accuracy, reverse=True):
        if st.total == 0:
            continue
        print(
            f"{name:15s}  "
            f"total={st.total:4d}  "
            f"acc={st.accuracy:6.3f}  "
            f"matches={st.matches:4d}  "
            f"off_by_1={st.off_by_1:4d}  "
            f"off_by_2+={st.off_by_2plus:4d}"
        )


def write_mismatches_csv(path: str, mismatches: List[dict]) -> None:
    import csv, os
    if not mismatches:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = ["city", "day", "rule", "settle_f", "pred_f", "diff", "vc_max_f"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(mismatches)
```

---

## CLI Script: `scripts/reverse_engineer_settlement.py`

This is a thin wrapper on top of the core harness.

```python
#!/usr/bin/env python

import argparse
from datetime import date
from typing import Optional, List

from analysis.reverse_engineering import (
    get_conn,
    evaluate_rules_over_range,
    print_stats,
    write_mismatches_csv,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reverse engineer mapping from VC 5-min temps to Kalshi/NWS daily highs."
    )
    p.add_argument("--city", action="append", help="City key (can be repeated)")
    p.add_argument("--start-day", type=str, help="Start date YYYY-MM-DD")
    p.add_argument("--end-day", type=str, help="End date YYYY-MM-DD")
    p.add_argument(
        "--out-csv",
        type=str,
        default="reports/settlement_rule_mismatches.csv",
        help="CSV path for mismatch days",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cities: Optional[List[str]] = args.city if args.city else None
    start_day = date.fromisoformat(args.start_day) if args.start_day else None
    end_day = date.fromisoformat(args.end_day) if args.end_day else None

    conn = get_conn()
    try:
        stats, mismatches = evaluate_rules_over_range(
            conn,
            cities=cities,
            start_day=start_day,
            end_day=end_day,
        )
    finally:
        conn.close()

    print_stats(stats)
    write_mismatches_csv(args.out_csv, mismatches)
    print(f"Wrote {len(mismatches)} mismatches to {args.out_c
```
