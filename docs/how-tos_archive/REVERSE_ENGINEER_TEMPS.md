# REVERSE_ENGINEER_TEMPS.md  
Reverse-engineering Kalshi / NWS Daily Highs from Visual Crossing 5-Minute Data

**Role of this document:**  
This file is a design + reference doc for a *deterministic* reverse-engineering harness that:

1. Uses **Kalshi/NWS settled daily highs** as the oracle.
2. Uses **Visual Crossing 5-minute temperatures** as the input signal.
3. Tests a *family of rules* that map the 5-minute VC series → an integer °F daily high.
4. Picks the best rule(s) and flags **rare edge cases** for manual inspection.

The target audience is the coding agent (Claude in VS Code) and future you when you revisit this logic.

---

## 0. Big Picture

**Goal:**  
Given a day’s full 5-minute Visual Crossing temperature series for a Kalshi weather city, predict exactly which **integer °F daily high** (the NWS / Kalshi settlement value) will be used.

We want this to be:

- **Simple:** deterministic rules, not a stochastic model.
- **Robust:** uses realistic assumptions about NWS / ASOS / Visual Crossing pipelines.
- **Accurate:** matches historical settled highs on as many days as possible; explicitly surfaces the rare exceptions.

Core idea:

> Treat the **settled high** (from NWS CLI / IEM) as the oracle, and search over a small family of deterministic rules that map the Visual Crossing 5-minute temps to that integer. Let history tell us which rule is “right”.

---

## 1. Ground Truth: What Kalshi and NWS Actually Do (Conceptually)

### 1.1 Kalshi settlement

- For each weather market (e.g., Chicago high temp at Midway), Kalshi’s rulebooks say:
  - The underlying is the **maximum temperature for the date published in the NWS Daily Climate Report** (CLI/CF6) for the specified station.
- That CLI report (for a station like KMDW, KAUS, etc.) contains **one integer °F max** for the 24-hour climate day.

So the ground-truth variable we care about is:

> `T_settle_F` = **CLI daily max in integer °F** at the Kalshi station for that date.

In this repo, you already have an excellent proxy for this in `wx.settlement` (from IEM/NCEI daily climate data). Treat `tmax_final` as `T_settle_F` unless/until you ingest direct Kalshi settlements.

### 1.2 NWS / ASOS temperature behavior (why edge cases exist)

From NWS / ASOS documentation (summarized):

- ASOS temperature sensors sample air temperature roughly every **2–5 seconds**.
- These are averaged into **1-minute values**, and then into **5-minute running averages** internally.
- Daily max/min used for climate products are based on an internal averaged series and stored in **integer °F** for the climate record (CLI/CF6).

Separately:

- Many archives (e.g., ISD / GHCN) store temps in **0.1°C** increments and may convert back/forth between °F and °C with rounding.
- Double-rounding (F→C →round→ back to F) can introduce **±1°F discrepancies** vs a naïve °F rounding path.

This is why:

- Visual Crossing (smooth, C-native, blended) and
- NWS CLI (station-specific, integer-F, ASOS-style)

can differ by 0–2°F on the daily max, and why rounding order matters.

**Key takeaway:**  
There is a small but real **rounding / representation ambiguity** (typically ±1°F) between VC’s float °F series and the integer °F CLI daily high.

---

## 2. Data & Alignment Assumptions (for the Harness)

We assume the following tables exist (or will be created) in Timescale:

### 2.1 Visual Crossing 5-minute temps

Hypertable, e.g. `wx.vc_minute_weather`:

```sql
CREATE TABLE IF NOT EXISTS wx.vc_minute_weather (
    city       TEXT NOT NULL,         -- 'chicago', 'austin', etc.
    ts_utc     TIMESTAMPTZ NOT NULL,  -- 5-min timestamp (UTC)
    temp_f     DOUBLE PRECISION,      -- Visual Crossing °F
    temp_c     DOUBLE PRECISION,      -- (optional) Visual Crossing °C
    -- other fields as needed
    PRIMARY KEY (city, ts_utc)
);
````

Assumption: 5-minute cadence from VC for all cities (or at least dense enough).

### 2.2 Settlement / CLI daily highs

Table, e.g. `wx.settlement`:

```sql
CREATE TABLE IF NOT EXISTS wx.settlement (
    city           TEXT NOT NULL,
    day            DATE NOT NULL,      -- CLI/Kalshi market date
    tmax_final_f   SMALLINT NOT NULL,  -- integer °F daily high (IEM/NCEI CLI)
    -- possibly other fields: source flags, tmax_iem, tmax_ncei, etc.
    PRIMARY KEY (city, day)
);
```

For reverse-engineering, we treat:

> `wx.settlement.tmax_final_f` as the “settled” `T_settle_F`.

If later you ingest **actual Kalshi settlements**, you can add a `kalshi.settlement` table and compare, but this harness works fine with `wx.settlement` alone.

### 2.3 City → Timezone mapping

For each of the six Kalshi weather cities:

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

For now we approximate the **climate day** as:

> local civil midnight → local civil midnight for that station’s timezone.

(You can refine later to “Local Standard Time” if you find systematic DST issues.)

---

## 3. Core Idea: Deterministic Rule Family

For each `(city, day)` with:

* `temps_f[t]` = the 5-minute VC °F temps over that day’s CLI window.
* `T_settle_F` = `wx.settlement.tmax_final_f` (the integer daily high).

We define several **candidate rules** `R_k(temps_f) → integer °F` that try to predict `T_settle_F`.

We then:

* Run all rules over all historical days.
* Measure accuracy (exact matches, off-by-1, etc.).
* Identify which rule(s) best approximate reality.
* Dump a CSV of **mismatch days** for inspection.

We’re not fitting parameters via ML here. We’re doing a structured **rule search** over:

* Basic rounding variants
* Simple plateau / spike-filter logic

All rules are deliberately simple, testable, and transparent.

---

## 4. Rule Definitions

Below we define a starting rule family. They can live in a Python module like `analysis/reverse_engineering.py`.

### 4.1 Dataclass for per-day series

```python
from dataclasses import dataclass
from datetime import date
from typing import List

@dataclass
class DaySeries:
    city: str
    day: date
    temps_f: List[float]   # 5-min VC temps in °F for the CLI window
    settle_f: int          # integer °F daily high from wx.settlement
```

### 4.2 Baseline rules

#### R1 – `max_round_f`

> Take the maximum VC °F and round to nearest integer.

```python
import math
from typing import Optional

def rule_max_round_f(day: DaySeries) -> Optional[int]:
    if not day.temps_f:
        return None
    return int(round(max(day.temps_f)))
```

#### R2 – `max_of_rounded`

> Round each 5-minute temp to integer °F, then take the max.

```python
def rule_max_of_rounded(day: DaySeries) -> Optional[int]:
    if not day.temps_f:
        return None
    rounded = [int(round(x)) for x in day.temps_f]
    return max(rounded)
```

#### R3 – `ceil_max`

> Ceiling of the maximum VC °F.

```python
def rule_ceil_max(day: DaySeries) -> Optional[int]:
    if not day.temps_f:
        return None
    return int(math.ceil(max(day.temps_f)))
```

#### R4 – `floor_max`

> Floor of the maximum VC °F.

```python
def rule_floor_max(day: DaySeries) -> Optional[int]:
    if not day.temps_f:
        return None
    return int(math.floor(max(day.temps_f)))
```

### 4.3 Plateau / spike-filter rules

These are meant to capture intuition like:

* “A single 5-minute spike might not reflect the official high.”
* “A 2-hour plateau around 90°F is more likely to define the daily high.”

#### R5 – `plateau(min_minutes_plateau)`

> Use the highest integer °F that appears in a contiguous run of at least `min_minutes_plateau` minutes (with 5-minute resolution). Fall back to `max_of_rounded` if none qualify.

```python
def rule_plateau(day: DaySeries, min_minutes_plateau: int = 20) -> Optional[int]:
    """
    Use the highest integer F that appears in a contiguous run of at least
    `min_minutes_plateau` minutes. Temps are assumed 5-min apart.
    Fallback: max_of_rounded.
    """
    if not day.temps_f:
        return None

    step_minutes = 5
    rounded = [int(round(x)) for x in day.temps_f]
    if len(rounded) == 1:
        return rounded[0]

    best_k = None
    i = 0
    n = len(rounded)

    while i < n:
        k = rounded[i]
        j = i + 1
        while j < n and rounded[j] == k:
            j += 1
        duration = (j - i) * step_minutes
        if duration >= min_minutes_plateau:
            if best_k is None or k > best_k:
                best_k = k
        i = j

    if best_k is not None:
        return best_k

    # Fallback if no plateau meets the duration requirement
    return rule_max_of_rounded(day)
```

You can instantiate variations:

* `plateau_10min` → `min_minutes_plateau=10`
* `plateau_20min` → `min_minutes_plateau=20`
* `plateau_30min` → `min_minutes_plateau=30`

#### R6 – `ignore_singletons`

> Ignore integer °F values that occur only once in the day. Take the max of the remaining; fall back to max if everything is a singleton.

```python
from collections import Counter

def rule_ignore_singletons(day: DaySeries) -> Optional[int]:
    if not day.temps_f:
        return None

    rounded = [int(round(x)) for x in day.temps_f]
    counts = Counter(rounded)
    candidates = [k for k, c in counts.items() if c >= 2]

    if candidates:
        return max(candidates)
    return max(rounded)
```

### 4.4 (Optional) Celsius-first rule

If you store `temp_c` in the VC minute table, you can also test a **C→F rounding path** that mimics C-native archives:

```python
def rule_c_first(day_c: DaySeries, temps_c: List[float]) -> Optional[int]:
    if not temps_c:
        return None
    max_c = max(temps_c)
    c_rounded = round(max_c)
    f_from_c = round(c_rounded * 9.0 / 5.0 + 32.0)
    return int(f_from_c)
```

Claude can wire this if/when `temp_c` is easily accessible.

---

## 5. Loading Data & Day Alignment

We need a way to:

1. Get the CLI window (start & end UTC) for a given `(city, day)`.
2. Load Visual Crossing temps for that window.
3. Load the settlement temp for that `(city, day)`.

### 5.1 CLI window in UTC

Use IANA timezones and treat the CLI day as **local midnight → next local midnight**:

```python
from datetime import datetime, date, timedelta
import pytz

CITY_META = {
    "chicago":      {"timezone": "America/Chicago"},
    "austin":       {"timezone": "America/Chicago"},
    "miami":        {"timezone": "America/New_York"},
    "denver":       {"timezone": "America/Denver"},
    "los_angeles":  {"timezone": "America/Los_Angeles"},
    "philadelphia": {"timezone": "America/New_York"},
}

def get_cli_window_utc(city: str, day: date):
    """
    Return (start_utc, end_utc) for the CLI day, interpreted as
    midnight-to-midnight in the city's local timezone.
    """
    tzname = CITY_META[city]["timezone"]
    tz = pytz.timezone(tzname)

    local_start = tz.localize(datetime(day.year, day.month, day.day, 0, 0, 0))
    local_end = local_start + timedelta(days=1)

    start_utc = local_start.astimezone(pytz.UTC)
    end_utc   = local_end.astimezone(pytz.UTC)
    return start_utc, end_utc
```

### 5.2 Loading a `DaySeries` from Timescale

Using SQLAlchemy or psycopg; here’s a conceptual version with SQLAlchemy’s `session` and `text`:

```python
from typing import Optional
from sqlalchemy import text

def load_day_series(session, city: str, day: date) -> Optional[DaySeries]:
    # 1) Settlement value (ground truth)
    settle_row = session.execute(
        text("""
            SELECT tmax_final_f
            FROM wx.settlement
            WHERE city = :city AND day = :day
        """),
        {"city": city, "day": day},
    ).fetchone()

    if not settle_row or settle_row[0] is None:
        return None

    settle_f = int(settle_row[0])

    # 2) Visual Crossing temps in CLI window
    start_utc, end_utc = get_cli_window_utc(city, day)

    rows = session.execute(
        text("""
            SELECT temp_f
            FROM wx.vc_minute_weather
            WHERE city = :city
              AND ts_utc >= :start_utc
              AND ts_utc <  :end_utc
            ORDER BY ts_utc
        """),
        {"city": city, "start_utc": start_utc, "end_utc": end_utc},
    ).fetchall()

    temps_f = [r[0] for r in rows if r[0] is not None]
    if not temps_f:
        return None

    return DaySeries(city=city, day=day, temps_f=temps_f, settle_f=settle_f)
```

---

## 6. Evaluation Harness

We want:

* A reusable **stats object** per rule.
* A way to iterate over many days and compute:

  * total days evaluated,
  * exact matches,
  * mean absolute error,
  * plus a list of mismatches for inspection.

### 6.1 RuleStats

```python
from dataclasses import dataclass

@dataclass
class RuleStats:
    name: str
    total: int = 0
    matches: int = 0
    sum_abs_error: int = 0

    def update(self, pred: int | None, actual: int):
        if pred is None:
            return
        self.total += 1
        if pred == actual:
            self.matches += 1
        self.sum_abs_error += abs(pred - actual)

    @property
    def accuracy(self) -> float:
        return self.matches / self.total if self.total else 0.0

    @property
    def mean_abs_error(self) -> float:
        return self.sum_abs_error / self.total if self.total else 0.0
```

### 6.2 Evaluating rules over a date range

```python
from datetime import timedelta
from typing import Dict, List, Tuple

def evaluate_rules_over_range(
    session,
    city: str,
    start_day: date,
    end_day: date,
):
    rules: Dict[str, callable] = {
        "max_round_f":       rule_max_round_f,
        "max_of_rounded":    rule_max_of_rounded,
        "ceil_max":          rule_ceil_max,
        "floor_max":         rule_floor_max,
        "plateau_20min":     lambda d: rule_plateau(d, min_minutes_plateau=20),
        "ignore_singletons": rule_ignore_singletons,
    }

    stats = {name: RuleStats(name) for name in rules}
    mismatches: List[dict] = []

    cur = start_day
    while cur <= end_day:
        day_series = load_day_series(session, city, cur)
        if day_series is None:
            cur += timedelta(days=1)
            continue

        # Apply all rules
        for name, fn in rules.items():
            pred = fn(day_series)
            stats[name].update(pred, day_series.settle_f)

        # Pick a baseline rule to track mismatches explicitly
        baseline_pred = rules["max_of_rounded"](day_series)
        if baseline_pred is not None and baseline_pred != day_series.settle_f:
            mismatches.append(
                {
                    "city": city,
                    "day": cur.isoformat(),
                    "settle_f": day_series.settle_f,
                    "baseline_rule": "max_of_rounded",
                    "baseline_pred": baseline_pred,
                    "error": baseline_pred - day_series.settle_f,
                    "vc_max_f": max(day_series.temps_f),
                }
            )

        cur += timedelta(days=1)

    return stats, mismatches
```

### 6.3 Reporting & CSV dump

```python
def print_stats(stats: Dict[str, RuleStats]):
    print("=== Rule performance ===")
    for name, st in sorted(stats.items(), key=lambda kv: kv[1].accuracy, reverse=True):
        print(
            f"{name:18s}  "
            f"acc={st.accuracy:.4f}  "
            f"matches={st.matches:4d}/{st.total:4d}  "
            f"MAE={st.mean_abs_error:.3f}"
        )
```

```python
import csv
import os

def write_mismatches_csv(filepath: str, mismatches: List[dict]):
    if not mismatches:
        print("No mismatches to write.")
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = sorted(mismatches[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mismatches)
    print(f"Wrote {len(mismatches)} mismatches to {filepath}")
```

---

## 7. CLI Script Wrapper

Create a script, e.g. `scripts/rev_eng_cli_from_vc.py`, that:

* Reads DB connection info from `DATABASE_URL`.
* Takes `--city`, `--start`, `--end`, `--out` arguments.
* Runs the evaluation and prints summary + writes mismatches.

Example skeleton:

```python
#!/usr/bin/env python

import os
from datetime import date
from argparse import ArgumentParser

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from analysis.reverse_engineering import (
    evaluate_rules_over_range,
    print_stats,
    write_mismatches_csv,
)

def parse_args():
    p = ArgumentParser()
    p.add_argument("--city", required=True, help="city key, e.g. 'chicago'")
    p.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="end date YYYY-MM-DD")
    p.add_argument("--out", default="reports/rev_eng_mismatches.csv",
                   help="output CSV for mismatches")
    return p.parse_args()

def main():
    args = parse_args()

    start_day = date.fromisoformat(args.start)
    end_day   = date.fromisoformat(args.end)

    db_url = os.environ["DATABASE_URL"]
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        stats, mismatches = evaluate_rules_over_range(
            session=session,
            city=args.city,
            start_day=start_day,
            end_day=end_day,
        )
        print_stats(stats)
        write_mismatches_csv(args.out, mismatches)

if __name__ == "__main__":
    main()
```

---

## 8. How This Helps in Practice

Once this harness is implemented and runs over your full history:

1. **You get a scoreboard**: which rule best approximates `tmax_final_f` for each city:

   * Overall accuracy (% exact matches),
   * Mean absolute error.
2. **You get a small list of “weird days”**:

   * The CSV of mismatches for the baseline rule (and you can extend to log mismatches per rule).
   * For those days, you can plot the 5-minute VC series and visually see:

     * Single-sample spikes,
     * Long plateaus,
     * Cases where VC is systematically off from the station.
3. From that, you can:

   * Choose a single “canonical” rule per city (e.g., `max_of_rounded` or `plateau_20min`).
   * Or implement a tiered logic (e.g., plateau rule first; fallback to max_of_rounded).

Ultimately, this gives you a **rules-based mapping** from VC 5-minute temps → Kalshi’s integer daily bracket, which you can:

* Use in **backtests** to simulate settlement from VC data.
* Use **live** to estimate the winning bracket in real time, with known failure modes and known rare edge days.

---

## 9. To-Do List for Claude (Coding Agent)

**High-level tasks:**

1. **Schema sanity check**

   * Confirm / create `wx.vc_minute_weather` and `wx.settlement` with at least:

     * `vc_minute_weather(city, ts_utc, temp_f)`
     * `settlement(city, day, tmax_final_f)`
2. **Create `analysis/reverse_engineering.py`**:

   * Add:

     * `DaySeries` dataclass
     * Rule functions:

       * `rule_max_round_f`
       * `rule_max_of_rounded`
       * `rule_ceil_max`
       * `rule_floor_max`
       * `rule_plateau(day, min_minutes_plateau=20)`
       * `rule_ignore_singletons`
     * `RuleStats` dataclass
     * `get_cli_window_utc(city, day)`
     * `load_day_series(session, city, day)`
     * `evaluate_rules_over_range`
     * `print_stats`
     * `write_mismatches_csv`
3. **Create `scripts/rev_eng_cli_from_vc.py`**:

   * Wraps evaluation in a CLI, reading `DATABASE_URL` and arguments.
   * Prints rule accuracy to stdout.
   * Writes mismatches to a CSV under `reports/`.
4. **Run initial experiments**:

   * For each city, over full backfill (e.g., 2022-01-01 → latest):

     * Run the script.
     * Record which rule has highest accuracy.
   * Inspect the mismatch CSVs for obvious patterns (short spikes, etc.).
5. **Report back in a short summary**:

   * For each city:

     * Best rule + accuracy + MAE.
     * Number of days with off-by-1, off-by-2 or more.
     * A few representative “weird days” for manual inspection.

This completes the first “reverse engineer temperatures” pass: you get a simple, testable, robust ruleset that’s grounded in actual historical Kalshi/NWS behavior and uses your Visual Crossing 5-minute temps as the input signal.

