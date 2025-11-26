I asked my friend how to do this, and he said to do the below and tell you.

# How to Pull All Historical Daily High Temps (TMAX) for ASOS Stations

You *can* absolutely do this historically without giving up on CLI/CF6 – and you don’t need brittle HTML scraping to do it.

There are really **three** data sources you care about:

1. **IEM “Tabular CLI” JSON API** – parsed daily climate (CLI) by station+year.
2. **IEM AFOS text archive** – raw CLI/CF6 text products via the `afos/retrieve.py` API.
3. **NCEI “daily-summaries” API** – canonical TMAX fallback/validator.

Below I’ll show you:

* URL patterns + pagination/chunking strategy for each.
* Python code stubs your agent can implement & test.
* Where to store `raw_json` / `raw_text` to future-proof.

I’ll keep the code framework-y so the agent can wire it directly into your repo.

---

## 1. IEM Tabular CLI JSON – easiest “daily CLI” feed

The IEM **Tabular CLI Report Data** page (`clitable.php`) is backed by a JSON API they explicitly advertise:

> “API: There is a JSON(P) webservice that backends this table presentation. You can directly access it here:
> `https://mesonet.agron.iastate.edu/json/cli.py?station=KMDW&year=2018`”

That API returns parsed daily climate values for a single **ASOS station** and **year**, e.g. KMDW for 2018. That’s ideal for daily TMAX, precipitation, etc.

### 1.1. URL pattern & parameters

For any ASOS station:

```text
https://mesonet.agron.iastate.edu/json/cli.py?station={STATION}&year={YYYY}
```

* `station` – 4-letter station ID (e.g. `KMDW`, `KDEN`, `KAUS`, `KMIA`, `KPHL`, `KLAX`).
* `year` – integer year, e.g. `2024`.

You call this **once per station per year**, so pagination is trivial: your “pagination” is just looping the years.

### 1.2. Code stub: fetch & parse CLI JSON

Your coding agent can start with something like:

```python
import requests
from dataclasses import dataclass
from typing import List, Dict, Any
import logging

log = logging.getLogger(__name__)

IEM_CLI_BASE = "https://mesonet.agron.iastate.edu/json/cli.py"

@dataclass
class DailyCLI:
    station: str
    date: str           # 'YYYY-MM-DD' (or whatever the JSON actually returns)
    tmax_f: float | None
    tmin_f: float | None
    precip_in: float | None
    snow_in: float | None
    raw: Dict[str, Any] # full row for future-proofing


def fetch_cli_year(station: str, year: int, session: requests.Session | None = None) -> List[DailyCLI]:
    """
    Fetch daily CLI data for a station+year from IEM JSON API.

    NOTE: The JSON schema typically contains 'data' rows and a 'columns'
    mapping. The agent should print the first response once and adjust
    field names accordingly.
    """
    sess = session or requests.Session()
    params = {
        "station": station,
        "year": year,
    }
    resp = sess.get(IEM_CLI_BASE, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    # Agent: inspect payload structure and adjust this parsing if needed
    log.info("CLI payload keys for %s %s: %s", station, year, list(payload.keys()))

    # IEM JSON services typically look like:
    # { "station": "KMDW", "year": 2018, "data": [...], "fields": [...] }
    data_rows = payload.get("data", [])
    fields = payload.get("fields") or payload.get("columns") or []

    # Heuristic: try to locate column indexes for max temp, min temp, precip, snow
    # Agent should confirm these names from a sample payload
    name_to_idx = {name: idx for idx, name in enumerate(fields)}
    # Example guesses – MUST be confirmed by printing fields:
    # e.g. fields might include 'max_temp_f', 'min_temp_f', 'precip_in', 'snow_in'
    tmax_key_candidates = ["max_temp_f", "max_temp", "max_tmpf"]
    tmin_key_candidates = ["min_temp_f", "min_temp", "min_tmpf"]
    pcp_key_candidates = ["precip_in", "precipitation", "precip"]
    snow_key_candidates = ["snow_in", "snow", "snowfall"]

    def try_idx(candidates):
        for key in candidates:
            if key in name_to_idx:
                return name_to_idx[key]
        return None

    idx_tmax = try_idx(tmax_key_candidates)
    idx_tmin = try_idx(tmin_key_candidates)
    idx_pcp = try_idx(pcp_key_candidates)
    idx_snow = try_idx(snow_key_candidates)

    results: List[DailyCLI] = []
    for row in data_rows:
        # row is typically a list aligned with fields[]
        # Again, agent should confirm by printing a sample.
        row_dict = {fields[i]: row[i] for i in range(len(fields))}

        date_str = row_dict.get("date") or row_dict.get("valid")  # adjust as needed
        tmax = float(row[idx_tmax]) if idx_tmax is not None and row[idx_tmax] not in ("M", None, "") else None
        tmin = float(row[idx_tmin]) if idx_tmin is not None and row[idx_tmin] not in ("M", None, "") else None
        pcp = float(row[idx_pcp]) if idx_pcp is not None and row[idx_pcp] not in ("M", None, "") else None
        snow = float(row[idx_snow]) if idx_snow is not None and row[idx_snow] not in ("M", None, "") else None

        results.append(
            DailyCLI(
                station=station,
                date=date_str,
                tmax_f=tmax,
                tmin_f=tmin,
                precip_in=pcp,
                snow_in=snow,
                raw=row_dict,
            )
        )

    return results
```

Things for the agent to **double-check**:

* Print `payload.keys()` and `payload["fields"]` / `payload["data"][0]` once to confirm column names and alignment.
* Write a tiny test that asserts e.g. for `KMDW, 2018` the first day’s `tmax_f` matches the HTML table (Jan 1, 2018 → 3°F) from the CLI table.

---

## 2. IEM AFOS text archive – raw CLI + CF6 products

For full raw payloads and cross-checking, use the **AFOS retrieve service** documented at:

> `/cgi-bin/afos/retrieve.py` with parameters like `pil`, `sdate`, `edate`, `fmt`, `limit`.

### 2.1. URL pattern for CLI & CF6

For Midway, you’ve already seen:

* CLI AFOS PIL: `CLIMDW`
* CF6 AFOS PIL: `CF6MDW`

The general pattern is:

```text
CLI{3-char station code}  e.g.  CLIMDW, CLIORD, etc.
CF6{3-char station code}  e.g.  CF6MDW, CF6DEN, etc.
```

Exact PILs are visible on IEM’s product pages for each station, and you can store a mapping for your 6 cities.

**Example: all CLI for Midway from 2024-01-01 to 2024-12-31**

```text
https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil=CLIMDW&fmt=text&sdate=2024-01-01&edate=2025-01-01&limit=9999
```

Key params (from docs):

* `pil` – required AFOS ID / product ID (e.g. `CLIMDW`, `CF6MDW`).
* `sdate`, `edate` – ISO dates or date-time; inclusive/exclusive.
* `fmt` – `text`, `html`, or `zip`.
* `limit` – max number of products (up to 9999).

### 2.2. Code stub: fetch CLI text products

```python
import requests
from dataclasses import dataclass
from typing import List

AFOS_RETRIEVE = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"

@dataclass
class AFOSTextProduct:
    pil: str
    ts_utc: str   # product timestamp as string (ISO / raw header)
    raw_text: str


def fetch_afos_products(pil: str, sdate: str, edate: str, session: requests.Session | None = None) -> List[AFOSTextProduct]:
    """
    Fetch raw NWS text products from IEM AFOS archive for a PIL and date range.

    sdate, edate: 'YYYY-MM-DD' (UTC). edate is exclusive.
    """
    sess = session or requests.Session()
    params = {
        "pil": pil,
        "sdate": sdate,
        "edate": edate,
        "fmt": "text",
        "limit": 9999,
        "order": "asc",
    }
    resp = sess.get(AFOS_RETRIEVE, params=params, timeout=60)
    resp.raise_for_status()
    text = resp.text

    # Products are concatenated; typical AFOS header structure:
    # CDUS43 KLOT 290639
    # CLIMDW
    # CLIMATE REPORT
    # ...
    #
    # Simplest approach: split on lines that look like WMO headers.
    lines = text.splitlines()
    products: List[AFOSTextProduct] = []
    current_lines: list[str] = []
    current_ts = None

    for line in lines:
        # Detect WMO header: e.g. 'CDUS43 KLOT 290639' at start of each product
        if line.strip().startswith("CDUS") and " CLIMD" in line:
            # flush previous
            if current_lines:
                products.append(
                    AFOSTextProduct(
                        pil=pil,
                        ts_utc=current_ts or "",
                        raw_text="\n".join(current_lines).strip(),
                    )
                )
                current_lines = []
            current_ts = line.strip()
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        products.append(
            AFOSTextProduct(
                pil=pil,
                ts_utc=current_ts or "",
                raw_text="\n".join(current_lines).strip(),
            )
        )

    return products
```

You only need this if you want **raw** CLI/CF6 text for archival / deep debugging. For daily TMAX, the JSON CLI service is easier, but AFOS text is your ground truth if you ever want to re-parse everything yourself.

### 2.3. Parsing CLI products for daily max (optional)

Inside each CLI product, the daily max is in the **TEMPERATURE (F)** section:

```text
TEMPERATURE (F)
 YESTERDAY
  MAXIMUM         87  11:53 AM  ...
```

You can write a small parser:

```python
import re

CLI_DATE_RE = re.compile(r"CLIMATE SUMMARY FOR ([A-Z]+ \d{1,2} \d{4})", re.I)
MAX_LINE_RE = re.compile(r"^\s*MAXIMUM\s+(-?\d+)", re.I)

def parse_cli_product_for_tmax(prod: AFOSTextProduct) -> tuple[str, int] | None:
    """
    Parse a single CLI text product and extract (date, tmax_f).
    Returns None if parsing fails.
    """
    lines = prod.raw_text.splitlines()

    # Find climate summary date
    date_str = None
    for line in lines:
        m = CLI_DATE_RE.search(line)
        if m:
            date_str = m.group(1).title()  # 'August 28 2024'
            break
    if not date_str:
        return None

    # Find 'MAXIMUM' line
    tmax = None
    for line in lines:
        m = MAX_LINE_RE.match(line)
        if m:
            tmax = int(m.group(1))
            break
    if tmax is None:
        return None

    # Convert date_str to ISO 'YYYY-MM-DD' in local control code
    # (agent: use datetime.strptime + mapping for month names)
    return date_str, tmax
```

**Tests the agent should run:**

* Fetch a small date range for `CLIMDW` (e.g. `sdate=2018-06-01&edate=2018-06-03`), parse each product, and check that the tmax values match `json/cli.py?station=KMDW&year=2018` for those dates.

---

## 3. NCEI “daily-summaries” API – canonical TMAX fallback

For long-term or extra validation, use NCEI’s **Access Data Service**:

* Base:
  `https://www.ncei.noaa.gov/access/services/data/v1`
* Dataset:
  `dataset=daily-summaries`
* Parameters:

  * `stations=USW00014819` (for KMDW – your agent can look up GHCND IDs for other airports).
  * `dataTypes=TMAX` (and/or `TMIN`, `PRCP` etc).
  * `startDate`, `endDate` in `YYYY-MM-DD`.
  * `units=standard` (for °F).
  * `format=json` or `csv`.
  * `includeStationName=1` for extra context.

Example URL (conceptual):

```text
https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00014819&dataTypes=TMAX&startDate=2024-01-01&endDate=2024-12-31&units=standard&includeStationName=1&format=json
```

NCEI docs confirm these parameters and examples.

### 3.1. Code stub: fetch NCEI daily TMAX

```python
import os
import requests
from dataclasses import dataclass
from typing import List, Dict, Any

NCEI_BASE = "https://www.ncei.noaa.gov/access/services/data/v1"

@dataclass
class DailyNCEI:
    station: str
    date: str
    tmax_f: float | None
    raw: Dict[str, Any]


def fetch_ncei_daily_tmax(
    station_id: str,  # e.g. 'USW00014819' for KMDW
    start_date: str,
    end_date: str,
    token: str | None = None,
    session: requests.Session | None = None,
) -> List[DailyNCEI]:
    """
    Fetch daily TMAX from NCEI daily-summaries.

    token: NCEI CDO or Access token if required (depends on dataset usage/volume).
    """
    sess = session or requests.Session()
    params = {
        "dataset": "daily-summaries",
        "stations": station_id,
        "dataTypes": "TMAX",
        "startDate": start_date,
        "endDate": end_date,
        "units": "standard",
        "includeStationName": "1",
        "format": "json",
    }
    headers = {}
    # For classic CDO API (v2), you use token in header; for access-data-service v1
    # it may not be necessary; agent should check docs if you hit auth errors.
    if token:
        headers["token"] = token

    resp = sess.get(NCEI_BASE, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    results: List[DailyNCEI] = []
    # Response is usually a list of dicts, each with keys like 'DATE', 'TMAX', 'STATION', 'NAME', etc.
    for row in data:
        date = row.get("DATE")
        tmax_raw = row.get("TMAX")
        tmax = float(tmax_raw) / 10.0 if tmax_raw not in (None, "", "NaN") else None  # some NCEI elements are in tenths
        results.append(
            DailyNCEI(
                station=row.get("STATION", station_id),
                date=date,
                tmax_f=tmax,
                raw=row,
            )
        )
    return results
```

Again, the agent should print one record and confirm:

* Are `DATE` and `TMAX` present?
* Is `TMAX` already in °F or in tenths? (NCEI doc: daily-summaries uses tenths of °C or °F depending on units; check a sample row.)

---

## 4. Putting it together & tests for the agent

Here’s the workflow I’d have your agent implement:

### 4.1. Settlement pipeline per (city, date)

1. **Primary daily TMAX**: `json/cli.py?station=KMDW&year=YYYY`

   * Use `fetch_cli_year` to get `DailyCLI` rows for the station/year.
   * Map `date -> tmax_f`.

2. **Raw NWS CLI text** via AFOS (optional but recommended for archival):

   * Use `fetch_afos_products("CLIMDW", sdate, edate)`.
   * Store `raw_text` to `wx.cli_raw` or as `raw_json` (string) in your settlement table.

3. **Raw CF6 monthly text** via AFOS (optional cross-check):

   * Similar to CLI, but with `pil=CF6MDW`.
   * Later you can build a parser to cross-check monthly stats.

4. **NCEI TMAX** (fallback / validation):

   * Use `fetch_ncei_daily_tmax("USW00014819", start_date, end_date)`.
   * Compare CLI-derived TMAX vs NCEI TMAX; if CLI missing, use NCEI.

5. **Write to `wx.settlement`**:

   * For each date, choose `tmax_final` based on your precedence (CLI > CF6 > ADS/NCEI).
   * Store raw payloads (`raw_cli_json`, `raw_cli_text`, `raw_ncei_json`) in JSONB columns for future-proofing.

### 4.2. Suggested tests

Have your agent write a few concrete tests:

```python
def test_cli_json_vs_html():
    # Hard-code a known station/year/day
    daily = fetch_cli_year("KMDW", 2018)
    jan01 = next(d for d in daily if d.date.startswith("2018-01-01"))
    # From the CLI HTML table for KMDW/2018, Jan 1 max = 3°F
    assert jan01.tmax_f == 3.0

def test_afos_cli_parse_matches_cli_json():
    # Fetch small AFOS range for CLIMDW around 2018-01-01
    prods = fetch_afos_products("CLIMDW", "2018-01-01", "2018-01-03")
    parsed = {}
    for p in prods:
        res = parse_cli_product_for_tmax(p)
        if res:
            date_str, tmax = res
            parsed[date_str] = tmax

    # Compare at least one date to CLI JSON
    cli_daily = fetch_cli_year("KMDW", 2018)
    jan01 = next(d for d in cli_daily if d.date.startswith("2018-01-01"))
    # convert AFOS date 'January 1 2018' to '2018-01-01' in test helper, then compare
    assert parsed["January 1 2018"] == int(jan01.tmax_f)

def test_ncei_tmax_reasonable():
    daily = fetch_ncei_daily_tmax("USW00014819", "2018-01-01", "2018-01-10")
    assert len(daily) > 0
    # Values should be in a realistic range
    for d in daily:
        assert -60.0 < (d.tmax_f or 0) < 130.0
```

Those give you both correctness and confidence that your scraping/API logic is wired correctly.

---

## 5. Summary of “how to pull it” for your agent

* **Don’t screen-scrape forecast.weather.gov** for CLI; it only has recent products and HTML parsing is fragile.
* **Use IEM’s JSON CLI API** (`json/cli.py?station=KMDW&year=YYYY`) as your primary daily dataset – it’s exactly the CLI 24-hour climate summary at local standard time, parsed and ready.
* **Use IEM AFOS `retrieve.py`** for raw CLI/CF6 text products when you want the original NWS bulletins; you can chunk by year (`sdate`, `edate`) and use `limit=9999` for bulk pulls.
* **Use NCEI `daily-summaries`** (`dataset=daily-summaries&dataTypes=TMAX`) as your deep-history and QA source for TMAX; you can request JSON/CSV with clear parameters.

If your agent implements the stubs above and writes the suggested tests, you’ll have a solid, fully historical TMAX pipeline that stays aligned with both NWS CLI and NCEI – and you’ll be storing full raw payloads for anything your future models or settlement audits might need.
