You can absolutely have *both* the “day-before” forecast and the “same-day” forecast for a given settlement day, and you can make your training and live pipelines line up cleanly. The trick is to:

* Treat forecasts as **(target_date, basis_date)** pairs (i.e., lead days), and
* Define features **“as of a snapshot time”** that only use basis runs that actually existed at that time.

Let’s walk through your Nov 21 example and then I’ll give you concrete instructions you can hand to your coding agent.

---

## 1. What VC actually gives you, relative to your trading timeline

From Visual Crossing:

* Historical forecasts are based on **full forecast model runs stored at midnight UTC each day**.
* With `forecastBasisDate=B`, the API can give you **D = B..B+14** forecast horizon in one call (days/hours, and minutes for near-term).

So for a settlement day **D = Nov 21**:

* There is a forecast run with `basis_date = Nov 20` (call it **B=20**). It forecasts D=21 as **lead_days=1**.
* There is another run with `basis_date = Nov 21` (call it **B=21**). It forecasts D=21 as **lead_days=0**.
* Each run has daily + hourly, and near-term **15-minute grids where the model supports it** (typically first 12–24h).

Your trading schedule:

* Market for high temp on **Nov 21** opens on **Nov 20 at 10:00 local**.
* Sometime later (midnight UTC, which is e.g. ~6pm previous local day for central time) the **B=21** run is produced. From that moment forward, you have a “same-day (T0) forecast” for Nov 21, including 15-min detail.

So for a given **real time snapshot** (t):

* The **set of available basis dates for target D** is ({ B \mid B \le \text{date}(t_{\text{UTC}})}).
* Among those, the **current forecast** is from (B_\text{current} = \max B).

Everything else is older runs you can use for drift.

---

## 2. How to think about T-2, T-1, T-0 in this context

Define:

[
\text{lead_days} = (D - B)
]

* For **B = Nov 19**, lead_days for D=Nov 21 is 2 (a T-2 forecast).
* For **B = Nov 20**, lead_days is 1 (T-1).
* For **B = Nov 21**, lead_days is 0 (T-0).

So at different times:

* **On Nov 20 at 10:00 local**:

  * Available basis runs: B = up to Nov 20.
  * For D=21:

    * T-2 from B=19
    * T-1 from B=20
    * T-0 from B=21 does **not** exist yet (basis run hasn’t happened).
* **On Nov 21 at 01:00 local**:

  * Available basis runs: B up to Nov 21.
  * For D=21:

    * T-2 (B=19), T-1 (B=20), and now T-0 (B=21) all exist.
    * T-0 run also gives you 15-minute curve (near-term).

From a modeling perspective:

* You don’t need a magic boolean “do I have the newest forecast?” — that’s just **“do I have lead_days=0 for this D yet?”**
* You *can* add a flag `has_lead0_forecast_for_D` if you want, but it’s functionally equivalent to “is the T0 features vector non-null”.

---

## 3. How to align training features with live features

The key is to define features **as of a snapshot time** (t), and **only use basis runs with `basis_datetime_utc <= t`**. That way your training set is using exactly the same information structure as live.

### Step 1 – Make the data model explicit in your DB

You already have the right fields in your VC tables:

* `VcForecastDaily`: `target_date`, `forecast_basis_date`, `lead_days` (derived), `data_type='historical_forecast'`/`'forecast'`.
* `VcForecastHourly`: `lead_hours = (datetime_utc - basis_datetime_utc)/3600`.
* `VcMinuteWeather`: `data_type='historical_forecast'` for minute grids, with `forecast_basis_date`, `lead_hours`.

Tell your agent:

> Make sure every forecast record we ingest has:
>
> * `target_date` (D),
> * `forecast_basis_date` (B),
> * `lead_days = (D - B)` (int),
> * `lead_hours` (for hourly/minute rows),
>   correctly set. All subsequent feature logic should be based on these, not on “T-1 vs T-0” strings.

### Step 2 – Choose a small set of snapshot times you care about

You don’t need to support “any minute of the day” for modeling; pick a few canonical snapshots per target date:

* **Snapshot A:** D−1 at 10:00 local (market opens / early trading).
* **Snapshot B:** D at, say, 00:30 local (just after the new basis run is available).
* **Snapshot C:** D at 08:00 local (closer to settlement, but before NWS Climate daily high window ends).

You can always add more later. But training on a few discrete snapshots keeps things manageable and comparably sized across history.

### Step 3 – For each snapshot, define the basis cutoff

For a given snapshot `(city_code, target_date D, snapshot_time_local)`:

1. Convert `snapshot_time_local` to UTC (`snapshot_time_utc`) using the city’s timezone.

2. Define a cutoff on basis:

   [
   B_\text{max} = \max B \text{ such that } \text{basis_datetime_utc} \le \text{snapshot_time_utc}
   ]

   In practice, with VC’s “midnight UTC” runs, you can approximate this as `B <= date(snapshot_time_utc)` (but your agent can use the precise `basis_datetime_utc` if you store it).

3. The **available basis dates** for D are all B in ([D-14, B_\text{max}]).

### Step 4 – Build a “snapshot feature builder” that respects this cutoff

Tell your agent to implement something like:

```python
def build_forecast_features_at_snapshot(
    session,
    city_code: str,
    target_date: date,
    snapshot_time_local: datetime,
    location_type: Literal["station", "city"],
) -> dict:
    """
    Use VcForecastDaily / VcForecastHourly / VcMinuteWeather with data_type='historical_forecast'
    to build features for target_date as of snapshot_time_local.
    """
```

Internally:

1. Compute `snapshot_time_utc`.

2. Query `VcForecastDaily` for all rows where:

   * `city_code`, `location_type` match,
   * `target_date == D`,
   * `data_type == 'historical_forecast'`,
   * `basis_datetime_utc <= snapshot_time_utc`.

3. From those rows, you have all `(B, D)` pairs and their `lead_days`.

Now compute features like:

* `high_lead0 = tempmax_f` for `lead_days=0` if present, else `None`.
* `high_lead1 = tempmax_f` for `lead_days=1`, `high_lead2`, … `high_lead14`.
* `lead_current = min(lead_days)` among the available basis runs (i.e., the freshest forecast).
* `drift_T1_vs_T2 = high_lead1 - high_lead2` if both exist.

For minute/hourly shape:

* For the **“current” basis** (B_\text{current}) (`lead_current`), query:

  * `VcMinuteWeather` where `forecast_basis_date == B_current`, `target_date == D` (via `datetime_local::date` or via lead_hours window).
  * If there are ~96 rows (15-min), compute detailed shape features; else fall back to hourly.

Important: for **Snapshot A (D−1 10:00)**, the “current” basis will be B=D−1 (lead_days=1), so `high_lead0` will be missing (no T0 yet). For **Snapshot B (D 00:30)**, B=D is available and `high_lead0` is now defined. That’s exactly the behavior you want.

You don’t need to manually track “did we get new forecast overnight” — `high_lead0` simply flips from null → non-null between snapshots.

---

## 4. How this matches live streaming

Live you’ll:

* Continuously ingest a **daily forecast snapshot** from VC (basis date = today, `data_type='forecast'`).
* Possibly also store intraday snapshots (same shape but different `basis_datetime_utc` / `source_system`).

You can reuse the **same snapshot feature builder**, just with:

* `data_type='forecast'` instead of `'historical_forecast'`.
* `basis_datetime_utc <= now` for cutoff (which in live may include multiple intraday snapshots if you log them).

The modeling contract stays the same:

* At any time `t`, for a given `D`, features are computed from all basis runs with `basis_datetime_utc <= t`.
* “Newest forecast” is just the most recent basis in that set; everything else is drift.

So your backtest and live are aligned by construction.

---

## 5. What to literally tell the coding agent

Here’s a block you can paste into your MD for them:

> **Forecast snapshots vs lead days**
>
> We want to model trading decisions for a target day D at different times before settlement (e.g. D−1 10:00, D 00:30, D 08:00). For each `(city_code, target_date=D, snapshot_time_local)` we must only use forecast runs that existed at that time.
>
> **Rules:**
>
> * Every forecast record in `VcForecastDaily`, `VcForecastHourly`, `VcMinuteWeather` must have:
>
>   * `forecast_basis_date` (B) and `forecast_basis_datetime_utc` (midnight UTC basis run) as stored by VC.
>   * `target_date` (D).
>   * `lead_days = (D - B)` (int).
> * At a snapshot time `t` (local), convert to `snapshot_time_utc` using the city’s timezone.
> * The **available basis runs** are all `B` where `basis_datetime_utc <= snapshot_time_utc`.
> * The **current run** is the max such `B`. `lead_current = (D - B_current)`.
>
> **Implement:**
>
> * A function `build_forecast_features_at_snapshot(session, city_code, target_date, snapshot_time_local, location_type)` that:
>
>   1. Finds all historical forecast rows with `target_date=D`, `basis_datetime_utc <= snapshot_time_utc`, and `location_type`.
>   2. Builds daily lead-based features: `high_lead0..high_lead14`, `lead_current`, drift features.
>   3. For the current basis run, builds intraday shape features from minutes if available, else from hourly, using a shared shape function that accepts `step_minutes` and returns the same feature schema.
>   4. Returns a flat dict of features, where `high_lead0` may be null if the same-day basis hasn’t happened yet (e.g. for D−1 snapshots).
>
> Live code will reuse this exact function, with `data_type='forecast'` instead of `'historical_forecast'`, and `snapshot_time_local = now`. This guarantees that training and live use the same information structure.

If they implement that, you’ll get exactly what you want:

* On Nov 20 at 10am, you see only T−1/T−2/etc. for Nov 21.
* After the midnight VC run (basis_date=21) becomes available, T0 appears in the feature vector.
* Both behavior and training/live alignment are controlled by the same snapshot logic, not manually maintained flags.

And you’re free to pick any snapshot times you like for training — the mechanism will behave consistently under the hood.
