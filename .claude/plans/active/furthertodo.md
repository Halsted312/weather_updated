Yep, there are a few more things I’d explicitly tell the agent so they don’t run off and build a 3-week “ideal” and miss what you actually need right now.

Below is something you can pretty much copy-paste (or adapt) as your reply to that plan.

---

### 0. Re-anchor the scope (Denver / Oct 2025 first)

> **Primary goal right now** is not to implement the full OpenAI-style “perfect pipeline”.
> Short-term goal is:
>
> * One city: **denver**
> * Date range: **2025-10-01 → 2025-10-31**
> * **2 Optuna trials** for ordinal and edge
> * Entire run uses **parquet only** (no DB) and **doesn’t break any existing scripts**.
>
> Once that is clean, fast, and repeatable end-to-end, then we can scale out to the full plan.

That’s the frame I’d set right at the top.

---

### 1. Use the *existing* scripts instead of inventing new ones

You already have a lot of the plumbing in `scripts/` and `models/pipeline/`:

```text
models/pipeline:
  01_build_dataset.py
  03_train_ordinal.py
  04_train_edge_classifier.py
  05_backtest_edge.py

scripts:
  build_dataset_from_parquets.py
  extract_raw_data_to_parquet.py
  run_multi_city_pipeline.py
  train_city_ordinal_optuna.py
  train_edge_classifier.py
  rebuild_all_cities_fresh.py
  ...
```

So I’d tell the agent explicitly:

1. **Do NOT create** a brand-new `scripts/extract_parquets_from_db.py`.

   * Instead, **extend** `scripts/extract_raw_data_to_parquet.py` so it does exactly what the plan describes (cities, date range, manifest, etc.). No duplicate “extractor” script. 
2. **Do NOT create** a new `run_staged_pipeline.py` orchestrator yet.

   * For now, use the existing entrypoints:

     * `scripts/build_dataset_from_parquets.py`
     * `scripts/train_city_ordinal_optuna.py`
     * `scripts/train_edge_classifier.py`
     * `scripts/run_multi_city_pipeline.py` if needed later
   * We can add a staged orchestrator *after* Denver/Oct ‘25 is clean.
3. Keep `models/pipeline/0x_*.py` working as thin wrappers; don’t break them with path changes until we’ve proven the new flow.

---

### 2. Correct the diagnosis about Philly’s 84-minute edge run

The plan claims Philly edge training is slow because it “hits the database” and that switching to parquet will fix it. That’s not quite true.

From the logs:

* Philly edge training:

  * auto-detects **candles_philadelphia.parquet**
  * loads 4.6M rows of candles from parquet
  * builds a `(day, bracket)` candle cache
  * then spends ~84 minutes processing 1068 days with 12 threads. 

Whereas:

* LA & Miami edge finished in <2 minutes because they reused **cached edge training data** parquet.
* Denver used cached edge data too and only failed on **PermissionError** when saving the model. 

So please update the plan assumptions:

> The main cause of Philly’s 84-minute run is **regenerating edge training data for ~1000 days**, not DB access for candles (we already use parquet there). The big win is:
>
> * solid, versioned **edge data caching**
> * and only regenerating when inputs truly change (ordinal model, candle parquet, or params).

I’d explicitly ask the agent to design a **simple cache-key / metadata file** for `edge_training_data_realistic.parquet` (city, date range, candles mtime, ordinal model mtime, key params) and to log *why* it chooses to regenerate vs reuse.

---

### 3. Permission strategy: don’t hardcode “halsted” and don’t go world-writable

The plan suggests:

```bash
sudo chown -R halsted:halsted models/
sudo chown -R halsted:halsted logs/
sudo chown -R halsted:halsted .claude/
chmod -R u+rw models/ logs/ .claude/
```



I’d reply:

* Don’t hardcode the username (`halsted`) into scripts. Use `$(whoami)` or make it a parameter if you really need it.
* I do **not** want everything to become world-writable `777`. If you change permissions, favor:

  * `chgrp` to a shared “weather” group
  * `chmod -R g+rwX`
  * optionally ACLs for defaults
* The only *must* is: `models/saved/{city}` and `logs/` must be writable by the user running the pipeline so Denver doesn’t fail saving `edge_classifier.pkl` again.

You can tell them: “Fix Denver once manually (remove root-owned files, chown the tree), then implement a small **preflight write check** in Python that attempts to create a temporary file in `models/saved/{city}` and fails early if not writable.”

---

### 4. Changes to make in **specific files**

This is where the `ls` you pasted is really helpful. I’d tell the agent:

#### 4.1 `scripts/extract_raw_data_to_parquet.py` (not a new extractor)

* Extend this script to behave like the “ideal” extractor in your plan:

  * `--cities`, `--start`, `--end`, `--workers`, `--output-dir`
  * writes a **manifest** with row counts and date ranges
  * atomic writes to `.tmp` + rename
* Don’t introduce a second extractor file; just evolve this one.

#### 4.2 `scripts/build_dataset_from_parquets.py`

This already exists and is our main offline builder.

Ask the agent to:

1. **Add date filters**:

   * Add `--start` and `--end` CLI args (dates).
   * Filter the `all_days` list to this range *before* train/test split:

     ```python
     all_days = sorted(raw_data.settlements['date_local'].unique())
     if args.start:
         all_days = [d for d in all_days if d >= args.start_date]
     if args.end:
         all_days = [d for d in all_days if d <= args.end_date]
     ```
   * This is critical for your **Denver / Oct 2025** test loop.
2. Add a **preflight check** for required parquets (as in the plan), but re-use it in both this script and `train_edge_classifier.py`. 
3. Log a tiny “dataset manifest” at the end (n_days, n_rows, columns, basic null percentages for a few key features).

#### 4.3 `scripts/train_city_ordinal_optuna.py`

You already use this as the general ordinal trainer.

Ask the agent to:

* For the **Denver Oct 2025 pilot**, always run it with `--use-cached --start-date 2025-10-01 --end-date 2025-10-31 --trials 2`.
* In code, improve:

  * Logs around “loading cached train/test datasets from: …” and the filtered date range.
  * Threading config: keep it **CPU only**, `thread_count` ≈ 28, and do **not** attempt to re-enable GPU given past issues.
* Do **not** yet expand the search space or `n_jobs` Optuna parallelism; that’s Phase 2 once correctness & speed are proven.

#### 4.4 `scripts/train_edge_classifier.py` **and** `models/pipeline/04_train_edge_classifier.py`

Plan already targets `scripts/train_edge_classifier.py` for parquet-only candles. Good. 

Ask them to:

* Ensure *all* candle access in edge training is either:

  * via the candles parquet (`models/candles/candles_{city}.parquet`), or
  * via the **existing cached edge training parquet** if already built.
* Add a **simple cache meta file** next to `edge_training_data_realistic.parquet`:

  * include `ordinal_model_path`, `candles_parquet_path`, `city`, `date_range`, and a short hash or mtimes
  * if nothing changed, skip regeneration and log: “Using cached edge training data (fresh)”.
* Make sure `models/saved/{city}` is created before saving `edge_classifier.pkl`.

#### 4.5 `scripts/run_multi_city_pipeline.py`

You already have this orchestrator.

Ask for **small tweaks only**:

* Add an optional **parquet preflight** at the top that uses the shared “required parquets” helper; if missing → log a clear error and exit (unless an explicit `--allow-db` flag is given).
* Keep `--continue-on-error` behavior but default to **fail-fast** (`continue-on-error` must be opt-in).
* Do *not* do the big directory restructure as part of this script yet.

---

### 5. Test suite: start tiny & data-driven, not with guessed schemas

The plan’s tests hard-code column names like `cloud_cover`, `wind_speed`, etc., which almost certainly don’t exactly match your actual columns. 

What I’d tell the agent:

> Start with **3–5 very targeted Pytests**, not a full “data_quality” test tree:
>
> 1. `test_train_test_no_overlap` – no overlapping `day` between train/test, and test days strictly after train days.
> 2. `test_core_columns_non_null` – in `train_data_full.parquet`, columns like `delta`, `settle_f`, `cutoff_time`, `snapshot_hour` have <5% nulls.
> 3. `test_candles_bid_ask_sane` – `yes_bid` ≤ `yes_ask` and both between 0 and 100.
> 4. `test_denver_oct_date_range` – for the Denver Oct 2025 run, only days in Oct 2025 appear.
>
> Derive expected column names from the actual DataFrame or code (e.g. `cloudcover`, `cloudcover_last_obs`), not from a guess.

Later you can grow this into the richer test suite from the plan, but don’t let “perfect tests” delay the basic pipeline validation.

---

### 6. Directory restructure & giant Optuna search space: **explicitly de-scope for now**

The plan’s Phase 5/6 is ambitious:

* Fully reorganize `models/saved/{city}` into datasets/models/params/metrics/logs. 
* Expand CatBoost search space and use parallel Optuna trials. 

I’d tell the agent:

> Please do **not** touch the directory layout or search space yet.
>
> * Any path changes risk breaking a *lot* of existing scripts (`train_all_cities_ordinal.py`, `verify_pipeline_parity.py`, `rebuild_all_cities_fresh.py`, etc.).
> * Any big Optuna expansion will increase runtime when we’re trying to *reduce* it.
>
> Let’s first:
>
> * Make Denver + Oct 2025 + 2 trials completely clean and repeatable.
> * Then do a separate, well-scoped PR for directory reorg + CatBoost tuning.

---

### 7. Ask for a **single concrete “Denver-Oct-2025” script** for you

Finally, ask them for something you can run top-to-bottom:

> Please add a small top-level script, e.g. `scripts/run_denver_oct2025_smoke_test.py`, that:
>
> * validates required parquets for `denver`
> * runs `build_dataset_from_parquets.py --city denver --start 2025-10-01 --end 2025-10-31`
> * runs `train_city_ordinal_optuna.py --city denver --use-cached --start-date 2025-10-01 --end-date 2025-10-31 --trials 2`
> * runs `train_edge_classifier.py --city denver --trials 2`
> * prints a short summary (rows, key metrics, total runtime)
>
> That will be our “golden path” to iterate on until it’s stable and quick.

---

If you paste some version of the above back to the agent, they’ll have a much tighter spec: **Denver, Oct-only, parquet-only, minimal code churn**, and very clear which parts of their big doc are “later” vs “now”.
