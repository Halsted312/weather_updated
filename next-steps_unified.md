
## 📄 To: Agent Coder — Phase 2E/F Plan (Bracket models ✅ + Unified model in parallel)

> **Goal:** Keep the current per‑bracket pipeline (less / 4×between / greater) moving to completion **and** add a **unified distribution** over all 6 daily brackets with *minimal code changes*. We will A/B both approaches in backtests and keep whichever dominates on calibration + P&L.

### Summary of what you’ll build (short)

* **Finish current path**: Chicago/less → backtest + acceptance. Then the 5 additional cities as planned.
* **Add a “Unified Head” (non‑invasive):** a tiny module that **couples** the six bracket probabilities at a timestamp into a *single coherent distribution that sums to 1*, via **logit‑softmax renormalization** (temperature‑tuned) or **pairwise coupling**. This reuses existing models/predictions; no retrain required.
* **Wire unified head** into:

  * `backtest/model_kelly` (optional `--unified-head` flag).
  * `scripts/rt_loop.py` (optional `--unified-head` flag).
* **(Optional after we measure U‑Head)**: a small **multiclass meta‑model** trained on the outputs of the base models (stacking) to directly predict the 6‑way distribution (again minimal code; just a new file and a trainer that reads OOF predictions).

---

## 0) Reasons & references (why this approach)

* You already estimate *exclusive* events (six brackets per city/day). Binary models per market **don’t enforce** that the six probabilities sum to 1. A light **probability coupling** layer fixes this while preserving your trained models. Standard approaches include **pairwise coupling** / **normalization of one‑vs‑rest** scores; see scikit‑learn’s notes on multiclass probability coupling and the literature on calibration/coupling and temperature/Dirichlet calibration.
* If you later want a single model, **ordinal** (cumulative‑link) or **multinomial** logistic are classic for ordered bins (temperature ranges). Pros/cons exist, especially with single‑class weeks; stacking on your current models is the lowest‑risk path.

**Bottom line:** Keep the bracket models (they’re working and map to markets). Add a tiny “unified head” so we also get a coherent, calibrated **6‑class** distribution for sizing and cross‑bucket risk.

---

## 1) Finish the bracket pipeline (no surprises)

**Action 1.1 — Complete Chicago/less**

* Train 90→7 windows (60 trials), ElasticNet.
* Run **maker‑first** backtest with JSON export.
* Generate acceptance artifacts.

**Action 1.2 — Small hardening patches (apply now)**

### 1.2.a `ml/dataset.py` — silence the pandas FutureWarning & lock dtypes

You saw:

```
FutureWarning: Downcasting object dtype arrays on .fillna ...
```

Patch the optional‑feature imputation to be dtype‑safe:

```python
# BEFORE (approx at ~line 340/455 depending on your version)
for col in optional_features:
    if col in df_after_critical.columns:
        df_after_critical[col] = df_after_critical[col].fillna(0.0)

# AFTER (dtype-stable)
for col in optional_features:
    if col in df_after_critical.columns:
        if not pd.api.types.is_float_dtype(df_after_critical[col]):
            df_after_critical[col] = pd.to_numeric(df_after_critical[col], errors="coerce")
        df_after_critical[col] = df_after_critical[col].astype("float32").fillna(0.0)
```

and where you add NA indicator columns, keep them compact:

```python
na_flag = f"{col}_is_na"
df_after_critical[na_flag] = df_after_critical[col].isna().astype("int8")
```

### 1.2.b `ml/logit_linear.py` — single‑class evaluation

You already patched `log_loss(..., labels=[0,1])`. Also force ECE/Brier to **handle single‑class** test sets gracefully:

```python
# wherever you compute metrics on y_true, y_pred
unique = np.unique(y_true)
if len(unique) == 1:
    # define safe metrics in degenerate case
    # Brier/log_loss are still defined if labels provided; ECE will be 0 if all p near 0 (for negative-only)
    brier = float(brier_score_loss(y_true, y_pred))
    ll = float(log_loss(y_true, y_pred, labels=[0,1], eps=1e-6))
    ece = float(ml.eval.expected_calibration_error(y_true, y_pred, labels=[0,1]))  # make eval accept labels
else:
    # normal path
```

*(If `ml/eval.py`’s ECE doesn’t accept `labels`, update it to tolerate single‑class by skipping empty bins and reporting bin counts.)*

### 1.2.c `ml/dataset.py` — assert bracket coverage per timestamp (guard rails)

We rely on exactly six markets per city/day. Add a **sanity check** (counts per `(event_date, bracket_key)`) when building test splits. If a day is missing any of the expected 6 brackets, **log and skip** that day in the unified path:

```python
# after you have features_df with market_ticker and event_date
group_cols = ["event_date", "bracket_key"]  # bracket_key must identify each of the 6 brackets
coverage = (features_df[group_cols]
            .drop_duplicates()
            .groupby("event_date").size())
if (coverage != 6).any():
    missing_days = list(coverage[coverage != 6].index)
    logger.warning(f"Missing full bracket set on {len(missing_days)} days: {missing_days[:5]}...")
```

*(Make sure you have a `bracket_key` column (e.g., `{type}_{floor}_{cap}`) — if not, derive it in `ml/features.py` from metadata you already join.)*

---

## 2) Add the **Unified Head** (non‑invasive; no retraining)

> **What it does:** For each `(city, event_date, timestamp)`, take the six bracket rows and their **calibrated** per‑bracket probabilities `p_i`. Produce a **coherent distribution** `q_i` across the six by either (A) **logit‑softmax renormalization** with temperature, or (B) **pairwise coupling**. This lives in a small new file and plugs into backtest & RT loop.

### 2.1 New file: `ml/unified_head.py`

Create this minimal module:

```python
# ml/unified_head.py
import numpy as np

def _safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)

def softmax_renorm(probs, tau=1.0, use_logit=True):
    """
    probs: array-like of shape (6,) per timestamp
    Returns q: normalized 6-way distribution summing to 1.
    """
    p = np.asarray(probs, dtype=float)
    if use_logit:
        s = _safe_logit(p) / max(tau, 1e-6)
        s = s - np.max(s)  # stabilize
        w = np.exp(s)
    else:
        s = p / max(tau, 1e-6)
        w = np.maximum(s, 1e-12)
    q = w / np.sum(w)
    return q

def dirichlet_temperature(probs, alpha):
    """
    Optional: map probs to Dirichlet-regularized mean; alpha is concentration (>0).
    q_i ∝ (p_i + alpha_i) - minimalistic variant for experimentation.
    """
    p = np.asarray(probs, dtype=float)
    a = np.full_like(p, fill_value=alpha, dtype=float)
    q = (p + a) / np.sum(p + a)
    return q

def couple_timestamp_rowset(df_rows, p_col="p_calibrated", method="softmax", tau=1.0):
    """
    df_rows: pandas DataFrame with exactly 6 rows for the timestamp (one per bracket).
    Returns array q of length 6 in the same row order.
    """
    probs = df_rows[p_col].to_numpy()
    if method == "softmax":
        q = softmax_renorm(probs, tau=tau, use_logit=True)
    else:
        # future: add pairwise coupling implementation or call into sklearn multiclass coupling
        q = softmax_renorm(probs, tau=tau, use_logit=True)
    return q
```

> Why softmax on **log‑odds**? You already calibrate each binary head (isotonic/sigmoid). Converting to log‑odds and renormalizing is a standard way to couple overlapping binary estimates into a consistent multinomial distribution; you can temperature‑tune `tau` on a dev fold to minimize multiclass log‑loss. See scikit‑learn’s multiclass probability notes and modern calibration literature.

### 2.2 Backtest integration (flag‑gated)

**File:** `backtest/run_backtest.py`
**Add CLI:**

```python
parser.add_argument("--unified-head", action="store_true",
    help="Couple bracket probs into a 6-way normalized distribution per timestamp")
parser.add_argument("--unified-tau", type=float, default=1.0,
    help="Temperature for unified head (softmax over logits)")
```

**Where predictions are loaded and edges computed** (in your model_kelly path), insert:

```python
from ml.unified_head import couple_timestamp_rowset

if args.unified_head:
    # group the predictions by (city, event_date, minute), couple into 6-way distribution
    # assuming df has columns: ['timestamp', 'event_date', 'bracket_key', 'p_calibrated', 'price_cents', ...]
    def _apply_unified(group):
        q = couple_timestamp_rowset(group, p_col="p_calibrated", method="softmax", tau=args.unified_tau)
        group["p_unified"] = q
        return group

    preds = preds.groupby(["event_date", "timestamp"], group_keys=False).apply(_apply_unified)
    # Use p_unified for edge instead of p_calibrated
    preds["p_for_edge"] = preds["p_unified"]
else:
    preds["p_for_edge"] = preds["p_calibrated"]
```

> **Note:** if any `(event_date, timestamp)` has ≠6 rows, drop it in unified mode (log a warning). This enforces clean coupling.

### 2.3 RT loop integration (flag‑gated)

**File:** `scripts/rt_loop.py`
Inside your signal generation step (after model inference + calibration), add:

```python
from ml.unified_head import couple_timestamp_rowset

if cfg.rt.unified_head:  # read from config or CLI
    # collect the 6 bracket rows for the current pending settlement date
    # ensure completeness: skip until all 6 p_model are present for the minute
    grouped = minute_df.groupby(["event_date", "timestamp"])
    def _couple(g):
        if len(g) != 6:
            g["p_for_edge"] = g["p_calibrated"]
            g["p_unified"] = np.nan
            return g
        q = couple_timestamp_rowset(g, p_col="p_calibrated", method="softmax", tau=cfg.rt.unified_tau)
        g["p_unified"] = q
        g["p_for_edge"] = g["p_unified"]
        return g
    minute_df = grouped.apply(_couple)
else:
    minute_df["p_for_edge"] = minute_df["p_calibrated"]
```

> This keeps RT behavior identical unless the flag is on. Your maker‑first engine then uses `p_for_edge` everywhere to compute edge and Kelly size.

### 2.4 Small analysis helper (optional)

**File:** `scripts/analyze_pilot.py`
Add an option to recompute **multiclass log‑loss** across the six bins using the coupled `p_unified` and the *single true bin per day*. This gives you **proper scoring** for the distribution.

```python
def multiclass_logloss(group):
    # group contains 6 rows (one per bracket) with columns: p (p_unified or p_calibrated), y (0/1 per bracket)
    p = group["p"].to_numpy()
    y = group["y"].to_numpy()  # one-hot with exactly one 1 if data is clean
    eps = 1e-12
    return float(-np.sum(y * np.log(np.clip(p, eps, 1 - eps))))
```

---

## 3) (Optional) True multiclass meta‑model (still minimal risk)

Once the unified head is in and A/B’d, if we want a **single learner** that predicts the 6‑way distribution:

1. **Out‑of‑fold base predictions:** During WF training, dump OOF `p_model` per bracket row (avoid leakage).
2. **Train meta‑model:** A multinomial logistic regression (softmax) with features:

   * Base model scores: `p_less`, `p_between_1..4`, `p_greater`
   * Market features: `minutes_to_close`, `spread_pct`, `temp_distance_to_bin`, etc.
3. **Calibrate** the 6‑way output with **temperature** or **Dirichlet calibration**.

**New file:** `ml/unified_meta.py` (later) with a simple `fit(X, y_multiclass)` that produces `proba_6`. This stays isolated and reuses existing artifacts; no disruption to your current trainer.

---

## 4) A/B experiment design & acceptance

**Datasets:** Use the same WF windows you already trained. Build two sets of predictions:

* **A (Current):** per‑market calibrated `p_model` (no coupling).
* **B (Unified‑Head):** the same predictions coupled into `p_unified`.

*(Later) C: unified meta‑model.*

**Metrics:**

* **Multiclass log‑loss** over the six brackets (per `(event_date, timestamp)`).
* **ECE (per class + macro)**
* **Backtest:** Net P&L, Sharpe, MaxDD, fees %, trades.

**Gates:** Keep existing (ECE ≤ 0.09, Sharpe ≥ 2.0, MaxDD ≤ 12%, fees ≤ 5%). Track **n_trades** to ensure practical signal density.

---

## 5) Concrete prompts & edits to run now

### 5.1 Finish Chicago/less → backtest → acceptance

**Command (training):**

```
python ml/train_walkforward.py \
  --city chicago --bracket less \
  --start 2025-08-02 --end 2025-11-13 \
  --feature-set elasticnet_rich \
  --penalties elasticnet --trials 60 \
  --train-days 90 --blend-weight 0.7 \
  --outdir models/production
```

**Command (backtest w/ JSON):**

```
python -m backtest.run_backtest \
  --city chicago --bracket less \
  --start-date 2025-08-02 --end-date 2025-11-13 \
  --strategy model_kelly \
  --models-dir models/production \
  --output-json backtest_chi_less_summary.json
```

**Accept:**

```
python scripts/generate_acceptance_report.py \
  --pilot-dir models/production/chicago/less \
  --config configs/elasticnet_chi_less.yaml \
  --output-dir acceptance_reports/phase2_chicago_less \
  --backtest-summary backtest_chi_less_summary.json
```

*(Make `configs/elasticnet_chi_less.yaml` by copy of `elasticnet_chi_greater.yaml` just changing `bracket: less`.)*

### 5.2 Add the **Unified Head** (code)

* Create `ml/unified_head.py` (snippet above).
* Patch `backtest/run_backtest.py` (flags + coupling hook).
* Patch `scripts/rt_loop.py` (flag + coupling hook).
* **Unit test** for coupling:

  * Sums to 1, monotonic w.r.t. tau, preserves ranking if tau=1.

**Backtest A/B (same windows):**

```
# A: baseline (no coupling)
python -m backtest.run_backtest --city chicago --bracket between --start-date ... --end-date ... --strategy model_kelly --models-dir models/trained --output-json backtest_A.json

# B: unified head on
python -m backtest.run_backtest --city chicago --bracket between --start-date ... --end-date ... --strategy model_kelly --models-dir models/trained --unified-head --unified-tau 1.2 --output-json backtest_B.json
```

**Analyze multiclass log‑loss + ECE** using `scripts/analyze_pilot.py` extension (snippet above).

---

## 6) Why three models vs one? And what to do now

* **Per‑bracket models (your current)**
  *Pros:* direct mapping to tradable markets; lets you inject bracket‑specific features (distance to bin floor/cap), and you already have them. *Cons:* probabilities aren’t inherently coherent across bins; extreme bins may have **single‑class** weeks.

* **Unified model (multiclass/ordinal)**
  *Pros:* produces a *coherent distribution* over bins; can borrow strength across bins; naturally handles relative ordering of temperature ranges. *Cons:* requires careful calibration; data shifts in bin definitions across cities/days must be encoded, and you’ll still need bracket features (distance to bin).

**Recommended staged approach:**

1. **Keep the current pipeline** to completion (less, then scale).
2. **Add the Unified Head** now — almost zero risk, immediate signal quality gains (coherent distribution).
3. After we measure gains, trial the **meta‑multiclass** learner (stacking) as the “one model” counterpart. If that wins on **multiclass log‑loss and P&L**, we can consider promoting it.

This is how practitioners typically evolve from many calibrated binary heads to coherent multiclass distributions, before committing to a wholesale architecture swap.

---

## 7) Open implementation details to confirm (so we don’t guess)

Please confirm or provide:

* **Schema of prediction CSVs** under `models/{tier}/{city}/{bracket}/win_*/preds_*.csv`
  *Need:* columns for `event_date`, `timestamp`, `market_ticker`, **bracket_key** (or alternatively columns `bracket_type`, `floor`, `cap`), `p_calibrated`, and market **price** used in backtest (`ask`, `bid`, or `mid`).
  *Why:* The unified head groups by `(event_date, timestamp)` and needs to know each of the six bracket entries. If `bracket_key` is not present, I’ll derive it from floor/cap/type.

* **Guaranteed bracket coverage per day**
  Are there days/timestamps where some bracket markets are missing? If so, in unified mode we’ll *skip* those timestamps (or switch to uncoupled p’s), and we’ll log counts in the backtest report.

* **RT loop**
  Confirm we should couple only when **all six** `p_model` for the minute are present. Otherwise, fall back to uncoupled `p_calibrated` for that minute.

---

## 8) Snippet pack (quick patches you can paste)

**8.1 `ml/eval.py` (ECE robust to single‑class)**

```python
def expected_calibration_error(y_true, y_prob, n_bins=15, labels=None, strategy="uniform"):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if labels is not None and len(np.unique(y_true)) == 1:
        # Single-class: define ECE as absolute calibration error in the occupied class only
        # i.e., if only negatives, ECE = mean(pred_prob) over negatives
        cls = labels[0] if np.all(y_true == labels[0]) else labels[1]
        if cls == 1:
            return float(np.mean(np.abs(1.0 - y_prob)))
        else:
            return float(np.mean(np.abs(y_prob)))
    # ... existing binning implementation ...
```

**8.2 `backtest/run_backtest.py` (new args)**

```python
parser.add_argument("--unified-head", action="store_true", help="Use unified coupling over 6 brackets")
parser.add_argument("--unified-tau", type=float, default=1.0, help="Temperature for unified head")
```

**8.3 `backtest/model_strategy.py` (use `p_for_edge`)**

Anywhere you currently do:

```python
edge = p_calibrated - price_prob
```

do:

```python
p_use = row.get("p_for_edge", row["p_calibrated"])
edge = p_use - price_prob
```

**8.4 `scripts/rt_loop.py` (flag in config)**

Add to YAML:

```yaml
rt:
  unified_head: true
  unified_tau: 1.2
```

Load in `rt_loop.py` and wire as in §2.3.

---

## 9) Deliverables checklist for this sprint

* ✅ Chicago/less training finished; **backtest JSON** + **07_backtest_acceptance.json**
* ✅ `ml/unified_head.py` created
* ✅ `backtest/run_backtest.py` & `scripts/rt_loop.py` patched with flags
* ✅ A/B backtest (no‑coupling vs unified‑head) summary table (Sharpe, MaxDD, fees%, n_trades)
* ✅ Multiclass log‑loss + per‑class ECE plots (optional)
* ✅ Feature completeness CSVs per window (already implemented, please attach for both brackets)

---

## 10) What to tell me / what I need to proceed fast

* Confirm the **prediction CSV schema** fields (see §7).
* Confirm **coverage policy** when fewer than 6 brackets exist at a timestamp.
* If `bracket_key` is not available, confirm which columns identify each bracket (e.g., `type/floor/cap`).
* Confirm whether to **default unified tau = 1.2** (good starting point; we’ll tune).

---

# Short answer to your strategy question

> **“Why three models? Why not one?”**

* Your current setup **already leverages bracket‑specific features** (e.g., `temp_to_floor`, `temp_to_cap`), and it maps 1:1 to tradable contracts. That’s a strong prior.
* A **unified distribution** is desirable for Kelly sizing and risk coherence. The **safest path** is to **keep** the bracket models and **couple** their calibrated outputs into one distribution (Unified Head). This takes *hours*, not weeks, and it’s standard practice to obtain consistent multiclass probabilities from overlapping binary models.
* If the Unified Head boosts scoring/P&L, you can then try a **true multiclass** (or **ordinal**) learner on top. That’s a *counter‑model* without ripping out what works.

---

## Sources (background & best practices)

* **Scikit‑learn on multiclass probabilities & coupling** (notes on one‑vs‑rest vs multinomial, and calibration).
* **Wu, Lin & Weng (2004)**: Probability estimates for multiclass classification via pairwise coupling (referenced in sklearn docs).
* **Kull et al. (NeurIPS 2019)**: Dirichlet calibration improves multiclass calibration beyond temperature scaling. Useful later if we want to calibrate the unified 6‑way distribution.
* **Ordinal / cumulative link models** (textbook overviews).

---

If you want, I’ll also draft the exact **git diffs** for each file once you confirm the prediction CSV schema fields (`p_calibrated`, `price_cents` or `price_prob`, and a `bracket_key`).
