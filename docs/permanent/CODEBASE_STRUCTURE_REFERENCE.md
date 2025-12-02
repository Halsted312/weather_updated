# Codebase Structure Reference

> Generated: 2025-11-30
> Purpose: Answer professor's questions about existing code infrastructure for market-clock model design

---

## 2.1 Files & Structure Confirmation

### Training Scripts Currently in Repo

| Script | Exact Path | Purpose |
|--------|------------|---------|
| `train_tod_v1_all_cities.py` | `/home/halsted/Python/weather_updated/scripts/train_tod_v1_all_cities.py` | TOD v1 per-city models (15-min intervals, 56 snapshots/day) |
| `train_all_cities_hourly.py` | `/home/halsted/Python/weather_updated/scripts/train_all_cities_hourly.py` | Hourly models (14 hours, 80 Optuna trials) |
| `train_all_cities_ordinal.py` | `/home/halsted/Python/weather_updated/scripts/train_all_cities_ordinal.py` | Baseline ordinal models (8 sparse hours, 30 trials) |
| `train_la_miami_ordinal.py` | `/home/halsted/Python/weather_updated/scripts/train_la_miami_ordinal.py` | LA/Miami with dynamic thresholds (fixed delta range issue) |
| `train_chicago_30min.py` | `/home/halsted/Python/weather_updated/scripts/train_chicago_30min.py` | Chicago hourly (legacy naming, actually 14 hourly snapshots) |

**Training Scripts by Model Type:**

- **TOD v1 per-city**: `scripts/train_tod_v1_all_cities.py`
- **Hourly models**: `scripts/train_all_cities_hourly.py`, `scripts/train_chicago_30min.py`
- **Baseline/experimental**: `scripts/train_all_cities_ordinal.py`, `scripts/train_la_miami_ordinal.py`

---

### Snapshot Builder Entry Points

| Builder | Exact Path | Purpose |
|---------|------------|---------|
| TOD Dataset Builder | `/home/halsted/Python/weather_updated/models/data/tod_dataset_builder.py` | 15-min/5-min interval snapshots for TOD v1 |
| Hourly Snapshot Builder | `/home/halsted/Python/weather_updated/models/data/snapshot_builder.py` | Standard hourly snapshots (8 or 14 hours) |

**No other custom snapshot builders exist** (no `*_market_clock_*`, `*_tminus1_*`, etc.)

---

### Live Inference Engine

| Component | Exact Path |
|-----------|------------|
| Main inference class | `/home/halsted/Python/weather_updated/models/inference/live_engine.py` |
| Model variant config | `/home/halsted/Python/weather_updated/config/live_trader_config.py` |

**Key class:** `LiveInferenceEngine`

**Model variant definitions in `config/live_trader_config.py`:**
```python
ORDINAL_MODEL_VARIANT = "tod_v1"  # Options: "baseline", "hourly", "tod_v1"

MODEL_VARIANTS = {
    "baseline": {
        "folder_suffix": "",
        "filename": "ordinal_catboost_optuna.pkl",
        "snapshot_hours": [10, 12, 14, 16, 18, 20, 22, 23],
        "requires_snapping": True,
    },
    "hourly": {
        "folder_suffix": "_hourly80",
        "filename": "ordinal_catboost_hourly_80trials.pkl",
        "snapshot_hours": list(range(10, 24)),
        "requires_snapping": True,
    },
    "tod_v1": {
        "folder_suffix": "_tod_v1",
        "filename": "ordinal_catboost_tod_v1.pkl",
        "snapshot_hours": None,  # Arbitrary timestamps
        "requires_snapping": False,
    },
}
```

---

### Data Loader Details

**File:** `/home/halsted/Python/weather_updated/models/data/loader.py`

**Database session factory:** Uses SQLAlchemy sessions passed from callers (no internal `create_session()`)

**Key loading functions:**

```python
def load_vc_observations(session, city_id, start_date, end_date) -> DataFrame
    # Source: wx.vc_minute_weather WHERE data_type='actual_obs'
    # Returns: datetime_local, datetime_utc, temp_f, humidity, windspeed_mph, etc.

def load_settlements(session, city_id, start_date, end_date) -> DataFrame
    # Source: wx.settlement
    # Returns: date_local, tmax_final, source_final, tmax_cli_f, tmax_iem_f, etc.

def load_historical_forecast_daily(session, city_id, target_date, basis_date) -> Optional[dict]
    # Source: wx.vc_forecast_daily WHERE forecast_basis_date=basis_date
    # Returns: tempmax_f, tempmin_f, humidity, precip_in, windspeed_mph, etc.

def load_historical_forecast_hourly(session, city_id, target_date, basis_date) -> DataFrame
    # Source: wx.vc_forecast_hourly WHERE forecast_basis_date=basis_date
    # Returns: target_datetime_local, lead_hours, temp_f, humidity, etc.

def load_inference_data(city_id, target_date, cutoff_time, session) -> dict
    # Combines: obs up to cutoff + T-1 forecast
    # Returns: {temps_sofar, timestamps_sofar, fcst_daily, fcst_hourly_df}
```

**Caching layers:**
- **Training**: Parquet dumps saved to `models/saved/{city}_tod_v1/train_data.parquet`
- **Inference**: In-memory prediction cache in `LiveInferenceEngine` (30-second cooldown)

---

## 2.2 Feature Engineering Assumptions

### Partial-Day Feature Implementation

**File:** `/home/halsted/Python/weather_updated/models/features/partial_day.py`

**Key function:**
```python
def compute_partial_day_features(temps_sofar: list[float]) -> FeatureSet
```

**Features computed:**
- `vc_max_f_sofar`, `vc_min_f_sofar`, `vc_mean_f_sofar`, `vc_std_f_sofar`
- `vc_q10_f_sofar` through `vc_q90_f_sofar` (percentiles)
- `vc_frac_part_sofar` (fractional part of max)
- `num_samples_sofar` (data quality)
- `t_base = int(round(max_f))` (baseline for delta)

**Time filtering:** This module receives pre-filtered `temps_sofar` list. The filtering happens in:
- `snapshot_builder.py`: Filters `datetime_local < cutoff` before calling
- `tod_dataset_builder.py`: Same filtering pattern

**No direct references to `snapshot_hour` or `cutoff_time`** in partial_day.py - it just receives the pre-filtered temperature list.

---

### Calendar / Time-of-Day Feature Implementation

**File:** `/home/halsted/Python/weather_updated/models/features/calendar.py`

**Function signature:**
```python
def compute_calendar_features(
    day: date,
    cutoff_time: Optional[datetime] = None,  # PRIMARY (for TOD v1)
    snapshot_hour: Optional[int] = None,     # LEGACY (for hourly/baseline)
) -> FeatureSet
```

**Dual-mode behavior:**
- If `cutoff_time` provided: Uses exact datetime for time-of-day features
- If only `snapshot_hour` provided: Reconstructs cutoff_time as `day + snapshot_hour:00`

**Features returned:**

| Feature | Source | Notes |
|---------|--------|-------|
| `snapshot_hour` | Legacy | Integer 0-23 |
| `snapshot_hour_sin`, `snapshot_hour_cos` | Legacy | Cyclical encoding of hour |
| `hour` | TOD v1 | Hour from cutoff_time (0-23) |
| `minute` | TOD v1 | Minute from cutoff_time (0-59) |
| `minutes_since_midnight` | TOD v1 | `hour * 60 + minute` |
| `hour_sin`, `hour_cos` | TOD v1 | Cyclical hour encoding |
| `minute_sin`, `minute_cos` | TOD v1 | Cyclical minute encoding |
| `time_of_day_sin`, `time_of_day_cos` | TOD v1 | Combined cyclical (minutes_since_midnight) |
| `doy_sin`, `doy_cos` | Day-level | Day of year cyclical |
| `week_sin`, `week_cos` | Day-level | Week of year cyclical |
| `month` | Day-level | Integer 1-12 |
| `is_weekend` | Day-level | Boolean |

**Existing features that could support "time since market open":**
- `minutes_since_midnight` - Could be adapted to `minutes_since_market_open` by subtracting market open time (e.g., 600 for 10:00 AM)
- No existing `minutes_since_observation_start` feature

---

### Quality / Coverage Features

**File:** `/home/halsted/Python/weather_updated/models/features/quality.py`

**Function signature:**
```python
def estimate_expected_samples(
    cutoff_time: Optional[datetime] = None,
    snapshot_hour: Optional[int] = None,
    day_start_hour: int = 6,  # <-- ASSUMPTION: observations start at 6 AM local
    step_minutes: int = 5,
) -> int
```

**Calculation:**
```python
if cutoff_time:
    minutes_elapsed = (cutoff_time.hour * 60 + cutoff_time.minute) - (day_start_hour * 60)
else:
    hours_elapsed = snapshot_hour - day_start_hour
    minutes_elapsed = hours_elapsed * 60

return minutes_elapsed // step_minutes
```

**Key assumption:** `day_start_hour=6` means observations are assumed to start at 06:00 local calendar day.

**Quality features returned:**
- `missing_fraction_sofar` = 1.0 - (actual_samples / expected_samples)
- `max_gap_minutes` = largest gap between consecutive observations
- `edge_max_flag` = 1 if max temp is in first/last 10% of window

**Same-day constraint:** No explicit assert, but all timestamp comparisons assume `datetime_local` is within the same calendar day as the target `day`.

---

### TOD vs Hourly Semantic Differences

**Current state of `cutoff_time` usage:**

| Module | Uses `cutoff_time`? | Uses `snapshot_hour`? | Notes |
|--------|--------------------|-----------------------|-------|
| `calendar.py` | Yes (primary) | Yes (fallback) | Backward compatible |
| `quality.py` | Yes (primary) | Yes (fallback) | Backward compatible |
| `partial_day.py` | N/A | N/A | Receives pre-filtered list |
| `shape.py` | N/A | N/A | Receives pre-filtered list |
| `forecast.py` | N/A | N/A | Uses T-1 forecast data directly |
| `snapshot_builder.py` | Yes | Yes (for building cutoff) | `build_snapshot_for_inference()` accepts both |
| `tod_dataset_builder.py` | Yes (only) | No | Always uses `cutoff_time` |

**Summary:** TOD v1 consistently uses `cutoff_time` throughout. A "market-clock" refactor would need to:
1. Change `day_start_hour=6` default to market open time
2. Potentially add `minutes_since_market_open` as a feature (easy derivation from `minutes_since_midnight`)

---

## 2.3 Model Training & Evaluation Infrastructure

### CatBoost Configuration

**From TOD v1 training script (`train_tod_v1_all_cities.py`):**

```python
# Default Optuna search space (in OrdinalDeltaTrainer)
param_distributions = {
    'depth': (4, 10),
    'iterations': (100, 1000),
    'learning_rate': (0.01, 0.3),
    'l2_leaf_reg': (1, 10),
    'border_count': (32, 255),
    'random_strength': (0.1, 10),
}

# Task type: CPU (no GPU config present)
task_type = "CPU"

# Default trials: 80 for TOD v1, 30 for baseline
n_trials = 80  # configurable via --trials flag
```

**Existing Optuna studies:**
- TOD v1: 40-80 trials per city (configurable)
- Hourly: 80 trials per city
- Baseline: 30 trials per city

---

### Multi-City vs Per-City Training

**TOD v1 script behavior:**
```python
# Per-city training (one model per city)
for city in cities:
    df_city = build_tod_snapshot_dataset(cities=[city], ...)
    trainer = OrdinalDeltaTrainer(base_model='catboost', n_trials=n_trials)
    trainer.train(df_train)
    # Save to models/saved/{city}_tod_v1/
```

**No "tod_v2_global" or all-cities-single-model config exists yet.**

**Dataset building supports multi-city:**
```python
# This works but not currently used for training
df_all = build_tod_snapshot_dataset(cities=['chicago', 'austin', 'denver', ...], ...)
```

---

### Metric Computation Code

**File:** `/home/halsted/Python/weather_updated/models/evaluation/metrics.py`

**Key functions:**

```python
def compute_delta_metrics(y_true, y_pred) -> dict:
    # Returns: delta_accuracy, delta_mae, off_by_1_rate, off_by_2plus_rate,
    #          within_1_rate, within_2_rate

def compute_settlement_metrics(t_settle, t_pred) -> dict:
    # Returns: settlement_accuracy, settlement_mae, settlement_rmse

def compute_probabilistic_metrics(y_true, proba, classes=None) -> dict:
    # Returns: log_loss, expected_calibration_error (ECE)

def compute_ordinal_metrics(y_true, proba, classes=None) -> dict:
    # Returns: mean_rank_error, ordinal_loss, cumulative_accuracy

def compute_all_metrics(y_true, y_pred, proba, t_base, t_settle,
                        delta_classes=None, thresholds=[80,85,90,95]) -> dict:
    # Comprehensive: combines all above metrics
```

**Location:** Reusable for market-clock model evaluation.

---

## 2.4 Data Availability for D-1 Windows

### D-1 Observations and Forecasts

**Loading D-1 observations (yesterday's temps):**
```python
from models.data.loader import load_vc_observations

# Example: Load observations from Nov 29 for Nov 30 market
df_obs_d1 = load_vc_observations(
    session=session,
    city_id='chicago',
    start_date=date(2025, 11, 29),  # D-1
    end_date=date(2025, 11, 29),
)
# Filter to 10:00 onward if needed:
df_obs_d1 = df_obs_d1[df_obs_d1['datetime_local'].dt.hour >= 10]
```

**Loading multiple forecast bases:**
```python
from models.data.loader import load_historical_forecast_daily

# T-1 forecast (D-1 basis for D target)
fcst_t1 = load_historical_forecast_daily(session, 'chicago', target_date=D, basis_date=D-1)

# T-2 forecast (D-2 basis for D target)
fcst_t2 = load_historical_forecast_daily(session, 'chicago', target_date=D, basis_date=D-2)
```

**Data availability:** Both are stored in existing tables used for TOD v1:
- `wx.vc_minute_weather` for observations
- `wx.vc_forecast_daily` / `wx.vc_forecast_hourly` for forecasts

---

### Mapping Kalshi Markets to (City, Event Date)

**Ticker parsing function (in trading scripts):**
```python
def parse_event_ticker(event_ticker: str) -> tuple[str, date]:
    """
    Example: 'KXHIGHCHI-25NOV28' -> ('chicago', date(2025, 11, 28))
    Format: {SERIES}-{YY}{MONTH}{DD}
    """
    series, date_part = event_ticker.split('-')[:2]
    city = SERIES_TO_CITY[series]
    # Parse date from '25NOV28' format
    ...
    return city, event_date

SERIES_TO_CITY = {
    "KXHIGHCHI": "chicago",
    "KXHIGHAUS": "austin",
    "KXHIGHDEN": "denver",
    "KXHIGHLAX": "los_angeles",
    "KXHIGHMIA": "miami",
    "KXHIGHPHIL": "philadelphia",
}
```

**Market open time:** Currently hard-coded as 10:00 AM ET in Kalshi. For market-clock model:
- `market_open_time = D-1 10:00 ET` (day before settlement)
- Settlement at market close on day D

**Database table for market metadata:**
```sql
-- kalshi.markets
SELECT ticker, city, event_date, strike_type, floor_strike, cap_strike, status
FROM kalshi.markets
WHERE series_ticker = 'KXHIGHCHI' AND event_date = '2025-11-30';
```

---

## 3. Files Reference Summary

### Live Inference Engine

**File:** `/home/halsted/Python/weather_updated/models/inference/live_engine.py`

**Class:** `LiveInferenceEngine`

**What it does:**
1. Loads all 6 city models at startup (variant-aware)
2. `predict()` method:
   - Gets current observations via `_get_observations()`
   - Builds snapshot features via `build_snapshot_for_inference()`
   - Calls CatBoost `predict_proba()`
   - Converts delta probs to bracket probs via `_delta_to_bracket_probs()`
3. Returns `PredictionResult` with bracket probabilities

**Key snippet (how model is called):**
```python
def predict(self, city, event_date, session, current_time=None, force_refresh=False):
    # Get observations up to cutoff
    temps_sofar, timestamps_sofar = self._get_observations(session, city, event_date, cutoff)

    # Build features
    cutoff_time, snapshot_hour = self._get_snapshot_params(current_time)
    features = build_snapshot_for_inference(
        city=city, day=event_date,
        temps_sofar=temps_sofar, timestamps_sofar=timestamps_sofar,
        cutoff_time=cutoff_time, snapshot_hour=snapshot_hour,
        fcst_daily=None,  # TODO: Add T-1 forecast loading
        fcst_hourly_df=None,
    )

    # Predict
    model = self.models[city]
    delta_probs = model.predict_proba(pd.DataFrame([features]))[0]

    # Convert to bracket probs
    bracket_probs = self._delta_to_bracket_probs(delta_probs, city, event_date, session, t_base)
    ...
```

---

### Main Trading / Execution Loop

**File:** `/home/halsted/Python/weather_updated/scripts/live_ws_trader.py`

**Class:** `LiveWebSocketTrader`

**Key flow:**
```python
class LiveWebSocketTrader:
    def __init__(self, dry_run=True, bet_size=50.0, max_daily_loss=500.0, cities=None):
        self.inference = LiveInferenceEngine(inference_cooldown_sec=30.0)
        self.client = KalshiClient(...)
        self.sizer = PositionSizer(...)
        self.order_books = {}  # ticker -> OrderBookState

    async def on_orderbook_update(self, ticker, yes_bid, yes_ask):
        city, event_date = parse_ticker(ticker)

        # Get model prediction
        prediction = self.inference.predict(city, event_date, session)
        model_prob = prediction.bracket_probs.get(ticker, 0.0)

        # Find best trade (maker vs taker, buy vs sell)
        side, action, price, ev, role = find_best_trade(
            model_prob=model_prob,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            min_ev_cents=3.0,
            maker_fill_prob=0.4,
        )

        if ev > MIN_EV_PER_CONTRACT_CENTS:
            # Size position
            size_result = self.sizer.calculate(ev, price, model_prob, prediction.settlement_std)

            # Place order
            if not self.dry_run:
                self.client.create_order(ticker, side, action, size_result.contracts, ...)
```

---

### Training Script for TOD v1

**File:** `/home/halsted/Python/weather_updated/scripts/train_tod_v1_all_cities.py`

**Key structure:**
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cities', nargs='+', default=ALL_CITIES)
    parser.add_argument('--trials', type=int, default=80)
    parser.add_argument('--interval', type=int, default=15)  # 15-min or 5-min
    args = parser.parse_args()

    session = create_session()

    for city in args.cities:
        # Build dataset
        df = build_tod_snapshot_dataset(
            cities=[city],
            start_date=date(2023, 1, 1),
            end_date=date(2025, 11, 27),
            session=session,
            snapshot_interval_min=args.interval,
        )

        # Train/test split (last 60 days for test)
        df_train = df[df['day'] < split_date]
        df_test = df[df['day'] >= split_date]

        # Train with Optuna
        trainer = OrdinalDeltaTrainer(base_model='catboost', n_trials=args.trials)
        trainer.train(df_train)

        # Evaluate
        y_pred = trainer.predict(df_test)
        metrics = compute_all_metrics(df_test['delta'], y_pred, ...)

        # Save
        output_dir = Path(f'models/saved/{city}_tod_v1')
        trainer.save(output_dir / 'ordinal_catboost_tod_v1.pkl')
        # Also saves: best_params.json, training_metadata.json, train_data.parquet
```

---

### Config / Constants

**Primary config file:** `/home/halsted/Python/weather_updated/config/live_trader_config.py`

**Key constants:**
```python
# Model selection
ORDINAL_MODEL_VARIANT = "tod_v1"
MODEL_VARIANTS = {...}  # See section 2.1 above

# Trading thresholds
MIN_EV_PER_CONTRACT_CENTS = 3.0
MIN_BRACKET_PROB = 0.35
MAX_BET_SIZE_USD = 50.0
MAX_DAILY_LOSS_USD = 500.0
KELLY_FRACTION = 0.25
BANKROLL_USD = 10000.0
MAKER_FILL_PROBABILITY = 0.4

# Data quality
MAX_FORECAST_AGE_HOURS = 24
MIN_OBSERVATIONS = 12
MAX_MODEL_STD_DEGF = 4.0

# TOD-specific
TOD_SNAPSHOT_INTERVAL_MIN = 15

# Series tickers
SERIES_TICKERS = ["KXHIGHCHI", "KXHIGHAUS", "KXHIGHDEN", "KXHIGHLAX", "KXHIGHMIA", "KXHIGHPHIL"]
```

---

### Metric Computation Utility

**File:** `/home/halsted/Python/weather_updated/models/evaluation/metrics.py`

**Usage example:**
```python
from models.evaluation.metrics import compute_all_metrics, compute_delta_metrics

# Basic delta metrics
delta_metrics = compute_delta_metrics(y_true=df['delta'], y_pred=y_pred)
# Returns: {'delta_accuracy': 0.58, 'delta_mae': 0.61, 'within_1_rate': 0.90, ...}

# Comprehensive metrics (including settlement, probabilistic, bracket)
all_metrics = compute_all_metrics(
    y_true=df['delta'],
    y_pred=y_pred,
    proba=proba,  # (n_samples, n_classes) from model.predict_proba()
    t_base=df['t_base'],
    t_settle=df['settle_f'],
    delta_classes=list(range(-2, 11)),  # 13 classes
    thresholds=[80, 85, 90, 95],  # For bracket Brier scores
)
```

---

## Quick Reference: File Paths

| Component | Path |
|-----------|------|
| **Training - TOD v1** | `scripts/train_tod_v1_all_cities.py` |
| **Training - Hourly** | `scripts/train_all_cities_hourly.py` |
| **Training - Baseline** | `scripts/train_all_cities_ordinal.py` |
| **Dataset - TOD** | `models/data/tod_dataset_builder.py` |
| **Dataset - Hourly** | `models/data/snapshot_builder.py` |
| **Data Loader** | `models/data/loader.py` |
| **Features - Partial Day** | `models/features/partial_day.py` |
| **Features - Calendar/TOD** | `models/features/calendar.py` |
| **Features - Quality** | `models/features/quality.py` |
| **Features - Shape** | `models/features/shape.py` |
| **Features - Forecast** | `models/features/forecast.py` |
| **Features - Rules** | `models/features/rules.py` |
| **Features - Base/Registry** | `models/features/base.py` |
| **Trainer** | `models/training/ordinal_trainer.py` |
| **Inference Engine** | `models/inference/live_engine.py` |
| **Metrics** | `models/evaluation/metrics.py` |
| **Config - Trading** | `config/live_trader_config.py` |
| **Config - Cities** | `src/config/cities.py` |
| **Config - VC Elements** | `src/config/vc_elements.py` |
| **Kalshi Client** | `src/kalshi/client.py` |
| **Trading - Fees** | `src/trading/fees.py` |
| **Trading - Risk** | `src/trading/risk.py` |
| **Live Trader - WS** | `scripts/live_ws_trader.py` |
| **Live Trader - Polling** | `scripts/live_active_trader.py` |
| **Live Trader - OpenMaker** | `open_maker/live_trader.py` |
| **Bracket Utils** | `open_maker/utils.py` |
| **DB Models** | `src/db/models.py` |

---

## Database Tables Reference

| Table | Schema | Purpose |
|-------|--------|---------|
| `wx.settlement` | Primary settlement source | Daily TMAX per city (ground truth) |
| `wx.vc_minute_weather` | 5-min observations | Actual temps, humidity, wind, etc. |
| `wx.vc_forecast_daily` | Daily forecast snapshots | T-1 forecasts with basis_date |
| `wx.vc_forecast_hourly` | Hourly forecast curves | 72-hour forecasts |
| `wx.vc_location` | Location dimension | Station/city metadata |
| `kalshi.markets` | Market metadata | Tickers, strikes, status |
| `kalshi.candles_1m` | 1-minute OHLCV | Price history |

---

## Key Assumptions to Note for Market-Clock Model

1. **`day_start_hour=6`** in quality.py assumes observations start at 6 AM local - would need adjustment for market-clock (10 AM ET market open)

2. **`minutes_since_midnight`** feature exists - can derive `minutes_since_market_open` easily

3. **Dual-mode support** (`cutoff_time` vs `snapshot_hour`) is already in place - new model can use `cutoff_time` exclusively

4. **No D-1 observation loading** currently in inference - would need to add for market-clock model that spans D-1 to D

5. **Per-city training** is current pattern - global model (all cities) would need new config entry

6. **CatBoost on CPU** - no GPU config present
