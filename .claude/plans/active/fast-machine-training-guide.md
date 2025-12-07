---
plan_id: fast-machine-training-guide
created: 2025-12-06
status: active
priority: high
agent: kalshi-weather-quant
---

# Fast Machine Training Guide - Kalshi Weather ML

## Overview

This guide documents the ML training pipeline for Kalshi weather derivatives trading. The system is designed for a **two-machine workflow**:

- **Slow Machine**: Has PostgreSQL/TimescaleDB database, used for data extraction
- **Fast Machine**: Has NVIDIA GPUs (3090/5090), used for model training

## 1. Directory Structure

```
weather_updated/
├── data/training_cache/{city}/
│   ├── full.parquet              # Full training dataset (~486K rows)
│   ├── train_data_full.parquet   # 80% train split (auto-created)
│   └── test_data_full.parquet    # 20% test split (auto-created)
│
├── models/
│   ├── raw_data/{city}/          # Raw parquet files extracted from DB
│   │   ├── settlements.parquet
│   │   ├── vc_observations.parquet
│   │   ├── vc_city_observations.parquet
│   │   ├── forecasts_daily.parquet
│   │   ├── forecasts_hourly.parquet
│   │   └── noaa_guidance.parquet
│   │
│   ├── candles/
│   │   └── candles_{city}.parquet  # Kalshi market candles (~11M rows)
│   │
│   ├── saved/{city}/               # Trained models
│   │   ├── ordinal_catboost_optuna.pkl
│   │   ├── best_params.json
│   │   ├── edge_classifier.pkl
│   │   └── edge_training_data_realistic.parquet
│   │
│   ├── training/
│   │   └── ordinal_trainer.py      # GPU-enabled CatBoost trainer
│   │
│   ├── features/
│   │   ├── station_city.py         # Station-city feature computation
│   │   └── pipeline.py             # Feature assembly
│   │
│   └── pipeline/
│       └── 03_train_ordinal.py     # Auto-split + train pipeline
│
├── scripts/
│   ├── train_city_ordinal_optuna.py    # Main ordinal training
│   ├── train_edge_classifier.py        # Edge classifier training
│   ├── backfill_station_city_features.py  # Backfill features
│   ├── build_dataset_parallel.py       # Build dataset (needs DB)
│   └── build_dataset_from_parquets.py  # Build from parquets (no DB)
```

## 2. Key Concepts

### Ordinal Regression
Trains K-1 binary classifiers for `P(delta >= k)` at each threshold. Settlement delta is bucketed into classes: -2, -1, 0, +1, +2.

### CatBoost GPU Training
- `task_type='GPU'` - Use NVIDIA GPU
- `devices='0'` - First GPU device
- `bootstrap_type='MVS'` - Minimum Variance Sampling (GPU-only, very fast)
- `border_count=128` - Fixed for GPU (must be power of 2)

### Day-Grouped Temporal Splits
All snapshots from the same day stay in the same fold (prevents lookahead leakage). Uses `DayGroupedTimeSeriesSplit`.

### Station-City Features
Compare airport station temps (e.g., KAUS) vs city-aggregate temps (e.g., Austin,TX):
- `station_city_temp_gap` - Current temperature gap
- `station_city_max_gap_sofar` - Max gap so far today
- `station_city_mean_gap_sofar` - Mean gap today
- `station_city_gap_std` - Gap consistency
- `city_warmer_flag` - 1 if city > station currently
- `station_city_gap_trend` - Gap change over time

## 3. Files Required on Fast Machine

Copy these from slow machine:

```bash
# Training data
data/training_cache/{city}/full.parquet

# Raw parquets (for building dataset without DB)
models/raw_data/{city}/
  - settlements.parquet
  - vc_observations.parquet
  - vc_city_observations.parquet
  - forecasts_daily.parquet
  - forecasts_hourly.parquet
  - noaa_guidance.parquet

# Candles (for edge classifier)
models/candles/candles_{city}.parquet

# Saved models (if continuing from existing)
models/saved/{city}/ordinal_catboost_optuna.pkl
models/saved/{city}/best_params.json
```

## 4. Training Commands

### Option A: Pipeline Script (Recommended)

This auto-creates train/test splits and trains:

```bash
cd /path/to/weather_updated
source .venv/bin/activate

python models/pipeline/03_train_ordinal.py \
  --city austin \
  --trials 80 \
  --cache-dir data/training_cache
```

### Option B: Direct Training Script

If train/test splits already exist:

```bash
PYTHONPATH=. python scripts/train_city_ordinal_optuna.py \
  --city austin \
  --trials 80 \
  --workers 8
```

### Option C: Build Dataset from Parquets (No DB)

If you need to rebuild the training dataset:

```bash
PYTHONPATH=. python scripts/build_dataset_from_parquets.py \
  --city austin \
  --output data/training_cache/austin/full.parquet
```

## 5. GPU Configuration

The `ordinal_trainer.py` has been configured for GPU training:

```python
# In _create_base_model() and Optuna objective:
params = {
    "task_type": "GPU",
    "devices": "0",
    "bootstrap_type": "MVS",  # GPU-optimized
    "border_count": 128,      # Power of 2 for GPU
    # ... other params
}
```

### GPU Search Space (Optuna)

```python
params = {
    "task_type": "GPU",
    "devices": "0",
    "depth": trial.suggest_int("depth", 5, 8),
    "iterations": trial.suggest_int("iterations", 400, 1500),
    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
    "border_count": 128,
    "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 40.0, log=True),
    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 30, 150),
    "random_strength": trial.suggest_float("random_strength", 0.1, 1.5),
    "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 0.9),
    "bootstrap_type": "MVS",
    "subsample": trial.suggest_float("subsample", 0.6, 0.95),
}
```

## 6. Station-City Feature Backfill

If training data is missing station-city features (all nulls):

```bash
PYTHONPATH=. python scripts/backfill_station_city_features.py \
  --city austin \
  --input data/training_cache/austin/full.parquet \
  --output data/training_cache/austin/full.parquet

# Then delete old splits to force regeneration
rm data/training_cache/austin/train_data_full.parquet
rm data/training_cache/austin/test_data_full.parquet

# Re-run training
python models/pipeline/03_train_ordinal.py --city austin --trials 80 --cache-dir data/training_cache
```

## 7. Edge Classifier Training

After ordinal model is trained:

```bash
PYTHONPATH=. python scripts/train_edge_classifier.py \
  --city austin \
  --trials 60 \
  --workers 12 \
  --optuna-metric sharpe \
  --min-trades-for-metric 50 \
  --threshold 0.5 \
  --sample-rate 4
```

### Regenerate Edge Data Only (Slow Machine)

```bash
PYTHONPATH=. python scripts/train_edge_classifier.py \
  --city austin \
  --threshold 0.5 \
  --sample-rate 4 \
  --regenerate-only
```

## 8. Troubleshooting

### "No CUDA-capable device detected"

```bash
# Check if GPU is visible
nvidia-smi
lspci | grep -i nvidia

# Load nvidia module
sudo modprobe nvidia

# Check CUDA
nvcc --version

# Reinstall if needed
pip install catboost --force-reinstall
```

### "Parquet files not found" (missing train/test splits)

Use the pipeline script which auto-creates splits:
```bash
python models/pipeline/03_train_ordinal.py --city austin --trials 80 --cache-dir data/training_cache
```

### PermissionError on parquet file

```bash
sudo rm models/saved/{city}/edge_training_data_realistic.parquet
```

### Station-city features all null (0% non-null)

Root cause was `build_dataset_parallel.py` having `include_station_city=False`.
Fix: Run backfill script (see section 6).

## 9. Expected Output Files

After successful training:

```
models/saved/austin/
├── ordinal_catboost_optuna.pkl    # Main ordinal model
├── best_params.json               # Optuna best params
├── ordinal_catboost_optuna.json   # Model metadata
├── final_metrics_austin.json      # Performance metrics
├── edge_classifier.pkl            # Edge classifier (if trained)
├── edge_classifier.json           # Edge metadata
└── edge_training_data_realistic.parquet  # Cached edge data
```

### Expected Metrics

**Ordinal Model:**
- Accuracy: ~43-45%
- MAE: ~1.3
- Within-1: ~58%
- Within-2: ~83%

**Edge Classifier:**
- Sharpe: 0.5-1.5 (not >5, which indicates leakage)
- Test AUC: <0.95 (sanity check)
- Filtered win rate: 55-70%

## 10. Key File References

| Component | File | Purpose |
|-----------|------|---------|
| Ordinal Trainer | `models/training/ordinal_trainer.py` | GPU-enabled CatBoost |
| Feature Pipeline | `models/features/pipeline.py` | Compose all features |
| Station-City | `models/features/station_city.py` | Station vs city gap features |
| Day Splits | `models/data/splits.py` | Temporal CV splits |
| Pipeline Script | `models/pipeline/03_train_ordinal.py` | Auto-split + train |
| Edge Classifier | `models/edge/classifier.py` | Trade signal classifier |
| Dataset Builder | `scripts/build_dataset_from_parquets.py` | No-DB dataset building |
| Backfill Script | `scripts/backfill_station_city_features.py` | Add missing features |

## 11. Cities Supported

| City | ICAO | NOAA Status |
|------|------|-------------|
| Austin | KAUS | Complete (NBM + HRRR) |
| Chicago | KMDW | NBM only (missing HRRR) |
| Denver | KDEN | No NOAA data |
| Los Angeles | KLAX | No NOAA data |
| Miami | KMIA | No NOAA data |
| Philadelphia | KPHL | No NOAA data |

## 12. Quick Start (Fast Machine)

```bash
# 1. Activate environment
cd /path/to/weather_updated
source .venv/bin/activate

# 2. Verify GPU
nvidia-smi

# 3. Train ordinal model (auto-splits full.parquet)
python models/pipeline/03_train_ordinal.py \
  --city austin \
  --trials 80 \
  --cache-dir data/training_cache

# 4. Check results
cat models/saved/austin/final_metrics_austin.json
```

---

*Created: 2025-12-06*
*Last Updated: 2025-12-06*
