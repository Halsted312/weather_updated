# Chicago Pipeline - Fast Computer Instructions

**Created**: 2025-12-08
**Status**: Chicago has train/test data + ordinal model, just needs edge classifier

---

## Pre-requisites

```bash
# Clone to LOCAL drive (not network mount!)
cd /path/to/local/fast/drive
git clone <your-repo-url> weather_updated
cd weather_updated

# Setup venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Chicago Status

| Component | Status |
|-----------|--------|
| train_data_full.parquet | 389,712 rows, 855 days |
| test_data_full.parquet | 97,128 rows, 213 days |
| ordinal_catboost_optuna.pkl | 31 MB |
| edge_training_data_realistic.parquet | 120,312 rows |
| **edge_classifier.pkl** | **MISSING** |

---

## Step 1: Train Edge Classifier

```bash
cd /path/to/weather_updated
source .venv/bin/activate

# Train the edge classifier for Chicago
# --from-parquet = uses local parquet files, no database needed
# --workers = parallel Optuna trials (use your CPU count)
# --trials = Optuna hyperparameter search iterations

python scripts/train_edge_classifier.py \
    --city chicago \
    --from-parquet \
    --workers 24 \
    --trials 100
```

**Flags explained:**
- `--from-parquet`: Uses `models/raw_data/chicago/` parquets instead of database
- `--workers 16`: Parallel Optuna trials (adjust to your CPU count)
- `--trials 50`: More trials = better hyperparameters (30-100 reasonable)
- `--threshold`: Edge threshold in F (auto-loaded from config if omitted)

---

## Step 2: Verify Model Created

```bash
ls -la models/saved/chicago/edge_classifier.*
# Should see:
#   edge_classifier.pkl (1-6 MB)
#   edge_classifier.json (metadata)
```

---

## Step 3: Test Inference

```bash
python scripts/test_inference_all_cities.py --city chicago
```

---

## Step 4: Push Results Back

```bash
git add models/saved/chicago/
git commit -m "Chicago edge classifier trained"
git push
```

---

## After Completion

Chicago will be **READY FOR LIVE** with:
- Ordinal model (bracket probability predictions)
- Edge classifier (identifies high-edge trading opportunities)

Pull on your main computer:
```bash
git pull
```

---

## Troubleshooting

If edge classifier training fails, check:
1. `edge_training_data_realistic.parquet` exists (120k rows)
2. Required columns present: `edge_realized`, `predicted_bracket`, `market_yes_bid`, etc.

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('models/saved/chicago/edge_training_data_realistic.parquet')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)[:20]}...')
"
```
