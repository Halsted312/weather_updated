# Fast Computer Setup Guide

## Prerequisites

Your fast computer should have:
- ✅ Python 3.11+
- ✅ Mounted access to project folder
- ✅ Virtual environment (or can create one)

## Setup (One-Time on Fast Computer)

```bash
# Navigate to mounted project
cd /path/to/mounted/weather_updated

# Create virtual environment (if needed)
python3.11 -m venv .venv_fast

# Activate
source .venv_fast/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import catboost, pandas, numpy; print('✅ Dependencies ready')"
```

---

## Running Edge Generation on Fast Computer

### **Denver (Full Date Range)**

```bash
cd /path/to/mounted/weather_updated
source .venv_fast/bin/activate  # or .venv if same

# Generate edge data (5-10 min)
python scripts/train_edge_classifier.py \
    --city denver \
    --threshold 0.5 \
    --sample-rate 4 \
    --regenerate-only \
    --workers 12
```

**Expected**:
- Loads Denver ordinal model (already trained)
- Loads Denver train/test parquets (already built)
- Generates edge data for ~1,068 days
- Saves to `models/saved/denver/edge_training_data_realistic.parquet`
- **Time**: ~5-10 minutes

---

### **Los Angeles (Full Date Range)**

First check if LA dataset is built:
```bash
ls -lh models/saved/los_angeles/train_data_full.parquet
ls -lh models/saved/los_angeles/ordinal_catboost_optuna.pkl
```

If **missing**, build first:
```bash
# Build LA dataset (~4 hours)
python scripts/build_dataset_from_parquets.py --city los_angeles --workers 14

# Train LA ordinal (~2 hours, 150 trials)
python scripts/train_city_ordinal_optuna.py \
    --city los_angeles \
    --use-cached \
    --trials 150 \
    --workers 14 \
    --objective weighted_auc
```

If **ready**, generate edge data:
```bash
# Generate edge data (3-5 min - smaller dataset than Miami)
python scripts/train_edge_classifier.py \
    --city los_angeles \
    --threshold 0.5 \
    --sample-rate 4 \
    --regenerate-only \
    --workers 12
```

---

## Threshold Sweep (After Edge Generation)

### **Denver**
```bash
python scripts/sweep_min_edge_threshold.py \
    --city denver \
    --thresholds 0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5 \
    --metric sharpe \
    --min-trades 500
```

**Output**: Will show optimal threshold (e.g., "OPTIMAL: threshold=1.25°F")

### **Los Angeles**
```bash
python scripts/sweep_min_edge_threshold.py \
    --city los_angeles \
    --thresholds 0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5 \
    --metric sharpe \
    --min-trades 500
```

---

## Final Edge Classifier Training

### **Denver** (Use optimal threshold from sweep)
```bash
# Example: sweep found 1.25 is optimal
python scripts/train_edge_classifier.py \
    --city denver \
    --threshold 1.25 \
    --sample-rate 4 \
    --trials 80 \
    --workers 12 \
    --optuna-metric sharpe
```

**Expected**: ~1-2 minutes (cache hit)

### **Los Angeles**
```bash
# Example: sweep found 1.0 is optimal
python scripts/train_edge_classifier.py \
    --city los_angeles \
    --threshold 1.0 \
    --sample-rate 4 \
    --trials 80 \
    --workers 12 \
    --optuna-metric sharpe
```

---

## Parallel Execution Strategy

You can run **Denver and LA in parallel** on the fast computer:

### **Terminal 1 (Denver)**
```bash
cd /path/to/mounted/weather_updated
source .venv_fast/bin/activate

# Denver edge generation
python scripts/train_edge_classifier.py \
    --city denver --threshold 0.5 --sample-rate 4 --regenerate-only --workers 6

# Then sweep + train
python scripts/sweep_min_edge_threshold.py --city denver --metric sharpe
python scripts/train_edge_classifier.py --city denver --threshold <OPTIMAL> --trials 80
```

### **Terminal 2 (Los Angeles)**
```bash
cd /path/to/mounted/weather_updated
source .venv_fast/bin/activate

# LA edge generation (if dataset+ordinal ready)
python scripts/train_edge_classifier.py \
    --city los_angeles --threshold 0.5 --sample-rate 4 --regenerate-only --workers 6

# Then sweep + train
python scripts/sweep_min_edge_threshold.py --city los_angeles --metric sharpe
python scripts/train_edge_classifier.py --city los_angeles --threshold <OPTIMAL> --trials 80
```

**Note**: Use `--workers 6` for each (total 12 workers, leaves CPU headroom)

---

## Verifying Outputs

After each city completes:
```bash
# Check edge cache exists
ls -lh models/saved/denver/edge_training_data_realistic.parquet
ls -lh models/saved/denver/edge_training_data_realistic.meta.json

# Check final models
ls -lh models/saved/denver/edge_classifier.pkl

# Review metrics
cat models/saved/denver/edge_classifier.json
```

---

## Troubleshooting

### "ModuleNotFoundError"
- Install dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (need 3.11+)

### "File not found" errors
- Verify mount point: `ls -lh models/saved/miami/`
- Check paths match between computers

### Permission errors
- Run permission fix from main computer first
- Or run: `chmod -R u+rw models/` on fast computer

---

*Created: 2025-12-07*
*For running edge generation on fast computer with mounted project folder*
