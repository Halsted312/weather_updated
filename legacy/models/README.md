# Legacy Model Code - Archived 2025-12-11

## Purpose
Archived model code that is no longer part of the active ML pipeline.

## Subfolders

### deprecated/
Old dataset builders superseded by the unified pipeline:
- `market_clock_dataset_builder.py` (51KB) - Original market clock builder
- `snapshot_builder.py` (16KB) - Old snapshot logic
- `tod_dataset_builder.py` (9KB) - TOD v1 dataset builder

### training/
Unused trainers:
- `logistic_trainer.py` - Logistic regression trainer (CatBoost is now primary)

### edge/
Experimental linear edge models:
- `linear_elastic.py` - ElasticNet edge model
- `linear_lasso.py` - Lasso edge model
- `linear_ridge.py` - Ridge edge model

## Current Pipeline
The active pipeline uses:
- `models/data/dataset.py` - Unified dataset builder
- `models/data/loader.py` - Data loading from DB
- `models/training/ordinal_trainer.py` - CatBoost ordinal regression
- `models/edge/classifier.py` - CatBoost edge classifier

## Revival Notes
The deprecated dataset builders contain useful reference code but have been
superseded by the unified approach in `models/data/dataset.py`.

The linear edge models were experimental alternatives to CatBoost.
They could be revived for comparison or ensemble purposes.
