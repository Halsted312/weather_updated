
# 1. Build dataset (~4 hours)
python scripts/build_dataset_from_parquets.py --city chicago --workers 14

# 2. Train ordinal (~2 hours, 150 trials)
python scripts/train_city_ordinal_optuna.py \
    --city chicago --use-cached --trials 150 --workers 14

# 3. Generate edge data (~1-2 hours)
python scripts/train_edge_classifier.py \
    --city chicago --threshold 0.5 --sample-rate 4 --regenerate-only --workers 20

# 4. Sweep
python scripts/sweep_min_edge_threshold.py --city chicago --metric sharpe

# 5. Check recent stability
# (use the script we've been running)














(.venv) (base) halsted@halsted:~/slow_weather_updated$ source .venv/bin/activate
(.venv) (base) halsted@halsted:~/slow_weather_updated$ python scripts/train_edge_classifier.py \
    --city denver \
    --threshold 0.5 \
    --sample-rate 4 \
    --regenerate-only \
    --workers 20
20:47:52 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_denver.parquet
============================================================
ML EDGE CLASSIFIER TRAINING
============================================================
City: denver
Optuna trials: 30
Optuna metric: filtered_precision
Workers: 20
Edge threshold: 0.5°F
Sample rate: every 4th snapshot
P&L mode: REALISTIC (with fees)
Maker fill probability: 40.0%
Ordinal model: models/saved/denver/ordinal_catboost_optuna.pkl (default)
Candle source: parquet (models/candles/candles_denver.parquet)
Settlement source: database

20:47:52 [INFO] __main__: ⚠️  Regenerating: ordinal model changed
20:47:52 [INFO] __main__:    Cached: 1765107178.7230484
20:47:52 [INFO] __main__:    Current: 1765107178.0
20:47:52 [INFO] __main__: Using ordinal model: models/saved/denver/ordinal_catboost_optuna.pkl
20:47:53 [INFO] __main__: Loaded train data: 389,712 rows
20:47:53 [INFO] __main__: Loaded test data: 97,128 rows
/home/halsted/slow_weather_updated/scripts/train_edge_classifier.py:472: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  df_combined = pd.concat(dfs, ignore_index=True)
20:47:55 [INFO] __main__: Combined data: 486,840 rows
20:47:55 [INFO] __main__: Generating edge data for denver with 20 workers...
20:47:55 [INFO] __main__: Processing 1068 unique days
20:47:55 [INFO] __main__: Loading model from models/saved/denver/ordinal_catboost_optuna.pkl...
20:47:55 [INFO] models.training.ordinal_trainer: Loaded ordinal model from models/saved/denver/ordinal_catboost_optuna.pkl
20:47:55 [INFO] __main__: Batch loading settlements...
20:47:56 [INFO] src.db.connection: Database engine created: localhost:5434/kalshi_weather
20:47:56 [INFO] __main__: Loaded 1062 settlements
20:47:56 [INFO] __main__: Days with settlement data: 1062
20:47:56 [INFO] __main__: Loading candles from parquet: models/candles/candles_denver.parquet
20:47:56 [INFO] __main__: Loading candles from parquet: models/candles/candles_denver.parquet
20:47:56 [INFO] __main__: Loaded 4,986,192 candle rows from parquet
20:47:57 [INFO] __main__: Filtered to 4,889,967 rows for requested dates
20:47:57 [INFO] __main__: Organizing 4,889,967 candle rows by (day, bracket)...
20:47:57 [INFO] __main__:   (This may take 10-20 minutes for 10M+ rows - please wait)
20:50:41 [INFO] __main__: Organized into 2,238 (day, bracket) entries from parquet
20:50:41 [INFO] __main__: Candle cache built: 2238 (day, bracket) entries
20:50:41 [INFO] __main__: Sample cache keys: [(datetime.date(2024, 12, 1), '47.5-48.5'), (datetime.date(2024, 12, 1), '49.5-50.5'), (datetime.date(2024, 12, 1), '51.5-52.5'), (datetime.date(2024, 12, 1), '53.5-54.5'), (datetime.date(2024, 12, 1), '47-48')]
20:50:50 [INFO] __main__: Processing 1062 days with 20 threads...
Processing days: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1062/1062 [1:24:40<00:00,  4.78s/it]
22:15:30 [INFO] __main__: Generated 42,426 edge samples
22:15:30 [INFO] __main__: Signals with outcomes: 18,527
22:15:31 [INFO] __main__: Cached edge data to models/saved/denver/edge_training_data_realistic.parquet
22:15:31 [INFO] __main__: Saved cache metadata: models/saved/denver/edge_training_data_realistic.meta.json

============================================================
EDGE DATA GENERATION COMPLETE (--regenerate-only)
============================================================
Output: models/saved/denver/edge_training_data_realistic.parquet
Rows: 42,426
Signals (non-no_trade): 21,347
Valid P&L rows: 18,527
Mean P&L: $0.0666

Copy this file to fast machine for training/sweeps.
(.venv) (base) halsted@halsted:~/slow_weather_updated$ python scripts/train_edge_classifier.py     --city los_angeles     --threshold 1.5     --sample-rate 4     --regenerate-only     --workers 28
22:32:35 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_los_angeles.parquet
============================================================
ML EDGE CLASSIFIER TRAINING
============================================================
City: los_angeles
Optuna trials: 30
Optuna metric: filtered_precision
Workers: 28
Edge threshold: 1.5°F
Sample rate: every 4th snapshot
P&L mode: REALISTIC (with fees)
Maker fill probability: 40.0%
Ordinal model: models/saved/los_angeles/ordinal_catboost_optuna.pkl (default)
Candle source: parquet (models/candles/candles_los_angeles.parquet)
Settlement source: database

22:32:35 [WARNING] __main__: ⚠️  Regenerating: cache metadata missing (old cache format)
22:32:35 [INFO] __main__: Using ordinal model: models/saved/los_angeles/ordinal_catboost_optuna.pkl
22:32:36 [INFO] __main__: Loaded train data: 389,712 rows
22:32:36 [INFO] __main__: Loaded test data: 97,128 rows
/home/halsted/slow_weather_updated/scripts/train_edge_classifier.py:472: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  df_combined = pd.concat(dfs, ignore_index=True)
22:32:38 [INFO] __main__: Combined data: 486,840 rows
22:32:39 [INFO] __main__: Generating edge data for los_angeles with 28 workers...
22:32:39 [INFO] __main__: Processing 1068 unique days
22:32:39 [INFO] __main__: Loading model from models/saved/los_angeles/ordinal_catboost_optuna.pkl...
22:32:39 [INFO] models.training.ordinal_trainer: Loaded ordinal model from models/saved/los_angeles/ordinal_catboost_optuna.pkl
22:32:39 [INFO] __main__: Batch loading settlements...
22:32:40 [INFO] src.db.connection: Database engine created: localhost:5434/kalshi_weather
22:32:40 [INFO] __main__: Loaded 1062 settlements
22:32:40 [INFO] __main__: Days with settlement data: 1062
22:32:40 [INFO] __main__: Loading candles from parquet: models/candles/candles_los_angeles.parquet
22:32:40 [INFO] __main__: Loading candles from parquet: models/candles/candles_los_angeles.parquet
22:32:40 [INFO] __main__: Loaded 4,518,635 candle rows from parquet
22:32:41 [INFO] __main__: Filtered to 4,419,442 rows for requested dates
22:32:41 [INFO] __main__: Organizing 4,419,442 candle rows by (day, bracket)...
22:32:41 [INFO] __main__:   (This may take 10-20 minutes for 10M+ rows - please wait)
22:34:51 [INFO] __main__: Organized into 1,962 (day, bracket) entries from parquet
22:34:51 [INFO] __main__: Candle cache built: 1962 (day, bracket) entries
22:34:51 [INFO] __main__: Sample cache keys: [(datetime.date(2025, 4, 1), '60.5-61.5'), (datetime.date(2025, 4, 1), '62.5-63.5'), (datetime.date(2025, 4, 1), '64.5-65.5'), (datetime.date(2025, 4, 1), '66.5-67.5'), (datetime.date(2025, 4, 1), '60-61')]
22:34:59 [INFO] __main__: Processing 1062 days with 28 threads...
Processing days: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1062/1062 [1:26:41<00:00,  4.90s/it]
00:01:42 [INFO] __main__: Generated 37,203 edge samples
00:01:42 [INFO] __main__: Signals with outcomes: 26,264
00:01:42 [INFO] __main__: Cached edge data to models/saved/los_angeles/edge_training_data_realistic.parquet
00:01:42 [INFO] __main__: Saved cache metadata: models/saved/los_angeles/edge_training_data_realistic.meta.json

============================================================
EDGE DATA GENERATION COMPLETE (--regenerate-only)
============================================================
Output: models/saved/los_angeles/edge_training_data_realistic.parquet
Rows: 37,203
Signals (non-no_trade): 33,329
Valid P&L rows: 26,264
Mean P&L: $0.1485

Copy this file to fast machine for training/sweeps.
(.venv) (base) halsted@halsted:~/slow_weather_updated$ 