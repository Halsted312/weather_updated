(.venv) (base) halsted@halsted:~/Python/weather_updated$ PYTHONPATH=. python scripts/train_edge_classifier.py     --city austin     --threshold 3.5     --sample-rate 4     --workers 28     --regenerate-only
00:05:28 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_austin.parquet
============================================================
ML EDGE CLASSIFIER TRAINING
============================================================
City: austin
Optuna trials: 30
Optuna metric: filtered_precision
Workers: 28
Edge threshold: 3.5°F
Sample rate: every 4th snapshot
P&L mode: REALISTIC (with fees)
Maker fill probability: 40.0%
Ordinal model: models/saved/austin/ordinal_catboost_optuna.pkl (default)
Candle source: parquet (models/candles/candles_austin.parquet)
Settlement source: database

00:05:28 [INFO] __main__: ⚠️  Regenerating: cached edge data not found
00:05:28 [INFO] __main__: Using ordinal model: models/saved/austin/ordinal_catboost_optuna.pkl
00:05:28 [INFO] __main__: Loaded train data: 163,608 rows
00:05:28 [INFO] __main__: Loaded test data: 82,992 rows
/home/halsted/Python/weather_updated/scripts/train_edge_classifier.py:474: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  df_combined = pd.concat(dfs, ignore_index=True)
00:05:29 [INFO] __main__: Combined data: 246,600 rows
00:05:29 [INFO] __main__: Generating edge data for austin with 28 workers...
00:05:29 [INFO] __main__: Processing 541 unique days
00:05:29 [INFO] __main__: Loading model from models/saved/austin/ordinal_catboost_optuna.pkl...
00:05:29 [INFO] models.training.ordinal_trainer: Loaded ordinal model from models/saved/austin/ordinal_catboost_optuna.pkl
00:05:29 [INFO] __main__: Batch loading settlements...
00:05:29 [INFO] src.db.connection: Database engine created: localhost:5434/kalshi_weather
00:05:29 [INFO] __main__: Loaded 535 settlements
00:05:29 [INFO] __main__: Days with settlement data: 535
00:05:29 [INFO] __main__: Loading candles from parquet: models/candles/candles_austin.parquet
00:05:29 [INFO] __main__: Loading candles from parquet: models/candles/candles_austin.parquet
00:05:30 [INFO] __main__: Loaded 11,245,787 candle rows from parquet
00:05:31 [INFO] __main__: Filtered to 5,931,036 rows for requested dates
00:05:31 [INFO] __main__: Organizing 5,931,036 candle rows by (day, bracket)...
00:05:31 [INFO] __main__:   (This may take 10-20 minutes for 10M+ rows - please wait)
00:09:44 [INFO] __main__: Organized into 2,867 (day, bracket) entries from parquet
00:09:44 [INFO] __main__: Candle cache built: 2867 (day, bracket) entries
00:09:44 [INFO] __main__: Sample cache keys: [(datetime.date(2023, 12, 12), '64.5-65.5'), (datetime.date(2023, 12, 12), '66.5-67.5'), (datetime.date(2023, 12, 12), '68.5-69.5'), (datetime.date(2023, 12, 12), '70.5-71.5'), (datetime.date(2023, 12, 12), '64-65')]
00:09:47 [INFO] __main__: Processing 535 days with 28 threads...
Processing days: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 535/535 [46:51<00:00,  5.25s/it]
00:56:39 [INFO] __main__: Generated 54,185 edge samples
00:56:39 [INFO] __main__: Signals with outcomes: 19,290
00:56:39 [INFO] __main__: Cached edge data to models/saved/austin/edge_training_data_realistic.parquet
00:56:39 [INFO] __main__: Saved cache metadata: models/saved/austin/edge_training_data_realistic.meta.json

============================================================
EDGE DATA GENERATION COMPLETE (--regenerate-only)
============================================================
Output: models/saved/austin/edge_training_data_realistic.parquet
Rows: 54,185
Signals (non-no_trade): 21,485
Valid P&L rows: 19,290
Mean P&L: $0.0919

Copy this file to fast machine for training/sweeps.
(.venv) (base) halsted@halsted:~/Python/weather_updated$ 














iles are HERE on this computer:
models/saved/austin/train_data_full.parquet  (42 MB)
models/saved/austin/test_data_full.parquet   (22 MB)
models/saved/austin/data_full.parquet        (114 MB)
Your fast computer doesn't have them. Quick fix - run from THIS computer:
git add models/saved/austin/
git commit -m "Austin data for fast computer"
git push
Then on FAST computer:
cd ~/Python/weather_updated
git pull
Then run your edge classifier command again on the fast computer.

(.venv) (base) halsted@halsted:~/Python/weather_updated$ PYTHONPATH=. python scripts/train_edge_classifier.py     --city austin     --threshold 3.5     --sample-rate 4     --workers 28     --regenerate-only
23:49:29 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_austin.parquet
============================================================
ML EDGE CLASSIFIER TRAINING
============================================================
City: austin
Optuna trials: 30
Optuna metric: filtered_precision
Workers: 28
Edge threshold: 3.5°F
Sample rate: every 4th snapshot
P&L mode: REALISTIC (with fees)
Maker fill probability: 40.0%
Ordinal model: models/saved/austin/ordinal_catboost_optuna.pkl (default)
Candle source: parquet (models/candles/candles_austin.parquet)
Settlement source: database

23:49:29 [INFO] __main__: ⚠️  Regenerating: cached edge data not found
23:49:29 [INFO] __main__: Using ordinal model: models/saved/austin/ordinal_catboost_optuna.pkl
23:49:29 [WARNING] __main__: Train data not found: models/saved/austin/train_data_full.parquet
23:49:29 [WARNING] __main__: Test data not found: models/saved/austin/test_data_full.parquet
Traceback (most recent call last):
  File "/home/halsted/Python/weather_updated/scripts/train_edge_classifier.py", line 1731, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/halsted/Python/weather_updated/scripts/train_edge_classifier.py", line 1552, in main
    df_combined = load_combined_data(args.city)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/halsted/Python/weather_updated/scripts/train_edge_classifier.py", line 472, in load_combined_data
    raise FileNotFoundError(f"No data found for {city}")
FileNotFoundError: No data found for austin
(.venv) (base) halsted@halsted:~/Python/weather_updated$ 


Done. Old edge files removed. Clean slate for Austin. Run on fast computer:
PYTHONPATH=. python scripts/train_edge_classifier.py \
    --city austin \
    --threshold 2.5 \
    --sample-rate 4 \
    --workers 24 \
    --regenerate-only
This generates edge_training_data_realistic.parquet. Then sweep and train classifier.

(.venv) (base) halsted@halsted:~/Python/weather_updated$ python scripts/train_edge_classifier.py     --city chicago     --from-parquet     --workers 24     --trials 100
20:35:52 [INFO] __main__: Using threshold from config: 10.0°F
20:35:52 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_chicago.parquet
20:35:52 [INFO] __main__: Auto-detected settlements parquet: models/raw_data/chicago/settlements.parquet
============================================================
ML EDGE CLASSIFIER TRAINING
============================================================
City: chicago
Optuna trials: 100
Optuna metric: filtered_precision
Workers: 24
Edge threshold: 10.0°F
Sample rate: every 6th snapshot
P&L mode: REALISTIC (with fees)
Maker fill probability: 40.0%
Ordinal model: models/saved/chicago/ordinal_catboost_optuna.pkl (default)
Candle source: parquet (models/candles/candles_chicago.parquet)
Settlement source: parquet (models/raw_data/chicago/settlements.parquet)
Mode: PARQUET-ONLY (no DB required)

20:35:52 [INFO] __main__: ⚠️  Regenerating: ordinal model changed
20:35:52 [INFO] __main__:    Cached: 1765214224.0
20:35:52 [INFO] __main__:    Current: 1765247638.801286
20:35:52 [INFO] __main__: Using ordinal model: models/saved/chicago/ordinal_catboost_optuna.pkl
20:35:52 [INFO] __main__: Loaded train data: 389,712 rows
20:35:52 [INFO] __main__: Loaded test data: 97,128 rows
/home/halsted/Python/weather_updated/scripts/train_edge_classifier.py:474: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  df_combined = pd.concat(dfs, ignore_index=True)
20:35:54 [INFO] __main__: Combined data: 486,840 rows
20:35:54 [INFO] __main__: Generating edge data for chicago with 24 workers...
20:35:54 [INFO] __main__: Processing 1068 unique days
20:35:54 [INFO] __main__: Loading model from models/saved/chicago/ordinal_catboost_optuna.pkl...
20:35:55 [INFO] models.training.ordinal_trainer: Loaded ordinal model from models/saved/chicago/ordinal_catboost_optuna.pkl
20:35:55 [INFO] __main__: Batch loading settlements...
20:35:55 [INFO] __main__: Loading settlements from parquet: models/raw_data/chicago/settlements.parquet
20:35:55 [INFO] __main__: Loaded 1068 settlements
20:35:55 [INFO] __main__: Days with settlement data: 1068
20:35:55 [INFO] __main__: Loading candles from parquet: models/candles/candles_chicago.parquet
20:35:55 [INFO] __main__: Loading candles from parquet: models/candles/candles_chicago.parquet
20:35:55 [INFO] __main__: Loaded 14,795,744 candle rows from parquet
20:35:57 [INFO] __main__: Filtered to 12,266,604 rows for requested dates
20:35:57 [INFO] __main__: Organizing 12,266,604 candle rows by (day, bracket)...
20:35:57 [INFO] __main__:   (This may take 10-20 minutes for 10M+ rows - please wait)
20:55:18 [INFO] __main__: Organized into 6,401 (day, bracket) entries from parquet
20:55:18 [INFO] __main__: Candle cache built: 6401 (day, bracket) entries
20:55:18 [INFO] __main__: Sample cache keys: [(datetime.date(2023, 4, 1), '50.5-51.5'), (datetime.date(2023, 4, 1), '52.5-53.5'), (datetime.date(2023, 4, 1), '54.5-55.5'), (datetime.date(2023, 4, 1), '56.5-57.5'), (datetime.date(2023, 4, 1), '50-51')]
20:55:26 [INFO] __main__: Processing 1068 days with 24 threads...
Processing days: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1068/1068 [1:03:37<00:00,  3.57s/it]
21:59:04 [INFO] __main__: Generated 80,359 edge samples
21:59:04 [INFO] __main__: Signals with outcomes: 6,415
21:59:04 [INFO] __main__: Cached edge data to models/saved/chicago/edge_training_data_realistic.parquet
21:59:04 [INFO] __main__: Saved cache metadata: models/saved/chicago/edge_training_data_realistic.meta.json
21:59:04 [INFO] __main__: Training on 6,979 edge signals

Class balance: 5210/6415 wins (81.2%)

--- REALISTIC P&L STATISTICS ---
Total samples with valid trades: 6,415
Average P&L per trade: $0.2978
Std P&L per trade: $0.4480
Total gross P&L: $2009.22
Total fees paid: $98.86
Total net P&L: $1910.36

Trade roles: {'taker': np.int64(5098), 'maker': np.int64(1317)}
Trade sides: {'yes': np.int64(6415)}
Trade actions: {'sell': np.int64(6315), 'buy': np.int64(100)}

Entry price range: 2¢ - 98¢
Average entry price: 48.8¢

============================================================
OPTUNA TRAINING (100 trials)
============================================================
21:59:04 [INFO] models.edge.classifier: Training EdgeClassifier with 100 Optuna trials
21:59:04 [INFO] models.edge.classifier: Using day-grouped time splits (DayGroupedTimeSeriesSplit)
21:59:04 [INFO] models.edge.classifier: Using 15 features: ['forecast_temp', 'market_temp', 'edge', 'confidence', 'forecast_uncertainty']...
21:59:04 [INFO] models.edge.classifier: Day splits: total_days=349, train+val_days=297, test_days=52
21:59:04 [INFO] models.edge.classifier: ✓ Leakage checks PASSED
21:59:04 [INFO] models.edge.classifier: Row-wise split: train=4687, val=309, test=385
21:59:04 [INFO] models.edge.classifier: Class balance - train: 81.9% positive
21:59:04 [INFO] models.edge.classifier: Starting Optuna optimization with 100 trials
21:59:35 [INFO] models.edge.classifier: Best trial score (filtered_precision): 0.9404
21:59:35 [INFO] models.edge.classifier: Best params: {'bootstrap_type': 'MVS', 'depth': 5, 'iterations': 428, 'learning_rate': 0.05551639412840685, 'l2_leaf_reg': 4.9599040490859565, 'min_data_in_leaf': 23, 'random_strength': 1.7637169234206294, 'colsample_bylevel': 0.6432305223526185, 'subsample': 0.7015965203814578, 'calibration_method': 'isotonic', 'decision_threshold': 0.8301634129416797}
21:59:35 [INFO] models.edge.classifier: Fitting final model on train+val combined...
21:59:37 [INFO] models.edge.classifier: Test AUC: 0.5725
21:59:37 [INFO] models.edge.classifier: Test accuracy: 34.0%
21:59:37 [INFO] models.edge.classifier: Baseline win rate: 73.2%
21:59:37 [INFO] models.edge.classifier: Filtered win rate: 100.0% (n_trades=28)
21:59:37 [INFO] models.edge.classifier: Mean PnL (all edges): 0.1881
21:59:37 [INFO] models.edge.classifier: Mean PnL (trades): 0.8557
21:59:37 [INFO] models.edge.classifier: Sharpe (trades): 9.0007
21:59:37 [WARNING] models.edge.classifier: ⚠️ Filtered win rate = 100.0% > 90% is unrealistic! Check for data leakage.

============================================================
RESULTS
============================================================
Test AUC: 0.5725
Test Accuracy: 34.0%

Baseline win rate: 73.2%
Filtered win rate: 100.0%
Improvement: +26.8pp

Trades recommended: 28/385 (7.3%)

Feature Importance:
  market_temp: 15.4379
  base_temp: 13.0419
  edge: 10.4947
  forecast_temp: 9.5098
  market_uncertainty: 7.7320
  confidence: 7.2536
  obs_fcst_max_gap: 6.8138
  predicted_delta: 5.5697
  forecast_uncertainty: 5.1587
  hours_to_event_close: 3.8971
  market_bid_ask_spread: 3.8164
  fcst_remaining_potential: 3.6610
  snapshot_hour: 3.6593
  minutes_since_market_open: 3.5165
  temp_volatility_30min: 0.4375

21:59:37 [INFO] models.edge.classifier: Saved model to models/saved/chicago/edge_classifier.pkl
21:59:37 [INFO] models.edge.classifier: Saved metadata to models/saved/chicago/edge_classifier.json
Model saved to: models/saved/chicago/edge_classifier
(.venv) (base) halsted@halsted:~/Python/weather_updated$ 



To buy more LFS storage:
Go to https://github.com/settings/billing
Scroll to "Git LFS Data"
Click "Add more data" to buy data packs ($5/month for 50GB storage + 50GB bandwidth)
Or view your current usage at: https://github.com/settings/billing/summary
On the other computer:
Before pulling, run:
sudo apt install git-lfs -y
git lfs install
git pull


(.venv) (base) halsted@halsted:~/Python/weather_updated$ python scripts/train_edge_classifier.py     --city chicago     --from-parquet     --workers 24     --trials 100
20:02:01 [INFO] __main__: Using threshold from config: 10.0°F
20:02:01 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_chicago.parquet
20:02:01 [INFO] __main__: Auto-detected settlements parquet: models/raw_data/chicago/settlements.parquet
============================================================
ML EDGE CLASSIFIER TRAINING
============================================================
City: chicago
Optuna trials: 100
Optuna metric: filtered_precision
Workers: 24
Edge threshold: 10.0°F
Sample rate: every 6th snapshot
P&L mode: REALISTIC (with fees)
Maker fill probability: 40.0%
Ordinal model: models/saved/chicago/ordinal_catboost_optuna.pkl (default)
Candle source: parquet (models/candles/candles_chicago.parquet)
Settlement source: parquet (models/raw_data/chicago/settlements.parquet)
Mode: PARQUET-ONLY (no DB required)

20:02:01 [INFO] __main__: ⚠️  Regenerating: ordinal model changed
20:02:01 [INFO] __main__:    Cached: 1765214224.0
20:02:01 [INFO] __main__:    Current: 1765214224.5963826
20:02:01 [INFO] __main__: Using ordinal model: models/saved/chicago/ordinal_catboost_optuna.pkl
Traceback (most recent call last):
  File "/home/halsted/Python/weather_updated/scripts/train_edge_classifier.py", line 1731, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/halsted/Python/weather_updated/scripts/train_edge_classifier.py", line 1552, in main
    df_combined = load_combined_data(args.city)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/halsted/Python/weather_updated/scripts/train_edge_classifier.py", line 458, in load_combined_data
    df_train = pd.read_parquet(train_path)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/halsted/Python/weather_updated/.venv/lib/python3.11/site-packages/pandas/io/parquet.py", line 669, in read_parquet
    return impl.read(
           ^^^^^^^^^^
  File "/home/halsted/Python/weather_updated/.venv/lib/python3.11/site-packages/pandas/io/parquet.py", line 265, in read
    pa_table = self.api.parquet.read_table(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/halsted/Python/weather_updated/.venv/lib/python3.11/site-packages/pyarrow/parquet/core.py", line 1844, in read_table
    dataset = ParquetDataset(
              ^^^^^^^^^^^^^^^
  File "/home/halsted/Python/weather_updated/.venv/lib/python3.11/site-packages/pyarrow/parquet/core.py", line 1413, in __init__
    [fragment], schema=schema or fragment.physical_schema,
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
  File "pyarrow/_dataset.pyx", line 1477, in pyarrow._dataset.Fragment.physical_schema.__get__
  File "pyarrow/error.pxi", line 155, in pyarrow.lib.pyarrow_internal_check_status
  File "pyarrow/error.pxi", line 92, in pyarrow.lib.check_status
pyarrow.lib.ArrowInvalid: Could not open Parquet input source '<Buffer>': Parquet file size is 0 bytes
(.venv) (base) halsted@halsted:~/Python/weather_updated$ 



GitHub Workflow - Yes, This Works
On this computer: Commit and push your current state
git add -A && git commit -m "sync for fast computer" && git push
On fast computer: Clone to LOCAL drive
cd /local/fast/drive
git clone <your-repo-url> weather_updated
cd weather_updated
pip install -e .  # or use existing venv
Run everything locally on fast computer (10-50x faster I/O)
When done on fast computer: Push results
git add models/saved/ models/raw_data/  # or whatever changed
git commit -m "Austin + Chicago models built"
git push
Back on this computer: Pull the results
git pull
Note: Large parquet files (50-100MB) in models/saved/ will bloat git history. Consider:
Adding them to .gitignore and using rsync/scp instead
Or using Git LFS for large files
Or just accepting the bloat since it's a private research repo
The local SSD vs network mount difference for parallel parquet processing is huge - probably 10-20x faster


(.venv) (base) halsted@halsted:~/slow_weather_updated$ PYTHONPATH=. python scripts/build_dataset_from_parquets.py --city austin
18:29:42 [INFO] __main__: ============================================================
18:29:42 [INFO] __main__: BUILD DATASET FROM PARQUETS: AUSTIN
18:29:42 [INFO] __main__: ============================================================
18:29:42 [INFO] __main__: Loading raw data from models/raw_data/austin...
18:29:42 [INFO] __main__:   Observations: 307,518 rows
18:29:42 [INFO] __main__:   City observations: 307,263 rows
18:29:42 [INFO] __main__:   Settlements: 1,068 rows
18:29:42 [INFO] __main__:   Daily forecasts: 7,693 rows
18:29:42 [INFO] __main__:   Hourly forecasts: 184,638 rows
18:29:42 [INFO] __main__:   NOAA guidance: 2,132 rows
18:29:43 [INFO] __main__:   Candles: 11,245,787 rows
18:29:43 [INFO] __main__:   Pre-grouping data by date...
18:29:44 [INFO] __main__:   Pre-grouped: 1068 obs days, 941 candle days
18:29:44 [INFO] __main__: 
Found 1068 days with settlements
18:29:44 [INFO] __main__: Date range: 2023-01-01 to 2025-12-03
18:29:44 [INFO] __main__: 
Days to build: 1068 (2023-01-01 to 2025-12-03)
18:29:44 [INFO] __main__: 
--- Pre-computing rolling stats ---
18:29:44 [INFO] __main__: Pre-computing obs_t15 stats for all days...
18:29:44 [INFO] __main__:   Pre-computed obs_t15 stats: 1058/1068 days have valid stats
18:29:44 [INFO] __main__: 
--- Building full dataset ---
18:29:44 [INFO] __main__: Building All days (1068 days)...
All days:   1%|█▎                                                                                                                                       | 10/1068 [01:46<3:17:00, 11.17s/it]