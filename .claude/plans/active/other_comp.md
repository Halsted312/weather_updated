All Paths Standardized to models/saved/{city}/
Changes Made
File	Change
03_train_ordinal.py	--cache-dir default: data/training_cache → models/saved
backfill_station_city_features.py	Now defaults to models/saved/{city}/full.parquet (no --input/--output needed)
rebuild_all_cities_fresh.py	base_dir: data/training_cache → models/saved
patch_cloudcover_all_cities.py	base_dir: data/training_cache → models/saved
Simplified Commands
# Backfill a city (defaults to models/saved/{city}/full.parquet)
PYTHONPATH=. python scripts/backfill_station_city_features.py --city chicago

# Train ordinal (defaults to models/saved/)
PYTHONPATH=. python models/pipeline/03_train_ordinal.py --city chicago --trials 80

# Multi-city pipeline (already uses models/saved/)
PYTHONPATH=. python scripts/run_multi_city_pipeline.py \
    --cities austin chicago denver los_angeles miami philadelphia \
    --start 2023-01-01 --end 2025-12-03 \
    --ordinal-trials 80 --edge-trials 30 \
    --skip-build --continue-on-error
Directory Structure
models/saved/
├── austin/
│   ├── full.parquet              # Dataset
│   ├── train_data_full.parquet   # Train split (auto-created)
│   ├── test_data_full.parquet    # Test split (auto-created)
│   ├── ordinal_catboost_optuna.pkl
│   └── edge_classifier.pkl
├── chicago/
│   └── ...
└── ...



What --continue-on-error means:
Without flag (default): If Denver fails, the script stops immediately. Miami/Philadelphia never run. With flag: If Denver fails, the script logs the error and continues to the next city. At the end you see which cities failed.
Updates Made:
File logging: All output goes to logs/multi_city_pipeline_{timestamp}.log
Timing per city: Shows how long each city took
Better summary: Lists success/fail with times and output files
Folder setup: Auto-creates logs/ and models/saved/ directories
Example output at end:
============================================================
PIPELINE SUMMARY
============================================================
Finished: 2025-12-06 22:30:00
Total time: 14400.0s (240.0 min, 4.00 hr)
Log file: logs/multi_city_pipeline_20251206_183000.log

Ordinal Training Results:
----------------------------------------
  [+] denver          SUCCESS    (45.2 min)
  [+] los_angeles     SUCCESS    (48.1 min)
  [+] miami           SUCCESS    (42.3 min)
  [+] philadelphia    SUCCESS    (44.8 min)

Edge Training Results:
----------------------------------------
  [+] denver          SUCCESS    (12.5 min)
  [+] los_angeles     SUCCESS    (13.2 min)
  [+] miami           SUCCESS    (11.8 min)
  [+] philadelphia    SUCCESS    (12.1 min)

Output Files:
----------------------------------------
  models/saved/denver/ordinal_catboost_optuna.pkl
  models/saved/denver/edge_classifier.pkl
  ...
Your overnight command:
PYTHONPATH=. python scripts/run_multi_city_pipeline.py \
    --cities denver los_angeles miami philadelphia \
    --start 2023-01-01 --end 2025-12-03 \
    --ordinal-trials 150 \
    --edge-trials 80 \
    --workers 8 \
    --skip-build \
    --continue-on-error
Check the log in the morning: tail -100 logs/multi_city_pipeline_*.log




11:55:08 [INFO] models.training.ordinal_trainer:   Threshold +11: trained (pos_rate=6.3%)
11:55:08 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['city_warmer_flag']
11:55:11 [INFO] models.training.ordinal_trainer:   Threshold +12: trained (pos_rate=5.5%)
11:55:11 [INFO] models.training.ordinal_trainer: Ordinal training complete: 24 classifiers

============================================================
EVALUATION
============================================================
11:55:11 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['city_warmer_flag']

Test Set Metrics:
----------------------------------------
  delta_accuracy: 0.2745
  delta_mae: 1.9369
  off_by_1_rate: 0.2732
  off_by_2plus_rate: 0.4523
  within_1_rate: 0.5477
  within_2_rate: 0.7410
11:55:13 [INFO] models.training.ordinal_trainer: Saved ordinal model to models/saved/austin/ordinal_catboost_optuna.pkl
11:55:13 [INFO] scripts.train_city_ordinal_optuna: 
Saved model to models/saved/austin/ordinal_catboost_optuna.pkl
11:55:13 [INFO] scripts.train_city_ordinal_optuna: Saved best params to models/saved/austin/best_params.json
11:55:13 [INFO] scripts.train_city_ordinal_optuna: Saved final metrics to models/saved/austin/final_metrics_austin.json

============================================================
FEATURE IMPORTANCE
============================================================

Top 30 Features:
                             feature  importance
0            temp_zscore_vs_forecast    9.903543
1                     fcst_obs_ratio    8.889688
2                   obs_fcst_max_gap    8.389570
3                   remaining_upside    6.318364
4            confidence_weighted_gap    5.610602
5           fcst_remaining_potential    4.650214
6               fcst_peak_hour_float    4.579328
7              gap_x_hours_remaining    4.377441
8              fcst_prev_hour_of_max    4.254851
9                  err_max_pos_sofar    1.724316
10        minutes_since_max_observed    1.649318
11          fcst_peak_band_width_min    1.481161
12             nbm_peak_window_max_f    1.253255
13            hrrr_peak_window_max_f    1.083003
14                  delta_vcmax_lag1    1.059533
15              log_abs_obs_fcst_gap    0.952356
16              obs_fcst_gap_squared    0.899914
17                 num_samples_sofar    0.883882
18  hrrr_minus_nbm_peak_window_max_f    0.862077
19             fcst_multi_t1_t2_diff    0.841512
20                   fcst_prev_std_f    0.794381
21            log_minutes_since_open    0.735938
22    log_expected_delta_uncertainty    0.720440
23                    err_mean_sofar    0.716041
24              hours_until_fcst_max    0.690957
25               fcst_dewpoint_range    0.641931
26                   nbm_t15_z_30d_f    0.641707
27        fcst_humidity_morning_mean    0.584867
28              fcst_cloudcover_mean    0.581684
29                           doy_sin    0.572453

----------------------------------------
Station-city feature importance:
                         feature  importance
47    station_city_max_gap_sofar    0.413686
85   station_city_mean_gap_sofar    0.188057
114         station_city_gap_std    0.075335
117        station_city_temp_gap    0.047181
126       station_city_gap_trend    0.025666
202  station_city_gap_x_fcst_gap    0.000000
  station_city_max_gap_sofar: rank 48/220
  station_city_mean_gap_sofar: rank 86/220
  station_city_gap_std: rank 115/220
  station_city_temp_gap: rank 118/220
  station_city_gap_trend: rank 127/220
  station_city_gap_x_fcst_gap: rank 203/220

----------------------------------------
Multi-horizon feature importance:
                  feature  importance
19  fcst_multi_t1_t2_diff    0.841512
57      fcst_multi_median    0.317684
60         fcst_multi_ema    0.295017
61   fcst_multi_range_pct    0.285673
63       fcst_multi_drift    0.265599
67          fcst_multi_cv    0.248498
73       fcst_multi_range    0.233919
76         fcst_multi_std    0.213431
86        fcst_multi_mean    0.180859
  fcst_multi_t1_t2_diff: rank 20/220
  fcst_multi_median: rank 58/220
  fcst_multi_ema: rank 61/220
  fcst_multi_range_pct: rank 62/220
  fcst_multi_drift: rank 64/220
  fcst_multi_cv: rank 68/220
  fcst_multi_range: rank 74/220
  fcst_multi_std: rank 77/220
  fcst_multi_mean: rank 87/220

============================================================
SUMMARY
============================================================
City: austin
Model: models/saved/austin/ordinal_catboost_optuna.pkl
Params: models/saved/austin/best_params.json
Metrics: models/saved/austin/final_metrics_austin.json
Training samples: 389,784
Test samples: 97,128
Optuna trials: 4
Best params: {'grow_policy': 'Lossguide', 'depth': 5, 'iterations': 863, 'learning_rate': 0.022261785203555637, 'border_count': 87, 'l2_leaf_reg': 37.75657476393088, 'min_data_in_leaf': 46, 'random_strength': 0.732288936155134, 'colsample_bylevel': 0.5319352458851474, 'subsample': 0.7223874032813048, 'max_leaves': 21}

Key Metrics:
  Accuracy: 27.4%
  MAE: 1.94
  Within 1: 54.8%
  Within 2: 74.1%

============================================================
DONE
============================================================
(.venv) (base) halsted@halsted:~/slow_weather_updated$ 



Just run this one command:
python models/pipeline/03_train_ordinal.py --city austin --trials 1 --cache-dir data/training_cache
That's it! It will:
Auto-detect full.parquet and split it into train/test (80/20)
Save the splits for future runs
Train with GPU (already configured in the code)
For full training after the test works:
python models/pipeline/03_train_ordinal.py --city austin --trials 80 --cache-dir data/training_cache


(.venv) (base) halsted@halsted:~/slow_weather_updated$ PYTHONPATH=. python scripts/train_city_ordinal_optuna.py \
  --city austin \
  --trials 1 \
  --use-cached \
  --cache-dir data/training_cache
11:28:53 [INFO] __main__: ============================================================
11:28:53 [INFO] __main__: AUSTIN Optuna Training (1 trials)
11:28:53 [INFO] __main__: Optimization objective: weighted_auc
11:28:53 [INFO] __main__: ============================================================
11:28:53 [ERROR] __main__: --use-cached specified but parquet files not found:
11:28:53 [ERROR] __main__:   Expected: data/training_cache/austin/train_data_full.parquet
11:28:53 [ERROR] __main__:   Expected: data/training_cache/austin/test_data_full.parquet
11:28:53 [ERROR] __main__: Either build the dataset first or remove --use-cached to rebuild from DB

The GPU settings are already in the code. Run this on your fast machine:
# Quick test - 1 trial to verify GPU works
PYTHONPATH=. python scripts/train_city_ordinal_optuna.py \
  --city austin \
  --trials 1 \
  --use-cached \
  --cache-dir data/training_cache

# If that works, run full training (much faster with GPU)
PYTHONPATH=. python scripts/train_city_ordinal_optuna.py \
  --city austin \
  --trials 80 \
  --use-cached \
  --cache-dir data/training_cache



Or use the pipeline script:
python models/pipeline/03_train_ordinal.py \
  --city austin \
  --trials 80 \
  --cache-dir data/training_cache
What changed:
task_type='GPU' - Uses CUDA
devices='0' - First GPU (use '0,1' for both GPUs)
bootstrap_type='MVS' - GPU-optimized sampling (fastest)
border_count=128 - Power of 2 for GPU efficiency


PYTHONPATH=. python scripts/backfill_station_city_features.py \
    --city chicago \
    --input data/training_cache/chicago/full.parquet \
    --output data/training_cache/chicago/full.parquet

    

PYTHONPATH=. python models/pipeline/03_train_ordinal.py \
  --city chicago \
  --trials 12 \
  --workers 24 \
  --cache-dir data/training_cache


.venv) (base) halsted@halsted:~/slow_weather_updated$ PYTHONPATH=. python models/pipeline/03_train_ordinal.py \
  --city chicago \
  --trials 150 \
  --workers 16 \
  --cache-dir data/training_cache
10:49:55 [INFO] train_ordinal_pipeline: Checking for cached data in data/training_cache/chicago...
10:49:55 [INFO] train_ordinal_pipeline: Auto-splitting data/training_cache/chicago/full.parquet with 20% test ratio...
10:49:56 [INFO] train_ordinal_pipeline:   Loaded: 486,912 rows, 256 columns
10:49:56 [INFO] train_ordinal_pipeline:   Total days: 1068
10:49:56 [INFO] train_ordinal_pipeline:   Train days: 855 (2023-01-01 to 2025-05-04)
10:49:56 [INFO] train_ordinal_pipeline:   Test days: 213 (2025-05-05 to 2025-12-03)
10:49:56 [INFO] train_ordinal_pipeline:   Train set: 389,784 rows (855 days)
10:49:56 [INFO] train_ordinal_pipeline:   Test set: 97,128 rows (213 days)
10:49:58 [INFO] train_ordinal_pipeline:   Saved: data/training_cache/chicago/train_data_full.parquet
10:49:58 [INFO] train_ordinal_pipeline:   Saved: data/training_cache/chicago/test_data_full.parquet
10:49:58 [INFO] train_ordinal_pipeline: Train/test split ready.
10:49:58 [INFO] train_ordinal_pipeline: Invoking train_city_ordinal_optuna with args: --city chicago --trials 150 --workers 16 --cv-splits 4 --cache-dir data/training_cache --use-cached
10:49:58 [INFO] scripts.train_city_ordinal_optuna: ============================================================
10:49:58 [INFO] scripts.train_city_ordinal_optuna: CHICAGO Optuna Training (150 trials)
10:49:58 [INFO] scripts.train_city_ordinal_optuna: Optimization objective: weighted_auc
10:49:58 [INFO] scripts.train_city_ordinal_optuna: ============================================================
10:49:58 [INFO] scripts.train_city_ordinal_optuna: Loading cached train/test datasets from: data/training_cache/chicago
10:49:58 [INFO] scripts.train_city_ordinal_optuna:   Train: data/training_cache/chicago/train_data_full.parquet
10:49:58 [INFO] scripts.train_city_ordinal_optuna:   Test:  data/training_cache/chicago/test_data_full.parquet
10:49:58 [INFO] scripts.train_city_ordinal_optuna: Loaded train: 389,784 rows, 256 columns
10:49:58 [INFO] scripts.train_city_ordinal_optuna: Loaded test:  97,128 rows, 256 columns
10:49:58 [INFO] scripts.train_city_ordinal_optuna: 
NOAA feature columns check:
10:49:58 [INFO] scripts.train_city_ordinal_optuna:   ✓ nbm_peak_window_max_f: 389,328/389,784 non-null (99.9%)
10:49:58 [INFO] scripts.train_city_ordinal_optuna:   ✓ hrrr_peak_window_max_f: 389,784/389,784 non-null (100.0%)
10:49:58 [INFO] scripts.train_city_ordinal_optuna:   ✓ nbm_t15_z_30d_f: 389,328/389,784 non-null (99.9%)
10:49:58 [INFO] scripts.train_city_ordinal_optuna:   ✓ hrrr_t15_z_30d_f: 389,784/389,784 non-null (100.0%)
10:49:58 [INFO] scripts.train_city_ordinal_optuna:   ✓ hrrr_minus_nbm_t15_z_30d_f: 389,328/389,784 non-null (99.9%)
10:49:58 [INFO] scripts.train_city_ordinal_optuna: 
Data range: 2023-01-01 to 2025-12-03
10:49:58 [INFO] scripts.train_city_ordinal_optuna: Training: 2023-01-01 to 2025-05-04 (855 days)
10:49:58 [INFO] scripts.train_city_ordinal_optuna: Testing:  2025-05-05 to 2025-12-03 (213 days)
10:49:58 [INFO] scripts.train_city_ordinal_optuna: 
Training samples: 389,784
10:49:58 [INFO] scripts.train_city_ordinal_optuna: Training days: 855
10:49:58 [INFO] scripts.train_city_ordinal_optuna: Test samples: 97,128
10:49:58 [INFO] scripts.train_city_ordinal_optuna: Test days: 213
10:49:58 [INFO] scripts.train_city_ordinal_optuna: 
Station-city features: ['station_city_temp_gap', 'station_city_max_gap_sofar', 'station_city_mean_gap_sofar', 'station_city_gap_std', 'station_city_gap_trend', 'station_city_gap_x_fcst_gap']
10:49:58 [INFO] scripts.train_city_ordinal_optuna:   station_city_temp_gap: 0/389,784 non-null (0.0%)
10:49:58 [INFO] scripts.train_city_ordinal_optuna:   station_city_max_gap_sofar: 0/389,784 non-null (0.0%)
10:49:58 [INFO] scripts.train_city_ordinal_optuna:   station_city_mean_gap_sofar: 0/389,784 non-null (0.0%)
10:49:59 [INFO] scripts.train_city_ordinal_optuna:   station_city_gap_std: 0/389,784 non-null (0.0%)
10:49:59 [INFO] scripts.train_city_ordinal_optuna:   station_city_gap_trend: 0/389,784 non-null (0.0%)
10:49:59 [INFO] scripts.train_city_ordinal_optuna:   station_city_gap_x_fcst_gap: 0/389,784 non-null (0.0%)
10:49:59 [INFO] scripts.train_city_ordinal_optuna: 
Multi-horizon features: ['fcst_multi_mean', 'fcst_multi_median', 'fcst_multi_ema', 'fcst_multi_std', 'fcst_multi_range', 'fcst_multi_t1_t2_diff', 'fcst_multi_drift', 'fcst_multi_cv', 'fcst_multi_range_pct']
10:49:59 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_mean: 389,784/389,784 non-null (100.0%)
10:49:59 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_median: 389,784/389,784 non-null (100.0%)
10:49:59 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_ema: 389,784/389,784 non-null (100.0%)
10:49:59 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_std: 389,784/389,784 non-null (100.0%)
10:49:59 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_range: 389,784/389,784 non-null (100.0%)
10:49:59 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_t1_t2_diff: 387,960/389,784 non-null (99.5%)
10:49:59 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_drift: 387,960/389,784 non-null (99.5%)
10:49:59 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_cv: 389,784/389,784 non-null (100.0%)
10:49:59 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_range_pct: 389,784/389,784 non-null (100.0%)

============================================================
OPTUNA TRAINING (150 trials, objective=weighted_auc)
============================================================
10:49:59 [INFO] models.training.ordinal_trainer: Training ordinal model (catboost) on 389784 samples
10:49:59 [INFO] models.training.ordinal_trainer: City delta range: [-12, 12]
10:49:59 [INFO] models.training.ordinal_trainer: Training 24 threshold classifiers: [-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
10:49:59 [INFO] models.training.ordinal_trainer: Starting Optuna tuning with 150 trials (objective=weighted_auc)
10:49:59 [INFO] models.training.ordinal_trainer: Weighted AUC thresholds: [-1, 0, 1, 2]
10:49:59 [INFO] models.training.ordinal_trainer: Threshold weights: {-1: 0.23301780148925946, 0: 0.24492018691054865, 1: 0.24220217084523027, 2: 0.22690704857309887}
Best trial: 0. Best value: 0.940243:   1%|█▎                                                                                                                                                                                              | 1/150 [00:30<1:16:53, 30.96s/it][W 2025-12-06 10:50:44,136] Trial 1 failed with parameters: {'bootstrap_type': 'Bayesian', 'grow_policy': 'SymmetricTree', 'depth': 5, 'iterations': 1040, 'learning_rate': 0.09666505740867626, 'border_count': 152, 'l2_leaf_reg': 26.55043156033748, 'min_data_in_leaf': 118, 'random_strength': 0.8870037215177122, 'colsample_bylevel': 0.8318608263552383, 'bagging_temperature': 0.8169848076905314} because of the following error: KeyboardInterrupt('').
Traceback (most recent call last):



  
(.venv) (base) halsted@halsted:~/slow_weather_updated$ PYTHONPATH=. python scripts/train_city_ordinal_optuna.py \
  --city chicago \
  --trials 150 \
  --workers 16 \
  --from-parquet \
  --parquet-path data/training_cache/chicago/full.parquet
usage: train_city_ordinal_optuna.py [-h] --city {chicago,austin,denver,los_angeles,miami,philadelphia} [--trials TRIALS] [--workers WORKERS] [--use-cached] [--cv-splits CV_SPLITS] [--holdout-pct HOLDOUT_PCT] [--no-station-city] [--start-date START_DATE]
                                    [--end-date END_DATE] [--objective {auc,within2,weighted_auc}] [--cache-dir CACHE_DIR]
train_city_ordinal_optuna.py: error: unrecognized arguments: --from-parquet --parquet-path data/training_cache/chicago/full.parquet
(.venv) (base) halsted@halsted:~/slow_weather_updated$ PYTHONPATH=. python scripts/train_city_ordinal_optuna.py \
  --city chicago \
  --trials 150 \
  --workers 16 \
  --from-parquet \
  --parquet-path data/training_cache/chicago/full.parquet




PYTHONPATH=. python scripts/build_dataset_from_parquets.py --city austin --workers 32




Usage on Fast Machine
# With 32 workers
PYTHONPATH=. python scripts/build_dataset_from_parquets.py --city austin --workers 8

# With default 8 workers
PYTHONPATH=. python scripts/build_dataset_from_parquets.py --city austin
Expected speedup:
Old: 3.5 hours (307k × 833 scans + sequential)
New: ~10-20 minutes on 32 cores (single scan + parallel process

(base) halsted@halsted:~/slow_weather_updated$ source /home/halsted/slow_weather_updated/.venv/bin/activate
(.venv) (base) halsted@halsted:~/slow_weather_updated$ PYTHONPATH=. python scripts/build_dataset_from_parquets.py --city austin --workers 8
09:42:24 [INFO] __main__: ============================================================
09:42:24 [INFO] __main__: BUILD DATASET FROM PARQUETS: AUSTIN
09:42:24 [INFO] __main__: ============================================================
09:42:24 [INFO] __main__: Loading raw data from models/raw_data/austin...
09:42:24 [INFO] __main__:   Observations: 307,518 rows
09:42:24 [INFO] __main__:   City observations: 307,263 rows
09:42:24 [INFO] __main__:   Settlements: 1,068 rows
09:42:24 [INFO] __main__:   Daily forecasts: 7,693 rows
09:42:24 [INFO] __main__:   Hourly forecasts: 184,638 rows
09:42:24 [INFO] __main__:   NOAA guidance: 2,132 rows
09:42:24 [INFO] __main__:   Candles: 11,245,787 rows
09:42:24 [INFO] __main__: 
Found 1068 days with settlements
09:42:24 [INFO] __main__: Date range: 2023-01-01 to 2025-12-03
09:42:24 [INFO] __main__: 
Train days: 855 (2023-01-01 to 2025-05-04)
09:42:24 [INFO] __main__: Test days: 213 (2025-05-05 to 2025-12-03)
09:42:24 [INFO] __main__: 
--- Pre-computing rolling stats ---
09:42:24 [INFO] __main__: Pre-computing obs_t15 stats for all days...
09:42:24 [INFO] __main__:   Pre-computed obs_t15 stats: 1058/1068 days have valid stats
09:42:24 [INFO] __main__: 
--- Building training dataset ---
09:42:24 [INFO] __main__: Building Training days with 8 workers (9 chunks, 855 days)...
09:42:25 [INFO] __main__: Loading raw data from models/raw_data/austin...
09:42:25 [INFO] __main__: Loading raw data from models/raw_data/austin...
09:42:25 [INFO] __main__: Loading raw data from models/raw_data/austin...
09:42:25 [INFO] __main__: Loading raw data from models/raw_data/austin...
09:42:25 [INFO] __main__: Loading raw data from models/raw_data/austin...
09:42:25 [INFO] __main__: Loading raw data from models/raw_data/austin...
09:42:25 [INFO] __main__: Loading raw data from models/raw_data/austin...
09:42:25 [INFO] __main__: Loading raw data from models/raw_data/austin...
Training days:   0%|                                                                                                                                                                                          | 0/855 [00:00<?, ?it/s]09:42:25 [INFO] __main__:   Observations: 307,518 rows
09:42:25 [INFO] __main__:   Observations: 307,518 rows
09:42:25 [INFO] __main__:   Observations: 307,518 rows
09:42:25 [INFO] __main__:   Observations: 307,518 rows
09:42:25 [INFO] __main__:   Observations: 307,518 rows
09:42:25 [INFO] __main__:   Observations: 307,518 rows
09:42:25 [INFO] __main__:   Observations: 307,518 rows
09:42:25 [INFO] __main__:   Observations: 307,518 rows
09:42:25 [INFO] __main__:   City observations: 307,263 rows
09:42:25 [INFO] __main__:   City observations: 307,263 rows
09:42:25 [INFO] __main__:   City observations: 307,263 rows
09:42:25 [INFO] __main__:   City observations: 307,263 rows
09:42:25 [INFO] __main__:   City observations: 307,263 rows
09:42:25 [INFO] __main__:   City observations: 307,263 rows
09:42:25 [INFO] __main__:   City observations: 307,263 rows
09:42:25 [INFO] __main__:   City observations: 307,263 rows
09:42:25 [INFO] __main__:   Settlements: 1,068 rows
09:42:25 [INFO] __main__:   Settlements: 1,068 rows
09:42:25 [INFO] __main__:   Settlements: 1,068 rows
09:42:25 [INFO] __main__:   Settlements: 1,068 rows
09:42:25 [INFO] __main__:   Settlements: 1,068 rows
09:42:25 [INFO] __main__:   Settlements: 1,068 rows
09:42:25 [INFO] __main__:   Settlements: 1,068 rows
09:42:25 [INFO] __main__:   Settlements: 1,068 rows
09:42:25 [INFO] __main__:   Daily forecasts: 7,693 rows
09:42:25 [INFO] __main__:   Daily forecasts: 7,693 rows
09:42:25 [INFO] __main__:   Daily forecasts: 7,693 rows
09:42:25 [INFO] __main__:   Daily forecasts: 7,693 rows
09:42:25 [INFO] __main__:   Daily forecasts: 7,693 rows
09:42:25 [INFO] __main__:   Daily forecasts: 7,693 rows
09:42:25 [INFO] __main__:   Daily forecasts: 7,693 rows
09:42:25 [INFO] __main__:   Daily forecasts: 7,693 rows
09:42:25 [INFO] __main__:   Hourly forecasts: 184,638 rows
09:42:25 [INFO] __main__:   Hourly forecasts: 184,638 rows
09:42:25 [INFO] __main__:   Hourly forecasts: 184,638 rows
09:42:25 [INFO] __main__:   Hourly forecasts: 184,638 rows
09:42:25 [INFO] __main__:   Hourly forecasts: 184,638 rows
09:42:25 [INFO] __main__:   Hourly forecasts: 184,638 rows
09:42:25 [INFO] __main__:   Hourly forecasts: 184,638 rows
09:42:25 [INFO] __main__:   Hourly forecasts: 184,638 rows
09:42:25 [INFO] __main__:   NOAA guidance: 2,132 rows
09:42:25 [INFO] __main__:   NOAA guidance: 2,132 rows
09:42:25 [INFO] __main__:   NOAA guidance: 2,132 rows
09:42:25 [INFO] __main__:   NOAA guidance: 2,132 rows
09:42:25 [INFO] __main__:   NOAA guidance: 2,132 rows
09:42:25 [INFO] __main__:   NOAA guidance: 2,132 rows
09:42:25 [INFO] __main__:   NOAA guidance: 2,132 rows
09:42:25 [INFO] __main__:   NOAA guidance: 2,132 rows
09:42:26 [INFO] __main__:   Candles: 11,245,787 rows
09:42:26 [INFO] __main__:   Candles: 11,245,787 rows
09:42:26 [INFO] __main__:   Candles: 11,245,787 rows
09:42:26 [INFO] __main__:   Candles: 11,245,787 rows
09:42:26 [INFO] __main__:   Candles: 11,245,787 rows
09:42:26 [INFO] __main__:   Candles: 11,245,787 rows
09:42:26 [INFO] __main__:   Candles: 11,245,787 rows
09:42:26 [INFO] __main__:   Candles: 11,245,787 rows



















# 2. Generate edge data (NOW parquet-based with --from-parquet)
PYTHONPATH=. python scripts/train_edge_classifier.py \
  --city austin \
  --threshold 0.5 \
  --sample-rate 4 \
  --regenerate-only \
  --from-parquet

# 3. Train edge classifier (no DB needed, reads cached edge data)
PYTHONPATH=. python scripts/train_edge_classifier.py \
  --city austin \
  --trials 60 \
  --workers 12 \
  --optuna-metric sharpe \
  --min-trades-for-metric 50 \
  --from-parquet
Code Changes Required


3. New Sweep Script (scripts/sweep_min_edge_threshold.py)
# Run sweep
PYTHONPATH=. python scripts/sweep_min_edge_threshold.py --city austin

# Custom thresholds
PYTHONPATH=. python scripts/sweep_min_edge_threshold.py \
    --city austin \
    --thresholds 0.5,0.75,1.0,1.25,1.5,2.0,2.5 \
    --min-trades 500
4. New Config File (config/edge_thresholds.py)
Stores per-city optimal thresholds
Used by train_edge_classifier.py as default when --threshold not specified
5. Training Script Integration (scripts/train_edge_classifier.py:1273-1277)
Now uses config/edge_thresholds.py for default threshold per city
Training Commands
Edge Classifier (on fast machine):
PYTHONPATH=. python scripts/train_edge_classifier.py \
  --city austin \
  --trials 150 \
  --workers 12 \
  --optuna-metric sharpe \
  --min-trades-for-metric 20



After sweep, update config/edge_thresholds.py with optimal values per city.


.venv) (base) halsted@halsted:/mnt/slow_weather_updated$ PYTHONPATH=. python scripts/train_edge_classifier.py \
  --city austin \
  --trials 150 \
  --workers 12 \
  --optuna-metric sharpe \
  --min-trades-for-metric 20
07:43:48 [INFO] __main__: Using threshold from config: 1.5°F
07:43:48 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_austin.parquet
============================================================
ML EDGE CLASSIFIER TRAINING
============================================================
City: austin
Optuna trials: 150
Optuna metric: sharpe
Workers: 12
Edge threshold: 1.5°F
Sample rate: every 6th snapshot
P&L mode: REALISTIC (with fees)
Maker fill probability: 40.0%
Ordinal model: models/saved/austin/ordinal_catboost_optuna.pkl (default)
Candle source: parquet (models/candles/candles_austin.parquet)

07:43:48 [INFO] __main__: Loading cached edge data from models/saved/austin/edge_training_data_realistic.parquet
07:43:48 [INFO] __main__: Training on 32,794 edge signals

Class balance: 12104/28544 wins (42.4%)

--- REALISTIC P&L STATISTICS ---
Total samples with valid trades: 28,544
Average P&L per trade: $0.0771
Std P&L per trade: $0.4538
Total gross P&L: $2608.01
Total fees paid: $407.78
Total net P&L: $2200.23

Trade roles: {'taker': np.int64(23606), 'maker': np.int64(4938)}
Trade sides: {'yes': np.int64(28544)}
Trade actions: {'buy': np.int64(15869), 'sell': np.int64(12675)}

Entry price range: 2¢ - 98¢
Average entry price: 36.4¢

============================================================
OPTUNA TRAINING (150 trials)
============================================================
07:43:48 [INFO] models.edge.classifier: Training EdgeClassifier with 150 Optuna trials
07:43:48 [INFO] models.edge.classifier: Using day-grouped time splits (DayGroupedTimeSeriesSplit)
07:43:48 [INFO] models.edge.classifier: Using 15 features: ['forecast_temp', 'market_temp', 'edge', 'confidence', 'forecast_uncertainty']...
07:43:48 [INFO] models.edge.classifier: Day splits: total_days=908, train+val_days=772, test_days=136
07:43:49 [INFO] models.edge.classifier: ✓ Leakage checks PASSED
07:43:49 [INFO] models.edge.classifier: Row-wise split: train=21886, val=1073, test=3223
07:43:49 [INFO] models.edge.classifier: Class balance - train: 44.8% positive
07:43:49 [INFO] models.edge.classifier: Starting Optuna optimization with 150 trials
07:47:23 [INFO] models.edge.classifier: Best trial score (sharpe): 0.7794
07:47:23 [INFO] models.edge.classifier: Best params: {'bootstrap_type': 'Bayesian', 'depth': 4, 'iterations': 715, 'learning_rate': 0.10215254191247841, 'l2_leaf_reg': 11.176987776008465, 'min_data_in_leaf': 79, 'random_strength': 0.8141604204825617, 'colsample_bylevel': 0.49925230853160235, 'bagging_temperature': 1.4671417600043133, 'calibration_method': 'sigmoid', 'decision_threshold': 0.7453098565463683}
07:47:23 [INFO] models.edge.classifier: Fitting final model on train+val combined...
07:47:26 [INFO] models.edge.classifier: Test AUC: 0.8491
07:47:26 [INFO] models.edge.classifier: Test accuracy: 68.4%
07:47:26 [INFO] models.edge.classifier: Baseline win rate: 41.1%
07:47:26 [INFO] models.edge.classifier: Filtered win rate: 88.3% (n_trades=402)
07:47:26 [INFO] models.edge.classifier: Mean PnL (all edges): 0.0313
07:47:26 [INFO] models.edge.classifier: Mean PnL (trades): 0.3228
07:47:26 [INFO] models.edge.classifier: Sharpe (trades): 0.9608

============================================================
RESULTS
============================================================
Test AUC: 0.8491
Test Accuracy: 68.4%

Baseline win rate: 41.1%
Filtered win rate: 88.3%
Improvement: +47.2pp

Trades recommended: 402/3223 (12.5%)

Feature Importance:
  edge: 13.6557
  market_temp: 13.3563
  market_uncertainty: 11.9782
  base_temp: 9.2370
  forecast_temp: 9.0746
  obs_fcst_max_gap: 7.9977
  confidence: 7.7256
  predicted_delta: 6.5745
  forecast_uncertainty: 5.9894
  fcst_remaining_potential: 3.9708
  minutes_since_market_open: 3.4430
  hours_to_event_close: 3.0970
  snapshot_hour: 2.9348
  temp_volatility_30min: 0.9655
  market_bid_ask_spread: 0.0000

07:47:26 [INFO] models.edge.classifier: Saved model to models/saved/austin/edge_classifier.pkl
07:47:26 [INFO] models.edge.classifier: Saved metadata to models/saved/austin/edge_classifier.json
Model saved to: models/saved/austin/edge_classifier
(.venv) (base) halsted@halsted:/mnt/slow_weather_updated$ ^C
(.venv) (base) halsted@halsted:/mnt/slow_weather_updated$ PYTHONPATH=. python scripts/sweep_min_edge_threshold.py --city austin
============================================================
MIN EDGE THRESHOLD SWEEP: AUSTIN
============================================================
Thresholds to test: [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
Min trades required: 500
Optimization metric: sharpe

07:51:17 [INFO] __main__: Loading edge data from models/saved/austin/edge_training_data_realistic.parquet
07:51:17 [INFO] __main__: Loaded 70,436 rows
07:51:17 [INFO] __main__: Edge range: [-13.13, 31.52]
07:51:17 [INFO] __main__: Min absolute edge in data: 0.00
07:51:17 [INFO] __main__: Rows with valid P&L: 28,544 (40.5%)

================================================================================
MIN EDGE THRESHOLD SWEEP RESULTS
================================================================================

 Threshold   N Trades     Mean PnL     Sharpe   Hit Rate    Total PnL
--------------------------------------------------------------------------------
      0.50     28,544 $     0.0771      0.170     42.4% $    2200.23
      0.75     28,544 $     0.0771      0.170     42.4% $    2200.23
      1.00     28,544 $     0.0771      0.170     42.4% $    2200.23
      1.25     28,544 $     0.0771      0.170     42.4% $    2200.23
      1.50     28,544 $     0.0771      0.170     42.4% $    2200.23
      1.75     28,530 $     0.0771      0.170     42.4% $    2199.30
      2.00     28,438 $     0.0771      0.170     42.5% $    2191.45
      2.50     27,563 $     0.0773      0.170     42.7% $    2129.43
--------------------------------------------------------------------------------

OPTIMAL THRESHOLD (maximize sharpe, min 500 trades): 2.50°F
  → N trades: 27,563
  → Mean P&L: $0.0773
  → Sharpe: 0.170
  → Hit rate: 42.7%
  → Total P&L: $2129.43

============================================================
SUGGESTED CONFIG UPDATE
============================================================

Add to config/edge_thresholds.py:

EDGE_MIN_THRESHOLD_F = {
    "austin": 2.50,
    # ... other cities
}

(.venv) (base) halsted@halsted:/mnt/slow_weather_updated$ PYTHONPATH=. python scripts/sweep_min_edge_threshold.py --city austin
============================================================
MIN EDGE THRESHOLD SWEEP: AUSTIN
============================================================
Thresholds to test: [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
Min trades required: 500
Optimization metric: sharpe

07:52:37 [INFO] __main__: Loading edge data from models/saved/austin/edge_training_data_realistic.parquet
07:52:37 [INFO] __main__: Loaded 70,436 rows
07:52:37 [INFO] __main__: Edge range: [-13.13, 31.52]
07:52:37 [INFO] __main__: Min absolute edge in data: 0.00
07:52:37 [INFO] __main__: Rows with valid P&L: 28,544 (40.5%)

================================================================================
MIN EDGE THRESHOLD SWEEP RESULTS
================================================================================

 Threshold   N Trades     Mean PnL     Sharpe   Hit Rate    Total PnL
--------------------------------------------------------------------------------
      0.50     28,544 $     0.0771      0.170     42.4% $    2200.23
      0.75     28,544 $     0.0771      0.170     42.4% $    2200.23
      1.00     28,544 $     0.0771      0.170     42.4% $    2200.23
      1.25     28,544 $     0.0771      0.170     42.4% $    2200.23
      1.50     28,544 $     0.0771      0.170     42.4% $    2200.23
      1.75     28,530 $     0.0771      0.170     42.4% $    2199.30
      2.00     28,438 $     0.0771      0.170     42.5% $    2191.45
      2.50     27,563 $     0.0773      0.170     42.7% $    2129.43
--------------------------------------------------------------------------------

OPTIMAL THRESHOLD (maximize sharpe, min 500 trades): 2.50°F
  → N trades: 27,563
  → Mean P&L: $0.0773
  → Sharpe: 0.170
  → Hit rate: 42.7%
  → Total P&L: $2129.43

============================================================
SUGGESTED CONFIG UPDATE
============================================================

Add to config/edge_thresholds.py:

EDGE_MIN_THRESHOLD_F = {
    "austin": 2.50,
    # ... other cities
}

(.venv) (base) halsted@halsted:/mnt/slow_weather_updated$ PYTHONPATH=. python scripts/sweep_min_edge_threshold.py --city austin --metric mean_pnl
============================================================
MIN EDGE THRESHOLD SWEEP: AUSTIN
============================================================
Thresholds to test: [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
Min trades required: 500
Optimization metric: mean_pnl

07:53:43 [INFO] __main__: Loading edge data from models/saved/austin/edge_training_data_realistic.parquet
07:53:43 [INFO] __main__: Loaded 70,436 rows
07:53:43 [INFO] __main__: Edge range: [-13.13, 31.52]
07:53:43 [INFO] __main__: Min absolute edge in data: 0.00
07:53:43 [INFO] __main__: Rows with valid P&L: 28,544 (40.5%)

================================================================================
MIN EDGE THRESHOLD SWEEP RESULTS
================================================================================

 Threshold   N Trades     Mean PnL     Sharpe   Hit Rate    Total PnL
--------------------------------------------------------------------------------
      0.50     28,544 $     0.0771      0.170     42.4% $    2200.23
      0.75     28,544 $     0.0771      0.170     42.4% $    2200.23
      1.00     28,544 $     0.0771      0.170     42.4% $    2200.23
      1.25     28,544 $     0.0771      0.170     42.4% $    2200.23
      1.50     28,544 $     0.0771      0.170     42.4% $    2200.23
      1.75     28,530 $     0.0771      0.170     42.4% $    2199.30
      2.00     28,438 $     0.0771      0.170     42.5% $    2191.45
      2.50     27,563 $     0.0773      0.170     42.7% $    2129.43
--------------------------------------------------------------------------------

OPTIMAL THRESHOLD (maximize mean_pnl, min 500 trades): 2.50°F
  → N trades: 27,563
  → Mean P&L: $0.0773
  → Sharpe: 0.170
  → Hit rate: 42.7%
  → Total P&L: $2129.43

============================================================
SUGGESTED CONFIG UPDATE
============================================================

Add to config/edge_thresholds.py:

EDGE_MIN_THRESHOLD_F = {
    "austin": 2.50,
    # ... other cities
}

(.venv) (base) halsted@halsted:/mnt/slow_weather_updated$ 







To Run Training on Your Other Computer:
PYTHONPATH=. .venv/bin/python scripts/train_city_ordinal_optuna.py \
  --city austin \
  --use-cached \
  --cache-dir models/saved \
  --trials 5 \
  --objective weighted_auc
This will:
Load models/saved/austin/train_data_full.parquet + test_data_full.parquet
Print NOAA column presence and fill rates
Run Optuna with updated search space
Use weighted AUC across thresholds [-1, 0, 1, 2]

Saved model to models/saved/austin/ordinal_catboost_optuna.pkl
07:21:07 [INFO] __main__: Saved best params to models/saved/austin/best_params.json
07:21:07 [INFO] __main__: Saved final metrics to models/saved/austin/final_metrics_austin.json

============================================================
FEATURE IMPORTANCE
============================================================

Top 30 Features:
                             feature  importance
0            temp_zscore_vs_forecast   18.312012
1                   obs_fcst_max_gap   12.750526
2                     fcst_obs_ratio   12.518199
3           fcst_remaining_potential    4.182513
4              gap_x_hours_remaining    3.446757
5              fcst_cloudcover_range    2.363932
6                 vc_frac_part_sofar    2.189674
7            confidence_weighted_gap    2.023329
8                fcst_humidity_range    1.807464
9                pred_ceil_max_sofar    1.779301
10                  remaining_upside    1.716449
11                 fcst_humidity_max    1.328787
12               fcst_cloudcover_max    1.321835
13                   fcst_prev_q25_f    1.138651
14              obs_fcst_gap_squared    1.123736
15  hrrr_minus_nbm_peak_window_max_f    1.089714
16        station_city_max_gap_sofar    1.080878
17                   fcst_prev_q90_f    0.997086
18              log_abs_obs_fcst_gap    0.858906
19        hrrr_minus_nbm_t15_z_30d_f    0.847076
20                   fcst_prev_max_f    0.745834
21              pred_floor_max_sofar    0.743768
22                  fcst_prev_mean_f    0.741841
23                    vc_max_f_sofar    0.725267
24                 err_max_pos_sofar    0.678416
25            hrrr_peak_window_max_f    0.678368
26                  hrrr_t15_z_30d_f    0.651978
27                  fcst_peak_temp_f    0.604109
28                    vc_std_f_sofar    0.560647
29                   t_forecast_base    0.544202

----------------------------------------
Station-city feature importance:
                         feature  importance
16    station_city_max_gap_sofar    1.080878
68   station_city_mean_gap_sofar    0.199196
70   station_city_gap_x_fcst_gap    0.191044
95          station_city_gap_std    0.105639
118       station_city_gap_trend    0.063798
159        station_city_temp_gap    0.012926
  station_city_max_gap_sofar: rank 17/220
  station_city_mean_gap_sofar: rank 69/220
  station_city_gap_x_fcst_gap: rank 71/220
  station_city_gap_std: rank 96/220
  station_city_gap_trend: rank 119/220
  station_city_temp_gap: rank 160/220

----------------------------------------
Multi-horizon feature importance:
                   feature  importance
36           fcst_multi_cv    0.482377
38       fcst_multi_median    0.448445
43        fcst_multi_range    0.406605
51   fcst_multi_t1_t2_diff    0.289451
56          fcst_multi_std    0.270939
64    fcst_multi_range_pct    0.212643
67          fcst_multi_ema    0.204647
94         fcst_multi_mean    0.107750
126       fcst_multi_drift    0.054894
  fcst_multi_cv: rank 37/220
  fcst_multi_median: rank 39/220
  fcst_multi_range: rank 44/220
  fcst_multi_t1_t2_diff: rank 52/220
  fcst_multi_std: rank 57/220
  fcst_multi_range_pct: rank 65/220
  fcst_multi_ema: rank 68/220
  fcst_multi_mean: rank 95/220
  fcst_multi_drift: rank 127/220

============================================================
SUMMARY
============================================================
City: austin
Model: models/saved/austin/ordinal_catboost_optuna.pkl
Params: models/saved/austin/best_params.json
Metrics: models/saved/austin/final_metrics_austin.json
Training samples: 5,472
Test samples: 912
Optuna trials: 100
Best params: {'bootstrap_type': 'Bernoulli', 'grow_policy': 'Lossguide', 'depth': 8, 'iterations': 1042, 'learning_rate': 0.02261533991680945, 'border_count': 147, 'l2_leaf_reg': 10.363671728194547, 'min_data_in_leaf': 74, 'random_strength': 0.7608742252839725, 'colsample_bylevel': 0.758355995786482, 'max_leaves': 19, 'subsample': 0.6527609636754904}

Key Metrics:
  Accuracy: 43.6%
  MAE: 1.31
  Within 1: 58.1%
  Within 2: 83.2%

============================================================
DONE
============================================================






Everything is set up. Here's what was done:
Completed
Candle date parsing bug FIXED in dataset.py:438
Augment script UPDATED with both NOAA + candle features (augment_austin_noaa_features.py)
Verification script created (verify_austin_augmented.py)
Next Steps for You
Option A: NOAA only (fast, ~2 min)
PYTHONPATH=. .venv/bin/python scripts/augment_austin_noaa_features.py --noaa-only
Option B: NOAA + candles (slower, ~15-20 min)
PYTHONPATH=. .venv/bin/python scripts/augment_austin_noaa_features.py
Then swap the files:
cd data/training_cache/austin/
mv full.parquet full_original.parquet
mv full_aug.parquet full.parquet
Then train:
PYTHONPATH=. .venv/bin/python scripts/train_city_ordinal_optuna.py \
  --city austin --cache-dir data/training_cache --trials 80 --objective auc
After the swap, the training script will load full.parquet which now has all the NOAA columns built in - no more "Missing columns" warnings!








Command to Run (with logs to logs/ folder):
cd ~/Documents/python/weather_updated
source .venv/bin/activate

# Run with logging to logs/ folder
PYTHONPATH=/home/halsted/Documents/python/weather_updated python models/pipeline/03_train_ordinal.py \
  --city austin \
  --trials 80 \
  --cv-splits 4 \
  2>&1 | tee logs/train_austin_ordinal_$(date +%Y%m%d_%H%M%S).log
What it does:
Uses cached parquets from models/saved/austin/
Runs 100 Optuna trials (4-fold CV each)
Objective: AUC (ROC-AUC for binary classifiers)
Logs to logs/train_austin_ordinal_TIMESTAMP.log
Saves model to: models/saved/austin/ordinal_catboost_optuna.json
Monitor progress:
tail -f logs/train_austin_ordinal_*.log
Ready to train with updated Optuna params + all 11 NOAA features! 🚀


Best trial: 56. Best value: 0.906894: 100%|██████████| 80/80 [22:40<00:00, 17.01s/it]
05:35:33 [INFO] models.training.ordinal_trainer: Best params: {'bootstrap_type': 'Bayesian', 'grow_policy': 'Lossguide', 'depth': 8, 'iterations': 779, 'learning_rate': 0.047009347253508436, 'border_count': 90, 'l2_leaf_reg': 26.464412065332443, 'min_data_in_leaf': 55, 'random_strength': 0.8832053242699806, 'colsample_bylevel': 0.6925574655385712, 'max_leaves': 27, 'bagging_temperature': 0.8128595679519613}
05:35:33 [INFO] models.training.ordinal_trainer: Best AUC: 0.9069
05:35:34 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:35:37 [INFO] models.training.ordinal_trainer:   Threshold -11: trained (pos_rate=94.5%)
05:35:37 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:35:40 [INFO] models.training.ordinal_trainer:   Threshold -10: trained (pos_rate=93.6%)
05:35:40 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:35:42 [INFO] models.training.ordinal_trainer:   Threshold -9: trained (pos_rate=92.4%)
05:35:42 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:35:45 [INFO] models.training.ordinal_trainer:   Threshold -8: trained (pos_rate=91.6%)
05:35:45 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:35:50 [INFO] models.training.ordinal_trainer:   Threshold -7: trained (pos_rate=89.6%)
05:35:50 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:35:56 [INFO] models.training.ordinal_trainer:   Threshold -6: trained (pos_rate=88.3%)
05:35:56 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:36:10 [INFO] models.training.ordinal_trainer:   Threshold -5: trained (pos_rate=85.8%)
05:36:10 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:36:21 [INFO] models.training.ordinal_trainer:   Threshold -4: trained (pos_rate=83.6%)
05:36:21 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:36:28 [INFO] models.training.ordinal_trainer:   Threshold -3: trained (pos_rate=80.0%)
05:36:28 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:36:42 [INFO] models.training.ordinal_trainer:   Threshold -2: trained (pos_rate=76.6%)
05:36:42 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:36:48 [INFO] models.training.ordinal_trainer:   Threshold -1: trained (pos_rate=70.9%)
05:36:49 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:37:00 [INFO] models.training.ordinal_trainer:   Threshold +0: trained (pos_rate=65.0%)
05:37:00 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:37:13 [INFO] models.training.ordinal_trainer:   Threshold +1: trained (pos_rate=46.5%)
05:37:13 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:37:20 [INFO] models.training.ordinal_trainer:   Threshold +2: trained (pos_rate=38.1%)
05:37:20 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:37:27 [INFO] models.training.ordinal_trainer:   Threshold +3: trained (pos_rate=30.2%)
05:37:27 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:37:31 [INFO] models.training.ordinal_trainer:   Threshold +4: trained (pos_rate=25.4%)
05:37:31 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:37:37 [INFO] models.training.ordinal_trainer:   Threshold +5: trained (pos_rate=19.8%)
05:37:37 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:37:43 [INFO] models.training.ordinal_trainer:   Threshold +6: trained (pos_rate=16.9%)
05:37:43 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:37:48 [INFO] models.training.ordinal_trainer:   Threshold +7: trained (pos_rate=13.8%)
05:37:49 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:37:56 [INFO] models.training.ordinal_trainer:   Threshold +8: trained (pos_rate=11.7%)
05:37:57 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:38:02 [INFO] models.training.ordinal_trainer:   Threshold +9: trained (pos_rate=9.2%)
05:38:02 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:38:08 [INFO] models.training.ordinal_trainer:   Threshold +10: trained (pos_rate=7.9%)
05:38:09 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:38:15 [INFO] models.training.ordinal_trainer:   Threshold +11: trained (pos_rate=6.3%)
05:38:15 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:38:20 [INFO] models.training.ordinal_trainer:   Threshold +12: trained (pos_rate=5.5%)
05:38:20 [INFO] models.training.ordinal_trainer: Ordinal training complete: 24 classifiers
05:38:20 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['nbm_peak_window_max_f', 'nbm_peak_window_revision_1h_f', 'hrrr_peak_window_max_f', 'hrrr_peak_window_revision_1h_f', 'ndfd_tmax_T1_f', 'ndfd_drift_T2_to_T1_f', 'hrrr_minus_nbm_peak_window_max_f', 'ndfd_minus_vc_T1_f', 'nbm_t15_z_30d_f', 'hrrr_t15_z_30d_f', 'hrrr_minus_nbm_t15_z_30d_f']
05:38:22 [INFO] models.training.ordinal_trainer: Saved ordinal model to models/saved/austin/ordinal_catboost_optuna.pkl
05:38:22 [INFO] scripts.train_city_ordinal_optuna: 
Saved model to models/saved/austin/ordinal_catboost_optuna.pkl
05:38:22 [INFO] scripts.train_city_ordinal_optuna: Saved best params to models/saved/austin/best_params.json
05:38:22 [INFO] scripts.train_city_ordinal_optuna: Saved final metrics to models/saved/austin/final_metrics_austin.json

============================================================
EVALUATION
============================================================

Test Set Metrics:
----------------------------------------
  delta_accuracy: 0.2800
  delta_mae: 1.9708
  off_by_1_rate: 0.2688
  off_by_2plus_rate: 0.4512
  within_1_rate: 0.5488
  within_2_rate: 0.7442

============================================================
FEATURE IMPORTANCE
============================================================

Top 30 Features:
                         feature  importance
0                 fcst_obs_ratio   11.088384
1        temp_zscore_vs_forecast    8.711377
2               obs_fcst_max_gap    7.236921
3               remaining_upside    6.197752
4           fcst_peak_hour_float    5.417268
5        confidence_weighted_gap    4.638874
6          gap_x_hours_remaining    4.055266
7          fcst_prev_hour_of_max    3.948648
8       fcst_remaining_potential    2.957120
9              err_max_pos_sofar    2.140387
10    minutes_since_max_observed    1.508594
11      fcst_peak_band_width_min    1.413019
12         fcst_multi_t1_t2_diff    1.126447
13          obs_fcst_gap_squared    1.104352
14          log_abs_obs_fcst_gap    1.090059
15             num_samples_sofar    0.992412
16              delta_vcmax_lag1    0.985529
17               fcst_prev_std_f    0.918941
18           fcst_dewpoint_range    0.910859
19          fcst_cloudcover_mean    0.838730
20            log_hours_to_close    0.824644
21           fcst_cloudcover_min    0.802837
22  fcst_humidity_afternoon_mean    0.777230
23         fcst_cloudcover_range    0.735391
24                err_mean_sofar    0.713537
25    fcst_humidity_morning_mean    0.704619
26             fcst_dewpoint_min    0.688915
27            fcst_humidity_mean    0.672908
28                       doy_sin    0.665186
29             fcst_humidity_min    0.622457

----------------------------------------
Station-city feature importance:
                         feature  importance
32    station_city_max_gap_sofar    0.572244
66   station_city_mean_gap_sofar    0.293926
80          station_city_gap_std    0.215430
115       station_city_gap_trend    0.045619
164  station_city_gap_x_fcst_gap    0.000116
165        station_city_temp_gap    0.000076
  station_city_max_gap_sofar: rank 33/212
  station_city_mean_gap_sofar: rank 67/212
  station_city_gap_std: rank 81/212
  station_city_gap_trend: rank 116/212
  station_city_gap_x_fcst_gap: rank 165/212
  station_city_temp_gap: rank 166/212

----------------------------------------
Multi-horizon feature importance:
                  feature  importance
12  fcst_multi_t1_t2_diff    1.126447
46       fcst_multi_drift    0.493940
47      fcst_multi_median    0.492960
51          fcst_multi_cv    0.441095
56        fcst_multi_mean    0.399135
57   fcst_multi_range_pct    0.372023
59       fcst_multi_range    0.360414
60         fcst_multi_ema    0.355457
71         fcst_multi_std    0.268288
  fcst_multi_t1_t2_diff: rank 13/212
  fcst_multi_drift: rank 47/212
  fcst_multi_median: rank 48/212
  fcst_multi_cv: rank 52/212
  fcst_multi_mean: rank 57/212
  fcst_multi_range_pct: rank 58/212
  fcst_multi_range: rank 60/212
  fcst_multi_ema: rank 61/212
  fcst_multi_std: rank 72/212

============================================================
SUMMARY
============================================================
City: austin
Model: models/saved/austin/ordinal_catboost_optuna.pkl
Params: models/saved/austin/best_params.json
Metrics: models/saved/austin/final_metrics_austin.json
Training samples: 389,784
Test samples: 97,128
Optuna trials: 80
Best params: {'bootstrap_type': 'Bayesian', 'grow_policy': 'Lossguide', 'depth': 8, 'iterations': 779, 'learning_rate': 0.047009347253508436, 'border_count': 90, 'l2_leaf_reg': 26.464412065332443, 'min_data_in_leaf': 55, 'random_strength': 0.8832053242699806, 'colsample_bylevel': 0.6925574655385712, 'max_leaves': 27, 'bagging_temperature': 0.8128595679519613}

Key Metrics:
  Accuracy: 28.0%
  MAE: 1.97
  Within 1: 54.9%
  Within 2: 74.4%

============================================================
DONE
============================================================








PYTHONPATH=. nohup python3 scripts/train_city_ordinal_optuna.py \
    --city chicago \
    --trials 100 \
    --cv-splits 5 \
    --objective auc \
    > logs/auc_training_20251205/chicago_auc_solo.log 2>&1 &
echo "Chicago PID: $!"

cd /home/halsted/Documents/python/weather_updated && \
PYTHONPATH=. .venv/bin/python scripts/train_edge_classifier.py \
  --city austin \
  --trials 80 \
  --optuna-metric sharpe \
  --workers 8 \
  --regenerate \
  2>&1 | tee logs/austin_edge_full_$(date +%Y%m%d_%H%M%S).log

  Denver training is at 61% (639/1055 days), about 17 min remaining for edge data generation, then Optuna. Now the workflow is much cleaner: On your fast machine:
# No --candle-parquet flag needed - auto-detected!
python scripts/train_edge_classifier.py --city los_angeles --trials 80 --optuna-metric sharpe --regenerate
Summary of changes:
export_kalshi_candles.py: Default output is now models/candles/
train_edge_classifier.py: Auto-detects parquet at models/candles/candles_{city}.parquet
Fixed Philadelphia ticker: KXHIGHPHL% → KXHIGHPHIL%
All 6 parquet files are ready in models/candles/ (212 MB total).
Updates complete:
Export script now defaults to models/candles/
Training script auto-detects parquet files - no --candle-parquet flag needed!
Fixed Philadelphia ticker pattern
Denver: 61% (639/1055 days) - ~17 min remaining for edge data, then Optuna On your fast machine, just run:
# Parquet files already in models/candles/ - auto-detected!
python scripts/train_edge_classifier.py --city los_angeles --trials 80 --optuna-metric sharpe --regenerate
python scripts/train_edge_classifier.py --city miami --trials 80 --optuna-metric sharpe --regenerate
python scripts/train_edge_classifier.py --city philadelphia --trials 80 --optuna-metric sharpe --regenerate
The training will print:
Candle source: parquet (models/candles/candles_los_angeles.parquet)