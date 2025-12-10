.venv) (base) halsted@halsted:~/Python/weather_updated$ python scripts/train_city_ordinal_optuna.py --city denver --trials 25 --use-cached
18:21:05 [INFO] __main__: ============================================================
18:21:05 [INFO] __main__: DENVER Optuna Training (25 trials)
18:21:05 [INFO] __main__: Optimization objective: weighted_auc
18:21:05 [INFO] __main__: ============================================================
18:21:05 [INFO] __main__: Loading cached train/test datasets from: models/saved/denver
18:21:05 [INFO] __main__:   Train: models/saved/denver/train_data_full.parquet
18:21:05 [INFO] __main__:   Test:  models/saved/denver/test_data_full.parquet
18:21:05 [INFO] __main__: Loaded train: 389,712 rows, 256 columns
18:21:05 [INFO] __main__: Loaded test:  97,128 rows, 256 columns
18:21:05 [INFO] __main__: 
NOAA feature columns check:
18:21:05 [INFO] __main__:   ✓ nbm_peak_window_max_f: 0/389,712 non-null (0.0%)
18:21:05 [INFO] __main__:   ✓ hrrr_peak_window_max_f: 0/389,712 non-null (0.0%)
18:21:05 [INFO] __main__:   ✓ nbm_t15_z_30d_f: 0/389,712 non-null (0.0%)
18:21:05 [INFO] __main__:   ✓ hrrr_t15_z_30d_f: 0/389,712 non-null (0.0%)
18:21:05 [INFO] __main__:   ✓ hrrr_minus_nbm_t15_z_30d_f: 0/389,712 non-null (0.0%)
18:21:05 [INFO] __main__: 
Data range: 2023-01-01 to 2025-12-03
18:21:05 [INFO] __main__: Training: 2023-01-01 to 2025-05-04 (855 days)
18:21:05 [INFO] __main__: Testing:  2025-05-05 to 2025-12-03 (213 days)
18:21:05 [INFO] __main__: 
Training samples: 389,712
18:21:05 [INFO] __main__: Training days: 855
18:21:05 [INFO] __main__: Test samples: 97,128
18:21:05 [INFO] __main__: Test days: 213
18:21:05 [INFO] __main__: 
Station-city features: ['station_city_temp_gap', 'station_city_max_gap_sofar', 'station_city_mean_gap_sofar', 'station_city_gap_std', 'city_warmer_flag', 'station_city_gap_trend', 'station_city_gap_x_fcst_gap']
18:21:05 [INFO] __main__:   station_city_temp_gap: 389,628/389,712 non-null (100.0%)
18:21:05 [INFO] __main__:   station_city_max_gap_sofar: 389,628/389,712 non-null (100.0%)
18:21:05 [INFO] __main__:   station_city_mean_gap_sofar: 389,628/389,712 non-null (100.0%)
18:21:05 [INFO] __main__:   station_city_gap_std: 389,628/389,712 non-null (100.0%)
18:21:05 [INFO] __main__:   city_warmer_flag: 389,628/389,712 non-null (100.0%)
18:21:05 [INFO] __main__:   station_city_gap_trend: 389,622/389,712 non-null (100.0%)
18:21:05 [INFO] __main__:   station_city_gap_x_fcst_gap: 389,628/389,712 non-null (100.0%)
18:21:05 [INFO] __main__: 
Multi-horizon features: ['fcst_multi_mean', 'fcst_multi_median', 'fcst_multi_ema', 'fcst_multi_std', 'fcst_multi_range', 'fcst_multi_t1_t2_diff', 'fcst_multi_drift', 'fcst_multi_cv', 'fcst_multi_range_pct']
18:21:05 [INFO] __main__:   fcst_multi_mean: 389,712/389,712 non-null (100.0%)
18:21:05 [INFO] __main__:   fcst_multi_median: 389,712/389,712 non-null (100.0%)
18:21:05 [INFO] __main__:   fcst_multi_ema: 389,712/389,712 non-null (100.0%)
18:21:05 [INFO] __main__:   fcst_multi_std: 389,712/389,712 non-null (100.0%)
18:21:05 [INFO] __main__:   fcst_multi_range: 389,712/389,712 non-null (100.0%)
18:21:05 [INFO] __main__:   fcst_multi_t1_t2_diff: 389,712/389,712 non-null (100.0%)
18:21:05 [INFO] __main__:   fcst_multi_drift: 389,712/389,712 non-null (100.0%)
18:21:05 [INFO] __main__:   fcst_multi_cv: 389,712/389,712 non-null (100.0%)
18:21:05 [INFO] __main__:   fcst_multi_range_pct: 389,712/389,712 non-null (100.0%)

============================================================
OPTUNA TRAINING (25 trials, objective=weighted_auc)
============================================================
18:21:05 [INFO] models.training.ordinal_trainer: Training ordinal model (catboost) on 389712 samples
18:21:05 [INFO] models.training.ordinal_trainer: City delta range: [-12, 12]
18:21:05 [INFO] models.training.ordinal_trainer: Training 24 threshold classifiers: [-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
18:21:05 [INFO] models.training.ordinal_trainer: Starting Optuna tuning with 25 trials (objective=weighted_auc)
18:21:05 [INFO] models.training.ordinal_trainer: Weighted AUC thresholds: [-1, 0, 1, 2]
18:21:05 [INFO] models.training.ordinal_trainer: Threshold weights: {-1: np.float64(0.2298528333699857), 0: np.float64(0.24258157649347278), 1: np.float64(0.24661820529346945), 2: np.float64(0.23422195018793415)}
Best trial: 8. Best value: 0.936735: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [15:31<00:00, 37.25s/it]
18:36:37 [INFO] models.training.ordinal_trainer: Best params: {'grow_policy': 'SymmetricTree', 'depth': 6, 'iterations': 1137, 'learning_rate': 0.07818610240549075, 'border_count': 204, 'l2_leaf_reg': 3.1236059728487238, 'min_data_in_leaf': 117, 'random_strength': 1.4856356362929928, 'colsample_bylevel': 0.6120568490452296, 'subsample': 0.6194979378889551}
18:36:37 [INFO] models.training.ordinal_trainer: Best Weighted AUC: 0.9367
18:36:40 [INFO] models.training.ordinal_trainer:   Threshold -11: trained (pos_rate=88.5%)
18:36:45 [INFO] models.training.ordinal_trainer:   Threshold -10: trained (pos_rate=87.6%)
18:36:47 [INFO] models.training.ordinal_trainer:   Threshold -9: trained (pos_rate=85.7%)
18:36:48 [INFO] models.training.ordinal_trainer:   Threshold -8: trained (pos_rate=84.2%)
18:36:50 [INFO] models.training.ordinal_trainer:   Threshold -7: trained (pos_rate=81.5%)
18:36:52 [INFO] models.training.ordinal_trainer:   Threshold -6: trained (pos_rate=80.1%)
18:36:54 [INFO] models.training.ordinal_trainer:   Threshold -5: trained (pos_rate=77.5%)
18:36:55 [INFO] models.training.ordinal_trainer:   Threshold -4: trained (pos_rate=75.4%)
18:36:57 [INFO] models.training.ordinal_trainer:   Threshold -3: trained (pos_rate=71.3%)
18:37:00 [INFO] models.training.ordinal_trainer:   Threshold -2: trained (pos_rate=69.0%)
18:37:01 [INFO] models.training.ordinal_trainer:   Threshold -1: trained (pos_rate=64.2%)
18:37:03 [INFO] models.training.ordinal_trainer:   Threshold +0: trained (pos_rate=58.6%)
18:37:06 [INFO] models.training.ordinal_trainer:   Threshold +1: trained (pos_rate=44.2%)
18:37:12 [INFO] models.training.ordinal_trainer:   Threshold +2: trained (pos_rate=37.4%)
18:37:13 [INFO] models.training.ordinal_trainer:   Threshold +3: trained (pos_rate=32.3%)
18:37:15 [INFO] models.training.ordinal_trainer:   Threshold +4: trained (pos_rate=29.2%)
18:37:17 [INFO] models.training.ordinal_trainer:   Threshold +5: trained (pos_rate=25.3%)
18:37:19 [INFO] models.training.ordinal_trainer:   Threshold +6: trained (pos_rate=22.7%)
18:37:21 [INFO] models.training.ordinal_trainer:   Threshold +7: trained (pos_rate=19.5%)
18:37:24 [INFO] models.training.ordinal_trainer:   Threshold +8: trained (pos_rate=17.2%)
18:37:25 [INFO] models.training.ordinal_trainer:   Threshold +9: trained (pos_rate=15.2%)
18:37:27 [INFO] models.training.ordinal_trainer:   Threshold +10: trained (pos_rate=13.0%)
18:37:29 [INFO] models.training.ordinal_trainer:   Threshold +11: trained (pos_rate=11.0%)
18:37:31 [INFO] models.training.ordinal_trainer:   Threshold +12: trained (pos_rate=9.5%)
18:37:31 [INFO] models.training.ordinal_trainer: Ordinal training complete: 24 classifiers

============================================================
EVALUATION
============================================================

Test Set Metrics:
----------------------------------------
  delta_accuracy: 0.1957
  delta_mae: 4.6009
  off_by_1_rate: 0.1607
  off_by_2plus_rate: 0.6436
  within_1_rate: 0.3564
  within_2_rate: 0.5058
18:37:33 [INFO] models.training.ordinal_trainer: Saved ordinal model to models/saved/denver/ordinal_catboost_optuna.pkl
18:37:33 [INFO] __main__: 
Saved model to models/saved/denver/ordinal_catboost_optuna.pkl
18:37:33 [INFO] __main__: Saved best params to models/saved/denver/best_params.json
18:37:33 [INFO] __main__: Saved final metrics to models/saved/denver/final_metrics_denver.json

============================================================
FEATURE IMPORTANCE
============================================================

Top 30 Features:
                           feature  importance
0            fcst_prev_hour_of_max   10.994010
1          temp_zscore_vs_forecast    9.325909
2             fcst_peak_hour_float    4.692571
3          confidence_weighted_gap    3.836804
4                   fcst_obs_ratio    3.308182
5              fcst_cloudcover_max    3.213779
6                 obs_fcst_max_gap    3.034073
7             log_abs_obs_fcst_gap    2.332155
8             obs_fcst_gap_squared    2.324329
9                    fcst_multi_cv    2.268940
10        fcst_peak_band_width_min    2.230776
11                delta_vcmax_lag1    2.162897
12            fcst_cloudcover_mean    2.139872
13      minutes_since_max_observed    2.073591
14           fcst_multi_t1_t2_diff    2.059343
15           gap_x_hours_remaining    1.766939
16  log_expected_delta_uncertainty    1.684912
17            fcst_multi_range_pct    1.582331
18             fcst_dewpoint_range    1.531671
19      expected_delta_uncertainty    1.487878
20               err_max_pos_sofar    1.414681
21      fcst_humidity_morning_mean    1.288947
22                fcst_drift_std_f    1.280438
23             fcst_humidity_range    1.268688
24                         doy_cos    1.190539
25        fcst_remaining_potential    1.095733
26                remaining_upside    1.080600
27                 fcst_prev_std_f    1.070763
28                           month    0.976738
29               fcst_dewpoint_min    0.904126

----------------------------------------
Station-city feature importance:
                         feature  importance
53    station_city_max_gap_sofar    0.406499
74   station_city_mean_gap_sofar    0.223615
106       station_city_gap_trend    0.040685
107  station_city_gap_x_fcst_gap    0.038592
108         station_city_gap_std    0.037303
184        station_city_temp_gap    0.000000
  station_city_max_gap_sofar: rank 54/220
  station_city_mean_gap_sofar: rank 75/220
  station_city_gap_trend: rank 107/220
  station_city_gap_x_fcst_gap: rank 108/220
  station_city_gap_std: rank 109/220
  station_city_temp_gap: rank 185/220

----------------------------------------
Multi-horizon feature importance:
                  feature  importance
9           fcst_multi_cv    2.268940
14  fcst_multi_t1_t2_diff    2.059343
17   fcst_multi_range_pct    1.582331
30         fcst_multi_std    0.847269
33       fcst_multi_range    0.811359
50       fcst_multi_drift    0.469530
56        fcst_multi_mean    0.374292
61         fcst_multi_ema    0.332189
66      fcst_multi_median    0.303320
  fcst_multi_cv: rank 10/220
  fcst_multi_t1_t2_diff: rank 15/220
  fcst_multi_range_pct: rank 18/220
  fcst_multi_std: rank 31/220
  fcst_multi_range: rank 34/220
  fcst_multi_drift: rank 51/220
  fcst_multi_mean: rank 57/220
  fcst_multi_ema: rank 62/220
  fcst_multi_median: rank 67/220

============================================================
SUMMARY
============================================================
City: denver
Model: models/saved/denver/ordinal_catboost_optuna.pkl
Params: models/saved/denver/best_params.json
Metrics: models/saved/denver/final_metrics_denver.json
Training samples: 389,712
Test samples: 97,128
Optuna trials: 25
Best params: {'grow_policy': 'SymmetricTree', 'depth': 6, 'iterations': 1137, 'learning_rate': 0.07818610240549075, 'border_count': 204, 'l2_leaf_reg': 3.1236059728487238, 'min_data_in_leaf': 117, 'random_strength': 1.4856356362929928, 'colsample_bylevel': 0.6120568490452296, 'subsample': 0.6194979378889551}

Key Metrics:
  Accuracy: 19.6%
  MAE: 4.60
  Within 1: 35.6%
  Within 2: 50.6%

============================================================
DONE
============================================================





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