Station-city features: ['station_city_temp_gap', 'station_city_max_gap_sofar', 'station_city_mean_gap_sofar', 'station_city_gap_std', 'city_warmer_flag', 'station_city_gap_trend', 'station_city_gap_x_fcst_gap']
19:30:01 [INFO] scripts.train_city_ordinal_optuna:   station_city_temp_gap: 389,664/389,772 non-null (100.0%)
19:30:01 [INFO] scripts.train_city_ordinal_optuna:   station_city_max_gap_sofar: 389,664/389,772 non-null (100.0%)
19:30:01 [INFO] scripts.train_city_ordinal_optuna:   station_city_mean_gap_sofar: 389,664/389,772 non-null (100.0%)
19:30:01 [INFO] scripts.train_city_ordinal_optuna:   station_city_gap_std: 389,664/389,772 non-null (100.0%)
19:30:01 [INFO] scripts.train_city_ordinal_optuna:   city_warmer_flag: 389,664/389,772 non-null (100.0%)
19:30:01 [INFO] scripts.train_city_ordinal_optuna:   station_city_gap_trend: 389,658/389,772 non-null (100.0%)
19:30:01 [INFO] scripts.train_city_ordinal_optuna:   station_city_gap_x_fcst_gap: 389,664/389,772 non-null (100.0%)
19:30:01 [INFO] scripts.train_city_ordinal_optuna: 
Multi-horizon features: ['fcst_multi_mean', 'fcst_multi_median', 'fcst_multi_ema', 'fcst_multi_std', 'fcst_multi_range', 'fcst_multi_t1_t2_diff', 'fcst_multi_drift', 'fcst_multi_cv', 'fcst_multi_range_pct']
19:30:01 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_mean: 389,772/389,772 non-null (100.0%)
19:30:01 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_median: 389,772/389,772 non-null (100.0%)
19:30:01 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_ema: 389,772/389,772 non-null (100.0%)
19:30:01 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_std: 389,772/389,772 non-null (100.0%)
19:30:01 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_range: 389,772/389,772 non-null (100.0%)
19:30:01 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_t1_t2_diff: 389,772/389,772 non-null (100.0%)
19:30:01 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_drift: 389,772/389,772 non-null (100.0%)
19:30:01 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_cv: 389,772/389,772 non-null (100.0%)
19:30:01 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_range_pct: 389,772/389,772 non-null (100.0%)

============================================================
OPTUNA TRAINING (150 trials, objective=weighted_auc)
============================================================
19:30:01 [INFO] models.training.ordinal_trainer: Training ordinal model (catboost) on 389772 samples
19:30:01 [INFO] models.training.ordinal_trainer: City delta range: [-12, 12]
19:30:01 [INFO] models.training.ordinal_trainer: Training 24 threshold classifiers: [-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
19:30:01 [INFO] models.training.ordinal_trainer: Starting Optuna tuning with 150 trials (objective=weighted_auc)
19:30:01 [INFO] models.training.ordinal_trainer: Weighted AUC thresholds: [-1, 0, 1, 2]
19:30:01 [INFO] models.training.ordinal_trainer: Threshold weights: {-1: np.float64(0.2300269020219402), 0: np.float64(0.24175037285441844), 1: np.float64(0.24598744586430718), 2: np.float64(0.23331766227003778)}
Best trial: 143. Best value: 0.945717: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 150/150 [2:01:52<00:00, 48.75s/it]
21:31:53 [INFO] models.training.ordinal_trainer: Best params: {'grow_policy': 'SymmetricTree', 'depth': 5, 'iterations': 842, 'learning_rate': 0.051610104720249614, 'border_count': 158, 'l2_leaf_reg': 39.35016207460371, 'min_data_in_leaf': 30, 'random_strength': 0.25216826092689326, 'colsample_bylevel': 0.5976934631215683, 'subsample': 0.817499467900865}
21:31:53 [INFO] models.training.ordinal_trainer: Best Weighted AUC: 0.9457
21:31:57 [INFO] models.training.ordinal_trainer:   Threshold -11: trained (pos_rate=93.9%)
21:32:01 [INFO] models.training.ordinal_trainer:   Threshold -10: trained (pos_rate=92.6%)
21:32:04 [INFO] models.training.ordinal_trainer:   Threshold -9: trained (pos_rate=90.3%)
21:32:06 [INFO] models.training.ordinal_trainer:   Threshold -8: trained (pos_rate=88.6%)
21:32:08 [INFO] models.training.ordinal_trainer:   Threshold -7: trained (pos_rate=86.2%)
21:32:11 [INFO] models.training.ordinal_trainer:   Threshold -6: trained (pos_rate=83.8%)
21:32:13 [INFO] models.training.ordinal_trainer:   Threshold -5: trained (pos_rate=80.3%)
21:32:16 [INFO] models.training.ordinal_trainer:   Threshold -4: trained (pos_rate=77.8%)
21:32:19 [INFO] models.training.ordinal_trainer:   Threshold -3: trained (pos_rate=72.6%)
21:32:23 [INFO] models.training.ordinal_trainer:   Threshold -2: trained (pos_rate=69.7%)
21:32:26 [INFO] models.training.ordinal_trainer:   Threshold -1: trained (pos_rate=64.1%)
21:32:28 [INFO] models.training.ordinal_trainer:   Threshold +0: trained (pos_rate=59.1%)
21:32:31 [INFO] models.training.ordinal_trainer:   Threshold +1: trained (pos_rate=43.7%)
21:32:34 [INFO] models.training.ordinal_trainer:   Threshold +2: trained (pos_rate=37.1%)
21:32:36 [INFO] models.training.ordinal_trainer:   Threshold +3: trained (pos_rate=29.8%)
21:32:39 [INFO] models.training.ordinal_trainer:   Threshold +4: trained (pos_rate=25.2%)
21:32:45 [INFO] models.training.ordinal_trainer:   Threshold +5: trained (pos_rate=21.0%)
21:32:50 [INFO] models.training.ordinal_trainer:   Threshold +6: trained (pos_rate=18.1%)
21:32:53 [INFO] models.training.ordinal_trainer:   Threshold +7: trained (pos_rate=14.3%)
21:32:56 [INFO] models.training.ordinal_trainer:   Threshold +8: trained (pos_rate=11.5%)
21:33:03 [INFO] models.training.ordinal_trainer:   Threshold +9: trained (pos_rate=9.4%)
21:33:09 [INFO] models.training.ordinal_trainer:   Threshold +10: trained (pos_rate=8.1%)
21:33:13 [INFO] models.training.ordinal_trainer:   Threshold +11: trained (pos_rate=6.0%)
21:33:16 [INFO] models.training.ordinal_trainer:   Threshold +12: trained (pos_rate=4.8%)
21:33:16 [INFO] models.training.ordinal_trainer: Ordinal training complete: 24 classifiers

============================================================
EVALUATION
============================================================

Test Set Metrics:
----------------------------------------
  delta_accuracy: 0.2219
  delta_mae: 2.1430
  off_by_1_rate: 0.2577
  off_by_2plus_rate: 0.5204
  within_1_rate: 0.4796
  within_2_rate: 0.6528
21:33:20 [INFO] models.training.ordinal_trainer: Saved ordinal model to models/saved/philadelphia/ordinal_catboost_optuna.pkl
21:33:20 [INFO] scripts.train_city_ordinal_optuna: 
Saved model to models/saved/philadelphia/ordinal_catboost_optuna.pkl
21:33:20 [INFO] scripts.train_city_ordinal_optuna: Saved best params to models/saved/philadelphia/best_params.json
21:33:20 [INFO] scripts.train_city_ordinal_optuna: Saved final metrics to models/saved/philadelphia/final_metrics_philadelphia.json

============================================================
FEATURE IMPORTANCE
============================================================

Top 30 Features:
                           feature  importance
0                 obs_fcst_max_gap    9.451294
1          temp_zscore_vs_forecast    7.872054
2                   fcst_obs_ratio    6.377408
3             log_abs_obs_fcst_gap    4.908186
4          confidence_weighted_gap    3.991402
5             obs_fcst_gap_squared    3.728403
6            fcst_prev_hour_of_max    3.603768
7             fcst_peak_hour_float    3.377173
8                 remaining_upside    3.268415
9            fcst_multi_t1_t2_diff    2.736762
10                delta_vcmax_lag1    2.631544
11        fcst_peak_band_width_min    2.440394
12      minutes_since_max_observed    2.180814
13           gap_x_hours_remaining    2.067639
14             fcst_cloudcover_min    1.952855
15            fcst_cloudcover_mean    1.843206
16                 fcst_prev_std_f    1.676733
17             fcst_cloudcover_max    1.664850
18                         doy_cos    1.435190
19               err_max_pos_sofar    1.279116
20        fcst_remaining_potential    1.120044
21               fcst_humidity_min    1.119361
22             fcst_dewpoint_range    1.098643
23      fcst_humidity_morning_mean    0.894309
24  log_expected_delta_uncertainty    0.807083
25      expected_delta_uncertainty    0.797236
26     fcst_drift_slope_f_per_lead    0.706378
27    fcst_humidity_afternoon_mean    0.701455
28               fcst_dewpoint_max    0.697293
29             max_evening_f_sofar    0.679716

----------------------------------------
Station-city feature importance:
                         feature  importance
48    station_city_max_gap_sofar    0.413868
83   station_city_mean_gap_sofar    0.177766
101         station_city_gap_std    0.091753
109       station_city_gap_trend    0.036632
137  station_city_gap_x_fcst_gap    0.000327
154        station_city_temp_gap    0.000000
  station_city_max_gap_sofar: rank 49/220
  station_city_mean_gap_sofar: rank 84/220
  station_city_gap_std: rank 102/220
  station_city_gap_trend: rank 110/220
  station_city_gap_x_fcst_gap: rank 138/220
  station_city_temp_gap: rank 155/220

----------------------------------------
Multi-horizon feature importance:
                  feature  importance
9   fcst_multi_t1_t2_diff    2.736762
36   fcst_multi_range_pct    0.537044
39       fcst_multi_drift    0.495702
42       fcst_multi_range    0.473348
45          fcst_multi_cv    0.444701
56        fcst_multi_mean    0.361564
62      fcst_multi_median    0.335100
63         fcst_multi_ema    0.332932
74         fcst_multi_std    0.223100
  fcst_multi_t1_t2_diff: rank 10/220
  fcst_multi_range_pct: rank 37/220
  fcst_multi_drift: rank 40/220
  fcst_multi_range: rank 43/220
  fcst_multi_cv: rank 46/220
  fcst_multi_mean: rank 57/220
  fcst_multi_median: rank 63/220
  fcst_multi_ema: rank 64/220
  fcst_multi_std: rank 75/220

============================================================
SUMMARY
============================================================
City: philadelphia
Model: models/saved/philadelphia/ordinal_catboost_optuna.pkl
Params: models/saved/philadelphia/best_params.json
Metrics: models/saved/philadelphia/final_metrics_philadelphia.json
Training samples: 389,772
Test samples: 97,128
Optuna trials: 150
Best params: {'grow_policy': 'SymmetricTree', 'depth': 5, 'iterations': 842, 'learning_rate': 0.051610104720249614, 'border_count': 158, 'l2_leaf_reg': 39.35016207460371, 'min_data_in_leaf': 30, 'random_strength': 0.25216826092689326, 'colsample_bylevel': 0.5976934631215683, 'subsample': 0.817499467900865}

Key Metrics:
  Accuracy: 22.2%
  MAE: 2.14
  Within 1: 48.0%
  Within 2: 65.3%

============================================================
DONE
============================================================
2025-12-06 21:33:20 [INFO] SUCCESS: Train ordinal model for philadelphia completed in 7400.1s (123.3 min)
2025-12-06 21:33:20 [INFO] 
PHILADELPHIA ORDINAL COMPLETED in 8844.5s (147.4 min)
2025-12-06 21:33:20 [INFO] 
============================================================
2025-12-06 21:33:20 [INFO] PHASE 2: EDGE CLASSIFIER TRAINING
2025-12-06 21:33:20 [INFO] ============================================================
2025-12-06 21:33:20 [INFO] 
############################################################
2025-12-06 21:33:20 [INFO] # EDGE TRAINING: DENVER
2025-12-06 21:33:20 [INFO] # Trials: 80, Threshold: 1.5°F, Sample rate: 4
2025-12-06 21:33:20 [INFO] ############################################################
2025-12-06 21:33:20 [INFO] 
============================================================
2025-12-06 21:33:20 [INFO] RUNNING: Train edge classifier for denver
2025-12-06 21:33:20 [INFO] Command: /home/halsted/Documents/python/weather_updated/.venv/bin/python scripts/train_edge_classifier.py --city denver --trials 80 --workers 12 --threshold 1.5 --sample-rate 4 --optuna-metric sharpe
2025-12-06 21:33:20 [INFO] ============================================================
21:33:21 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_denver.parquet
============================================================
ML EDGE CLASSIFIER TRAINING
============================================================
City: denver
Optuna trials: 80
Optuna metric: sharpe
Workers: 12
Edge threshold: 1.5°F
Sample rate: every 4th snapshot
P&L mode: REALISTIC (with fees)
Maker fill probability: 40.0%
Ordinal model: models/saved/denver/ordinal_catboost_optuna.pkl (default)
Candle source: parquet (models/candles/candles_denver.parquet)
Settlement source: database

21:33:21 [INFO] __main__: Loading cached edge data from models/saved/denver/edge_training_data_realistic.parquet
21:33:21 [INFO] __main__: Training on 8,503 edge signals

Class balance: 3816/7544 wins (50.6%)

--- REALISTIC P&L STATISTICS ---
Total samples with valid trades: 7,544
Average P&L per trade: $0.0782
Std P&L per trade: $0.4165
Total gross P&L: $716.59
Total fees paid: $126.66
Total net P&L: $589.93

Trade roles: {'taker': np.int64(7201), 'maker': np.int64(343)}
Trade sides: {'yes': np.int64(7544)}
Trade actions: {'sell': np.int64(4120), 'buy': np.int64(3424)}

Entry price range: 2¢ - 98¢
Average entry price: 32.6¢

============================================================
OPTUNA TRAINING (80 trials)
============================================================
21:33:21 [INFO] models.edge.classifier: Training EdgeClassifier with 80 Optuna trials
21:33:21 [INFO] models.edge.classifier: Using day-grouped time splits (DayGroupedTimeSeriesSplit)
21:33:21 [INFO] models.edge.classifier: Using 15 features: ['forecast_temp', 'market_temp', 'edge', 'confidence', 'forecast_uncertainty']...
21:33:21 [INFO] models.edge.classifier: Day splits: total_days=307, train+val_days=261, test_days=46
21:33:21 [INFO] models.edge.classifier: ✓ Leakage checks PASSED
21:33:21 [INFO] models.edge.classifier: Row-wise split: train=5276, val=236, test=1119
21:33:21 [INFO] models.edge.classifier: Class balance - train: 54.8% positive
21:33:21 [INFO] models.edge.classifier: Starting Optuna optimization with 80 trials
21:34:41 [INFO] models.edge.classifier: Best trial score (sharpe): -0.2603
21:34:41 [INFO] models.edge.classifier: Best params: {'bootstrap_type': 'MVS', 'depth': 7, 'iterations': 820, 'learning_rate': 0.020515182345890736, 'l2_leaf_reg': 7.2567200685422035, 'min_data_in_leaf': 100, 'random_strength': 1.6906302744171742, 'colsample_bylevel': 0.8884749871364059, 'subsample': 0.6644326278355579, 'calibration_method': 'isotonic', 'decision_threshold': 0.6700286506157892}
21:34:41 [INFO] models.edge.classifier: Fitting final model on train+val combined...
21:34:45 [INFO] models.edge.classifier: Test AUC: 0.8408
21:34:45 [INFO] models.edge.classifier: Test accuracy: 76.3%
21:34:45 [INFO] models.edge.classifier: Baseline win rate: 52.3%
21:34:45 [INFO] models.edge.classifier: Filtered win rate: 84.5% (n_trades=464)
21:34:45 [INFO] models.edge.classifier: Mean PnL (all edges): 0.0696
21:34:45 [INFO] models.edge.classifier: Mean PnL (trades): 0.2282
21:34:45 [INFO] models.edge.classifier: Sharpe (trades): 0.6714

============================================================
RESULTS
============================================================
Test AUC: 0.8408
Test Accuracy: 76.3%

Baseline win rate: 52.3%
Filtered win rate: 84.5%
Improvement: +32.2pp

Trades recommended: 464/1119 (41.5%)

Feature Importance:
  edge: 20.1178
  base_temp: 13.9871
  market_temp: 11.3539
  forecast_temp: 8.5239
  confidence: 6.4043
  forecast_uncertainty: 6.1753
  obs_fcst_max_gap: 5.2660
  predicted_delta: 4.5891
  fcst_remaining_potential: 4.4932
  hours_to_event_close: 4.0803
  minutes_since_market_open: 3.8610
  market_uncertainty: 3.3935
  snapshot_hour: 3.2401
  market_bid_ask_spread: 3.1600
  temp_volatility_30min: 1.3546

Traceback (most recent call last):
  File "/home/halsted/Documents/python/weather_updated/scripts/train_edge_classifier.py", line 1547, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/halsted/Documents/python/weather_updated/scripts/train_edge_classifier.py", line 1540, in main
    classifier.save(save_path, city=args.city)
  File "/home/halsted/Documents/python/weather_updated/models/edge/classifier.py", line 714, in save
    joblib.dump(
  File "/home/halsted/Documents/python/weather_updated/.venv/lib/python3.11/site-packages/joblib/numpy_pickle.py", line 599, in dump
    with open(filename, "wb") as f:
         ^^^^^^^^^^^^^^^^^^^^
PermissionError: [Errno 13] Permission denied: 'models/saved/denver/edge_classifier.pkl'
2025-12-06 21:34:45 [ERROR] FAILED: Train edge classifier for denver (exit code 1) after 85.2s
2025-12-06 21:34:45 [INFO] 
############################################################
2025-12-06 21:34:45 [INFO] # EDGE TRAINING: LOS_ANGELES
2025-12-06 21:34:45 [INFO] # Trials: 80, Threshold: 1.5°F, Sample rate: 4
2025-12-06 21:34:45 [INFO] ############################################################
2025-12-06 21:34:45 [INFO] 
============================================================
2025-12-06 21:34:45 [INFO] RUNNING: Train edge classifier for los_angeles
2025-12-06 21:34:45 [INFO] Command: /home/halsted/Documents/python/weather_updated/.venv/bin/python scripts/train_edge_classifier.py --city los_angeles --trials 80 --workers 12 --threshold 1.5 --sample-rate 4 --optuna-metric sharpe
2025-12-06 21:34:45 [INFO] ============================================================
21:34:46 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_los_angeles.parquet
============================================================
ML EDGE CLASSIFIER TRAINING
============================================================
City: los_angeles
Optuna trials: 80
Optuna metric: sharpe
Workers: 12
Edge threshold: 1.5°F
Sample rate: every 4th snapshot
P&L mode: REALISTIC (with fees)
Maker fill probability: 40.0%
Ordinal model: models/saved/los_angeles/ordinal_catboost_optuna.pkl (default)
Candle source: parquet (models/candles/candles_los_angeles.parquet)
Settlement source: database

21:34:46 [INFO] __main__: Loading cached edge data from models/saved/los_angeles/edge_training_data_realistic.parquet
21:34:46 [INFO] __main__: Training on 5,512 edge signals

Class balance: 1258/4992 wins (25.2%)

--- REALISTIC P&L STATISTICS ---
Total samples with valid trades: 4,992
Average P&L per trade: $-0.0191
Std P&L per trade: $0.4028
Total gross P&L: $-19.57
Total fees paid: $75.74
Total net P&L: $-95.31

Trade roles: {'taker': np.int64(4762), 'maker': np.int64(230)}
Trade sides: {'yes': np.int64(4992)}
Trade actions: {'buy': np.int64(3869), 'sell': np.int64(1123)}

Entry price range: 2¢ - 98¢
Average entry price: 29.9¢

============================================================
OPTUNA TRAINING (80 trials)
============================================================
21:34:46 [INFO] models.edge.classifier: Training EdgeClassifier with 80 Optuna trials
21:34:46 [INFO] models.edge.classifier: Using day-grouped time splits (DayGroupedTimeSeriesSplit)
21:34:46 [INFO] models.edge.classifier: Using 15 features: ['forecast_temp', 'market_temp', 'edge', 'confidence', 'forecast_uncertainty']...
21:34:46 [INFO] models.edge.classifier: Day splits: total_days=245, train+val_days=208, test_days=37
21:34:46 [INFO] models.edge.classifier: ✓ Leakage checks PASSED
21:34:46 [INFO] models.edge.classifier: Row-wise split: train=4086, val=176, test=442
21:34:46 [INFO] models.edge.classifier: Class balance - train: 25.7% positive
21:34:46 [INFO] models.edge.classifier: Starting Optuna optimization with 80 trials
21:35:19 [INFO] models.edge.classifier: Best trial score (sharpe): -1000000.0000
21:35:19 [INFO] models.edge.classifier: Best params: {'bootstrap_type': 'MVS', 'depth': 4, 'iterations': 572, 'learning_rate': 0.07161329380132488, 'l2_leaf_reg': 11.581010943779246, 'min_data_in_leaf': 99, 'random_strength': 0.04914072542132186, 'colsample_bylevel': 0.4296432678984522, 'subsample': 0.9229021670427989, 'calibration_method': 'none', 'decision_threshold': 0.6700800851998743}
21:35:19 [INFO] models.edge.classifier: Fitting final model on train+val combined...
21:35:19 [INFO] models.edge.classifier: Test AUC: 0.5775
21:35:19 [INFO] models.edge.classifier: Test accuracy: 68.8%
21:35:19 [INFO] models.edge.classifier: Baseline win rate: 23.3%
21:35:19 [INFO] models.edge.classifier: Filtered win rate: 2.7% (n_trades=37)
21:35:19 [INFO] models.edge.classifier: Mean PnL (all edges): 0.0166
21:35:19 [INFO] models.edge.classifier: Mean PnL (trades): -0.2200
21:35:19 [INFO] models.edge.classifier: Sharpe (trades): -1.3449

============================================================
RESULTS
============================================================
Test AUC: 0.5775
Test Accuracy: 68.8%

Baseline win rate: 23.3%
Filtered win rate: 2.7%
Improvement: +-20.6pp

Trades recommended: 37/442 (8.4%)

Feature Importance:
  edge: 21.4346
  market_temp: 12.1281
  forecast_temp: 10.4969
  base_temp: 10.3798
  forecast_uncertainty: 8.6425
  obs_fcst_max_gap: 8.3273
  fcst_remaining_potential: 5.8518
  confidence: 5.8209
  market_uncertainty: 5.4549
  predicted_delta: 4.0307
  minutes_since_market_open: 2.7773
  hours_to_event_close: 2.4462
  snapshot_hour: 1.2047
  temp_volatility_30min: 1.0042
  market_bid_ask_spread: 0.0000

21:35:19 [INFO] models.edge.classifier: Saved model to models/saved/los_angeles/edge_classifier.pkl
21:35:19 [INFO] models.edge.classifier: Saved metadata to models/saved/los_angeles/edge_classifier.json
Model saved to: models/saved/los_angeles/edge_classifier
2025-12-06 21:35:20 [INFO] SUCCESS: Train edge classifier for los_angeles completed in 34.3s (0.6 min)
2025-12-06 21:35:20 [INFO] 
LOS_ANGELES EDGE COMPLETED in 34.3s (0.6 min)
2025-12-06 21:35:20 [INFO] 
############################################################
2025-12-06 21:35:20 [INFO] # EDGE TRAINING: MIAMI
2025-12-06 21:35:20 [INFO] # Trials: 80, Threshold: 1.5°F, Sample rate: 4
2025-12-06 21:35:20 [INFO] ############################################################
2025-12-06 21:35:20 [INFO] 
============================================================
2025-12-06 21:35:20 [INFO] RUNNING: Train edge classifier for miami
2025-12-06 21:35:20 [INFO] Command: /home/halsted/Documents/python/weather_updated/.venv/bin/python scripts/train_edge_classifier.py --city miami --trials 80 --workers 12 --threshold 1.5 --sample-rate 4 --optuna-metric sharpe
2025-12-06 21:35:20 [INFO] ============================================================
21:35:21 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_miami.parquet
============================================================
ML EDGE CLASSIFIER TRAINING
============================================================
City: miami
Optuna trials: 80
Optuna metric: sharpe
Workers: 12
Edge threshold: 1.5°F
Sample rate: every 4th snapshot
P&L mode: REALISTIC (with fees)
Maker fill probability: 40.0%
Ordinal model: models/saved/miami/ordinal_catboost_optuna.pkl (default)
Candle source: parquet (models/candles/candles_miami.parquet)
Settlement source: database

21:35:21 [INFO] __main__: Loading cached edge data from models/saved/miami/edge_training_data_realistic.parquet
21:35:21 [INFO] __main__: Training on 28,305 edge signals

Class balance: 8016/24428 wins (32.8%)

--- REALISTIC P&L STATISTICS ---
Total samples with valid trades: 24,428
Average P&L per trade: $0.0394
Std P&L per trade: $0.4696
Total gross P&L: $1292.33
Total fees paid: $330.06
Total net P&L: $962.27

Trade roles: {'taker': np.int64(19601), 'maker': np.int64(4827)}
Trade sides: {'yes': np.int64(24428)}
Trade actions: {'buy': np.int64(18017), 'sell': np.int64(6411)}

Entry price range: 2¢ - 98¢
Average entry price: 35.8¢

============================================================
OPTUNA TRAINING (80 trials)
============================================================
21:35:21 [INFO] models.edge.classifier: Training EdgeClassifier with 80 Optuna trials
21:35:21 [INFO] models.edge.classifier: Using day-grouped time splits (DayGroupedTimeSeriesSplit)
21:35:21 [INFO] models.edge.classifier: Using 15 features: ['forecast_temp', 'market_temp', 'edge', 'confidence', 'forecast_uncertainty']...
21:35:21 [INFO] models.edge.classifier: Day splits: total_days=777, train+val_days=660, test_days=117
21:35:21 [INFO] models.edge.classifier: ✓ Leakage checks PASSED
21:35:21 [INFO] models.edge.classifier: Row-wise split: train=17737, val=831, test=2331
21:35:21 [INFO] models.edge.classifier: Class balance - train: 36.4% positive
21:35:21 [INFO] models.edge.classifier: Starting Optuna optimization with 80 trials
21:36:39 [INFO] models.edge.classifier: Best trial score (sharpe): 0.5816
21:36:39 [INFO] models.edge.classifier: Best params: {'bootstrap_type': 'MVS', 'depth': 5, 'iterations': 213, 'learning_rate': 0.046458417198972896, 'l2_leaf_reg': 1.7823398571754112, 'min_data_in_leaf': 32, 'random_strength': 1.2740028515020243, 'colsample_bylevel': 0.7674860667110626, 'subsample': 0.8279892501771295, 'calibration_method': 'sigmoid', 'decision_threshold': 0.5729778762801732}
21:36:39 [INFO] models.edge.classifier: Fitting final model on train+val combined...
21:36:40 [INFO] models.edge.classifier: Test AUC: 0.6223
21:36:40 [INFO] models.edge.classifier: Test accuracy: 83.2%
21:36:40 [INFO] models.edge.classifier: Baseline win rate: 17.1%
21:36:40 [INFO] models.edge.classifier: Filtered win rate: 53.0% (n_trades=100)
21:36:40 [INFO] models.edge.classifier: Mean PnL (all edges): -0.1397
21:36:40 [INFO] models.edge.classifier: Mean PnL (trades): 0.0356
21:36:40 [INFO] models.edge.classifier: Sharpe (trades): 0.0812

============================================================
RESULTS
============================================================
Test AUC: 0.6223
Test Accuracy: 83.2%

Baseline win rate: 17.1%
Filtered win rate: 53.0%
Improvement: +35.9pp

Trades recommended: 100/2331 (4.3%)

Feature Importance:
  obs_fcst_max_gap: 18.7294
  market_uncertainty: 15.3856
  edge: 10.3704
  market_temp: 9.5035
  forecast_temp: 7.4947
  forecast_uncertainty: 7.2983
  confidence: 6.7559
  base_temp: 6.1747
  predicted_delta: 5.7150
  snapshot_hour: 3.3281
  fcst_remaining_potential: 3.2307
  minutes_since_market_open: 2.8608
  hours_to_event_close: 2.3837
  temp_volatility_30min: 0.7692
  market_bid_ask_spread: 0.0000

21:36:40 [INFO] models.edge.classifier: Saved model to models/saved/miami/edge_classifier.pkl
21:36:40 [INFO] models.edge.classifier: Saved metadata to models/saved/miami/edge_classifier.json
Model saved to: models/saved/miami/edge_classifier
2025-12-06 21:36:41 [INFO] SUCCESS: Train edge classifier for miami completed in 80.9s (1.3 min)
2025-12-06 21:36:41 [INFO] 
MIAMI EDGE COMPLETED in 80.9s (1.3 min)
2025-12-06 21:36:41 [INFO] 
############################################################
2025-12-06 21:36:41 [INFO] # EDGE TRAINING: PHILADELPHIA
2025-12-06 21:36:41 [INFO] # Trials: 80, Threshold: 1.5°F, Sample rate: 4
2025-12-06 21:36:41 [INFO] ############################################################
2025-12-06 21:36:41 [INFO] 
============================================================
2025-12-06 21:36:41 [INFO] RUNNING: Train edge classifier for philadelphia
2025-12-06 21:36:41 [INFO] Command: /home/halsted/Documents/python/weather_updated/.venv/bin/python scripts/train_edge_classifier.py --city philadelphia --trials 80 --workers 12 --threshold 1.5 --sample-rate 4 --optuna-metric sharpe
2025-12-06 21:36:41 [INFO] ============================================================
21:36:41 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_philadelphia.parquet
============================================================
ML EDGE CLASSIFIER TRAINING
============================================================
City: philadelphia
Optuna trials: 80
Optuna metric: sharpe
Workers: 12
Edge threshold: 1.5°F
Sample rate: every 4th snapshot
P&L mode: REALISTIC (with fees)
Maker fill probability: 40.0%
Ordinal model: models/saved/philadelphia/ordinal_catboost_optuna.pkl (default)
Candle source: parquet (models/candles/candles_philadelphia.parquet)
Settlement source: database

21:36:41 [INFO] __main__: Using ordinal model: models/saved/philadelphia/ordinal_catboost_optuna.pkl
21:36:41 [INFO] __main__: Loaded train data: 389,772 rows
21:36:42 [INFO] __main__: Loaded test data: 97,128 rows
21:36:45 [INFO] __main__: Combined data: 486,900 rows
21:36:47 [INFO] __main__: Generating edge data for philadelphia with 12 workers...
21:36:47 [INFO] __main__: Processing 1068 unique days
21:36:47 [INFO] __main__: Loading model from models/saved/philadelphia/ordinal_catboost_optuna.pkl...
21:36:47 [INFO] models.training.ordinal_trainer: Loaded ordinal model from models/saved/philadelphia/ordinal_catboost_optuna.pkl
21:36:47 [INFO] __main__: Batch loading settlements...
21:36:47 [INFO] src.db.connection: Database engine created: localhost:5434/kalshi_weather
21:36:47 [INFO] __main__: Loaded 1068 settlements
21:36:47 [INFO] __main__: Days with settlement data: 1068
21:36:47 [INFO] __main__: Loading candles from parquet: models/candles/candles_philadelphia.parquet
21:36:47 [INFO] __main__: Loading candles from parquet: models/candles/candles_philadelphia.parquet
21:36:48 [INFO] __main__: Loaded 4,677,580 candle rows from parquet
21:36:49 [INFO] __main__: Filtered to 4,647,479 rows for requested dates
21:40:51 [INFO] __main__: Organized into 2,268 (day, bracket) entries from parquet
21:40:51 [INFO] __main__: Candle cache built: 2268 (day, bracket) entries
21:40:51 [INFO] __main__: Sample cache keys: [(datetime.date(2024, 12, 1), '35.5-36.5'), (datetime.date(2024, 12, 1), '37.5-38.5'), (datetime.date(2024, 12, 1), '39.5-40.5'), (datetime.date(2024, 12, 1), '41.5-42.5'), (datetime.date(2024, 12, 1), '35-36')]
21:41:04 [INFO] __main__: Processing 1068 days with 12 threads...
Processing days:  64%|██████████████████████████████████████████████████████████████████████████████████▌                                              | 684/1068 [48:08<07:19,  1.14s/it]
============================================================
2025-12-06 21:33:20 [INFO] SUCCESS: Train ordinal model for philadelphia completed in 7400.1s (123.3 min)
2025-12-06 21:33:20 [INFO] 
PHILADELPHIA ORDINAL COMPLETED in 8844.5s (147.4 min)
2025-12-06 21:33:20 [INFO] 
============================================================
2025-12-06 21:33:20 [INFO] PHASE 2: EDGE CLASSIFIER TRAINING
2025-12-06 21:33:20 [INFO] ============================================================
2025-12-06 21:33:20 [INFO] 
############################################################
2025-12-06 21:33:20 [INFO] # EDGE TRAINING: DENVER
2025-12-06 21:33:20 [INFO] # Trials: 80, Threshold: 1.5°F, Sample rate: 4
2025-12-06 21:33:20 [INFO] ############################################################
2025-12-06 21:33:20 [INFO] 
============================================================
2025-12-06 21:33:20 [INFO] RUNNING: Train edge classifier for denver
2025-12-06 21:33:20 [INFO] Command: /home/halsted/Documents/python/weather_updated/.venv/bin/python scripts/train_edge_classifier.py --city denver --trials 80 --workers 12 --threshold 1.5 --sample-rate 4 --optuna-metric sharpe
2025-12-06 21:33:20 [INFO] ============================================================
21:33:21 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_denver.parquet
============================================================
ML EDGE CLASSIFIER TRAINING
============================================================
City: denver
Optuna trials: 80
Optuna metric: sharpe
Workers: 12
Edge threshold: 1.5°F
Sample rate: every 4th snapshot
P&L mode: REALISTIC (with fees)
Maker fill probability: 40.0%
Ordinal model: models/saved/denver/ordinal_catboost_optuna.pkl (default)
Candle source: parquet (models/candles/candles_denver.parquet)
Settlement source: database

21:33:21 [INFO] __main__: Loading cached edge data from models/saved/denver/edge_training_data_realistic.parquet
21:33:21 [INFO] __main__: Training on 8,503 edge signals

Class balance: 3816/7544 wins (50.6%)

--- REALISTIC P&L STATISTICS ---
Total samples with valid trades: 7,544
Average P&L per trade: $0.0782
Std P&L per trade: $0.4165
Total gross P&L: $716.59
Total fees paid: $126.66
Total net P&L: $589.93

Trade roles: {'taker': np.int64(7201), 'maker': np.int64(343)}
Trade sides: {'yes': np.int64(7544)}
Trade actions: {'sell': np.int64(4120), 'buy': np.int64(3424)}

Entry price range: 2¢ - 98¢
Average entry price: 32.6¢

============================================================
OPTUNA TRAINING (80 trials)
============================================================
21:33:21 [INFO] models.edge.classifier: Training EdgeClassifier with 80 Optuna trials
21:33:21 [INFO] models.edge.classifier: Using day-grouped time splits (DayGroupedTimeSeriesSplit)
21:33:21 [INFO] models.edge.classifier: Using 15 features: ['forecast_temp', 'market_temp', 'edge', 'confidence', 'forecast_uncertainty']...
21:33:21 [INFO] models.edge.classifier: Day splits: total_days=307, train+val_days=261, test_days=46
21:33:21 [INFO] models.edge.classifier: ✓ Leakage checks PASSED
21:33:21 [INFO] models.edge.classifier: Row-wise split: train=5276, val=236, test=1119
21:33:21 [INFO] models.edge.classifier: Class balance - train: 54.8% positive
21:33:21 [INFO] models.edge.classifier: Starting Optuna optimization with 80 trials
21:34:41 [INFO] models.edge.classifier: Best trial score (sharpe): -0.2603
21:34:41 [INFO] models.edge.classifier: Best params: {'bootstrap_type': 'MVS', 'depth': 7, 'iterations': 820, 'learning_rate': 0.020515182345890736, 'l2_leaf_reg': 7.2567200685422035, 'min_data_in_leaf': 100, 'random_strength': 1.6906302744171742, 'colsample_bylevel': 0.8884749871364059, 'subsample': 0.6644326278355579, 'calibration_method': 'isotonic', 'decision_threshold': 0.6700286506157892}
21:34:41 [INFO] models.edge.classifier: Fitting final model on train+val combined...
21:34:45 [INFO] models.edge.classifier: Test AUC: 0.8408
21:34:45 [INFO] models.edge.classifier: Test accuracy: 76.3%
Processing days:  65%|████████████████████████████████████████████████████████████████████████████████████                                             | 696/1068 [49:04<08:29,  