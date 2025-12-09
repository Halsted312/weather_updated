(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:33:20 [INFO] # Trials: 80, Threshold: 1.5°F, Sample rate: 4
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:33:20 [INFO] ############################################################
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:33:20 [INFO] 
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:33:20 [INFO] RUNNING: Train edge classifier for denver
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:33:20 [INFO] Command: /home/halsted/Documents/python/weather_updated/.venv/bin/python scripts/train_edge_classifier.py --city denver --trials 80 --workers 12 --threshold 1.5 --sample-rate 4 --optuna-metric sharpe
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:33:20 [INFO] ============================================================
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:33:21 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_denver.parquet
21:33:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ML EDGE CLASSIFIER TRAINING
ML: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ City: denver
City:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Optuna trials: 80
Command 'Optuna' not found, did you mean:
  command 'optuna' from deb python3-optuna (3.5.0-1)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Optuna metric: sharpe
Command 'Optuna' not found, did you mean:
  command 'optuna' from deb python3-optuna (3.5.0-1)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Workers: 12
Workers:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Edge threshold: 1.5°F
Command 'Edge' not found, did you mean:
  command 'edge' from deb n2n (1.3.1~svn3789-7)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Sample rate: every 4th snapshot
Command 'Sample' not found, did you mean:
  command 'ample' from deb ample (0.5.7-13)
  command 'yample' from deb yample (0.30-5)
  command 'sample' from deb barcode (0.99-7)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ P&L mode: REALISTIC (with fees)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Maker fill probability: 40.0%
Command 'Maker' not found, did you mean:
  command 'faker' from deb faker (0.9.3-2)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Ordinal model: models/saved/denver/ordinal_catboost_optuna.pkl (default)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Candle source: parquet (models/candles/candles_denver.parquet)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Settlement source: database
Settlement: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:33:21 [INFO] __main__: Loading cached edge data from models/saved/denver/edge_training_data_realistic.parquet
21:33:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:33:21 [INFO] __main__: Training on 8,503 edge signals
21:33:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Class balance: 3816/7544 wins (50.6%)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ --- REALISTIC P&L STATISTICS ---
[1] 3524282
L: command not found
---: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Total samples with valid trades: 7,544
[1]+  Exit 127                --- REALISTIC P
Total: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Average P&L per trade: $0.0782
[1] 3524335
L: command not found
Command 'Average' not found, did you mean:
  command 'average' from deb argyll (3.1.0+repack-1)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Std P&L per trade: $0.4165
[2] 3524360
L: command not found
Command 'Std' not found, did you mean:
  command 'std' from snap std (1.0.1)
  command 'atd' from deb at (3.2.5-1ubuntu1)
  command 'rtd' from deb skycat (3.1.2+starlink1~b+dfsg-7build1)
  command 'td' from deb textdraw (0.2+ds-0+nmu1build3)
See 'snap info <snapname>' for additional versions.
[1]-  Exit 127                Average P
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Total gross P&L: $716.59
[3] 3524385
L:: command not found
Total: command not found
[2]-  Exit 127                Std P
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Total fees paid: $126.66
Total: command not found
[3]+  Exit 127                Total gross P
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Total net P&L: $589.93
[1] 3524427
L:: command not found
Total: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Trade roles: {'taker': np.int64(7201), 'maker': np.int64(343)}
bash: syntax error near unexpected token `('
[1]+  Exit 127                Total net P
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Trade sides: {'yes': np.int64(7544)}
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Trade actions: {'sell': np.int64(4120), 'buy': np.int64(3424)}
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Entry price range: 2¢ - 98¢
Entry: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Average entry price: 32.6¢
Command 'Average' not found, did you mean:
  command 'average' from deb argyll (3.1.0+repack-1)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ OPTUNA TRAINING (80 trials)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:33:21 [INFO] models.edge.classifier: Training EdgeClassifier with 80 Optuna trials
21:33:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:33:21 [INFO] models.edge.classifier: Using day-grouped time splits (DayGroupedTimeSeriesSplit)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:33:21 [INFO] models.edge.classifier: Using 15 features: ['forecast_temp', 'market_temp', 'edge', 'confidence', 'forecast_uncertainty']...
21:33:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:33:21 [INFO] models.edge.classifier: Day splits: total_days=307, train+val_days=261, test_days=46
21:33:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:33:21 [INFO] models.edge.classifier: ✓ Leakage checks PASSED
21:33:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:33:21 [INFO] models.edge.classifier: Row-wise split: train=5276, val=236, test=1119
21:33:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:33:21 [INFO] models.edge.classifier: Class balance - train: 54.8% positive
21:33:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:33:21 [INFO] models.edge.classifier: Starting Optuna optimization with 80 trials
21:33:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:41 [INFO] models.edge.classifier: Best trial score (sharpe): -0.2603
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:41 [INFO] models.edge.classifier: Best params: {'bootstrap_type': 'MVS', 'depth': 7, 'iterations': 820, 'learning_rate': 0.020515182345890736, 'l2_leaf_reg': 7.2567200685422035, 'min_data_in_leaf': 100, 'random_strength': 1.6906302744171742, 'colsample_bylevel': 0.8884749871364059, 'subsample': 0.6644326278355579, 'calibration_method': 'isotonic', 'decision_threshold': 0.6700286506157892}
21:34:41: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:41 [INFO] models.edge.classifier: Fitting final model on train+val combined...
21:34:41: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:45 [INFO] models.edge.classifier: Test AUC: 0.8408
21:34:45: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:45 [INFO] models.edge.classifier: Test accuracy: 76.3%
21:34:45: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:45 [INFO] models.edge.classifier: Baseline win rate: 52.3%
21:34:45: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:45 [INFO] models.edge.classifier: Filtered win rate: 84.5% (n_trades=464)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:45 [INFO] models.edge.classifier: Mean PnL (all edges): 0.0696
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:45 [INFO] models.edge.classifier: Mean PnL (trades): 0.2282
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:45 [INFO] models.edge.classifier: Sharpe (trades): 0.6714
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ RESULTS
RESULTS: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Test AUC: 0.8408
Command 'Test' not found, did you mean:
  command 'jest' from deb jest (29.6.2~ds1+~cs73.45.28-5)
  command 'test' from deb coreutils (9.4-3ubuntu6.1)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Test Accuracy: 76.3%
Command 'Test' not found, did you mean:
  command 'test' from deb coreutils (9.4-3ubuntu6.1)
  command 'jest' from deb jest (29.6.2~ds1+~cs73.45.28-5)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Baseline win rate: 52.3%
Baseline: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Filtered win rate: 84.5%
Filtered: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Improvement: +32.2pp
Improvement:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Trades recommended: 464/1119 (41.5%)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Feature Importance:
Feature: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   edge: 20.1178
Command 'edge:' not found, did you mean:
  command 'edge' from deb n2n (1.3.1~svn3789-7)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   base_temp: 13.9871
base_temp:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   market_temp: 11.3539
market_temp:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   forecast_temp: 8.5239
forecast_temp:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   confidence: 6.4043
confidence:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   forecast_uncertainty: 6.1753
forecast_uncertainty:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   obs_fcst_max_gap: 5.2660
obs_fcst_max_gap:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   predicted_delta: 4.5891
predicted_delta:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   fcst_remaining_potential: 4.4932
fcst_remaining_potential:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   hours_to_event_close: 4.0803
hours_to_event_close:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   minutes_since_market_open: 3.8610
minutes_since_market_open:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   market_uncertainty: 3.3935
market_uncertainty:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   snapshot_hour: 3.2401
snapshot_hour:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   market_bid_ask_spread: 3.1600
market_bid_ask_spread:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   temp_volatility_30min: 1.3546
temp_volatility_30min:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Traceback (most recent call last):
bash: syntax error near unexpected token `most'
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   File "/home/halsted/Documents/python/weather_updated/scripts/train_edge_classifier.py", line 1547, in <module>
bash: syntax error near unexpected token `newline'
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$     sys.exit(main())
bash: syntax error near unexpected token `main'
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$              ^^^^^^
^^^^^^: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   File "/home/halsted/Documents/python/weather_updated/scripts/train_edge_classifier.py", line 1540, in main
Command 'File' not found, did you mean:
  command 'file' from deb file (1:5.45-2)
  command 'zile' from deb zile (2.6.2-2)
  command 'kile' from deb kile (4:2.9.93-2)
  command 'vile' from deb vile (9.8y-3)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$     classifier.save(save_path, city=args.city)
bash: syntax error near unexpected token `save_path,'
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   File "/home/halsted/Documents/python/weather_updated/models/edge/classifier.py", line 714, in save
Command 'File' not found, did you mean:
  command 'vile' from deb vile (9.8y-3)
  command 'zile' from deb zile (2.6.2-2)
  command 'kile' from deb kile (4:2.9.93-2)
  command 'file' from deb file (1:5.45-2)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$     joblib.dump(
bash: syntax error near unexpected token `newline'
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   File "/home/halsted/Documents/python/weather_updated/.venv/lib/python3.11/site-packages/joblib/numpy_pickle.py", line 599, in dump
Command 'File' not found, did you mean:
  command 'kile' from deb kile (4:2.9.93-2)
  command 'file' from deb file (1:5.45-2)
  command 'zile' from deb zile (2.6.2-2)
  command 'vile' from deb vile (9.8y-3)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$     with open(filename, "wb") as f:
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$          ^^^^^^^^^^^^^^^^^^^^
^^^^^^^^^^^^^^^^^^^^: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ PermissionError: [Errno 13] Permission denied: 'models/saved/denver/edge_classifier.pkl'
PermissionError:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:34:45 [ERROR] FAILED: Train edge classifier for denver (exit code 1) after 85.2s
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:34:45 [INFO] 
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ############################################################
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:34:45 [INFO] # EDGE TRAINING: LOS_ANGELES
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:34:45 [INFO] # Trials: 80, Threshold: 1.5°F, Sample rate: 4
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:34:45 [INFO] ############################################################
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:34:45 [INFO] 
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:34:45 [INFO] RUNNING: Train edge classifier for los_angeles
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:34:45 [INFO] Command: /home/halsted/Documents/python/weather_updated/.venv/bin/python scripts/train_edge_classifier.py --city los_angeles --trials 80 --workers 12 --threshold 1.5 --sample-rate 4 --optuna-metric sharpe
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:34:45 [INFO] ============================================================
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:46 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_los_angeles.parquet
21:34:46: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ML EDGE CLASSIFIER TRAINING
ML: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ City: los_angeles
City:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Optuna trials: 80
Command 'Optuna' not found, did you mean:
  command 'optuna' from deb python3-optuna (3.5.0-1)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Optuna metric: sharpe
Command 'Optuna' not found, did you mean:
  command 'optuna' from deb python3-optuna (3.5.0-1)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Workers: 12
Workers:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Edge threshold: 1.5°F
Command 'Edge' not found, did you mean:
  command 'edge' from deb n2n (1.3.1~svn3789-7)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Sample rate: every 4th snapshot
Command 'Sample' not found, did you mean:
  command 'sample' from deb barcode (0.99-7)
  command 'yample' from deb yample (0.30-5)
  command 'ample' from deb ample (0.5.7-13)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ P&L mode: REALISTIC (with fees)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Maker fill probability: 40.0%
Command 'Maker' not found, did you mean:
  command 'faker' from deb faker (0.9.3-2)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Ordinal model: models/saved/los_angeles/ordinal_catboost_optuna.pkl (default)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Candle source: parquet (models/candles/candles_los_angeles.parquet)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Settlement source: database
Settlement: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:46 [INFO] __main__: Loading cached edge data from models/saved/los_angeles/edge_training_data_realistic.parquet
21:34:46: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:46 [INFO] __main__: Training on 5,512 edge signals
21:34:46: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Class balance: 1258/4992 wins (25.2%)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ --- REALISTIC P&L STATISTICS ---
[1] 3525778
L: command not found
---: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Total samples with valid trades: 4,992
Total: command not found
[1]+  Exit 127                --- REALISTIC P
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Average P&L per trade: $-0.0191
[1] 3525814
L: command not found
Command 'Average' not found, did you mean:
  command 'average' from deb argyll (3.1.0+repack-1)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Std P&L per trade: $0.4028
[2] 3525839
L: command not found
Command 'Std' not found, did you mean:
  command 'std' from snap std (1.0.1)
  command 'atd' from deb at (3.2.5-1ubuntu1)
  command 'td' from deb textdraw (0.2+ds-0+nmu1build3)
  command 'rtd' from deb skycat (3.1.2+starlink1~b+dfsg-7build1)
See 'snap info <snapname>' for additional versions.
[1]-  Exit 127                Average P
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Total gross P&L: $-19.57
[2]+  Exit 127                Std P
[1] 3525872
L:: command not found
Total: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Total fees paid: $75.74
Total: command not found
[1]+  Exit 127                Total gross P
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Total net P&L: $-95.31
[1] 3525913
L:: command not found
Total: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
n[1]+  Exit 127                Total net P
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Trade roles: {'taker': np.int64(4762), 'maker': np.int64(230)}
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Trade sides: {'yes': np.int64(4992)}
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Trade actions: {'buy': np.int64(3869), 'sell': np.int64(1123)}
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Entry price range: 2¢ - 98¢
Entry: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Average entry price: 29.9¢
Command 'Average' not found, did you mean:
  command 'average' from deb argyll (3.1.0+repack-1)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ OPTUNA TRAINING (80 trials)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:46 [INFO] models.edge.classifier: Training EdgeClassifier with 80 Optuna trials
21:34:46: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:46 [INFO] models.edge.classifier: Using day-grouped time splits (DayGroupedTimeSeriesSplit)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:46 [INFO] models.edge.classifier: Using 15 features: ['forecast_temp', 'market_temp', 'edge', 'confidence', 'forecast_uncertainty']...
21:34:46: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:46 [INFO] models.edge.classifier: Day splits: total_days=245, train+val_days=208, test_days=37
21:34:46: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:46 [INFO] models.edge.classifier: ✓ Leakage checks PASSED
21:34:46: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:46 [INFO] models.edge.classifier: Row-wise split: train=4086, val=176, test=442
21:34:46: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:46 [INFO] models.edge.classifier: Class balance - train: 25.7% positive
21:34:46: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:34:46 [INFO] models.edge.classifier: Starting Optuna optimization with 80 trials
21:34:46: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:19 [INFO] models.edge.classifier: Best trial score (sharpe): -1000000.0000
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:19 [INFO] models.edge.classifier: Best params: {'bootstrap_type': 'MVS', 'depth': 4, 'iterations': 572, 'learning_rate': 0.07161329380132488, 'l2_leaf_reg': 11.581010943779246, 'min_data_in_leaf': 99, 'random_strength': 0.04914072542132186, 'colsample_bylevel': 0.4296432678984522, 'subsample': 0.9229021670427989, 'calibration_method': 'none', 'decision_threshold': 0.6700800851998743}
21:35:19: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:19 [INFO] models.edge.classifier: Fitting final model on train+val combined...
21:35:19: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:19 [INFO] models.edge.classifier: Test AUC: 0.5775
21:35:19: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:19 [INFO] models.edge.classifier: Test accuracy: 68.8%
21:35:19: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:19 [INFO] models.edge.classifier: Baseline win rate: 23.3%
21:35:19: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:19 [INFO] models.edge.classifier: Filtered win rate: 2.7% (n_trades=37)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:19 [INFO] models.edge.classifier: Mean PnL (all edges): 0.0166
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:19 [INFO] models.edge.classifier: Mean PnL (trades): -0.2200
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:19 [INFO] models.edge.classifier: Sharpe (trades): -1.3449
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ RESULTS
RESULTS: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Test AUC: 0.5775
Command 'Test' not found, did you mean:
  command 'test' from deb coreutils (9.4-3ubuntu6.1)
  command 'jest' from deb jest (29.6.2~ds1+~cs73.45.28-5)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Test Accuracy: 68.8%
Command 'Test' not found, did you mean:
  command 'jest' from deb jest (29.6.2~ds1+~cs73.45.28-5)
  command 'test' from deb coreutils (9.4-3ubuntu6.1)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Baseline win rate: 23.3%
Baseline: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Filtered win rate: 2.7%
Filtered: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Improvement: +-20.6pp
Improvement:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Trades recommended: 37/442 (8.4%)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Feature Importance:
Feature: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   edge: 21.4346
Command 'edge:' not found, did you mean:
  command 'edge' from deb n2n (1.3.1~svn3789-7)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   market_temp: 12.1281
market_temp:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   forecast_temp: 10.4969
forecast_temp:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   base_temp: 10.3798
base_temp:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   forecast_uncertainty: 8.6425
forecast_uncertainty:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   obs_fcst_max_gap: 8.3273
obs_fcst_max_gap:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   fcst_remaining_potential: 5.8518
fcst_remaining_potential:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   confidence: 5.8209
confidence:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   market_uncertainty: 5.4549
market_uncertainty:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   predicted_delta: 4.0307
predicted_delta:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   minutes_since_market_open: 2.7773
minutes_since_market_open:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   hours_to_event_close: 2.4462
hours_to_event_close:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   snapshot_hour: 1.2047
snapshot_hour:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   temp_volatility_30min: 1.0042
temp_volatility_30min:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   market_bid_ask_spread: 0.0000
market_bid_ask_spread:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:19 [INFO] models.edge.classifier: Saved model to models/saved/los_angeles/edge_classifier.pkl
21:35:19: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:19 [INFO] models.edge.classifier: Saved metadata to models/saved/los_angeles/edge_classifier.json
21:35:19: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Model saved to: models/saved/los_angeles/edge_classifier
Model: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:35:20 [INFO] SUCCESS: Train edge classifier for los_angeles completed in 34.3s (0.6 min)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:35:20 [INFO] 
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ LOS_ANGELES EDGE COMPLETED in 34.3s (0.6 min)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:35:20 [INFO] 
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ############################################################
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:35:20 [INFO] # EDGE TRAINING: MIAMI
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:35:20 [INFO] # Trials: 80, Threshold: 1.5°F, Sample rate: 4
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:35:20 [INFO] ############################################################
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:35:20 [INFO] 
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:35:20 [INFO] RUNNING: Train edge classifier for miami
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:35:20 [INFO] Command: /home/halsted/Documents/python/weather_updated/.venv/bin/python scripts/train_edge_classifier.py --city miami --trials 80 --workers 12 --threshold 1.5 --sample-rate 4 --optuna-metric sharpe
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:35:20 [INFO] ============================================================
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:21 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_miami.parquet
21:35:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ML EDGE CLASSIFIER TRAINING
ML: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ City: miami
City:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Optuna trials: 80
Command 'Optuna' not found, did you mean:
  command 'optuna' from deb python3-optuna (3.5.0-1)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Optuna metric: sharpe
Command 'Optuna' not found, did you mean:
  command 'optuna' from deb python3-optuna (3.5.0-1)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Workers: 12
Workers:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Edge threshold: 1.5°F
Command 'Edge' not found, did you mean:
  command 'edge' from deb n2n (1.3.1~svn3789-7)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Sample rate: every 4th snapshot
Command 'Sample' not found, did you mean:
  command 'sample' from deb barcode (0.99-7)
  command 'ample' from deb ample (0.5.7-13)
  command 'yample' from deb yample (0.30-5)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ P&L mode: REALISTIC (with fees)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Maker fill probability: 40.0%
Command 'Maker' not found, did you mean:
  command 'faker' from deb faker (0.9.3-2)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Ordinal model: models/saved/miami/ordinal_catboost_optuna.pkl (default)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Candle source: parquet (models/candles/candles_miami.parquet)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Settlement source: database
Settlement: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:21 [INFO] __main__: Loading cached edge data from models/saved/miami/edge_training_data_realistic.parquet
21:35:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:21 [INFO] __main__: Training on 28,305 edge signals
21:35:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Class balance: 8016/24428 wins (32.8%)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ --- REALISTIC P&L STATISTICS ---
[1] 3527117
L: command not found
---: command not found
[1]+  Exit 127                --- REALISTIC P
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Total samples with valid trades: 24,428
Total: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Average P&L per trade: $0.0394
[1] 3527155
L: command not found
Command 'Average' not found, did you mean:
  command 'average' from deb argyll (3.1.0+repack-1)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Std P&L per trade: $0.4696
[2] 3527181
L: command not found
Command 'Std' not found, did you mean:
  command 'std' from snap std (1.0.1)
  command 'atd' from deb at (3.2.5-1ubuntu1)
  command 'rtd' from deb skycat (3.1.2+starlink1~b+dfsg-7build1)
  command 'td' from deb textdraw (0.2+ds-0+nmu1build3)
See 'snap info <snapname>' for additional versions.
[1]-  Exit 127                Average P
[2]+  Exit 127                Std P
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Total gross P&L: $1292.33
[1] 3527209
L:: command not found
Total: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Total fees paid: $330.06
Total: command not found
[1]+  Exit 127                Total gross P
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Total net P&L: $962.27
[1] 3527282
L:: command not found
Total: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
[1]+  Exit 127                Total net P
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Trade roles: {'taker': np.int64(19601), 'maker': np.int64(4827)}
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Trade sides: {'yes': np.int64(24428)}
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Trade actions: {'buy': np.int64(18017), 'sell': np.int64(6411)}
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Entry price range: 2¢ - 98¢
Entry: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Average entry price: 35.8¢
Command 'Average' not found, did you mean:
  command 'average' from deb argyll (3.1.0+repack-1)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ OPTUNA TRAINING (80 trials)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:21 [INFO] models.edge.classifier: Training EdgeClassifier with 80 Optuna trials
21:35:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:21 [INFO] models.edge.classifier: Using day-grouped time splits (DayGroupedTimeSeriesSplit)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:21 [INFO] models.edge.classifier: Using 15 features: ['forecast_temp', 'market_temp', 'edge', 'confidence', 'forecast_uncertainty']...
21:35:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:21 [INFO] models.edge.classifier: Day splits: total_days=777, train+val_days=660, test_days=117
21:35:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:21 [INFO] models.edge.classifier: ✓ Leakage checks PASSED
21:35:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:21 [INFO] models.edge.classifier: Row-wise split: train=17737, val=831, test=2331
21:35:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:21 [INFO] models.edge.classifier: Class balance - train: 36.4% positive
21:35:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:35:21 [INFO] models.edge.classifier: Starting Optuna optimization with 80 trials
21:35:21: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:39 [INFO] models.edge.classifier: Best trial score (sharpe): 0.5816
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:39 [INFO] models.edge.classifier: Best params: {'bootstrap_type': 'MVS', 'depth': 5, 'iterations': 213, 'learning_rate': 0.046458417198972896, 'l2_leaf_reg': 1.7823398571754112, 'min_data_in_leaf': 32, 'random_strength': 1.2740028515020243, 'colsample_bylevel': 0.7674860667110626, 'subsample': 0.8279892501771295, 'calibration_method': 'sigmoid', 'decision_threshold': 0.5729778762801732}
21:36:39: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:39 [INFO] models.edge.classifier: Fitting final model on train+val combined...
21:36:39: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:40 [INFO] models.edge.classifier: Test AUC: 0.6223
21:36:40: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:40 [INFO] models.edge.classifier: Test accuracy: 83.2%
21:36:40: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:40 [INFO] models.edge.classifier: Baseline win rate: 17.1%
21:36:40: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:40 [INFO] models.edge.classifier: Filtered win rate: 53.0% (n_trades=100)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:40 [INFO] models.edge.classifier: Mean PnL (all edges): -0.1397
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:40 [INFO] models.edge.classifier: Mean PnL (trades): 0.0356
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:40 [INFO] models.edge.classifier: Sharpe (trades): 0.0812
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ RESULTS
RESULTS: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Test AUC: 0.6223
Command 'Test' not found, did you mean:
  command 'test' from deb coreutils (9.4-3ubuntu6.1)
  command 'jest' from deb jest (29.6.2~ds1+~cs73.45.28-5)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Test Accuracy: 83.2%
Command 'Test' not found, did you mean:
  command 'test' from deb coreutils (9.4-3ubuntu6.1)
  command 'jest' from deb jest (29.6.2~ds1+~cs73.45.28-5)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Baseline win rate: 17.1%
Baseline: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Filtered win rate: 53.0%
Filtered: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Improvement: +35.9pp
Improvement:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Trades recommended: 100/2331 (4.3%)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Feature Importance:
Feature: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   obs_fcst_max_gap: 18.7294
��████████�obs_fcst_max_gap:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   market_uncertainty: 15.3856
market_uncertainty:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   edge: 10.3704
Command 'edge:' not found, did you mean:
  command 'edge' from deb n2n (1.3.1~svn3789-7)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   market_temp: 9.5035
market_temp:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   forecast_temp: 7.4947
forecast_temp:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   forecast_uncertainty: 7.2983
forecast_uncertainty:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   confidence: 6.7559
confidence:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   base_temp: 6.1747
base_temp:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   predicted_delta: 5.7150
predicted_delta:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   snapshot_hour: 3.3281
snapshot_hour:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   fcst_remaining_potential: 3.2307
fcst_remaining_potential:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   minutes_since_market_open: 2.8608
minutes_since_market_open:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   hours_to_event_close: 2.3837
hours_to_event_close:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   temp_volatility_30min: 0.7692
temp_volatility_30min:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$   market_bid_ask_spread: 0.0000
market_bid_ask_spread:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:40 [INFO] models.edge.classifier: Saved model to models/saved/miami/edge_classifier.pkl
21:36:40: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:40 [INFO] models.edge.classifier: Saved metadata to models/saved/miami/edge_classifier.json
21:36:40: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Model saved to: models/saved/miami/edge_classifier
Model: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:36:41 [INFO] SUCCESS: Train edge classifier for miami completed in 80.9s (1.3 min)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:36:41 [INFO] 
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ MIAMI EDGE COMPLETED in 80.9s (1.3 min)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:36:41 [INFO] 
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ############################################################
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:36:41 [INFO] # EDGE TRAINING: PHILADELPHIA
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:36:41 [INFO] # Trials: 80, Threshold: 1.5°F, Sample rate: 4
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:36:41 [INFO] ############################################################
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:36:41 [INFO] 
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:36:41 [INFO] RUNNING: Train edge classifier for philadelphia
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:36:41 [INFO] Command: /home/halsted/Documents/python/weather_updated/.venv/bin/python scripts/train_edge_classifier.py --city philadelphia --trials 80 --workers 12 --threshold 1.5 --sample-rate 4 --optuna-metric sharpe
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 2025-12-06 21:36:41 [INFO] ============================================================
2025-12-06: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:41 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_philadelphia.parquet
21:36:41: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ML EDGE CLASSIFIER TRAINING
ML: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ ============================================================
============================================================: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ City: philadelphia
City:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Optuna trials: 80
Command 'Optuna' not found, did you mean:
  command 'optuna' from deb python3-optuna (3.5.0-1)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Optuna metric: sharpe
Command 'Optuna' not found, did you mean:
  command 'optuna' from deb python3-optuna (3.5.0-1)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Workers: 12
Workers:: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Edge threshold: 1.5°F
Command 'Edge' not found, did you mean:
  command 'edge' from deb n2n (1.3.1~svn3789-7)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Sample rate: every 4th snapshot
Command 'Sample' not found, did you mean:
  command 'sample' from deb barcode (0.99-7)
  command 'yample' from deb yample (0.30-5)
  command 'ample' from deb ample (0.5.7-13)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ P&L mode: REALISTIC (with fees)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Maker fill probability: 40.0%
Command 'Maker' not found, did you mean:
  command 'faker' from deb faker (0.9.3-2)
Try: sudo apt install <deb name>
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Ordinal model: models/saved/philadelphia/ordinal_catboost_optuna.pkl (default)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Candle source: parquet (models/candles/candles_philadelphia.parquet)
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Settlement source: database
Settlement: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:41 [INFO] __main__: Using ordinal model: models/saved/philadelphia/ordinal_catboost_optuna.pkl
21:36:41: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:41 [INFO] __main__: Loaded train data: 389,772 rows
21:36:41: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:42 [INFO] __main__: Loaded test data: 97,128 rows
21:36:42: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:45 [INFO] __main__: Combined data: 486,900 rows
21:36:45: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:47 [INFO] __main__: Generating edge data for philadelphia with 12 workers...
21:36:47: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:47 [INFO] __main__: Processing 1068 unique days
21:36:47: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:47 [INFO] __main__: Loading model from models/saved/philadelphia/ordinal_catboost_optuna.pkl...
21:36:47: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:47 [INFO] models.training.ordinal_trainer: Loaded ordinal model from models/saved/philadelphia/ordinal_catboost_optuna.pkl
21:36:47: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:47 [INFO] __main__: Batch loading settlements...
21:36:47: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:47 [INFO] src.db.connection: Database engine created: localhost:5434/kalshi_weather
21:36:47: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:47 [INFO] __main__: Loaded 1068 settlements
21:36:47: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:47 [INFO] __main__: Days with settlement data: 1068
21:36:47: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:47 [INFO] __main__: Loading candles from parquet: models/candles/candles_philadelphia.parquet
21:36:47: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:47 [INFO] __main__: Loading candles from parquet: models/candles/candles_philadelphia.parquet
21:36:47: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:48 [INFO] __main__: Loaded 4,677,580 candle rows from parquet
21:36:48: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:36:49 [INFO] __main__: Filtered to 4,647,479 rows for requested dates
21:36:49: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:40:51 [INFO] __main__: Organized into 2,268 (day, bracket) entries from parquet
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:40:51 [INFO] __main__: Candle cache built: 2268 (day, bracket) entries
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:40:51 [INFO] __main__: Sample cache keys: [(datetime.date(2024, 12, 1), '35.5-36.5'), (datetime.date(2024, 12, 1), '37.5-38.5'), (datetime.date(2024, 12, 1), '39.5-40.5'), (datetime.date(2024, 12, 1), '41.5-42.5'), (datetime.date(2024, 12, 1), '35-36')]
bash: syntax error near unexpected token `('
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ 21:41:04 [INFO] __main__: Processing 1068 days with 12 threads...
21:41:04: command not found
(.venv) (base) halsted@halsted:~/Documents/python/weather_updated$ Processing days:  64%|██████████████████████████████████████████████████████████████████████████████████▌    