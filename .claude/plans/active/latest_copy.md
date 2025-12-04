cd /mnt/slow_weather_updated
source .venv/bin/activate

# Run with ALL fixes (177 features, -12 to +12 delta range)
PYTHONPATH=. nohup python3 scripts/train_city_ordinal_optuna.py \
  --city austin \
  --trials 100 \
  --cv-splits 4 \
  --workers 18 \
  --use-cached \
  > logs/austin_FINAL_177features.log 2>&1 &

echo $! > /tmp/training_pid.txt
echo "Training with FULL delta range! PID: $(cat /tmp/training_pid.txt)"

# Verify it's using correct range
tail -f logs/austin_FINAL_177features.log | grep "delta range"


# On your faster computer
cd /mnt/slow_weather_updated
source .venv/bin/activate

# Kill old training
pkill -f "train_city_ordinal_optuna.py.*austin"

# Run with ALL features (177 total)
PYTHONPATH=. nohup python3 scripts/train_city_ordinal_optuna.py \
  --city austin \
  --trials 100 \
  --cv-splits 4 \
  --workers 18 \
  --use-cached \
  > logs/austin_100trials_all_features.log 2>&1 &

echo $! > /tmp/training_pid.txt
echo "Training with 177 features! PID: $(cat /tmp/training_pid.txt)"

# Monitor
tail -f logs/austin_100trials_all_features.log | grep "Training.*features"








============================================================
OPTUNA TRAINING (150 trials)
============================================================
Best trial: 31. Best value: 0.906501:  63%|██████▎   | 94/150 [07:19<04:35,  4.92s/it](.venv) (base) halsted@halsted:/mnt/slow_weather_updated$ tail -30 logs/austin_150trials.log
14:07:20 [INFO] __main__: 
Training samples: 129,928
14:07:20 [INFO] __main__: Training days: 855
14:07:20 [INFO] __main__: Test samples: 32,376
14:07:20 [INFO] __main__: Test days: 213
14:07:20 [INFO] __main__: 
Station-city features: ['station_city_temp_gap', 'station_city_max_gap_sofar', 'station_city_mean_gap_sofar', 'station_city_gap_std', 'station_city_gap_trend']
14:07:20 [INFO] __main__:   station_city_temp_gap: 129,896/129,928 non-null (100.0%)
14:07:20 [INFO] __main__:   station_city_max_gap_sofar: 129,896/129,928 non-null (100.0%)
14:07:20 [INFO] __main__:   station_city_mean_gap_sofar: 129,896/129,928 non-null (100.0%)
14:07:20 [INFO] __main__:   station_city_gap_std: 129,896/129,928 non-null (100.0%)
14:07:20 [INFO] __main__:   station_city_gap_trend: 129,894/129,928 non-null (100.0%)
14:07:20 [INFO] __main__: 
Multi-horizon features: ['fcst_multi_mean', 'fcst_multi_median', 'fcst_multi_ema', 'fcst_multi_std', 'fcst_multi_range', 'fcst_multi_t1_t2_diff', 'fcst_multi_drift']
14:07:20 [INFO] __main__:   fcst_multi_mean: 129,928/129,928 non-null (100.0%)
14:07:20 [INFO] __main__:   fcst_multi_median: 129,928/129,928 non-null (100.0%)
14:07:20 [INFO] __main__:   fcst_multi_ema: 129,928/129,928 non-null (100.0%)
14:07:20 [INFO] __main__:   fcst_multi_std: 129,928/129,928 non-null (100.0%)
14:07:20 [INFO] __main__:   fcst_multi_range: 129,928/129,928 non-null (100.0%)
14:07:20 [INFO] __main__:   fcst_multi_t1_t2_diff: 129,928/129,928 non-null (100.0%)
14:07:20 [INFO] __main__:   fcst_multi_drift: 129,928/129,928 non-null (100.0%)
14:07:22 [INFO] models.training.ordinal_trainer: Training ordinal model (catboost) on 129928 samples
14:07:22 [INFO] models.training.ordinal_trainer: City delta range: [-10, 10]
14:07:22 [INFO] models.training.ordinal_trainer: Training 20 threshold classifiers: [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
14:07:22 [INFO] models.training.ordinal_trainer: Starting Optuna tuning with 150 trials

============================================================
OPTUNA TRAINING (150 trials)
============================================================
Best trial: 31. Best value: 0.906501:  88%|████████▊ | 132/150 [09:58<01:08,  3.83s/it](.venv) (base) halsted@halsted:/mnt/slow_weather_updated$ tail -30 logs/austin_150trials.log
[1]+  Exit 1                  PYTHONPATH=. nohup python3 scripts/train_city_ordinal_optuna.py --city austin --trials 150 --cv-splits 5 --workers 18 --use-cached > logs/austin_150trials.log 2>&1
14:18:31 [INFO] __main__: Saved best params to models/saved/austin/best_params.json
14:18:31 [INFO] __main__: Saved final metrics to models/saved/austin/final_metrics_austin.json

============================================================
EVALUATION
============================================================

Test Set Metrics:
----------------------------------------
  delta_accuracy: 0.2300
  delta_mae: 2.5368
  off_by_1_rate: 0.2481
  off_by_2plus_rate: 0.5218
  within_1_rate: 0.4782
  within_2_rate: 0.6825

============================================================
FEATURE IMPORTANCE
============================================================
Traceback (most recent call last):
  File "/mnt/slow_weather_updated/scripts/train_city_ordinal_optuna.py", line 436, in <module>
    sys.exit(main())
             ^^^^^^
  File "/mnt/slow_weather_updated/scripts/train_city_ordinal_optuna.py", line 384, in main
    fi = trainer.get_feature_importance()
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/slow_weather_updated/models/training/ordinal_trainer.py", line 542, in get_feature_importance
    imp = clf.get_feature_importance()
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'dict' object has no attribute 'get_feature_importance'
(.venv) (base) halsted@halsted:/mnt/slow_weather_updated$ 


✅ Code fixed! Delta range now -10 to +10 for all cities. Copy these commands to your faster computer:
# ============================================================
# AUSTIN TRAINING - 100 TRIALS (18 WORKERS)
# ============================================================

# 1. Kill any existing training
pkill -f "train_city_ordinal_optuna.py.*austin"

# 2. Navigate to project
cd /mnt/slow_weather_updated

# 3. Activate venv
source .venv/bin/activate

# 4. Create logs directory
mkdir -p logs

# 5. Start training in background
PYTHONPATH=. nohup python3 scripts/train_city_ordinal_optuna.py \
  --city austin \
  --trials 150 \
  --cv-splits 5 \
  --workers 18 \
  --use-cached \
  > logs/austin_150trials.log 2>&1 &

# 6. Save PID
echo $! > /tmp/training_pid.txt
echo "Training started! PID: $(cat /tmp/training_pid.txt)"

# ============================================================
# MONITORING COMMANDS
# ============================================================

# Watch trial progress (live)
tail -f logs/austin_150trials.log | grep --line-buffered -E "Trial|Best value|Accuracy|MAE"

# Check current trial number
grep -oP "Best trial: \K\d+" logs/austin_150trials.log | tail -1

# Check if process is running
ps -p $(cat /tmp/training_pid.txt) -o pid,etime,%cpu,%mem,cmd

# Quick status (last 30 lines)
tail -30 logs/austin_150trials.log

# ============================================================
# TO KILL TRAINING
# ============================================================
kill $(cat /tmp/training_pid.txt)
# OR
pkill -f "train_city_ordinal_optuna.py.*austin"
Key changes made:
✅ Delta range: -10 to +10 (21 symmetric classes)
✅ Workers: 18 (for faster computer)
✅ Trials: 100
✅ CV splits: 4
Expected time: ~1-2 hours on faster computer with 18 workers









PYTHONPATH=. python scripts/train_edge_classifier.py \
  --city austin \
  --trials 20 \
  --sample-rate 10 \
  --optuna-metric sharpe \
  --cv-splits 4 \
  --workers 12

(.venv) (base) halsted@halsted:/mnt/slow_weather_updated$ PYTHONPATH=. python scripts/train_edge_classifier.py \
  --city austin \
  --trials 20 \
  --sample-rate 10 \
  --optuna-metric sharpe \
  --cv-splits 4 \
  --workers 12
============================================================
ML EDGE CLASSIFIER TRAINING
============================================================
City: austin
Optuna trials: 20
Workers: 12
Edge threshold: 1.5°F
Sample rate: every 10th snapshot

19:57:35 [INFO] __main__: Using ordinal model: models/saved/austin/ordinal_catboost_optuna.pkl
19:57:35 [INFO] __main__: Loaded train data: 411,672 rows
19:57:36 [INFO] __main__: Loaded test data: 72,504 rows
19:57:36 [INFO] __main__: Combined data: 484,176 rows
19:57:36 [INFO] __main__: Generating edge data for austin with 12 workers...
19:57:36 [INFO] __main__: Processing 1062 unique days
19:57:36 [INFO] __main__: Loading model from models/saved/austin/ordinal_catboost_optuna.pkl...
19:57:36 [INFO] models.training.ordinal_trainer: Loaded ordinal model from models/saved/austin/ordinal_catboost_optuna.pkl
19:57:36 [INFO] __main__: Batch loading settlements...
19:57:36 [INFO] src.db.connection: Database engine created: localhost:5434/kalshi_weather
19:57:37 [INFO] __main__: Loaded 1062 settlements
19:57:37 [INFO] __main__: Days with settlement data: 1062
19:57:37 [INFO] __main__: Batch loading ALL candles (this may take a moment)...
19:57:37 [INFO] __main__: Loading candles for 1062 days (checking both old and new ticker formats)...
19:57:46 [ERROR] __main__: No candles loaded from database!
19:57:46 [INFO] __main__: Candle cache built: 0 (day, bracket) entries
19:57:46 [ERROR] __main__: Candle cache is EMPTY! No candles loaded.
19:57:46 [ERROR] __main__: No edge data generated
(.venv) (base) halsted@halsted:/mnt/slow_weather_updated$ 