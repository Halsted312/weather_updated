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
  --trials 100 \
  --cv-splits 4 \
  --workers 18 \
  --use-cached \
  > logs/austin_100trials.log 2>&1 &

# 6. Save PID
echo $! > /tmp/training_pid.txt
echo "Training started! PID: $(cat /tmp/training_pid.txt)"

# ============================================================
# MONITORING COMMANDS
# ============================================================

# Watch trial progress (live)
tail -f logs/austin_100trials.log | grep --line-buffered -E "Trial|Best value|Accuracy|MAE"

# Check current trial number
grep -oP "Best trial: \K\d+" logs/austin_100trials.log | tail -1

# Check if process is running
ps -p $(cat /tmp/training_pid.txt) -o pid,etime,%cpu,%mem,cmd

# Quick status (last 30 lines)
tail -30 logs/austin_100trials.log

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