PYTHONPATH=. python scripts/train_edge_classifier.py \
  --city austin \
  --trials 20 \
  --sample-rate 10 \
  --optuna-metric sharpe \
  --cv-splits 4 \
  --workers 20

(.venv) (base) halsted@halsted:/mnt/slow_weather_updated$ PYTHONPATH=. python scripts/train_edge_classifier.py \
  --city austin \
  --trials 20 \
  --sample-rate 10 \
  --optuna-metric sharpe \
  --cv-splits 4 \
  --workers 20
============================================================
ML EDGE CLASSIFIER TRAINING
============================================================
City: austin
Optuna trials: 20
Workers: 20
Edge threshold: 1.5°F
Sample rate: every 10th snapshot

19:37:43 [INFO] __main__: Using ordinal model: models/saved/austin/ordinal_catboost_optuna.pkl
19:37:44 [INFO] __main__: Loaded train data: 411,672 rows
19:37:44 [INFO] __main__: Loaded test data: 72,504 rows
19:37:44 [INFO] __main__: Combined data: 484,176 rows
19:37:44 [INFO] __main__: Generating edge data for austin with 20 workers...
19:37:44 [INFO] __main__: Processing 1062 unique days
19:37:44 [INFO] __main__: Loading model from models/saved/austin/ordinal_catboost_optuna.pkl...
19:37:45 [INFO] models.training.ordinal_trainer: Loaded ordinal model from models/saved/austin/ordinal_catboost_optuna.pkl
19:37:45 [INFO] __main__: Batch loading settlements...
19:37:45 [INFO] src.db.connection: Database engine created: localhost:5434/kalshi_weather
19:37:45 [INFO] __main__: Loaded 1062 settlements
19:37:45 [INFO] __main__: Days with settlement data: 1062
19:37:45 [INFO] __main__: Batch loading ALL candles (this may take a moment)...
19:37:45 [INFO] __main__: Loading candles for 1062 days (checking both old and new ticker formats)...
19:38:02 [INFO] __main__: Processing 1062 days with 20 threads...
Processing days: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1062/1062 [11:45<00:00,  1.51it/s]
19:49:48 [WARNING] __main__: No edge data generated
19:49:48 [ERROR] __main__: No edge data generated
(.venv) (base) halsted@halsted:/mnt/slow_weather_updated$ 