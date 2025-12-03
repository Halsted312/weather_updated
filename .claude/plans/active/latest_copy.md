PYTHONPATH=. python scripts/train_edge_classifier.py \
  --city austin \
  --trials 20 \
  --sample-rate 10 \
  --optuna-metric sharpe \
  --cv-splits 4 \
  --workers 16

  (.venv) (base) halsted@halsted:/mnt/slow_weather_updated$ PYTHONPATH=. python scripts/train_edge_classifier.py \
  --city austin \
  --trials 20 \
  --sample-rate 10 \
  --optuna-metric sharpe \
  --cv-splits 4 \
  --workers 16
============================================================
ML EDGE CLASSIFIER TRAINING
============================================================
City: austin
Optuna trials: 20
Workers: 16
Edge threshold: 1.5°F
Sample rate: every 10th snapshot

18:08:10 [INFO] __main__: Using ordinal model: models/saved/austin/ordinal_catboost_optuna.pkl
18:08:11 [INFO] __main__: Loaded train data: 411,672 rows
18:08:11 [INFO] __main__: Loaded test data: 72,504 rows
18:08:11 [INFO] __main__: Combined data: 484,176 rows
18:08:11 [INFO] __main__: Generating edge data for austin with 16 workers...
18:08:11 [INFO] __main__: Processing 1062 unique days
18:08:11 [INFO] __main__: Loading model from models/saved/austin/ordinal_catboost_optuna.pkl...
18:08:11 [INFO] models.training.ordinal_trainer: Loaded ordinal model from models/saved/austin/ordinal_catboost_optuna.pkl
18:08:11 [INFO] __main__: Batch loading settlements...
18:08:12 [INFO] src.db.connection: Database engine created: localhost:5434/kalshi_weather
18:08:12 [INFO] __main__: Loaded 1062 settlements
18:08:12 [INFO] __main__: Days with settlement data: 1062
18:08:12 [INFO] __main__: Batch loading ALL candles (this may take a moment)...
18:08:12 [INFO] __main__: Loading candles for 1062 days...
18:08:25 [INFO] __main__: Processing 1062 days with 16 threads...
Processing days: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1062/1062 [10:50<00:00,  1.63it/s]
18:19:16 [WARNING] __main__: No edge data generated
18:19:16 [ERROR] __main__: No edge data generated
(.venv) (base) halsted@halsted:/mnt/slow_weather_updated$ 