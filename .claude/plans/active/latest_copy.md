PYTHONPATH=. python scripts/train_edge_classifier.py \
  --city austin \
  --trials 30 \
  --sample-rate 10 \
  --optuna-metric sharpe \
  --cv-splits 4 \
  --workers 12

  