# Chicago validation (10 trials, ~10-15 min)
PYTHONPATH=. python3 scripts/train_city_ordinal_optuna.py \
  --city chicago \
  --trials 10 \
  --cv-splits 3 \
  --workers 18 \
  --use-cached