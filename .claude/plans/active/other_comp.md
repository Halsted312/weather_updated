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