#!/bin/bash
# Train Austin and Chicago with AUC objective for comparison
# Same settings as overnight run (200 trials, cv=5) but with AUC

set -e

CITIES=("austin" "chicago")
TRIALS=100
CV_SPLITS=5
OBJECTIVE="auc"
LOG_DIR="logs/auc_training_$(date +%Y%m%d)"

mkdir -p "$LOG_DIR"

echo "=============================================="
echo "AUC TRAINING - Austin & Chicago"
echo "=============================================="
echo "Started: $(date)"
echo "Trials: $TRIALS"
echo "CV Splits: $CV_SPLITS"
echo "Objective: $OBJECTIVE"
echo "=============================================="

START_TIME=$(date +%s)

for city in "${CITIES[@]}"; do
    echo ""
    echo "=============================================="
    echo "TRAINING: $city (AUC objective)"
    echo "Started: $(date)"
    echo "=============================================="

    CITY_START=$(date +%s)

    PYTHONPATH=. python3 scripts/train_city_ordinal_optuna.py \
        --city "$city" \
        --trials $TRIALS \
        --cv-splits $CV_SPLITS \
        --objective "$OBJECTIVE" \
        2>&1 | tee "$LOG_DIR/${city}_auc_training.log"

    # Copy model to _AUC.pkl for comparison
    if [ -f "models/saved/$city/ordinal_catboost_optuna.pkl" ]; then
        cp "models/saved/$city/ordinal_catboost_optuna.pkl" "models/saved/$city/ordinal_catboost_optuna_AUC.pkl"
        echo "Saved: models/saved/$city/ordinal_catboost_optuna_AUC.pkl"
    fi

    CITY_END=$(date +%s)
    CITY_DURATION=$((CITY_END - CITY_START))
    CITY_MINUTES=$((CITY_DURATION / 60))

    echo ""
    echo "$city completed in $CITY_MINUTES minutes"
    echo ""
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo ""
echo "=============================================="
echo "ALL COMPLETE"
echo "=============================================="
echo "Finished: $(date)"
echo "Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo ""
echo "Models saved:"
for city in "${CITIES[@]}"; do
    echo "  - models/saved/$city/ordinal_catboost_optuna_AUC.pkl"
done
echo "=============================================="
