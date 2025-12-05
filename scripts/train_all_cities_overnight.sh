#!/bin/bash
# Overnight training script for all 6 cities
# Runs sequentially to avoid resource contention
# 200 trials, cv=5, objective=within2

set -e  # Exit on error

CITIES=("austin" "chicago" "denver" "los_angeles" "miami" "philadelphia")
TRIALS=200
CV_SPLITS=5
OBJECTIVE="within2"
LOG_DIR="logs/overnight_training_$(date +%Y%m%d)"

# Create log directory
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "OVERNIGHT ORDINAL TRAINING - ALL CITIES"
echo "=============================================="
echo "Started: $(date)"
echo "Trials: $TRIALS"
echo "CV Splits: $CV_SPLITS"
echo "Objective: $OBJECTIVE"
echo "Log directory: $LOG_DIR"
echo "=============================================="

# Track timing
START_TIME=$(date +%s)

for city in "${CITIES[@]}"; do
    echo ""
    echo "=============================================="
    echo "TRAINING: $city"
    echo "Started: $(date)"
    echo "=============================================="

    CITY_START=$(date +%s)

    PYTHONPATH=. python3 scripts/train_city_ordinal_optuna.py \
        --city "$city" \
        --trials $TRIALS \
        --cv-splits $CV_SPLITS \
        --objective "$OBJECTIVE" \
        2>&1 | tee "$LOG_DIR/${city}_training.log"

    CITY_END=$(date +%s)
    CITY_DURATION=$((CITY_END - CITY_START))
    CITY_MINUTES=$((CITY_DURATION / 60))

    echo ""
    echo "$city completed in $CITY_MINUTES minutes"
    echo "Log saved to: $LOG_DIR/${city}_training.log"
    echo ""
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo ""
echo "=============================================="
echo "ALL CITIES COMPLETE"
echo "=============================================="
echo "Finished: $(date)"
echo "Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo ""
echo "Results saved to:"
for city in "${CITIES[@]}"; do
    echo "  - models/saved/$city/ordinal_catboost_optuna.pkl"
done
echo ""
echo "Logs saved to: $LOG_DIR/"
echo "=============================================="
