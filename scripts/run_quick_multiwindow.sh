#!/bin/bash
# Quick multi-window backtest with 4 windows for faster testing
# Uses smaller trial counts for speed

set -e  # Exit on error

# Configuration
CITY="chicago"
BRACKET="between"
TRAIN_DAYS=90
ELASTICNET_TRIALS=5   # Reduced for speed
CATBOOST_TRIALS=2     # Reduced for speed
RESULTS_DIR="results/multiwindow_quick"

# Create results directory
mkdir -p $RESULTS_DIR

# Define 4 evenly-spaced windows
declare -a WINDOWS=(
    "2024-07-19:2024-10-23:2024-10-17:2024-10-23"
    "2024-12-06:2025-03-12:2025-03-06:2025-03-12"
    "2025-04-25:2025-07-30:2025-07-24:2025-07-30"
    "2025-07-04:2025-10-08:2025-10-02:2025-10-08"
)

echo "======================================================================="
echo "QUICK MULTI-WINDOW BACKTEST (4 Windows)"
echo "======================================================================="
echo "City: $CITY | Bracket: $BRACKET"
echo "Windows: ${#WINDOWS[@]} | Train days: $TRAIN_DAYS | Test days: 7 each"
echo "Total test coverage: $((${#WINDOWS[@]} * 7)) days"
echo "Note: Using reduced trials for speed (EN=$ELASTICNET_TRIALS, CB=$CATBOOST_TRIALS)"
echo "======================================================================="

# Function to train a model
train_model() {
    local model_type=$1
    local start_date=$2
    local end_date=$3
    local trials=$4

    echo ""
    echo "Training $model_type: $start_date to $end_date (trials=$trials)..."

    python ml/train_walkforward.py \
        --city $CITY \
        --bracket $BRACKET \
        --start $start_date \
        --end $end_date \
        --train-days $TRAIN_DAYS \
        --model-type $model_type \
        --trials $trials \
        --feature-set baseline

    if [ $? -eq 0 ]; then
        echo "✓ $model_type training complete"
        return 0
    else
        echo "✗ $model_type training failed"
        return 1
    fi
}

# Function to run backtest
run_backtest() {
    local model_type=$1
    local test_start=$2
    local test_end=$3
    local window_id=$4

    local output_file="$RESULTS_DIR/${model_type}_win${window_id}.json"

    echo "Backtesting $model_type: $test_start to $test_end..."

    python backtest/run_backtest.py \
        --city $CITY \
        --bracket $BRACKET \
        --strategy model_kelly \
        --model-type $model_type \
        --start-date $test_start \
        --end-date $test_end \
        --output-json $output_file \
        --initial-cash 10000

    if [ $? -eq 0 ]; then
        echo "✓ Saved results to $output_file"
        return 0
    else
        echo "✗ Backtest failed"
        return 1
    fi
}

# Main execution loop
WINDOW_NUM=1
TOTAL_START=$(date +%s)

for window in "${WINDOWS[@]}"; do
    IFS=':' read -r start_date end_date test_start test_end <<< "$window"

    echo ""
    echo "======================================================================="
    echo "WINDOW $WINDOW_NUM/${#WINDOWS[@]}"
    echo "Training: $start_date to $end_date | Testing: $test_start to $test_end"
    echo "======================================================================="

    WINDOW_START=$(date +%s)

    # Train and test both models
    train_model "elasticnet" "$start_date" "$end_date" "$ELASTICNET_TRIALS" && \
        run_backtest "elasticnet" "$test_start" "$test_end" "$WINDOW_NUM"

    train_model "catboost" "$start_date" "$end_date" "$CATBOOST_TRIALS" && \
        run_backtest "catboost" "$test_start" "$test_end" "$WINDOW_NUM"

    WINDOW_END=$(date +%s)
    WINDOW_TIME=$((WINDOW_END - WINDOW_START))
    echo "Window $WINDOW_NUM completed in ${WINDOW_TIME} seconds"

    WINDOW_NUM=$((WINDOW_NUM + 1))
done

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo ""
echo "======================================================================="
echo "AGGREGATING RESULTS"
echo "======================================================================="

# Run aggregation script
python scripts/aggregate_backtest_results.py --results-dir $RESULTS_DIR

echo ""
echo "======================================================================="
echo "QUICK MULTI-WINDOW BACKTEST COMPLETE"
echo "======================================================================="
echo "Total execution time: ${TOTAL_TIME} seconds ($(($TOTAL_TIME / 60)) minutes)"
echo "Results saved to: $RESULTS_DIR/"
echo "Aggregated summary: $RESULTS_DIR/aggregated_summary.json"
echo ""
echo "To view summary:"
echo "  cat $RESULTS_DIR/aggregated_summary.json | python -m json.tool"
echo ""
echo "To run full backtest with more trials:"
echo "  ./scripts/run_multiwindow_backtest.sh"
echo "======================================================================="