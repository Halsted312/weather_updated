#!/bin/bash
# Comprehensive multi-window backtest for Kalshi weather models
# Trains and tests 8 evenly-spaced windows across the full data range

set -e  # Exit on error

# Configuration
CITY="chicago"
BRACKET="between"
TRAIN_DAYS=90
ELASTICNET_TRIALS=40
CATBOOST_TRIALS=60
RESULTS_DIR="results/multiwindow"

# Create results directory
mkdir -p $RESULTS_DIR

# Define the 8 windows (start_date, end_date, test_start, test_end)
declare -a WINDOWS=(
    "2024-03-08:2024-06-12:2024-06-06:2024-06-12"
    "2024-05-10:2024-08-14:2024-08-08:2024-08-14"
    "2024-07-19:2024-10-23:2024-10-17:2024-10-23"
    "2024-09-27:2025-01-01:2024-12-26:2025-01-01"
    "2024-12-06:2025-03-12:2025-03-06:2025-03-12"
    "2025-02-14:2025-05-21:2025-05-15:2025-05-21"
    "2025-04-25:2025-07-30:2025-07-24:2025-07-30"
    "2025-07-04:2025-10-08:2025-10-02:2025-10-08"
)

echo "======================================================================="
echo "MULTI-WINDOW BACKTEST PIPELINE"
echo "======================================================================="
echo "City: $CITY | Bracket: $BRACKET"
echo "Windows: ${#WINDOWS[@]} | Train days: $TRAIN_DAYS | Test days: 7 each"
echo "Total test coverage: $((${#WINDOWS[@]} * 7)) days"
echo "======================================================================="

# Function to train a model
train_model() {
    local model_type=$1
    local start_date=$2
    local end_date=$3
    local trials=$4

    echo ""
    echo "Training $model_type: $start_date to $end_date..."

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
        --output-json $output_file

    if [ $? -eq 0 ]; then
        echo "✓ Saved results to $output_file"
    else
        echo "✗ Backtest failed"
        return 1
    fi
}

# Main execution loop
WINDOW_NUM=1
for window in "${WINDOWS[@]}"; do
    IFS=':' read -r start_date end_date test_start test_end <<< "$window"

    echo ""
    echo "======================================================================="
    echo "WINDOW $WINDOW_NUM/${#WINDOWS[@]}"
    echo "Training: $start_date to $end_date | Testing: $test_start to $test_end"
    echo "======================================================================="

    # Train ElasticNet
    if train_model "elasticnet" "$start_date" "$end_date" "$ELASTICNET_TRIALS"; then
        # Run ElasticNet backtest
        run_backtest "elasticnet" "$test_start" "$test_end" "$WINDOW_NUM"
    fi

    # Train CatBoost
    if train_model "catboost" "$start_date" "$end_date" "$CATBOOST_TRIALS"; then
        # Run CatBoost backtest
        run_backtest "catboost" "$test_start" "$test_end" "$WINDOW_NUM"
    fi

    WINDOW_NUM=$((WINDOW_NUM + 1))
done

echo ""
echo "======================================================================="
echo "AGGREGATING RESULTS"
echo "======================================================================="

# Run aggregation script
python scripts/aggregate_backtest_results.py --results-dir $RESULTS_DIR

echo ""
echo "======================================================================="
echo "MULTI-WINDOW BACKTEST COMPLETE"
echo "======================================================================="
echo "Results saved to: $RESULTS_DIR/"
echo "Aggregated summary: $RESULTS_DIR/aggregated_summary.json"
echo ""
echo "To view summary: cat $RESULTS_DIR/aggregated_summary.json | python -m json.tool"
echo "======================================================================="