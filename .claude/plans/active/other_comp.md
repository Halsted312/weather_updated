PYTHONPATH=. nohup python3 scripts/train_city_ordinal_optuna.py \
    --city chicago \
    --trials 100 \
    --cv-splits 5 \
    --objective auc \
    > logs/auc_training_20251205/chicago_auc_solo.log 2>&1 &
echo "Chicago PID: $!"