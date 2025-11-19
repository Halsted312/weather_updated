#!/usr/bin/env python3
"""
CatBoost regression helpers for near-term EV modeling.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold

from ml.catboost_model import build_monotone_constraints

logger = logging.getLogger(__name__)


def _regression_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "depth": trial.suggest_int("depth", 3, 8),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "use_temp_to_floor": trial.suggest_categorical("use_temp_to_floor", [True, False]),
        "use_temp_to_cap": trial.suggest_categorical("use_temp_to_cap", [True, False]),
        "use_spread": trial.suggest_categorical("use_spread", [True, False]),
        "minutes_constraint": trial.suggest_categorical("minutes_constraint", [-1, 0, 1]),
        "task_type": "CPU",
        "random_state": 42,
        "loss_function": "RMSE",
        "verbose": False,
        "early_stopping_rounds": 50,
    }
    return params


def tune_ev_catboost(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    bracket_type: str,
    feature_cols: List[str],
    n_trials: int = 30,
    seed: int = 42,
) -> Tuple[Dict[str, Any], optuna.Study]:
    logger.info("Starting EV CatBoost tuning with %s trials...", n_trials)

    def objective(trial: optuna.Trial) -> float:
        params = _regression_search_space(trial)
        use_temp_to_floor = params.pop("use_temp_to_floor")
        use_temp_to_cap = params.pop("use_temp_to_cap")
        use_spread = params.pop("use_spread")
        minutes_constraint = params.pop("minutes_constraint")

        monotone = build_monotone_constraints(
            feature_cols,
            bracket_type,
            use_temp_to_floor,
            use_temp_to_cap,
            use_spread,
            minutes_constraint,
        )

        gkf = GroupKFold(n_splits=4)
        scores = []
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            train_pool = Pool(X.iloc[train_idx], y[train_idx])
            val_pool = Pool(X.iloc[val_idx], y[val_idx])
            model = CatBoostRegressor(
                **params,
                monotone_constraints=monotone,
            )
            model.fit(train_pool, eval_set=val_pool, verbose=False)
            pred = model.predict(val_pool)
            rmse = mean_squared_error(y[val_idx], pred, squared=False)
            scores.append(rmse)
            trial.report(rmse, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    logger.info("Best EV CatBoost trial: rmse=%.4f params=%s", study.best_value, study.best_params)
    return study.best_params, study


def fit_ev_catboost(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: List[str],
    bracket_type: str,
    params: Dict[str, Any],
) -> CatBoostRegressor:
    params = params.copy()
    use_temp_to_floor = params.pop("use_temp_to_floor", True)
    use_temp_to_cap = params.pop("use_temp_to_cap", True)
    use_spread = params.pop("use_spread", False)
    minutes_constraint = params.pop("minutes_constraint", 0)

    monotone = build_monotone_constraints(
        feature_cols,
        bracket_type,
        use_temp_to_floor,
        use_temp_to_cap,
        use_spread,
        minutes_constraint,
    )

    model = CatBoostRegressor(
        **params,
        monotone_constraints=monotone,
        loss_function=params.get("loss_function", "RMSE"),
    )
    train_pool = Pool(X, y)
    model.fit(train_pool, verbose=False)
    return model


def save_ev_artifacts(
    model: CatBoostRegressor,
    win_dir: Path,
    model_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    study: optuna.Study,
) -> None:
    win_dir.mkdir(parents=True, exist_ok=True)
    model_path = win_dir / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    params_path = win_dir / f"{model_name}_params.json"
    payload = {
        "best_params": params,
        "metrics": metrics,
        "optuna": {
            "best_value": study.best_value,
            "n_trials": len(study.trials),
        },
    }
    with open(params_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


__all__ = ["tune_ev_catboost", "fit_ev_catboost", "save_ev_artifacts"]
