#!/usr/bin/env python3
"""Train per-minute regressor to predict daily CLI Tmax from partial-day temperatures."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
import torch
from torch import nn
import optuna

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostRegressor = None

from db.connection import get_session
from ml.city_config import CITY_CONFIG
from numpy.lib.stride_tricks import sliding_window_view


SEQ_WINDOW_MINUTES = 180
SEQ_STEP_MINUTES = 5
BIAS_BIN_MINUTES = 30
@dataclass
class SplineParams:
    smoothing_multiplier: float = 0.5
    degree: int = 3
    horizon_minutes: int = 60


@dataclass
class SeqParams:
    hidden_dim: int = 128
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 25
    batch_size: int = 256
    sequence_minutes: int = 180
    step_minutes: int = 5


@dataclass
class Dataset:
    features: pd.DataFrame
    labels: pd.Series
    dates: pd.Series
    minute_of_day: pd.Series
    timestamps: pd.Series


CATBOOST_MONOTONE_HINTS: Dict[str, int] = {
    "minute_of_day": 1,
    "minute_norm": 1,
    "running_max": 1,
    "running_min": 1,
    "delta_from_max": -1,
    "rolling_mean_30": 1,
    "rolling_mean_60": 1,
    "prior_tmax": 1,
}


def split_date_windows(unique_dates: List[date]) -> Tuple[List[date], List[date], List[date]]:
    """Return train/validation/test date buckets."""
    total = len(unique_dates)
    if total < 40:
        raise ValueError("Need at least 40 distinct days for train/validation/test splits")

    n_test = max(7, int(total * 0.2))
    test_dates = unique_dates[-n_test:]
    train_val = unique_dates[:-n_test]

    n_val = max(5, int(len(train_val) * 0.2))
    if len(train_val) - n_val < 10:
        raise ValueError("Train window too small after reserving validation/test sets")

    val_dates = train_val[-n_val:]
    train_dates = train_val[:-n_val]
    return train_dates, val_dates, test_dates


def load_minute_obs(city: str, start: date, end: date) -> pd.DataFrame:
    loc_id = CITY_CONFIG[city]["loc_id"]
    with get_session() as session:
        query = text(
            """
            SELECT ts_utc, temp_f
            FROM wx.minute_obs
            WHERE loc_id = :loc_id
              AND ts_utc >= :start_dt
              AND ts_utc < :end_dt
            ORDER BY ts_utc
            """
        )
        params = {
            "loc_id": loc_id,
            "start_dt": datetime.combine(start, datetime.min.time(), tzinfo=ZoneInfo("UTC")),
            "end_dt": datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=ZoneInfo("UTC")),
        }
        df = pd.read_sql(query, session.bind, params=params)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def load_settlements(city: str, start: date, end: date) -> pd.DataFrame:
    with get_session() as session:
        query = text(
            """
            SELECT date_local, tmax_cli as tmax
            FROM wx.settlement
            WHERE city = :city
              AND date_local BETWEEN :start_date AND :end_date
              AND tmax_cli IS NOT NULL
            ORDER BY date_local
            """
        )
        sett = pd.read_sql(query, session.bind, params={"city": city, "start_date": start, "end_date": end})
    sett["date_local"] = pd.to_datetime(sett["date_local"]).dt.date
    return sett


def build_day_groups(df_full: pd.DataFrame) -> Dict[date, pd.DataFrame]:
    groups: Dict[date, pd.DataFrame] = {}
    for day, df_day in df_full.groupby("date_local"):
        df_day = df_day.sort_values("timestamp").reset_index(drop=True)
        groups[day] = df_day
    return groups


def build_feature_frame(city: str, start: date, end: date) -> Dataset:
    tz = ZoneInfo(CITY_CONFIG[city]["timezone"])
    minutes = load_minute_obs(city, start, end)
    if minutes.empty:
        raise ValueError("No minute observations found")
    settlements = load_settlements(city, start, end)
    if settlements.empty:
        raise ValueError("No CLI settlements found")

    minutes["ts_local"] = minutes["ts_utc"].dt.tz_convert(tz)
    minutes["date_local"] = minutes["ts_local"].dt.date
    minutes["minute_of_day"] = minutes["ts_local"].dt.hour * 60 + minutes["ts_local"].dt.minute
    minutes = minutes.merge(settlements, left_on="date_local", right_on="date_local", how="inner")

    grouped = minutes.groupby("date_local", sort=True)
    frames: List[pd.DataFrame] = []
    for day, df_day in grouped:
        df_day = df_day.sort_values("ts_local").reset_index(drop=True)
        df_day["temp_f"] = pd.to_numeric(df_day["temp_f"], errors="coerce")
        df_day = df_day.dropna(subset=["temp_f"])
        if df_day.empty:
            continue
        df_day["ts_utc"] = df_day["ts_utc"]
        df_day["running_max"] = df_day["temp_f"].cummax()
        df_day["running_min"] = df_day["temp_f"].cummin()
        df_day["delta_from_max"] = df_day["running_max"] - df_day["temp_f"]
        df_day["rolling_mean_30"] = df_day["temp_f"].rolling(window=6, min_periods=1).mean()
        df_day["rolling_mean_60"] = df_day["temp_f"].rolling(window=12, min_periods=1).mean()
        df_day["rolling_std_60"] = df_day["temp_f"].rolling(window=12, min_periods=2).std().fillna(0.0)
        df_day["slope_30"] = (df_day["temp_f"] - df_day["temp_f"].shift(6)).fillna(0.0) / 30.0
        df_day["slope_60"] = (df_day["temp_f"] - df_day["temp_f"].shift(12)).fillna(0.0) / 60.0
        df_day["running_max_slope"] = (df_day["running_max"] - df_day["running_max"].shift(6)).fillna(0.0) / 30.0
        df_day["minute_norm"] = df_day["minute_of_day"] / (24 * 60)
        frames.append(df_day)

    if not frames:
        raise ValueError("No usable feature rows")
    feat_df = pd.concat(frames, ignore_index=True)

    settlements_sorted = settlements.sort_values("date_local")
    settlements_sorted["prior_tmax"] = settlements_sorted["tmax"].shift(1)
    feat_df = feat_df.merge(settlements_sorted[["date_local", "prior_tmax"]], on="date_local", how="left")
    feat_df["prior_tmax"] = feat_df["prior_tmax"].ffill()

    feature_cols = [
        "minute_of_day",
        "minute_norm",
        "temp_f",
        "running_max",
        "running_min",
        "delta_from_max",
        "rolling_mean_30",
        "rolling_mean_60",
        "rolling_std_60",
        "slope_30",
        "slope_60",
        "running_max_slope",
        "prior_tmax",
    ]
    feat_df = feat_df.dropna(subset=feature_cols + ["tmax"])
    feat_df["timestamp"] = pd.to_datetime(feat_df["ts_utc"], utc=True)

    return Dataset(
        features=feat_df[feature_cols],
        labels=feat_df["tmax"],
        dates=feat_df["date_local"],
        minute_of_day=feat_df["minute_of_day"],
        timestamps=feat_df["timestamp"],
    )


def build_catboost_monotone(feature_cols: List[str]) -> List[int]:
    return [CATBOOST_MONOTONE_HINTS.get(col, 0) for col in feature_cols]


def train_catboost_component(
    data: Dataset,
    mask_train: np.ndarray,
    mask_val: np.ndarray,
    mask_trainval: np.ndarray,
    n_trials: int,
) -> Tuple[Optional[CatBoostRegressor], np.ndarray, Dict[str, Any]]:
    if n_trials <= 0 or CatBoostRegressor is None:
        return None, np.full(len(data.labels), np.nan), {}

    X = data.features.to_numpy()
    y = data.labels.to_numpy()
    monotone = build_catboost_monotone(list(data.features.columns))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            "iterations": trial.suggest_int("iterations", 300, 800),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        }
        model = CatBoostRegressor(
            loss_function="RMSE",
            allow_writing_files=False,
            verbose=False,
            monotone_constraints=monotone,
            **params,
        )
        model.fit(X[mask_train], y[mask_train], verbose=False)
        preds = model.predict(X[mask_val])
        return mean_absolute_error(y[mask_val], preds)

    study = optuna.create_study(direction="minimize", study_name="catboost_tmax")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_trial.params
    final_model = CatBoostRegressor(
        loss_function="RMSE",
        allow_writing_files=False,
        verbose=False,
        monotone_constraints=monotone,
        **best_params,
    )
    final_model.fit(X[mask_trainval], y[mask_trainval], verbose=False)
    preds = final_model.predict(X)
    meta = {"params": best_params, "monotone": monotone}
    return final_model, preds, meta


def _fit_spline_with_retry(x: np.ndarray, y: np.ndarray, degree: int, smoothing: float) -> Optional[UnivariateSpline]:
    """Fit a spline while progressively relaxing smoothing if VC data is noisy."""
    cur_s = smoothing
    for _ in range(5):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            try:
                spline = UnivariateSpline(x, y, k=degree, s=cur_s)
            except Exception:
                spline = None
        has_fp_warning = any("fp" in str(w.message) for w in caught)
        if spline is not None and not has_fp_warning:
            return spline
        cur_s *= 2.0
    return spline


def fit_spline_forecaster(day_df: pd.DataFrame, params: SplineParams) -> float:
    """Return spline-based Tmax prediction for a single day snapshot."""
    if len(day_df) < max(8, params.degree + 2):
        return float(day_df["running_max"].iloc[-1])
    x = day_df["minute_of_day"].to_numpy()
    y = day_df["temp_f"].to_numpy()
    degree = max(1, min(params.degree, len(x) - 1))
    smoothing = max(len(x) * params.smoothing_multiplier, len(x) * 0.1)
    spline = _fit_spline_with_retry(x, y, degree, smoothing)
    if spline is None:
        return float(day_df["running_max"].iloc[-1])

    horizon_end = min(24 * 60, x[-1] + params.horizon_minutes)
    num_points = max(20, params.horizon_minutes // 5)
    future_minutes = np.linspace(x[-1], horizon_end, num_points, dtype=float)
    preds = spline(future_minutes)
    upper = float(day_df["running_max"].max())
    val = float(max(upper, preds.max()))
    lower = float(day_df["running_min"].min())
    return max(lower - 10.0, min(val, upper + 10.0))


def generate_spline_predictions(
    day_groups: Dict[date, pd.DataFrame],
    params: SplineParams,
    num_rows: int,
) -> np.ndarray:
    preds = np.empty(num_rows, dtype=float)
    for df_day in day_groups.values():
        running_preds = []
        for idx in range(len(df_day)):
            cutoff_df = df_day.iloc[: idx + 1]
            running_preds.append(fit_spline_forecaster(cutoff_df, params))
        preds[df_day["row_id"].to_numpy()] = running_preds
    return preds


class TempSequenceNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.head(last).squeeze(-1)


def build_sequence_dataset(day_groups: Dict[date, pd.DataFrame], params: SeqParams):
    """Create sequence tensors for neural model."""
    seq_len = max(1, params.sequence_minutes // max(1, params.step_minutes))
    all_windows: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    for df_day in day_groups.values():
        if len(df_day) <= seq_len:
            continue
        temps = df_day["temp_f"].to_numpy(dtype=np.float32, copy=False)
        minutes = df_day["minute_of_day"].to_numpy(dtype=np.float32, copy=False)
        running_max = df_day["running_max"].to_numpy(dtype=np.float32, copy=False)
        prior = np.float32(df_day["prior_tmax"].iloc[0])
        prior_col = np.full_like(minutes, prior, dtype=np.float32)
        arr = np.stack(
            [
                temps,
                running_max,
                minutes / (24 * 60.0),
                prior_col,
            ],
            axis=1,
        )
        windows = sliding_window_view(arr, window_shape=seq_len, axis=0)
        # windows shape: (len - seq_len + 1, seq_len, features); drop final window to align labels
        if len(windows) <= 1:
            continue
        windows = windows[:-1]
        step = max(1, params.step_minutes // 5)
        windows = windows[::step]
        targets = df_day["tmax"].to_numpy(dtype=np.float32, copy=False)[seq_len:]
        targets = targets[::step]
        if len(windows) != len(targets):
            length = min(len(windows), len(targets))
            windows = windows[:length]
            targets = targets[:length]
        all_windows.append(windows.astype(np.float32, copy=False))
        all_targets.append(targets)
    if not all_windows:
        return None, None
    return np.concatenate(all_windows, axis=0), np.concatenate(all_targets, axis=0)


def train_sequence_model(day_groups: Dict[date, pd.DataFrame], params: SeqParams) -> TempSequenceNet:
    samples, targets = build_sequence_dataset(day_groups, params)
    if samples is None:
        raise ValueError("Not enough data for sequence model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TempSequenceNet(input_dim=samples.shape[-1], hidden_dim=params.hidden_dim, dropout=params.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params.lr)
    loss_fn = nn.MSELoss()
    X = torch.tensor(samples, dtype=torch.float32, device=device)
    y = torch.tensor(targets, dtype=torch.float32, device=device)
    dataset_size = len(y)
    batch_size = min(params.batch_size, dataset_size)
    for epoch in range(params.epochs):
        perm = torch.randperm(dataset_size, device=device)
        X = X[perm]
        y = y[perm]
        for idx in range(0, dataset_size, batch_size):
            xb = X[idx: idx + batch_size]
            yb = y[idx: idx + batch_size]
            opt.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
    model.eval()
    return model.cpu()


def predict_sequence_component(
    model: TempSequenceNet,
    day_groups: Dict[date, pd.DataFrame],
    num_rows: int,
    params: SeqParams,
) -> np.ndarray:
    """Generate per-row sequence predictions using trained model."""

    seq_len = max(1, params.sequence_minutes // max(1, params.step_minutes))
    preds = np.full(num_rows, np.nan, dtype=float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    for df_day in day_groups.values():
        df_day = df_day.sort_values("timestamp").reset_index(drop=True)
        if len(df_day) <= seq_len:
            continue

        features = np.stack(
            (
                df_day["temp_f"].to_numpy(dtype=np.float32, copy=False),
                df_day["running_max"].to_numpy(dtype=np.float32, copy=False),
                (df_day["minute_of_day"].to_numpy(dtype=np.float32, copy=False) / (24 * 60.0)),
                np.full(len(df_day), df_day["prior_tmax"].iloc[0], dtype=np.float32),
            ),
            axis=1,
        )
        windows = sliding_window_view(features, window_shape=seq_len, axis=0)
        if len(windows) <= 1:
            continue

        windows = windows[:-1]
        step = max(1, params.step_minutes // 5)
        windows = windows[::step]
        row_ids = df_day["row_id"].to_numpy()
        target_rows = row_ids[seq_len : seq_len + len(windows) * step : step]
        if len(target_rows) != len(windows):
            length = min(len(target_rows), len(windows))
            target_rows = target_rows[:length]
            windows = windows[:length]
        windows = windows.astype(np.float32, copy=False)

        for start in range(0, len(windows), 512):
            batch = np.ascontiguousarray(windows[start : start + 512])
            with torch.no_grad():
                tensor = torch.from_numpy(batch).to(device)
                output = model(tensor).cpu().numpy()
            preds[target_rows[start : start + len(output)]] = output

    return preds


def tune_gbdt_hyperparams(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    val_X: pd.DataFrame,
    val_y: pd.Series,
    n_trials: int,
) -> HistGradientBoostingRegressor:
    if n_trials <= 0:
        model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.15, max_iter=400)
        model.fit(pd.concat([train_X, val_X]), pd.concat([train_y, val_y]))
        return model

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3),
            "max_iter": trial.suggest_int("max_iter", 200, 600),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 120),
        }
        model = HistGradientBoostingRegressor(**params)
        model.fit(train_X, train_y)
        preds = model.predict(val_X)
        return mean_absolute_error(val_y, preds)

    study = optuna.create_study(direction="minimize", study_name="gbdt_tmax")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_trial.params
    model = HistGradientBoostingRegressor(**best_params)
    model.fit(pd.concat([train_X, val_X]), pd.concat([train_y, val_y]))
    return model


def tune_spline_params(
    day_groups: Dict[date, pd.DataFrame],
    val_mask: np.ndarray,
    df_full: pd.DataFrame,
    n_trials: int,
) -> SplineParams:
    if n_trials <= 0:
        return SplineParams()

    y_true = df_full["tmax"].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = SplineParams(
            smoothing_multiplier=trial.suggest_float("smooth_mult", 0.1, 3.0),
            degree=trial.suggest_int("spline_degree", 2, 5),
            horizon_minutes=trial.suggest_int("horizon_minutes", 30, 180),
        )
        preds = generate_spline_predictions(day_groups, params, len(df_full))
        mask = ~np.isnan(preds) & val_mask
        if not mask.any():
            return float("inf")
        return mean_absolute_error(y_true[mask], preds[mask])

    study = optuna.create_study(direction="minimize", study_name="spline_tmax")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial.params
    return SplineParams(
        smoothing_multiplier=best["smooth_mult"],
        degree=int(best["spline_degree"]),
        horizon_minutes=int(best["horizon_minutes"]),
    )


def tune_sequence_params(
    train_groups: Dict[date, pd.DataFrame],
    full_groups: Dict[date, pd.DataFrame],
    df_full: pd.DataFrame,
    val_mask: np.ndarray,
    base_params: SeqParams,
    n_trials: int,
) -> Tuple[SeqParams, TempSequenceNet]:
    if n_trials <= 0:
        model = train_sequence_model(train_groups, base_params)
        return base_params, model

    y_true = df_full["tmax"].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = SeqParams(
            hidden_dim=trial.suggest_int("seq_hidden", 64, 256),
            dropout=trial.suggest_float("seq_dropout", 0.0, 0.3),
            lr=trial.suggest_float("seq_lr", 5e-4, 3e-3, log=True),
            epochs=trial.suggest_int("seq_epochs", 15, 40),
            batch_size=trial.suggest_int("seq_batch", 128, 512, step=64),
            sequence_minutes=trial.suggest_int("seq_window", 120, 240, step=15),
            step_minutes=trial.suggest_int("seq_step", 3, 10),
        )
        try:
            model = train_sequence_model(train_groups, params)
        except ValueError:
            return float("inf")
        preds = predict_sequence_component(model, full_groups, len(df_full), params)
        mask = ~np.isnan(preds) & val_mask
        if not mask.any():
            return float("inf")
        return mean_absolute_error(y_true[mask], preds[mask])

    study = optuna.create_study(direction="minimize", study_name="seq_tmax")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial.params
    tuned = SeqParams(
        hidden_dim=int(best["seq_hidden"]),
        dropout=float(best["seq_dropout"]),
        lr=float(best["seq_lr"]),
        epochs=int(best["seq_epochs"]),
        batch_size=int(best["seq_batch"]),
        sequence_minutes=int(best["seq_window"]),
        step_minutes=int(best["seq_step"]),
    )
    model = train_sequence_model(train_groups, tuned)
    return tuned, model


def tune_component_weights(
    component_preds: np.ndarray,
    labels: np.ndarray,
    val_mask: np.ndarray,
    n_trials: int,
    component_names: List[str],
) -> np.ndarray:
    num_components = len(component_names)

    if n_trials <= 0:
        return np.full(num_components, 1.0 / max(1, num_components))

    def objective(trial: optuna.Trial) -> float:
        weights = np.array(
            [trial.suggest_float(f"w_{name}", 0.0, 1.0) for name in component_names],
            dtype=float,
        )
        if weights.sum() == 0:
            weights = np.full(num_components, 1.0 / max(1, num_components))
        weights = weights / weights.sum()
        preds = np.nansum(component_preds * weights[:, None], axis=0)
        mask = np.isfinite(preds) & val_mask
        if not mask.any():
            return float("inf")
        return mean_absolute_error(labels[mask], preds[mask])

    study = optuna.create_study(direction="minimize", study_name="tmax_weights")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = np.array(
        [study.best_trial.params[f"w_{name}"] for name in component_names],
        dtype=float,
    )
    if best.sum() == 0:
        best = np.full(num_components, 1.0 / max(1, num_components))
    return best / best.sum()


def calibrate_linear(preds: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    valid_mask = np.isfinite(preds) & mask
    if not valid_mask.any():
        return preds, {"slope": 1.0, "intercept": 0.0}
    reg = LinearRegression()
    reg.fit(preds[valid_mask].reshape(-1, 1), labels[valid_mask])
    calibrated = reg.predict(preds.reshape(-1, 1))
    return calibrated, {"slope": float(reg.coef_[0]), "intercept": float(reg.intercept_)}


def export_sequence_model(model: TempSequenceNet, params: SeqParams, export_path: str) -> None:
    seq_len = max(1, params.sequence_minutes // max(1, params.step_minutes))
    input_dim = model.gru.input_size
    dummy = torch.zeros(1, seq_len, input_dim)
    traced = torch.jit.trace(model.cpu(), dummy)
    traced.save(export_path)


def train_and_eval(args, cutoffs: List[int]) -> None:
    city = args.city
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    data = build_feature_frame(city, start, end)
    unique_dates = sorted(data.dates.unique())
    train_dates, val_dates, test_dates = split_date_windows(unique_dates)

    mask_train = data.dates.isin(train_dates)
    mask_val = data.dates.isin(val_dates)
    mask_trainval = data.dates.isin(list(train_dates) + list(val_dates))
    mask_test = data.dates.isin(test_dates)
    mask_train_arr = mask_train.to_numpy()
    mask_val_arr = mask_val.to_numpy()
    mask_trainval_arr = mask_trainval.to_numpy()

    df_full = data.features.copy()
    df_full["date_local"] = data.dates
    df_full["minute_of_day"] = data.minute_of_day
    df_full["timestamp"] = data.timestamps
    df_full["tmax"] = data.labels
    df_full["row_id"] = np.arange(len(df_full))

    day_groups = build_day_groups(df_full)
    train_groups = {d: day_groups[d] for d in train_dates}
    trainval_groups = {d: day_groups[d] for d in list(train_dates) + list(val_dates)}

    print(f"Train days: {len(train_dates)}, val days: {len(val_dates)}, test days: {len(test_dates)}")

    gbdt_model = tune_gbdt_hyperparams(
        data.features.loc[mask_train],
        data.labels.loc[mask_train],
        data.features.loc[mask_val],
        data.labels.loc[mask_val],
        args.optuna_trials,
    )
    preds_gbdt = gbdt_model.predict(data.features)

    spline_params = tune_spline_params(train_groups, mask_val_arr, df_full, args.optuna_trials)
    spline_preds = generate_spline_predictions(day_groups, spline_params, len(df_full))

    base_seq_params = SeqParams(
        hidden_dim=args.seq_hidden_dim,
        dropout=args.seq_dropout,
        lr=1e-3,
        epochs=args.seq_epochs,
        batch_size=args.seq_batch_size,
        sequence_minutes=args.seq_window_minutes,
        step_minutes=args.seq_step_minutes,
    )
    try:
        seq_params, _ = tune_sequence_params(
            train_groups,
            day_groups,
            df_full,
            mask_val_arr,
            base_seq_params,
            args.seq_optuna_trials,
        )
        seq_model = train_sequence_model(trainval_groups, seq_params)
        seq_preds = predict_sequence_component(seq_model, day_groups, len(df_full), seq_params)
    except ValueError as exc:
        print(f"Sequence model skipped: {exc}")
        seq_params = base_seq_params
        seq_model = None
        seq_preds = np.full(len(df_full), np.nan)

    cat_model = None
    cat_preds = np.full(len(df_full), np.nan)
    cat_meta: Dict[str, Any] = {}
    if args.enable_catboost:
        if CatBoostRegressor is None:
            raise RuntimeError("CatBoost is not installed. Install catboost or disable --enable-catboost")
        cat_model, cat_preds, cat_meta = train_catboost_component(
            data,
            mask_train_arr,
            mask_val_arr,
            mask_trainval_arr,
            args.catboost_trials,
        )

    component_outputs: Dict[str, np.ndarray] = {
        "gbdt": preds_gbdt,
        "spline": spline_preds,
        "seq": seq_preds,
    }
    if cat_model is not None:
        component_outputs["catboost"] = cat_preds

    component_names = list(component_outputs.keys())
    component_stack = np.vstack([component_outputs[name] for name in component_names])
    weights = tune_component_weights(
        component_stack,
        data.labels.to_numpy(),
        mask_val_arr,
        args.optuna_trials,
        component_names,
    )
    weight_map = {name: float(weight) for name, weight in zip(component_names, weights.tolist())}
    numer = np.nansum(component_stack * weights[:, None], axis=0)
    denom = np.nansum((~np.isnan(component_stack)) * weights[:, None], axis=0)
    ensemble_pred = np.where(denom > 0, numer / denom, preds_gbdt)

    calibrated_pred, calibration = calibrate_linear(ensemble_pred, data.labels.to_numpy(), mask_trainval_arr)
    minutes_array = data.minute_of_day.to_numpy()
    minute_bins = (minutes_array // BIAS_BIN_MINUTES) * BIAS_BIN_MINUTES
    residuals_all = calibrated_pred - data.labels.to_numpy()
    bias_series = pd.Series(residuals_all[mask_trainval_arr], index=minute_bins[mask_trainval_arr])
    bias_map = bias_series.groupby(level=0).mean().to_dict()
    if bias_map:
        bias_lookup = pd.Series(bias_map)
        calibrated_pred = calibrated_pred - bias_lookup.reindex(minute_bins).fillna(0.0).to_numpy()
        residuals_all = calibrated_pred - data.labels.to_numpy()

    results = data.features.copy()
    results["date"] = data.dates
    results["pred_raw"] = ensemble_pred
    results["pred"] = calibrated_pred
    for name, preds in component_outputs.items():
        results[f"pred_{name}"] = preds
    results["actual"] = data.labels
    results["timestamp"] = data.timestamps

    overall_mae = mean_absolute_error(results.loc[mask_test, "actual"], results.loc[mask_test, "pred"])
    print(f"Overall test MAE (all minutes): {overall_mae:.2f}°F")

    for cutoff in cutoffs:
        subset = results[(results["minute_of_day"] <= cutoff) & mask_test]
        if subset.empty:
            continue
        last_rows = subset.sort_values(["date", "minute_of_day"]).groupby("date").tail(1)
        mae = mean_absolute_error(last_rows["actual"], last_rows["pred"])
        bias = (last_rows["pred"] - last_rows["actual"]).mean()
        print(f"Cutoff {cutoff//60:02d}:{cutoff%60:02d} -> {len(last_rows)} days, MAE {mae:.2f}°F, bias {bias:.2f}")

    results["residual"] = residuals_all

    train_residuals = residuals_all[mask_trainval_arr]
    base_sigma = float(np.std(train_residuals)) if len(train_residuals) else float(np.std(residuals_all))
    if not np.isfinite(base_sigma) or base_sigma <= 0:
        base_sigma = 4.0

    sigma_est = np.full(len(residuals_all), base_sigma)
    sigma_feature_candidates = [
        "minute_of_day",
        "running_std_60",
        "slope_30",
        "slope_60",
        "delta_from_max",
    ]
    sigma_cols = [col for col in sigma_feature_candidates if col in df_full.columns]

    if sigma_cols:
        sigma_features = df_full[sigma_cols].copy().fillna(0.0)
        sigma_model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.1, max_iter=300)
        try:
            sigma_model.fit(
                sigma_features.iloc[mask_trainval_arr],
                np.abs(train_residuals),
            )
            sigma_est = sigma_model.predict(sigma_features).clip(min=0.5)
        except ValueError:
            print("Dynamic sigma model skipped (insufficient data)")

    print(f"Residual std dev (train+val): {base_sigma:.2f}°F")
    print(f"Median dynamic sigma estimate: {np.median(sigma_est):.2f}°F")

    results["sigma_est"] = sigma_est
    results["pred_p10"] = results["pred"] + norm.ppf(0.10) * results["sigma_est"]
    results["pred_p90"] = results["pred"] + norm.ppf(0.90) * results["sigma_est"]

    if args.export_csv:
        component_cols = [f"pred_{name}" for name in component_names]
        cols = [
            "timestamp",
            "date",
            "minute_of_day",
            "pred",
            "pred_raw",
            *component_cols,
            "actual",
            "residual",
            "sigma_est",
            "pred_p10",
            "pred_p90",
        ]
        results[cols].to_csv(args.export_csv, index=False)
        print(f"Saved per-minute predictions to {args.export_csv}")

    metadata = {
        "city": city,
        "start": args.start,
        "end": args.end,
        "gbdt_params": gbdt_model.get_params(),
        "spline_params": vars(spline_params),
        "sequence_params": vars(seq_params),
        "weights": weights.tolist(),
        "component_weights": weight_map,
        "components": component_names,
        "calibration": calibration,
        "bias_correction": {
            "bin_minutes": BIAS_BIN_MINUTES,
            "per_bin": {str(int(k)): float(v) for k, v in bias_map.items()},
        },
        "cutoffs": args.cutoffs,
        "sigma_features": sigma_cols,
        "quantiles": {"p10": "pred_p10", "p90": "pred_p90"},
    }

    if cat_meta:
        metadata["catboost_params"] = cat_meta

    if args.export_seq_model and seq_model is not None:
        export_sequence_model(seq_model, seq_params, args.export_seq_model)
        metadata["seq_model_path"] = args.export_seq_model

    metadata_path = args.export_metadata
    if not metadata_path and args.export_csv:
        metadata_path = str(Path(args.export_csv).with_suffix(".json"))
    if metadata_path:
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        print(f"Wrote metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Tmax regressor from minute data")
    parser.add_argument("--city", required=True, choices=list(CITY_CONFIG.keys()))
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--cutoffs", nargs="*", type=str, default=["14:00", "16:00", "18:00"],
                        help="Local time cutoffs (HH:MM)")
    parser.add_argument("--export-csv", help="Optional path to save per-minute predictions")
    parser.add_argument("--export-metadata", help="Optional path for model metadata JSON")
    parser.add_argument("--export-seq-model", help="Optional TorchScript path for the GRU (fast inference)")
    parser.add_argument("--optuna-trials", type=int, default=25, help="Trials for GBDT/spline weighting Optuna sweeps")
    parser.add_argument("--seq-optuna-trials", type=int, default=5, help="Trials for GRU Optuna tuning (set 0 to skip)")
    parser.add_argument("--seq-epochs", type=int, default=30, help="Base number of GRU epochs when Optuna disabled")
    parser.add_argument("--seq-hidden-dim", type=int, default=128, help="Base GRU hidden size when Optuna disabled")
    parser.add_argument("--seq-dropout", type=float, default=0.1, help="Base dropout for GRU head")
    parser.add_argument("--seq-batch-size", type=int, default=256, help="Base batch size when Optuna disabled")
    parser.add_argument("--seq-window-minutes", type=int, default=SEQ_WINDOW_MINUTES, help="History window (minutes)")
    parser.add_argument("--seq-step-minutes", type=int, default=SEQ_STEP_MINUTES, help="Stride (minutes)")
    parser.add_argument("--enable-catboost", action="store_true", help="Add CatBoost regression component to ensemble")
    parser.add_argument("--catboost-trials", type=int, default=25, help="Optuna trials for CatBoost component")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    cutoffs = []
    for c in args.cutoffs:
        hour, minute = map(int, c.split(":"))
        cutoffs.append(hour * 60 + minute)

    train_and_eval(args, cutoffs)


if __name__ == "__main__":
    main()
