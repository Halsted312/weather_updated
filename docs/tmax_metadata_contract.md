# Tmax Artifact Contract

This contract documents the fields emitted by the updated Tmax ensemble so
downstream agents can load predictions without reading the trainer.

## `results/tmax_preds_<city>_<tag>.csv`

| Column | Description |
| --- | --- |
| `timestamp` | UTC minute for the snapshot |
| `date` | Local date corresponding to the row |
| `minute_of_day` | Local minute-of-day (0–1439) |
| `pred_raw` | Uncalibrated ensemble output |
| `pred` | Calibrated prediction used for trading |
| `pred_gbdt`, `pred_spline`, `pred_seq`, `pred_catboost` | Component-level predictions (CatBoost present when enabled) |
| `actual` | Observed CLI Tmax used during training |
| `residual` | `pred - actual` |
| `sigma_est` | Dynamic sigma from the residual regressor |
| `pred_p10`, `pred_p90` | Quantile estimates computed via `pred ± z * sigma_est` |

These CSVs are suffixed with the provided batch `--tag` so automation can keep
multiple versions side-by-side (e.g., `tmax_preds_chicago_latest.csv`).

## `results/tmax_model_<city>_<tag>.json`

Top-level fields:

| Field | Description |
| --- | --- |
| `city`, `start`, `end` | Training window metadata |
| `components` | Ordered list of component names (e.g., `['gbdt','spline','seq','catboost']`) |
| `component_weights` | Mapping `{component: weight}` after Optuna tuning |
| `gbdt_params`, `spline_params`, `sequence_params`, `catboost_params` | Model-specific hyperparameters |
| `calibration` | Linear calibration slope/intercept |
| `sigma_features` | Columns used by the sigma regressor |
| `quantiles` | Column names for the exported quantiles (`p10`, `p90`) |
| `seq_model_path` | TorchScript artifact path when `--export-seq-model` used |

Consumers should rely on `components` + `component_weights` rather than
assuming a fixed ordering. The new contract ensures additional components (e.g.,
future CatBoost heads) can be added without breaking loaders.
