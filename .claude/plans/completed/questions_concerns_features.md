
python models/pipeline/03_train_ordinal.py --city austin --trials 80 --cv-splits 4 --workers 16

python models/pipeline/04_train_edge_classifier.py \
  --city austin \
  --trials 150 \
  --workers 20 \
  --sample-rate 1 \
  --optuna-metric filtered_precision

This is the error I was seeing:
02:30:42,892 - delta_sweep_gpu_pipeline - INFO - Best model placeholder saved to models/saved/chicago/delta_range_sweep_gpu/best_model_gpu.pkl
(.venv) (base) halsted@halsted:/mnt/slow_weather_updated$ python models/pipeline/03_train_ordinal.py --city chicago --trials 80 --cv-splits 4 --workers 28
03:02:04 [INFO] train_ordinal_pipeline: Invoking train_city_ordinal_optuna with args: --city chicago --trials 80 --workers 28 --cv-splits 4 --use-cached
03:02:04 [INFO] scripts.train_city_ordinal_optuna: ============================================================
03:02:04 [INFO] scripts.train_city_ordinal_optuna: CHICAGO Optuna Training (80 trials)
03:02:04 [INFO] scripts.train_city_ordinal_optuna: ============================================================
03:02:06 [INFO] src.db.connection: Database engine created: localhost:5434/kalshi_weather
03:02:06 [INFO] scripts.train_city_ordinal_optuna: Available data: 2023-01-01 to 2025-11-27
03:02:06 [INFO] scripts.train_city_ordinal_optuna: Total days: 1062
03:02:06 [INFO] scripts.train_city_ordinal_optuna: Training: 2023-01-01 to 2025-04-28 (850 days)
03:02:06 [INFO] scripts.train_city_ordinal_optuna: Testing:  2025-04-29 to 2025-11-27 (212 days)
03:02:06 [INFO] scripts.train_city_ordinal_optuna: Loading cached datasets...
03:02:07 [INFO] scripts.train_city_ordinal_optuna: Loaded train: 387,504 rows, test: 96,672 rows
03:02:07 [INFO] scripts.train_city_ordinal_optuna: 
Training samples: 387,504
03:02:07 [INFO] scripts.train_city_ordinal_optuna: Training days: 850
03:02:07 [INFO] scripts.train_city_ordinal_optuna: Test samples: 96,672
03:02:07 [INFO] scripts.train_city_ordinal_optuna: Test days: 212
03:02:07 [INFO] scripts.train_city_ordinal_optuna: 
Station-city features: ['station_city_temp_gap', 'station_city_max_gap_sofar', 'station_city_mean_gap_sofar', 'station_city_gap_std', 'station_city_gap_trend']
03:02:07 [INFO] scripts.train_city_ordinal_optuna:   station_city_temp_gap: 0/387,504 non-null (0.0%)
03:02:07 [INFO] scripts.train_city_ordinal_optuna:   station_city_max_gap_sofar: 0/387,504 non-null (0.0%)
03:02:07 [INFO] scripts.train_city_ordinal_optuna:   station_city_mean_gap_sofar: 0/387,504 non-null (0.0%)
03:02:07 [INFO] scripts.train_city_ordinal_optuna:   station_city_gap_std: 0/387,504 non-null (0.0%)
03:02:07 [INFO] scripts.train_city_ordinal_optuna:   station_city_gap_trend: 0/387,504 non-null (0.0%)
03:02:07 [INFO] scripts.train_city_ordinal_optuna: 
Multi-horizon features: ['fcst_multi_std', 'fcst_multi_mean', 'fcst_multi_drift']
03:02:07 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_std: 387,504/387,504 non-null (100.0%)
03:02:07 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_mean: 387,504/387,504 non-null (100.0%)
03:02:07 [INFO] scripts.train_city_ordinal_optuna:   fcst_multi_drift: 0/387,504 non-null (0.0%)

============================================================
OPTUNA TRAINING (80 trials)
============================================================
03:02:07 [INFO] models.training.ordinal_trainer: Training ordinal model (catboost) on 387504 samples
03:02:07 [WARNING] models.training.base_trainer: Missing columns (will fill with NaN): ['settle_f_lag1', 'settle_f_lag2', 'settle_f_lag7', 'vc_max_f_lag1', 'vc_max_f_lag7', 'delta_vcmax_lag1']
03:02:07 [INFO] models.training.ordinal_trainer: City delta range: [-10, 10]
03:02:07 [INFO] models.training.ordinal_trainer: Training 20 threshold classifiers: [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
03:02:07 [INFO] models.training.ordinal_trainer: Starting Optuna tuning with 80 trials


agenet told me:
Why fcst_multi_* is mostly null: the builder requests T-1..T-6 forecasts (load_multi_horizon_forecasts → load_historical_forecast_daily), but compute_multi_horizon_features only fills values if it sees at least two horizons with tempmax_f. If most entries in fcst_multi are None (e.g., only T-1 exists; T-2..T-6 missing in DB), all three features go to None. fcst_multi_std/mean are 100% because they’re falling back to zeros when ≥2 temps exist; fcst_multi_drift is None if either T-1 or T-6 is missing, which is why it’s 0% non-null. To fix: ensure historical daily forecasts for lead days 1–6 are ingested for the whole date span, then rebuild the dataset (Step 1) so the multi-horizon dict has real values.

Station-city features at 0%: the code already sets include_station_city=True by default, loads city obs via _load_city_observations (wx.vc_minute_weather with location_type='city'). If that query returns empty (e.g., no city-level obs ingested or wrong vc_location_id), all five station_city_* fields stay None. Fix is data, not a flag: ingest city-aggregate obs for the date range, then rebuild with Step 1.

Lag features warnings: the current snapshot pipeline never calls the lag helpers (compute_lag_features/add_lag_features_to_dataframe), so settle_f_lag*, vc_max_f_lag*, delta_vcmax_lag1 aren’t present in the parquet, hence the “Missing columns (will fill with NaN)” warning. They’re not being computed at all, so they’re not a leakage risk; they just aren’t used. If you want them, you’d need to wire add_lag_features_to_dataframe into the dataset build after snapshots are assembled. These lags are based on prior days’ settle_f and VC max, so they’re time-safe if implemented.

Flags: you already ran with defaults that request all features (include_multi_horizon=True, include_station_city=True, include_market=True, include_meteo=True). The absence is due to missing upstream data (multi-horizon forecasts, city obs) and unused lag code, not a missing CLI flag.

Recommended next steps:

Verify/ingest historical daily forecasts for T-1..T-6 for Chicago; re-run 01_build_dataset.py (rebuild) so fcst_multi_* gets populated.
Verify city-level obs ingestion for Chicago; re-run 01_build_dataset.py so station_city_* fills.
If you want lag features, add the add_lag_features_to_dataframe step in the dataset build and rebuild; otherwise ignore the warning.



Findings on the three gaps:

Multi-horizon forecasts (fcst_multi_*)
Origin: built in compute_multi_horizon_features (models/features/forecast.py) after load_multi_horizon_forecasts (models/data/loader.py) fetches T-1..T-6 via load_historical_forecast_daily.
Why fcst_multi_drift is all null: drift needs both T-1 and T-6 tempmax; the code sets drift to None if either is missing. You have data for T-1..T-? such that fcst_multi_std/mean get values (at least two temps exist), but T-6 (or T-1) is missing for most rows, so drift stays None.
Ingestion link: daily historical forecasts live in wx.vc_forecast_daily. They’re populated by scripts/ingest_vc_historical_forecast.py (backfill) and scripts/poll_vc_live_daemon.py (live snapshots). If those tables don’t have T-6 for your dates/city, drift won’t fill. Next check: select date(forecast_basis_date), count(*) from wx.vc_forecast_daily where data_type='historical_forecast' and city_id='chicago' group by 1 order by 1; and specifically basis_date = target_date-6. If sparse, rerun scripts/ingest_vc_historical_forecast.py --city chicago --start ... --end ... with enough history.
Station-city features (station_city_*)
Origin: build_snapshot (models/features/pipeline.py) pulls city obs via _load_city_observations (models/data/dataset.py) -> wx.vc_minute_weather with location_type='city'. If that query returns empty, all five fields are filled with None.
Ingestion link: city obs are ingested by scripts/ingest_vc_obs_backfill.py when run with --location-type city (or default all locations) for your city code (CHI, DEN, etc.). If you only ingested station obs, city obs are missing. Next check: select min(datetime_local), max(datetime_local), count(*) from wx.vc_minute_weather mw join wx.vc_location l on mw.vc_location_id=l.id where l.location_type='city' and l.city_code='CHI'; If empty or short, rerun ingest_vc_obs_backfill.py --city-code CHI --location-type city --all-history (or desired window), then rebuild the dataset (Step 1).
Lag features (settle_f_lag*, vc_max_f_lag*, delta_vcmax_lag1)
Origin: functions exist in models/features/calendar.py (compute_lag_features, add_lag_features_to_dataframe), but they are not wired into the snapshot builder. build_snapshot never calls them, and the dataset builder (models/data/dataset.py) doesn’t add them post-assembly. That’s why they’re absent and get NaN fill warnings. This is a missing linkage, not data leakage.
If you want them, you need to call add_lag_features_to_dataframe after assembling the snapshot DataFrame in build_dataset/build_dataset_parallel, before saving parquets. These lags are time-safe (use prior days’ settle_f/VC max).
What to do now:

For multi-horizon: verify wx.vc_forecast_daily coverage for T-1..T-6; backfill with scripts/ingest_vc_historical_forecast.py if gaps, then rerun 01_build_dataset.py (rebuild).
For station-city: verify city obs in wx.vc_minute_weather with location_type='city'; backfill with scripts/ingest_vc_obs_backfill.py --location-type city --city-code CHI (or all-history) if missing, then rerun 01_build_dataset.py.
For lags: wire add_lag_features_to_dataframe into the dataset build, or ignore the warning. They’re currently unused by design.

DEBUGGING: Null Features in Training (Dec 2, 2025)
Database Audit Results
=== vc_minute_weather (observations) ===
ALL cities/station: 307K rows (2022-12-31 to 2025-12-02) ✓ GOOD
ALL cities/city:    ~1,300 rows (2025-11-27 to 2025-12-02) ✗ ONLY 5 DAYS!

=== vc_forecast_daily (historical forecasts) ===
ALL cities: T-0, T-1, T-2, T-3 only (1,060 rows each)
MISSING: T-4, T-5, T-6 for all cities

=== forecast_snapshot ===
EMPTY (0 rows)

=== minute_obs ===
EMPTY (0 rows)
Root Causes (Confirmed)
Issue	Root Cause	Impact
Station-city (0%)	City obs never backfilled historically	All 5 station_city_* features null
fcst_multi_drift (0%)	Forecasts ingested with --horizon-days 4 instead of 7	Drift requires T-6, only have T-0 to T-3
Lag features	Code exists but not wired into dataset builder	6 lag columns missing from parquet
FIX PLAN
Step 1: Backfill City Observations (ALL cities)
# Takes ~2-3 hours per city due to VC API rate limits
for city in CHI AUS DEN LAX MIA PHL; do
  python scripts/ingest_vc_obs_backfill.py \
    --city-code $city \
    --location-type city \
    --all-history \
    --batch-days 7
done
Step 2: Backfill T-4 to T-6 Forecasts (ALL cities)
# Re-run with horizon=7 to get leads 0-6
python scripts/ingest_vc_historical_forecast.py \
  --start-date 2023-01-01 \
  --end-date 2025-11-27 \
  --horizon-days 7
Note: This will upsert, so existing T-0 to T-3 won't be duplicated.
Step 3: Wire Up Lag Features (Code Change)
In models/data/dataset.py, after building DataFrame:
from models.features.calendar import add_lag_features_to_dataframe
df_combined = pd.DataFrame(all_rows)
df_combined = add_lag_features_to_dataframe(df_combined)  # ADD THIS
Step 4: Rebuild Datasets
# Delete cached parquets
rm models/saved/chicago/train_data_full.parquet
rm models/saved/chicago/test_data_full.parquet

# Rebuild
python models/pipeline/01_build_dataset.py --city chicago --workers 28
Time Estimates
Task	Time	API Calls
City obs backfill (6 cities)	~12-18 hours	~6000 calls
Forecast backfill (T-4,5,6)	~2-3 hours	~2000 calls
Dataset rebuild	~30 min	0
Quick Workaround (Skip for Now)
Train with reduced features:
python models/pipeline/03_train_ordinal.py --city chicago --no-station-city --trials 80
Model will work, just without station-city and with drift=null.