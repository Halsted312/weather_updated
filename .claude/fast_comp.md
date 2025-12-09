(.venv) (base) halsted@halsted:~/Python/weather_updated$ python scripts/train_edge_classifier.py     --city chicago     --from-parquet     --workers 24     --trials 100
20:02:01 [INFO] __main__: Using threshold from config: 10.0°F
20:02:01 [INFO] __main__: Auto-detected candle parquet: models/candles/candles_chicago.parquet
20:02:01 [INFO] __main__: Auto-detected settlements parquet: models/raw_data/chicago/settlements.parquet
============================================================
ML EDGE CLASSIFIER TRAINING
============================================================
City: chicago
Optuna trials: 100
Optuna metric: filtered_precision
Workers: 24
Edge threshold: 10.0°F
Sample rate: every 6th snapshot
P&L mode: REALISTIC (with fees)
Maker fill probability: 40.0%
Ordinal model: models/saved/chicago/ordinal_catboost_optuna.pkl (default)
Candle source: parquet (models/candles/candles_chicago.parquet)
Settlement source: parquet (models/raw_data/chicago/settlements.parquet)
Mode: PARQUET-ONLY (no DB required)

20:02:01 [INFO] __main__: ⚠️  Regenerating: ordinal model changed
20:02:01 [INFO] __main__:    Cached: 1765214224.0
20:02:01 [INFO] __main__:    Current: 1765214224.5963826
20:02:01 [INFO] __main__: Using ordinal model: models/saved/chicago/ordinal_catboost_optuna.pkl
Traceback (most recent call last):
  File "/home/halsted/Python/weather_updated/scripts/train_edge_classifier.py", line 1731, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/halsted/Python/weather_updated/scripts/train_edge_classifier.py", line 1552, in main
    df_combined = load_combined_data(args.city)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/halsted/Python/weather_updated/scripts/train_edge_classifier.py", line 458, in load_combined_data
    df_train = pd.read_parquet(train_path)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/halsted/Python/weather_updated/.venv/lib/python3.11/site-packages/pandas/io/parquet.py", line 669, in read_parquet
    return impl.read(
           ^^^^^^^^^^
  File "/home/halsted/Python/weather_updated/.venv/lib/python3.11/site-packages/pandas/io/parquet.py", line 265, in read
    pa_table = self.api.parquet.read_table(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/halsted/Python/weather_updated/.venv/lib/python3.11/site-packages/pyarrow/parquet/core.py", line 1844, in read_table
    dataset = ParquetDataset(
              ^^^^^^^^^^^^^^^
  File "/home/halsted/Python/weather_updated/.venv/lib/python3.11/site-packages/pyarrow/parquet/core.py", line 1413, in __init__
    [fragment], schema=schema or fragment.physical_schema,
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
  File "pyarrow/_dataset.pyx", line 1477, in pyarrow._dataset.Fragment.physical_schema.__get__
  File "pyarrow/error.pxi", line 155, in pyarrow.lib.pyarrow_internal_check_status
  File "pyarrow/error.pxi", line 92, in pyarrow.lib.check_status
pyarrow.lib.ArrowInvalid: Could not open Parquet input source '<Buffer>': Parquet file size is 0 bytes
(.venv) (base) halsted@halsted:~/Python/weather_updated$ 



GitHub Workflow - Yes, This Works
On this computer: Commit and push your current state
git add -A && git commit -m "sync for fast computer" && git push
On fast computer: Clone to LOCAL drive
cd /local/fast/drive
git clone <your-repo-url> weather_updated
cd weather_updated
pip install -e .  # or use existing venv
Run everything locally on fast computer (10-50x faster I/O)
When done on fast computer: Push results
git add models/saved/ models/raw_data/  # or whatever changed
git commit -m "Austin + Chicago models built"
git push
Back on this computer: Pull the results
git pull
Note: Large parquet files (50-100MB) in models/saved/ will bloat git history. Consider:
Adding them to .gitignore and using rsync/scp instead
Or using Git LFS for large files
Or just accepting the bloat since it's a private research repo
The local SSD vs network mount difference for parallel parquet processing is huge - probably 10-20x faster


(.venv) (base) halsted@halsted:~/slow_weather_updated$ PYTHONPATH=. python scripts/build_dataset_from_parquets.py --city austin
18:29:42 [INFO] __main__: ============================================================
18:29:42 [INFO] __main__: BUILD DATASET FROM PARQUETS: AUSTIN
18:29:42 [INFO] __main__: ============================================================
18:29:42 [INFO] __main__: Loading raw data from models/raw_data/austin...
18:29:42 [INFO] __main__:   Observations: 307,518 rows
18:29:42 [INFO] __main__:   City observations: 307,263 rows
18:29:42 [INFO] __main__:   Settlements: 1,068 rows
18:29:42 [INFO] __main__:   Daily forecasts: 7,693 rows
18:29:42 [INFO] __main__:   Hourly forecasts: 184,638 rows
18:29:42 [INFO] __main__:   NOAA guidance: 2,132 rows
18:29:43 [INFO] __main__:   Candles: 11,245,787 rows
18:29:43 [INFO] __main__:   Pre-grouping data by date...
18:29:44 [INFO] __main__:   Pre-grouped: 1068 obs days, 941 candle days
18:29:44 [INFO] __main__: 
Found 1068 days with settlements
18:29:44 [INFO] __main__: Date range: 2023-01-01 to 2025-12-03
18:29:44 [INFO] __main__: 
Days to build: 1068 (2023-01-01 to 2025-12-03)
18:29:44 [INFO] __main__: 
--- Pre-computing rolling stats ---
18:29:44 [INFO] __main__: Pre-computing obs_t15 stats for all days...
18:29:44 [INFO] __main__:   Pre-computed obs_t15 stats: 1058/1068 days have valid stats
18:29:44 [INFO] __main__: 
--- Building full dataset ---
18:29:44 [INFO] __main__: Building All days (1068 days)...
All days:   1%|█▎                                                                                                                                       | 10/1068 [01:46<3:17:00, 11.17s/it]