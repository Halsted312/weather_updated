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