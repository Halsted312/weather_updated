-- Austin Data Source Validation (2023-01-01 → Latest)
-- Read-only health check before training

\echo '================================================================================'
\echo 'AUSTIN DATA SOURCE VALIDATION'
\echo '================================================================================'
\echo ''

-- Step 1: Check NOAA guidance coverage by model
\echo '1. NOAA GUIDANCE COVERAGE (NBM, HRRR, NDFD)'
\echo '--------------------------------------------------------------------------------'
SELECT
    model,
    COUNT(DISTINCT target_date) as unique_days,
    MIN(target_date) as first_date,
    MAX(target_date) as last_date
FROM wx.weather_more_apis_guidance
WHERE city_id = 'austin'
  AND target_date >= '2023-01-01'
GROUP BY model
ORDER BY model;

\echo ''
\echo '2. NOAA GUIDANCE VALUE SANITY (Temp ranges per model)'
\echo '--------------------------------------------------------------------------------'
SELECT
    model,
    COUNT(*) as total_rows,
    ROUND(MIN(peak_window_max_f)::numeric, 1) as min_temp_f,
    ROUND(MAX(peak_window_max_f)::numeric, 1) as max_temp_f,
    ROUND(AVG(peak_window_max_f)::numeric, 1) as avg_temp_f,
    COUNT(*) FILTER (WHERE peak_window_max_f IS NULL) as null_count,
    COUNT(*) FILTER (WHERE peak_window_max_f < 0 OR peak_window_max_f > 120) as out_of_range
FROM wx.weather_more_apis_guidance
WHERE city_id = 'austin'
  AND target_date >= '2023-01-01'
GROUP BY model
ORDER BY model;

\echo ''
\echo '3. MISSING MODEL DAYS (Days without NBM or HRRR)'
\echo '--------------------------------------------------------------------------------'
WITH date_series AS (
    SELECT generate_series(
        '2023-01-01'::date,
        (SELECT MAX(target_date) FROM wx.weather_more_apis_guidance WHERE city_id = 'austin'),
        '1 day'::interval
    )::date AS target_date
),
model_coverage AS (
    SELECT
        ds.target_date,
        MAX(CASE WHEN g.model = 'nbm' THEN 1 ELSE 0 END) as has_nbm,
        MAX(CASE WHEN g.model = 'hrrr' THEN 1 ELSE 0 END) as has_hrrr,
        MAX(CASE WHEN g.model = 'ndfd' THEN 1 ELSE 0 END) as has_ndfd
    FROM date_series ds
    LEFT JOIN wx.weather_more_apis_guidance g
        ON ds.target_date = g.target_date
        AND g.city_id = 'austin'
    GROUP BY ds.target_date
)
SELECT
    COUNT(*) FILTER (WHERE has_nbm = 0) as missing_nbm_days,
    COUNT(*) FILTER (WHERE has_hrrr = 0) as missing_hrrr_days,
    COUNT(*) FILTER (WHERE has_ndfd = 0) as missing_ndfd_days,
    COUNT(*) as total_days
FROM model_coverage;

\echo ''
\echo '4. VC OBSERVATIONS COVERAGE (wx.vc_minute_weather, actual_obs)'
\echo '--------------------------------------------------------------------------------'
SELECT
    COUNT(DISTINCT DATE(datetime_local)) as unique_obs_days,
    MIN(DATE(datetime_local)) as first_obs_date,
    MAX(DATE(datetime_local)) as last_obs_date,
    COUNT(*) as total_obs_points,
    COUNT(*) FILTER (WHERE temp_f IS NULL) as null_temp_count
FROM wx.vc_minute_weather vm
JOIN wx.vc_location vl ON vm.vc_location_id = vl.id
WHERE vl.city_code = 'AUS'
  AND vl.location_type = 'station'
  AND vm.data_type = 'actual_obs'
  AND DATE(datetime_local) >= '2023-01-01';

\echo ''
\echo '5. VC DAILY FORECASTS COVERAGE (wx.vc_forecast_daily)'
\echo '--------------------------------------------------------------------------------'
SELECT
    COUNT(DISTINCT target_date) as unique_fcst_days,
    MIN(target_date) as first_fcst_date,
    MAX(target_date) as last_fcst_date,
    COUNT(*) as total_fcst_rows,
    COUNT(*) FILTER (WHERE tempmax_f IS NULL) as null_tempmax_count
FROM wx.vc_forecast_daily
JOIN wx.vc_location vl ON vc_forecast_daily.vc_location_id = vl.id
WHERE vl.city_code = 'AUS'
  AND vl.location_type = 'city'
  AND data_type = 'historical_forecast'
  AND target_date >= '2023-01-01';

\echo ''
\echo '6. VC HOURLY FORECASTS COVERAGE (wx.vc_forecast_hourly)'
\echo '--------------------------------------------------------------------------------'
SELECT
    COUNT(DISTINCT DATE(target_datetime_local)) as unique_fcst_days,
    MIN(DATE(target_datetime_local)) as first_fcst_date,
    MAX(DATE(target_datetime_local)) as last_fcst_date,
    COUNT(*) as total_hourly_fcst_rows,
    COUNT(*) FILTER (WHERE temp_f IS NULL) as null_temp_count
FROM wx.vc_forecast_hourly
JOIN wx.vc_location vl ON vc_forecast_hourly.vc_location_id = vl.id
WHERE vl.city_code = 'AUS'
  AND vl.location_type = 'city'
  AND data_type = 'historical_forecast'
  AND DATE(target_datetime_local) >= '2023-01-01';

\echo ''
\echo '7. SETTLEMENT DATA COVERAGE (wx.settlement)'
\echo '--------------------------------------------------------------------------------'
SELECT
    COUNT(DISTINCT date_local) as unique_settle_days,
    MIN(date_local) as first_settle_date,
    MAX(date_local) as last_settle_date,
    COUNT(*) FILTER (WHERE tmax_final IS NULL) as null_tmax_count
FROM wx.settlement
WHERE city = 'austin'
  AND date_local >= '2023-01-01';

\echo ''
\echo '================================================================================'
\echo 'VALIDATION COMPLETE'
\echo '================================================================================'
