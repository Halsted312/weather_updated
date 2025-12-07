-- Global check: Are VC historical_forecast minutes ALWAYS NULL across all cities?

\echo '================================================================================'
\echo 'GLOBAL VC MINUTE HISTORICAL FORECAST NULL CHECK'
\echo '================================================================================'
\echo ''

-- Check if any VC historical_forecast minutes have non-NULL temp_f
SELECT
    vl.city_code,
    COUNT(*) AS total_points,
    SUM(CASE WHEN vm.temp_f IS NOT NULL THEN 1 ELSE 0 END) AS non_null_temp_points,
    MIN(vm.datetime_local) AS min_dt,
    MAX(vm.datetime_local) AS max_dt
FROM wx.vc_minute_weather vm
JOIN wx.vc_location vl ON vm.vc_location_id = vl.id
WHERE vm.data_type = 'historical_forecast'
GROUP BY vl.city_code
ORDER BY vl.city_code;

\echo ''
\echo 'Checking for any non-NULL temps (should be empty if all NULL):'
\echo '--------------------------------------------------------------------------------'

-- Sanity peek at raw JSON where temp is not null, if any exist
SELECT
    vl.city_code,
    vm.datetime_local,
    vm.temp_f,
    vm.raw_json->>'temp' AS raw_temp
FROM wx.vc_minute_weather vm
JOIN wx.vc_location vl ON vm.vc_location_id = vl.id
WHERE vm.data_type = 'historical_forecast'
  AND vm.temp_f IS NOT NULL
ORDER BY vm.datetime_local
LIMIT 20;

\echo ''
\echo '================================================================================'
\echo 'CHECK COMPLETE'
\echo '================================================================================'
