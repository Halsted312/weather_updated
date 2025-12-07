-- Validation Queries for Austin 15-min Forecast Data
-- Run with: psql -U kalshi -d kalshi_weather -f scripts/validate_austin_15min_data.sql

\echo '================================================================================'
\echo 'AUSTIN 15-MIN FORECAST VALIDATION'
\echo '================================================================================'
\echo ''

-- Query 1: Coverage - Check all dates present
\echo '1. COVERAGE CHECK: Verify all dates have data'
\echo '--------------------------------------------------------------------------------'
WITH expected_dates AS (
    SELECT generate_series(
        '2025-05-01'::date,
        '2025-12-01'::date,
        '1 day'::interval
    )::date AS event_date
),
actual_coverage AS (
    SELECT DISTINCT
        DATE(datetime_local) as event_date,
        forecast_basis_date
    FROM wx.vc_minute_weather vm
    JOIN wx.vc_location vl ON vm.vc_location_id = vl.id
    WHERE vl.city_code = 'AUS'
      AND vl.location_type = 'city'
      AND data_type = 'historical_forecast'
)
SELECT
    COUNT(*) FILTER (WHERE ac.event_date IS NULL) as missing_dates,
    COUNT(*) FILTER (WHERE ac.event_date IS NOT NULL) as present_dates,
    COUNT(*) as total_expected
FROM expected_dates ed
LEFT JOIN actual_coverage ac ON ed.event_date = ac.event_date;

\echo ''

-- Query 2: Minute-level granularity check
\echo '2. GRANULARITY CHECK: Verify 15-min intervals (~96 points/day)'
\echo '--------------------------------------------------------------------------------'
SELECT
    COUNT(*) as days_checked,
    MIN(point_count) as min_points,
    MAX(point_count) as max_points,
    ROUND(AVG(point_count), 1) as avg_points,
    COUNT(*) FILTER (WHERE point_count < 90) as sparse_days,
    COUNT(*) FILTER (WHERE point_count >= 90 AND point_count <= 100) as ok_days
FROM (
    SELECT
        DATE(datetime_local) as event_date,
        COUNT(*) as point_count
    FROM wx.vc_minute_weather vm
    JOIN wx.vc_location vl ON vm.vc_location_id = vl.id
    WHERE vl.city_code = 'AUS'
      AND vl.location_type = 'city'
      AND data_type = 'historical_forecast'
      AND DATE(datetime_local) BETWEEN '2025-05-01' AND '2025-12-01'
    GROUP BY event_date
) counts;

\echo ''

-- Query 3: Timezone validation
\echo '3. TIMEZONE CHECK: Verify America/Chicago and correct offsets'
\echo '--------------------------------------------------------------------------------'
SELECT
    timezone,
    tzoffset_minutes,
    COUNT(*) as records,
    CASE
        WHEN tzoffset_minutes NOT IN (-360, -300) THEN '⚠️  WRONG OFFSET'
        ELSE '✅ OK'
    END as tz_check
FROM wx.vc_minute_weather vm
JOIN wx.vc_location vl ON vm.vc_location_id = vl.id
WHERE vl.city_code = 'AUS'
  AND data_type = 'historical_forecast'
  AND DATE(datetime_local) BETWEEN '2025-05-01' AND '2025-12-01'
GROUP BY timezone, tzoffset_minutes
ORDER BY records DESC;

\echo ''

-- Query 4: 1900 datetime bug check (CRITICAL)
\echo '4. DATETIME BUG CHECK: Detect 1900 dates (should be ZERO)'
\echo '--------------------------------------------------------------------------------'
SELECT
    COUNT(*) as bug_count,
    MIN(datetime_local) as earliest_bad,
    MAX(datetime_local) as latest_bad
FROM wx.vc_minute_weather vm
JOIN wx.vc_location vl ON vm.vc_location_id = vl.id
WHERE vl.city_code = 'AUS'
  AND data_type = 'historical_forecast'
  AND EXTRACT(YEAR FROM datetime_local) < 2000;

\echo ''

-- Query 5: Basis date semantics (should be T-1)
\echo '5. BASIS DATE CHECK: Verify forecast_basis_date = event_date - 1'
\echo '--------------------------------------------------------------------------------'
WITH lead_check AS (
    SELECT
        DATE(datetime_local) as event_date,
        forecast_basis_date,
        (DATE(datetime_local) - forecast_basis_date) as lead_days
    FROM wx.vc_minute_weather vm
    JOIN wx.vc_location vl ON vm.vc_location_id = vl.id
    WHERE vl.city_code = 'AUS'
      AND vl.location_type = 'city'
      AND data_type = 'historical_forecast'
      AND DATE(datetime_local) BETWEEN '2025-05-01' AND '2025-12-01'
)
SELECT
    lead_days,
    COUNT(DISTINCT event_date) as days_count,
    CASE
        WHEN lead_days != 1 THEN '⚠️  WRONG LEAD'
        ELSE '✅ OK'
    END as check
FROM lead_check
GROUP BY lead_days
ORDER BY lead_days;

\echo ''

-- Query 6: Data freshness check
\echo '6. FRESHNESS CHECK: Verify forecast_basis_date < event_date'
\echo '--------------------------------------------------------------------------------'
SELECT
    COUNT(*) as violations,
    CASE
        WHEN COUNT(*) = 0 THEN '✅ OK - All forecasts made before event'
        ELSE '⚠️  ERROR - Some forecasts after event!'
    END as check
FROM wx.vc_minute_weather vm
JOIN wx.vc_location vl ON vm.vc_location_id = vl.id
WHERE vl.city_code = 'AUS'
  AND data_type = 'historical_forecast'
  AND forecast_basis_date >= DATE(datetime_local);

\echo ''

-- Query 7: NULL temperature check
\echo '7. NULL CHECK: Verify no missing temperatures'
\echo '--------------------------------------------------------------------------------'
SELECT
    COUNT(*) as total_points,
    SUM(CASE WHEN temp_f IS NULL THEN 1 ELSE 0 END) as null_temp_count,
    ROUND(100.0 * SUM(CASE WHEN temp_f IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as null_pct,
    CASE
        WHEN SUM(CASE WHEN temp_f IS NULL THEN 1 ELSE 0 END) = 0 THEN '✅ OK'
        ELSE '⚠️  HAS NULLS'
    END as check
FROM wx.vc_minute_weather vm
JOIN wx.vc_location vl ON vm.vc_location_id = vl.id
WHERE vl.city_code = 'AUS'
  AND vl.location_type = 'city'
  AND data_type = 'historical_forecast'
  AND DATE(datetime_local) BETWEEN '2025-05-01' AND '2025-12-01';

\echo ''

-- Query 8: Temperature sanity check
\echo '8. SANITY CHECK: Verify temperature range (Austin: 30-110°F typical)'
\echo '--------------------------------------------------------------------------------'
SELECT
    MIN(temp_f) as min_temp,
    MAX(temp_f) as max_temp,
    ROUND(AVG(temp_f), 1) as avg_temp,
    CASE
        WHEN MIN(temp_f) < 0 OR MAX(temp_f) > 120 THEN '⚠️  OUT OF RANGE'
        ELSE '✅ OK'
    END as sanity_check
FROM wx.vc_minute_weather vm
JOIN wx.vc_location vl ON vm.vc_location_id = vl.id
WHERE vl.city_code = 'AUS'
  AND data_type = 'historical_forecast'
  AND DATE(datetime_local) BETWEEN '2025-05-01' AND '2025-12-01';

\echo ''
\echo '================================================================================'
\echo 'VALIDATION COMPLETE'
\echo '================================================================================'
