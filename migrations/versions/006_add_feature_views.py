"""Add feature schema with midnight_forecast_path and intraday_decision_points views.

Revision ID: 006
Revises: 005
Create Date: 2025-11-26

Creates feature views for the midnight heuristic trading strategy:
- feature.midnight_forecast_path: One row per (city, event_date) with 3-day forecast path
- feature.intraday_decision_points: Decision times at midnight, pre_high_2h, pre_high_1h
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create feature schema
    op.execute("CREATE SCHEMA IF NOT EXISTS feature;")

    # Create midnight_forecast_path view
    # This joins forecast snapshots to get 3-day path + predicted high time
    op.execute("""
        CREATE OR REPLACE VIEW feature.midnight_forecast_path AS
        WITH daily_forecasts AS (
            -- Get daily forecasts for lead_days 0, 1, 2
            SELECT
                city,
                target_date AS event_date,
                basis_date,
                lead_days,
                tempmax_fcst_f
            FROM wx.forecast_snapshot
            WHERE lead_days <= 2
              AND basis_date = target_date  -- Only midnight forecast (lead_days=0 row)
        ),
        pivoted_forecasts AS (
            -- Pivot to get t0, t1, t2 in same row
            SELECT
                f0.city,
                f0.event_date,
                f0.basis_date,
                f0.tempmax_fcst_f AS tempmax_t0,
                f1.tempmax_fcst_f AS tempmax_t1,
                f2.tempmax_fcst_f AS tempmax_t2
            FROM (SELECT * FROM daily_forecasts WHERE lead_days = 0) f0
            LEFT JOIN (
                SELECT city, basis_date, tempmax_fcst_f
                FROM wx.forecast_snapshot
                WHERE lead_days = 1
            ) f1 ON f0.city = f1.city AND f0.basis_date = f1.basis_date
            LEFT JOIN (
                SELECT city, basis_date, tempmax_fcst_f
                FROM wx.forecast_snapshot
                WHERE lead_days = 2
            ) f2 ON f0.city = f2.city AND f0.basis_date = f2.basis_date
        ),
        hourly_max AS (
            -- Find predicted high hour from hourly forecast curve for today
            SELECT DISTINCT ON (city, basis_date)
                city,
                basis_date,
                EXTRACT(HOUR FROM target_hour_local) +
                    EXTRACT(MINUTE FROM target_hour_local) / 60.0 AS predicted_high_hour_of_day,
                temp_fcst_f AS predicted_high_temp_f
            FROM wx.forecast_snapshot_hourly
            WHERE lead_hours < 24  -- Only today's hours (0-23)
            ORDER BY city, basis_date, temp_fcst_f DESC, target_hour_local ASC
        ),
        yesterday_high_hour AS (
            -- Get yesterday's actual high hour from minute observations
            SELECT
                loc_id,
                ts_utc::date AS obs_date,
                EXTRACT(HOUR FROM ts_utc AT TIME ZONE 'UTC') +
                    EXTRACT(MINUTE FROM ts_utc AT TIME ZONE 'UTC') / 60.0 AS actual_high_hour_of_day
            FROM (
                SELECT DISTINCT ON (loc_id, ts_utc::date)
                    loc_id, ts_utc, temp_f
                FROM wx.minute_obs
                ORDER BY loc_id, ts_utc::date, temp_f DESC, ts_utc ASC
            ) sub
        ),
        tomorrow_high_hour AS (
            -- Get tomorrow's predicted high hour from hourly forecast
            SELECT DISTINCT ON (city, basis_date)
                city,
                basis_date,
                EXTRACT(HOUR FROM target_hour_local) +
                    EXTRACT(MINUTE FROM target_hour_local) / 60.0 AS tomorrow_high_hour_of_day
            FROM wx.forecast_snapshot_hourly
            WHERE lead_hours >= 24 AND lead_hours < 48  -- Tomorrow's hours
            ORDER BY city, basis_date, temp_fcst_f DESC, target_hour_local ASC
        ),
        city_stations AS (
            -- Map cities to station IDs
            SELECT 'chicago' AS city, 'KMDW' AS loc_id UNION ALL
            SELECT 'austin', 'KAUS' UNION ALL
            SELECT 'denver', 'KDEN' UNION ALL
            SELECT 'los_angeles', 'KLAX' UNION ALL
            SELECT 'miami', 'KMIA' UNION ALL
            SELECT 'philadelphia', 'KPHL'
        )
        SELECT
            pf.city,
            pf.event_date,
            pf.basis_date,
            -- 3-day forecast path
            pf.tempmax_t0,
            pf.tempmax_t1,
            pf.tempmax_t2,
            -- Trend features
            (COALESCE(pf.tempmax_t2, pf.tempmax_t1, pf.tempmax_t0) - pf.tempmax_t0) / 2.0 AS trend_3d,
            pf.tempmax_t1 - pf.tempmax_t0 AS delta_t1_vs_t0,
            pf.tempmax_t2 - pf.tempmax_t0 AS delta_t2_vs_t0,
            -- Predicted high time (float hour-of-day)
            hm.predicted_high_hour_of_day,
            hm.predicted_high_temp_f,
            -- Weighted high time: 0.2*yesterday + 0.7*today + 0.1*tomorrow
            0.2 * COALESCE(yh.actual_high_hour_of_day, hm.predicted_high_hour_of_day) +
            0.7 * hm.predicted_high_hour_of_day +
            0.1 * COALESCE(th.tomorrow_high_hour_of_day, hm.predicted_high_hour_of_day)
                AS weighted_high_hour_of_day,
            -- Labels from settlement
            s.tmax_final,
            -- Actual high hour from observations (earliest if ties)
            (
                SELECT EXTRACT(HOUR FROM ts_utc AT TIME ZONE 'UTC') +
                       EXTRACT(MINUTE FROM ts_utc AT TIME ZONE 'UTC') / 60.0
                FROM wx.minute_obs mo
                WHERE mo.loc_id = cs.loc_id
                  AND mo.ts_utc::date = pf.event_date
                ORDER BY mo.temp_f DESC, mo.ts_utc ASC
                LIMIT 1
            ) AS actual_high_hour_of_day,
            s.tmax_final - pf.tempmax_t0 AS delta_midnight
        FROM pivoted_forecasts pf
        LEFT JOIN hourly_max hm ON pf.city = hm.city AND pf.basis_date = hm.basis_date
        LEFT JOIN city_stations cs ON pf.city = cs.city
        LEFT JOIN yesterday_high_hour yh ON cs.loc_id = yh.loc_id AND yh.obs_date = pf.event_date - INTERVAL '1 day'
        LEFT JOIN tomorrow_high_hour th ON pf.city = th.city AND pf.basis_date = th.basis_date
        LEFT JOIN wx.settlement s ON pf.city = s.city AND pf.event_date = s.date_local;
    """)

    # Create intraday_decision_points view
    op.execute("""
        CREATE OR REPLACE VIEW feature.intraday_decision_points AS
        WITH base AS (
            SELECT * FROM feature.midnight_forecast_path
        ),
        decision_times AS (
            -- Generate decision times: midnight, pre_high_2h, pre_high_1h
            SELECT
                city,
                event_date,
                'midnight' AS decision_type,
                0.0 AS decision_hour_of_day
            FROM base
            UNION ALL
            SELECT
                city,
                event_date,
                'pre_high_2h' AS decision_type,
                GREATEST(weighted_high_hour_of_day - 2.0, 0.0) AS decision_hour_of_day
            FROM base
            WHERE weighted_high_hour_of_day IS NOT NULL
            UNION ALL
            SELECT
                city,
                event_date,
                'pre_high_1h' AS decision_type,
                GREATEST(weighted_high_hour_of_day - 1.0, 0.0) AS decision_hour_of_day
            FROM base
            WHERE weighted_high_hour_of_day IS NOT NULL
        ),
        city_timezones AS (
            SELECT 'chicago' AS city, 'America/Chicago' AS tz UNION ALL
            SELECT 'austin', 'America/Chicago' UNION ALL
            SELECT 'denver', 'America/Denver' UNION ALL
            SELECT 'los_angeles', 'America/Los_Angeles' UNION ALL
            SELECT 'miami', 'America/New_York' UNION ALL
            SELECT 'philadelphia', 'America/New_York'
        ),
        city_stations AS (
            SELECT 'chicago' AS city, 'KMDW' AS loc_id UNION ALL
            SELECT 'austin', 'KAUS' UNION ALL
            SELECT 'denver', 'KDEN' UNION ALL
            SELECT 'los_angeles', 'KLAX' UNION ALL
            SELECT 'miami', 'KMIA' UNION ALL
            SELECT 'philadelphia', 'KPHL'
        )
        SELECT
            dt.city,
            dt.event_date,
            dt.decision_type,
            dt.decision_hour_of_day,
            -- Convert hour-of-day to UTC timestamp
            (dt.event_date + (dt.decision_hour_of_day || ' hours')::interval)
                AT TIME ZONE tz.tz AT TIME ZONE 'UTC' AS decision_time_utc,
            -- Forecast state
            b.tempmax_t0,
            -- Hourly forecast temp at decision time (approximate to nearest hour)
            (
                SELECT temp_fcst_f
                FROM wx.forecast_snapshot_hourly fh
                WHERE fh.city = dt.city
                  AND fh.basis_date = dt.event_date
                  AND EXTRACT(HOUR FROM fh.target_hour_local) = FLOOR(dt.decision_hour_of_day)
                LIMIT 1
            ) AS temp_fcst_at_decision,
            -- Observed state at decision time
            (
                SELECT temp_f
                FROM wx.minute_obs mo
                WHERE mo.loc_id = cs.loc_id
                  AND mo.ts_utc >= (dt.event_date + (dt.decision_hour_of_day || ' hours')::interval)
                        AT TIME ZONE tz.tz AT TIME ZONE 'UTC' - INTERVAL '5 minutes'
                  AND mo.ts_utc < (dt.event_date + (dt.decision_hour_of_day || ' hours')::interval)
                        AT TIME ZONE tz.tz AT TIME ZONE 'UTC' + INTERVAL '5 minutes'
                ORDER BY mo.ts_utc
                LIMIT 1
            ) AS temp_obs_at_decision,
            -- Max observed temp up to decision time
            (
                SELECT MAX(temp_f)
                FROM wx.minute_obs mo
                WHERE mo.loc_id = cs.loc_id
                  AND mo.ts_utc::date = dt.event_date
                  AND mo.ts_utc <= (dt.event_date + (dt.decision_hour_of_day || ' hours')::interval)
                        AT TIME ZONE tz.tz AT TIME ZONE 'UTC'
            ) AS temp_max_so_far,
            -- Labels
            b.tmax_final,
            -- Winning bracket ticker (join with markets where result='yes')
            (
                SELECT ticker
                FROM kalshi.markets km
                WHERE km.city = dt.city
                  AND km.event_date = dt.event_date
                  AND km.result = 'yes'
                LIMIT 1
            ) AS bin_that_settled
        FROM decision_times dt
        JOIN base b ON dt.city = b.city AND dt.event_date = b.event_date
        JOIN city_timezones tz ON dt.city = tz.city
        JOIN city_stations cs ON dt.city = cs.city;
    """)

    # Create index on the midnight_forecast_path for fast lookups
    # (Views don't have indexes, but we can create a materialized view later if needed)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS feature.intraday_decision_points;")
    op.execute("DROP VIEW IF EXISTS feature.midnight_forecast_path;")
    op.execute("DROP SCHEMA IF EXISTS feature;")
