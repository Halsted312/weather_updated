-- Migration: Add sim.market_open_log table
-- Date: 2025-11-27
-- Purpose: Track market open events detected via WebSocket

-- Create the table to log market open events
CREATE TABLE IF NOT EXISTS sim.market_open_log (
    id SERIAL PRIMARY KEY,
    series_ticker VARCHAR(20) NOT NULL,
    event_ticker VARCHAR(50) NOT NULL,
    market_ticker VARCHAR(50) NOT NULL,
    city VARCHAR(20),
    event_date DATE,
    opened_at TIMESTAMPTZ NOT NULL,
    raw_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_market_open_log_event_ticker
    ON sim.market_open_log(event_ticker);

CREATE INDEX IF NOT EXISTS idx_market_open_log_opened_at
    ON sim.market_open_log(opened_at);

CREATE INDEX IF NOT EXISTS idx_market_open_log_city_date
    ON sim.market_open_log(city, event_date);

-- Add comments
COMMENT ON TABLE sim.market_open_log IS
    'Logs market open events detected via WebSocket market_lifecycle channel';

COMMENT ON COLUMN sim.market_open_log.series_ticker IS
    'Kalshi series ticker (e.g., KXHIGHCHI)';

COMMENT ON COLUMN sim.market_open_log.event_ticker IS
    'Kalshi event ticker (e.g., KXHIGHCHI-25NOV28)';

COMMENT ON COLUMN sim.market_open_log.market_ticker IS
    'Kalshi market ticker (e.g., KXHIGHCHI-25NOV28-B35.5)';

COMMENT ON COLUMN sim.market_open_log.city IS
    'City ID extracted from series ticker';

COMMENT ON COLUMN sim.market_open_log.event_date IS
    'Event date extracted from event ticker';

COMMENT ON COLUMN sim.market_open_log.opened_at IS
    'Timestamp when the market open event was received';

COMMENT ON COLUMN sim.market_open_log.raw_data IS
    'Full JSON payload from WebSocket message';
