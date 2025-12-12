.PHONY: help install dev db-up db-down db-reset db-shell migrate test lint format clean

# Default target
help:
	@echo "Kalshi Weather Pipeline - Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install production dependencies"
	@echo "  make dev          - Install development dependencies"
	@echo ""
	@echo "Database:"
	@echo "  make db-up        - Start TimescaleDB container"
	@echo "  make db-down      - Stop TimescaleDB container"
	@echo "  make db-reset     - Reset database (destroy & recreate)"
	@echo "  make db-shell     - Open psql shell"
	@echo "  make migrate      - Run Alembic migrations"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run all tests"
	@echo "  make test-db      - Test database connection"
	@echo "  make test-kalshi  - Test Kalshi client"
	@echo "  make test-vc      - Test Visual Crossing client"
	@echo ""
	@echo "Quality:"
	@echo "  make lint         - Run linter (ruff)"
	@echo "  make format       - Format code (black)"
	@echo ""
	@echo "Ingestion:"
	@echo "  make ingest-markets     - Backfill Kalshi markets"
	@echo "  make ingest-candles     - Backfill Kalshi candles"
	@echo "  make ingest-vc-obs      - Backfill Visual Crossing observations"
	@echo "  make ingest-vc-forecasts - Backfill VC historical forecasts"
	@echo "  make ingest-settlement  - Backfill NWS settlements"

# Setup
install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# Database commands
db-up:
	docker-compose up -d db
	@echo "Waiting for TimescaleDB to be ready..."
	@sleep 5
	@docker-compose exec db pg_isready -U kalshi -d kalshi_weather

db-down:
	docker-compose down

db-reset:
	docker-compose down -v
	docker-compose up -d db
	@echo "Waiting for TimescaleDB to be ready..."
	@sleep 5
	@$(MAKE) migrate

db-shell:
	docker-compose exec db psql -U kalshi -d kalshi_weather

db-logs:
	docker-compose logs -f db

# Migrations
migrate:
	alembic upgrade head

migrate-new:
	@read -p "Migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

# Testing
test:
	pytest tests/ -v

test-db:
	pytest tests/test_db_connection.py -v

test-kalshi:
	pytest tests/test_kalshi_client.py -v

test-vc:
	pytest tests/test_weather_vc.py -v

test-fees:
	pytest tests/test_fees.py -v

# Code quality
lint:
	ruff check src/ scripts/ tests/

format:
	black src/ scripts/ tests/

# Ingestion scripts
ingest-markets:
	python scripts/ingestion/kalshi/backfill_kalshi_markets.py

ingest-candles:
	python scripts/ingestion/kalshi/backfill_kalshi_candles.py

ingest-vc-obs:
	python scripts/ingestion/vc/ingest_vc_obs_backfill.py

ingest-vc-forecasts:
	python scripts/ingestion/vc/ingest_vc_historical_forecast_parallel.py

ingest-settlement:
	python scripts/ingestion/settlement/ingest_settlement_multi.py

# Clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
