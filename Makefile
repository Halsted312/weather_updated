.PHONY: help init init-dev db-up db-down db-migrate db-reset install test lint format clean
.DEFAULT_GOAL := help

# Detect docker compose CLI
DOCKER_COMPOSE_BIN := $(shell command -v docker-compose 2>/dev/null)
ifeq ($(DOCKER_COMPOSE_BIN),)
DOCKER_COMPOSE := docker compose
else
DOCKER_COMPOSE := docker-compose
endif

# Directories
PROJECT_DIR := $(shell pwd)
DATA_DIR := $(PROJECT_DIR)/data

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(CYAN)Kalshi Weather Data Pipeline - Makefile Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

init: ## Initialize project (install deps + start Docker)
	@echo "$(CYAN)Initializing project...$(NC)"
	@$(MAKE) install
	@$(MAKE) db-up
	@echo "$(GREEN)✓ Project initialized!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Copy .env.example to .env and add your KALSHI_API_KEY"
	@echo "  2. Run 'make db-migrate' to create database schema"
	@echo "  3. Run 'make ingest-chicago-demo' to test data fetching"

init-dev: ## Initialize development environment with all tools
	@echo "$(CYAN)Setting up development environment...$(NC)"
	pip install -e ".[dev]"
	@$(MAKE) db-up --profile tools
	@echo "$(GREEN)✓ Dev environment ready!$(NC)"
	@echo "pgAdmin available at http://localhost:5151 (admin@kalshi.local / admin)"

install: ## Install Python dependencies
	@echo "$(CYAN)Installing dependencies...$(NC)"
	pip install -e .
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

# Docker & Database Commands
db-up: ## Start PostgreSQL container
	@echo "$(CYAN)Starting PostgreSQL...$(NC)"
	$(DOCKER_COMPOSE) up -d postgres
	@echo "$(GREEN)✓ PostgreSQL started on localhost:5444$(NC)"

db-down: ## Stop PostgreSQL container
	@echo "$(CYAN)Stopping PostgreSQL...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ PostgreSQL stopped$(NC)"

db-migrate: ## Run database migrations
	@echo "$(CYAN)Running database migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)✓ Migrations complete$(NC)"

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "$(YELLOW)⚠ WARNING: This will destroy all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
		if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(MAKE) db-down; \
		docker volume rm kalshi_weather_mom_postgres_data 2>/dev/null || true; \
		$(MAKE) db-up; \
		$(MAKE) db-migrate; \
		echo "$(GREEN)✓ Database reset complete$(NC)"; \
	fi

# Data Ingestion Commands
ingest-chicago-demo: ## Fetch 7-day sample data for Chicago
	@echo "$(CYAN)Fetching Chicago demo data (7 days)...$(NC)"
	python scripts/discover_chicago.py --days 7 --output $(DATA_DIR)/raw/chicago
	@echo "$(GREEN)✓ Demo data saved to $(DATA_DIR)/raw/chicago$(NC)"

ingest-chicago-100d: ## Fetch 100 days of Chicago data
	@echo "$(CYAN)Fetching Chicago data (100 days)...$(NC)"
	python scripts/discover_chicago.py --days 100 --output $(DATA_DIR)/raw/chicago
	@echo "$(GREEN)✓ Data saved to $(DATA_DIR)/raw/chicago$(NC)"

ingest-all-cities: ## Fetch 100 days for all cities
	@echo "$(CYAN)Fetching data for all cities (100 days)...$(NC)"
	python scripts/discover_all_cities.py --days 100 --output $(DATA_DIR)/raw
	@echo "$(GREEN)✓ All cities data fetched$(NC)"

load-to-db: ## Load fetched data into database
	@echo "$(CYAN)Loading data to database...$(NC)"
	python ingest/load_to_db.py --data-dir $(DATA_DIR)/raw
	@echo "$(GREEN)✓ Data loaded to database$(NC)"

# Visual Crossing Weather Commands
backfill-wx-demo: ## Backfill 3 days Visual Crossing data (Chicago only)
	@echo "$(CYAN)Backfilling Visual Crossing demo data (3 days, Chicago)...$(NC)"
	python ingest/backfill_visualcrossing.py \
		--start-date $$(date -d '3 days ago' +%Y-%m-%d) \
		--end-date $$(date +%Y-%m-%d) \
		--cities chicago
	@echo "$(GREEN)✓ Demo weather data backfilled$(NC)"

backfill-wx: ## Backfill 100 days Visual Crossing data (all cities)
	@echo "$(CYAN)Backfilling Visual Crossing data (100 days, all cities)...$(NC)"
	python ingest/backfill_visualcrossing.py \
		--start-date $$(date -d '100 days ago' +%Y-%m-%d) \
		--end-date $$(date +%Y-%m-%d) \
		--cities all
	@echo "$(GREEN)✓ Weather data backfilled for all cities$(NC)"

backfill-wx-fast: ## Backfill 100 days Visual Crossing data (skip per-city refresh)
	@echo "$(CYAN)Backfilling Visual Crossing data (fast mode)...$(NC)"
	python ingest/backfill_visualcrossing.py \
		--start-date $$(date -d '100 days ago' +%Y-%m-%d) \
		--end-date $$(date +%Y-%m-%d) \
		--cities all \
		--skip-refresh
	@echo "$(GREEN)✓ Weather data backfilled (run poll-wx-live to refresh 1-min grid)$(NC)"

poll-wx-live: ## Start Visual Crossing real-time poller (background)
	@echo "$(CYAN)Starting Visual Crossing real-time poller...$(NC)"
	@nohup python ingest/poll_visualcrossing.py > /tmp/wx_poller.log 2>&1 &
	@echo $$! > /tmp/wx_poller.pid
	@echo "$(GREEN)✓ Poller started (PID: $$(cat /tmp/wx_poller.pid))$(NC)"
	@echo "  Log: /tmp/wx_poller.log"
	@echo "  Stop with: kill $$(cat /tmp/wx_poller.pid)"

poll-wx-stop: ## Stop Visual Crossing poller
	@if [ -f /tmp/wx_poller.pid ]; then \
		kill $$(cat /tmp/wx_poller.pid) 2>/dev/null || true; \
		rm /tmp/wx_poller.pid; \
		echo "$(GREEN)✓ Poller stopped$(NC)"; \
	else \
		echo "$(YELLOW)No poller PID found$(NC)"; \
	fi

# Development Commands
lint: ## Run linters (ruff + mypy)
	@echo "$(CYAN)Running linters...$(NC)"
	ruff check .
	mypy kalshi ingest weather db scripts
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black
	@echo "$(CYAN)Formatting code...$(NC)"
	black .
	@echo "$(GREEN)✓ Code formatted$(NC)"

clean: ## Clean generated files and caches
	@echo "$(CYAN)Cleaning...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(NC)"

# Quick workflow shortcuts
quick-start: install db-up db-migrate ingest-chicago-demo load-to-db ## Quick start: setup + fetch demo data

full-pipeline: ingest-all-cities load-to-db ## Full pipeline: ingest all data
