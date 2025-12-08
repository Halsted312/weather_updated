"""Pytest fixtures and configuration."""

import os
import sys

import pytest
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()


@pytest.fixture(scope="session")
def db_engine():
    """Get database engine for testing."""
    from src.db.connection import get_engine, close_engine

    engine = get_engine()
    yield engine
    close_engine()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Get database session for testing."""
    from src.db.connection import get_session_factory

    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="session")
def settings():
    """Get application settings."""
    from src.config.settings import get_settings

    return get_settings()


# ============================================================================
# Denver Oct 2025 Smoke Test Fixtures
# ============================================================================

@pytest.fixture
def project_root():
    """Return project root directory."""
    from pathlib import Path
    return Path(__file__).parent.parent


@pytest.fixture
def models_dir(project_root):
    """Return models directory."""
    return project_root / "models"


# Cities available for testing
CITIES = ["denver", "los_angeles", "miami", "philadelphia", "chicago", "austin"]


@pytest.fixture(params=CITIES)
def all_cities(request):
    """Parametrize tests across all cities."""
    return request.param
