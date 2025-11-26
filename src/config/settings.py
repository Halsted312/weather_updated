"""
Application settings loaded from environment variables.
Uses pydantic-settings for validation and type coercion.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql://kalshi:kalshi@localhost:5433/kalshi_weather"
    db_host: str = "localhost"
    db_port: int = 5433
    db_name: str = "kalshi_weather"
    db_user: str = "kalshi"
    db_password: str = "kalshi"

    # Kalshi API
    kalshi_api_key: str
    kalshi_private_key_path: str = "./weather.pem"
    kalshi_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"

    # Visual Crossing
    vc_api_key: str
    vc_base_url: str = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    wx_minute_interval: int = 5

    # NOAA (optional)
    noaa_api_key: Optional[str] = None

    # Application
    log_level: str = "INFO"
    data_dir: str = "./data"
    max_workers: int = 4

    @property
    def private_key_path(self) -> Path:
        """Return Path object for private key."""
        return Path(self.kalshi_private_key_path)

    def get_database_url(self) -> str:
        """Get the database URL, constructing from parts if needed."""
        if self.database_url:
            return self.database_url
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
