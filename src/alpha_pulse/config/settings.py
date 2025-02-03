"""
Configuration settings for AlphaPulse.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .exchanges import ExchangeConfig, get_exchange_config


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = "postgresql+psycopg2://username:password@localhost:5432/timescaledb"
    echo: bool = False


@dataclass
class PathConfig:
    """Path configuration settings."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.parent)
    trained_models: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.parent / "trained_models")
    feature_cache: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.parent / "feature_cache")
    logs: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.parent / "logs")


@dataclass
class Settings:
    """Global configuration settings."""
    exchange: ExchangeConfig = field(default_factory=lambda: get_exchange_config("binance") or ExchangeConfig(
        id="binance",
        name="Binance",
        description="Default exchange configuration"
    ))
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    DATABASE_URL: str = field(default_factory=lambda: "sqlite:///data.db")

    def update_exchange(self, exchange_id: str, api_key: str = "", api_secret: str = "") -> None:
        """
        Update exchange configuration.
        
        Args:
            exchange_id: Exchange identifier
            api_key: Optional API key
            api_secret: Optional API secret
        """
        if config := get_exchange_config(exchange_id):
            config.api_key = api_key
            config.api_secret = api_secret
            self.exchange = config


# Global settings instance
settings = Settings()

# Export commonly used paths
BASE_DIR = settings.paths.base_dir
TRAINED_MODELS_DIR = settings.paths.trained_models
FEATURE_CACHE_DIR = settings.paths.feature_cache
LOGS_DIR = settings.paths.logs

# Create directories if they don't exist
for directory in [TRAINED_MODELS_DIR, FEATURE_CACHE_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)