"""
Configuration settings for AlphaPulse.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExchangeConfig:
    """Exchange configuration settings."""
    api_key: str = ""
    api_secret: str = ""
    id: str = "mock"


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = "sqlite:///data.db"
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
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    DATABASE_URL: str = field(default_factory=lambda: "sqlite:///data.db")


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