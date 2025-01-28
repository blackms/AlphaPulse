"""
Configuration settings for AlphaPulse.
"""
from dataclasses import dataclass, field


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
class Settings:
    """Global configuration settings."""
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    DATABASE_URL: str = field(default_factory=lambda: "sqlite:///data.db")


# Global settings instance
settings = Settings()