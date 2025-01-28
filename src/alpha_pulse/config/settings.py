"""
Configuration settings for AlphaPulse.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExchangeConfig:
    """Exchange configuration settings."""
    api_key: str = ""
    api_secret: str = ""
    id: str = "mock"


@dataclass
class Settings:
    """Global configuration settings."""
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)


# Global settings instance
settings = Settings()