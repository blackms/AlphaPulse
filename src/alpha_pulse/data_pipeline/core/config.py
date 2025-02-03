"""
Configuration module for the data pipeline.

This module provides configuration classes and validation for all data pipeline
components, following the Single Responsibility Principle.
"""
from dataclasses import dataclass, field
from typing import Dict

from alpha_pulse.data_pipeline.core.models import DataPipelineError


class ConfigurationError(DataPipelineError):
    """Error raised for invalid configurations."""
    pass


@dataclass
class StorageConfig:
    """Configuration for data storage operations."""
    batch_size: int = 500
    max_connections: int = 10
    timeout: float = 30.0  # seconds
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.batch_size < 1:
            raise ConfigurationError("batch_size must be positive")
        if self.max_connections < 1:
            raise ConfigurationError("max_connections must be positive")
        if self.timeout <= 0:
            raise ConfigurationError("timeout must be positive")


@dataclass
class DataFetchConfig:
    """Configuration for data fetching operations."""
    batch_size: int = 1000
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    timeout: float = 30.0  # seconds
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.batch_size < 1:
            raise ConfigurationError("batch_size must be positive")
        if self.max_retries < 0:
            raise ConfigurationError("max_retries cannot be negative")
        if self.retry_delay <= 0:
            raise ConfigurationError("retry_delay must be positive")
        if self.timeout <= 0:
            raise ConfigurationError("timeout must be positive")


@dataclass
class MarketDataConfig:
    """Configuration for market data operations."""
    update_interval: float = 1.0  # seconds
    max_symbols: int = 100
    cache_duration: int = 60  # seconds
    timeframe_durations: Dict[str, int] = field(default_factory=lambda: {
        "1m": 60,      # 1 minute in seconds
        "5m": 300,     # 5 minutes
        "15m": 900,    # 15 minutes
        "1h": 3600,    # 1 hour
        "4h": 14400,   # 4 hours
        "1d": 86400,   # 1 day
    })
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.update_interval <= 0:
            raise ConfigurationError("update_interval must be positive")
        if self.max_symbols < 1:
            raise ConfigurationError("max_symbols must be positive")
        if self.cache_duration < 0:
            raise ConfigurationError("cache_duration cannot be negative")
        if not self.timeframe_durations:
            raise ConfigurationError("timeframe_durations cannot be empty")
        
        for timeframe, duration in self.timeframe_durations.items():
            if duration <= 0:
                raise ConfigurationError(
                    f"Duration for timeframe {timeframe} must be positive"
                )

    def validate_timeframe(self, timeframe: str) -> None:
        """
        Validate timeframe string.

        Args:
            timeframe: Timeframe string to validate

        Raises:
            ConfigurationError: If timeframe is not supported
        """
        if timeframe not in self.timeframe_durations:
            raise ConfigurationError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported timeframes: {list(self.timeframe_durations.keys())}"
            )


@dataclass
class DataPipelineConfig:
    """Main configuration for the data pipeline."""
    storage: StorageConfig = field(default_factory=StorageConfig)
    fetcher: DataFetchConfig = field(default_factory=DataFetchConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)