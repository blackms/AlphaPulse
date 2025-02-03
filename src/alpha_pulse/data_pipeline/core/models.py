"""
Core models for the data pipeline.

This module defines the fundamental data models and configurations used
throughout the data pipeline.
"""
from dataclasses import dataclass
from datetime import datetime


class DataPipelineError(Exception):
    """Base exception for data pipeline errors."""
    pass


@dataclass
class StorageConfig:
    """Configuration for data storage operations."""
    batch_size: int = 500
    max_connections: int = 10
    timeout: float = 30.0  # seconds


@dataclass
class DataFetchConfig:
    """Configuration for data fetching operations."""
    batch_size: int = 1000
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    timeout: float = 30.0  # seconds


@dataclass
class MarketDataConfig:
    """Configuration for market data operations."""
    update_interval: float = 1.0  # seconds
    max_symbols: int = 100
    cache_duration: int = 60  # seconds


# Common timeframe durations
TIMEFRAME_DURATIONS = {
    "1m": 60,          # 1 minute in seconds
    "5m": 300,         # 5 minutes
    "15m": 900,        # 15 minutes
    "1h": 3600,        # 1 hour
    "4h": 14400,       # 4 hours
    "1d": 86400,       # 1 day
}


def validate_timeframe(timeframe: str) -> None:
    """
    Validate timeframe string.

    Args:
        timeframe: Timeframe string to validate

    Raises:
        ValueError: If timeframe is not supported
    """
    if timeframe not in TIMEFRAME_DURATIONS:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            f"Supported timeframes: {list(TIMEFRAME_DURATIONS.keys())}"
        )