"""
Error definitions for the data pipeline.

This module provides a hierarchy of custom exceptions for the data pipeline,
following the Open/Closed Principle for error handling.
"""
from typing import Optional


class DataPipelineError(Exception):
    """Base exception for all data pipeline errors."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        """
        Initialize error.

        Args:
            message: Error message
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        """Get string representation including cause if available."""
        if self.cause:
            return f"{super().__str__()} | Caused by: {str(self.cause)}"
        return super().__str__()


class ConfigurationError(DataPipelineError):
    """Error raised for invalid configurations."""
    pass


class StorageError(DataPipelineError):
    """Base class for storage-related errors."""
    pass


class ConnectionError(StorageError):
    """Error raised when database connection fails."""
    pass


class QueryError(StorageError):
    """Error raised when database query fails."""
    pass


class DataFetchError(DataPipelineError):
    """Base class for data fetching errors."""
    pass


class NetworkError(DataFetchError):
    """Error raised when network operations fail."""
    pass


class RateLimitError(DataFetchError):
    """Error raised when rate limits are exceeded."""
    pass


class ValidationError(DataPipelineError):
    """Error raised when data validation fails."""
    pass


class MarketDataError(DataPipelineError):
    """Base class for market data errors."""
    pass


class ProviderError(MarketDataError):
    """Error raised by market data providers."""
    pass


class HistoricalDataError(MarketDataError):
    """Error raised by historical data operations."""
    pass


class RealTimeDataError(MarketDataError):
    """Error raised by real-time data operations."""
    pass


def wrap_error(
    error: Exception,
    error_class: type[DataPipelineError],
    message: Optional[str] = None
) -> DataPipelineError:
    """
    Wrap an exception in a data pipeline error.

    Args:
        error: Original exception
        error_class: Type of error to create
        message: Optional custom message

    Returns:
        Wrapped error
    """
    if isinstance(error, DataPipelineError):
        return error
    
    return error_class(
        message or str(error),
        cause=error
    )