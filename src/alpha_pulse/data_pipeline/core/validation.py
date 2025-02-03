"""
Validation module for the data pipeline.

This module provides validation functions and utilities for data pipeline
components, following the Single Responsibility Principle.
"""
from datetime import datetime
from typing import List, Optional

from alpha_pulse.exchanges import OHLCV, ExchangeType
from alpha_pulse.data_pipeline.core.errors import ValidationError


def validate_exchange_type(exchange_type: ExchangeType) -> None:
    """
    Validate exchange type.

    Args:
        exchange_type: Exchange type to validate

    Raises:
        ValidationError: If exchange type is invalid
    """
    if not isinstance(exchange_type, ExchangeType):
        raise ValidationError(
            f"Invalid exchange type: {exchange_type}. "
            f"Must be an instance of ExchangeType enum."
        )


def validate_symbol(symbol: str) -> None:
    """
    Validate trading symbol.

    Args:
        symbol: Symbol to validate

    Raises:
        ValidationError: If symbol is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValidationError("Symbol must be a non-empty string")
    
    if '/' not in symbol:
        raise ValidationError(
            f"Invalid symbol format: {symbol}. "
            f"Must be in format 'BASE/QUOTE' (e.g., 'BTC/USDT')"
        )


def validate_timeframe(timeframe: str, valid_timeframes: List[str]) -> None:
    """
    Validate timeframe string.

    Args:
        timeframe: Timeframe to validate
        valid_timeframes: List of valid timeframe strings

    Raises:
        ValidationError: If timeframe is invalid
    """
    if not timeframe or not isinstance(timeframe, str):
        raise ValidationError("Timeframe must be a non-empty string")
    
    if timeframe not in valid_timeframes:
        raise ValidationError(
            f"Invalid timeframe: {timeframe}. "
            f"Must be one of: {', '.join(valid_timeframes)}"
        )


def validate_time_range(
    start_time: Optional[datetime],
    end_time: Optional[datetime]
) -> None:
    """
    Validate time range.

    Args:
        start_time: Start time
        end_time: End time

    Raises:
        ValidationError: If time range is invalid
    """
    if start_time and end_time and start_time > end_time:
        raise ValidationError("Start time must be before end time")


def validate_ohlcv(ohlcv: OHLCV) -> None:
    """
    Validate OHLCV data.

    Args:
        ohlcv: OHLCV data to validate

    Raises:
        ValidationError: If OHLCV data is invalid
    """
    if not ohlcv.timestamp:
        raise ValidationError("OHLCV timestamp is required")
    
    if ohlcv.high < ohlcv.low:
        raise ValidationError("High price cannot be lower than low price")
    
    if ohlcv.open < 0 or ohlcv.high < 0 or ohlcv.low < 0 or ohlcv.close < 0:
        raise ValidationError("Prices cannot be negative")
    
    if ohlcv.volume < 0:
        raise ValidationError("Volume cannot be negative")


def validate_ohlcv_list(data: List[OHLCV]) -> None:
    """
    Validate list of OHLCV data.

    Args:
        data: List of OHLCV data to validate

    Raises:
        ValidationError: If any OHLCV data is invalid
    """
    if not data:
        raise ValidationError("OHLCV data list cannot be empty")
    
    # Validate each OHLCV record
    for ohlcv in data:
        validate_ohlcv(ohlcv)
    
    # Validate timestamps are ordered
    for i in range(1, len(data)):
        if data[i].timestamp <= data[i-1].timestamp:
            raise ValidationError("OHLCV timestamps must be strictly increasing")


def validate_batch_size(batch_size: int) -> None:
    """
    Validate batch size.

    Args:
        batch_size: Batch size to validate

    Raises:
        ValidationError: If batch size is invalid
    """
    if not isinstance(batch_size, int):
        raise ValidationError("Batch size must be an integer")
    
    if batch_size < 1:
        raise ValidationError("Batch size must be positive")


def validate_limit(limit: Optional[int]) -> None:
    """
    Validate limit parameter.

    Args:
        limit: Limit to validate

    Raises:
        ValidationError: If limit is invalid
    """
    if limit is not None:
        if not isinstance(limit, int):
            raise ValidationError("Limit must be an integer")
        
        if limit < 1:
            raise ValidationError("Limit must be positive")