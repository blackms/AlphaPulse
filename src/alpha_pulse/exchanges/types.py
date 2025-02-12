"""
Exchange type definitions.
"""
from enum import Enum


class ExchangeType(str, Enum):
    """Supported exchange types."""
    BINANCE = "binance"
    BYBIT = "bybit"