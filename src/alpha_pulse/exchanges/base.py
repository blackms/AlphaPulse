"""
Base types for exchange operations.
"""
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Optional


@dataclass
class Balance:
    """Account balance information."""
    total: Decimal
    available: Decimal
    locked: Decimal

    @property
    def free(self) -> Decimal:
        """Free balance for trading."""
        return self.available


@dataclass
class OHLCV:
    """OHLCV candle data."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal