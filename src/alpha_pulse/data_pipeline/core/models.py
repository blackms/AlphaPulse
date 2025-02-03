"""
Core domain models for the data pipeline.

This module defines the fundamental data models used throughout the data pipeline,
focusing on domain concepts and value objects.
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

from alpha_pulse.exchanges import ExchangeType


@dataclass(frozen=True)
class TimeRange:
    """
    Value object representing a time range.
    
    Immutable to prevent accidental modifications and ensure thread safety.
    """
    start: datetime
    end: datetime

    def __post_init__(self):
        """Validate time range on creation."""
        if self.start > self.end:
            raise ValueError("Start time must be before end time")

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp is within range."""
        return self.start <= timestamp <= self.end

    def overlaps(self, other: 'TimeRange') -> bool:
        """Check if this range overlaps with another."""
        return (
            self.contains(other.start) or
            self.contains(other.end) or
            other.contains(self.start) or
            other.contains(self.end)
        )


@dataclass(frozen=True)
class MarketSymbol:
    """
    Value object representing a market symbol.
    
    Immutable to prevent accidental modifications and ensure thread safety.
    """
    base: str
    quote: str

    def __post_init__(self):
        """Validate symbol components."""
        if not self.base or not self.quote:
            raise ValueError("Base and quote currencies are required")

    @classmethod
    def from_string(cls, symbol: str) -> 'MarketSymbol':
        """Create from string representation."""
        try:
            base, quote = symbol.split('/')
            return cls(base=base, quote=quote)
        except ValueError:
            raise ValueError(
                f"Invalid symbol format: {symbol}. "
                f"Must be in format 'BASE/QUOTE' (e.g., 'BTC/USDT')"
            )

    def __str__(self) -> str:
        """Get string representation."""
        return f"{self.base}/{self.quote}"


@dataclass(frozen=True)
class PriceUpdate:
    """
    Value object representing a price update.
    
    Immutable to prevent accidental modifications and ensure thread safety.
    """
    exchange: ExchangeType
    symbol: MarketSymbol
    price: Decimal
    timestamp: datetime
    volume: Optional[Decimal] = None

    def __post_init__(self):
        """Validate price update."""
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.volume is not None and self.volume < 0:
            raise ValueError("Volume cannot be negative")


@dataclass(frozen=True)
class MarketDepth:
    """
    Value object representing market depth.
    
    Immutable to prevent accidental modifications and ensure thread safety.
    """
    exchange: ExchangeType
    symbol: MarketSymbol
    timestamp: datetime
    bids: list[tuple[Decimal, Decimal]]  # List of (price, amount) tuples
    asks: list[tuple[Decimal, Decimal]]  # List of (price, amount) tuples

    def __post_init__(self):
        """Validate market depth."""
        if not self.bids and not self.asks:
            raise ValueError("At least one bid or ask is required")
        
        # Validate bid/ask prices and amounts
        for price, amount in self.bids + self.asks:
            if price <= 0:
                raise ValueError("Prices must be positive")
            if amount <= 0:
                raise ValueError("Amounts must be positive")
        
        # Validate bid/ask ordering
        if len(self.bids) > 1:
            for i in range(1, len(self.bids)):
                if self.bids[i][0] >= self.bids[i-1][0]:
                    raise ValueError("Bids must be in descending price order")
        
        if len(self.asks) > 1:
            for i in range(1, len(self.asks)):
                if self.asks[i][0] <= self.asks[i-1][0]:
                    raise ValueError("Asks must be in ascending price order")