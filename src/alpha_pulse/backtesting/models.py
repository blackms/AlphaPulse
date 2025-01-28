"""
Data models for the backtesting framework.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Position:
    """Represents a trading position."""
    entry_price: float
    entry_time: datetime
    size: float
    pnl: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None