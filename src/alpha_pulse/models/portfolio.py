"""
Portfolio data structures used across services.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Position:
    """Represents a single portfolio position."""

    symbol: str
    quantity: float
    current_price: float
    average_cost: float
    position_type: str = "long"
    sector: Optional[str] = None
    strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Portfolio:
    """Aggregated portfolio view."""

    portfolio_id: str
    name: str
    total_value: float
    cash_balance: float
    positions: Dict[str, Position] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
