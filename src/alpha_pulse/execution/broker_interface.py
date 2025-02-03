"""
Trading broker interface definitions.
"""
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order information."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    timestamp: Optional[float] = None


@dataclass
class OrderResult:
    """Order execution result."""
    success: bool
    order_id: Optional[str] = None
    filled_quantity: Optional[float] = None
    filled_price: Optional[float] = None
    error: Optional[str] = None


@dataclass
class Position:
    """Position information."""
    symbol: str
    quantity: float
    avg_entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: Optional[float] = None


class BrokerInterface:
    """Abstract base class for trading brokers."""
    
    def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value."""
        raise NotImplementedError
        
    def get_available_margin(self) -> Decimal:
        """Get available margin for trading."""
        raise NotImplementedError
        
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        raise NotImplementedError
        
    def get_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        raise NotImplementedError
        
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        raise NotImplementedError
        
    async def place_order(self, order: Order) -> OrderResult:
        """Place a new order."""
        raise NotImplementedError
        
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        raise NotImplementedError
        
    def update_market_data(self, symbol: str, price: float) -> None:
        """Update market data."""
        raise NotImplementedError
        
    async def initialize_spot_position(
        self,
        symbol: str,
        quantity: float,
        price: float
    ) -> None:
        """Initialize spot position."""
        raise NotImplementedError