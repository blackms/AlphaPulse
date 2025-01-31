"""
Abstract base class defining the interface for broker implementations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None  # Required for limit orders
    stop_price: Optional[float] = None  # Required for stop orders
    order_id: Optional[str] = None  # Set by broker upon placement
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    timestamp: datetime = datetime.now()


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: datetime = datetime.now()


class BrokerInterface(ABC):
    """Abstract base class for broker implementations."""

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """Place a new order with the broker.
        
        Args:
            order: Order object containing order details
            
        Returns:
            Updated Order object with broker-assigned ID and initial status
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order.
        
        Args:
            order_id: Unique identifier for the order to cancel
            
        Returns:
            True if cancellation was successful, False otherwise
        """
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get current state of an order.
        
        Args:
            order_id: Unique identifier for the order
            
        Returns:
            Order object if found, None otherwise
        """
        pass

    @abstractmethod
    def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter orders
            
        Returns:
            List of active Order objects
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            Position object if position exists, None otherwise
        """
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions.
        
        Returns:
            Dictionary mapping symbols to Position objects
        """
        pass

    @abstractmethod
    def get_account_balance(self) -> float:
        """Get current account cash balance.
        
        Returns:
            Current cash balance
        """
        pass

    @abstractmethod
    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions).
        
        Returns:
            Total portfolio value
        """
        pass

    @abstractmethod
    def update_market_data(self, symbol: str, current_price: float) -> None:
        """Update market data for position tracking and PnL calculation.
        
        Args:
            symbol: The trading symbol
            current_price: Current market price
        """
        pass