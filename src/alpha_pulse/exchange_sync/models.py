"""
Data models for the exchange synchronization module.

These models represent the core data structures used throughout the module,
providing a clean and consistent interface between components.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List


@dataclass
class PortfolioItem:
    """
    Portfolio item representing a single asset holding.
    
    This class encapsulates all data related to an asset in a portfolio,
    including quantities and price information.
    """
    asset: str
    quantity: float
    current_price: Optional[float] = None
    avg_entry_price: Optional[float] = None
    updated_at: Optional[datetime] = None
    
    @property
    def value(self) -> Optional[float]:
        """Calculate current value of the portfolio item."""
        if self.current_price is None:
            return None
        return self.quantity * self.current_price
    
    @property
    def profit_loss(self) -> Optional[float]:
        """Calculate profit/loss amount."""
        if self.current_price is None or self.avg_entry_price is None:
            return None
        return self.quantity * (self.current_price - self.avg_entry_price)
    
    @property
    def profit_loss_percentage(self) -> Optional[float]:
        """Calculate profit/loss percentage."""
        if self.current_price is None or self.avg_entry_price is None or self.avg_entry_price == 0:
            return None
        return ((self.current_price / self.avg_entry_price) - 1) * 100


@dataclass
class OrderData:
    """
    Order data representing a trade order.
    
    This class encapsulates all data related to an exchange order,
    including order details and execution information.
    """
    order_id: str
    asset: str
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    status: str
    timestamp: datetime
    exchange_id: str
    order_type: Optional[str] = None  # 'market', 'limit', etc.
    filled_quantity: Optional[float] = None
    fee: Optional[float] = None
    fee_currency: Optional[str] = None
    
    @property
    def is_completed(self) -> bool:
        """Check if the order is completed."""
        return self.status.lower() in ['closed', 'filled', 'completed']
    
    @property
    def is_buy(self) -> bool:
        """Check if this is a buy order."""
        return self.side.lower() == 'buy'
    
    @property
    def is_sell(self) -> bool:
        """Check if this is a sell order."""
        return self.side.lower() == 'sell'


@dataclass
class SyncResult:
    """
    Result of a synchronization operation.
    
    This class provides a standardized way to track the results of 
    synchronization operations, including success/failure status and statistics.
    """
    success: bool
    items_processed: int = 0
    items_synced: int = 0
    errors: List[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.errors is None:
            self.errors = []
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate the duration of the sync operation in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def add_error(self, error: str) -> None:
        """Add an error message to the result."""
        self.errors.append(error)
        self.success = False
    
    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred during synchronization."""
        return len(self.errors) > 0