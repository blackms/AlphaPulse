"""
Core interfaces for hedging components.
"""
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, List, Protocol, runtime_checkable

from .types import GridLevel, GridMetrics, MarketState, PositionState


@runtime_checkable
class MarketDataProvider(Protocol):
    """Protocol for market data providers."""
    
    async def get_current_price(self, symbol: str) -> Decimal:
        """Get current market price."""
        ...
        
    async def get_funding_rates(self, symbol: str) -> List[Decimal]:
        """Get funding rates."""
        ...
        
    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        limit: int
    ) -> List[Dict]:
        """Get historical market data."""
        ...


@runtime_checkable
class RiskManager(Protocol):
    """Protocol for risk management."""
    
    def calculate_position_size(
        self,
        price: Decimal,
        volatility: Decimal,
        available_margin: Decimal
    ) -> Decimal:
        """Calculate safe position size."""
        ...
        
    def validate_risk_limits(
        self,
        position: PositionState,
        market: MarketState
    ) -> bool:
        """Check if position meets risk limits."""
        ...
        
    def calculate_stop_loss(
        self,
        position: PositionState,
        market: MarketState
    ) -> Decimal:
        """Calculate dynamic stop loss level."""
        ...


class OrderManager(ABC):
    """Abstract base class for order management."""
    
    @abstractmethod
    async def place_grid_orders(
        self,
        levels: List[GridLevel],
        market: MarketState
    ) -> Dict[str, GridLevel]:
        """Place grid orders."""
        pass
        
    @abstractmethod
    async def cancel_orders(self, order_ids: List[str]) -> None:
        """Cancel orders."""
        pass
        
    @abstractmethod
    async def update_risk_orders(
        self,
        position: PositionState,
        market: MarketState
    ) -> None:
        """Update stop loss and take profit orders."""
        pass


class GridCalculator(ABC):
    """Abstract base class for grid calculations."""
    
    @abstractmethod
    def calculate_grid_levels(
        self,
        market: MarketState,
        position: PositionState
    ) -> List[GridLevel]:
        """Calculate grid levels."""
        pass
        
    @abstractmethod
    def adjust_for_funding(
        self,
        levels: List[GridLevel],
        funding_rate: Decimal
    ) -> List[GridLevel]:
        """Adjust grid levels based on funding rate."""
        pass
        
    @abstractmethod
    def validate_levels(
        self,
        levels: List[GridLevel],
        market: MarketState
    ) -> List[GridLevel]:
        """Validate and filter grid levels."""
        pass


class MetricsCalculator(ABC):
    """Abstract base class for metrics calculation."""
    
    @abstractmethod
    def update_metrics(
        self,
        current: GridMetrics,
        position: PositionState,
        market: MarketState
    ) -> GridMetrics:
        """Update performance metrics."""
        pass
        
    @abstractmethod
    def calculate_pnl(
        self,
        position: PositionState,
        market: MarketState
    ) -> Dict[str, Decimal]:
        """Calculate PnL metrics."""
        pass


class TechnicalAnalyzer(ABC):
    """Abstract base class for technical analysis."""
    
    @abstractmethod
    async def get_support_resistance(
        self,
        market: MarketState,
        lookback: int = 20
    ) -> List[Dict]:
        """Get support and resistance levels."""
        pass
        
    @abstractmethod
    def calculate_volatility(
        self,
        prices: List[Decimal],
        window: int = 20
    ) -> Decimal:
        """Calculate price volatility."""
        pass
        
    @abstractmethod
    def detect_trend(
        self,
        prices: List[Decimal],
        window: int = 20
    ) -> str:
        """Detect price trend."""
        pass


class StateManager(ABC):
    """Abstract base class for state management."""
    
    @abstractmethod
    def update_position(
        self,
        current: PositionState,
        **kwargs
    ) -> PositionState:
        """Update position state."""
        pass
        
    @abstractmethod
    def update_metrics(
        self,
        current: GridMetrics,
        **kwargs
    ) -> GridMetrics:
        """Update metrics state."""
        pass
        
    @abstractmethod
    def get_status(self) -> Dict:
        """Get current strategy status."""
        pass