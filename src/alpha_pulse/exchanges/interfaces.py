"""
Exchange interfaces defining core functionality contracts.
"""
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, List, Optional, Any
from datetime import datetime


from .base import Balance, OHLCV


class MarketDataProvider(ABC):
    """Interface for market data operations."""
    
    @abstractmethod
    async def get_ticker_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for symbol."""
        pass
    
    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """Fetch OHLCV candles."""
        pass


class TradingOperations(ABC):
    """Interface for trading operations."""
    
    @abstractmethod
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        price: Optional[Decimal] = None,
        order_type: str = "market"
    ) -> Dict[str, Any]:
        """Execute trade order."""
        pass
    
    @abstractmethod
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get order history."""
        pass


class AccountOperations(ABC):
    """Interface for account operations."""
    
    @abstractmethod
    async def get_balances(self) -> Dict[str, Balance]:
        """Get account balances."""
        pass
    
    @abstractmethod
    async def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value in base currency."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions."""
        pass
    
    @abstractmethod
    async def get_trading_fees(self) -> Dict[str, Decimal]:
        """Get trading fees."""
        pass


class ExchangeConnection(ABC):
    """Interface for exchange connection lifecycle."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize exchange connection."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close exchange connection."""
        pass


class BaseExchange(MarketDataProvider, TradingOperations, AccountOperations, ExchangeConnection):
    """
    Base adapter interface combining all exchange operations.
    
    This interface provides a complete exchange implementation contract.
    Concrete implementations should inherit from this interface
    and implement all required methods.
    """
    
    @abstractmethod
    async def get_average_entry_price(self, symbol: str) -> Optional[Decimal]:
        """Calculate average entry price for symbol."""
        pass


class ExchangeConfiguration:
    """Exchange configuration data class."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        options: Optional[Dict[str, Any]] = None
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.options = options or {}
        


class ExchangeError(Exception):
    """Base class for exchange-related errors."""
    pass


class ConnectionError(ExchangeError):
    """Raised when exchange connection fails."""
    pass


class OrderError(ExchangeError):
    """Raised when order execution fails."""
    pass


class MarketDataError(ExchangeError):
    """Raised when market data operations fail."""
    pass


class ConfigurationError(ExchangeError):
    """Raised when exchange configuration is invalid."""
    pass