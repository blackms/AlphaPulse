"""
Interface definitions for AlphaPulse data pipeline components.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

from .models import OHLCV


class IExchange(ABC):
    """Abstract base class for exchange implementations."""

    @abstractmethod
    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict:
        """Fetch historical market data."""
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        pass

    @abstractmethod
    def get_available_pairs(self) -> List[str]:
        """Get list of available trading pairs."""
        pass


class IExchangeFactory(ABC):
    """Abstract factory for creating exchange instances."""

    @abstractmethod
    def create_exchange(self, exchange_id: str) -> IExchange:
        """Create an exchange instance."""
        pass


class IDataStorage(ABC):
    """Abstract base class for data storage implementations."""

    @abstractmethod
    def save_historical_data(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        data: Dict
    ) -> None:
        """Save historical market data."""
        pass

    @abstractmethod
    def get_historical_data(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict:
        """Retrieve historical market data."""
        pass

    @abstractmethod
    def save_ohlcv(self, data: List[OHLCV]) -> None:
        """Save OHLCV data."""
        pass

    @abstractmethod
    def get_latest_ohlcv(
        self,
        exchange_id: str,
        symbol: str
    ) -> Optional[datetime]:
        """Get timestamp of latest OHLCV record."""
        pass


# Alias for backward compatibility
ExchangeInterface = IExchange