"""
Core interfaces for the data pipeline.

This module defines the fundamental interfaces that all data pipeline components
must implement, following Interface Segregation Principle.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from alpha_pulse.exchanges import OHLCV, ExchangeType


class IOHLCVStorage(ABC):
    """Interface for OHLCV data storage operations."""
    
    @abstractmethod
    def save_ohlcv(self, data: List[OHLCV]) -> None:
        """Save OHLCV data."""
        pass
    
    @abstractmethod
    def get_latest_ohlcv(
        self,
        exchange_type: ExchangeType,
        symbol: str
    ) -> Optional[datetime]:
        """Get timestamp of latest OHLCV record."""
        pass


class IHistoricalDataStorage(ABC):
    """Interface for historical data storage operations."""
    
    @abstractmethod
    def get_historical_data(
        self,
        exchange_type: ExchangeType,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[OHLCV]:
        """Retrieve historical market data."""
        pass


class IDataStorage(IOHLCVStorage, IHistoricalDataStorage):
    """Combined interface for all data storage operations."""
    pass


class IDataFetcher(ABC):
    """Interface for components that fetch market data."""
    
    @abstractmethod
    async def fetch_ohlcv(
        self,
        exchange_type: ExchangeType,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
        testnet: bool = False
    ) -> List[OHLCV]:
        """Fetch OHLCV data from a source."""
        pass


class IMarketDataProvider(ABC):
    """Interface for real-time market data providers."""
    
    @abstractmethod
    async def start(self, symbols: List[str]) -> None:
        """Start market data updates."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop market data updates."""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close provider connection."""
        pass


class IHistoricalDataManager(ABC):
    """Interface for historical data management."""
    
    @abstractmethod
    async def ensure_data_available(
        self,
        exchange_type: ExchangeType,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> bool:
        """Ensure historical data is available for the specified period."""
        pass
    
    @abstractmethod
    def get_historical_data(
        self,
        exchange_type: ExchangeType,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[OHLCV]:
        """Get historical data for the specified period."""
        pass


class IRealTimeDataManager(ABC):
    """Interface for real-time data management."""
    
    @abstractmethod
    async def start(self, symbols: List[str]) -> None:
        """Start real-time data updates."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop real-time data updates."""
        pass
    
    @abstractmethod
    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest cached price for a symbol."""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price directly from source."""
        pass