"""
Interface definitions for AlphaPulse data pipeline components.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

from alpha_pulse.exchanges import OHLCV, ExchangeType


class IDataStorage(ABC):
    """Abstract base class for data storage implementations."""
    
    @abstractmethod
    def save_historical_data(
        self,
        exchange_type: ExchangeType,
        symbol: str,
        timeframe: str,
        data: Dict
    ) -> None:
        """Save historical market data.
        
        Args:
            exchange_type: Type of exchange
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            data: Market data to save
        """
        pass
    
    @abstractmethod
    def get_historical_data(
        self,
        exchange_type: ExchangeType,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict:
        """Retrieve historical market data.
        
        Args:
            exchange_type: Type of exchange
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            
        Returns:
            Dict containing historical market data
        """
        pass
    
    @abstractmethod
    def save_ohlcv(self, data: List[OHLCV]) -> None:
        """Save OHLCV data.
        
        Args:
            data: List of OHLCV records to save
        """
        pass
    
    @abstractmethod
    def get_latest_ohlcv(
        self,
        exchange_type: ExchangeType,
        symbol: str
    ) -> Optional[datetime]:
        """Get timestamp of latest OHLCV record.
        
        Args:
            exchange_type: Type of exchange
            symbol: Trading pair symbol
            
        Returns:
            Timestamp of latest record or None if no records exist
        """
        pass