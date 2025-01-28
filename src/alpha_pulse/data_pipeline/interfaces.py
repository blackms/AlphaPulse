"""
Interface definitions for AlphaPulse data pipeline components.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional


class ExchangeInterface(ABC):
    """Abstract base class for exchange implementations."""

    @abstractmethod
    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict:
        """
        Fetch historical market data from the exchange.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            timeframe: Candle timeframe (e.g., "1m", "1h", "1d")
            start_time: Start time for data fetch
            end_time: End time for data fetch

        Returns:
            Dict containing OHLCV data
        """
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")

        Returns:
            Current price
        """
        pass

    @abstractmethod
    def get_available_pairs(self) -> List[str]:
        """
        Get list of available trading pairs.

        Returns:
            List of trading pair symbols
        """
        pass