"""
Real-time data manager for the data pipeline.

This module provides a manager that coordinates real-time market data operations,
handling data providers and storage for live market data.
"""
from datetime import datetime
from typing import Dict, List, Optional, Set

from loguru import logger

from alpha_pulse.exchanges import OHLCV, ExchangeType
from alpha_pulse.data_pipeline.core.interfaces import (
    IRealTimeDataManager,
    IOHLCVStorage,
    IMarketDataProvider
)
from alpha_pulse.data_pipeline.core.models import (
    MarketDataConfig,
    DataPipelineError
)


class RealTimeDataError(DataPipelineError):
    """Error raised by real-time data operations."""
    pass


class RealTimeDataManager(IRealTimeDataManager):
    """Manager for real-time market data operations."""

    def __init__(
        self,
        provider: IMarketDataProvider,
        storage: Optional[IOHLCVStorage] = None,
        config: Optional[MarketDataConfig] = None
    ):
        """
        Initialize real-time data manager.

        Args:
            provider: Market data provider implementation
            storage: Optional storage for persisting real-time data
            config: Manager configuration
        """
        self.provider = provider
        self.storage = storage
        self.config = config or MarketDataConfig()
        
        self._active_symbols: Set[str] = set()
        self._last_stored: Dict[str, datetime] = {}
        self._prices: Dict[str, float] = {}
        self._last_update: Dict[str, datetime] = {}

    async def start(self, symbols: List[str]) -> None:
        """
        Start real-time data updates.

        Args:
            symbols: List of symbols to track

        Raises:
            RealTimeDataError: If start operation fails
        """
        try:
            self._active_symbols.update(symbols)
            await self.provider.start(symbols)
            logger.info(f"Started real-time updates for {len(symbols)} symbols")

        except Exception as e:
            raise RealTimeDataError(f"Failed to start real-time updates: {str(e)}")

    def stop(self) -> None:
        """Stop real-time data updates."""
        try:
            self.provider.stop()
            self._active_symbols.clear()
            self._prices.clear()
            self._last_update.clear()
            logger.info("Stopped real-time updates")

        except Exception as e:
            logger.error(f"Error stopping real-time updates: {str(e)}")

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get latest cached price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Latest price or None if not available
        """
        if symbol not in self._active_symbols:
            return None

        # Check if price is stale
        if symbol in self._last_update:
            age = (datetime.now() - self._last_update[symbol]).total_seconds()
            if age > self.config.cache_duration:
                return None

        return self._prices.get(symbol)

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price directly from provider.

        Args:
            symbol: Trading symbol

        Returns:
            Current price or None if not available

        Raises:
            RealTimeDataError: If price fetch fails
        """
        try:
            price = await self.provider.get_current_price(symbol)
            if price is not None:
                self._prices[symbol] = price
                self._last_update[symbol] = datetime.now()
                
                # Store price update if storage is available
                if self.storage and symbol in self._active_symbols:
                    ohlcv = OHLCV(
                        exchange="realtime",
                        symbol=symbol,
                        timeframe="1m",
                        timestamp=datetime.now(),
                        open=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=0.0  # Real-time price updates don't include volume
                    )
                    self.storage.save_ohlcv([ohlcv])
                    self._last_stored[symbol] = datetime.now()
            
            return price

        except Exception as e:
            raise RealTimeDataError(f"Failed to get current price: {str(e)}")

    async def close(self) -> None:
        """Clean up manager resources."""
        self.stop()
        await self.provider.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()