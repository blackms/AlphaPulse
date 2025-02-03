"""
Exchange-based market data provider implementation.

This module provides real-time market data functionality from cryptocurrency exchanges.
"""
import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Set

from loguru import logger

from alpha_pulse.exchanges import ExchangeType, ExchangeFactory, BaseExchange
from alpha_pulse.data_pipeline.core.interfaces import IMarketDataProvider
from alpha_pulse.data_pipeline.core.models import (
    MarketDataConfig,
    DataPipelineError
)


class MarketDataError(DataPipelineError):
    """Error raised by market data operations."""
    pass


class ExchangeDataProvider(IMarketDataProvider):
    """Real-time market data provider for cryptocurrency exchanges."""

    def __init__(
        self,
        exchange_type: ExchangeType,
        testnet: bool = False,
        config: Optional[MarketDataConfig] = None
    ):
        """
        Initialize exchange data provider.

        Args:
            exchange_type: Type of exchange to use
            testnet: Whether to use testnet
            config: Provider configuration
        """
        self.exchange_type = exchange_type
        self.testnet = testnet
        self.config = config or MarketDataConfig()
        
        self.exchange: Optional[BaseExchange] = None
        self._active_symbols: Set[str] = set()
        self._prices: Dict[str, float] = {}
        self._decimal_prices: Dict[str, Decimal] = {}
        self._last_update: Dict[str, datetime] = {}
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

    async def _initialize(self) -> None:
        """Initialize exchange connection."""
        if not self.exchange:
            try:
                self.exchange = await ExchangeFactory.create_exchange(
                    self.exchange_type,
                    testnet=self.testnet
                )
                logger.info(
                    f"Initialized {self.exchange_type.value} "
                    f"data provider (testnet={self.testnet})"
                )
            except Exception as e:
                raise MarketDataError(f"Failed to initialize exchange: {str(e)}")

    async def start(self, symbols: List[str]) -> None:
        """
        Start market data updates.

        Args:
            symbols: List of symbols to track

        Raises:
            MarketDataError: If provider fails to start
        """
        try:
            await self._initialize()

            if len(symbols) > self.config.max_symbols:
                raise ValueError(
                    f"Number of symbols ({len(symbols)}) exceeds maximum "
                    f"allowed ({self.config.max_symbols})"
                )

            self._active_symbols.update(symbols)
            self._running = True

            # Start update loop if not already running
            if not self._update_task or self._update_task.done():
                self._update_task = asyncio.create_task(self._update_loop())
                logger.info("Started market data updates")

        except Exception as e:
            raise MarketDataError(f"Failed to start provider: {str(e)}")

    def stop(self) -> None:
        """Stop market data updates."""
        self._running = False
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
        logger.info("Stopped market data updates")

    async def _update_loop(self) -> None:
        """Main update loop for market data."""
        while self._running:
            try:
                await self._update_prices()
                await asyncio.sleep(self.config.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
                await asyncio.sleep(self.config.update_interval * 2)

    async def _update_prices(self) -> None:
        """Update prices for all active symbols."""
        if not self.exchange:
            await self._initialize()

        try:
            for symbol in self._active_symbols:
                price = await self.exchange.get_ticker_price(symbol)
                if price is not None:
                    self._prices[symbol] = float(price)
                    self._decimal_prices[symbol] = price
                    self._last_update[symbol] = datetime.now()

        except Exception as e:
            logger.error(f"Error updating prices: {str(e)}")

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
        Get current price directly from exchange.

        Args:
            symbol: Trading symbol

        Returns:
            Current price or None if not available

        Raises:
            MarketDataError: If price fetch fails
        """
        if not self.exchange:
            await self._initialize()

        try:
            price = await self.exchange.get_ticker_price(symbol)
            if price is not None:
                self._prices[symbol] = float(price)
                self._decimal_prices[symbol] = price
                self._last_update[symbol] = datetime.now()
                return float(price)
            return None

        except Exception as e:
            raise MarketDataError(f"Failed to get current price: {str(e)}")

    def get_decimal_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get latest cached price as Decimal.

        Args:
            symbol: Trading symbol

        Returns:
            Latest price as Decimal or None if not available
        """
        if symbol not in self._active_symbols:
            return None

        # Check if price is stale
        if symbol in self._last_update:
            age = (datetime.now() - self._last_update[symbol]).total_seconds()
            if age > self.config.cache_duration:
                return None

        return self._decimal_prices.get(symbol)

    async def close(self) -> None:
        """Clean up provider resources."""
        self.stop()
        if self.exchange:
            try:
                await self.exchange.close()
                self.exchange = None
            except Exception as e:
                logger.error(f"Error closing exchange: {str(e)}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()