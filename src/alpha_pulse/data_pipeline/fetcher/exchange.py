"""
Exchange-based data fetcher implementation.

This module provides functionality for fetching market data from cryptocurrency exchanges.
"""
import asyncio
from datetime import datetime
from typing import List, Optional

from loguru import logger

from alpha_pulse.exchanges import (
    OHLCV,
    ExchangeType,
    ExchangeFactory,
    BaseExchange
)
from alpha_pulse.data_pipeline.core.interfaces import IDataFetcher
from alpha_pulse.data_pipeline.core.models import (
    DataFetchConfig,
    DataPipelineError
)


class DataFetchError(DataPipelineError):
    """Error raised by data fetching operations."""
    pass


class ExchangeFetcher(IDataFetcher):
    """Fetcher implementation for cryptocurrency exchanges."""

    def __init__(self, config: Optional[DataFetchConfig] = None):
        """
        Initialize exchange fetcher.

        Args:
            config: Fetcher configuration
        """
        self.config = config or DataFetchConfig()
        self._exchange: Optional[BaseExchange] = None

    async def _get_exchange(
        self,
        exchange_type: ExchangeType,
        testnet: bool
    ) -> BaseExchange:
        """
        Get or create exchange instance.

        Args:
            exchange_type: Type of exchange
            testnet: Whether to use testnet

        Returns:
            Exchange instance
        """
        if not self._exchange:
            try:
                self._exchange = await ExchangeFactory.create_exchange(
                    exchange_type,
                    testnet=testnet
                )
                logger.debug(
                    f"Created exchange instance for {exchange_type.value}"
                )
            except Exception as e:
                raise DataFetchError(f"Failed to create exchange: {str(e)}")

        return self._exchange

    async def fetch_ohlcv(
        self,
        exchange_type: ExchangeType,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
        testnet: bool = False
    ) -> List[OHLCV]:
        """
        Fetch OHLCV data with retry logic.

        Args:
            exchange_type: Type of exchange to use
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            since: Start time for data fetch
            limit: Number of candles to fetch
            testnet: Whether to use testnet

        Returns:
            List of OHLCV objects

        Raises:
            DataFetchError: If fetch operation fails after retries
        """
        retries = 0
        while retries < self.config.max_retries:
            try:
                # Get exchange instance
                exchange = await self._get_exchange(exchange_type, testnet)

                # Convert datetime to milliseconds timestamp if provided
                since_ts = int(since.timestamp() * 1000) if since else None

                # Fetch OHLCV data
                data = await exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since_ts,
                    limit=limit
                )

                logger.debug(
                    f"Fetched {len(data)} OHLCV records for {symbol} "
                    f"from {exchange_type.value}"
                )
                return data

            except Exception as e:
                retries += 1
                if retries >= self.config.max_retries:
                    raise DataFetchError(
                        f"Failed to fetch OHLCV data after {retries} retries: {str(e)}"
                    )
                
                logger.warning(
                    f"Fetch attempt {retries} failed, retrying in "
                    f"{self.config.retry_delay} seconds..."
                )
                await asyncio.sleep(self.config.retry_delay)

    async def close(self) -> None:
        """Clean up exchange resources."""
        if self._exchange:
            try:
                await self._exchange.close()
                self._exchange = None
            except Exception as e:
                logger.error(f"Error closing exchange: {str(e)}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()