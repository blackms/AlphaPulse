"""
Historical data manager for the data pipeline.

This module provides a manager that coordinates between fetchers and storage
to handle historical market data operations.
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from loguru import logger

from alpha_pulse.exchanges import OHLCV, ExchangeType
from alpha_pulse.data_pipeline.core.interfaces import (
    IHistoricalDataManager,
    IHistoricalDataStorage,
    IDataFetcher,
    IOHLCVStorage
)
from alpha_pulse.data_pipeline.core.errors import DataPipelineError
from alpha_pulse.data_pipeline.core.config import MarketDataConfig
from alpha_pulse.data_pipeline.core.validation import validate_timeframe

# Get timeframe durations from MarketDataConfig's default
_default_config = MarketDataConfig()
TIMEFRAME_DURATIONS = _default_config.timeframe_durations


class HistoricalDataError(DataPipelineError):
    """Error raised by historical data operations."""
    pass


class HistoricalDataManager(IHistoricalDataManager):
    """Manager for historical market data operations."""

    def __init__(
        self,
        storage: IHistoricalDataStorage,
        fetcher: IDataFetcher
    ):
        """
        Initialize historical data manager.

        Args:
            storage: Historical data storage implementation
            fetcher: Data fetcher implementation
        """
        self.storage = storage
        self.fetcher = fetcher

    async def ensure_data_available(
        self,
        exchange_type: ExchangeType,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> bool:
        """
        Ensure historical data is available for the specified period.

        Args:
            exchange_type: Type of exchange
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_time: Start time for data check
            end_time: End time for data check

        Returns:
            True if data is available, False otherwise

        Raises:
            HistoricalDataError: If data check fails
        """
        try:
            validate_timeframe(timeframe, list(TIMEFRAME_DURATIONS.keys()))

            # Get existing data range
            data = self.get_historical_data(
                exchange_type, symbol, timeframe,
                start_time, end_time
            )

            if not data:
                # No data exists, fetch all
                await self._fetch_and_store_data(
                    exchange_type, symbol, timeframe,
                    start_time, end_time
                )
                return True

            # Check for gaps in data
            duration = timedelta(seconds=TIMEFRAME_DURATIONS[timeframe])
            current_time = start_time
            last_timestamp = None

            for ohlcv in data:
                if last_timestamp:
                    expected_time = last_timestamp + duration
                    if ohlcv.timestamp > expected_time:
                        # Gap found, fetch missing data
                        await self._fetch_and_store_data(
                            exchange_type, symbol, timeframe,
                            last_timestamp, ohlcv.timestamp
                        )
                last_timestamp = ohlcv.timestamp

            # Check if we need to fetch data after the last record
            if last_timestamp and last_timestamp < end_time:
                await self._fetch_and_store_data(
                    exchange_type, symbol, timeframe,
                    last_timestamp, end_time
                )

            return True

        except Exception as e:
            raise HistoricalDataError(
                f"Failed to ensure data availability: {str(e)}"
            )

    async def _fetch_and_store_data(
        self,
        exchange_type: ExchangeType,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> None:
        """
        Fetch and store historical data for a time range.

        Args:
            exchange_type: Type of exchange
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_time: Start time for data fetch
            end_time: End time for data fetch

        Raises:
            HistoricalDataError: If fetch or store operation fails
        """
        try:
            # Fetch data
            data = await self.fetcher.fetch_ohlcv(
                exchange_type=exchange_type,
                symbol=symbol,
                timeframe=timeframe,
                since=start_time
            )

            if data:
                # Filter data to requested time range
                filtered_data = [
                    d for d in data 
                    if start_time <= d.timestamp <= end_time
                ]
                
                if isinstance(self.storage, IOHLCVStorage):
                    self.storage.save_ohlcv(filtered_data)
                    logger.info(
                        f"Fetched and stored {len(filtered_data)} records for "
                        f"{symbol} from {exchange_type.value}"
                    )

        except Exception as e:
            raise HistoricalDataError(
                f"Failed to fetch and store data: {str(e)}"
            )

    def get_historical_data(
        self,
        exchange_type: ExchangeType,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[OHLCV]:
        """
        Get historical data for the specified period.

        Args:
            exchange_type: Type of exchange
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_time: Start time for data retrieval
            end_time: End time for data retrieval

        Returns:
            List of OHLCV objects

        Raises:
            HistoricalDataError: If data retrieval fails
        """
        try:
            validate_timeframe(timeframe, list(TIMEFRAME_DURATIONS.keys()))
            return self.storage.get_historical_data(
                exchange_type, symbol, timeframe,
                start_time, end_time
            )

        except Exception as e:
            raise HistoricalDataError(
                f"Failed to get historical data: {str(e)}"
            )