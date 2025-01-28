"""
Data fetching module for AlphaPulse.
"""
from datetime import datetime, timedelta, UTC
from typing import List, Optional

from loguru import logger

from alpha_pulse.data_pipeline.interfaces import IExchange, IExchangeFactory, IDataStorage
from alpha_pulse.data_pipeline.models import OHLCV


class DataFetcher:
    """Handles fetching and storing market data."""

    def __init__(
        self,
        exchange_factory: IExchangeFactory,
        storage: IDataStorage
    ):
        """
        Initialize data fetcher.

        Args:
            exchange_factory: Factory for creating exchange instances
            storage: Data storage implementation
        """
        self.exchange_factory = exchange_factory
        self.storage = storage
        logger.info("Initialized DataFetcher")

    def fetch_ohlcv(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """
        Fetch OHLCV data from exchange.

        Args:
            exchange_id: ID of the exchange
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            since: Start time for data fetch
            limit: Number of candles to fetch

        Returns:
            List of OHLCV objects
        """
        exchange = self.exchange_factory.create_exchange(exchange_id)
        
        # Convert datetime to milliseconds timestamp if provided
        since_ts = int(since.timestamp() * 1000) if since else None
        
        # Fetch raw OHLCV data
        raw_data = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since_ts,
            limit=limit
        )
        
        # Convert to OHLCV objects
        ohlcv_data = []
        for candle in raw_data:
            timestamp = datetime.fromtimestamp(candle[0] / 1000, UTC)
            ohlcv = OHLCV(
                exchange=exchange_id,
                symbol=symbol,
                timestamp=timestamp,
                open=candle[1],
                high=candle[2],
                low=candle[3],
                close=candle[4],
                volume=candle[5]
            )
            ohlcv_data.append(ohlcv)
        
        return ohlcv_data

    def update_historical_data(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        days_back: int = 30,
        end_time: Optional[datetime] = None
    ) -> None:
        """
        Fetch and store historical market data.

        Args:
            exchange_id: ID of the exchange to fetch from
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            days_back: Number of days of historical data to fetch
            end_time: End time for data fetch (defaults to now)
        """
        if end_time is None:
            end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=days_back)
        
        logger.info(
            f"Fetching {days_back} days of {timeframe} data for {symbol} "
            f"from {exchange_id}"
        )
        
        # Fetch OHLCV data
        ohlcv_data = self.fetch_ohlcv(
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            since=start_time
        )
        
        # Save to storage
        self.storage.save_ohlcv(ohlcv_data)
        
        logger.info("Data fetched and stored successfully")

    def get_historical_data(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[OHLCV]:
        """
        Retrieve historical market data.

        Args:
            exchange_id: ID of the exchange
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_time: Start time for data retrieval
            end_time: End time for data retrieval

        Returns:
            List of OHLCV objects
        """
        return self.storage.get_historical_data(
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )