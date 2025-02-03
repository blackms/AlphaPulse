"""
Data fetching module for AlphaPulse.
"""
from datetime import datetime, timedelta, UTC
from typing import List, Optional

from loguru import logger

from alpha_pulse.exchanges import ExchangeType, ExchangeFactory, OHLCV
from .interfaces import IDataStorage


class DataFetcher:
    """Handles fetching and storing market data."""

    def __init__(self, storage: IDataStorage):
        """
        Initialize data fetcher.

        Args:
            storage: Data storage implementation
        """
        self.storage = storage
        logger.info("Initialized DataFetcher")

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
        Fetch OHLCV data from exchange.

        Args:
            exchange_type: Type of exchange to use
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            since: Start time for data fetch
            limit: Number of candles to fetch
            testnet: Whether to use testnet

        Returns:
            List of OHLCV objects
        """
        # Create and initialize exchange
        exchange = await ExchangeFactory.create_exchange(exchange_type, testnet)

        try:
            # Convert datetime to milliseconds timestamp if provided
            since_ts = int(since.timestamp() * 1000) if since else None

            # Fetch OHLCV data
            return await exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_ts,
                limit=limit
            )

        finally:
            # Clean up exchange
            await exchange.__aexit__(None, None, None)

    async def update_historical_data(
        self,
        exchange_type: ExchangeType,
        symbol: str,
        timeframe: str,
        days_back: int = 30,
        end_time: Optional[datetime] = None,
        testnet: bool = False
    ) -> None:
        """
        Fetch and store historical market data.

        Args:
            exchange_type: Type of exchange to fetch from
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            days_back: Number of days of historical data to fetch
            end_time: End time for data fetch (defaults to now)
            testnet: Whether to use testnet
        """
        logger.info(
            f"Fetching {days_back} days of {timeframe} data for {symbol} "
            f"from {exchange_type.value}"
        )

        if end_time is None:
            end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=days_back)

        # Fetch OHLCV data from the exchange
        ohlcv_data = await self.fetch_ohlcv(
            exchange_type=exchange_type,
            symbol=symbol,
            timeframe=timeframe,
            since=start_time,
            testnet=testnet
        )
        
        # Convert exchange OHLCV objects to mapped OHLCV objects for storage
        from alpha_pulse.data_pipeline.models import OHLCVRecord as DB_OHLCV
        db_ohlcv_data = []
        for ohlcv in ohlcv_data:
            db_ohlcv_data.append(DB_OHLCV(
                exchange=exchange_type.value,
                symbol=symbol,
                timeframe=timeframe,
                timestamp=ohlcv.timestamp,
                open=ohlcv.open,
                high=ohlcv.high,
                low=ohlcv.low,
                close=ohlcv.close,
                volume=ohlcv.volume
            ))
        
        # Save the converted OHLCV data to storage
        self.storage.save_ohlcv(db_ohlcv_data)

        logger.info("Data fetched and stored successfully")