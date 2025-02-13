"""
Data fetcher implementation.
"""
from datetime import datetime, timedelta, UTC
from typing import List, Optional

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime

from ..exchanges.factories import ExchangeFactory
from ..exchanges.base import OHLCV


class IDataStorage(ABC):
    """Interface for data storage."""
    
    @abstractmethod
    async def save_ohlcv(self, records: List[OHLCV]) -> None:
        """Save OHLCV records."""
        pass
    
    @abstractmethod
    async def get_latest_ohlcv(
        self,
        exchange: str,
        symbol: str
    ) -> Optional[datetime]:
        """Get latest OHLCV timestamp."""
        pass


class DataFetcher:
    """Fetches and stores market data."""
    
    def __init__(self, exchange_factory: ExchangeFactory, storage: IDataStorage):
        self.exchange_factory = exchange_factory
        self.storage = storage
    
    async def fetch_ohlcv(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """
        Fetch OHLCV data from exchange.
        
        Args:
            exchange_id: Exchange identifier
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            since: Start time
            limit: Maximum number of candles
            
        Returns:
            List of OHLCV candles
        """
        exchange = await self.exchange_factory.create_exchange(exchange_id)
        try:
            since_ts = int(since.timestamp() * 1000) if since else None
            return await exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_ts,
                limit=limit
            )
        finally:
            await exchange.close()
    
    async def update_historical_data(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = "1h",
        days_back: int = 30
    ) -> None:
        """
        Update historical data in storage.
        
        Args:
            exchange_id: Exchange identifier
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            days_back: Number of days to fetch
        """
        since = datetime.now(UTC) - timedelta(days=days_back)
        data = await self.fetch_ohlcv(
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            since=since
        )
        
        # Add metadata to OHLCV records
        for record in data:
            record.exchange = exchange_id
            record.symbol = symbol
            record.timeframe = timeframe
        
        await self.storage.save_ohlcv(data)