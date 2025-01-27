from datetime import datetime, timedelta, UTC
from typing import List, Optional
from loguru import logger

from .interfaces import IExchange, IExchangeFactory, IDataStorage
from .models import OHLCV

class DataFetcher:
    def __init__(self, exchange_factory: IExchangeFactory, storage: IDataStorage):
        self.exchange_factory = exchange_factory
        self.storage = storage
        
    def _create_ohlcv_models(
        self,
        exchange_id: str,
        symbol: str,
        ohlcv_data: List
    ) -> List[OHLCV]:
        """Convert raw OHLCV data to database models"""
        models = []
        for data in ohlcv_data:
            timestamp, open_price, high, low, close, volume = data
            # Create timezone-aware datetime
            ts = datetime.fromtimestamp(timestamp / 1000, tz=UTC)
            models.append(
                OHLCV(
                    exchange=exchange_id,
                    symbol=symbol,
                    timestamp=ts,
                    open=float(open_price),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume)
                )
            )
        return models
        
    def fetch_ohlcv(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[OHLCV]:
        """Fetch OHLCV data from exchange and convert to database models"""
        try:
            exchange = self.exchange_factory.create_exchange(exchange_id)
            ohlcv_data = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
            return self._create_ohlcv_models(exchange_id, symbol, ohlcv_data)
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {str(e)}")
            raise
            
    def update_historical_data(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = '1h',
        days_back: int = 30
    ) -> None:
        """Update historical data for a symbol"""
        since = datetime.now(UTC) - timedelta(days=days_back)
        
        try:
            data = self.fetch_ohlcv(
                exchange_id=exchange_id,
                symbol=symbol,
                timeframe=timeframe,
                since=since
            )
            
            if data:
                self.storage.save_ohlcv(data)
                logger.info(f"Updated historical data for {symbol} on {exchange_id}")
            
        except Exception as e:
            logger.error(f"Error updating historical data: {str(e)}")
            raise