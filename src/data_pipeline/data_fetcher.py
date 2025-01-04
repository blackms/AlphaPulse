from datetime import datetime, timedelta
import time
from typing import List, Optional
import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .exchange import ExchangeManager
from .models import OHLCV
from .database import get_db
from config.settings import settings

class DataFetcher:
    def __init__(self):
        self.exchange_manager = ExchangeManager()
        
    def fetch_ohlcv(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[OHLCV]:
        """Fetch OHLCV data from exchange and convert to database models"""
        exchange = self.exchange_manager.get_exchange(exchange_id)
        
        try:
            # Convert datetime to timestamp in milliseconds
            since_ts = int(since.timestamp() * 1000) if since else None
            
            # Fetch data from exchange
            ohlcv_data = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since_ts,
                limit=limit
            )
            
            # Convert to OHLCV models
            models = []
            for data in ohlcv_data:
                timestamp, open_price, high, low, close, volume = data
                models.append(
                    OHLCV(
                        exchange=exchange_id,
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(timestamp / 1000),
                        open=float(open_price),
                        high=float(high),
                        low=float(low),
                        close=float(close),
                        volume=float(volume)
                    )
                )
            
            return models
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {str(e)}")
            raise
            
    def store_ohlcv(self, data: List[OHLCV]) -> None:
        """Store OHLCV data in database"""
        with get_db() as db:
            try:
                db.bulk_save_objects(data)
                db.commit()
                logger.info(f"Stored {len(data)} OHLCV records")
            except IntegrityError as e:
                db.rollback()
                logger.error(f"Database integrity error: {str(e)}")
                raise
            
    def update_historical_data(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = '1h',
        days_back: int = 30
    ) -> None:
        """Update historical data for a symbol"""
        since = datetime.utcnow() - timedelta(days=days_back)
        
        try:
            data = self.fetch_ohlcv(
                exchange_id=exchange_id,
                symbol=symbol,
                timeframe=timeframe,
                since=since
            )
            
            if data:
                self.store_ohlcv(data)
                logger.info(f"Updated historical data for {symbol} on {exchange_id}")
            
        except Exception as e:
            logger.error(f"Error updating historical data: {str(e)}")
            raise 