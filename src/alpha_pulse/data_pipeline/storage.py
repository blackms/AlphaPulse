from datetime import datetime
from typing import List, Optional, Any
from loguru import logger
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import desc
from sqlalchemy.exc import IntegrityError

from .interfaces import IDataStorage
from .models import OHLCV
from .database import get_db

class SQLAlchemyStorage(IDataStorage):
    def save_ohlcv(self, data: List[OHLCV]) -> None:
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

    def get_latest_ohlcv(self, exchange: str, symbol: str) -> Optional[datetime]:
        """Get the timestamp of the latest OHLCV record for a symbol"""
        with get_db() as db:
            try:
                latest = db.query(OHLCV)\
                    .filter(OHLCV.exchange == exchange)\
                    .filter(OHLCV.symbol == symbol)\
                    .order_by(desc(OHLCV.timestamp))\
                    .first()
                return latest.timestamp if latest else None
            except Exception as e:
                logger.error(f"Error fetching latest OHLCV: {str(e)}")
                raise

    def get_historical_data(self, exchange: str, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Retrieve historical OHLCV data from the database.

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            timeframe: Time interval (e.g., '1h', '1d')

        Returns:
            DataFrame containing OHLCV data
        """
        with get_db() as db:
            try:
                # Query all OHLCV records for the symbol
                records = db.query(OHLCV)\
                    .filter(OHLCV.exchange == exchange)\
                    .filter(OHLCV.symbol == symbol)\
                    .order_by(OHLCV.timestamp)\
                    .all()
                
                # Convert to DataFrame
                df = pd.DataFrame([{
                    'timestamp': r.timestamp,
                    'open': r.open,
                    'high': r.high,
                    'low': r.low,
                    'close': r.close,
                    'volume': r.volume
                } for r in records])
                
                if not df.empty:
                    df.set_index('timestamp', inplace=True)
                
                return df
                
            except Exception as e:
                logger.error(f"Error fetching historical data: {str(e)}")
                raise