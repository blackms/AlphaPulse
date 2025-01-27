from datetime import datetime
from typing import List, Optional, Any
from loguru import logger
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