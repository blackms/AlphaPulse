"""
Data storage module for AlphaPulse.
"""
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from loguru import logger
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from alpha_pulse.data_pipeline.interfaces import IDataStorage
from alpha_pulse.config.settings import settings
from alpha_pulse.data_pipeline.models import OHLCV, Base


class SQLAlchemyStorage(IDataStorage):
    """SQLAlchemy-based data storage implementation."""

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize storage connection.

        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url or settings.DATABASE_URL
        self.engine = sa.create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"Initialized SQLAlchemyStorage with URL: {self.database_url}")

        # Create tables if they don't exist
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created")

    def save_historical_data(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        data: Dict
    ) -> None:
        """
        Save historical market data.

        Args:
            exchange_id: ID of the exchange
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            data: Dict containing OHLCV data
        """
        ohlcv_data = []
        for i in range(len(data['open'])):
            ohlcv = OHLCV(
                exchange=exchange_id,
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),  # Should be adjusted based on actual data
                open=data['open'][i],
                high=data['high'][i],
                low=data['low'][i],
                close=data['close'][i],
                volume=data['volume'][i]
            )
            ohlcv_data.append(ohlcv)

        self.save_ohlcv(ohlcv_data)

    def save_ohlcv(self, data: List[OHLCV]) -> None:
        """
        Save OHLCV data to storage.

        Args:
            data: List of OHLCV objects to save
        """
        with self.Session() as session:
            session.add_all(data)
            session.commit()

        logger.info(f"Saved {len(data)} OHLCV records")

    def get_historical_data(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
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
        with self.Session() as session:
            query = session.query(OHLCV).filter(
                OHLCV.exchange == exchange_id,
                OHLCV.symbol == symbol,
                OHLCV.timeframe == timeframe
            )

            if start_time:
                query = query.filter(OHLCV.timestamp >= start_time)
            if end_time:
                query = query.filter(OHLCV.timestamp <= end_time)

            query = query.order_by(OHLCV.timestamp)
            result = query.all()

        logger.info(
            f"Retrieved {len(result)} OHLCV records for {symbol} "
            f"from {exchange_id}"
        )

        return result

    def get_latest_ohlcv(
        self,
        exchange_id: str,
        symbol: str
    ) -> Optional[datetime]:
        """
        Get timestamp of latest OHLCV record.

        Args:
            exchange_id: Exchange identifier
            symbol: Trading pair symbol

        Returns:
            Timestamp of latest record or None if no records exist
        """
        with self.Session() as session:
            result = session.query(OHLCV.timestamp)\
                .filter(
                    OHLCV.exchange == exchange_id,
                    OHLCV.symbol == symbol
                )\
                .order_by(OHLCV.timestamp.desc())\
                .first()

            return result[0] if result else None