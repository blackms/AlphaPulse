"""
SQL storage implementation for the data pipeline.

This module provides SQLAlchemy-based storage functionality for market data.
"""
from datetime import datetime
from typing import List, Optional

from loguru import logger
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import select

from alpha_pulse.exchanges import OHLCV, ExchangeType
from alpha_pulse.config.settings import settings
from alpha_pulse.data_pipeline.core.interfaces import IDataStorage
from alpha_pulse.data_pipeline.core.models import (
    StorageConfig,
    DataPipelineError
)
from alpha_pulse.data_pipeline.models import Base, OHLCVRecord


class StorageError(DataPipelineError):
    """Error raised by storage operations."""
    pass


class SQLAlchemyStorage(IDataStorage):
    """SQLAlchemy-based data storage implementation."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        config: Optional[StorageConfig] = None
    ):
        """
        Initialize SQL storage.

        Args:
            database_url: Database connection URL
            config: Storage configuration
        """
        self.config = config or StorageConfig()
        self.database_url = database_url or settings.DATABASE_URL
        self.engine = None
        self.Session = None
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize database connection and create tables."""
        if self._initialized:
            return

        try:
            self.engine = sa.create_engine(
                self.database_url,
                pool_size=self.config.max_connections,
                pool_timeout=self.config.timeout
            )
            self.Session = sessionmaker(bind=self.engine)
            
            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)
            self._initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            raise StorageError(f"Database initialization failed: {str(e)}")

    def save_ohlcv(self, data: List[OHLCV]) -> None:
        """
        Save OHLCV data to storage.

        Args:
            data: List of OHLCV records to save

        Raises:
            StorageError: If save operation fails
        """
        if not data:
            return

        self._initialize()

        with self.Session() as session:
            try:
                # Convert OHLCV objects to database records
                db_records = []
                for ohlcv in data:
                    db_record = OHLCVRecord(
                        exchange=ohlcv.exchange,
                        symbol=ohlcv.symbol,
                        timeframe=ohlcv.timeframe,
                        timestamp=ohlcv.timestamp,
                        open=float(ohlcv.open),
                        high=float(ohlcv.high),
                        low=float(ohlcv.low),
                        close=float(ohlcv.close),
                        volume=float(ohlcv.volume)
                    )
                    db_records.append(db_record)

                # Save records in batches
                for i in range(0, len(db_records), self.config.batch_size):
                    batch = db_records[i:i + self.config.batch_size]
                    session.add_all(batch)
                    session.commit()

                logger.info(f"Saved {len(data)} OHLCV records")

            except Exception as e:
                session.rollback()
                raise StorageError(f"Failed to save OHLCV data: {str(e)}")

    def get_historical_data(
        self,
        exchange_type: ExchangeType,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[OHLCV]:
        """
        Retrieve historical market data.

        Args:
            exchange_type: Type of exchange
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_time: Start time for data retrieval
            end_time: End time for data retrieval

        Returns:
            List of OHLCV objects

        Raises:
            StorageError: If retrieval operation fails
        """
        self._initialize()

        with self.Session() as session:
            try:
                query = select(OHLCVRecord).where(
                    OHLCVRecord.exchange == exchange_type.value,
                    OHLCVRecord.symbol == symbol,
                    OHLCVRecord.timeframe == timeframe
                )

                if start_time:
                    query = query.where(OHLCVRecord.timestamp >= start_time)
                if end_time:
                    query = query.where(OHLCVRecord.timestamp <= end_time)

                query = query.order_by(OHLCVRecord.timestamp)
                result = session.execute(query).scalars().all()

                # Convert database records to OHLCV objects
                data = [record.to_ohlcv() for record in result]
                logger.info(
                    f"Retrieved {len(data)} OHLCV records for {symbol} "
                    f"from {exchange_type.value}"
                )
                return data

            except Exception as e:
                raise StorageError(f"Failed to retrieve historical data: {str(e)}")

    def get_latest_ohlcv(
        self,
        exchange_type: ExchangeType,
        symbol: str
    ) -> Optional[datetime]:
        """
        Get timestamp of latest OHLCV record.

        Args:
            exchange_type: Type of exchange
            symbol: Trading pair symbol

        Returns:
            Timestamp of latest record or None if no records exist

        Raises:
            StorageError: If retrieval operation fails
        """
        self._initialize()

        with self.Session() as session:
            try:
                result = session.query(OHLCVRecord.timestamp)\
                    .filter(
                        OHLCVRecord.exchange == exchange_type.value,
                        OHLCVRecord.symbol == symbol
                    )\
                    .order_by(OHLCVRecord.timestamp.desc())\
                    .first()

                return result[0] if result else None

            except Exception as e:
                raise StorageError(f"Failed to get latest OHLCV: {str(e)}")

    def __enter__(self):
        """Context manager entry."""
        self._initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.engine:
            self.engine.dispose()