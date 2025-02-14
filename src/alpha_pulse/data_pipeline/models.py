"""
Market data models for AlphaPulse data pipeline.
"""
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
from sqlalchemy import Column, Integer, Float, String, DateTime, MetaData
from sqlalchemy.orm import declarative_base

from alpha_pulse.exchanges import ExchangeType, OHLCV


@dataclass
class MarketData:
    """Container for market data used by agents and models."""
    prices: pd.DataFrame  # Historical price data
    volumes: pd.DataFrame  # Trading volume data
    fundamentals: Optional[Dict[str, Any]] = None  # Fundamental data
    sentiment: Optional[Dict[str, float]] = None  # Sentiment scores
    technical_indicators: Optional[Dict[str, pd.DataFrame]] = None  # Technical indicators
    timestamp: datetime = None  # Data timestamp

# Create a metadata object with proper naming convention
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

metadata = MetaData(naming_convention=convention)
Base = declarative_base(metadata=metadata)


class OHLCVRecord(Base):
    """Database model for OHLCV market data."""
    
    __tablename__ = 'ohlcv_data'
    
    id = Column(Integer, primary_key=True)
    exchange = Column(String(50), nullable=False, index=True)  # ExchangeType value
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    @classmethod
    def from_ohlcv(cls, exchange_type: ExchangeType, symbol: str, timeframe: str, ohlcv: OHLCV) -> 'OHLCVRecord':
        """Create database record from OHLCV model.
        
        Args:
            exchange_type: Type of exchange
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            ohlcv: OHLCV data
            
        Returns:
            Database record
        """
        # Ensure timestamp is timezone-aware
        timestamp = ohlcv.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
            
        return cls(
            exchange=exchange_type.value,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            open=float(ohlcv.open),
            high=float(ohlcv.high),
            low=float(ohlcv.low),
            close=float(ohlcv.close),
            volume=float(ohlcv.volume)
        )
    
    def to_ohlcv(self) -> OHLCV:
        """Convert database record to OHLCV model.
        
        Returns:
            OHLCV model
        """
        # Ensure timestamp is timezone-aware
        timestamp = self.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
            
        return OHLCV(
            timestamp=timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            exchange=self.exchange,
            symbol=self.symbol,
            timeframe=self.timeframe
        )