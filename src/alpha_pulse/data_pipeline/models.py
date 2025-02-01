"""
Database models for AlphaPulse data pipeline.
"""
from datetime import datetime
from typing import Dict, Any
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, ForeignKey, MetaData
from sqlalchemy.orm import declarative_base, relationship

from alpha_pulse.exchanges import ExchangeType, OHLCV

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
    """Database model for OHLCV data."""
    
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
        return cls(
            exchange=exchange_type.value,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=ohlcv.timestamp,
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
        return OHLCV(
            timestamp=self.timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume
        )


class Strategy(Base):
    """Trading strategy model."""
    
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    type = Column(String(20), nullable=False)  # e.g., 'ta', 'ml', etc.
    parameters = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Relationships
    trades = relationship("Trade", back_populates="strategy", cascade="all, delete-orphan")
    
    def __init__(
        self,
        name: str,
        type: str,
        parameters: Dict[str, Any],
        created_at: datetime
    ):
        """Initialize strategy.
        
        Args:
            name: Strategy name
            type: Strategy type (e.g., 'ta', 'ml')
            parameters: Strategy parameters
            created_at: Creation timestamp
        """
        self.name = name
        self.type = type
        self.parameters = parameters
        self.created_at = created_at
    
    def __repr__(self) -> str:
        """Get string representation."""
        return f"Strategy({self.name}, {self.type}, {self.parameters})"


class Trade(Base):
    """Trade execution model."""
    
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id', ondelete='CASCADE'), nullable=False)
    exchange = Column(String(50), nullable=False, index=True)  # ExchangeType value
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="trades")
    
    def __init__(
        self,
        strategy_id: int,
        exchange_type: ExchangeType,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        timestamp: datetime
    ):
        """Initialize trade.
        
        Args:
            strategy_id: ID of the strategy that generated the trade
            exchange_type: Type of exchange where the trade was executed
            symbol: Trading pair symbol
            side: Trade side ('buy' or 'sell')
            amount: Trade amount
            price: Trade price
            timestamp: Trade timestamp
        """
        self.strategy_id = strategy_id
        self.exchange = exchange_type.value
        self.symbol = symbol
        self.side = side
        self.amount = amount
        self.price = price
        self.timestamp = timestamp
    
    def __repr__(self) -> str:
        """Get string representation."""
        return (
            f"Trade({self.exchange}, {self.symbol}, {self.side}, "
            f"amount={self.amount:.8f}, price={self.price:.2f})"
        )