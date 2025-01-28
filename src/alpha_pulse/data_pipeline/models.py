"""
Database models for AlphaPulse data pipeline.
"""
from datetime import datetime
from typing import Dict, Any
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, ForeignKey, MetaData
from sqlalchemy.orm import declarative_base, relationship

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


class OHLCV(Base):
    """OHLCV (Open, High, Low, Close, Volume) data model."""

    __tablename__ = 'ohlcv_data'

    id = Column(Integer, primary_key=True)
    exchange = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    def __init__(
        self,
        exchange: str,
        symbol: str,
        timestamp: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        timeframe: str = "1h"
    ):
        """
        Initialize OHLCV record.

        Args:
            exchange: Exchange identifier
            symbol: Trading pair symbol
            timestamp: Candle timestamp
            open: Opening price
            high: Highest price
            low: Lowest price
            close: Closing price
            volume: Trading volume
            timeframe: Candle timeframe (default: "1h")
        """
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def __repr__(self) -> str:
        """Get string representation."""
        return (
            f"OHLCV({self.exchange}, {self.symbol}, {self.timestamp}, "
            f"O:{self.open:.2f}, H:{self.high:.2f}, L:{self.low:.2f}, "
            f"C:{self.close:.2f}, V:{self.volume:.2f})"
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
        """
        Initialize strategy.

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
    exchange = Column(String(50), nullable=False, index=True)
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
        exchange: str,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        timestamp: datetime
    ):
        """
        Initialize trade.

        Args:
            strategy_id: ID of the strategy that generated the trade
            exchange: Exchange where the trade was executed
            symbol: Trading pair symbol
            side: Trade side ('buy' or 'sell')
            amount: Trade amount
            price: Trade price
            timestamp: Trade timestamp
        """
        self.strategy_id = strategy_id
        self.exchange = exchange
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