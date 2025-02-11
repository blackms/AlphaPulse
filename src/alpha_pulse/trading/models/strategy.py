"""
Trading strategy models for AlphaPulse.
"""
from datetime import datetime
from typing import Dict, Any, List
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
        exchange: str,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        timestamp: datetime
    ):
        """Initialize trade.
        
        Args:
            strategy_id: ID of the strategy that generated the trade
            exchange: Exchange identifier
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