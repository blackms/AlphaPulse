"""
SQLAlchemy models for the AlphaPulse database.

This module defines the SQLAlchemy ORM models for the database tables used by the
AlphaPulse data pipeline.
"""
import enum
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Enum, 
    ForeignKey, Text, Boolean, UniqueConstraint, MetaData
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

# Create a metadata object with naming conventions for constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}
metadata = MetaData(naming_convention=convention)

# Create a base class for all models
Base = declarative_base(metadata=metadata)


class SyncStatusEnum(enum.Enum):
    """Enum for sync status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SyncStatus(Base):
    """Model for the sync_status table."""
    __tablename__ = "sync_status"
    
    id = Column(Integer, primary_key=True)
    exchange_id = Column(String, nullable=False)
    data_type = Column(String, nullable=False)
    status = Column(String, nullable=False)
    last_sync = Column(DateTime(timezone=True))
    next_sync = Column(DateTime(timezone=True))
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        UniqueConstraint('exchange_id', 'data_type', name='uq_sync_status_exchange_data_type'),
    )


class ExchangeBalance(Base):
    """Model for the exchange_balances table."""
    __tablename__ = "exchange_balances"
    
    id = Column(Integer, primary_key=True)
    exchange_id = Column(String, nullable=False)
    currency = Column(String, nullable=False)
    available = Column(Float)
    locked = Column(Float)
    total = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        UniqueConstraint('exchange_id', 'currency', name='uq_exchange_balances_exchange_currency'),
    )


class ExchangePosition(Base):
    """Model for the exchange_positions table."""
    __tablename__ = "exchange_positions"
    
    id = Column(Integer, primary_key=True)
    exchange_id = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    quantity = Column(Float)
    entry_price = Column(Float)
    current_price = Column(Float)
    unrealized_pnl = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        UniqueConstraint('exchange_id', 'symbol', name='uq_exchange_positions_exchange_symbol'),
    )


class ExchangeOrder(Base):
    """Model for the exchange_orders table."""
    __tablename__ = "exchange_orders"
    
    id = Column(Integer, primary_key=True)
    exchange_id = Column(String, nullable=False)
    order_id = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    order_type = Column(String)
    side = Column(String)
    price = Column(Float)
    amount = Column(Float)
    filled = Column(Float)
    status = Column(String)
    timestamp = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        UniqueConstraint('exchange_id', 'order_id', name='uq_exchange_orders_exchange_order_id'),
    )


class ExchangePrice(Base):
    """Model for the exchange_prices table."""
    __tablename__ = "exchange_prices"
    
    id = Column(Integer, primary_key=True)
    exchange_id = Column(String, nullable=False)
    base_currency = Column(String, nullable=False)
    quote_currency = Column(String, nullable=False)
    price = Column(Float)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        UniqueConstraint('exchange_id', 'base_currency', 'quote_currency', 
                         name='uq_exchange_prices_exchange_currencies'),
    )