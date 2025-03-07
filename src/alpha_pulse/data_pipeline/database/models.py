"""
Database models for AlphaPulse.

This module defines SQLAlchemy ORM models for database tables.
"""
import uuid
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, 
    ForeignKey, Text, Numeric, JSON, Table, MetaData
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func

# Create base model class
Base = declarative_base()

# Define metadata
metadata = MetaData(schema="alphapulse")


class BaseModel(Base):
    """Base model with common fields."""
    
    __abstract__ = True
    __table_args__ = {'schema': 'alphapulse'}
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class User(BaseModel):
    """User model for authentication and authorization."""
    
    __tablename__ = 'users'
    
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    role = Column(String(20), nullable=False)
    last_login = Column(DateTime(timezone=True))
    
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"


class ApiKey(BaseModel):
    """API key model for API authentication."""
    
    __tablename__ = 'api_keys'
    
    user_id = Column(Integer, ForeignKey('alphapulse.users.id', ondelete='CASCADE'), nullable=False)
    api_key = Column(String(255), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    permissions = Column(JSON, nullable=False, default=lambda: [])
    expires_at = Column(DateTime(timezone=True))
    last_used = Column(DateTime(timezone=True))
    
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self):
        return f"<ApiKey(id={self.id}, name='{self.name}', user_id={self.user_id})>"


class Portfolio(BaseModel):
    """Portfolio model for tracking investment portfolios."""
    
    __tablename__ = 'portfolios'
    
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="portfolio", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Portfolio(id={self.id}, name='{self.name}')>"
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        position_value = sum(p.current_value for p in self.positions if p.current_value is not None)
        return position_value


class Position(BaseModel):
    """Position model for tracking holdings in a portfolio."""
    
    __tablename__ = 'positions'
    
    portfolio_id = Column(Integer, ForeignKey('alphapulse.portfolios.id', ondelete='CASCADE'), nullable=False)
    symbol = Column(String(20), nullable=False)
    quantity = Column(Numeric(18, 8), nullable=False)
    entry_price = Column(Numeric(18, 8), nullable=False)
    current_price = Column(Numeric(18, 8))
    
    portfolio = relationship("Portfolio", back_populates="positions")
    
    def __repr__(self):
        return f"<Position(id={self.id}, symbol='{self.symbol}', quantity={self.quantity})>"
    
    @property
    def current_value(self) -> Optional[float]:
        """Calculate current position value."""
        if self.current_price is not None:
            return float(self.quantity) * float(self.current_price)
        return None
    
    @property
    def entry_value(self) -> float:
        """Calculate entry position value."""
        return float(self.quantity) * float(self.entry_price)
    
    @property
    def pnl(self) -> Optional[float]:
        """Calculate profit/loss."""
        if self.current_value is not None:
            return self.current_value - self.entry_value
        return None
    
    @property
    def pnl_percentage(self) -> Optional[float]:
        """Calculate profit/loss percentage."""
        if self.pnl is not None and self.entry_value != 0:
            return (self.pnl / self.entry_value) * 100
        return None


class Trade(BaseModel):
    """Trade model for tracking executed trades."""
    
    __tablename__ = 'trades'
    
    portfolio_id = Column(Integer, ForeignKey('alphapulse.portfolios.id', ondelete='CASCADE'), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    quantity = Column(Numeric(18, 8), nullable=False)
    price = Column(Numeric(18, 8), nullable=False)
    fees = Column(Numeric(18, 8))
    order_type = Column(String(20), nullable=False)  # 'market', 'limit', etc.
    status = Column(String(20), nullable=False)  # 'pending', 'filled', 'cancelled', etc.
    executed_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    external_id = Column(String(100))  # ID from external exchange
    
    portfolio = relationship("Portfolio", back_populates="trades")
    
    def __repr__(self):
        return f"<Trade(id={self.id}, symbol='{self.symbol}', side='{self.side}', quantity={self.quantity})>"
    
    @property
    def value(self) -> float:
        """Calculate trade value."""
        return float(self.quantity) * float(self.price)
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost including fees."""
        if self.fees is not None:
            return self.value + float(self.fees)
        return self.value


class Alert(BaseModel):
    """Alert model for system alerts and notifications."""
    
    __tablename__ = 'alerts'
    
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    severity = Column(String(20), nullable=False)  # 'info', 'warning', 'critical', etc.
    source = Column(String(50), nullable=False)  # Component that generated the alert
    tags = Column(JSON, nullable=False, default=lambda: [])
    acknowledged = Column(Boolean, nullable=False, default=False)
    acknowledged_by = Column(String(50))
    acknowledged_at = Column(DateTime(timezone=True))
    
    def __repr__(self):
        return f"<Alert(id={self.id}, title='{self.title}', severity='{self.severity}')>"


# Note: Metrics are stored in a regular table since TimescaleDB is not available
class Metric:
    """Representation of a time-series metric."""
    
    def __init__(
        self, 
        metric_name: str, 
        value: float, 
        labels: Dict[str, str] = None, 
        timestamp: datetime = None
    ):
        self.metric_name = metric_name
        self.value = value
        self.labels = labels or {}
        self.timestamp = timestamp or datetime.now()
    
    def __repr__(self):
        return f"<Metric(name='{self.metric_name}', value={self.value}, timestamp={self.timestamp})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric_name': self.metric_name,
            'value': self.value,
            'labels': self.labels,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Metric':
        """Create from dictionary."""
        timestamp = data.get('timestamp')
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            metric_name=data['metric_name'],
            value=data['value'],
            labels=data.get('labels', {}),
            timestamp=timestamp
        )