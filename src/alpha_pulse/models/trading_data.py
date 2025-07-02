"""
Encrypted trading data models for AlphaPulse.

This module contains SQLAlchemy models for trading data with
field-level encryption for sensitive information.
"""
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, 
    ForeignKey, Index, UniqueConstraint, Float
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.declarative import declarative_base

from .encrypted_fields import (
    EncryptedString,
    EncryptedFloat,
    EncryptedInteger,
    EncryptedJSON,
    EncryptedSearchableString,
    SearchTokenIndex,
    create_encrypted_column
)

Base = declarative_base()


class TradingAccount(Base):
    """Model for trading accounts with encrypted sensitive data."""
    
    __tablename__ = "trading_accounts"
    
    id = Column(Integer, primary_key=True)
    account_id = Column(String(50), unique=True, nullable=False)
    
    # Encrypted fields
    account_number = Column(
        EncryptedSearchableString(
            encryption_context="trading_account"
        ),
        nullable=False
    )
    account_number_search = Column(
        SearchTokenIndex(
            source_field="account_number",
            encryption_context="trading_account"
        ),
        index=True
    )
    
    exchange_name = Column(String(50), nullable=False)
    account_type = Column(String(20), nullable=False)  # spot, margin, futures
    
    # Encrypted balance information
    balance = Column(
        EncryptedFloat(encryption_context="account_balance"),
        nullable=False,
        default=0.0
    )
    available_balance = Column(
        EncryptedFloat(encryption_context="account_balance"),
        nullable=False,
        default=0.0
    )
    
    # Encrypted API credentials reference
    api_key_reference = Column(
        EncryptedString(encryption_context="api_credentials"),
        nullable=True
    )
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    positions = relationship("Position", back_populates="account")
    trades = relationship("Trade", back_populates="account")
    
    __table_args__ = (
        Index("idx_account_search", "account_number_search"),
        Index("idx_account_exchange", "exchange_name", "account_type"),
    )
    
    @validates("account_number")
    def validate_account_number(self, key, value):
        """Validate and set search token for account number."""
        if value:
            # The search token will be automatically generated
            self.account_number_search = value
        return value


class Position(Base):
    """Model for trading positions with encrypted sensitive data."""
    
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True)
    position_id = Column(String(50), unique=True, nullable=False)
    account_id = Column(Integer, ForeignKey("trading_accounts.id"), nullable=False)
    
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # long, short
    
    # Encrypted position details
    size = Column(
        EncryptedFloat(encryption_context="position_size"),
        nullable=False
    )
    entry_price = Column(
        EncryptedFloat(encryption_context="position_pricing"),
        nullable=False
    )
    current_price = Column(
        EncryptedFloat(encryption_context="position_pricing"),
        nullable=True
    )
    
    # Encrypted P&L calculations
    unrealized_pnl = Column(
        EncryptedFloat(encryption_context="position_pnl"),
        nullable=True
    )
    realized_pnl = Column(
        EncryptedFloat(encryption_context="position_pnl"),
        nullable=True,
        default=0.0
    )
    
    # Risk metrics (encrypted)
    stop_loss = Column(
        EncryptedFloat(encryption_context="position_risk"),
        nullable=True
    )
    take_profit = Column(
        EncryptedFloat(encryption_context="position_risk"),
        nullable=True
    )
    max_loss = Column(
        EncryptedFloat(encryption_context="position_risk"),
        nullable=True
    )
    
    # Metadata
    opened_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    closed_at = Column(DateTime, nullable=True)
    status = Column(String(20), default="open", nullable=False)
    
    # Encrypted metadata
    metadata = Column(
        EncryptedJSON(encryption_context="position_metadata"),
        nullable=True
    )
    
    # Relationships
    account = relationship("TradingAccount", back_populates="positions")
    trades = relationship("Trade", back_populates="position")
    
    __table_args__ = (
        Index("idx_position_account", "account_id"),
        Index("idx_position_symbol", "symbol"),
        Index("idx_position_status", "status"),
    )


class Trade(Base):
    """Model for individual trades with encrypted execution details."""
    
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(50), unique=True, nullable=False)
    account_id = Column(Integer, ForeignKey("trading_accounts.id"), nullable=False)
    position_id = Column(Integer, ForeignKey("positions.id"), nullable=True)
    
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # buy, sell
    trade_type = Column(String(20), nullable=False)  # market, limit, stop
    
    # Encrypted execution details
    quantity = Column(
        EncryptedFloat(encryption_context="trade_execution"),
        nullable=False
    )
    price = Column(
        EncryptedFloat(encryption_context="trade_execution"),
        nullable=False
    )
    fee = Column(
        EncryptedFloat(encryption_context="trade_execution"),
        nullable=True,
        default=0.0
    )
    
    # Encrypted financial details
    gross_value = Column(
        EncryptedFloat(encryption_context="trade_financial"),
        nullable=False
    )
    net_value = Column(
        EncryptedFloat(encryption_context="trade_financial"),
        nullable=False
    )
    
    # Order details (encrypted)
    order_id = Column(
        EncryptedString(encryption_context="trade_order"),
        nullable=True
    )
    client_order_id = Column(
        EncryptedString(encryption_context="trade_order"),
        nullable=True
    )
    
    # Execution metadata
    executed_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Encrypted metadata
    execution_details = Column(
        EncryptedJSON(encryption_context="trade_details"),
        nullable=True
    )
    
    # Relationships
    account = relationship("TradingAccount", back_populates="trades")
    position = relationship("Position", back_populates="trades")
    
    __table_args__ = (
        Index("idx_trade_account", "account_id"),
        Index("idx_trade_symbol", "symbol"),
        Index("idx_trade_executed", "executed_at"),
    )


class RiskMetrics(Base):
    """Model for risk metrics with encrypted sensitive calculations."""
    
    __tablename__ = "risk_metrics"
    
    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey("trading_accounts.id"), nullable=False)
    
    # Encrypted risk metrics
    portfolio_value = Column(
        EncryptedFloat(encryption_context="risk_metrics"),
        nullable=False
    )
    total_exposure = Column(
        EncryptedFloat(encryption_context="risk_metrics"),
        nullable=False
    )
    leverage = Column(
        EncryptedFloat(encryption_context="risk_metrics"),
        nullable=False,
        default=1.0
    )
    
    # VaR calculations (encrypted)
    var_95 = Column(
        EncryptedFloat(encryption_context="risk_var"),
        nullable=True
    )
    var_99 = Column(
        EncryptedFloat(encryption_context="risk_var"),
        nullable=True
    )
    cvar_95 = Column(
        EncryptedFloat(encryption_context="risk_var"),
        nullable=True
    )
    
    # Performance metrics (encrypted)
    sharpe_ratio = Column(
        EncryptedFloat(encryption_context="risk_performance"),
        nullable=True
    )
    sortino_ratio = Column(
        EncryptedFloat(encryption_context="risk_performance"),
        nullable=True
    )
    max_drawdown = Column(
        EncryptedFloat(encryption_context="risk_performance"),
        nullable=True
    )
    
    # Correlation and beta (encrypted)
    correlation_matrix = Column(
        EncryptedJSON(encryption_context="risk_correlation"),
        nullable=True
    )
    beta_values = Column(
        EncryptedJSON(encryption_context="risk_beta"),
        nullable=True
    )
    
    # Timestamp
    calculated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index("idx_risk_account", "account_id"),
        Index("idx_risk_calculated", "calculated_at"),
    )


class PortfolioSnapshot(Base):
    """Model for portfolio snapshots with encrypted holdings."""
    
    __tablename__ = "portfolio_snapshots"
    
    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey("trading_accounts.id"), nullable=False)
    
    # Encrypted portfolio details
    total_value = Column(
        EncryptedFloat(encryption_context="portfolio_value"),
        nullable=False
    )
    cash_balance = Column(
        EncryptedFloat(encryption_context="portfolio_value"),
        nullable=False
    )
    
    # Encrypted holdings
    holdings = Column(
        EncryptedJSON(encryption_context="portfolio_holdings"),
        nullable=False
    )
    
    # Encrypted allocation details
    allocations = Column(
        EncryptedJSON(encryption_context="portfolio_allocation"),
        nullable=False
    )
    
    # Performance (encrypted)
    daily_return = Column(
        EncryptedFloat(encryption_context="portfolio_performance"),
        nullable=True
    )
    cumulative_return = Column(
        EncryptedFloat(encryption_context="portfolio_performance"),
        nullable=True
    )
    
    # Timestamp
    snapshot_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index("idx_snapshot_account", "account_id"),
        Index("idx_snapshot_time", "snapshot_time"),
        UniqueConstraint("account_id", "snapshot_time", name="uq_account_snapshot"),
    )


# Utility functions for working with encrypted trading data
def mask_sensitive_value(value: Any, visible_chars: int = 4) -> str:
    """
    Mask sensitive values for display purposes.
    
    Args:
        value: The value to mask
        visible_chars: Number of characters to show
        
    Returns:
        Masked string
    """
    str_value = str(value)
    if len(str_value) <= visible_chars:
        return "*" * len(str_value)
    
    return str_value[:visible_chars] + "*" * (len(str_value) - visible_chars)


def create_trading_tables(engine):
    """Create all trading tables with encryption."""
    Base.metadata.create_all(engine)
    

def drop_trading_tables(engine):
    """Drop all trading tables."""
    Base.metadata.drop_all(engine)


# Export models
__all__ = [
    "TradingAccount",
    "Position",
    "Trade",
    "RiskMetrics",
    "PortfolioSnapshot",
    "create_trading_tables",
    "drop_trading_tables",
    "mask_sensitive_value"
]