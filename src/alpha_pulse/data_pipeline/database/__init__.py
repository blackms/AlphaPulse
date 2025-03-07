"""
Database module for AlphaPulse.

This module provides database access and models for the AlphaPulse system.
"""
from .connection import (
    connection_manager, 
    get_pg_connection, 
    get_redis_connection, 
    execute_in_transaction,
    execute_with_retry
)
from .models import (
    User, ApiKey, Portfolio, Position, Trade, Alert, Metric, BaseModel
)
from .repository import (
    Repository,
    UserRepository,
    ApiKeyRepository,
    PortfolioRepository,
    PositionRepository,
    TradeRepository,
    AlertRepository,
    MetricRepository
)

__all__ = [
    # Connection
    'connection_manager',
    'get_pg_connection',
    'get_redis_connection',
    'execute_in_transaction',
    'execute_with_retry',
    
    # Models
    'User',
    'ApiKey',
    'Portfolio',
    'Position',
    'Trade',
    'Alert',
    'Metric',
    'BaseModel',
    
    # Repositories
    'Repository',
    'UserRepository',
    'ApiKeyRepository',
    'PortfolioRepository',
    'PositionRepository',
    'TradeRepository',
    'AlertRepository',
    'MetricRepository'
]