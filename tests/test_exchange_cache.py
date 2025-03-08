"""
Tests for exchange data cache functionality.

This module provides tests for the exchange data cache components,
including the database models, repository, and synchronization scheduler.
"""
import asyncio
import os
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import patch, MagicMock, AsyncMock

from sqlalchemy.ext.asyncio import AsyncSession

from alpha_pulse.data_pipeline.database.connection import get_pg_connection
from alpha_pulse.data_pipeline.database.exchange_cache import (
    ExchangeCacheRepository, ExchangeSync, ExchangeOrder, 
    ExchangePosition, ExchangeBalance, ExchangePrice, SyncStatus
)
from alpha_pulse.data_pipeline.scheduler import (
    ExchangeDataSynchronizer, DataType
)
from alpha_pulse.exchanges.interfaces import BaseExchange
from alpha_pulse.exchanges.base import Balance


@pytest.fixture
async def db_session():
    """Provide a database session for testing."""
    # Use in-memory SQLite for testing
    os.environ["DB_TYPE"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = ":memory:"
    
    async with get_pg_connection() as session:
        # Create tables
        from alpha_pulse.data_pipeline.database.models import BaseModel
        await session.run_sync(lambda sync_session: BaseModel.metadata.create_all(sync_session.connection()))
        yield session


@pytest.fixture
def mock_exchange():
    """Provide a mock exchange for testing."""
    exchange = AsyncMock(spec=BaseExchange)
    
    # Mock the get_balances method
    mock_balances = {
        "BTC": Balance(total=Decimal("1.5"), available=Decimal("1.2"), locked=Decimal("0.3")),
        "ETH": Balance(total=Decimal("10.0"), available=Decimal("9.5"), locked=Decimal("0.5")),
        "USDT": Balance(total=Decimal("5000.0"), available=Decimal("4500.0"), locked=Decimal("500.0"))
    }
    exchange.get_balances.return_value = mock_balances
    
    # Mock the get_positions method
    mock_positions = {
        "BTC": {
            "symbol": "BTC",
            "quantity": 1.5,
            "entry_price": 45000.0,
            "current_price": 47000.0,
            "unrealized_pnl": 3000.0,
            "liquidation_price": None
        },
        "ETH": {
            "symbol": "ETH",
            "quantity": 10.0,
            "entry_price": 2500.0,
            "current_price": 2800.0,
            "unrealized_pnl": 3000.0,
            "liquidation_price": None
        }
    }
    exchange.get_positions.return_value = mock_positions
    
    # Mock the get_order_history method
    mock_orders = [
        {
            "id": "12345",
            "symbol": "BTC/USDT",
            "type": "market",
            "side": "buy",
            "price": 45000.0,
            "amount": 1.0,
            "cost": 45000.0,
            "filled": 1.0,
            "status": "filled",
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)
        },
        {
            "id": "12346",
            "symbol": "ETH/USDT",
            "type": "limit",
            "side": "buy",
            "price": 2500.0,
            "amount": 5.0,
            "cost": 12500.0,
            "filled": 5.0,
            "status": "filled",
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)
        }
    ]
    exchange.get_order_history.return_value = mock_orders
    
    # Mock the get_ticker_price method
    async def mock_get_ticker_price(symbol):
        prices = {
            "BTC/USDT": Decimal("47000.0"),
            "ETH/USDT": Decimal("2800.0")
        }
        return prices.get(symbol, Decimal("0"))
    
    exchange.get_ticker_price.side_effect = mock_get_ticker_price
    
    return exchange


@pytest.mark.asyncio
async def test_repository_save_and_get_balances(db_session, mock_exchange):
    """Test saving and retrieving balances."""
    # Arrange
    repository = ExchangeCacheRepository(db_session)
    exchange_id = "bybit"
    balances = await mock_exchange.get_balances()
    
    # Act
    count = await repository.save_balances(exchange_id, balances)
    saved_balances = await repository.get_balances(exchange_id)
    balances_dict = await repository.get_balances_dict(exchange_id)
    
    # Assert
    assert count == 3  # Three balances saved
    assert len(saved_balances) == 3
    assert "BTC" in balances_dict
    assert balances_dict["BTC"].total == Decimal("1.5")
    assert balances_dict["USDT"].available == Decimal("4500.0")


@pytest.mark.asyncio
async def test_repository_save_and_get_positions(db_session, mock_exchange):
    """Test saving and retrieving positions."""
    # Arrange
    repository = ExchangeCacheRepository(db_session)
    exchange_id = "bybit"
    positions = await mock_exchange.get_positions()
    
    # Act
    count = await repository.save_positions(exchange_id, positions)
    saved_positions = await repository.get_positions(exchange_id)
    positions_dict = await repository.get_positions_dict(exchange_id)
    
    # Assert
    assert count == 2  # Two positions saved
    assert len(saved_positions) == 2
    assert "BTC" in positions_dict
    assert positions_dict["BTC"]["quantity"] == 1.5
    assert positions_dict["ETH"]["entry_price"] == 2500.0


@pytest.mark.asyncio
async def test_repository_save_and_get_orders(db_session, mock_exchange):
    """Test saving and retrieving orders."""
    # Arrange
    repository = ExchangeCacheRepository(db_session)
    exchange_id = "bybit"
    orders = await mock_exchange.get_order_history()
    
    # Act
    count = await repository.save_orders(exchange_id, orders)
    saved_orders = await repository.get_orders(exchange_id)
    
    # Assert
    assert count == 2  # Two orders saved
    assert len(saved_orders) == 2
    assert saved_orders[0].symbol == "BTC/USDT" or saved_orders[0].symbol == "ETH/USDT"
    assert saved_orders[1].symbol == "BTC/USDT" or saved_orders[1].symbol == "ETH/USDT"


@pytest.mark.asyncio
async def test_synchronizer_sync_operations(db_session, mock_exchange):
    """Test synchronizer operations."""
    # Arrange
    with patch('alpha_pulse.data_pipeline.scheduler.get_pg_connection') as mock_get_conn:
        # Setup mock connection context manager
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = db_session
        mock_get_conn.return_value = mock_context
        
        synchronizer = ExchangeDataSynchronizer()
        synchronizer._exchanges = {"bybit": mock_exchange}
        
        # Act - Test sync orders
        repository = ExchangeCacheRepository(db_session)
        order_count = await synchronizer._sync_orders(mock_exchange, "bybit", repository)
        
        # Assert
        assert order_count == 2
        
        # Act - Test sync balances
        balance_count = await synchronizer._sync_balances(mock_exchange, "bybit", repository)
        
        # Assert
        assert balance_count == 3
        
        # Act - Test sync positions
        position_count = await synchronizer._sync_positions(mock_exchange, "bybit", repository)
        
        # Assert
        assert position_count == 2


@pytest.mark.asyncio
async def test_portfolio_accessor_with_cache(db_session, mock_exchange):
    """Test portfolio accessor using cache."""
    with patch('alpha_pulse.api.data.portfolio.get_pg_connection') as mock_get_conn:
        # Setup mock connection context manager
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = db_session
        mock_get_conn.return_value = mock_context
        
        # Arrange - Save some test data
        repository = ExchangeCacheRepository(db_session)
        await repository.save_balances("bybit", await mock_exchange.get_balances())
        await repository.save_positions("bybit", await mock_exchange.get_positions())
        
        # Import portfolio accessor
        from alpha_pulse.api.data.portfolio import PortfolioDataAccessor
        
        # Create accessor with mocked exchange
        accessor = PortfolioDataAccessor()
        accessor._exchange = mock_exchange
        accessor._initialized = True
        accessor._exchange_id = "bybit"
        
        # Act - Get portfolio from cache
        portfolio_data = await accessor._get_portfolio_from_cache()
        
        # Assert
        assert portfolio_data is not None
        assert portfolio_data["data_source"] == "cache"
        assert portfolio_data["cash"] == 5000.0
        assert len(portfolio_data["positions"]) == 2
        
        # Check position details
        btc_position = next((p for p in portfolio_data["positions"] if p["symbol"] == "BTC"), None)
        assert btc_position is not None
        assert btc_position["quantity"] == 1.5
        assert btc_position["current_price"] == 47000.0