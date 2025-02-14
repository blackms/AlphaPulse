"""
Tests for the paper broker implementation.
"""
import pytest
from decimal import Decimal
from datetime import datetime
import asyncio

from alpha_pulse.execution.paper_broker import PaperBroker
from alpha_pulse.execution.broker_interface import Position
from alpha_pulse.exchanges.base import Balance


@pytest.fixture
def paper_broker():
    """Create paper broker instance."""
    return PaperBroker(initial_balance=100000.0)


@pytest.mark.asyncio
async def test_initial_balance():
    """Test initial balance setup."""
    initial_balance = 50000.0
    broker = PaperBroker(initial_balance=initial_balance)
    assert broker.balance == Decimal(str(initial_balance))


@pytest.mark.asyncio
async def test_get_balances(paper_broker):
    """Test getting account balances."""
    balances = await paper_broker.get_balances()
    assert isinstance(balances, dict)
    assert "USDT" in balances
    assert isinstance(balances["USDT"], Balance)
    assert balances["USDT"].total == Decimal("100000.0")


@pytest.mark.asyncio
async def test_get_ticker_price(paper_broker):
    """Test getting ticker price."""
    # Test base currency pair
    price = await paper_broker.get_ticker_price("USDT/USDT")
    assert price == Decimal("1.0")
    
    # Test non-existent symbol
    price = await paper_broker.get_ticker_price("BTC/USDT")
    assert price is None
    
    # Test after updating market data
    paper_broker.update_market_data("BTC/USDT", 50000.0)
    price = await paper_broker.get_ticker_price("BTC/USDT")
    assert price == Decimal("50000.0")


@pytest.mark.asyncio
async def test_get_portfolio_value(paper_broker):
    """Test getting portfolio value."""
    # Initial value should be balance
    value = await paper_broker.get_portfolio_value()
    assert value == Decimal("100000.0")
    
    # Add a position and update market price
    paper_broker.positions["BTC/USDT"] = Position(
        symbol="BTC/USDT",
        quantity=1.0,
        avg_entry_price=45000.0
    )
    paper_broker.update_market_data("BTC/USDT", 50000.0)
    
    # Value should include position value
    value = await paper_broker.get_portfolio_value()
    assert value == Decimal("100000.0") + Decimal("50000.0")


@pytest.mark.asyncio
async def test_fetch_ohlcv(paper_broker):
    """Test fetching OHLCV data."""
    # Update market price
    paper_broker.update_market_data("BTC/USDT", 50000.0)
    
    # Fetch candles
    candles = await paper_broker.fetch_ohlcv("BTC/USDT")
    assert len(candles) == 1
    candle = candles[0]
    assert candle.open == Decimal("50000.0")
    assert candle.high == Decimal("50000.0")
    assert candle.low == Decimal("50000.0")
    assert candle.close == Decimal("50000.0")
    assert candle.volume == Decimal("0")


@pytest.mark.asyncio
async def test_execute_trade_buy(paper_broker):
    """Test executing buy trade."""
    # Setup
    symbol = "BTC/USDT"
    paper_broker.update_market_data(symbol, 50000.0)
    initial_balance = paper_broker.balance
    
    # Execute buy trade
    result = await paper_broker.execute_trade(
        symbol=symbol,
        side="buy",
        amount=1.0,
        price=50000.0
    )
    
    # Verify result
    assert result["success"]
    assert result["symbol"] == symbol
    assert result["side"] == "buy"
    assert result["amount"] == 1.0
    assert result["price"] == 50000.0
    assert result["status"] == "filled"
    
    # Verify balance and position
    assert paper_broker.balance == initial_balance - Decimal("50000.0")
    assert symbol in paper_broker.positions
    assert paper_broker.positions[symbol].quantity == 1.0
    assert paper_broker.positions[symbol].avg_entry_price == 50000.0


@pytest.mark.asyncio
async def test_execute_trade_sell(paper_broker):
    """Test executing sell trade."""
    # Setup - create initial position
    symbol = "BTC/USDT"
    paper_broker.update_market_data(symbol, 50000.0)
    paper_broker.positions[symbol] = Position(
        symbol=symbol,
        quantity=1.0,
        avg_entry_price=45000.0
    )
    initial_balance = paper_broker.balance
    
    # Execute sell trade
    result = await paper_broker.execute_trade(
        symbol=symbol,
        side="sell",
        amount=1.0,
        price=50000.0
    )
    
    # Verify result
    assert result["success"]
    assert result["symbol"] == symbol
    assert result["side"] == "sell"
    assert result["amount"] == 1.0
    assert result["price"] == 50000.0
    assert result["status"] == "filled"
    
    # Verify balance and position
    assert paper_broker.balance == initial_balance + Decimal("50000.0")
    assert symbol not in paper_broker.positions


@pytest.mark.asyncio
async def test_execute_trade_insufficient_funds(paper_broker):
    """Test executing trade with insufficient funds."""
    # Try to buy more than balance allows
    result = await paper_broker.execute_trade(
        symbol="BTC/USDT",
        side="buy",
        amount=1000.0,
        price=50000.0
    )
    
    assert not result["success"]
    assert "Insufficient funds" in result["error"]


@pytest.mark.asyncio
async def test_execute_trade_insufficient_position(paper_broker):
    """Test executing sell trade with insufficient position."""
    result = await paper_broker.execute_trade(
        symbol="BTC/USDT",
        side="sell",
        amount=1.0,
        price=50000.0
    )
    
    assert not result["success"]
    assert "Insufficient position" in result["error"]


@pytest.mark.asyncio
async def test_get_positions(paper_broker):
    """Test getting positions."""
    # Setup
    symbol = "BTC/USDT"
    paper_broker.update_market_data(symbol, 50000.0)
    paper_broker.positions[symbol] = Position(
        symbol=symbol,
        quantity=1.0,
        avg_entry_price=45000.0
    )
    
    # Get positions
    positions = await paper_broker.get_positions()
    
    # Verify
    assert symbol in positions
    position = positions[symbol]
    assert position["symbol"] == symbol
    assert position["quantity"] == 1.0
    assert position["avg_entry_price"] == 45000.0
    assert position["current_price"] == 50000.0
    assert position["unrealized_pnl"] == 5000.0


@pytest.mark.asyncio
async def test_get_average_entry_price(paper_broker):
    """Test getting average entry price."""
    # Setup
    symbol = "BTC/USDT"
    paper_broker.positions[symbol] = Position(
        symbol=symbol,
        quantity=1.0,
        avg_entry_price=45000.0
    )
    
    # Get entry price
    entry_price = await paper_broker.get_average_entry_price(symbol)
    assert entry_price == Decimal("45000.0")
    
    # Test non-existent position
    entry_price = await paper_broker.get_average_entry_price("ETH/USDT")
    assert entry_price is None


def test_update_market_data(paper_broker):
    """Test updating market data."""
    symbol = "BTC/USDT"
    price = 50000.0
    
    # Update market data
    paper_broker.update_market_data(symbol, price)
    assert paper_broker.market_prices[symbol] == Decimal(str(price))
    
    # Test with non-standard symbol format
    paper_broker.update_market_data("BTC", price)
    assert paper_broker.market_prices["BTC/USDT"] == Decimal(str(price))