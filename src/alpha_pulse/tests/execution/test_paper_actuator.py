"""
Tests for the paper trading actuator implementation.
"""
import pytest
from decimal import Decimal
from datetime import datetime
import asyncio

from alpha_pulse.execution.paper_actuator import PaperActuator, PaperTrade


@pytest.fixture
def paper_actuator():
    """Create paper actuator instance."""
    config = {
        "initial_balance": 100000,
        "slippage": 0.001,
        "fee_rate": 0.001
    }
    return PaperActuator(config)


def test_initialization(paper_actuator):
    """Test actuator initialization."""
    assert paper_actuator.paper_mode is True
    assert paper_actuator.slippage == 0.001
    assert paper_actuator.fee_rate == 0.001
    assert paper_actuator.initial_balance == Decimal("100000")
    assert paper_actuator.cash_balance == Decimal("100000")
    assert isinstance(paper_actuator.positions, dict)
    assert isinstance(paper_actuator.trade_history, list)


@pytest.mark.asyncio
async def test_execute_trade_buy(paper_actuator):
    """Test executing buy trade."""
    result = await paper_actuator.execute_trade(
        symbol="BTC/USDT",
        side="buy",
        quantity=1.0,
        price=50000.0
    )
    
    assert result["status"] == "executed"
    assert result["symbol"] == "BTC/USDT"
    assert result["side"] == "buy"
    assert result["quantity"] == 1.0
    assert abs(result["price"] - 50050.0) < 0.01  # Price with slippage
    assert "BTC/USDT" in paper_actuator.positions
    assert len(paper_actuator.trade_history) == 1


@pytest.mark.asyncio
async def test_execute_trade_sell(paper_actuator):
    """Test executing sell trade."""
    # First buy to create position
    await paper_actuator.execute_trade(
        symbol="BTC/USDT",
        side="buy",
        quantity=1.0,
        price=50000.0
    )
    
    # Then sell
    result = await paper_actuator.execute_trade(
        symbol="BTC/USDT",
        side="sell",
        quantity=1.0,
        price=51000.0
    )
    
    assert result["status"] == "executed"
    assert result["symbol"] == "BTC/USDT"
    assert result["side"] == "sell"
    assert result["quantity"] == 1.0
    assert abs(result["price"] - 50949.0) < 0.01  # Price with slippage
    assert "BTC/USDT" not in paper_actuator.positions
    assert len(paper_actuator.trade_history) == 2


@pytest.mark.asyncio
async def test_execute_trade_insufficient_funds(paper_actuator):
    """Test executing trade with insufficient funds."""
    result = await paper_actuator.execute_trade(
        symbol="BTC/USDT",
        side="buy",
        quantity=10000.0,
        price=50000.0
    )
    
    assert result["status"] == "rejected"
    assert "insufficient_funds" in result["reason"]
    assert len(paper_actuator.trade_history) == 0


@pytest.mark.asyncio
async def test_get_portfolio_value(paper_actuator):
    """Test getting portfolio value."""
    # Initial value should be cash balance
    value = await paper_actuator.get_portfolio_value()
    assert value == float(paper_actuator.cash_balance)
    
    # Add position and check value
    await paper_actuator.execute_trade(
        symbol="BTC/USDT",
        side="buy",
        quantity=1.0,
        price=50000.0
    )
    
    value = await paper_actuator.get_portfolio_value()
    assert abs(value - 100000.0) < 100  # Allow for slippage and fees


@pytest.mark.asyncio
async def test_get_positions(paper_actuator):
    """Test getting positions."""
    # Add position
    await paper_actuator.execute_trade(
        symbol="BTC/USDT",
        side="buy",
        quantity=1.0,
        price=50000.0
    )
    
    positions = await paper_actuator.get_positions()
    assert "BTC/USDT" in positions
    position = positions["BTC/USDT"]
    assert position["quantity"] == 1.0
    assert abs(position["avg_entry"] - 50050.0) < 1  # Price with slippage
    assert "unrealized_pnl" in position
    assert "unrealized_pnl_pct" in position


@pytest.mark.asyncio
async def test_get_trade_history(paper_actuator):
    """Test getting trade history."""
    # Execute some trades
    await paper_actuator.execute_trade(
        symbol="BTC/USDT",
        side="buy",
        quantity=1.0,
        price=50000.0
    )
    await paper_actuator.execute_trade(
        symbol="ETH/USDT",
        side="buy",
        quantity=10.0,
        price=3000.0
    )
    
    # Get all trades
    trades = await paper_actuator.get_trade_history()
    assert len(trades) == 2
    assert isinstance(trades[0], PaperTrade)
    
    # Filter by symbol
    btc_trades = await paper_actuator.get_trade_history(symbol="BTC/USDT")
    assert len(btc_trades) == 1
    assert btc_trades[0].symbol == "BTC/USDT"
    
    # Filter by time
    now = datetime.now()
    recent_trades = await paper_actuator.get_trade_history(
        start_time=now.replace(hour=0, minute=0, second=0)
    )
    assert len(recent_trades) == 2


def test_apply_slippage(paper_actuator):
    """Test slippage application."""
    price = 50000.0
    
    # Buy slippage (price increases)
    buy_price = paper_actuator._apply_slippage(price, "buy")
    assert buy_price > price
    assert abs(buy_price - 50050.0) < 0.01
    
    # Sell slippage (price decreases)
    sell_price = paper_actuator._apply_slippage(price, "sell")
    assert sell_price < price
    assert abs(sell_price - 49950.0) < 0.01


@pytest.mark.asyncio
async def test_validate_trade(paper_actuator):
    """Test trade validation."""
    # Valid trade
    assert await paper_actuator._validate_trade(
        side="buy",
        trade_value=Decimal("50000"),
        fees=Decimal("50")
    )
    
    # Invalid trade (insufficient funds)
    assert not await paper_actuator._validate_trade(
        side="buy",
        trade_value=Decimal("1000000"),
        fees=Decimal("1000")
    )
    
    # Sell trade (always valid in paper trading)
    assert await paper_actuator._validate_trade(
        side="sell",
        trade_value=Decimal("50000"),
        fees=Decimal("50")
    )


@pytest.mark.asyncio
async def test_update_portfolio(paper_actuator):
    """Test portfolio updates."""
    symbol = "BTC/USDT"
    quantity = 1.0
    price = 50000.0
    trade_value = Decimal("50000")
    fees = Decimal("50")
    
    # Test buy update
    await paper_actuator._update_portfolio(
        symbol=symbol,
        side="buy",
        quantity=quantity,
        price=price,
        trade_value=trade_value,
        fees=fees
    )
    
    assert paper_actuator.cash_balance == Decimal("100000") - trade_value - fees
    assert symbol in paper_actuator.positions
    assert paper_actuator.positions[symbol]["quantity"] == quantity
    
    # Test sell update
    await paper_actuator._update_portfolio(
        symbol=symbol,
        side="sell",
        quantity=quantity,
        price=price,
        trade_value=trade_value,
        fees=fees
    )
    
    assert symbol not in paper_actuator.positions
    assert paper_actuator.cash_balance == Decimal("100000") - fees - fees  # Both buy and sell fees