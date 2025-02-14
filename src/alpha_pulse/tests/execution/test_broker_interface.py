"""
Tests for broker interface types and data classes.
"""
import pytest
import asyncio
from decimal import Decimal
from datetime import datetime

from alpha_pulse.execution.broker_interface import (
    OrderSide,
    OrderType,
    OrderStatus,
    Order,
    OrderResult,
    Position,
    BrokerInterface
)


def test_order_side_enum():
    """Test OrderSide enumeration."""
    assert OrderSide.BUY.value == "buy"
    assert OrderSide.SELL.value == "sell"


def test_order_type_enum():
    """Test OrderType enumeration."""
    assert OrderType.MARKET.value == "market"
    assert OrderType.LIMIT.value == "limit"
    assert OrderType.STOP.value == "stop"


def test_order_status_enum():
    """Test OrderStatus enumeration."""
    assert OrderStatus.PENDING.value == "pending"
    assert OrderStatus.FILLED.value == "filled"
    assert OrderStatus.CANCELLED.value == "cancelled"
    assert OrderStatus.REJECTED.value == "rejected"


def test_order_creation():
    """Test Order dataclass creation."""
    order = Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
        order_type=OrderType.LIMIT,
        price=50000.0,
        stop_price=49000.0,
        order_id="test_id",
        status=OrderStatus.PENDING,
        filled_quantity=0.5,
        filled_price=50000.0,
        timestamp=1234567890.0
    )
    
    assert order.symbol == "BTC/USDT"
    assert order.side == OrderSide.BUY
    assert order.quantity == 1.0
    assert order.order_type == OrderType.LIMIT
    assert order.price == 50000.0
    assert order.stop_price == 49000.0
    assert order.order_id == "test_id"
    assert order.status == OrderStatus.PENDING
    assert order.filled_quantity == 0.5
    assert order.filled_price == 50000.0
    assert order.timestamp == 1234567890.0


def test_order_creation_defaults():
    """Test Order dataclass with default values."""
    order = Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
        order_type=OrderType.MARKET
    )
    
    assert order.price is None
    assert order.stop_price is None
    assert order.order_id is None
    assert order.status == OrderStatus.PENDING
    assert order.filled_quantity == 0.0
    assert order.filled_price is None
    assert order.timestamp is None


def test_order_result_creation():
    """Test OrderResult dataclass creation."""
    result = OrderResult(
        success=True,
        order_id="test_id",
        filled_quantity=1.0,
        filled_price=50000.0,
        error=None
    )
    
    assert result.success is True
    assert result.order_id == "test_id"
    assert result.filled_quantity == 1.0
    assert result.filled_price == 50000.0
    assert result.error is None


def test_order_result_with_error():
    """Test OrderResult dataclass with error."""
    result = OrderResult(
        success=False,
        error="Insufficient funds"
    )
    
    assert result.success is False
    assert result.order_id is None
    assert result.filled_quantity is None
    assert result.filled_price is None
    assert result.error == "Insufficient funds"


def test_position_creation():
    """Test Position dataclass creation."""
    position = Position(
        symbol="BTC/USDT",
        quantity=1.0,
        avg_entry_price=50000.0,
        unrealized_pnl=1000.0,
        realized_pnl=500.0,
        timestamp=1234567890.0
    )
    
    assert position.symbol == "BTC/USDT"
    assert position.quantity == 1.0
    assert position.avg_entry_price == 50000.0
    assert position.unrealized_pnl == 1000.0
    assert position.realized_pnl == 500.0
    assert position.timestamp == 1234567890.0


def test_position_creation_defaults():
    """Test Position dataclass with default values."""
    position = Position(
        symbol="BTC/USDT",
        quantity=1.0,
        avg_entry_price=50000.0
    )
    
    assert position.unrealized_pnl == 0.0
    assert position.realized_pnl == 0.0
    assert position.timestamp is None


def test_broker_interface_abstract_methods():
    """Test BrokerInterface abstract methods raise NotImplementedError."""
    broker = BrokerInterface()
    
    with pytest.raises(NotImplementedError):
        broker.get_portfolio_value()
        
    with pytest.raises(NotImplementedError):
        broker.get_available_margin()
        
    with pytest.raises(NotImplementedError):
        broker.get_position("BTC/USDT")
        
    with pytest.raises(NotImplementedError):
        broker.get_positions()
        
    with pytest.raises(NotImplementedError):
        broker.get_order("test_id")
        
    with pytest.raises(NotImplementedError):
        asyncio.run(broker.place_order(None))
        
    with pytest.raises(NotImplementedError):
        asyncio.run(broker.cancel_order("test_id"))
        
    with pytest.raises(NotImplementedError):
        broker.update_market_data("BTC/USDT", 50000.0)
        
    with pytest.raises(NotImplementedError):
        asyncio.run(broker.initialize_spot_position("BTC/USDT", 1.0, 50000.0))