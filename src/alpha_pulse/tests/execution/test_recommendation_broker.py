"""
Tests for the recommendation-only broker implementation.
"""
import pytest
from datetime import datetime

from alpha_pulse.execution.recommendation_broker import RecommendationOnlyBroker
from alpha_pulse.execution.broker_interface import Order, OrderSide, OrderType, OrderStatus


@pytest.fixture
def recommendation_broker():
    """Create recommendation broker instance."""
    return RecommendationOnlyBroker()


def test_initialization(recommendation_broker):
    """Test broker initialization."""
    assert isinstance(recommendation_broker._orders, dict)
    assert isinstance(recommendation_broker._positions, dict)


def test_place_order(recommendation_broker):
    """Test placing order (recommendation only)."""
    order = Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
        order_type=OrderType.LIMIT,
        price=50000.0
    )
    
    result = recommendation_broker.place_order(order)
    
    assert result.status == OrderStatus.REJECTED
    assert isinstance(result.timestamp, datetime)


def test_cancel_order(recommendation_broker):
    """Test cancelling order (recommendation only)."""
    result = recommendation_broker.cancel_order("test_order_id")
    assert result is True


def test_get_order(recommendation_broker):
    """Test getting order details."""
    result = recommendation_broker.get_order("test_order_id")
    assert result is None


def test_get_orders(recommendation_broker):
    """Test getting all orders."""
    result = recommendation_broker.get_orders()
    assert isinstance(result, list)
    assert len(result) == 0


def test_get_position(recommendation_broker):
    """Test getting position details."""
    result = recommendation_broker.get_position("BTC/USDT")
    assert result is None


def test_get_positions(recommendation_broker):
    """Test getting all positions."""
    result = recommendation_broker.get_positions()
    assert isinstance(result, dict)
    assert len(result) == 0


def test_get_account_balance(recommendation_broker):
    """Test getting account balance."""
    result = recommendation_broker.get_account_balance()
    assert result == 0.0


def test_get_portfolio_value(recommendation_broker):
    """Test getting portfolio value."""
    result = recommendation_broker.get_portfolio_value()
    assert result == 0.0


def test_update_market_data(recommendation_broker):
    """Test updating market data."""
    # Should be no-op
    recommendation_broker.update_market_data("BTC/USDT", 50000.0)
    # No assertion needed as this is a no-op function