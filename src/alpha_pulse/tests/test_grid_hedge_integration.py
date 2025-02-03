"""
Tests for grid hedge bot integration with different trading modes.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from ..execution.broker_factory import create_broker, TradingMode
from ..execution.broker_interface import Order, OrderStatus, Position
from ..hedging.grid_hedge_config import GridHedgeConfig, GridDirection
from ..hedging.grid_hedge_bot import GridHedgeBot


def create_test_config():
    """Create a test grid configuration."""
    return GridHedgeConfig.create_symmetric_grid(
        symbol="BTCUSDT",
        center_price=40000.0,
        grid_spacing=100.0,
        num_levels=3,
        position_step_size=0.001,
        max_position_size=0.01,
        grid_direction=GridDirection.BOTH
    )


def test_recommendation_mode():
    """Test that recommendation mode only logs orders without execution."""
    # Create broker and bot
    broker = create_broker(TradingMode.RECOMMENDATION)
    config = create_test_config()
    bot = GridHedgeBot(broker, config)
    
    # Execute strategy
    with patch('loguru.logger.info') as mock_logger:
        bot.execute(40000.0)
        
        # Verify that orders were logged but not executed
        assert mock_logger.call_count > 0
        assert len(broker.get_orders()) == 0
        assert len(broker.get_positions()) == 0


def test_paper_trading_mode():
    """Test that paper trading mode simulates order execution."""
    # Create broker and bot
    broker = create_broker(TradingMode.PAPER)
    config = create_test_config()
    bot = GridHedgeBot(broker, config)
    
    # Initial price at center
    current_price = 40000.0
    
    # Execute strategy
    bot.execute(current_price)
    
    # Verify orders were placed
    orders = broker.get_orders()
    assert len(orders) > 0
    
    # Verify grid levels
    buy_orders = [o for o in orders if o.price < current_price]
    sell_orders = [o for o in orders if o.price > current_price]
    assert len(buy_orders) > 0
    assert len(sell_orders) > 0
    
    # Verify order properties
    for order in orders:
        assert order.symbol == config.symbol
        assert order.quantity <= config.max_position_size
        assert order.status in [OrderStatus.PENDING, OrderStatus.FILLED]


@pytest.mark.parametrize("exchange", ["binance", "bybit"])
def test_real_trading_mode(exchange):
    """Test real trading mode with mock exchange."""
    # Mock exchange API credentials
    api_key = "test_key"
    api_secret = "test_secret"
    
    # Create mock exchange responses
    mock_order_response = {
        "orderId": "123",
        "status": "NEW",
        "executedQty": "0",
        "price": "39900"
    }
    
    mock_position_response = {
        "symbol": "BTCUSDT",
        "positionAmt": "0.001",
        "entryPrice": "40000",
        "unrealizedPnl": "0"
    }
    
    # Create broker with mocked exchange
    with patch(f"..exchanges.{exchange}.{exchange.capitalize()}Exchange") as mock_exchange:
        # Configure mock exchange
        mock_exchange.return_value.place_order.return_value = mock_order_response
        mock_exchange.return_value.get_position.return_value = mock_position_response
        mock_exchange.return_value.get_positions.return_value = [mock_position_response]
        
        broker = create_broker(
            TradingMode.REAL,
            exchange_name=exchange,
            api_key=api_key,
            api_secret=api_secret,
            testnet=True
        )
        
        config = create_test_config()
        bot = GridHedgeBot(broker, config)
        
        # Execute strategy
        bot.execute(40000.0)
        
        # Verify exchange interactions
        assert mock_exchange.return_value.place_order.called
        assert mock_exchange.return_value.get_position.called


def test_grid_rebalancing():
    """Test that grid rebalances correctly when price moves."""
    broker = create_broker(TradingMode.PAPER)
    config = create_test_config()
    bot = GridHedgeBot(broker, config)
    
    # Initial execution at center price
    bot.execute(40000.0)
    initial_orders = set(o.order_id for o in broker.get_orders())
    
    # Move price up significantly
    bot.execute(40500.0)
    new_orders = set(o.order_id for o in broker.get_orders())
    
    # Verify that orders were rebalanced
    assert initial_orders != new_orders


def test_position_limits():
    """Test that grid respects position limits."""
    broker = create_broker(TradingMode.PAPER)
    config = create_test_config()
    bot = GridHedgeBot(broker, config)
    
    # Execute strategy
    bot.execute(40000.0)
    
    # Get all positions
    positions = broker.get_positions()
    
    # Verify position sizes
    total_position = sum(abs(pos.quantity) for pos in positions.values())
    assert total_position <= config.max_position_size


def test_invalid_config():
    """Test that invalid configurations are rejected."""
    with pytest.raises(ValueError):
        # Try to create invalid config with upper price below lower price
        GridHedgeConfig(
            symbol="BTCUSDT",
            grid_direction=GridDirection.BOTH,
            grid_levels=[],
            upper_price=39000.0,  # Invalid: upper < lower
            lower_price=40000.0,
            grid_spacing=100.0,
            max_position_size=0.01,
            position_step_size=0.001
        )