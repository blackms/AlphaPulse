"""
Unit tests for the GridHedgeBot implementation.
"""
import pytest
import pytest_asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime

from alpha_pulse.hedging.grid.bot import GridHedgeBot
from alpha_pulse.hedging.common.types import GridState, MarketState, PositionState
from alpha_pulse.execution.broker_interface import BrokerInterface
from alpha_pulse.data_pipeline.providers.exchange import ExchangeDataProvider


@pytest.fixture
def mock_broker():
    """Create mock broker interface."""
    broker = AsyncMock(spec=BrokerInterface)
    return broker


@pytest.fixture
def mock_data_provider():
    """Create mock exchange data provider."""
    provider = AsyncMock(spec=ExchangeDataProvider)
    provider.get_current_price = AsyncMock(return_value=Decimal('50000.00'))
    provider.initialize = AsyncMock()
    return provider


@pytest.fixture
def default_config():
    """Default test configuration."""
    return {
        'grid_spacing_pct': '0.01',  # 1%
        'num_levels': 5,
        'min_price_distance': '1.0',
        'max_position_size': '1.0',
        'max_drawdown': '0.1',  # 10%
        'stop_loss_pct': '0.04',  # 4%
        'var_limit': '10000',  # $10k
        'max_active_orders': 50,
        'rebalance_interval_seconds': 60
    }


@pytest.fixture
def mock_position():
    """Create mock position state."""
    return PositionState(
        spot_quantity=Decimal('0'),
        futures_quantity=Decimal('0'),
        avg_entry_price=Decimal('50000.00'),
        unrealized_pnl=Decimal('0'),
        realized_pnl=Decimal('0'),
        funding_paid=Decimal('0'),
        last_update=datetime.now()
    )


@pytest_asyncio.fixture
async def grid_bot(mock_broker, mock_data_provider, default_config, mock_position):
    """Create test grid bot instance."""
    # Create bot with mocked components
    bot = GridHedgeBot(
        broker=mock_broker,
        data_provider=mock_data_provider,
        symbol='BTC/USD',
        config=default_config
    )
    
    # Mock internal components
    bot.order_manager = AsyncMock()
    bot.order_manager.get_active_orders = MagicMock(return_value={})
    bot.order_manager.cancel_orders = AsyncMock()
    bot.order_manager.place_grid_orders = AsyncMock()
    bot.order_manager.update_risk_orders = AsyncMock()
    
    bot.risk_manager = MagicMock()
    bot.risk_manager.validate_risk_limits = MagicMock(return_value=True)
    
    bot.grid_calculator = MagicMock()
    bot.grid_calculator.calculate_grid_levels = MagicMock(return_value=[])
    
    bot.state_manager = MagicMock()
    bot.state_manager.state = GridState.ACTIVE
    bot.state_manager.position = mock_position
    bot.state_manager.update_state = MagicMock()
    bot.state_manager.record_rebalance = MagicMock()
    bot.state_manager.get_status = MagicMock(return_value={'state': GridState.ACTIVE})
    
    return bot


@pytest.mark.asyncio
async def test_init_with_default_config(grid_bot, default_config):
    """Test bot initialization with default configuration."""
    assert grid_bot.symbol == 'BTC/USD'
    assert grid_bot.grid_spacing == Decimal('0.01')
    assert grid_bot.num_levels == 5
    assert grid_bot.min_price_distance == Decimal('1.0')
    assert grid_bot.max_position_size == Decimal('1.0')
    assert grid_bot.max_drawdown == Decimal('0.1')
    assert grid_bot.stop_loss_pct == Decimal('0.04')
    assert grid_bot.var_limit == Decimal('10000')
    assert grid_bot.max_active_orders == 50
    assert grid_bot.rebalance_interval == 60


@pytest.mark.asyncio
async def test_start_success(grid_bot, mock_data_provider):
    """Test successful bot start."""
    # Setup
    mock_data_provider.get_current_price.return_value = Decimal('50000.00')
    
    # Execute
    await grid_bot.start()
    
    # Verify
    mock_data_provider.get_current_price.assert_called_once_with('BTC/USD')
    grid_bot.state_manager.update_state.assert_called_with(GridState.ACTIVE)
    grid_bot.order_manager.place_grid_orders.assert_called_once()


@pytest.mark.asyncio
async def test_start_no_price(grid_bot, mock_data_provider):
    """Test bot start with no price available."""
    # Setup
    mock_data_provider.get_current_price.return_value = None
    
    # Execute and verify
    with pytest.raises(ValueError, match="Could not get price for BTC/USD"):
        await grid_bot.start()
    
    grid_bot.state_manager.update_state.assert_called_with(GridState.ERROR)


@pytest.mark.asyncio
async def test_stop_success(grid_bot):
    """Test successful bot stop."""
    # Setup - Start the bot first
    await grid_bot.start()
    
    # Execute
    await grid_bot.stop()
    
    # Verify
    grid_bot.order_manager.cancel_orders.assert_called_once()
    grid_bot.state_manager.update_state.assert_called_with(GridState.STOPPED)


@pytest.mark.asyncio
async def test_execute_inactive_bot(grid_bot):
    """Test execute when bot is not active."""
    # Setup
    grid_bot.state_manager.state = GridState.STOPPED
    
    # Execute
    await grid_bot.execute(50000.00)
    
    # Verify no actions taken
    grid_bot.risk_manager.validate_risk_limits.assert_not_called()
    grid_bot.grid_calculator.calculate_grid_levels.assert_not_called()


@pytest.mark.asyncio
async def test_execute_risk_exceeded(grid_bot):
    """Test execute when risk limits are exceeded."""
    # Setup
    await grid_bot.start()
    grid_bot.risk_manager.validate_risk_limits.return_value = False
    
    # Execute
    await grid_bot.execute(50000.00)
    
    # Verify
    grid_bot.risk_manager.validate_risk_limits.assert_called_once()
    grid_bot.state_manager.update_state.assert_called_with(GridState.STOPPED)


@pytest.mark.asyncio
async def test_execute_success(grid_bot):
    """Test successful execute iteration."""
    # Setup
    grid_bot.grid_calculator.calculate_grid_levels.reset_mock()  # Reset the mock to clear start() call
    
    # Execute
    await grid_bot.execute(50000.00)
    
    # Verify
    grid_bot.risk_manager.validate_risk_limits.assert_called_once()
    grid_bot.grid_calculator.calculate_grid_levels.assert_called_once()
    grid_bot.order_manager.place_grid_orders.assert_called_once()
    grid_bot.state_manager.record_rebalance.assert_called_once()


@pytest.mark.asyncio
async def test_get_status(grid_bot):
    """Test getting bot status."""
    # Setup
    expected_status = {'state': GridState.ACTIVE}
    grid_bot.state_manager.get_status.return_value = expected_status
    
    # Execute
    status = grid_bot.get_status()
    
    # Verify
    assert status == expected_status
    grid_bot.state_manager.get_status.assert_called_once()


@pytest.mark.asyncio
async def test_create_success():
    """Test successful bot creation via factory method."""
    # Setup
    mock_broker = AsyncMock(spec=BrokerInterface)
    config = {'grid_spacing_pct': '0.02'}
    
    # Execute
    with patch('alpha_pulse.hedging.grid.bot.ExchangeDataProvider') as mock_provider_cls:
        mock_provider = AsyncMock(spec=ExchangeDataProvider)
        mock_provider.initialize = AsyncMock()
        mock_provider_cls.return_value = mock_provider
        mock_provider.get_current_price.return_value = Decimal('50000.00')
        
        bot = await GridHedgeBot.create(
            broker=mock_broker,
            symbol='BTC/USD',
            config=config
        )
    
    # Verify
    assert isinstance(bot, GridHedgeBot)
    assert bot.symbol == 'BTC/USD'
    assert bot.grid_spacing == Decimal('0.02')
    assert bot.state_manager.state == GridState.ACTIVE